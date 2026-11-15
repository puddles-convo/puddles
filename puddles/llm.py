"""The OpenAI model behind the critic, researcher and proposer.

The model is set in configuration (PUDDLES_AGENT_MODEL), never in code. So is the output limit
of each agent (`output_limits` in the run's config file); every request carries one.
"""
import json
import os
from typing import Any, Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from loop.llm import CallUsage, estimate_tokens
from puddles import usage
from puddles.config import load_config, output_limit

T = TypeVar("T", bound=BaseModel)
Check = Callable[[T], list[str]]
Tool = Callable[[str], list[dict]]


class AgentFailed(RuntimeError):
    pass


class LLM(Protocol):
    def structured(self, system: str, user: str, schema: type[T]) -> T: ...
    def structured_with_search(self, system: str, user: str, schema: type[T], search: Tool,
                               max_searches: int) -> T: ...


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web through Exa. Returns titles, URLs and page text.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"], "additionalProperties": False},
    },
}


class OpenAILLM:
    """Reports its own usage: each request is logged and traced when a meter is active."""

    reports_usage = True

    def __init__(self, model: Optional[str] = None, output_limits: Optional[dict[str, int]] = None):
        from openai import OpenAI

        name = model or os.environ.get("PUDDLES_AGENT_MODEL")
        if not name:
            raise AgentFailed("PUDDLES_AGENT_MODEL is not set")
        self.model: str = name
        self.output_limits = output_limits if output_limits is not None else load_config().output_limits
        self.client = OpenAI()
        temperature = os.environ.get("PUDDLES_AGENT_TEMPERATURE")
        self.extra = {"temperature": float(temperature)} if temperature else {}
        # Only on the requests that carry the search tool; see docs/specs/provider-settings.md.
        effort = os.environ.get("PUDDLES_AGENT_TOOL_REASONING_EFFORT")
        self.tool_extra = {"reasoning_effort": effort} if effort else {}

    def _request(self, send: Callable[..., Any], messages: list[dict], **options: Any) -> Any:
        """One API request, with the agent's output limit. Returns the response.

        The agent is the one named by `puddles.usage.metering`. With a meter active the request
        gets a `sent` and a `done` line in the usage log and its own span with its token counts.
        An `openai.OpenAIError` other than `LengthFinishReasonError` becomes `AgentFailed`.
        """
        from openai import LengthFinishReasonError, OpenAIError

        meter, agent = usage.current()
        limit = output_limit(self.output_limits, agent)
        options = {**self.extra, **options, "max_completion_tokens": limit}

        def sent() -> Any:
            try:
                return send(model=self.model, messages=messages, **options)
            except LengthFinishReasonError:
                raise
            except OpenAIError as exc:
                reason = f"the request to {self.model} failed: {type(exc).__name__}: {str(exc)[:300]}"
                usage.note_failed(agent or "agent", reason)
                raise AgentFailed(reason) from exc

        if meter is None:
            return sent()
        from puddles.tracing import model_call  # here, so that importing this module does not import MLflow

        with model_call(agent or "agent", self.model, {"messages": messages},
                        prompt_tokens=estimate_tokens(json.dumps(messages)), max_tokens=limit, meter=meter) as call:
            try:
                resp = sent()
            except LengthFinishReasonError as exc:
                call.finish("(cut off at the output limit)", _usage(exc.completion, messages))
                raise
            call.finish(resp.choices[0].message.model_dump(exclude_none=True), _usage(resp, messages))
            return resp

    def _parse(self, messages: list[dict], schema: type[T]) -> T:
        from openai import LengthFinishReasonError

        try:
            resp = self._request(self.client.chat.completions.parse, messages, response_format=schema)
        except LengthFinishReasonError as exc:
            reason = f"the answer was cut off at the output limit before it was a whole {schema.__name__}"
            usage.note_failed(usage.current()[1] or "agent", reason)
            raise AgentFailed(reason) from exc
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            reason = f"model refused or returned no {schema.__name__}"
            usage.note_failed(usage.current()[1] or "agent", reason)
            raise AgentFailed(reason)
        return parsed

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        return self._parse([{"role": "system", "content": system},
                            {"role": "user", "content": user}], schema)

    def structured_with_search(self, system: str, user: str, schema: type[T], search: Tool,
                               max_searches: int) -> T:
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": user}]
        searches = 0
        while searches < max_searches:
            resp = self._request(self.client.chat.completions.create, messages, tools=[SEARCH_TOOL],
                                 **self.tool_extra)
            choice = resp.choices[0]
            msg = choice.message
            if getattr(choice, "finish_reason", None) == "length":
                reason = "a search turn was cut off at the output limit"
                usage.note_failed(usage.current()[1] or "agent", reason)
                raise AgentFailed(reason)
            if not msg.tool_calls:
                break
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                if searches >= max_searches:
                    result: object = {"error": "search limit reached"}
                else:
                    searches += 1
                    query = _search_query(call)
                    result = ({"query": query, "results": search(query)} if query is not None else
                              {"error": 'the call must be search with arguments {"query": "<non-blank text>"}'})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        return self._parse(messages, schema)


def _search_query(call: Any) -> Optional[str]:
    """The query of a well-formed call to the search tool, or None."""
    function = getattr(call, "function", None)
    if getattr(function, "name", None) != SEARCH_TOOL["function"]["name"]:
        return None
    try:
        args = json.loads(getattr(function, "arguments", None) or "")
    except (TypeError, ValueError):
        return None
    query = args.get("query") if isinstance(args, dict) else None
    return query if isinstance(query, str) and query.strip() else None


def _usage(resp: Any, messages: list[dict]) -> CallUsage:
    """The token counts the API returned, or an estimate from the text when it returned none."""
    counts = getattr(resp, "usage", None)
    tokens_in = getattr(counts, "prompt_tokens", None)
    tokens_out = getattr(counts, "completion_tokens", None)
    if isinstance(tokens_in, int) and isinstance(tokens_out, int):
        return CallUsage(tokens_in, tokens_out)
    try:
        answer = json.dumps(resp.choices[0].message.model_dump(exclude_none=True), default=str)
    except Exception:
        answer = ""
    return CallUsage(estimate_tokens(json.dumps(messages)), estimate_tokens(answer), estimated=True)


def call_checked(produce: Callable[[str], T], check: Check, what: str) -> T:
    """Run a model step, retrying once with the reasons it failed its checks.

    `what` is the agent's name. When a meter is active (`puddles.usage.metering`), a first
    answer that fails is counted for that agent, and a second failure writes a `failed` line.
    """
    feedback = ""
    for attempt in (1, 2):
        try:
            out = produce(feedback)
        except ValidationError as e:
            problems = [str(e)]
        else:
            problems = check(out)
            if not problems:
                return out
        if attempt == 2:
            reason = f"{what} failed its checks twice: {'; '.join(problems)}"
            usage.note_failed(what, reason)
            raise AgentFailed(reason)
        usage.note_first_answer_failed(what)
        feedback = ("\n\nYour previous answer failed these checks. Fix them:\n- "
                    + "\n- ".join(problems))
    raise AssertionError("unreachable")
