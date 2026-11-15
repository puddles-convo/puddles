"""The OpenAI model behind the critic, researcher and proposer.

The model is set in configuration (PUDDLES_AGENT_MODEL), never in code.
"""
import json
import os
from typing import Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

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
    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI

        name = model or os.environ.get("PUDDLES_AGENT_MODEL")
        if not name:
            raise AgentFailed("PUDDLES_AGENT_MODEL is not set")
        self.model: str = name
        self.client = OpenAI()
        temperature = os.environ.get("PUDDLES_AGENT_TEMPERATURE")
        self.extra = {"temperature": float(temperature)} if temperature else {}

    def _parse(self, messages: list[dict], schema: type[T]) -> T:
        resp = self.client.chat.completions.parse(model=self.model, messages=messages,  # type: ignore[arg-type]
                                                  response_format=schema, **self.extra)  # type: ignore[arg-type]
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise AgentFailed(f"model refused or returned no {schema.__name__}")
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
            resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                       tools=[SEARCH_TOOL], **self.extra)  # type: ignore
            msg = resp.choices[0].message
            if not msg.tool_calls:
                break
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                if searches >= max_searches:
                    result: object = {"error": "search limit reached"}
                else:
                    query = json.loads(call.function.arguments)["query"]
                    result = {"query": query, "results": search(query)}
                    searches += 1
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        return self._parse(messages, schema)


def call_checked(produce: Callable[[str], T], check: Check, what: str) -> T:
    """Run a model step, retrying once with the reasons it failed its checks."""
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
            raise AgentFailed(f"{what} failed its checks twice: {'; '.join(problems)}")
        feedback = ("\n\nYour previous answer failed these checks. Fix them:\n- "
                    + "\n- ".join(problems))
    raise AssertionError("unreachable")
