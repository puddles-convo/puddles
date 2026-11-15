"""MLflow tracing: where traces go, and a span around every chat-model call.

`loop.llm` talks to its models over plain HTTP, so MLflow's automatic OpenAI tracing does not
see those calls. `TracedClient` wraps any `ChatClient` and opens one span per request instead.

Every span opened here carries the request's token counts. When a meter is active
(`puddles.usage.metering`), the same request is also written to the usage log, before and after.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey
from pydantic import BaseModel

from loop.errors import SchemaError
from loop.llm import CallUsage, ChatClient, complete_json_with, estimate_tokens, estimate_usage, prompt_text
from puddles.usage import Call, Meter, current, model_name

T = TypeVar("T", bound=BaseModel)

DEFAULT_TRACKING_URI = "sqlite:///runs/mlflow.db"
SQLITE_PREFIX = "sqlite:///"


def setup(tracking_uri: str | None = None, experiment: str = "puddles") -> None:
    """Point MLflow at the tracking store, pick the experiment and turn on autologging."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    if uri.startswith(SQLITE_PREFIX):
        db = uri[len(SQLITE_PREFIX):]
        if db and db != ":memory:":
            Path(db).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    mlflow.langchain.autolog()
    mlflow.openai.autolog()


class ModelCall:
    """One request inside its span. Call `finish` once, when the answer has arrived."""

    def __init__(self, span: Any, meter: Optional[Meter], call: Optional[Call]) -> None:
        self.span, self.meter, self.call = span, meter, call

    def finish(self, output: Any, usage: CallUsage) -> None:
        self.span.set_outputs(output)
        self.span.set_attribute(SpanAttributeKey.CHAT_USAGE, {
            "input_tokens": usage.tokens_in, "output_tokens": usage.tokens_out,
            "total_tokens": usage.tokens_in + usage.tokens_out})
        self.span.set_attribute("puddles.tokens_estimated", usage.estimated)
        if self.meter is not None and self.call is not None:
            self.span.set_attribute("puddles.cost_usd", self.meter.done(self.call, usage))


@contextmanager
def model_call(agent: str, model: str, inputs: dict, *, prompt_tokens: int, max_tokens: int,
               meter: Optional[Meter] = None) -> Iterator[ModelCall]:
    """One span named llm:<agent> and, with a meter, the `sent` line of one model request.

    The `sent` line is written before the block runs. `finish` writes the `done` line. If the
    block raises first, there is no `done` line and the request is counted at its estimate.
    """
    with mlflow.start_span(name=f"llm:{agent}", span_type=SpanType.CHAT_MODEL) as span:
        span.set_inputs(inputs)
        span.set_attribute("puddles.model", model)
        call = meter.sent(agent, model, prompt_tokens, max_tokens) if meter is not None else None
        if call is not None:
            span.set_attribute("puddles.call_id", call.id)
        yield ModelCall(span, meter, call)


class TracedClient:
    """A ChatClient that records each request as an MLflow span named llm:<name>.

    `meter` is optional; without one the meter set by `puddles.usage.metering` is used, if any.
    """

    def __init__(self, inner: ChatClient, name: str, meter: Optional[Meter] = None) -> None:
        self.inner = inner
        self.name = name
        self.meter = meter

    def _meter(self) -> Optional[Meter]:
        return self.meter if self.meter is not None else current()[0]

    def _send(self, send: Callable[..., str], system: str, messages: list[dict[str, str]],
              **options: Any) -> str:
        inputs = {"system": system, "messages": [dict(m) for m in messages], **options}
        prompt = estimate_tokens(prompt_text(system, messages))
        if options.get("response_format") is not None:
            prompt += estimate_tokens(json.dumps(options["response_format"]))
        with model_call(self.name, model_name(self.inner), inputs, prompt_tokens=prompt,
                        max_tokens=options["max_tokens"], meter=self._meter()) as call:
            text = send(system, messages, **options)
            call.finish(text, getattr(self.inner, "last_usage", None) or estimate_usage(system, messages, text))
            return text

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str:
        return self._send(self.inner.complete, system, messages, max_tokens=max_tokens,
                          temperature=temperature, seed=seed)

    def request(self, system: str, messages: list[dict[str, str]], *, max_tokens: int,
                temperature: float = 0.7, seed: int | None = None,
                response_format: dict[str, Any] | None = None, timeout: float | None = None) -> str:
        return self._send(self.inner.request, system, messages,  # type: ignore[attr-defined]
                          max_tokens=max_tokens, temperature=temperature, seed=seed,
                          response_format=response_format, timeout=timeout)

    def complete_json(self, system: str, messages: list[dict[str, str]], schema: type[T], *,
                      max_tokens: int, temperature: float = 0.0, seed: int | None = None,
                      deadline_s: float | None = None) -> T:
        """`complete_json` of the inner client, with one span and one pair of usage lines per attempt."""
        meter = self._meter()

        def first_failed() -> None:
            if meter is not None:
                meter.first_answer_failed(self.name)

        try:
            return complete_json_with(self.request, self.inner, system, messages, schema,
                                      max_tokens=max_tokens, temperature=temperature, seed=seed,
                                      deadline_s=deadline_s, on_first_failure=first_failed)
        except SchemaError as exc:
            if meter is not None:
                meter.failed(self.name, str(exc))
            raise


class TracedLLM:
    """Wraps a scripted `puddles.llm.LLM` that does not report its own calls.

    Each call becomes one span and one pair of usage lines, under the agent named by
    `puddles.usage.metering`, with token counts estimated from the text. `OpenAILLM` is not
    wrapped: it reports each of its requests itself, with the counts the API returned.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def _run(self, system: str, user: str, produce: Callable[[], T]) -> T:
        meter, agent = current()
        prompt = estimate_tokens(system + user)
        with model_call(agent or "agent", model_name(self.inner), {"system": system, "user": user},
                        prompt_tokens=prompt, max_tokens=0, meter=meter) as call:
            out = produce()
            text = out.model_dump_json()
            call.finish(text, CallUsage(prompt, estimate_tokens(text), estimated=True))
            return out

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        return self._run(system, user, lambda: self.inner.structured(system, user, schema))

    def structured_with_search(self, system: str, user: str, schema: type[T],
                               search: Callable[[str], list[dict]], max_searches: int) -> T:
        return self._run(system, user, lambda: self.inner.structured_with_search(
            system, user, schema, search, max_searches))
