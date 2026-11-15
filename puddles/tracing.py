"""MLflow tracing: where traces go, and a span around every chat-model call.

`loop.llm` talks to its models over plain HTTP, so MLflow's automatic OpenAI tracing does not
see those calls. `TracedClient` wraps any `ChatClient` and opens one span per call instead.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from loop.llm import ChatClient

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


class TracedClient:
    """A ChatClient that records each call as an MLflow span named llm:<name>."""

    def __init__(self, inner: ChatClient, name: str) -> None:
        self.inner = inner
        self.name = name

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str:
        with mlflow.start_span(name=f"llm:{self.name}", span_type=SpanType.CHAT_MODEL) as span:
            span.set_inputs({"system": system, "messages": [dict(m) for m in messages],
                             "max_tokens": max_tokens, "temperature": temperature, "seed": seed})
            text = self.inner.complete(system, messages, max_tokens=max_tokens,
                                       temperature=temperature, seed=seed)
            span.set_outputs(text)
            return text
