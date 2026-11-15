"""Chat model clients: the protocol, an OpenAI-compatible HTTP client and a fake."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

from loop.errors import ConfigError, LLMError


class ChatClient(Protocol):
    """Anything that turns a system message and chat messages into reply text."""

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str: ...


class OpenAICompatClient:
    """Client for a server exposing POST {base_url}/chat/completions."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 60.0, extra_body: dict[str, Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str:
        """Send one chat request. Any failure, from encoding to parsing, is an LLMError."""
        try:
            return self._complete(system, messages, max_tokens=max_tokens,
                                  temperature=temperature, seed=seed)
        except LLMError:
            raise
        except urllib.error.HTTPError as exc:
            raise LLMError(f"HTTP {exc.code} from {self.base_url}") from exc
        except Exception as exc:
            raise LLMError(f"request to {self.base_url} failed: "
                           f"{type(exc).__name__}: {exc}") from exc

    def _complete(self, system: str, messages: list[dict[str, str]], *,
                  max_tokens: int, temperature: float, seed: int | None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        body.update(self.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions",
                                         data=json.dumps(body).encode("utf-8"),
                                         headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        try:
            content = json.loads(raw)["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LLMError(f"unexpected response body from {self.base_url}") from exc
        if not isinstance(content, str):
            raise LLMError(f"unexpected response body from {self.base_url}")
        return content


Replies = list[str] | Callable[[str, list[dict[str, str]]], str]


class FakeClient:
    """Scripted client for tests: a list of replies in order, or a callable."""

    def __init__(self, replies: Replies) -> None:
        self._replies = replies if callable(replies) else list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str:
        messages = [dict(m) for m in messages]
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "seed": seed})
        if callable(self._replies):
            return self._replies(system, messages)
        if not self._replies:
            raise LLMError("FakeClient has no replies left")
        return self._replies.pop(0)


def client_from_env(prefix: str) -> OpenAICompatClient:
    """Build a client from {prefix}_URL, {prefix}_MODEL and optional {prefix}_API_KEY."""
    values = {}
    for name in (f"{prefix}_URL", f"{prefix}_MODEL"):
        value = os.environ.get(name)
        if not value:
            raise ConfigError(f"environment variable {name} is not set")
        values[name] = value
    return OpenAICompatClient(values[f"{prefix}_URL"], values[f"{prefix}_MODEL"],
                              api_key=os.environ.get(f"{prefix}_API_KEY") or None)
