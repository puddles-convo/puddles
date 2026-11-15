"""Chat model clients: the protocol, an OpenAI-compatible HTTP client and a fake."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from loop.errors import ConfigError, DeadlineExceeded, LLMError, SchemaError

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class CallUsage:
    """The tokens of one request. `estimated` is true when the server sent no usage."""

    tokens_in: int
    tokens_out: int
    estimated: bool = False


def estimate_tokens(text: str) -> int:
    """Characters divided by four, rounded up."""
    return (len(text) + 3) // 4


def prompt_text(system: str, messages: list[dict[str, str]]) -> str:
    return system + "".join(str(m.get("content") or "") for m in messages)


def estimate_usage(system: str, messages: list[dict[str, str]], reply: str) -> CallUsage:
    return CallUsage(estimate_tokens(prompt_text(system, messages)), estimate_tokens(reply), estimated=True)


class ChatClient(Protocol):
    """Anything that turns a system message and chat messages into reply text."""

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str: ...


class JSONChatClient(ChatClient, Protocol):
    """A chat client that can also return a validated model."""

    def complete_json(self, system: str, messages: list[dict[str, str]], schema: type[T], *,
                      max_tokens: int, temperature: float = 0.0, seed: int | None = None,
                      deadline_s: float | None = None) -> T: ...


# One request. `complete_json_with` is given one of these, so a wrapper can trace each attempt.
Request = Callable[..., str]

REFUSED_STATUSES = (400, 422)  # how a server says it does not take `response_format`
JSON_ONLY = ("Reply with one JSON object that matches this JSON schema, and nothing else. "
             "No code fence, no explanation.\n{schema}")
FIX_REPLY = "That reply did not match the schema:\n{problems}\nReply again with one JSON object and nothing else."


def parse_json_reply(text: str, schema: type[T]) -> T:
    """Validate a reply as `schema`. A code fence or words around the object are dropped."""
    body = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fenced:
        body = fenced.group(1).strip()
    elif not body.startswith("{") and "{" in body and "}" in body:
        body = body[body.index("{"):body.rindex("}") + 1]
    return schema.model_validate_json(body)


def complete_json_with(request: Request, state: Any, system: str, messages: list[dict[str, str]],
                       schema: type[T], *, max_tokens: int, temperature: float = 0.0,
                       seed: int | None = None, deadline_s: float | None = None,
                       on_first_failure: Callable[[], None] | None = None) -> T:
    """Ask for `schema` through `request`, which sends one request and returns the reply text.

    The schema goes in `response_format`. If the server refuses that (HTTP 400 or 422), the same
    call is sent again with the schema in the system message, and `state.supports_response_format`
    is set to false so later calls skip the refused request. A reply that fails validation is
    retried once with the reasons; a second failure raises SchemaError.

    `deadline_s` covers the whole call, every request included. Each request is given the time
    left as its timeout, and an answer that arrives after the deadline is discarded. Either way
    the call raises DeadlineExceeded.
    """
    started = time.monotonic()
    schema_json = schema.model_json_schema()

    def left() -> float | None:
        return None if deadline_s is None else deadline_s - (time.monotonic() - started)

    def out_of_time() -> bool:
        remaining = left()
        return remaining is not None and remaining <= 0

    def send(msgs: list[dict[str, str]]) -> str:
        use_format = bool(getattr(state, "supports_response_format", True))
        while True:
            if out_of_time():
                raise DeadlineExceeded(f"no time left of the {deadline_s} seconds allowed")
            if use_format:
                sys_text = system
                fmt: dict[str, Any] | None = {
                    "type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema_json}}
            else:
                sys_text = system + "\n\n" + JSON_ONLY.format(schema=json.dumps(schema_json))
                fmt = None
            try:
                text = request(sys_text, msgs, max_tokens=max_tokens, temperature=temperature, seed=seed,
                               response_format=fmt, timeout=left())
            except LLMError as exc:
                if use_format and exc.status in REFUSED_STATUSES:
                    use_format = False
                    continue
                if out_of_time():
                    raise DeadlineExceeded(f"no answer within the {deadline_s} seconds allowed") from exc
                raise
            if out_of_time():
                raise DeadlineExceeded(f"the answer came after the {deadline_s} seconds allowed")
            if not use_format:
                try:
                    state.supports_response_format = False
                except AttributeError:
                    pass
            return text

    msgs = list(messages)
    for attempt in (1, 2):
        text = send(msgs)
        try:
            return parse_json_reply(text, schema)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'reply'}: {e['msg']}"
                                 for e in exc.errors())
            if attempt == 2:
                raise SchemaError(f"reply did not match {schema.__name__} twice: {problems}") from exc
            if on_first_failure is not None:
                on_first_failure()
            said = [{"role": "assistant", "content": text}] if text.strip() else []
            msgs = [*messages, *said, {"role": "user", "content": FIX_REPLY.format(problems=problems)}]
    raise AssertionError("unreachable")


class OpenAICompatClient:
    """Client for a server exposing POST {base_url}/chat/completions.

    After each request `last_usage` holds its tokens: read from the response's `usage`, or
    estimated from the text when the server sent none.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 60.0, extra_body: dict[str, Any] | None = None,
                 max_tokens_param: str = "max_tokens") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self.max_tokens_param = max_tokens_param  # the body field for the output limit
        self.last_usage: CallUsage | None = None
        self.supports_response_format = True

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str:
        """Send one chat request. Any failure, from encoding to parsing, is an LLMError."""
        return self.request(system, messages, max_tokens=max_tokens, temperature=temperature, seed=seed)

    def complete_json(self, system: str, messages: list[dict[str, str]], schema: type[T], *,
                      max_tokens: int, temperature: float | None = 0.0, seed: int | None = None,
                      deadline_s: float | None = None) -> T:
        """Return a validated `schema`. See `complete_json_with` for the fallback, retry and deadline.

        `temperature=None` sends no temperature, so the server uses its default."""
        return complete_json_with(self.request, self, system, messages, schema, max_tokens=max_tokens,
                                  temperature=temperature, seed=seed, deadline_s=deadline_s)

    def request(self, system: str, messages: list[dict[str, str]], *, max_tokens: int,
                temperature: float | None = 0.7, seed: int | None = None,
                response_format: dict[str, Any] | None = None, timeout: float | None = None) -> str:
        """One HTTP request. `timeout` can only shorten the client's own timeout."""
        self.last_usage = None
        try:
            return self._complete(system, messages, max_tokens=max_tokens, temperature=temperature,
                                  seed=seed, response_format=response_format, timeout=timeout)
        except LLMError:
            raise
        except urllib.error.HTTPError as exc:
            error = LLMError(f"HTTP {exc.code} from {self.base_url}")
            error.status = exc.code
            raise error from exc
        except Exception as exc:
            raise LLMError(f"request to {self.base_url} failed: "
                           f"{type(exc).__name__}: {exc}") from exc

    def _complete(self, system: str, messages: list[dict[str, str]], *,
                  max_tokens: int, temperature: float | None, seed: int | None,
                  response_format: dict[str, Any] | None = None, timeout: float | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            self.max_tokens_param: max_tokens,
        }
        if temperature is not None:  # None: no field, so the server's default applies
            body["temperature"] = temperature
        if seed is not None:
            body["seed"] = seed
        if response_format is not None:
            body["response_format"] = response_format
        body.update(self.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions",
                                         data=json.dumps(body).encode("utf-8"),
                                         headers=headers, method="POST")
        wait = self.timeout if timeout is None else min(self.timeout, timeout)
        with urllib.request.urlopen(request, timeout=wait) as response:
            raw = response.read()
        try:
            payload = json.loads(raw)
            content = payload["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LLMError(f"unexpected response body from {self.base_url}") from exc
        if not isinstance(content, str):
            raise LLMError(f"unexpected response body from {self.base_url}")
        self.last_usage = _read_usage(payload) or estimate_usage(system, messages, content)
        return content


def _read_usage(payload: Any) -> CallUsage | None:
    """The token counts in a response, or None when they are missing or not whole numbers."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None
    counts = (usage.get("prompt_tokens"), usage.get("completion_tokens"))
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts):
        return None
    return CallUsage(counts[0], counts[1])


Replies = list[str] | Callable[[str, list[dict[str, str]]], str]


class FakeClient:
    """Scripted client for tests: a list of replies in order, or a callable."""

    def __init__(self, replies: Replies) -> None:
        self._replies = replies if callable(replies) else list(replies)
        self.calls: list[dict[str, Any]] = []
        self.supports_response_format = True

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str:
        return self.request(system, messages, max_tokens=max_tokens, temperature=temperature, seed=seed)

    def complete_json(self, system: str, messages: list[dict[str, str]], schema: type[T], *,
                      max_tokens: int, temperature: float = 0.0, seed: int | None = None,
                      deadline_s: float | None = None) -> T:
        """The scripted replies are JSON text; the retry and the deadline work as on a real client."""
        return complete_json_with(self.request, self, system, messages, schema, max_tokens=max_tokens,
                                  temperature=temperature, seed=seed, deadline_s=deadline_s)

    def request(self, system: str, messages: list[dict[str, str]], *, max_tokens: int,
                temperature: float = 0.7, seed: int | None = None,
                response_format: dict[str, Any] | None = None, timeout: float | None = None) -> str:
        messages = [dict(m) for m in messages]
        call = {"system": system, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed}
        if response_format is not None:
            call["response_format"] = response_format
        self.calls.append(call)
        if callable(self._replies):
            return self._replies(system, messages)
        if not self._replies:
            raise LLMError("FakeClient has no replies left")
        return self._replies.pop(0)


def client_from_env(prefix: str) -> OpenAICompatClient:
    """Build a client from {prefix}_URL, {prefix}_MODEL and optional {prefix}_API_KEY.

    {prefix}_EXTRA_BODY, also optional, is a JSON object merged into every request body. It is
    where a server-specific switch goes, such as the one that turns a model's thinking mode off.
    {prefix}_MAX_TOKENS_PARAM, also optional, names the body field for the output limit:
    `max_tokens` (the default) or `max_completion_tokens`.
    """
    values = {}
    for name in (f"{prefix}_URL", f"{prefix}_MODEL"):
        value = os.environ.get(name)
        if not value:
            raise ConfigError(f"environment variable {name} is not set")
        values[name] = value
    extra_name = f"{prefix}_EXTRA_BODY"
    extra: Any = None
    if os.environ.get(extra_name):
        try:
            extra = json.loads(os.environ[extra_name])
        except ValueError as exc:
            raise ConfigError(f"environment variable {extra_name} is not valid JSON") from exc
        if not isinstance(extra, dict):
            raise ConfigError(f"environment variable {extra_name} must be a JSON object")
    param_name = f"{prefix}_MAX_TOKENS_PARAM"
    param = os.environ.get(param_name) or "max_tokens"
    if param not in ("max_tokens", "max_completion_tokens"):
        raise ConfigError(f"environment variable {param_name} must be max_tokens or max_completion_tokens")
    return OpenAICompatClient(values[f"{prefix}_URL"], values[f"{prefix}_MODEL"],
                              api_key=os.environ.get(f"{prefix}_API_KEY") or None, extra_body=extra,
                              max_tokens_param=param)
