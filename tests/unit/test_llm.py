import http.client
import io
import json
import socket
import urllib.error

import pytest

from loop import llm
from loop.errors import ConfigError, LLMError
from loop.llm import FakeClient, OpenAICompatClient, client_from_env

MESSAGES = [{"role": "user", "content": "hi"}]


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, body=None, raises=None):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        if raises is not None:
            raise raises
        return _Response(body if isinstance(body, bytes) else json.dumps(body).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return seen


def _ok(content="Hello there."):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_openai_client_sends_expected_request(monkeypatch):
    seen = _patch_urlopen(monkeypatch, _ok())
    client = OpenAICompatClient("http://host/v1/", "m", api_key="k", timeout=5.0,
                                extra_body={"top_p": 0.5})
    assert client.complete("sys", MESSAGES, max_tokens=50, temperature=0.2, seed=3) == "Hello there."
    request = seen["request"]
    assert request.full_url == "http://host/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer k"
    assert seen["timeout"] == 5.0
    assert json.loads(request.data) == {
        "model": "m",
        "messages": [{"role": "system", "content": "sys"}, *MESSAGES],
        "max_tokens": 50, "temperature": 0.2, "seed": 3, "top_p": 0.5,
    }


def test_openai_client_omits_seed_and_auth_when_absent(monkeypatch):
    seen = _patch_urlopen(monkeypatch, _ok())
    OpenAICompatClient("http://host/v1", "m").complete("sys", MESSAGES, max_tokens=10)
    body = json.loads(seen["request"].data)
    assert "seed" not in body
    assert body["temperature"] == 0.7
    assert seen["request"].get_header("Authorization") is None


@pytest.mark.parametrize("raises", [
    urllib.error.HTTPError("http://host/v1/chat/completions", 500, "boom", {}, None),
    urllib.error.URLError("refused"),
    socket.timeout("timed out"),
    TimeoutError("timed out"),
])
def test_openai_client_transport_failures_raise_llm_error(monkeypatch, raises):
    _patch_urlopen(monkeypatch, raises=raises)
    with pytest.raises(LLMError):
        OpenAICompatClient("http://host/v1", "m").complete("s", MESSAGES, max_tokens=10)


class _BrokenResponse(_Response):
    def __init__(self, exc):
        super().__init__(b"")
        self._exc = exc

    def read(self, *args):
        raise self._exc


@pytest.mark.parametrize("read_error", [
    http.client.IncompleteRead(b'{"choi', 40),
    http.client.RemoteDisconnected("closed"),
    ConnectionResetError("reset"),
    socket.timeout("timed out while reading"),
])
def test_openai_client_failures_while_reading_body_raise_llm_error(monkeypatch, read_error):
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda request, timeout: _BrokenResponse(read_error))
    with pytest.raises(LLMError):
        OpenAICompatClient("http://host/v1", "m").complete("s", MESSAGES, max_tokens=10)


@pytest.mark.parametrize("raises", [http.client.BadStatusLine("garbage"),
                                    http.client.RemoteDisconnected("closed")])
def test_openai_client_connection_level_http_errors_raise_llm_error(monkeypatch, raises):
    _patch_urlopen(monkeypatch, raises=raises)
    with pytest.raises(LLMError):
        OpenAICompatClient("http://host/v1", "m").complete("s", MESSAGES, max_tokens=10)


def _nested(depth):
    value = 0
    for _ in range(depth):
        value = [value]
    return value


def test_openai_client_too_deeply_nested_response_raises_llm_error(monkeypatch):
    _patch_urlopen(monkeypatch, b"[" * 10000 + b"0" + b"]" * 10000)
    with pytest.raises(LLMError):
        OpenAICompatClient("http://host/v1", "m").complete("s", MESSAGES, max_tokens=10)


@pytest.mark.parametrize("extra", [{"extension": _nested(10000)}, {"extension": object()}])
def test_openai_client_unencodable_request_raises_llm_error(monkeypatch, extra):
    seen = _patch_urlopen(monkeypatch, _ok())
    with pytest.raises(LLMError):
        OpenAICompatClient("http://host/v1", "m", extra_body=extra).complete(
            "s", MESSAGES, max_tokens=10)
    assert "request" not in seen  # nothing was sent


def test_openai_client_any_other_failure_raises_llm_error(monkeypatch):
    _patch_urlopen(monkeypatch, raises=RuntimeError("something unforeseen"))
    with pytest.raises(LLMError, match="RuntimeError"):
        OpenAICompatClient("http://host/v1", "m").complete("s", MESSAGES, max_tokens=10)


def test_openai_client_does_not_swallow_keyboard_interrupt(monkeypatch):
    _patch_urlopen(monkeypatch, raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        OpenAICompatClient("http://host/v1", "m").complete("s", MESSAGES, max_tokens=10)


def test_openai_client_unusable_base_url_raises_llm_error():
    with pytest.raises(LLMError):
        OpenAICompatClient("not-a-url/v1", "m").complete("s", MESSAGES, max_tokens=10)


@pytest.mark.parametrize("body", [b"not json", b"\xff\xfe", {"choices": []}, {"nope": 1},
                                  {"choices": [{"message": {"content": None}}]}])
def test_openai_client_unexpected_body_raises_llm_error(monkeypatch, body):
    _patch_urlopen(monkeypatch, body)
    with pytest.raises(LLMError):
        OpenAICompatClient("http://host/v1", "m").complete("s", MESSAGES, max_tokens=10)


def test_fake_client_returns_replies_in_order_and_records_calls():
    client = FakeClient(["one", "two"])
    assert client.complete("s1", MESSAGES, max_tokens=5) == "one"
    assert client.complete("s2", [], max_tokens=6, temperature=0.9, seed=4) == "two"
    assert client.calls == [
        {"system": "s1", "messages": MESSAGES, "max_tokens": 5, "temperature": 0.7, "seed": None},
        {"system": "s2", "messages": [], "max_tokens": 6, "temperature": 0.9, "seed": 4},
    ]
    with pytest.raises(LLMError):
        client.complete("s3", [], max_tokens=5)


def test_fake_client_accepts_a_callable():
    client = FakeClient(lambda system, messages: f"{system}:{len(messages)}")
    assert client.complete("s", MESSAGES, max_tokens=5) == "s:1"


def test_client_from_env_reads_variables(monkeypatch):
    monkeypatch.setenv("HELPER_URL", "http://host/v1")
    monkeypatch.setenv("HELPER_MODEL", "qwen")
    monkeypatch.setenv("HELPER_API_KEY", "secret")
    client = client_from_env("HELPER")
    assert isinstance(client, OpenAICompatClient)
    assert (client.base_url, client.model, client.api_key) == ("http://host/v1", "qwen", "secret")


def test_client_from_env_key_is_optional(monkeypatch):
    monkeypatch.setenv("SIMUSER_URL", "http://host/v1")
    monkeypatch.setenv("SIMUSER_MODEL", "qwen")
    monkeypatch.delenv("SIMUSER_API_KEY", raising=False)
    assert client_from_env("SIMUSER").api_key is None


@pytest.mark.parametrize("missing", ["HELPER_URL", "HELPER_MODEL"])
def test_client_from_env_missing_variable_names_it(monkeypatch, missing):
    monkeypatch.setenv("HELPER_URL", "http://host/v1")
    monkeypatch.setenv("HELPER_MODEL", "qwen")
    monkeypatch.delenv(missing)
    with pytest.raises(ConfigError, match=missing):
        client_from_env("HELPER")
