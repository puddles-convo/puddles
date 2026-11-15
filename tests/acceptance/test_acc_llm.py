"""HTTP contract tested against a loopback http.server, never a real endpoint."""

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import socket
import threading
import time

import pytest


@pytest.fixture
def api():
    return importlib.import_module("loop.llm")


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    state = dict(status=200, body=b'{"choices":[{"message":{"content":"A reply."}}]}',
                 delay=0, disconnect=False, declared_length=None, calls=[])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["calls"].append(dict(path=self.path, headers=self.headers, body=json.loads(body)))
            if state["disconnect"]:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if state["delay"]:
                time.sleep(state["delay"])
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            length = state["declared_length"] if state["declared_length"] is not None else len(state["body"])
            self.send_header("Content-Length", str(length))
            self.end_headers()
            try:
                self.wfile.write(state["body"])
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the timeout test closes the connection.

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_http_post_payload_reply_and_no_authorization_by_default(api, endpoint):
    client = api.OpenAICompatClient(endpoint["url"], "test-model")
    messages = [{"role": "user", "content": "Café?"}, {"role": "assistant", "content": "Yes."}]
    before = deepcopy(messages)
    assert client.complete("System first.", messages, max_tokens=43) == "A reply."
    assert len(endpoint["calls"]) == 1
    request = endpoint["calls"][0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"].get("Authorization") is None
    assert request["body"] == dict(
        model="test-model", messages=[{"role": "system", "content": "System first."}] + before,
        max_tokens=43, temperature=0.7,
    )


def test_http_explicit_options_auth_and_extra_body(api, endpoint):
    # A dummy string, not an API key; it only reaches the test-owned HTTP server.
    client = api.OpenAICompatClient(endpoint["url"], "test-model", api_key="acceptance-dummy",
                                     extra_body={"stream": False, "chat_template_kwargs": {"enable_thinking": False}})
    client.complete("S", [], max_tokens=17, temperature=0.2, seed=0)
    request = endpoint["calls"][0]
    assert request["headers"]["Authorization"] == "Bearer acceptance-dummy"
    assert request["body"] == dict(
        model="test-model", messages=[{"role": "system", "content": "S"}],
        max_tokens=17, temperature=0.2, seed=0, stream=False,
        chat_template_kwargs={"enable_thinking": False},
    )


@pytest.mark.parametrize("content", ["", " \nCafé.\t "])
def test_http_returns_content_without_stripping(api, endpoint, content):
    endpoint["body"] = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    assert api.OpenAICompatClient(endpoint["url"], "m").complete("s", [], max_tokens=9) == content


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_http_failures_raise_llm_error(api, endpoint, status):
    from loop.errors import LLMError
    endpoint["status"] = status
    with pytest.raises(LLMError):
        api.OpenAICompatClient(endpoint["url"], "m").complete("s", [], max_tokens=1)


@pytest.mark.parametrize("body", [
    b"not JSON", b"{}", b'{"choices":[]}',
    b'{"choices":[{"message":{}}]}', b'{"choices":[{"message":{"content":null}}]}',
    b'{"choices":[{"message":{"content":42}}]}',
])
def test_unexpected_response_raises_llm_error(api, endpoint, body):
    from loop.errors import LLMError
    endpoint["body"] = body
    with pytest.raises(LLMError):
        api.OpenAICompatClient(endpoint["url"], "m").complete("s", [], max_tokens=1)


def test_transport_disconnect_raises_llm_error(api, endpoint):
    from loop.errors import LLMError
    endpoint["disconnect"] = True
    with pytest.raises(LLMError):
        api.OpenAICompatClient(endpoint["url"], "m").complete("s", [], max_tokens=1)


def test_truncated_http_body_raises_llm_error(api, endpoint):
    from loop.errors import LLMError
    endpoint["declared_length"] = len(endpoint["body"]) + 50
    with pytest.raises(LLMError):
        api.OpenAICompatClient(endpoint["url"], "m").complete("s", [], max_tokens=1)


def test_response_exceeding_json_parser_depth_raises_llm_error(api, endpoint):
    from loop.errors import LLMError
    # This is only 20 KB. A server-controlled unexpected body must not expose
    # the decoder's RecursionError or abort the batch.
    endpoint["body"] = b"[" * 10000 + b"0" + b"]" * 10000
    with pytest.raises(LLMError):
        api.OpenAICompatClient(endpoint["url"], "m").complete("s", [], max_tokens=1)


def test_request_encoding_never_leaks_encoder_exception(api, endpoint):
    from loop.errors import LLMError
    nested = 0
    for _ in range(10000):
        nested = [nested]
    client = api.OpenAICompatClient(endpoint["url"], "m", extra_body={"extension": nested})
    # Every value has a JSON-compatible type. Encoding may reject the depth,
    # but that is a request failure and must stay inside the LLMError boundary.
    try:
        reply = client.complete("s", [], max_tokens=1)
    except LLMError:
        return
    assert reply == "A reply."


@pytest.mark.parametrize("failed_role", ["helper", "user"])
def test_truncated_transport_fails_conversation_and_batch_continues(
    api, endpoint, tmp_path, policy, scenario_factory, client_factory, failed_role,
):
    from loop.runner import run_batch
    endpoint["declared_length"] = len(endpoint["body"]) + 50
    broken = api.OpenAICompatClient(endpoint["url"], "m")
    helper = broken if failed_role == "helper" else client_factory([])
    user = broken if failed_role == "user" else client_factory(["Worry A", "Worry B"])
    out = tmp_path / "records.jsonl"
    result = run_batch(helper, user, policy, [scenario_factory("a"), scenario_factory("b")],
                       round=1, out_path=out)
    assert result.records == []
    assert [failure.scenario_id for failure in result.failures] == ["a", "b"]
    assert all(failure.reason.strip() for failure in result.failures)
    assert out.read_text() == ""
    assert len(endpoint["calls"]) == 2


def test_timeout_raises_llm_error(api, endpoint):
    from loop.errors import LLMError
    endpoint["delay"] = 0.25
    with pytest.raises(LLMError):
        api.OpenAICompatClient(endpoint["url"], "m", timeout=0.03).complete("s", [], max_tokens=1)


def test_fake_list_records_calls_defaults_and_exhaustion(api):
    from loop.errors import LLMError
    client = api.FakeClient(["First", "Second"])
    messages = [{"role": "user", "content": "Hi"}]
    assert client.complete("S", messages, max_tokens=9) == "First"
    assert client.complete("T", [], max_tokens=12, temperature=0.1, seed=8) == "Second"
    assert client.calls == [
        dict(system="S", messages=messages, max_tokens=9, temperature=0.7, seed=None),
        dict(system="T", messages=[], max_tokens=12, temperature=0.1, seed=8),
    ]
    with pytest.raises(LLMError):
        client.complete("U", [], max_tokens=1)
    assert len(client.calls) == 3  # Every call, including the exhausted one.


def test_fake_callable_uses_system_and_messages_and_records_call(api):
    client = api.FakeClient(lambda system, messages: system + ": " + messages[-1]["content"])
    assert client.complete("Echo", [{"role": "user", "content": "hello"}], max_tokens=5, seed=2) == "Echo: hello"
    assert client.calls == [dict(system="Echo", messages=[{"role": "user", "content": "hello"}],
                                 max_tokens=5, temperature=0.7, seed=2)]


@pytest.mark.parametrize("prefix", ["HELPER", "SIMUSER"])
@pytest.mark.parametrize("missing", ["URL", "MODEL"])
def test_missing_environment_variable_is_named(api, clean_env, monkeypatch, prefix, missing):
    from loop.errors import ConfigError
    if missing != "URL":
        monkeypatch.setenv(f"{prefix}_URL", "http://127.0.0.1:1/v1")
    if missing != "MODEL":
        monkeypatch.setenv(f"{prefix}_MODEL", "test-model")
    with pytest.raises(ConfigError) as caught:
        api.client_from_env(prefix)
    assert f"{prefix}_{missing}" in str(caught.value)


@pytest.mark.parametrize("prefix", ["HELPER", "SIMUSER"])
@pytest.mark.parametrize("with_key", [False, True])
def test_environment_factory_creates_working_client(api, clean_env, monkeypatch, endpoint, prefix, with_key):
    monkeypatch.setenv(f"{prefix}_URL", endpoint["url"])
    monkeypatch.setenv(f"{prefix}_MODEL", "chosen-model")
    if with_key:
        monkeypatch.setenv(f"{prefix}_API_KEY", "acceptance-dummy")
    client = api.client_from_env(prefix)
    assert isinstance(client, api.OpenAICompatClient)
    assert client.complete("s", [], max_tokens=3) == "A reply."
    request = endpoint["calls"][0]
    assert request["body"]["model"] == "chosen-model"
    assert request["headers"].get("Authorization") == ("Bearer acceptance-dummy" if with_key else None)
