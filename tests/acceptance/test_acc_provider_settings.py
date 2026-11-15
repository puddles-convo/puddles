"""Independent provider contract checks; HTTP is captured or handled in memory."""
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType

import httpx
import mlflow
import openai
import pytest
from pydantic import BaseModel

from loop import llm as compat
from loop.errors import ConfigError
from puddles import exa, tracing
from puddles.agents import researcher
from puddles.decision_log import DecisionLog
from puddles.llm import AgentFailed, OpenAILLM, call_checked
from puddles.usage import Meter, ModelPrice, Prices, current, metering, read_usage
from schemas import PatternBrief, ResearcherInput

ROOT = Path(__file__).resolve().parents[2]
MODEL = "acceptance-provider"
QUERY = "open questions before advice"
URL = "https://example.invalid/counselling"


class Answer(BaseModel):
    answer: str


def completion(content=None, *, tool=False, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [{"id": "search-1", "type": "function", "function": {
            "name": "search", "arguments": json.dumps({"query": QUERY})}}]
    return {"id": "acceptance", "object": "chat.completion", "created": 0, "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("PUDDLES_AGENT_TEMPERATURE", "PUDDLES_AGENT_TOOL_REASONING_EFFORT", "PUDDLES_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    for prefix in ("HELPER", "SIMUSER", "ACCEPTANCE"):
        for suffix in ("MAX_TOKENS_PARAM", "EXTRA_BODY", "API_KEY"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    # This replaces the SDK constructor for all tests, including accidental extra calls.
    # The actual SDK still serializes, parses and raises its own error types.
    original = openai.OpenAI
    clients = []

    def install(handler):
        def factory(*args, **kwargs):
            client = original(api_key="acceptance-dummy", base_url="http://127.0.0.1/v1",
                              max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
            clients.append(client)
            return client
        monkeypatch.setattr(openai, "OpenAI", factory)

    def unexpected(request):
        pytest.fail(f"Unexpected SDK request: {request.method} {request.url}")

    install(unexpected)
    yield install
    for client in clients:
        client.close()


@pytest.fixture
def traced(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


def load_head(monkeypatch, path, name):
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{path}"],
                            check=True, capture_output=True, text=True).stdout
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, f"HEAD:{path}", "exec"), module.__dict__)
    return module


def capture_http(monkeypatch):
    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def send(request, timeout):
        seen.append((request.full_url, dict(request.header_items()), request.data, timeout))
        return Response(json.dumps(completion('{"answer":"ok"}')).encode())

    monkeypatch.setattr(compat.urllib.request, "urlopen", send)
    return seen


def invoke(client, method):
    args = ("system", [{"role": "user", "content": "hello"}])
    kwargs = dict(max_tokens=41, temperature=0.9, seed=7)
    if method == "complete_json":
        return client.complete_json(*args, Answer, **kwargs)
    return getattr(client, method)(*args, **kwargs)


@pytest.mark.parametrize("method", ["complete", "complete_json", "request"])
def test_unset_compat_requests_match_head_bytes(monkeypatch, method):
    old = load_head(monkeypatch, "loop/llm.py", "_acceptance_head_compat")
    seen = capture_http(monkeypatch)
    for module in (old, compat):
        monkeypatch.setenv("HELPER_URL", "http://127.0.0.1/v1/")
        monkeypatch.setenv("HELPER_MODEL", MODEL)
        monkeypatch.setenv("HELPER_EXTRA_BODY", '{"temperature":1,"extension":{"x":[1]}}')
        invoke(module.client_from_env("HELPER"), method)
    assert len(seen) == 2 and seen[0] == seen[1]


@pytest.mark.parametrize("prefix", ["HELPER", "SIMUSER", "ACCEPTANCE"])
@pytest.mark.parametrize("setting", [None, "", "max_tokens", "max_completion_tokens"])
def test_environment_limit_field_on_all_public_methods(monkeypatch, prefix, setting):
    seen = capture_http(monkeypatch)
    monkeypatch.setenv(f"{prefix}_URL", "http://127.0.0.1/v1")
    monkeypatch.setenv(f"{prefix}_MODEL", MODEL)
    if setting is not None:
        monkeypatch.setenv(f"{prefix}_MAX_TOKENS_PARAM", setting)
    client = compat.client_from_env(prefix)
    for method in ("complete", "complete_json", "request"):
        invoke(client, method)
    field = setting or "max_tokens"
    other = "max_tokens" if field == "max_completion_tokens" else "max_completion_tokens"
    assert len(seen) == 3
    for _, _, raw, _ in seen:
        body = json.loads(raw)
        assert body[field] == 41 and other not in body


@pytest.mark.parametrize("setting", ["none", "MAX_TOKENS", "max_tokens ", " "])
def test_invalid_environment_limit_is_rejected_before_io(monkeypatch, setting):
    seen = capture_http(monkeypatch)
    monkeypatch.setenv("ACCEPTANCE_URL", "http://127.0.0.1/v1")
    monkeypatch.setenv("ACCEPTANCE_MODEL", MODEL)
    monkeypatch.setenv("ACCEPTANCE_MAX_TOKENS_PARAM", setting)
    with pytest.raises(ConfigError) as caught:
        compat.client_from_env("ACCEPTANCE")
    assert all(s in str(caught.value) for s in (
        "ACCEPTANCE_MAX_TOKENS_PARAM", "max_tokens", "max_completion_tokens"))
    assert seen == []


def test_extra_body_overrides_all_collisions_and_passes_unknown_fields(monkeypatch):
    seen = capture_http(monkeypatch)
    extra = {"model": "replacement", "messages": [], "temperature": None, "seed": 99,
             "max_completion_tokens": 88, "response_format": {"unknown": [False, None]}}
    client = compat.OpenAICompatClient("http://127.0.0.1/v1", MODEL, extra_body=extra,
                                       max_tokens_param="max_completion_tokens")
    client.request("system", [], max_tokens=1, temperature=0.7, seed=3,
                   response_format={"type": "json_object"})
    assert json.loads(seen[0][2]) == extra


def sdk_llm(install, handler, cls=OpenAILLM):
    install(handler)
    return cls(model=MODEL, output_limits={"default": 123, "researcher": 234})


@pytest.mark.parametrize("effort", [None, "", "none", "high"])
def test_reasoning_effort_is_limited_to_every_tool_request(isolated, monkeypatch, effort):
    if effort is not None:
        monkeypatch.setenv("PUDDLES_AGENT_TOOL_REASONING_EFFORT", effort)
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json=completion('{"answer":"ok"}', tool="tools" in body))

    llm = sdk_llm(isolated, respond)
    llm.structured_with_search("system", "user", Answer, lambda q: [], 2)
    llm.structured("system", "user", Answer)
    assert ["tools" in b for b in bodies] == [True, True, False, False]
    for body in bodies:
        if effort and "tools" in body:
            assert body["reasoning_effort"] == effort
        else:
            assert "reasoning_effort" not in body


def test_unset_agent_requests_match_head_through_real_sdk(isolated, monkeypatch):
    old = load_head(monkeypatch, "puddles/llm.py", "_acceptance_head_agents")
    batches = []
    for cls in (old.OpenAILLM, OpenAILLM):
        bodies = []

        def respond(request):
            body = json.loads(request.content)
            bodies.append(body)
            return httpx.Response(200, json=completion('{"answer":"ok"}', tool="tools" in body))

        llm = sdk_llm(isolated, respond, cls)
        llm.structured("system", "user", Answer)
        llm.structured_with_search("system", "user", Answer, lambda q: [], 2)
        batches.append(deepcopy(bodies))
    assert len(batches[0]) == 4 and batches[0] == batches[1]


@pytest.mark.parametrize("where", ["structured", "first_tool", "later_tool", "final"])
@pytest.mark.parametrize("metered", [False, True])
def test_sdk_provider_failure_preserves_cause_message_and_usage(isolated, tmp_path, traced, where, metered):
    bodies = []
    message = "provider refused sk-key-like-text: " + "x" * 400

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        fail = (where in ("structured", "first_tool") or
                where == "later_tool" and len(bodies) == 2 or where == "final" and "tools" not in body)
        if fail:
            return httpx.Response(400, json={"error": {"message": message, "type": "invalid_request_error"}})
        return httpx.Response(200, json=completion(tool=True))

    llm = sdk_llm(isolated, respond)
    meter = Meter(tmp_path, Prices({MODEL: ModelPrice(1, 1)}), round=9) if metered else None
    agent = "critic" if where == "structured" else "researcher"
    with metering(meter, agent), pytest.raises(AgentFailed) as caught:
        if where == "structured":
            llm.structured("system", "user", Answer)
        else:
            llm.structured_with_search("system", "user", Answer, lambda q: [], 2)
    cause = caught.value.__cause__
    assert isinstance(cause, openai.BadRequestError)
    reason = str(caught.value)
    assert type(cause).__name__ in reason and reason.endswith(str(cause)[:300])
    assert str(cause)[:301] not in reason
    rows = read_usage(tmp_path / "log/usage.jsonl")
    if metered:
        failed = [r for r in rows if r["event"] == "failed"]
        assert [(r["agent"], r["round"], r["reason"]) for r in failed] == [(agent, 9, reason)]
        sent = [r for r in rows if r["event"] == "sent"]
        done = [r for r in rows if r["event"] == "done"]
        assert len(sent) == len(bodies) and len(done) == len(sent) - 1
        assert sent[-1]["call_id"] not in {r["call_id"] for r in done}
    else:
        assert rows == []


@pytest.mark.parametrize("metered", [False, True])
def test_length_error_keeps_reason_cause_and_completed_usage(isolated, tmp_path, traced, metered):
    llm = sdk_llm(isolated, lambda r: httpx.Response(200, json=completion('{"answer":', finish="length")))
    meter = Meter(tmp_path, Prices({MODEL: ModelPrice(1, 1)})) if metered else None
    with metering(meter, "proposer"), pytest.raises(AgentFailed) as caught:
        llm.structured("system", "user", Answer)
    assert isinstance(caught.value.__cause__, openai.LengthFinishReasonError)
    assert str(caught.value) == "the answer was cut off at the output limit before it was a whole Answer"
    rows = read_usage(tmp_path / "log/usage.jsonl")
    assert [r["event"] for r in rows] == (["sent", "done", "failed"] if metered else [])


@pytest.mark.parametrize("error", [openai.OpenAIError("SDK failure"), KeyboardInterrupt(), SystemExit(7)])
def test_base_sdk_error_is_wrapped_but_process_controls_escape(isolated, error):
    llm = sdk_llm(isolated, lambda r: pytest.fail("No SDK call expected"))

    def send(**kwargs):
        raise error

    expected = AgentFailed if isinstance(error, openai.OpenAIError) else type(error)
    with pytest.raises(expected) as caught:
        llm._request(send, [])
    if expected is AgentFailed:
        assert caught.value.__cause__ is error and "OpenAIError: SDK failure" in str(caught.value)
    else:
        assert caught.value is error


@pytest.mark.parametrize("prior_failed_check", [False, True])
def test_call_checked_never_retries_agent_failed(tmp_path, prior_failed_check):
    calls = []
    error = AgentFailed("provider refused")

    def produce(feedback):
        calls.append(feedback)
        if prior_failed_check and len(calls) == 1:
            return Answer(answer="bad")
        raise error

    meter = Meter(tmp_path)
    with metering(meter, "critic"), pytest.raises(AgentFailed) as caught:
        call_checked(produce, lambda out: ["bad answer"], "critic")
    assert caught.value is error and len(calls) == (2 if prior_failed_check else 1)
    assert read_usage(tmp_path / "log/usage.jsonl") == []


def test_exa_failure_inside_tool_loop_closes_search_and_model_usage(isolated, monkeypatch, tmp_path, traced):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(tool=True))

    exa_calls = []

    def refused(url, **kwargs):
        exa_calls.append(url)
        return httpx.Response(503, request=httpx.Request("POST", url), text="unavailable")

    monkeypatch.setenv("EXA_API_KEY", "acceptance-dummy")
    monkeypatch.setattr(exa.httpx, "post", refused)
    llm = sdk_llm(isolated, respond)
    meter = Meter(tmp_path, Prices({MODEL: ModelPrice(1, 1)}, exa_search_usd=0.01))
    inp = ResearcherInput(pattern=PatternBrief(id="p1", name="n", description="d"), earlier_queries=[])
    with metering(meter, "researcher"), pytest.raises(AgentFailed, match="the search failed: Exa search failed"):
        researcher.run(inp, llm, exa.search, DecisionLog(tmp_path))
    assert len(requests) == 1 and len(exa_calls) == 2
    raw = (tmp_path / "log/usage.jsonl").read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert raw.endswith("\n")
    sent = [r for r in rows if r["event"] == "sent"]
    done = [r for r in rows if r["event"] == "done"]
    assert len(sent) == len(done) == 3
    assert {r["call_id"] for r in sent} == {r["call_id"] for r in done}
    assert all(r["agent"] == "researcher" for r in rows)
    assert all("error" in r for r in done if r["kind"] == "search")


@pytest.mark.parametrize("failure", ["critic", "researcher", "proposer", "search"])
def test_cli_finishes_and_logs_each_agent_failure(isolated, monkeypatch, tmp_path, failure, capsys):
    from scripts import run_graph_round as cli

    def respond(request):
        body = json.loads(request.content)
        agent = current()[1]
        if agent == failure:
            return httpx.Response(400, json={"error": {"message": "provider refused this agent"}})
        if agent == "critic":
            user = body["messages"][-1]["content"]
            ids = re.findall(r'<conversation id="([^"]+)"', user)
            answer = dict(found=True, name="advice-before-understanding", description="Advice before questions.",
                          seen_in=1, out_of=len(ids), example_conversation_ids=ids[:1])
        elif "tools" in body:
            return httpx.Response(200, json=completion(tool=not any(m["role"] == "tool" for m in body["messages"])))
        else:
            answer = {"queries": [QUERY], "sources": [{"url": URL, "title": "Skills", "technique": "Ask first."}]}
        return httpx.Response(200, json=completion(json.dumps(answer)))

    llm = sdk_llm(isolated, respond)
    monkeypatch.setattr(cli, "FakeLLM", lambda: llm)

    def search(query):
        if failure == "search":
            raise RuntimeError("Exa unavailable twice")
        return [{"url": URL, "title": "Skills", "text": "Ask first."}]

    monkeypatch.setattr(cli, "fake_search", search)
    # Supply prices through the existing loader interface.
    from puddles.config import RunConfig
    import puddles.graph
    monkeypatch.setattr(puddles.graph, "load_config", lambda: RunConfig(
        Prices({MODEL: ModelPrice(0, 0)}), {"default": 123}))
    result = cli.main(["--fake", "--root", str(tmp_path / "run"), "--policy",
                       str(ROOT / "policies/policy-v1.md"), "--scenarios", str(ROOT / "fixtures/scenarios.jsonl"),
                       "--max-turns", "1", "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"])
    mlflow.flush_trace_async_logging()
    assert result == 0
    events = list(DecisionLog(tmp_path / "run").rounds())
    assert len(events) == 1 and events[0].edit is None
    event = events[0]
    assert event.outcome == ("no_evidence" if failure in ("researcher", "search") else "no_pattern")
    assert f"{'researcher' if failure == 'search' else failure} failed:" in event.reason
    assert ("Exa unavailable twice" if failure == "search" else "provider refused this agent") in event.reason
    assert f"round 1: {event.outcome}" in capsys.readouterr().out


def test_cli_refuses_missing_exa_key_before_round(isolated, monkeypatch, tmp_path, capsys):
    from scripts import run_graph_round as cli

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(compat, "client_from_env", lambda prefix: compat.FakeClient([]))
    monkeypatch.setattr(cli, "run_round", lambda *a, **k: pytest.fail("Round must not start"))
    assert cli.main(["--root", str(tmp_path), "--policy", str(ROOT / "policies/policy-v1.md"),
                     "--scenarios", str(ROOT / "fixtures/scenarios.jsonl")]) == 2
    assert "EXA_API_KEY" in capsys.readouterr().err
    assert list(DecisionLog(tmp_path).rounds()) == []
