"""docs/specs/provider-settings.md: provider settings and provider errors. All offline: no test calls a server."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import mlflow
import openai
import pytest
from openai.types.chat import ChatCompletion, ParsedChatCompletion

from loop import llm as loop_llm
from loop.errors import ConfigError
from loop.llm import OpenAICompatClient, client_from_env
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.agents import researcher
from puddles.decision_log import DecisionLog
from puddles.graph import RoundDeps, run_round
from puddles.llm import AgentFailed, OpenAILLM, call_checked
from puddles.memory import FakeMemory
from puddles.round import stand_in_gate
from puddles.usage import Meter, ModelPrice, Prices, metering, read_usage
from schemas import CriticOutput, PatternBrief, ResearcherInput, ResearcherOutput
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
MESSAGES = [{"role": "user", "content": "hi"}]
MODEL = "agent-model"
QUERY = "counselling skills exploring before giving advice"
URL = "https://counselling.example.ac.uk/skills"
SECRET = "sk-never-in-a-reason"
LONG = "Function tools with reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions. " * 6


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("PUDDLES_AGENT_TEMPERATURE", "PUDDLES_AGENT_TOOL_REASONING_EFFORT", "PUDDLES_CONFIG",
                 "HELPER_EXTRA_BODY", "HELPER_MAX_TOKENS_PARAM", "HELPER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)


@pytest.fixture
def traced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow keeps some trace files in ./mlruns
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


# ---------- loop.llm: the request body, captured at urlopen as tests/unit/test_llm.py does ----------

class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def capture(monkeypatch, content='{"found": false, "reason": "fine"}'):
    seen: list = []

    def fake_urlopen(request, timeout):
        seen.append(request)
        return _Response(json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode())

    monkeypatch.setattr(loop_llm.urllib.request, "urlopen", fake_urlopen)
    return seen


def body(request) -> dict:
    return json.loads(request.data)


def helper_env(monkeypatch, **extra):
    monkeypatch.setenv("HELPER_URL", "http://host/v1")
    monkeypatch.setenv("HELPER_MODEL", "m")
    for name, value in extra.items():
        monkeypatch.setenv(f"HELPER_{name}", value)


def test_constructor_default_sends_the_same_bytes_as_before(monkeypatch):
    seen = capture(monkeypatch)
    OpenAICompatClient("http://host/v1", "m").complete("sys", MESSAGES, max_tokens=50, temperature=0.7)
    before = {"model": "m", "messages": [{"role": "system", "content": "sys"}, *MESSAGES],
              "max_tokens": 50, "temperature": 0.7}
    assert seen[0].data == json.dumps(before).encode("utf-8")


def test_constructor_takes_max_completion_tokens(monkeypatch):
    seen = capture(monkeypatch)
    client = OpenAICompatClient("http://host/v1", "m", max_tokens_param="max_completion_tokens")
    client.complete("sys", MESSAGES, max_tokens=50)
    client.complete_json("sys", MESSAGES, CriticOutput, max_tokens=60)
    client.request("sys", MESSAGES, max_tokens=70)
    sent = [body(r) for r in seen]
    assert [b["max_completion_tokens"] for b in sent] == [50, 60, 70]
    assert not any("max_tokens" in b for b in sent)
    assert list(sent[0]) == ["model", "messages", "max_completion_tokens", "temperature"]


@pytest.mark.parametrize("value, field", [
    (None, "max_tokens"), ("", "max_tokens"),
    ("max_tokens", "max_tokens"), ("max_completion_tokens", "max_completion_tokens"),
], ids=["unset", "empty", "max_tokens", "max_completion_tokens"])
def test_max_tokens_param_names_the_body_field(monkeypatch, value, field):
    seen = capture(monkeypatch)
    helper_env(monkeypatch, **({} if value is None else {"MAX_TOKENS_PARAM": value}))
    client = client_from_env("HELPER")
    client.complete("sys", MESSAGES, max_tokens=80)
    client.complete_json("sys", MESSAGES, CriticOutput, max_tokens=90)
    other = ({"max_tokens", "max_completion_tokens"} - {field}).pop()
    assert [body(r)[field] for r in seen] == [80, 90]
    assert not any(other in body(r) for r in seen)


def test_unset_max_tokens_param_sends_the_same_bytes_as_the_default(monkeypatch):
    seen = capture(monkeypatch)
    helper_env(monkeypatch)
    client_from_env("HELPER").complete("sys", MESSAGES, max_tokens=80, temperature=0.9, seed=3)
    OpenAICompatClient("http://host/v1", "m").complete("sys", MESSAGES, max_tokens=80, temperature=0.9, seed=3)
    assert seen[0].data == seen[1].data


@pytest.mark.parametrize("value", ["max_output_tokens", "MAX_TOKENS", " max_tokens"])
def test_a_wrong_max_tokens_param_is_refused_when_the_client_is_built(monkeypatch, value):
    seen = capture(monkeypatch)
    helper_env(monkeypatch, MAX_TOKENS_PARAM=value)
    with pytest.raises(ConfigError) as caught:
        client_from_env("HELPER")
    message = str(caught.value)
    assert "HELPER_MAX_TOKENS_PARAM" in message
    assert "max_tokens" in message and "max_completion_tokens" in message
    assert seen == []


# ---------- section 2: EXTRA_BODY replaces the same field (behaviour that already existed) ----------

def test_extra_body_replaces_temperature_whatever_the_caller_passed(monkeypatch):
    seen = capture(monkeypatch)
    helper_env(monkeypatch, EXTRA_BODY='{"temperature": 1}')
    client = client_from_env("HELPER")
    client.complete("sys", MESSAGES, max_tokens=80, temperature=0.7)
    client.complete("sys", MESSAGES, max_tokens=80, temperature=0.9)
    client.complete_json("sys", MESSAGES, CriticOutput, max_tokens=80, temperature=0.0)
    assert [body(r)["temperature"] for r in seen] == [1, 1, 1]


def test_extra_body_fields_are_merged_without_inspection(monkeypatch):
    seen = capture(monkeypatch)
    helper_env(monkeypatch, MAX_TOKENS_PARAM="max_completion_tokens",
               EXTRA_BODY='{"temperature": 1, "reasoning_effort": "none", "anything": {"x": [1]}}')
    client_from_env("HELPER").complete("sys", MESSAGES, max_tokens=80, temperature=0.7)
    sent = body(seen[0])
    assert sent["reasoning_effort"] == "none" and sent["anything"] == {"x": [1]}
    assert sent["temperature"] == 1 and sent["max_completion_tokens"] == 80 and "max_tokens" not in sent


# ---------- puddles.llm.OpenAILLM: the SDK client replaced by a stand-in that records its arguments ----------

def completion(message: dict, finish: str = "stop") -> dict:
    return {"id": "c", "object": "chat.completion", "created": 0, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def tool_call(query=QUERY, call_id="t1") -> ChatCompletion:
    return ChatCompletion.model_validate(completion({"content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "search", "arguments": json.dumps({"query": query})}}
    ]}, "tool_calls"))


def no_tool_call() -> ChatCompletion:
    return ChatCompletion.model_validate(completion({"content": "done searching"}))


def parsed(schema, value: dict):
    return ParsedChatCompletion[schema].model_validate(completion({"content": json.dumps(value), "parsed": value}))


def bad_request(message: str = LONG) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions",
                            headers={"Authorization": f"Bearer {SECRET}"})
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return openai.BadRequestError(f"Error code: 400 - {message}", response=response, body=None)


def cut_off() -> openai.LengthFinishReasonError:
    return openai.LengthFinishReasonError(
        completion=ChatCompletion.model_validate(completion({"content": '{"found": tr'}, "length")))


class StandInClient:
    """`chat.completions.create` and `.parse`: each records its keyword arguments and gives the next answer."""

    def __init__(self, create=(), parse=()):
        self.calls: list[tuple[str, dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._send("create", list(create)),
                                                                parse=self._send("parse", list(parse))))

    def _send(self, kind, answers):
        def send(**kwargs):
            self.calls.append((kind, {**kwargs, "messages": list(kwargs["messages"])}))
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        return send


def agent_llm(create=(), parse=()) -> OpenAILLM:
    llm = OpenAILLM(model=MODEL, output_limits={"default": 900})
    llm.client = StandInClient(create, parse)
    return llm


FOUND_NOTHING = {"found": False, "reason": "The conversations went well."}
RESEARCH = {"queries": [QUERY], "sources": [{"url": URL, "title": "Skills", "technique": "Explore first."}]}


def search_once(llm: OpenAILLM, search=lambda q: [{"url": URL, "title": "Skills", "text": "..."}]):
    return llm.structured_with_search("sys", "user", ResearcherOutput, search, 3)


def test_tool_requests_carry_reasoning_effort_and_the_others_do_not(monkeypatch):
    monkeypatch.setenv("PUDDLES_AGENT_TOOL_REASONING_EFFORT", "none")
    llm = agent_llm(create=[tool_call(), no_tool_call()],
                    parse=[parsed(ResearcherOutput, RESEARCH), parsed(CriticOutput, FOUND_NOTHING)])
    search_once(llm)
    llm.structured("sys", "user", CriticOutput)
    kinds = [(kind, kwargs.get("reasoning_effort")) for kind, kwargs in llm.client.calls]
    assert kinds == [("create", "none"), ("create", "none"), ("parse", None), ("parse", None)]
    assert all("tools" in kwargs for kind, kwargs in llm.client.calls if kind == "create")
    assert not any("reasoning_effort" in kwargs for kind, kwargs in llm.client.calls if kind == "parse")


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_no_request_gets_reasoning_effort_when_the_variable_is_unset(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("PUDDLES_AGENT_TOOL_REASONING_EFFORT", value)
    llm = agent_llm(create=[tool_call(), no_tool_call()],
                    parse=[parsed(ResearcherOutput, RESEARCH), parsed(CriticOutput, FOUND_NOTHING)])
    search_once(llm)
    llm.structured("sys", "user", CriticOutput)
    keys = [(kind, sorted(kwargs)) for kind, kwargs in llm.client.calls]
    create_keys = ["max_completion_tokens", "messages", "model", "tools"]
    parse_keys = ["max_completion_tokens", "messages", "model", "response_format"]
    assert keys == [("create", create_keys), ("create", create_keys), ("parse", parse_keys), ("parse", parse_keys)]


# ---------- section 4: a provider error fails the agent ----------

# Where the error is raised: (create answers, parse answers) for structured_with_search, or structured.
WHERE = {
    "structured": ("structured", [], [bad_request()]),
    "first tool request": ("search", [bad_request()], []),
    "later tool request": ("search", [tool_call(), bad_request()], []),
    "final request": ("search", [tool_call(), no_tool_call()], [bad_request()]),
}


def run_where(where: str, error: BaseException):
    method, create, parse = WHERE[where]
    create = [error if isinstance(a, openai.BadRequestError) else a for a in create]
    parse = [error if isinstance(a, openai.BadRequestError) else a for a in parse]
    llm = agent_llm(create, parse)
    if method == "structured":
        return llm, lambda: llm.structured("sys", "user", CriticOutput)
    return llm, lambda: search_once(llm)


def lines(root: Path, event: str) -> list[dict]:
    return [line for line in read_usage(root / "log" / "usage.jsonl") if line.get("event") == event]


@pytest.mark.parametrize("where", list(WHERE))
def test_a_provider_error_becomes_agent_failed_without_a_meter(tmp_path, monkeypatch, where):
    monkeypatch.chdir(tmp_path)
    error = bad_request()
    llm, call = run_where(where, error)
    with pytest.raises(AgentFailed) as caught:
        call()
    reason = str(caught.value)
    assert caught.value.__cause__ is error
    assert "BadRequestError" in reason and MODEL in reason
    assert str(error)[:300] in reason and str(error)[:301] not in reason
    assert SECRET not in reason and "Authorization" not in reason
    assert not (tmp_path / "log").exists()


@pytest.mark.parametrize("where", list(WHERE))
def test_a_provider_error_writes_one_failed_line_and_no_done_line(tmp_path, traced, where):
    error = bad_request()
    llm, call = run_where(where, error)
    meter = Meter(tmp_path, Prices(models={MODEL: ModelPrice(0.0, 0.0)}), round=2)
    agent = "critic" if where == "structured" else "researcher"
    with metering(meter, agent), pytest.raises(AgentFailed) as caught:
        call()
    assert caught.value.__cause__ is error
    assert [(f["round"], f["agent"], f["reason"]) for f in lines(tmp_path, "failed")] == [(2, agent, str(caught.value))]
    sent, done = lines(tmp_path, "sent"), lines(tmp_path, "done")
    assert len(sent) == len(llm.client.calls) and len(done) == len(sent) - 1
    assert sent[-1]["call_id"] not in {d["call_id"] for d in done}


@pytest.mark.parametrize("error", [
    openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
    openai.OpenAIError("something else from the SDK"),
], ids=["APIConnectionError", "OpenAIError"])
def test_any_openai_error_is_named_in_the_reason(error):
    llm, call = run_where("structured", error)
    with pytest.raises(AgentFailed) as caught:
        call()
    assert type(error).__name__ in str(caught.value) and str(error) in str(caught.value)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("metered", [False, True], ids=["no meter", "meter"])
@pytest.mark.parametrize("where", ["structured", "final request"])
def test_length_finish_reason_error_keeps_its_own_handling(tmp_path, traced, metered, where):
    error = cut_off()
    llm, call = run_where(where, error)
    meter = Meter(tmp_path, Prices(models={MODEL: ModelPrice(0.0, 0.0)})) if metered else None
    schema = CriticOutput if where == "structured" else ResearcherOutput
    with metering(meter, "critic"), pytest.raises(AgentFailed) as caught:
        call()
    reason = f"the answer was cut off at the output limit before it was a whole {schema.__name__}"
    assert str(caught.value) == reason and caught.value.__cause__ is error
    if metered:
        assert [f["reason"] for f in lines(tmp_path, "failed")] == [reason]
        assert len(lines(tmp_path, "done")) == len(lines(tmp_path, "sent"))  # the cut-off answer is still counted
    else:
        assert not (tmp_path / "log").exists()


@pytest.mark.parametrize("metered", [False, True], ids=["no meter", "meter"])
@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(3)], ids=["KeyboardInterrupt", "SystemExit"])
@pytest.mark.parametrize("where", list(WHERE))
def test_keyboard_interrupt_and_system_exit_pass_through(tmp_path, traced, metered, error, where):
    llm, call = run_where(where, error)
    meter = Meter(tmp_path, Prices(models={MODEL: ModelPrice(0.0, 0.0)})) if metered else None
    with metering(meter, "researcher"), pytest.raises(type(error)):
        call()
    assert lines(tmp_path, "failed") == []


def test_call_checked_does_not_retry_an_agent_failed(tmp_path):
    calls = []

    def produce(feedback):
        calls.append(feedback)
        raise AgentFailed("the request failed")

    meter = Meter(tmp_path)
    with metering(meter, "critic"), pytest.raises(AgentFailed, match="the request failed"):
        call_checked(produce, lambda out: [], "critic")
    assert calls == [""]
    assert meter.usage.get("critic") is None and lines(tmp_path, "failed") == []


def test_a_provider_error_in_the_researcher_is_not_retried(tmp_path):
    error = bad_request()
    llm = agent_llm(create=[error])
    with pytest.raises(AgentFailed) as caught:
        researcher.run(research_input(), llm, FakeSearch({}), DecisionLog(tmp_path))
    assert caught.value.__cause__ is error and len(llm.client.calls) == 1


# ---------- section 5: a search that fails ----------

def failing_search(query):
    raise RuntimeError("Exa search failed: 503 Service Unavailable")


def research_input():
    return ResearcherInput(pattern=PatternBrief(id="p1", name="n", description="d"), earlier_queries=[])


@pytest.mark.parametrize("kind", ["scripted", "openai"])
def test_a_search_that_raises_runtime_error_fails_the_researcher(tmp_path, kind):
    if kind == "scripted":
        llm = ScriptedLLM()
        llm.add(ResearcherOutput, RESEARCH, RESEARCH)
    else:
        llm = agent_llm(create=[tool_call()])
    with pytest.raises(AgentFailed) as caught:
        researcher.run(research_input(), llm, failing_search, DecisionLog(tmp_path))
    assert str(caught.value) == "the search failed: Exa search failed: 503 Service Unavailable"
    assert isinstance(caught.value.__cause__, RuntimeError)
    if kind == "scripted":
        assert len(llm.prompts[ResearcherOutput]) == 1  # not retried by call_checked


# ---------- a whole round, built as tests/test_graph.py builds one ----------

POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")
PATTERN = {"found": True, "name": "advice-before-understanding",
           "description": "Gives advice in its first reply before asking about the situation.",
           "seen_in": 1, "out_of": len(SCENARIOS), "example_conversation_ids": ["conv-r01-fixture-0001"]}


@pytest.mark.parametrize("failure", ["provider", "search"])
def test_a_researcher_failure_finishes_the_round_as_no_evidence(tmp_path, traced, failure):
    if failure == "provider":
        llm = agent_llm(create=[bad_request()], parse=[parsed(CriticOutput, PATTERN)])
        search = FakeSearch({})
        expected = "BadRequestError: Error code: 400 - Function tools with reasoning_effort are not supported"
    else:
        llm = agent_llm(create=[tool_call()], parse=[parsed(CriticOutput, PATTERN)])
        search = failing_search
        expected = "the search failed: Exa search failed: 503 Service Unavailable"
    gate_calls = []
    deps = RoundDeps(helper_client=loop_llm.FakeClient(_fake_helper), user_client=loop_llm.FakeClient(_fake_user),
                     llm=llm, search=search, memory=FakeMemory(), log=DecisionLog(tmp_path), root=tmp_path,
                     gate=lambda r: gate_calls.append(r) or stand_in_gate(r, "reverted"),
                     usage=Meter(tmp_path, Prices(models={MODEL: ModelPrice(0.0, 0.0)})))
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)

    assert state["result"].outcome == "no_evidence" and state["event"].outcome == "no_evidence"
    assert state["result"].research is None and gate_calls == []
    events = [e for e in DecisionLog(tmp_path).events() if e.event == "round"]
    assert len(events) == 1 and events[0].outcome == "no_evidence"
    assert events[0].reason.startswith("researcher failed: ") and expected in events[0].reason
    assert SECRET not in events[0].reason
    if failure == "provider":
        assert [f["agent"] for f in lines(tmp_path, "failed")] == ["researcher"]
