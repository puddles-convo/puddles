"""Stage 1: token counts, cost, the usage log, schema output on a plain endpoint, the transcript.

The numbered tests are the eleven in docs/plan/stage-1-client-and-io.md. All offline: the only
servers are loopback ones started here.
"""
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import httpx
import mlflow
import pytest
from pydantic import BaseModel

from loop.errors import ConfigError, DeadlineExceeded, LLMError, SchemaError
from loop.llm import CallUsage, FakeClient, OpenAICompatClient, client_from_env, estimate_tokens
from loop.policy import load_policy
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios, run_batch
from puddles import exa, tracing
from puddles.agents.render import transcript
from puddles.config import DEFAULT_PATH, load_config
from puddles.decision_log import DecisionLog
from puddles.graph import RoundDeps, run_round
from puddles.llm import AgentFailed, OpenAILLM, call_checked
from puddles.memory import FakeMemory
from puddles.tracing import TracedClient
from puddles.usage import FAKE_MODEL, Meter, MissingPrice, ModelPrice, Prices, metering, read_usage, spend
from schemas import CriticOutput, ProposerOutput, ResearcherOutput, ScoredConversation
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")
URL = "https://counselling.example.ac.uk/skills"
QUERY = "counselling skills exploring before giving advice"
MESSAGES = [{"role": "user", "content": "I can't see the point any more."}]
PRICES = Prices(models={"m": ModelPrice(1.0, 2.0)}, exa_search_usd=0.005)
TOKEN_USAGE = "mlflow.chat.tokenUsage"

PATTERN = {"found": True, "name": "advice-before-understanding",
           "description": "Gives advice in its first reply before asking about the situation.",
           "seen_in": 1, "out_of": len(SCENARIOS), "example_conversation_ids": ["conv-r01-fixture-0001"]}
BAD_PATTERN = {"found": True, "seen_in": 1, "out_of": 1, "example_conversation_ids": []}
RESEARCH = {"queries": [QUERY], "sources": [{"url": URL, "title": "Skills", "technique": "Explore first."}]}
EDIT = {"action": "add", "rule_text": "Ask one open question before any suggestion.", "summary": "Ask first."}


class Verdict(BaseModel):
    risk: Literal["none", "concern", "crisis"]
    reason: str


GOOD = '{"risk": "crisis", "reason": "Says there is no point."}'
BAD = '{"risk": "maybe"}'


def reply(content, tokens=None, *, status=200, delay=0.0):
    """One scripted answer of the fake chat server."""
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if tokens is not None:
        body["usage"] = {"prompt_tokens": tokens[0], "completion_tokens": tokens[1],
                         "total_tokens": tokens[0] + tokens[1]}
    return {"status": status, "body": body, "delay": delay}


@contextmanager
def chat_server(answers):
    """A loopback server. `answers` is a list used in order, or a function of the request body."""
    calls = []
    queue = None if callable(answers) else list(answers)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            answer = answers(body) if queue is None else queue.pop(0)
            if answer["delay"]:
                time.sleep(answer["delay"])
            raw = json.dumps(answer["body"]).encode()
            try:
                self.send_response(answer["status"])
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client gave up waiting, as the deadline test intends

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.delenv("PUDDLES_CONFIG", raising=False)


@pytest.fixture
def traced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow keeps some trace files in ./mlruns
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


def llm_spans():
    mlflow.flush_trace_async_logging()
    return [s for t in mlflow.search_traces(return_type="list") for s in t.data.spans
            if s.name.startswith("llm:")]


def lines(root, event=None, **match):
    found = read_usage(Path(root) / "log" / "usage.jsonl")
    return [line for line in found if (event is None or line["event"] == event)
            and all(line.get(k) == v for k, v in match.items())]


def scripted_llm(*answers):
    llm = ScriptedLLM()
    for schema, out in answers or ((CriticOutput, PATTERN), (ResearcherOutput, RESEARCH), (ProposerOutput, EDIT)):
        llm.add(schema, out)
    return llm


def make_deps(root, llm, **overrides):
    values = dict(helper_client=FakeClient(_fake_helper), user_client=FakeClient(_fake_user), llm=llm,
                  search=FakeSearch({QUERY: [{"url": URL, "title": "Skills", "text": "..."}]}),
                  memory=FakeMemory(), log=DecisionLog(root), root=root)
    values.update(overrides)
    return RoundDeps(**values)


# 1
def test_usage_is_read_from_the_response_and_estimated_when_absent():
    with chat_server([reply("Hello there.", (123, 45)), reply("Hello there.")]) as (url, _):
        client = OpenAICompatClient(url, "m")
        assert client.last_usage is None
        assert client.complete("sys", MESSAGES, max_tokens=50) == "Hello there."
        assert client.last_usage == CallUsage(123, 45, estimated=False)

        assert client.complete("sys", MESSAGES, max_tokens=50) == "Hello there."
        prompt_chars = len("sys") + len(MESSAGES[0]["content"])
        assert client.last_usage == CallUsage((prompt_chars + 3) // 4, 3, estimated=True)
        assert client.last_usage.tokens_in > 0 and client.last_usage.tokens_out > 0


@pytest.mark.parametrize("usage", [None, "many", {"prompt_tokens": "7"}, {"prompt_tokens": 7},
                                   {"prompt_tokens": -1, "completion_tokens": 2}])
def test_an_unusable_usage_field_is_estimated_and_does_not_fail(usage):
    answer = reply("Hello there.")
    answer["body"]["usage"] = usage
    with chat_server([answer]) as (url, _):
        client = OpenAICompatClient(url, "m")
        assert client.complete("sys", MESSAGES, max_tokens=50) == "Hello there."
        assert client.last_usage.estimated and client.last_usage.tokens_out == estimate_tokens("Hello there.")


def test_a_failed_request_leaves_no_usage_from_the_one_before():
    with chat_server([reply("ok", (5, 1)), reply("", status=500)]) as (url, _):
        client = OpenAICompatClient(url, "m")
        client.complete("sys", MESSAGES, max_tokens=50)
        with pytest.raises(LLMError) as caught:
            client.complete("sys", MESSAGES, max_tokens=50)
        assert caught.value.status == 500 and client.last_usage is None


# 2
def test_complete_json_validates_retries_once_and_raises_after_two_bad_replies(tmp_path, traced):
    meter = Meter(tmp_path, PRICES, round=4)
    answers = [reply(GOOD, (11, 9)),
               reply(BAD, (10, 5)), reply(GOOD, (20, 7)),
               reply(BAD, (10, 5)), reply("not json at all", (30, 6))]
    with chat_server(answers) as (url, calls):
        client = TracedClient(OpenAICompatClient(url, "m"), "crisis_checker", meter=meter)

        assert client.complete_json("sys", MESSAGES, Verdict, max_tokens=60) == Verdict(
            risk="crisis", reason="Says there is no point.")
        assert calls[0]["response_format"]["json_schema"]["schema"] == Verdict.model_json_schema()
        assert calls[0]["max_tokens"] == 60 and calls[0]["temperature"] == 0.0

        meter.start_round(5)
        assert client.complete_json("sys", MESSAGES, Verdict, max_tokens=60).risk == "crisis"
        tally = meter.usage["crisis_checker"]
        assert (tally.tokens_in, tally.tokens_out, tally.calls) == (30, 12, 2)  # both attempts
        assert tally.first_answer_failed == 1 and tally.failed == 0
        retry = calls[2]["messages"]
        assert retry[-2] == {"role": "assistant", "content": BAD}
        assert retry[-1]["role"] == "user" and "did not match the schema" in retry[-1]["content"]

        meter.start_round(6)
        with pytest.raises(SchemaError):
            client.complete_json("sys", MESSAGES, Verdict, max_tokens=60)
        assert len(calls) == 5  # one retry, no more
        tally = meter.usage["crisis_checker"]
        assert (tally.tokens_in, tally.tokens_out, tally.calls) == (40, 11, 2)
        assert tally.first_answer_failed == 1 and tally.failed == 1
        assert [(f["round"], f["agent"]) for f in lines(tmp_path, "failed")] == [(6, "crisis_checker")]
    assert len(lines(tmp_path, "sent")) == len(lines(tmp_path, "done")) == 5


def test_complete_json_without_a_wrapper_and_on_the_fake_client():
    with chat_server([reply(BAD), reply("```json\n" + GOOD + "\n```")]) as (url, calls):
        assert OpenAICompatClient(url, "m").complete_json("sys", MESSAGES, Verdict, max_tokens=60).risk == "crisis"
        assert len(calls) == 2
    fake = FakeClient([BAD, GOOD, BAD, BAD])
    assert fake.complete_json("sys", MESSAGES, Verdict, max_tokens=60).risk == "crisis"
    with pytest.raises(SchemaError):
        fake.complete_json("sys", MESSAGES, Verdict, max_tokens=60)
    assert len(fake.calls) == 4 and "response_format" in fake.calls[0]


# 3
def test_complete_json_falls_back_when_the_server_refuses_response_format():
    refusal = reply("", status=400)
    with chat_server([refusal, reply("Here it is: " + GOOD, (40, 9)), reply(GOOD, (40, 9))]) as (url, calls):
        client = OpenAICompatClient(url, "m")
        assert client.supports_response_format
        assert client.complete_json("sys", MESSAGES, Verdict, max_tokens=60).risk == "crisis"
        assert "response_format" in calls[0] and "response_format" not in calls[1]
        system = calls[1]["messages"][0]["content"]
        assert system.startswith("sys") and json.dumps(Verdict.model_json_schema()) in system
        assert calls[1]["messages"][1:] == MESSAGES

        # The refusal is remembered: the next call does not send the refused request again.
        assert not client.supports_response_format
        assert client.complete_json("sys", MESSAGES, Verdict, max_tokens=60).risk == "crisis"
        assert len(calls) == 3 and "response_format" not in calls[2]


def test_a_refusal_followed_by_a_bad_reply_is_still_retried_once():
    with chat_server([reply("", status=400), reply(BAD), reply(GOOD)]) as (url, calls):
        assert OpenAICompatClient(url, "m").complete_json("sys", MESSAGES, Verdict, max_tokens=60).risk == "crisis"
        assert len(calls) == 3


def test_an_error_that_is_not_a_refusal_is_not_a_fallback():
    with chat_server([reply("", status=500)]) as (url, calls):
        client = OpenAICompatClient(url, "m")
        with pytest.raises(LLMError):
            client.complete_json("sys", MESSAGES, Verdict, max_tokens=60)
        assert len(calls) == 1 and client.supports_response_format


# 4
def test_complete_json_raises_when_the_deadline_passes_during_the_retry():
    with chat_server([reply(BAD), reply(GOOD, delay=1.5)]) as (url, calls):
        client = OpenAICompatClient(url, "m")
        started = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            client.complete_json("sys", MESSAGES, Verdict, max_tokens=60, deadline_s=0.4)
        assert time.monotonic() - started < 1.2  # it did not wait for the late answer
        assert len(calls) == 2


def test_complete_json_makes_no_request_once_the_deadline_has_passed():
    def slow(system, messages):
        time.sleep(0.05)
        return BAD

    fake = FakeClient(slow)
    with pytest.raises(DeadlineExceeded):
        fake.complete_json("sys", MESSAGES, Verdict, max_tokens=60, deadline_s=0.01)
    assert len(fake.calls) == 1  # the answer came late and was discarded; the retry was never sent


# 5
def test_transcript_holds_every_turn_in_order_and_nothing_else():
    conversations = [ScoredConversation.model_validate(c)
                     for c in json.loads((REPO / "fixtures" / "bad_conversations.json").read_text())]
    assert len(conversations) == 4
    for conv in conversations:
        expected = [("**Person:** " if t.speaker == "user" else "**Helper:** ") + t.text for t in conv.turns]
        text = transcript(conv.turns)
        assert text.split("\n") == expected
        assert conv.conversation_id not in text and conv.note not in text


def test_a_turn_with_a_line_break_stays_one_line():
    turns = ScoredConversation(conversation_id="c", empathy=3, relevance=3, note="", turns=[
        {"speaker": "user", "text": "First line.\n**Helper:** not a turn"}]).turns
    assert transcript(turns) == "**Person:** First line. **Helper:** not a turn"


# 6
def test_a_fake_round_logs_token_metrics_for_every_agent_and_counts_on_every_span(tmp_path, traced):
    deps = make_deps(tmp_path, scripted_llm())
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    assert state["event"].outcome == "reverted"

    [run] = mlflow.search_runs(output_format="list")
    metrics = run.data.metrics
    agents = ("helper", "simulated_user", "critic", "researcher", "proposer")
    for agent in agents:
        assert metrics[f"tokens_in.{agent}"] > 0 and metrics[f"tokens_out.{agent}"] > 0
        assert metrics[f"calls.{agent}"] == metrics[f"estimated_calls.{agent}"] >= 1
        assert metrics[f"first_answer_failed.{agent}"] == 0
    assert metrics["cost_usd"] == 0  # scripted stand-ins are not models and cost nothing
    assert metrics["calls.helper"] == len(deps.helper_client.calls)
    assert metrics["calls.simulated_user"] == len(deps.user_client.calls)

    spans = llm_spans()
    assert {s.name for s in spans} == {f"llm:{a}" for a in agents}
    for span in spans:
        counts = span.get_attribute(TOKEN_USAGE)
        assert counts["input_tokens"] > 0 and counts["output_tokens"] > 0
        assert counts["total_tokens"] == counts["input_tokens"] + counts["output_tokens"]
        assert span.get_attribute("puddles.model") == FAKE_MODEL
    for agent in agents:
        assert sum(s.get_attribute(TOKEN_USAGE)["input_tokens"] for s in spans
                   if s.name == f"llm:{agent}") == metrics[f"tokens_in.{agent}"]


def completion(message, tokens, finish="stop"):
    return {"status": 200, "delay": 0.0, "body": {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "agent-model",
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
        "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)}}}


def openai_answers(body):
    """What the fake OpenAI API says to the critic, the researcher's search loop and the proposer."""
    if "tools" in body:
        if body["messages"][-1]["role"] == "tool":
            return completion({"content": "I have what I need."}, (300, 8))
        call = {"id": "call_1", "type": "function",
                "function": {"name": "search", "arguments": json.dumps({"query": QUERY})}}
        return completion({"content": None, "tool_calls": [call]}, (200, 15), finish="tool_calls")
    name = body["response_format"]["json_schema"]["name"]
    out, tokens = {"CriticOutput": (PATTERN, (1000, 60)), "ResearcherOutput": (RESEARCH, (400, 50)),
                   "ProposerOutput": (EDIT, (700, 40))}[name]
    return completion({"content": json.dumps(out)}, tokens)


@pytest.fixture
def openai_llm(monkeypatch):
    """A real OpenAILLM, talking to a loopback server through the OpenAI library."""
    with chat_server(openai_answers) as (url, calls):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", url)
        monkeypatch.delenv("PUDDLES_AGENT_TEMPERATURE", raising=False)
        llm = OpenAILLM(model="agent-model", output_limits={"default": 900, "researcher": 1200})
        llm.calls = calls
        yield llm


def test_each_request_of_the_real_client_has_its_own_span_and_counts(tmp_path, traced, openai_llm):
    prices = Prices(models={"agent-model": ModelPrice(2.0, 8.0)})
    deps = make_deps(tmp_path, openai_llm, usage=Meter(tmp_path, prices))
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    assert state["event"].outcome == "reverted" and len(openai_llm.calls) == 5

    def counts(agent):
        return sorted((s.get_attribute(TOKEN_USAGE)["input_tokens"], s.get_attribute(TOKEN_USAGE)["output_tokens"])
                      for s in llm_spans() if s.name == f"llm:{agent}")

    assert counts("critic") == [(1000, 60)]
    assert counts("researcher") == [(200, 15), (300, 8), (400, 50)]  # two calls in the search loop, then the answer
    assert counts("proposer") == [(700, 40)]

    [run] = mlflow.search_runs(output_format="list")
    metrics = run.data.metrics
    assert (metrics["tokens_in.researcher"], metrics["tokens_out.researcher"]) == (900, 73)
    assert metrics["calls.researcher"] == 3 and metrics["estimated_calls.researcher"] == 0
    assert metrics["cost_usd"] == pytest.approx((2600 * 2.0 + 173 * 8.0) / 1e6)
    done = lines(tmp_path, "done", model="agent-model")
    assert sum(d["cost_usd"] for d in done) == pytest.approx(metrics["cost_usd"])
    assert all(not d["estimated"] and d["round"] == 1 for d in done)


# 7
def test_complete_json_through_traced_client_leaves_one_span_per_attempt(tmp_path, traced):
    with chat_server([reply(BAD, (10, 5)), reply(GOOD, (20, 7))]) as (url, _):
        client = TracedClient(OpenAICompatClient(url, "m"), "crisis_checker", meter=Meter(tmp_path, PRICES))
        assert client.complete_json("sys", MESSAGES, Verdict, max_tokens=60).risk == "crisis"

    spans = sorted(llm_spans(), key=lambda s: s.start_time_ns)
    assert [s.name for s in spans] == ["llm:crisis_checker"] * 2
    assert [s.span_type for s in spans] == ["CHAT_MODEL"] * 2
    assert [s.get_attribute(TOKEN_USAGE) for s in spans] == [
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}]
    assert [s.outputs for s in spans] == [BAD, GOOD]
    assert [s.get_attribute("puddles.tokens_estimated") for s in spans] == [False, False]
    assert spans[1].get_attribute("puddles.cost_usd") == pytest.approx((20 * 1.0 + 7 * 2.0) / 1e6)
    assert "response_format" in spans[0].inputs


# 8
def test_a_model_with_no_price_stops_the_run_before_the_first_call(tmp_path, traced):
    helper, user, memory = FakeClient(_fake_helper), FakeClient(_fake_user), FakeMemory()
    helper.model = "helper-model"
    user.model = "m"
    llm = scripted_llm()
    deps = make_deps(tmp_path, llm, helper_client=helper, user_client=user, memory=memory,
                     usage=Meter(tmp_path, PRICES))

    with pytest.raises(MissingPrice, match="helper-model"):
        run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    assert helper.calls == [] and user.calls == [] and not llm.prompts
    assert memory.entries == {} and memory.audiences == {}
    assert not (tmp_path / "log").exists() and not (tmp_path / "conversations").exists()
    assert mlflow.search_runs(output_format="list") == []
    assert issubclass(MissingPrice, ConfigError)


def test_exa_with_no_price_stops_the_run_before_the_first_call(tmp_path, traced):
    deps = make_deps(tmp_path, scripted_llm(), search=exa.search, usage=Meter(tmp_path, Prices()))
    with pytest.raises(MissingPrice, match="Exa"):
        run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    assert deps.helper_client.calls == [] and not (tmp_path / "log").exists()


def test_the_shipped_config_has_no_prices_and_a_limit_for_every_call():
    config = load_config()
    assert config.prices.models == {} and config.prices.exa_search_usd is None
    assert config.output_limits["default"] > 0
    with pytest.raises(MissingPrice):
        config.prices.cost("any-model", 1, 1)
    with pytest.raises(MissingPrice):
        Meter("unused", config.prices).sent("helper", "any-model", 10, 10)


def test_the_config_file_is_read_and_checked(tmp_path, monkeypatch):
    path = tmp_path / "run.toml"
    path.write_text('[prices]\nexa_search_usd = 0.005\n[prices.models]\n'
                    '"m" = { input_usd_per_million = 1.5, output_usd_per_million = 6 }\n'
                    '"local" = { input_usd_per_million = 0, output_usd_per_million = 0 }\n'
                    "[output_limits]\ndefault = 500\ncritic = 800\n")
    monkeypatch.setenv("PUDDLES_CONFIG", str(path))
    config = load_config()
    assert config.prices.cost("m", 1_000_000, 500_000) == pytest.approx(4.5)
    assert config.prices.cost("local", 10, 10) == 0 and config.prices.search() == 0.005
    assert config.output_limits == {"default": 500, "critic": 800}
    config.prices.check(["m", "local", FAKE_MODEL], search=True)

    for bad in ('[prices.models]\n"m" = { input_usd_per_million = 1 }\n[output_limits]\ndefault = 1\n',
                '[prices.models]\n"m" = { input_usd_per_million = -1, output_usd_per_million = 1 }\n'
                "[output_limits]\ndefault = 1\n",
                "[output_limits]\ncritic = 800\n", "[output_limits]\ndefault = 0\n", "not toml ["):
        path.write_text(bad)
        with pytest.raises(ConfigError):
            load_config()
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.toml")
    assert DEFAULT_PATH == REPO / "config" / "run.toml"


# 9
def test_the_done_lines_of_a_round_add_up_to_the_spans(tmp_path, traced):
    helper, user = FakeClient(_fake_helper), FakeClient(_fake_user)
    helper.model = user.model = "m"
    deps = make_deps(tmp_path, scripted_llm(), helper_client=helper, user_client=user,
                     usage=Meter(tmp_path, PRICES))
    run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)

    spans, done, sent = llm_spans(), lines(tmp_path, "done", round=1), lines(tmp_path, "sent", round=1)
    assert len(done) == len(spans) == len(sent) and done == lines(tmp_path, "done")
    assert {d["call_id"] for d in done} == {s["call_id"] for s in sent} == {
        s.get_attribute("puddles.call_id") for s in spans}
    for agent in ("helper", "simulated_user", "critic", "researcher", "proposer"):
        mine = [s.get_attribute(TOKEN_USAGE) for s in spans if s.name == f"llm:{agent}"]
        for key, field in (("input_tokens", "tokens_in"), ("output_tokens", "tokens_out")):
            assert sum(c[key] for c in mine) == sum(d[field] for d in done if d["agent"] == agent) > 0
    cost = sum(d["cost_usd"] for d in done)
    assert cost > 0 and cost == pytest.approx(sum(s.get_attribute("puddles.cost_usd") for s in spans))
    assert spend(deps.usage.path) == pytest.approx(cost)
    [run] = mlflow.search_runs(output_format="list")
    assert run.data.metrics["cost_usd"] == pytest.approx(cost)
    by_id = {s["call_id"]: s for s in sent}
    assert all(by_id[d["call_id"]]["max_cost_usd"] > 0 for d in done if d["model"] == "m")


# 10
def test_a_batch_killed_midway_keeps_its_finished_calls_and_counts_the_open_one(tmp_path, traced):
    def helper(system, messages):
        if len(helper_client.calls) == 3:
            raise KeyboardInterrupt  # the process is stopped while the third request is out
        return _fake_helper(system, messages)

    helper_client, user_client = FakeClient(helper), FakeClient(_fake_user)
    helper_client.model = user_client.model = "m"
    meter = Meter(tmp_path, PRICES, round=2)
    with pytest.raises(KeyboardInterrupt), metering(meter):
        run_batch(TracedClient(helper_client, "helper"), TracedClient(user_client, "simulated_user"),
                  load_policy(POLICY), SCENARIOS, round=2)

    sent, done = lines(tmp_path, "sent"), lines(tmp_path, "done")
    finished = len(user_client.calls) + len(helper_client.calls) - 1
    assert len(done) == finished >= 4 and len(sent) == finished + 1
    [cut_off] = [s for s in sent if s["call_id"] not in {d["call_id"] for d in done}]
    assert cut_off["agent"] == "helper" and cut_off["max_cost_usd"] > 0
    assert spend(meter.path) == pytest.approx(sum(d["cost_usd"] for d in done) + cut_off["max_cost_usd"])

    # A last line torn by the crash is skipped, not fatal.
    with meter.path.open("a") as f:
        f.write('{"event": "done", "call_id": "')
    assert spend(meter.path) == pytest.approx(sum(d["cost_usd"] for d in done) + cut_off["max_cost_usd"])


# 11
def test_a_request_through_openai_llm_carries_the_configured_output_limit(tmp_path, openai_llm):
    out = openai_llm.structured("sys", "user", CriticOutput)
    assert out.name == PATTERN["name"]
    assert openai_llm.calls[0]["max_completion_tokens"] == 900  # no agent named: the default

    searched = []
    with metering(None, "researcher"):
        openai_llm.structured_with_search("sys", "user", ResearcherOutput,
                                          lambda q: searched.append(q) or [{"url": URL}], 3)
    assert searched == [QUERY]
    assert [c["max_completion_tokens"] for c in openai_llm.calls[1:]] == [1200, 1200, 1200]
    assert not (tmp_path / "log").exists()  # no meter, so no usage log and no spans


def test_openai_llm_without_limits_takes_them_from_the_config_file(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert OpenAILLM(model="agent-model").output_limits == load_config().output_limits


def test_an_answer_cut_off_at_the_limit_fails_the_agent_and_is_still_counted(tmp_path, traced, monkeypatch):
    cut = completion({"content": '{"found": tr'}, (1000, 900), finish="length")
    with chat_server([cut]) as (url, _):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", url)
        llm = OpenAILLM(model="agent-model", output_limits={"default": 900})
        meter = Meter(tmp_path, Prices(models={"agent-model": ModelPrice(2.0, 8.0)}), round=3)
        with metering(meter, "critic"), pytest.raises(AgentFailed, match="output limit"):
            llm.structured("sys", "user", CriticOutput)
    [done] = lines(tmp_path, "done")
    assert (done["tokens_in"], done["tokens_out"]) == (1000, 900)
    assert [(f["round"], f["agent"]) for f in lines(tmp_path, "failed")] == [(3, "critic")]


def test_an_exa_search_leaves_a_sent_and_a_done_line_per_request(tmp_path, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    posts = []

    def post(url, **kwargs):
        posts.append(kwargs)
        request = httpx.Request("POST", url)
        if len(posts) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"results": [{"url": URL, "title": "Skills", "text": "..."}]},
                              request=request)

    monkeypatch.setattr(exa.httpx, "post", post)
    meter = Meter(tmp_path, PRICES, round=7)
    with metering(meter, "researcher"):
        assert exa.search(QUERY) == [{"url": URL, "title": "Skills", "text": "..."}]

    found = lines(tmp_path)
    assert [line["event"] for line in found] == ["sent", "done", "sent", "done"]  # the retry is its own request
    assert [line["call_id"] for line in found[:2]] == [found[0]["call_id"]] * 2
    assert found[0]["call_id"] != found[2]["call_id"]
    for line in found:
        assert line["kind"] == "search" and line["agent"] == "researcher" and line["round"] == 7
        assert line.get("cost_usd", line.get("max_cost_usd")) == 0.005
        assert not {"model", "tokens_in", "tokens_out", "max_tokens", "prompt_tokens_estimate"} & set(line)
    assert "error" in found[1] and "error" not in found[3]
    assert meter.usage["researcher"].searches == 2 and meter.metrics()["searches.researcher"] == 2
    assert spend(meter.path) == pytest.approx(0.010)

    with metering(Meter(tmp_path / "unpriced", Prices()), "researcher"), pytest.raises(MissingPrice):
        exa.search(QUERY)
    assert len(posts) == 2  # refused before the request went out


# Counting retries
def test_a_first_answer_that_fails_its_checks_is_counted_for_the_round(tmp_path, traced):
    llm = scripted_llm((CriticOutput, BAD_PATTERN), (CriticOutput, PATTERN), (ResearcherOutput, RESEARCH),
                       (ProposerOutput, EDIT))
    run_round(make_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)
    [run] = mlflow.search_runs(output_format="list")
    assert run.data.metrics["first_answer_failed.critic"] == 1
    assert run.data.metrics["first_answer_failed.researcher"] == 0
    assert run.data.metrics["calls.critic"] == 2  # the failed attempt is counted too
    assert lines(tmp_path, "failed") == []


def test_an_agent_that_fails_twice_leaves_a_failed_line_naming_the_round_and_agent(tmp_path, traced):
    llm = scripted_llm((CriticOutput, BAD_PATTERN), (CriticOutput, BAD_PATTERN))
    state = run_round(make_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)
    assert state["event"].outcome == "no_pattern"
    [failed] = lines(tmp_path, "failed")
    assert (failed["round"], failed["agent"]) == (1, "critic") and "failed its checks twice" in failed["reason"]
    [run] = mlflow.search_runs(output_format="list")
    assert run.data.metrics["first_answer_failed.critic"] == 1


def test_call_checked_outside_a_metered_block_changes_nothing():
    answers = iter(["bad", "good"])
    assert call_checked(lambda feedback: next(answers), lambda out: [] if out == "good" else ["bad"], "x") == "good"


def test_client_from_env_reads_an_extra_body(monkeypatch):
    monkeypatch.setenv("HELPER_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("HELPER_MODEL", "m")
    monkeypatch.delenv("HELPER_API_KEY", raising=False)
    monkeypatch.setenv("HELPER_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
    assert client_from_env("HELPER").extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setenv("HELPER_EXTRA_BODY", "[1]")
    with pytest.raises(ConfigError, match="HELPER_EXTRA_BODY"):
        client_from_env("HELPER")
    monkeypatch.delenv("HELPER_EXTRA_BODY")
    assert client_from_env("HELPER").extra_body == {}
