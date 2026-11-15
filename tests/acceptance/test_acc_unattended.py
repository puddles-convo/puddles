"""Independent contract probes. All service replies are local scripted objects/transports.

These tests deliberately assert the contract, including the joins and crash recovery;
known failures remain failures, rather than xfails. No builder test helpers are used.
"""
import json
import re
import socket
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import mlflow
import openai
import pytest
from openai.types.chat import ChatCompletion, ParsedChatCompletion

import scripts.run_loop as driver
from loop.errors import LLMError
from loop.llm import FakeClient
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.agents import proposer, researcher
from puddles.config import LoopConfig
from puddles.decision_log import DecisionLog
from puddles.eval.judge import FallbackConfig, JudgeFailed, SingleJudge
from puddles.graph import RoundDeps, run_round
from puddles.llm import AgentFailed, OpenAILLM
from puddles.loop import run_loop
from puddles.memory import FakeMemory, PastClient, PastError, now, recall_records
from puddles.policy import PolicyEditError, apply_edit, parse_policy
from puddles.round import finish_round, prepare_round, retry_pending, run_agents, stand_in_gate
from puddles.usage import Meter, metering, read_usage
from schemas import (
    ConversationRecord, CriticOutput, JudgeOutput, ProposerInput,
    ProposerOutput, ResearcherOutput, ScoredConversation, Turn,
)

REPO = Path(__file__).resolve().parents[2]
START = REPO / "policies/policy-v1.md"
POOL = load_scenarios(REPO / "fixtures/scenarios.jsonl")
URL = "https://example.invalid/counselling"
QUERY = "listening before giving advice"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """A programming error in a probe must not reach a real service."""
    connect = socket.socket.connect

    def local_only(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"acceptance probe attempted external network: {address[0]}")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    monkeypatch.delenv("PUDDLES_AGENT_TEMPERATURE", raising=False)
    monkeypatch.delenv("PUDDLES_AGENT_TOOL_REASONING_EFFORT", raising=False)


@pytest.fixture
def tracked(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


class Replies:
    def __init__(self):
        self.answers = {}

    def structured(self, system, user, schema):
        answer = self.answers[schema]
        return schema.model_validate(answer(user) if callable(answer) else answer)

    def structured_with_search(self, system, user, schema, search, max_searches):
        answer = self.structured(system, user, schema)
        for query in answer.queries[:max_searches]:
            search(query)
        return answer


def deps_for(root, *, judge=None, memory=None, helper=None):
    llm = Replies()
    llm.answers[CriticOutput] = {"found": False, "reason": "No recurring problem."}
    return RoundDeps(helper_client=helper or FakeClient(_fake_helper),
                     user_client=FakeClient(_fake_user), llm=llm, search=lambda q: [],
                     memory=memory if memory is not None else FakeMemory(), log=DecisionLog(root),
                     root=root, start_policy=START, judge_panel=judge)


def drive(deps, cap=6, before=None):
    def one(policy_path, drawn, round_):
        if before:
            before(round_)
        return run_round(deps, policy_path=policy_path, scenarios=drawn, max_turns=3, seed=19)["event"]

    return run_loop(LoopConfig(max_rounds=cap, conversations_per_round=1), log=deps.log,
                    memory=deps.memory, root=deps.root, start_policy=START, scenarios=POOL[:1],
                    run_one=one, out=lambda text: None)


def lines(root, name="loop"):
    path = root / f"log/{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def config(tmp_path, monkeypatch, *, noise="0.25", measured="true", version=None, safety_price=True):
    val, train = tmp_path / "val.txt", tmp_path / "train.txt"
    val.write_text(POOL[0].scenario_id + "\n")
    train.write_text(POOL[1].scenario_id + "\n")
    path = tmp_path / "config.toml"
    text = '[prices]\nexa_search_usd = 0.01\n[output_limits]\ndefault = 300\n'
    if safety_price:
        text += '[prices.models."safety-offline"]\ninput_usd_per_million = 0.0\noutput_usd_per_million = 0.0\n'
    text += (f'[loop]\nnoise = {noise}\nnoise_measured = {measured}\nvalidation_ids = "{val}"\n'
             f'train_ids = "{train}"\nmax_turns = 4\nseed = 23\nconversations_per_round = 1\n')
    if version:
        text += f'judge_version = "{version}"\n'
    path.write_text(text)
    monkeypatch.setenv("PUDDLES_CONFIG", str(path))


def real_constructors(monkeypatch, judge):
    """Offline constructors; the driver's gate and round adapter are left intact."""
    import loop.llm
    import puddles.eval.judge
    import puddles.eval.safety
    import puddles.llm
    import puddles.memory

    monkeypatch.setenv("EXA_API_KEY", "offline")
    monkeypatch.setattr(loop.llm, "client_from_env", lambda prefix: FakeClient(
        _fake_helper if prefix == "HELPER" else _fake_user))
    monkeypatch.setattr(puddles.llm, "OpenAILLM", Replies)
    monkeypatch.setattr(puddles.memory, "PastClient", FakeMemory)
    monkeypatch.setattr(puddles.eval.judge, "judge_from_env", lambda: judge)
    monkeypatch.setattr(puddles.eval.safety.OpenAIYesNo, "from_env",
                        classmethod(lambda cls: SimpleNamespace(model="safety-offline")))
    monkeypatch.setattr(driver, "load_esconv", lambda split: POOL)


def conversation():
    return ConversationRecord(id="conv-independent", written_by="test", created_at=now(), parent_ids=[],
                              source="real", scenario_id="independent", split="validation",
                              policy_version="policy-v1", turns=[Turn(speaker="user", text="I feel alone."),
                              Turn(speaker="helper", text="That sounds painful.")])


def answer(label="Reflection of feelings", **extra):
    return JudgeOutput(empathy=3, relevance=4, emotion_before=4, emotion_after=3,
                       strategies=[label], note="The helper acknowledges the feeling.", **extra)


class JudgeReplies:
    model = "fake"

    def __init__(self, response):
        self.response, self.calls = response, 0

    def judge(self, system, user):
        self.calls += 1
        value = self.response(self.calls) if callable(self.response) else self.response
        if isinstance(value, Exception):
            raise value
        helper_count = len(re.findall(r"^\d+\. helper:", user, re.MULTILINE))
        if len(value.strategies) == 1:
            value = value.model_copy(update={"strategies": value.strategies * helper_count})
        return value, 10, 5


# 1.1, 1.2: independently exercise real-mode construction and the actual single-model adapter.
def test_real_gate_and_round_share_single_judge_and_validation_settings(tmp_path, monkeypatch, tracked):
    from puddles.gate import _Gate
    config(tmp_path, monkeypatch)
    judge = SingleJudge(FallbackConfig(model="fake"), JudgeReplies(answer()))
    real_constructors(monkeypatch, judge)
    captured = []
    actual = driver.run_round

    def run(deps, **kwargs):
        captured.append(deps)
        deps.llm.answers[CriticOutput] = {"found": False, "reason": "No pattern."}
        return actual(deps, **kwargs)

    monkeypatch.setattr(driver, "run_round", run)
    assert driver.main(["--root", str(tmp_path / "run"), "--gate", "real", "--max-rounds", "1",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]) == 0
    [deps] = captured
    assert deps.judge_panel is judge and isinstance(deps.gate, _Gate)
    gate = deps.gate
    assert gate.deps.judge.judge is judge
    assert gate.ids == [POOL[0].scenario_id] and gate.deps.scenarios == POOL[:1]
    assert (gate.noise, gate.deps.noise_measured, gate.deps.max_turns, gate.deps.seed) == (0.25, True, 4, 23)


def test_safety_price_refusal_precedes_round_calls(tmp_path, monkeypatch, capsys):
    config(tmp_path, monkeypatch, safety_price=False)
    real_constructors(monkeypatch, SingleJudge(FallbackConfig(model="fake"), JudgeReplies(answer())))
    monkeypatch.setattr(driver, "run_round", lambda *a, **kw: pytest.fail("a call escaped preflight"))
    assert driver.main(["--root", str(tmp_path / "run"), "--gate", "real", "--max-rounds", "1"]) == 2
    assert "safety-offline" in capsys.readouterr().err


def test_one_refused_single_judgment_is_unscored_while_another_is_used(tmp_path, tracked):
    llm = JudgeReplies(lambda n: AgentFailed("provider refusal") if n == 1 else answer())
    deps = deps_for(tmp_path, judge=SingleJudge(FallbackConfig(model="fake"), llm))
    state = run_round(deps, policy_path=START, scenarios=POOL[:2], max_turns=3)
    assert len(state["unscored"]) == len(state["scored"]) == 1
    saved = lines(tmp_path, "decisions")
    assert saved[0]["event"] == "round" and state["unscored"][0] in saved[0]["reason"]
    scores = (tmp_path / "scores/round-01-train.jsonl").read_text().splitlines()
    assert len(scores) == 1 and json.loads(scores[0])["conversation_id"] == state["scored"][0].conversation_id


# 1.3, 1.4: guard ordering with otherwise unavailable services.
def test_real_standin_keep_refused_and_revert_announces_pilot(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(driver, "load_loop_config", lambda: (_ for _ in ()).throw(AssertionError("preflight reached")))
    assert driver.main(["--gate", "kept", "--root", str(tmp_path)]) == 2
    capsys.readouterr()
    monkeypatch.setattr(driver, "load_loop_config", lambda: (_ for _ in ()).throw(ValueError("offline")))
    assert driver.main(["--gate", "reverted", "--root", str(tmp_path)]) == 2
    assert capsys.readouterr().out.splitlines()[0].startswith("pilot:")


@pytest.mark.parametrize("noise,measured", [("inf", "true"), ("-inf", "true"), ("nan", "true"),
                                          ("0", "true"), ("0.25", "false")])
def test_noise_refusal_never_constructs_clients(tmp_path, monkeypatch, noise, measured):
    import loop.llm
    config(tmp_path, monkeypatch, noise=noise, measured=measured)
    monkeypatch.setattr(loop.llm, "client_from_env", lambda *a: pytest.fail("client constructed"))
    assert driver.main(["--gate", "real", "--root", str(tmp_path), "--max-rounds", "1"]) == 2


# 1.5: a failing freeze check cannot be ignored; fallback still needs version agreement.
def test_unfrozen_panel_and_mismatched_fallback_refused_before_run(tmp_path, monkeypatch):
    import puddles.eval.calibrate
    config(tmp_path, monkeypatch)
    real_constructors(monkeypatch, SimpleNamespace(kind="panel", version="panel-offline"))
    monkeypatch.setattr(puddles.eval.calibrate, "check_frozen", lambda j: (_ for _ in ()).throw(ValueError("unfrozen")))
    assert driver.main(["--gate", "real", "--root", str(tmp_path / "panel"), "--max-rounds", "1"]) == 2
    config(tmp_path, monkeypatch, version="a-different-judge")
    real_constructors(monkeypatch, SingleJudge(FallbackConfig(model="fake"), JudgeReplies(answer())))
    assert driver.main(["--gate", "real", "--root", str(tmp_path / "fallback"), "--max-rounds", "1"]) == 2


# 1.6: recovery reaches the actual prepare_round and graph, with non-ASCII long errors.
def test_round_error_recovery_and_three_error_stop(tmp_path, tracked):
    deps = deps_for(tmp_path)
    calls = []

    def once(round_):
        calls.append(round_)
        if len(calls) == 1:
            raise httpx.ReadTimeout("時" * 450)

    summary = drive(deps, cap=2, before=once)
    assert calls == [1, 1, 2] and not summary.failed
    error = lines(tmp_path)[0]
    assert error["round"] == 1 and error["error"] == "ReadTimeout: " + "時" * 300
    bad = deps_for(tmp_path / "bad")
    summary = drive(bad, before=lambda n: (_ for _ in ()).throw(RuntimeError("service")))
    assert summary.failed and [x["event"] for x in lines(bad.root)] == ["round_error"] * 3 + ["stopped"]


# 1.7: this fails because finish_round adds unscored IDs to the reason used by nothing_scored.
@pytest.mark.parametrize("failure", ["timeout", "refusal", "429"])
def test_judge_outage_stops_after_three_actual_graph_rounds(tmp_path, tracked, failure):
    request = httpx.Request("POST", "https://offline.invalid/judge")
    errors = {"timeout": httpx.ReadTimeout("slow", request=request),
              "refusal": AgentFailed("model refused"),
              "429": openai.RateLimitError("busy", response=httpx.Response(429, request=request), body={})}
    judge = SingleJudge(FallbackConfig(model="fake"), JudgeReplies(errors[failure]))
    deps = deps_for(tmp_path, judge=judge)
    summary = drive(deps, cap=6)
    events = deps.log.rounds()
    assert len(events) == 3, (len(events), summary.stop_reason, events[0].reason)
    assert summary.failed and "no training conversation scored" in summary.stop_reason


def test_zero_score_streak_survives_process_restart(tmp_path, tracked):
    def unavailable(system, messages):
        raise LLMError("helper down")

    deps = deps_for(tmp_path, helper=FakeClient(unavailable))

    def killed(round_):
        if round_ == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        drive(deps, before=killed)
    assert len(deps.log.rounds()) == 2
    restarted = deps_for(tmp_path, helper=FakeClient(unavailable))
    summary = drive(restarted)
    assert len(restarted.log.rounds()) == 3, (len(restarted.log.rounds()), summary.stop_reason)
    assert summary.failed


def test_round_error_streak_survives_process_restart(tmp_path, tracked):
    deps = deps_for(tmp_path)
    calls = []

    def failing(round_):
        calls.append(round_)
        if len(calls) == 3:
            raise KeyboardInterrupt
        raise httpx.ReadTimeout("service down")

    with pytest.raises(KeyboardInterrupt):
        drive(deps, before=failing)
    assert len([x for x in lines(tmp_path) if x["event"] == "round_error"]) == 2
    restarted = deps_for(tmp_path)
    summary = drive(restarted, before=lambda n: (_ for _ in ()).throw(httpx.ReadTimeout("still down")))
    errors = [x for x in lines(tmp_path) if x["event"] == "round_error"]
    assert len(errors) == 3, (len(errors), summary.stop_reason)
    assert summary.failed


# 1.8: real driver exit status, stop record fields and spend cap, including a restarted cap.
def test_failure_exit_and_spend_stop_are_logged(tmp_path, monkeypatch, tracked, capsys):
    config(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, "run_round", lambda *a, **kw: (_ for _ in ()).throw(httpx.ReadTimeout("offline")))
    root = tmp_path / "error"
    assert driver.main(["--fake", "--root", str(root), "--max-rounds", "4",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]) == 3
    stopped = lines(root)[-1]
    assert stopped["event"] == "stopped" and stopped["reason"] == "3 round errors in a row"
    assert {"at", "spend_usd", "rounds", "policy_in_force"} <= stopped.keys()
    assert "stopped: 3 round errors in a row" in capsys.readouterr().out
    deps = deps_for(tmp_path / "spend")
    summary = run_loop(LoopConfig(spend_limit_usd=0.5, round_cost_estimate_usd=1.0), log=deps.log,
                       memory=deps.memory, root=deps.root, start_policy=START, scenarios=POOL,
                       run_one=lambda *a: pytest.fail("spend cap ignored"), out=lambda text: None)
    assert not summary.failed and summary.stop_reason.startswith("spend limit")
    assert lines(deps.root)[-1]["rounds"] == 0


def pattern_response(user):
    cid = re.search(r"conv-r\d+-[\w-]+", user).group(0)
    return {"found": True, "name": "advice-before-listening", "description": "Advice first.",
            "seen_in": 1, "out_of": 1, "example_conversation_ids": [cid]}


def record_fixture(root):
    """A valid round and record, before or after copies, without rewriting any files."""
    llm = Replies()
    llm.answers[CriticOutput] = {"found": True, "name": "advice-before-listening", "description": "Advice first.",
                                "seen_in": 1, "out_of": 1, "example_conversation_ids": ["conv-independent"]}
    llm.answers[ResearcherOutput] = {"queries": [QUERY], "sources": [{"url": URL, "technique": "Explore first."}]}
    llm.answers[ProposerOutput] = {"action": "reword", "rule_id": "C1", "rule_text": "Ask before suggesting.",
                                 "summary": "Explore first."}
    log, memory = DecisionLog(root), FakeMemory()
    memory.ensure_training_audience()
    conv = ScoredConversation(conversation_id="conv-independent", empathy=2, relevance=2, note="Advice first.",
                              turns=conversation().turns)
    result = run_agents(1, [conv], START, log, memory, llm,
                        lambda q: [{"url": URL, "title": "Skills", "text": "Ask first."}], root)
    return result, log, memory


# 2.1 and 2.2: use actual HTTP decoding at the round boundary, not just an exception-only fake.
def test_memory_html_and_timeouts_are_unavailable_and_round_continues(tmp_path, tracked):
    def reply(request):
        if request.url.path.endswith("/recall"):
            return httpx.Response(200, text="<html>busy</html>")
        raise httpx.ReadTimeout("audience down", request=request)

    memory = PastClient(api_key="offline", transport=httpx.MockTransport(reply))
    deps = deps_for(tmp_path, memory=memory)
    assert drive(deps, cap=1).rounds == [(1, "no_pattern")]
    reads = lines(tmp_path, "memory_reads")
    assert {x["reader"] for x in reads if x.get("unavailable")} >= {"critic", "round"}
    assert all(x["record_ids"] == [] and x["error"] for x in reads)
    empty = FakeMemory()
    empty.ensure_training_audience()
    assert recall_records(empty, deps.log, "q", "critic", tmp_path) == []
    assert "unavailable" not in lines(tmp_path, "memory_reads")[-1]


class DownMemory(FakeMemory):
    def __init__(self):
        super().__init__()
        self.calls = []

    def ingest(self, entry):
        self.calls.append(entry.id)
        raise PastError("503")


# 2.3: pre-existing pending copies are fine; a crash before their first copy exposes the join.
def test_old_pending_copy_retried_once_at_each_driver_start(tmp_path):
    result, log, memory = record_fixture(tmp_path)
    down = DownMemory()
    finish_round(result, stand_in_gate(result, "reverted"), log, down, tmp_path, START)
    down.calls.clear()
    expected = set(ref.id for ref in log.rounds()[0].records.values())
    for start in range(5):
        prepare_round(log, down, tmp_path, START)
        retry_pending(log, down, tmp_path)
        assert Counter(down.calls) == Counter({record_id: start + 1 for record_id in expected})
    assert all(e.status == "pending" and e.attempt == 6 for e in log.copy_events().values())


def test_crash_before_first_copy_does_not_retry_it_twice_at_one_start(tmp_path, monkeypatch):
    result, log, memory = record_fixture(tmp_path)
    import puddles.round

    # Kill at a real write boundary: the round event is durable, but no memory-copy event exists.
    with monkeypatch.context() as patch:
        patch.setattr(puddles.round, "copy_to_memory", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            finish_round(result, stand_in_gate(result, "reverted"), log, memory, tmp_path, START)
    assert log.missing_copies() and not log.copy_events()
    down = DownMemory()
    prepare_round(DecisionLog(tmp_path), down, tmp_path, START)
    retry_pending(DecisionLog(tmp_path), down, tmp_path)
    assert Counter(down.calls) == Counter({ref.id: 1 for ref in log.rounds()[0].records.values()}), down.calls


# 2.4: all API methods must reject a missing required field, not only ingest/recall.
@pytest.mark.parametrize("body", [{}, {"status": None}, {"status": ["completed"]}])
def test_malformed_ingestion_status_is_past_error(tmp_path, body):
    memory = PastClient(api_key="offline", transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)))
    with pytest.raises(PastError):
        memory.ingestion_status("ing-offline")


@pytest.mark.parametrize("body", [{"results": [{"sources": [{"metadata": []}]}]},
                                 {"results": [{"sources": 0}]}, {}, {"results": "empty"}])
def test_malformed_recall_shape_cannot_reach_agents(tmp_path, body):
    memory = PastClient(api_key="offline", transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)))
    assert recall_records(memory, DecisionLog(tmp_path), "q", "proposer", tmp_path) == []
    assert lines(tmp_path, "memory_reads")[-1]["unavailable"] is True


# 2.5: valid hash, stale hash, metadata absent and metadata belonging to another root.
def test_recall_requires_this_roots_metadata_and_loads_local_content(tmp_path):
    result, log, memory = record_fixture(tmp_path)
    finish_round(result, stand_in_gate(result, "reverted"), log, memory, tmp_path, START)
    ref = log.ref("pattern-r01")
    own = {"record_id": ref.id, "sha256": ref.sha256}

    class Recalled(FakeMemory):
        def __init__(self, metadata):
            super().__init__()
            self.metadata = metadata

        def recall(self, *a, **kw):
            return [{"content": "pattern-r01: INJECTED OLD CONTENT", "sources": [
                {"sourceId": "pattern-r01", "metadata": self.metadata}]}]

    for metadata in ({}, {**own, "sha256": "0" * 64}, {**own, "record_id": "pattern-r999"}):
        assert recall_records(Recalled(metadata), log, "q", "critic", tmp_path) == []
    [record] = recall_records(Recalled(own), log, "q", "critic", tmp_path)
    assert record.description == "Advice first." and "INJECTED" not in record.model_dump_json()


# 3.1 and 3.2: quotes inside a marker; unknown label still rejected; persisted flags survive adapter.
@pytest.mark.parametrize("label", ['  “helper TURN 7 — \'reflection OF feelings\'”  ', "7) REFLECTION OF FEELINGS"])
def test_strategy_normalization_handles_nested_quotes_and_case(label):
    llm = JudgeReplies(answer(label))
    result = SingleJudge(FallbackConfig(model="fake"), llm).score(conversation())
    assert result.score.strategies == ["Reflection of feelings"] and llm.calls == 1
    with pytest.raises(JudgeFailed):
        SingleJudge(FallbackConfig(model="fake"), JudgeReplies(answer("Turn 7: Invented strategy"))).score(conversation())


def test_safety_flags_survive_graph_score_serialization(tmp_path, tracked):
    judge = SingleJudge(FallbackConfig(model="fake"), JudgeReplies(answer(
        evidence="The helper ignored a risk at turn 2.", safety_flags=["missed_risk"])))
    deps = deps_for(tmp_path, judge=judge)
    drive(deps, cap=1)
    score = json.loads((tmp_path / "scores/round-01-train.jsonl").read_text())
    assert score["safety_flags"] == ["missed_risk"] and score["evidence"] == "The helper ignored a risk at turn 2."


# 3.3: with protection on, the agent fails closed and policy application independently protects C8.
@pytest.mark.parametrize("action", ["remove", "reword"])
def test_c8_repeated_edit_attempt_fails_without_a_candidate(tmp_path, monkeypatch, action):
    monkeypatch.setenv("PUDDLES_PROTECT_C8", "1")
    policy = parse_policy(START.read_text())
    inp = ProposerInput(policy_text=START.read_text(), at_length_limit=False,
                        pattern={"id": "pattern-r01", "name": "advice", "description": "Advice first."}, sources=[])
    llm = Replies()
    text = "Listen only." if action == "reword" else None
    llm.answers[ProposerOutput] = {"action": action, "rule_id": "C8", "rule_text": text, "summary": "Edit."}
    with metering(Meter(tmp_path, round=1), "proposer"), pytest.raises(AgentFailed, match="C8"):
        proposer.run(inp, policy.version, policy, DecisionLog(tmp_path), llm)
    with pytest.raises(PolicyEditError, match="C8"):
        apply_edit(policy, action, "C8", text, "policy-v2-candidate")
    assert not (tmp_path / "policies").exists()


def test_c8_is_editable_by_default_like_integration(tmp_path, monkeypatch):
    monkeypatch.delenv("PUDDLES_PROTECT_C8", raising=False)
    monkeypatch.delenv("PUDDLES_PARK_REVERTED", raising=False)
    text = START.read_text()
    policy = parse_policy(text)
    inp = ProposerInput(policy_text=text, at_length_limit=False,
                        pattern={"id": "pattern-r01", "name": "advice"}, sources=[])
    log = DecisionLog(tmp_path)
    for action, wording in (("reword", "Keep listening."), ("remove", None)):
        out = ProposerOutput(action=action, rule_id="C8", rule_text=wording, summary="Edit.")
        assert proposer.check(out, inp, policy, log) == []
        candidate, touched = apply_edit(policy, action, "C8", wording, "policy-v2-candidate")
        assert touched == "C8"
        assert candidate.version == "policy-v2-candidate"
        assert candidate.safety == policy.safety
        assert {rid: rule for rid, rule in candidate.conversation.items() if rid != "C8"} == {
            rid: rule for rid, rule in policy.conversation.items() if rid != "C8"}
        if action == "reword":
            assert candidate.conversation["C8"] == wording
        else:
            assert "C8" not in candidate.conversation
    for setting in (None, "1"):
        if setting is None:
            monkeypatch.delenv("PUDDLES_PROTECT_C8", raising=False)
        else:
            monkeypatch.setenv("PUDDLES_PROTECT_C8", setting)
        for action, wording in (("reword", "Keep listening."), ("remove", None)):
            out = ProposerOutput(action=action, rule_id="S1", rule_text=wording, summary="Edit.")
            assert any("safety rule" in problem for problem in proposer.check(out, inp, policy, log))
            with pytest.raises(PolicyEditError, match="protected safety rule"):
                apply_edit(policy, action, "S1", wording, "policy-v2-candidate")


def completion(message, finish="stop", parsed=False):
    data = {"id": "offline", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    return (ParsedChatCompletion[ResearcherOutput] if parsed else ChatCompletion).model_validate(data)


def tool_response(calls, finish="tool_calls"):
    return completion({"content": None, "tool_calls": [{"id": f"call-{n}", "type": "function", "function": {
        "name": name, "arguments": args}} for n, (name, args) in enumerate(calls)]}, finish)


def tool_llm(monkeypatch, responses, output):
    import openai
    sent = []
    responses = iter(responses)

    def create(**kwargs):
        sent.append(kwargs)
        return next(responses)

    def parse(**kwargs):
        sent.append(kwargs)
        return completion({"content": json.dumps(output), "parsed": output}, parsed=True)

    monkeypatch.setattr(openai, "OpenAI", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create, parse=parse))))
    return OpenAILLM(model="fake", output_limits={"default": 200}), sent


# 3.4: every bad call consumes a slot, and length wins over even a well-formed search call.
@pytest.mark.parametrize("name,args", [("search", "null"), ("search", '{"query": false}'),
                                       ("search", '{"query": "\\t"}'), ("search", "{}"),
                                       ("other", '{"query": "q"}'), ("search", "{")])
def test_bad_search_call_consumes_slot_and_returns_error(tmp_path, monkeypatch, name, args):
    llm, sent = tool_llm(monkeypatch, [tool_response([(name, args), ("search", '{"query": "next"}')])],
                         {"queries": ["next"], "sources": []})
    llm.structured_with_search("sys", "user", ResearcherOutput, lambda q: pytest.fail("over budget search"), 1)
    results = [json.loads(m["content"]) for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert len(results) == 2 and all("error" in result for result in results)


def test_cutoff_search_turn_fails_and_logs_researcher(tmp_path, monkeypatch):
    llm, sent = tool_llm(monkeypatch, [tool_response([("search", '{"query": "q"}')], "length")], {})
    with metering(Meter(tmp_path, round=1), "researcher"), pytest.raises(AgentFailed, match="cut off"):
        llm.structured_with_search("sys", "user", ResearcherOutput, lambda q: pytest.fail("cutoff searched"), 3)
    failed = [x for x in read_usage(tmp_path / "log/usage.jsonl") if x["event"] == "failed"]
    assert len(failed) == 1 and failed[0]["agent"] == "researcher" and len(sent) == 1


# 3.5 plus section-1 join: real Exa retry wrapper, mocked HTTP, full graph, same-agent streak.
def test_exa_503_stops_loop_after_three_researcher_failures(tmp_path, monkeypatch, tracked):
    from puddles import exa
    monkeypatch.setenv("EXA_API_KEY", "offline")
    calls = []

    def unavailable(url, **kwargs):
        calls.append(url)
        return httpx.Response(503, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", unavailable)
    deps = deps_for(tmp_path)
    # Search is called through exa.search, so run_round also checks its price.
    from puddles.usage import Prices
    deps.usage = Meter(tmp_path, Prices(exa_search_usd=0.01))
    deps.search = exa.search

    deps.llm.answers[CriticOutput] = pattern_response
    deps.llm.answers[ResearcherOutput] = {"queries": [QUERY], "sources": []}
    summary = drive(deps)
    failed = [x for x in read_usage(tmp_path / "log/usage.jsonl") if x["event"] == "failed"]
    assert summary.failed and summary.stop_reason == "researcher failed in 3 rounds in a row"
    assert [(x["round"], x["agent"]) for x in failed] == [(1, "researcher"), (2, "researcher"), (3, "researcher")]
    assert len(calls) == 6 and all(e.outcome == "no_evidence" for e in deps.log.rounds())


def test_zero_search_results_end_no_evidence_without_failed_usage(tmp_path, tracked):
    deps = deps_for(tmp_path)
    deps.llm.answers[CriticOutput] = pattern_response
    deps.llm.answers[ResearcherOutput] = {"queries": [QUERY], "sources": []}
    summary = drive(deps, cap=1)
    assert summary.rounds == [(1, "no_evidence")] and not summary.failed
    assert not [x for x in read_usage(tmp_path / "log/usage.jsonl") if x["event"] == "failed"]
