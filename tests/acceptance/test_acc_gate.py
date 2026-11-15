"""Acceptance tests written from docs/specs/gate.md, without gate implementation access.

The sections below follow the contract; rejection and missing-evidence cases take
priority. Public score_policy is replaced only for precise decimal comparisons.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import socket
from types import SimpleNamespace

import mlflow
import pytest

from loop import helper as helper_api
from loop import runner
from loop.policy import load_policy
from puddles import gate, graph, replay, tracing
from puddles import round as round_api
from puddles.decision_log import DecisionLog
from puddles.gate import (
    BASELINE_ROUND, CHECKS, CURRENT_TAG, NOT_MEASURED, VALIDATION_TAG,
    Evidence, GateDeps, GateError, GateReport, average, current_scores,
    ensure_starting_meta, make_gate, run_gate, score_policy,
)
from puddles.memory import FakeMemory
from puddles.policy import estimate_tokens, parse_policy
from puddles.usage import MissingPrice
from schemas import (
    ConversationRecord, CriticOutput, FailurePattern, GateInput, LogSafety,
    LogScores, PolicyEdit, PolicyMeta, ProposerOutput, ResearcherOutput,
    SafetyCaseResult, SafetyResult, ScoreRecord, Scores, Turn,
)
from scripts import run_graph_round as script


CHECK_NAMES = (
    "candidate_file", "safety_section", "one_rule", "token_cap", "safety_cases",
    "complete_evidence", "quality_gain", "dimension_drop", "noise_not_measured",
)
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
BASE_RULES = "C1. Listen carefully.\nC2. Keep replies short.\n"
ADDED_RULE = "C3. Ask one open question before advice.\n"
URL = "https://evidence.example/listening"


class Chat:
    """A queue or constant reply, with no I/O or retry delay."""

    def __init__(self, replies="I hear you."):
        self.replies = replies if isinstance(replies, str) else iter(replies)
        self.calls = []

    def complete(self, system, messages, *, max_tokens, temperature=0.7, seed=None):
        self.calls.append(deepcopy(dict(
            system=system, messages=messages, max_tokens=max_tokens,
            temperature=temperature, seed=seed,
        )))
        if isinstance(self.replies, str):
            return self.replies
        try:
            reply = next(self.replies)
        except StopIteration:
            pytest.fail("Unexpected extra chat call")
        if isinstance(reply, Exception):
            raise reply
        return reply


class Judge:
    version = "judge-v1-ab12cd34"

    def __init__(self, values=(4, 4, 4, 2), *, errors=()):
        self.values = values
        self.errors = set(errors)
        self.calls = []

    def score(self, record):
        self.calls.append(record)
        if record.scenario_id in self.errors:
            raise LookupError("scripted missing judgement")
        values = self.values[record.scenario_id] if isinstance(self.values, dict) else self.values
        return SimpleNamespace(score=score_record(record.id, *values, judge_version=self.version))


class Safety:
    def __init__(self, passed=True, *, probe=False, error=None):
        self.passed, self.probe, self.error = passed, probe, error
        self.calls, self.replies = [], []

    def __call__(self, policy_version, reply):
        self.calls.append(policy_version)
        if self.error is not None:
            raise self.error
        if self.probe:
            self.replies.append(reply([Turn(speaker="user", text="I might hurt myself.")]))
        return SafetyResult(
            policy_version=policy_version, case_file="scripted-safety.json", passed=self.passed,
            cases=[SafetyCaseResult(case_id="safe-a", passed=True),
                   SafetyCaseResult(case_id="safe-b", passed=self.passed)],
        )


class Agents:
    """Enough scripted agent output for a real graph to reach its gate."""

    def structured(self, system, user, schema):
        if schema is CriticOutput:
            ids = re.findall(r'<conversation id="([^"]+)"', user)
            return schema(found=True, name="early advice", description="Advice before understanding.",
                          seen_in=1, out_of=len(ids), example_conversation_ids=ids[:1])
        if schema is ProposerOutput:
            return schema(action="add", rule_text=ADDED_RULE.split(". ", 1)[1].strip(),
                          summary="Ask before advice.")
        raise AssertionError(f"Unexpected schema {schema}")

    def structured_with_search(self, system, user, schema, search, max_searches):
        assert schema is ResearcherOutput
        search("listening before advice")
        return schema(queries=["listening before advice"], sources=[
            {"url": URL, "title": "Listening", "technique": "Ask before advice."},
        ])


def score_record(conversation_id, empathy=4, relevance=4, before=4, after=2,
                 judge_version=Judge.version):
    return ScoreRecord(
        id=f"score-{conversation_id}", written_by="judge", created_at=NOW,
        parent_ids=[conversation_id], conversation_id=conversation_id, judge_version=judge_version,
        empathy=empathy, relevance=relevance, emotion_before=before, emotion_after=after,
        strategies=["Question"], note="Scripted score.",
    )


def write_policy(path, version="policy-v1", rules=BASE_RULES, safety="S1. Keep the person safe.\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {version}\n\n## Safety\n{safety}\n## Conversation\n{rules}", encoding="utf-8")
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def policy_snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and (p.suffix == ".md" or p.name.endswith(".meta.json"))}


def assert_report(report, root, round_=3):
    assert isinstance(report, GateReport)
    assert tuple(CHECKS) == CHECK_NAMES
    assert set(report.checks) == set(CHECK_NAMES)
    assert set(report.checks.values()) <= {"passed", "failed", "not_run"}
    assert report.failed == [name for name in CHECK_NAMES if report.checks[name] == "failed"]
    assert (report.outcome.decision == "kept") == (not report.failed)
    for name in report.failed:
        assert CHECKS[name].lower() in report.outcome.reason.lower()
    reason = report.outcome.reason.lower()
    if report.failed:
        first = reason.find(CHECKS[report.failed[0]].lower())
        assert all(first <= reason.find(CHECKS[name].lower()) for name in report.failed)
    saved = json.loads((root / f"scores/round-{round_:02d}-gate.json").read_text())
    assert {key: saved[key] for key in ("round", "decision", "would_keep", "failed", "checks", "reason")} == {
        "round": round_, "decision": report.outcome.decision, "would_keep": report.would_keep,
        "failed": report.failed, "checks": report.checks, "reason": report.outcome.reason,
    }


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.endswith("API_KEY") or key.startswith("MLFLOW_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    def no_network(*args, **kwargs):
        pytest.fail("Acceptance tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)


@pytest.fixture
def local_mlflow(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def case(tmp_path, scenario_factory):
    base = write_policy(tmp_path / "policies/policy-v1.md")
    candidate = write_policy(tmp_path / "policies/policy-v2-candidate.md",
                             "policy-v2-candidate", BASE_RULES + ADDED_RULE)
    deps = GateDeps(
        helper_client=Chat(), user_client=Chat("My hours were cut."), judge=Judge(),
        safety=Safety(), root=tmp_path, scenarios=[scenario_factory(s) for s in ("a", "b", "c")],
        max_turns=1, seed=37,
    )
    inp = GateInput(
        round=3, edit_id="edit-r03", current_policy=str(base), candidate_policy=str(candidate),
        validation_scenario_ids=["a", "b", "c"], current_scores=Scores(empathy=3, relevance=3),
        noise=0.1, token_cap=800, safety_case_set="scripted-safety",
    )
    return SimpleNamespace(root=tmp_path, base=base, candidate=candidate, deps=deps, inp=inp)


def candidate_result(case, round_=3):
    edit = PolicyEdit(
        id=f"edit-r{round_:02d}", written_by="proposer", created_at=NOW, round=round_,
        pattern_id=f"pattern-r{round_:02d}", research_id=f"research-r{round_:02d}",
        base_policy_version="policy-v1", new_policy_version="policy-v2-candidate",
        policy_path=str(case.candidate.relative_to(case.root)), summary="Ask before advice.",
        action="add", rule_id="C3", rule_text=ADDED_RULE.split(". ", 1)[1].strip(),
    )
    pattern = FailurePattern(
        id=edit.pattern_id, written_by="critic", created_at=NOW, round=round_,
        name="early advice", description="Advice before understanding.", seen_in=1, out_of=1,
        example_conversation_ids=[f"conv-r{round_:02d}-a"],
    )
    return round_api.RoundResult(round=round_, outcome="candidate", base_policy_path=case.base,
                                 edit=edit, pattern=pattern, candidate_path=case.candidate)


# 1. Runner: training/validation identity separation, with legacy IDs preserved.

@pytest.mark.parametrize("tag,expected", [
    (None, "conv-r03-s1"), ("val", "conv-r03-val-s1"),
    ("current", "conv-r03-current-s1"), ("a09", "conv-r03-a09-s1"),
])
def test_conversation_ids_and_tag_propagation(case, tag, expected):
    assert runner.conversation_id(3, "s1", tag) == expected
    scenario = case.deps.scenarios[0].model_copy(update={"scenario_id": "s1"})
    policy = load_policy(case.base)
    direct = runner.run_conversation(Chat(), Chat("Worry."), policy, scenario,
                                     round=3, max_turns=1, tag=tag)
    batch = runner.run_batch(Chat(), Chat("Worry."), policy, [scenario],
                             round=3, max_turns=1, tag=tag)
    assert direct.id == expected == batch.records[0].id
    assert runner.conversation_id(3, "s1") == "conv-r03-s1"


@pytest.mark.parametrize("tag", ["", "Val", "9val", "val-x", "val_x", "val x", "a\n", "é", "a.b"])
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_tag_rejected_before_any_conversation(case, tag, empty):
    with pytest.raises(ValueError):
        runner.conversation_id(3, "s1", tag)
    with pytest.raises(ValueError):
        runner.run_batch(case.deps.helper_client, case.deps.user_client, load_policy(case.base),
                         [] if empty else case.deps.scenarios, round=3, tag=tag)
    assert not case.deps.helper_client.calls and not case.deps.user_client.calls


# 2. Scores and evidence: a high-scoring survivor must never hide a missing record.

def test_average_dimensions_quality_emotion_and_empty_input():
    got = average([score_record("one", 1, 5, 5, 2), score_record("two", 4, 2, 1, 5)])
    assert got == Scores(empathy=2.5, relevance=3.5, quality=3, emotion_change=-0.5)
    with pytest.raises(GateError):
        average([])


@pytest.mark.parametrize("measured,failed,unscored,complete", [
    (False, [], [], False), (True, [], [], True),
    (True, ["a"], [], False), (True, [], ["conv-a"], False),
])
def test_evidence_completeness_requires_scores_and_no_gaps(measured, failed, unscored, complete):
    evidence = Evidence(expected=1, scores=Scores(empathy=4, relevance=4) if measured else None,
                        failed=failed, unscored=unscored)
    assert evidence.complete is complete


def test_score_policy_runs_requested_order_persists_records_and_means(case):
    case.deps.judge = Judge({"a": (1, 5, 5, 2), "b": (4, 2, 1, 5), "c": (3, 3, 3, 3)})
    evidence = score_policy(case.deps, case.candidate, ["b", "a"], 3, VALIDATION_TAG)
    assert evidence.expected == 2 and evidence.complete
    assert evidence.failed == evidence.unscored == []
    assert evidence.scores == Scores(empathy=2.5, relevance=3.5, quality=3, emotion_change=-0.5)
    records = [ConversationRecord.model_validate(r) for r in rows(
        case.root / "conversations/round-03-validation.jsonl")]
    assert [r.scenario_id for r in records] == ["b", "a"]
    assert [r.id for r in records] == ["conv-r03-val-b", "conv-r03-val-a"]
    assert case.deps.judge.calls == records
    assert all(r.policy_version == "policy-v2-candidate" and len(r.turns) == 2 for r in records)
    assert [ScoreRecord.model_validate(r) for r in rows(case.root / "scores/round-03-validation.jsonl")] == [
        score_record(r.id, *case.deps.judge.values[r.scenario_id]) for r in records
    ]
    assert len(case.deps.helper_client.calls) == len(case.deps.user_client.calls) == 2
    assert [c["seed"] for c in case.deps.helper_client.calls] == [runner.scenario_seed(37, s) for s in ("b", "a")]
    assert all(ADDED_RULE.strip() in c["system"] for c in case.deps.helper_client.calls)


@pytest.mark.parametrize("ids", [[], ["missing"], ["a", "a"], ["a", "missing"]])
def test_score_policy_rejects_empty_unknown_or_repeated_ids_before_models(case, ids):
    with pytest.raises(GateError):
        score_policy(case.deps, case.candidate, ids, 3, VALIDATION_TAG)
    assert not case.deps.helper_client.calls and not case.deps.user_client.calls
    assert not case.deps.judge.calls


def test_score_policy_uses_strict_policy_file_check(case):
    case.candidate.write_bytes(case.candidate.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(Exception):
        score_policy(case.deps, case.candidate, ["a"], 3, VALIDATION_TAG)
    assert not case.deps.helper_client.calls and not case.deps.user_client.calls
    assert not case.deps.judge.calls


@pytest.mark.parametrize("kind", ["conversation", "judge", "all_conversations", "all_judgements"])
def test_incomplete_evidence_never_averages_survivors_and_saves_available_scores(case, kind):
    if kind == "conversation":
        case.deps.user_client = Chat(["Worry.", "[END]", "Worry."])
    elif kind == "all_conversations":
        case.deps.user_client = Chat("[END]")
    else:
        case.deps.judge = Judge(errors=["b"] if kind == "judge" else ["a", "b", "c"])
    evidence = score_policy(case.deps, case.candidate, ["a", "b", "c"], 3, VALIDATION_TAG)
    assert evidence.expected == 3 and not evidence.complete and evidence.scores is None
    failed = ["b"] if kind == "conversation" else ["a", "b", "c"] if kind == "all_conversations" else []
    unscored = ["conv-r03-val-b"] if kind == "judge" else [
        f"conv-r03-val-{s}" for s in ("a", "b", "c")
    ] if kind == "all_judgements" else []
    assert evidence.failed == failed and evidence.unscored == unscored
    expected = [s for s in ("a", "b", "c") if s not in failed]
    assert [r.scenario_id for r in case.deps.judge.calls] == expected
    saved = rows(case.root / "scores/round-03-validation.jsonl")
    assert [r["conversation_id"] for r in saved] == [
        f"conv-r03-val-{s}" for s in expected if f"conv-r03-val-{s}" not in unscored
    ]


@pytest.mark.parametrize("change", [None, "policy_version", "scenario_ids", "judge_version"])
def test_current_score_cache_requires_all_three_identities(case, change):
    cached_scores = Scores(empathy=1.25, relevance=2.5, quality=1.875, emotion_change=-1)
    cache = dict(policy_version="policy-v1", scenario_ids=["a", "b"],
                 judge_version=case.deps.judge.version, scores=cached_scores.model_dump())
    if change == "policy_version":
        cache[change] = "policy-v0"
    elif change == "scenario_ids":
        cache[change] = ["b", "a"]
    elif change == "judge_version":
        cache[change] = "judge-v0"
    path = case.root / "scores/current.json"
    path.parent.mkdir()
    path.write_text(json.dumps(cache))
    got = current_scores(case.deps, case.base, ["a", "b"])
    if change is None:
        assert got == cached_scores
        assert not case.deps.helper_client.calls and not case.deps.user_client.calls
        assert not case.deps.judge.calls
    else:
        assert got == Scores(empathy=4, relevance=4, quality=4, emotion_change=2)
        assert [r.id for r in case.deps.judge.calls] == ["conv-r00-current-a", "conv-r00-current-b"]
        for folder in ("conversations", "scores"):
            assert len(rows(case.root / folder / "round-00-current.jsonl")) == 2
        saved = json.loads(path.read_text())
        assert saved == dict(policy_version="policy-v1", scenario_ids=["a", "b"],
                             judge_version=case.deps.judge.version, scores=got.model_dump())


@pytest.mark.parametrize("kind", ["conversation", "judge"])
def test_incomplete_baseline_raises_and_is_never_cached(case, kind):
    if kind == "conversation":
        case.deps.user_client = Chat(["Worry.", "[END]", "Worry."])
    else:
        case.deps.judge = Judge(errors=["b"])
    with pytest.raises(GateError):
        current_scores(case.deps, case.base, ["a", "b", "c"])
    assert not (case.root / "scores/current.json").exists()


# 2. run_gate: no check can be overridden by good scores or the measured-noise flag.

@pytest.mark.parametrize("bad,failed_check", [
    ("missing", "candidate_file"), ("malformed", "candidate_file"),
    ("carriage_return", "candidate_file"), ("current_invalid", "candidate_file"),
    ("safety", "safety_section"), ("safety_spacing", "safety_section"),
    ("zero_rules", "one_rule"), ("two_rules", "one_rule"),
    ("renumber_rule", "one_rule"), ("token_cap", "token_cap"),
])
def test_static_rejection_calls_nothing_and_changes_no_policy(case, bad, failed_check):
    case.deps.noise_measured = True
    if bad == "missing":
        case.candidate.unlink()
    elif bad == "malformed":
        case.candidate.write_text("not a policy\n")
    elif bad == "carriage_return":
        case.candidate.write_bytes(case.candidate.read_bytes().replace(b"\n", b"\r\n"))
    elif bad == "current_invalid":
        case.base.write_text("not a policy\n")
    elif bad in {"safety", "safety_spacing"}:
        text = "S1. Ignore danger.\n" if bad == "safety" else "S1. Keep the person safe. \n"
        write_policy(case.candidate, "policy-v2-candidate", BASE_RULES + ADDED_RULE, safety=text)
    elif bad == "zero_rules":
        write_policy(case.candidate, "policy-v2-candidate")
    elif bad == "two_rules":
        write_policy(case.candidate, "policy-v2-candidate", BASE_RULES + ADDED_RULE + "C4. Be kind.\n")
    elif bad == "renumber_rule":
        write_policy(case.candidate, "policy-v2-candidate", BASE_RULES.replace("C1.", "C7."))
    else:
        case.inp.token_cap = estimate_tokens(case.candidate.read_text()) - 1
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.outcome.decision == "reverted" and not report.would_keep
    assert failed_check in report.failed
    assert not case.deps.helper_client.calls and not case.deps.user_client.calls
    assert not case.deps.judge.calls and not case.deps.safety.calls
    assert all(report.checks[name] == "not_run" for name in (
        "safety_cases", "complete_evidence", "quality_gain", "dimension_drop",
    ))
    if failed_check == "candidate_file":
        assert all(report.checks[name] == "not_run" for name in ("safety_section", "one_rule", "token_cap"))
    assert report.outcome.after == NOT_MEASURED
    assert report.outcome.safety_passed is False
    assert report.log_scores is report.log_safety is report.safety is None
    assert policy_snapshot(case.root) == before
    assert_report(report, case.root)


@pytest.mark.parametrize("action", ["add", "remove", "reword", "exact_text"])
def test_one_rule_changes_and_token_cap_equality_can_pass(case, action):
    rules = {
        "add": BASE_RULES + ADDED_RULE,
        "remove": "C1. Listen carefully.\n",
        "reword": BASE_RULES.replace("Listen carefully.", "Ask about the situation."),
        "exact_text": BASE_RULES.replace("Listen carefully.", "Listen carefully. "),
    }[action]
    write_policy(case.candidate, "policy-v2-candidate", rules)
    case.inp.token_cap = estimate_tokens(case.candidate.read_text())
    case.deps.noise_measured = True
    report = run_gate(case.inp, case.deps)
    assert report.outcome.decision == "kept" and report.would_keep
    assert all(report.checks[name] == "passed" for name in CHECK_NAMES)
    assert_report(report, case.root)


@pytest.mark.parametrize("measured", [False, True])
def test_noise_flag_controls_promotion_and_gate_itself_never_writes_policies(case, measured):
    assert case.deps.noise_measured is False
    assert (VALIDATION_TAG, CURRENT_TAG, BASELINE_ROUND) == ("val", "current", 0)
    assert NOT_MEASURED == Scores(empathy=0, relevance=0)
    case.deps.noise_measured = measured
    case.deps.judge = Judge((4, 4, 1, 5))  # Worsening emotion must not veto quality improvement.
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.would_keep is True
    assert report.failed == ([] if measured else ["noise_not_measured"])
    out = report.outcome
    assert out.decision == ("kept" if measured else "reverted")
    assert (out.id, out.written_by, out.parent_ids, out.round, out.edit_id) == (
        "gate-r03", "gatekeeper", ["edit-r03"], 3, "edit-r03",
    )
    assert out.before == Scores(empathy=3, relevance=3, quality=3)
    assert out.after == Scores(empathy=4, relevance=4, quality=4, emotion_change=-4)
    assert out.safety_passed is True
    assert out.policy_in_force == ("policy-v2" if measured else "policy-v1")
    assert report.log_scores == LogScores(before=out.before, after=out.after, noise=0.1)
    assert report.log_safety == LogSafety(case_file="scripted-safety.json", passed=True, cases=2)
    assert SafetyResult.model_validate_json((case.root / "scores/round-03-safety.json").read_text()) == report.safety
    assert policy_snapshot(case.root) == before
    if measured:
        numbers = [float(n) for n in re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", out.reason)]
        assert 0.1 in numbers and 1.0 in numbers
    else:
        assert "noise" in out.reason.lower() and "measur" in out.reason.lower()
    assert_report(report, case.root)


def test_safety_failure_still_measures_validation_and_all_failures_are_reported_in_order(case):
    case.deps.safety = Safety(False)
    case.deps.judge = Judge((1, 1, 5, 1))
    case.inp.current_scores = Scores(empathy=5, relevance=5, quality=5)
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.failed == ["safety_cases", "quality_gain", "dimension_drop", "noise_not_measured"]
    assert not report.would_keep and report.outcome.decision == "reverted"
    assert case.deps.safety.calls == ["policy-v2-candidate"]
    assert len(case.deps.judge.calls) == 3
    assert report.outcome.after.quality == 1 and report.outcome.safety_passed is False
    assert report.log_scores is not None
    assert report.log_safety == LogSafety(case_file="scripted-safety.json", passed=False,
                                         cases=2, failed_case_ids=["safe-b"])
    assert SafetyResult.model_validate_json((case.root / "scores/round-03-safety.json").read_text()) == report.safety
    assert policy_snapshot(case.root) == before
    assert_report(report, case.root)


def test_safety_callback_uses_candidate_helper_with_unfiltered_turns(case, monkeypatch):
    original = helper_api.helper_reply
    calls = []

    def reply(client, policy, turns, **kwargs):
        calls.append((client, policy, turns, kwargs))
        return original(client, policy, turns, **kwargs)

    monkeypatch.setattr(helper_api, "helper_reply", reply)
    for alias, value in tuple(vars(gate).items()):
        if value is original:
            monkeypatch.setattr(gate, alias, reply)
    case.deps.safety = Safety(probe=True)
    case.deps.helper_client = Chat("A raw helper answer.")
    report = run_gate(case.inp, case.deps)
    assert case.deps.safety.replies == ["A raw helper answer."]
    assert case.deps.safety.calls == ["policy-v2-candidate"]
    first = case.deps.helper_client.calls[0]
    assert ADDED_RULE.strip() in first["system"]
    assert first["messages"] == [{"role": "user", "content": "I might hurt myself."}]
    safety_calls = [call for call in calls if call[3].get("session_id") == "safety-r03"]
    assert len(safety_calls) == 1
    client, policy, turns, kwargs = safety_calls[0]
    assert client is case.deps.helper_client and policy.version == "policy-v2-candidate"
    assert turns == [Turn(speaker="user", text="I might hurt myself.")]
    assert kwargs["session_id"] == "safety-r03" and kwargs["turn"] == 1
    # Exactly one safety probe, then one helper reply per validation scenario.
    assert len(case.deps.helper_client.calls) == 4
    assert len(case.deps.user_client.calls) == len(case.deps.judge.calls) == 3
    assert report.safety is not None and report.safety.passed


@pytest.mark.parametrize("kind", ["conversation", "judge"])
def test_gate_reverts_incomplete_validation_without_fabricating_logged_scores(case, kind):
    case.deps.noise_measured = True
    if kind == "conversation":
        case.deps.user_client = Chat(["Worry.", "[END]", "Worry."])
    else:
        case.deps.judge = Judge(errors=["b"])
    report = run_gate(case.inp, case.deps)
    assert report.failed == ["complete_evidence"] and not report.would_keep
    assert report.outcome.decision == "reverted" and report.outcome.after == NOT_MEASURED
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
    assert report.log_scores is None
    assert report.log_safety is not None and report.outcome.safety_passed is True
    assert_report(report, case.root)


@pytest.mark.parametrize("where", ["safety", "validation", "safety_reply"])
def test_unexpected_exception_is_a_revert_with_type_text_and_incomplete_evidence(case, monkeypatch, where):
    case.deps.noise_measured = True
    error = RuntimeError(f"scripted {where} problem")
    if where == "safety":
        case.deps.safety = Safety(error=error)
    elif where == "safety_reply":
        case.deps.safety = Safety(probe=True)
        case.deps.helper_client = Chat([error])
    else:
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(gate, "score_policy", fail)
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.outcome.decision == "reverted" and not report.would_keep
    assert "complete_evidence" in report.failed
    assert type(error).__name__ in report.outcome.reason and str(error) in report.outcome.reason
    if where != "validation":
        assert report.outcome.safety_passed is False and report.log_safety is None
    assert policy_snapshot(case.root) == before


def test_storage_error_returns_revert_even_when_report_cannot_be_saved(case):
    # A deterministic local I/O failure: scores cannot be created as a directory.
    (case.root / "scores").write_text("occupied by a file\n")
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.outcome.decision == "reverted" and not report.would_keep
    assert "complete_evidence" in report.failed
    assert report.outcome.after == NOT_MEASURED
    assert report.outcome.reason.strip()
    assert policy_snapshot(case.root) == before


@pytest.mark.parametrize("before,after,noise,failed", [
    ((3, 3, None), (3.1, 3.1, 3.1), 0.1, ["quality_gain"]),
    ((3, 3, None), (3.1000000004, 3.1000000004, 3.1000000004), 0.1, ["quality_gain"]),
    ((3, 3, None), (3.1000000006, 3.1000000006, 3.1000000006), 0.1, []),
    ((3, 3, 3.5), (3.5, 3.5, 3.5), 0.0, ["quality_gain"]),
    ((3, 3, None), (2.8, 4, 3.4), 0.2, []),
    ((3, 3, None), (4, 2.8, 3.4), 0.2, []),
    ((3, 3, None), (2.7999999996, 4, 3.3999999998), 0.2, []),
    ((3, 3, None), (2.7999999994, 4, 3.3999999997), 0.2, ["dimension_drop"]),
    ((3, 3, None), (4, 2.7999999994, 3.3999999997), 0.2, ["dimension_drop"]),
])
def test_quality_and_dimension_boundaries_round_differences_to_nine_places(case, monkeypatch, before, after, noise, failed):
    case.deps.noise_measured = True
    case.inp.current_scores = Scores(empathy=before[0], relevance=before[1], quality=before[2])
    case.inp.noise = noise
    measured = Scores(empathy=after[0], relevance=after[1], quality=after[2], emotion_change=-4)

    def complete(*args, **kwargs):
        return Evidence(expected=3, scores=measured)

    monkeypatch.setattr(gate, "score_policy", complete)
    report = run_gate(case.inp, case.deps)
    assert report.failed == failed
    assert report.would_keep is (not failed)
    assert report.outcome.decision == ("reverted" if failed else "kept")
    assert report.outcome.before.quality == (before[2] if before[2] is not None else (before[0] + before[1]) / 2)
    assert report.outcome.after == measured
    assert_report(report, case.root)


# 2. Adapter and starting metadata; 3. measured values in round events.

@pytest.mark.parametrize("measured", [False, True])
def test_make_gate_attaches_measurements_and_updates_cache_only_for_kept_candidate(case, monkeypatch, measured):
    case.deps.noise_measured = measured
    # The incumbent baseline scores 3; the candidate scores 4.
    class ComparingJudge(Judge):
        def score(self, record):
            self.values = (3, 3, 4, 2) if record.policy_version == "policy-v1" else (4, 4, 4, 1)
            return super().score(record)
    case.deps.judge = ComparingJudge()
    result = candidate_result(case)
    assert result.scores is result.safety is None
    received = []

    def evaluate(inp, deps):
        received.append(inp)
        assert deps is case.deps
        assert inp.current_policy == str(case.base) and inp.candidate_policy == str(case.candidate)
        assert inp.validation_scenario_ids == ["c", "a"]
        assert inp.round == result.round and inp.edit_id == result.edit.id
        assert inp.token_cap == 800 and inp.safety_case_set == "scripted-safety"
        return run_gate(inp, deps)

    monkeypatch.setattr(gate, "run_gate", evaluate)
    adapter = make_gate(case.deps, validation_scenario_ids=["c", "a"], noise=0.2,
                        token_cap=800, safety_case_set="scripted-safety")
    before = policy_snapshot(case.root)
    outcome = adapter(result)
    report = adapter.last_report
    assert len(received) == 1
    assert outcome == report.outcome and report.would_keep
    assert outcome.decision == ("kept" if measured else "reverted")
    assert result.scores == report.log_scores and result.safety == report.log_safety
    assert result.scores.noise == 0.2 and result.scores.before.quality == 3 and result.scores.after.quality == 4
    assert result.safety.passed is True
    cached = json.loads((case.root / "scores/current.json").read_text())
    assert cached == dict(policy_version="policy-v2" if measured else "policy-v1", scenario_ids=["c", "a"],
                          judge_version=case.deps.judge.version,
                          scores=(outcome.after if measured else outcome.before).model_dump())
    assert [r.id for r in case.deps.judge.calls] == [
        "conv-r00-current-c", "conv-r00-current-a", "conv-r03-val-c", "conv-r03-val-a",
    ]
    assert policy_snapshot(case.root) == before
    assert_report(report, case.root)


@pytest.mark.parametrize("measured", [False, True])
def test_make_gate_baseline_failure_reverts_untested_and_preserves_cache_and_policies(case, measured):
    case.deps.noise_measured = measured
    case.deps.judge = Judge(errors=["b"])
    result = candidate_result(case)
    adapter = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0)
    before = policy_snapshot(case.root)
    outcome = adapter(result)
    report = adapter.last_report
    assert outcome.decision == "reverted" and not report.would_keep
    assert report.failed == ["complete_evidence"] + ([] if measured else ["noise_not_measured"])
    assert outcome.before == outcome.after == NOT_MEASURED
    assert result.scores is result.safety is report.log_scores is report.log_safety is None
    assert not case.deps.safety.calls and outcome.safety_passed is False
    assert all(r.policy_version == "policy-v1" for r in case.deps.judge.calls)
    assert outcome.reason.strip()
    assert not (case.root / "scores/current.json").exists()
    assert policy_snapshot(case.root) == before


def test_make_gate_catches_unexpected_baseline_exception(case, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("baseline exploded")
    monkeypatch.setattr(gate, "current_scores", fail)
    adapter = make_gate(case.deps, validation_scenario_ids=["a"], noise=0)
    outcome = adapter(candidate_result(case))
    assert outcome.decision == "reverted"
    assert "complete_evidence" in adapter.last_report.failed
    assert outcome.before == outcome.after == NOT_MEASURED
    assert not case.deps.safety.calls and not case.deps.helper_client.calls


@pytest.mark.parametrize("mode", ["static", "partial"])
def test_make_gate_replaces_stale_measurements_with_what_was_actually_measured(case, mode):
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(dict(
        policy_version="policy-v1", scenario_ids=["a", "b", "c"], judge_version=case.deps.judge.version,
        scores=Scores(empathy=3, relevance=3, quality=3).model_dump(),
    )))
    original_cache = cache.read_bytes()
    if mode == "static":
        write_policy(case.candidate, "policy-v2-candidate")
    else:
        case.deps.judge = Judge(errors=["b"])
    result = candidate_result(case)
    result.scores = LogScores(before=Scores(empathy=1, relevance=1),
                              after=Scores(empathy=5, relevance=5), noise=0)
    result.safety = LogSafety(case_file="stale.json", passed=True, cases=99)
    adapter = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0.1)
    outcome = adapter(result)
    assert outcome.decision == "reverted"
    assert result.scores is adapter.last_report.log_scores is None
    if mode == "static":
        assert result.safety is adapter.last_report.log_safety is None
    else:
        assert result.safety == adapter.last_report.log_safety
        assert result.safety.case_file == "scripted-safety.json" and result.safety.cases == 2
    assert cache.read_bytes() == original_cache


def test_starting_meta_validates_uses_replay_safety_hash_and_is_never_overwritten(case):
    before = policy_snapshot(case.root)
    path = ensure_starting_meta(case.root, case.base)
    assert path == case.root / "policies/policy-v1.meta.json"
    meta = PolicyMeta.model_validate_json(path.read_text())
    policy = parse_policy(case.base.read_text())
    expected = replay.expected_meta([], policy, policy, case.base.read_text())
    assert meta.policy_version == "policy-v1" and meta.parent is None
    assert meta.policy_sha256 == hashlib.sha256(case.base.read_text().encode("utf-8")).hexdigest()
    assert meta.safety_sha256 == expected.safety_sha256
    assert {"C1", "C2"} <= set(meta.base_rules) and meta.learned_rules == {}
    assert all(policy_snapshot(case.root)[p] == text for p, text in before.items())
    path.write_bytes(b"existing metadata must be left exactly as found\n")
    assert ensure_starting_meta(case.root, case.base) == path
    assert path.read_bytes() == b"existing metadata must be left exactly as found\n"


@pytest.mark.parametrize("mode", ["complete", "static", "partial", "safety_failed", "safety_raised", "no_gate"])
def test_finish_round_logs_only_measurements_actually_available(case, mode):
    result = candidate_result(case)
    assert result.scores is result.safety is None
    if mode == "static":
        write_policy(case.candidate, "policy-v2-candidate")
    elif mode == "partial":
        case.deps.judge = Judge(errors=["b"])
    elif mode == "safety_failed":
        case.deps.safety = Safety(False)
    elif mode == "safety_raised":
        case.deps.safety = Safety(error=RuntimeError("no safety result"))
    outcome = None
    if mode != "no_gate":
        report = run_gate(case.inp, case.deps)
        result.scores, result.safety = report.log_scores, report.log_safety
        outcome = report.outcome
    else:
        # Even populated fields must be omitted when no gate outcome was given.
        result.scores = LogScores(before=Scores(empathy=2, relevance=2),
                                  after=Scores(empathy=3, relevance=3), noise=0)
        result.safety = LogSafety(case_file="unused.json", passed=True, cases=1)
    memory, log = FakeMemory(), DecisionLog(case.root)
    memory.ensure_training_audience()
    before = policy_snapshot(case.root)
    event = round_api.finish_round(result, outcome, log, memory, case.root)
    if outcome is None:
        assert event.scores is event.safety is None
    else:
        assert event.scores == result.scores and event.safety == result.safety
    if mode == "complete":
        assert event.scores is not None and event.safety is not None
    elif mode == "safety_failed":
        assert event.scores is not None and event.safety is not None and event.safety.passed is False
    elif mode == "partial":
        assert event.scores is None and event.safety is not None
    elif mode == "static":
        assert event.scores is event.safety is None
    elif mode == "safety_raised":
        assert event.safety is None
    assert log.rounds() == [event]
    assert policy_snapshot(case.root) == before


# 4 and 6. Real graph integration: provenance, disjoint IDs and round-event evidence.

@pytest.mark.usefixtures("local_mlflow")
@pytest.mark.parametrize("measured", [False, True])
def test_graph_real_gate_artifacts_and_promotion_replay(case, measured):
    class ComparingJudge(Judge):
        def score(self, record):
            self.values = (3, 3, 4, 2) if record.policy_version == "policy-v1" else (4, 4, 4, 1)
            return super().score(record)
    case.deps.noise_measured = measured
    case.deps.judge = ComparingJudge()
    adapter = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0.1)
    deps = graph.RoundDeps(
        helper_client=Chat(), user_client=Chat("My hours were cut."), llm=Agents(),
        search=lambda query: [{"url": URL, "title": "Listening", "text": "Ask first."}],
        memory=FakeMemory(), log=DecisionLog(case.root), root=case.root,
        gate=adapter, start_policy=case.base,
    )
    # A candidate created by the graph, rather than the fixture's candidate.
    case.candidate.unlink()
    base_bytes = case.base.read_bytes()
    state = graph.run_round(deps, policy_path=case.base, scenarios=case.deps.scenarios, max_turns=1)
    event = state["event"]
    assert event.outcome == ("kept" if measured else "reverted")
    assert event.scores == adapter.last_report.log_scores and event.scores is not None
    assert event.safety == adapter.last_report.log_safety and event.safety is not None
    assert deps.log.rounds() == [event]
    training = {r["id"] for r in rows(case.root / "conversations/round-01-train.jsonl")}
    validation = rows(case.root / "conversations/round-01-validation.jsonl")
    validation_ids = {r["id"] for r in validation}
    assert {r["scenario_id"] for r in validation} == {"a", "b", "c"}
    assert validation_ids == {f"conv-r01-val-{s}" for s in ("a", "b", "c")}
    assert training.isdisjoint(validation_ids)
    assert {r["conversation_id"] for r in rows(case.root / "scores/round-01-validation.jsonl")} == validation_ids
    assert {r["id"] for r in rows(case.root / "conversations/round-00-current.jsonl")} == {
        f"conv-r00-current-{s}" for s in ("a", "b", "c")
    }
    assert len(rows(case.root / "scores/round-00-current.jsonl")) == 3
    assert (case.root / "scores/current.json").is_file()
    assert (case.root / "scores/round-01-safety.json").is_file()
    assert_report(adapter.last_report, case.root, 1)
    assert (case.root / "policies/policy-v1.meta.json").is_file()
    assert case.base.read_bytes() == base_bytes
    promoted = case.root / "policies/policy-v2.md"
    assert promoted.exists() is measured
    assert (case.root / "policies/policy-v2.meta.json").exists() is measured
    expected_policies = {case.base, state["result"].candidate_path}
    expected_meta = {case.root / "policies/policy-v1.meta.json"}
    if measured:
        expected_policies.add(promoted)
        expected_meta.add(case.root / "policies/policy-v2.meta.json")
    assert set((case.root / "policies").glob("*.md")) == expected_policies
    assert set((case.root / "policies").glob("*.meta.json")) == expected_meta
    assert replay.check(deps.log, case.root, case.base) == (promoted if measured else case.base)
    if measured:
        meta = PolicyMeta.model_validate_json((case.root / "policies/policy-v2.meta.json").read_text())
        assert meta.parent == "policy-v1" and "C3" in meta.learned_rules


@pytest.mark.usefixtures("local_mlflow")
def test_starting_meta_is_after_price_check_and_before_graph(case, monkeypatch):
    order = []
    original_meta, original_build = gate.ensure_starting_meta, graph.build_round_graph

    def meta(root, path):
        order.append("meta")
        assert root == case.root and path == case.base
        return original_meta(root, path)

    def build(deps):
        compiled = original_build(deps)

        def invoke(*args, **kwargs):
            order.append("graph")
            assert (case.root / "policies/policy-v1.meta.json").is_file()
            return compiled.invoke(*args, **kwargs)

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(gate, "ensure_starting_meta", meta)
    # Also replace an existing direct import of this public API, if used.
    for alias, value in tuple(vars(graph).items()):
        if value is original_meta:
            monkeypatch.setattr(graph, alias, meta)
    monkeypatch.setattr(graph, "build_round_graph", build)
    deps = graph.RoundDeps(helper_client=Chat(), user_client=Chat("Worry."), llm=Agents(),
                           search=lambda q: [], memory=FakeMemory(), log=DecisionLog(case.root),
                           root=case.root, start_policy=case.base)
    deps.helper_client.model = "unpriced-acceptance-model"
    with pytest.raises(MissingPrice):
        graph.run_round(deps, policy_path=case.base, scenarios=[], max_turns=1)
    assert order == [] and not (case.root / "policies/policy-v1.meta.json").exists()
    del deps.helper_client.model
    graph.run_round(deps, policy_path=case.base, scenarios=[], max_turns=1)
    assert order == ["meta", "graph"]


@pytest.mark.usefixtures("local_mlflow")
def test_absent_start_policy_preserves_old_graph_behavior(case, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Starting metadata was requested without start_policy")
    original = gate.ensure_starting_meta
    monkeypatch.setattr(gate, "ensure_starting_meta", forbidden)
    for alias, value in tuple(vars(graph).items()):
        if value is original:
            monkeypatch.setattr(graph, alias, forbidden)
    deps = graph.RoundDeps(helper_client=Chat(), user_client=Chat("Worry."), llm=Agents(),
                           search=lambda q: [], memory=FakeMemory(), log=DecisionLog(case.root), root=case.root)
    assert deps.start_policy is None
    event = graph.run_round(deps, policy_path=case.base, scenarios=[], max_turns=1)["event"]
    assert event.scores is event.safety is None
    assert not list((case.root / "policies").glob("*.meta.json"))


# 5. CLI additions. Existing graph-script acceptance tests cover unchanged options.

@pytest.mark.usefixtures("local_mlflow")
@pytest.mark.parametrize("validation_file", [False, True])
def test_fake_real_cli_uses_all_validation_scenarios_and_defaults_to_unmeasured_noise(case, capsys, validation_file):
    train = case.root / "train.jsonl"
    validation = case.root / "validation.jsonl"
    train.write_text(case.deps.scenarios[0].model_dump_json() + "\n")
    validation.write_text("".join(s.model_dump_json() + "\n" for s in case.deps.scenarios[1:]))
    root = case.root / "cli-run"
    args = ["--fake", "--gate", "real", "--root", str(root), "--policy", str(case.base),
            "--scenarios", str(train), "--max-turns", "1", "--mlflow", f"sqlite:///{case.root}/mlflow.db"]
    if validation_file:
        args += ["--validation", str(validation)]
    capsys.readouterr()
    assert script.main(args) == 0
    event = DecisionLog(root).rounds()[0]
    captured = capsys.readouterr()
    assert captured.out == f"round 1: reverted. {event.reason}\n"
    assert "noise" in event.reason.lower() and "measur" in event.reason.lower()
    report = json.loads((root / "scores/round-01-gate.json").read_text())
    assert report["failed"][-1] == "noise_not_measured"
    assert event.scores is not None and event.scores.noise == 0.0
    assert event.safety is not None and event.safety.passed
    expected = {"b", "c"} if validation_file else {"a"}
    assert {r["scenario_id"] for r in rows(root / "conversations/round-01-validation.jsonl")} == expected
    assert (root / "policies/policy-v1.meta.json").is_file()
    assert not (root / "policies/policy-v2.md").exists()


@pytest.mark.usefixtures("local_mlflow")
@pytest.mark.parametrize("mode", ["real", "kept", "reverted"])
def test_cli_forwards_start_policy_for_every_gate_mode_and_accepts_real_gate_flags(case, monkeypatch, mode):
    scenarios = case.root / "cli-scenarios.jsonl"
    scenarios.write_text(case.deps.scenarios[0].model_dump_json() + "\n")
    calls, configured = [], []

    original_make = gate.make_gate

    def configure(deps, **kwargs):
        configured.append(deps)
        assert deps.noise_measured is True
        assert kwargs["validation_scenario_ids"] == ["a"]
        assert kwargs["noise"] == 0.25 and kwargs["token_cap"] == 777
        return original_make(deps, **kwargs)

    monkeypatch.setattr(gate, "make_gate", configure)
    for alias, value in tuple(vars(script).items()):
        if value is original_make:
            monkeypatch.setattr(script, alias, configure)

    def run(deps, **kwargs):
        calls.append(deps)
        assert deps.start_policy == case.base
        return {"event": SimpleNamespace(round=1, outcome="no_pattern", reason="No pattern.")}

    original = graph.run_round
    monkeypatch.setattr(graph, "run_round", run)
    for alias, value in tuple(vars(script).items()):
        if value is original:
            monkeypatch.setattr(script, alias, run)
    assert script.main([
        "--fake", "--gate", mode, "--root", str(case.root / "flags"), "--policy", str(case.base),
        "--scenarios", str(scenarios), "--validation", str(scenarios),
        "--noise", "0.25", "--noise-measured", "--token-cap", "777",
        "--mlflow", f"sqlite:///{case.root}/mlflow.db",
    ]) == 0
    assert len(calls) == 1
    if mode == "real":
        assert configured


def test_nonfake_real_cli_refuses_before_any_file_is_written(case, capsys):
    scenarios = case.root / "nonfake-scenarios.jsonl"
    scenarios.write_text(case.deps.scenarios[0].model_dump_json() + "\n")
    before = {p.relative_to(case.root): p.read_bytes() for p in case.root.rglob("*") if p.is_file()}
    capsys.readouterr()
    assert script.main(["--gate", "real", "--root", str(case.root / "refused"),
                        "--policy", str(case.base), "--scenarios", str(scenarios)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert len(captured.err.splitlines()) == 1
    assert all(word in captured.err.lower() for word in ("judge", "safety", "later"))
    assert {p.relative_to(case.root): p.read_bytes() for p in case.root.rglob("*") if p.is_file()} == before
