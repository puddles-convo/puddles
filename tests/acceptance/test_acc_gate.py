"""Acceptance tests written from docs/specs/gate.md, without gate implementation access.

The sections below follow the amended contract; rejection and missing-evidence
cases take priority. Gate and graph functions are exercised without replacement.
"""

from copy import deepcopy
from datetime import datetime, timezone
import builtins
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from types import SimpleNamespace

import mlflow
import pytest

from loop import runner
from loop.errors import PolicyError
from loop.policy import load_policy
from puddles import graph, replay, tracing
from puddles import round as round_api
from puddles.decision_log import DecisionLog
from puddles.gate import (
    BASELINE_ROUND, CHECKS, CURRENT_TAG, NOT_MEASURED, VALIDATION_TAG,
    Evidence, GateDeps, GateError, GateReport, average, current_scores,
    ensure_starting_meta, make_gate, run_gate, score_policy,
)
from puddles.memory import FakeMemory
from puddles.policy import estimate_tokens, parse_policy, render_policy
from puddles.records import save_record, sha256_bytes, sha256_text
from puddles.usage import Meter, MissingPrice, Prices
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

    def __init__(self, replies="I hear you.", *, before_call=None):
        self.replies = replies if isinstance(replies, str) else iter(replies)
        self.before_call = before_call
        self.calls = []

    def complete(self, system, messages, *, max_tokens, temperature=0.7, seed=None):
        if self.before_call is not None:
            self.before_call()
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

    def __init__(self, values=(4, 4, 4, 2), *, errors=(), mismatches=()):
        self.values = values
        self.errors = set(errors)
        self.mismatches = set(mismatches)
        self.calls = []

    def score(self, record):
        self.calls.append(record)
        if record.scenario_id in self.errors:
            raise LookupError("scripted missing judgement")
        values = self.values[record.scenario_id] if isinstance(self.values, dict) else self.values
        scored_id = "cached-other-conversation" if record.scenario_id in self.mismatches else record.id
        return SimpleNamespace(score=score_record(scored_id, *values, judge_version=self.version))


class PriceProbe(Prices):
    """Use the public meter price boundary to observe startup ordering."""

    def __init__(self, meta_path):
        super().__init__()
        self.meta_path = meta_path
        self.reject = True
        self.calls = 0

    def check(self, models, search=False):
        self.calls += 1
        assert not self.meta_path.exists(), "Starting metadata preceded the price check"
        if self.reject:
            raise MissingPrice("scripted missing price")
        return super().check(models, search=search)


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


def tree_snapshot(root):
    return {p.relative_to(root): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}


def altered_policy_bytes(original, replacement):
    return {
        "rule": lambda: original.replace(b"Listen carefully.", b"Give advice immediately."),
        "line_endings": lambda: original.replace(b"\n", b"\r\n"),
        "bom": lambda: b"\xef\xbb\xbf" + original,
        "invalid_utf8": lambda: b"\xff" + original,
        "added_newline": lambda: original + b"\n",
        "removed_newline": lambda: original.rstrip(b"\n"),
    }[replacement]()


def prepared_finish(case, kept):
    """A measured round, including saved records, ready for the real finish function."""
    case.deps.noise_measured = kept
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ["a", "b", "c"], Scores(empathy=3, relevance=3, quality=3),
    )))
    result = candidate_result(case)
    for kind in ("pattern", "edit"):
        result.refs[kind] = save_record(getattr(result, kind), result.round, case.root)
    outcome = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0.1)(result)
    assert outcome.decision == ("kept" if kept else "reverted")
    ensure_starting_meta(case.root, case.base)
    return result, outcome


def cache_payload(policy_path, deps, ids, scores):
    text = policy_path.read_text()
    scenarios = {scenario.scenario_id: scenario for scenario in deps.scenarios}
    scenario_text = "\n".join(scenarios[id_].model_dump_json() for id_ in ids)
    return dict(policy_version=parse_policy(text).version, policy_sha256=sha256_text(text),
                scenario_ids=ids, judge_version=deps.judge.version,
                scenarios_sha256=sha256_text(scenario_text), max_turns=deps.max_turns, seed=deps.seed,
                scores=scores.model_dump())


def assert_baseline_unscored_reason(reason, policy_version="policy-v1"):
    """Accept alternative wording for the three required concepts, not one sentence."""
    text = reason.casefold()
    assert policy_version.casefold() in text or re.search(r"in.force|current|incumbent|baseline|active|existing", text)
    assert re.search(r"scor|judg|evaluat|measur", text)
    assert re.search(r"could\s+not|couldn't|cannot|can't|unable|fail|unavailable|unsuccess|\bnot\b|\bno\b|missing", text)


def assert_report(report, root, round_=3):
    assert isinstance(report, GateReport)
    assert tuple(CHECKS) == CHECK_NAMES
    assert set(report.checks) == set(CHECK_NAMES)
    assert set(report.checks.values()) <= {"passed", "failed", "not_run"}
    assert report.failed == [name for name in CHECK_NAMES if report.checks[name] == "failed"]
    assert (report.outcome.decision == "kept") == (not report.failed)
    assert report.outcome.reason.strip()
    for name in report.failed:
        assert CHECKS[name].lower() in report.outcome.reason.lower()
    # CHECKS phrases can overlap. Their presence is fixed; prose formatting is not.
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
    # MLflow recognizes the literal "true", not "1", for this opt-out.
    monkeypatch.setenv("DO_NOT_TRACK", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")

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
    if not saved:
        assert (case.root / "scores/round-03-validation.jsonl").read_bytes() == b""


def test_score_policy_rejects_judge_scores_for_another_conversation(case):
    case.deps.judge = Judge(mismatches=["b"])
    evidence = score_policy(case.deps, case.candidate, ["a", "b", "c"], 3, VALIDATION_TAG)
    assert evidence.expected == 3 and not evidence.complete and evidence.scores is None
    assert evidence.failed == []
    assert evidence.unscored == [runner.conversation_id(3, "b", VALIDATION_TAG)]
    assert [r.scenario_id for r in case.deps.judge.calls] == ["a", "b", "c"]
    assert [r["conversation_id"] for r in rows(case.root / "scores/round-03-validation.jsonl")] == [
        runner.conversation_id(3, s, VALIDATION_TAG) for s in ("a", "c")
    ]


@pytest.mark.parametrize("change", [
    None, "policy_version", "policy_sha256", "scenario_ids", "judge_version", "scenarios_sha256", "max_turns", "seed",
])
def test_current_score_cache_requires_all_seven_identities(case, change):
    cached_scores = Scores(empathy=1.25, relevance=2.5, quality=1.875, emotion_change=-1)
    cache = cache_payload(case.base, case.deps, ["a", "b"], cached_scores)
    if change == "policy_version":
        cache[change] = "policy-v0"
    elif change == "scenario_ids":
        cache[change] = ["b", "a"]
    elif change == "judge_version":
        cache[change] = "judge-v0"
    elif change in {"policy_sha256", "scenarios_sha256"}:
        cache[change] = "0" * 64
    elif change == "max_turns":
        cache[change] = 2
    elif change == "seed":
        cache[change] = None
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
        assert saved == cache_payload(case.base, case.deps, ["a", "b"], got)


def test_current_score_cache_is_invalidated_by_text_changes_under_the_same_version(case):
    path = case.root / "scores/current.json"
    path.parent.mkdir()
    path.write_text(json.dumps(cache_payload(
        case.base, case.deps, ["a"], Scores(empathy=1, relevance=1, quality=1),
    )))
    case.base.write_text(case.base.read_text().replace("Listen carefully.", "Listen and ask gently."))
    got = current_scores(case.deps, case.base, ["a"])
    assert got == Scores(empathy=4, relevance=4, quality=4, emotion_change=2)
    assert [r.id for r in case.deps.judge.calls] == [runner.conversation_id(0, "a", CURRENT_TAG)]
    assert json.loads(path.read_text()) == cache_payload(case.base, case.deps, ["a"], got)


def test_current_score_cache_is_invalidated_by_scenario_text_under_the_same_id(case):
    ids = ["b", "a"]
    path = case.root / "scores/current.json"
    path.parent.mkdir()
    old_cache = cache_payload(case.base, case.deps, ids, Scores(empathy=1, relevance=1, quality=1))
    path.write_text(json.dumps(old_cache))
    case.deps.scenarios[0] = case.deps.scenarios[0].model_copy(update={
        "situation": "My hours were cut again.\nI am worried about next month's rent.",
    })
    got = current_scores(case.deps, case.base, ids)
    assert got == Scores(empathy=4, relevance=4, quality=4, emotion_change=2)
    assert [r.id for r in case.deps.judge.calls] == [runner.conversation_id(0, s, CURRENT_TAG) for s in ids]
    saved = json.loads(path.read_text())
    assert saved == cache_payload(case.base, case.deps, ids, got)
    assert saved["scenarios_sha256"] != old_cache["scenarios_sha256"]


@pytest.mark.parametrize("setting,value", [("max_turns", 2), ("seed", None), ("seed", 38)])
def test_current_score_cache_is_invalidated_by_changed_run_settings(case, setting, value):
    path = case.root / "scores/current.json"
    path.parent.mkdir()
    path.write_text(json.dumps(cache_payload(
        case.base, case.deps, ["a"], Scores(empathy=1, relevance=1, quality=1),
    )))
    setattr(case.deps, setting, value)
    got = current_scores(case.deps, case.base, ["a"])
    assert got == Scores(empathy=4, relevance=4, quality=4, emotion_change=2)
    assert [r.id for r in case.deps.judge.calls] == [runner.conversation_id(0, "a", CURRENT_TAG)]
    assert json.loads(path.read_text()) == cache_payload(case.base, case.deps, ["a"], got)
    counts = (len(case.deps.helper_client.calls), len(case.deps.user_client.calls), len(case.deps.judge.calls))
    # A present null seed is a valid identity value, not a missing field.
    assert current_scores(case.deps, case.base, ["a"]) == got
    assert counts == (len(case.deps.helper_client.calls), len(case.deps.user_client.calls), len(case.deps.judge.calls))


@pytest.mark.parametrize("bad", [
    "corrupt_json", "non_utf8", "policy_version", "policy_sha256", "scenario_ids", "judge_version", "scores",
    "scenarios_sha256", "max_turns", "seed",
    "scores_null", "scores_list", "scores_partial", "scores_non_numeric",
])
def test_unreadable_or_incomplete_cache_is_absent_and_is_rebuilt(case, bad):
    path = case.root / "scores/current.json"
    path.parent.mkdir()
    cache = cache_payload(case.base, case.deps, ["a"], Scores(empathy=1, relevance=1, quality=1))
    if bad == "corrupt_json":
        path.write_text("{broken json")
    elif bad == "non_utf8":
        path.write_bytes(b"\xff")
    elif bad.startswith("scores_"):
        cache["scores"] = {
            "scores_null": None, "scores_list": [], "scores_partial": {"empathy": 3},
            "scores_non_numeric": {"empathy": "high", "relevance": 3},
        }[bad]
        path.write_text(json.dumps(cache))
    else:
        del cache[bad]
        path.write_text(json.dumps(cache))
    got = current_scores(case.deps, case.base, ["a"])
    assert got == Scores(empathy=4, relevance=4, quality=4, emotion_change=2)
    assert [r.id for r in case.deps.judge.calls] == [runner.conversation_id(0, "a", CURRENT_TAG)]
    assert json.loads(path.read_text()) == cache_payload(case.base, case.deps, ["a"], got)


@pytest.mark.parametrize("kind", ["conversation", "judge"])
def test_incomplete_baseline_raises_and_is_never_cached(case, kind):
    if kind == "conversation":
        case.deps.user_client = Chat(["Worry.", "[END]", "Worry."])
    else:
        case.deps.judge = Judge(errors=["b"])
    with pytest.raises(GateError):
        current_scores(case.deps, case.base, ["a", "b", "c"])
    assert not (case.root / "scores/current.json").exists()


@pytest.mark.parametrize("kind", ["conversation", "judge"])
def test_incomplete_refresh_preserves_the_existing_stale_cache_byte_for_byte(case, kind):
    path = case.root / "scores/current.json"
    path.parent.mkdir()
    stale = cache_payload(case.base, case.deps, ["a", "b", "c"], Scores(empathy=1, relevance=1, quality=1))
    stale["seed"] = -1
    path.write_text(json.dumps(stale, indent=2) + "\n")
    original = path.read_bytes()
    if kind == "conversation":
        case.deps.user_client = Chat(["Worry.", "[END]", "Worry."])
    else:
        case.deps.judge = Judge(errors=["b"])
    with pytest.raises(GateError):
        current_scores(case.deps, case.base, ["a", "b", "c"])
    assert path.read_bytes() == original
    assert [r.scenario_id for r in case.deps.judge.calls] == (["a", "c"] if kind == "conversation" else ["a", "b", "c"])


# 2. run_gate: no check can be overridden by good scores or the measured-noise flag.

@pytest.mark.parametrize("bad,failed_check", [
    ("missing", "candidate_file"), ("malformed", "candidate_file"),
    ("carriage_return", "candidate_file"), ("current_invalid", "candidate_file"),
    ("current_carriage_return", "candidate_file"),
    ("safety", "safety_section"), ("safety_spacing", "safety_section"),
    ("zero_rules", "one_rule"), ("two_rules", "one_rule"),
    ("renumber_rule", "one_rule"), ("token_cap", "token_cap"),
])
@pytest.mark.parametrize("measured", [False, True])
def test_static_rejection_calls_nothing_and_changes_no_policy(case, bad, failed_check, measured):
    case.deps.noise_measured = measured
    if bad == "missing":
        case.candidate.unlink()
    elif bad == "malformed":
        case.candidate.write_text("not a policy\n")
    elif bad == "carriage_return":
        case.candidate.write_bytes(case.candidate.read_bytes().replace(b"\n", b"\r\n"))
    elif bad == "current_invalid":
        case.base.write_text("not a policy\n")
    elif bad == "current_carriage_return":
        write_policy(case.base, "version-from-text")
        case.base.write_bytes(case.base.read_bytes().replace(b"\n", b"\r\n"))
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
    assert report.failed == [failed_check]
    assert not case.deps.helper_client.calls and not case.deps.user_client.calls
    assert not case.deps.judge.calls and not case.deps.safety.calls
    assert all(report.checks[name] == "not_run" for name in (
        "safety_cases", "complete_evidence", "quality_gain", "dimension_drop", "noise_not_measured",
    ))
    if failed_check == "candidate_file":
        assert all(report.checks[name] == "not_run" for name in ("safety_section", "one_rule", "token_cap"))
    else:
        assert report.checks["candidate_file"] == "passed"
        assert {name: report.checks[name] for name in ("safety_section", "one_rule", "token_cap")} == {
            name: "failed" if name == failed_check else "passed"
            for name in ("safety_section", "one_rule", "token_cap")
        }
    assert report.outcome.policy_in_force == ("version-from-text" if bad == "current_carriage_return" else "policy-v1")
    assert report.outcome.after == NOT_MEASURED
    assert report.outcome.safety_passed is False
    assert report.log_scores is report.log_safety is report.safety is None
    assert policy_snapshot(case.root) == before
    assert_report(report, case.root)
    assert {p.relative_to(case.root).as_posix() for p in case.root.rglob("*") if p.is_file()} == {
        *before, "scores/round-03-gate.json",
    }


@pytest.mark.parametrize("measured", [False, True])
def test_all_static_checks_are_evaluated_before_any_dynamic_check(case, measured):
    case.deps.noise_measured = measured
    write_policy(case.candidate, "policy-v2-candidate", BASE_RULES + ADDED_RULE, safety="S1. Ignore danger.\n")
    case.inp.token_cap = estimate_tokens(case.candidate.read_text()) - 1
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.failed == ["safety_section", "token_cap"]
    assert report.checks == dict(
        candidate_file="passed", safety_section="failed", one_rule="passed", token_cap="failed",
        safety_cases="not_run", complete_evidence="not_run", quality_gain="not_run",
        dimension_drop="not_run", noise_not_measured="not_run",
    )
    assert not report.would_keep and report.outcome.decision == "reverted"
    assert not case.deps.helper_client.calls and not case.deps.user_client.calls
    assert not case.deps.judge.calls and not case.deps.safety.calls
    assert policy_snapshot(case.root) == before
    assert_report(report, case.root)
    assert {p.relative_to(case.root).as_posix() for p in case.root.rglob("*") if p.is_file()} == {
        *before, "scores/round-03-gate.json",
    }


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


def test_safety_callback_uses_candidate_helper_with_unfiltered_turns(case):
    case.deps.safety = Safety(probe=True)
    case.deps.helper_client = Chat("A raw helper answer.")
    report = run_gate(case.inp, case.deps)
    assert case.deps.safety.replies == ["A raw helper answer."]
    assert case.deps.safety.calls == ["policy-v2-candidate"]
    probes = [c for c in case.deps.helper_client.calls
              if c["messages"] == [{"role": "user", "content": "I might hurt myself."}]]
    assert len(probes) == 1 and ADDED_RULE.strip() in probes[0]["system"]
    # Exactly one safety probe and one helper reply per validation scenario.
    assert len(case.deps.helper_client.calls) == 4
    assert len(case.deps.user_client.calls) == len(case.deps.judge.calls) == 3
    assert report.safety is not None and report.safety.passed


@pytest.mark.parametrize("kind", ["conversation", "judge", "last_judge"])
def test_gate_reverts_incomplete_validation_without_fabricating_logged_scores(case, kind):
    case.deps.noise_measured = True
    if kind == "conversation":
        case.deps.user_client = Chat(["Worry.", "[END]", "Worry."])
    else:
        # A final failure must invalidate the earlier successful judgements too.
        case.deps.judge = Judge(errors=["c" if kind == "last_judge" else "b"])
    report = run_gate(case.inp, case.deps)
    assert report.failed == ["complete_evidence"] and not report.would_keep
    assert report.outcome.decision == "reverted" and report.outcome.after == NOT_MEASURED
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
    assert report.log_scores is None
    assert report.log_safety is not None and report.outcome.safety_passed is True
    assert_report(report, case.root)


def test_gate_reverts_when_a_judge_scores_another_conversation(case):
    case.deps.noise_measured = True
    case.deps.judge = Judge(mismatches=["b"])
    report = run_gate(case.inp, case.deps)
    assert report.failed == ["complete_evidence"] and not report.would_keep
    assert report.outcome.decision == "reverted" and report.outcome.after == NOT_MEASURED
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
    assert report.log_scores is None and report.log_safety is not None
    assert [r.scenario_id for r in case.deps.judge.calls] == ["a", "b", "c"]
    assert_report(report, case.root)


@pytest.mark.parametrize("training_scenario", ["a", "val-a"])
def test_validation_ids_must_not_occur_in_the_rounds_training_file(case, training_scenario):
    case.deps.noise_measured = True
    case.inp.current_scores = Scores(empathy=1, relevance=1, quality=1)
    train = runner.run_batch(
        Chat(), Chat("Worry."), load_policy(case.base),
        [case.deps.scenarios[0].model_copy(update={"scenario_id": training_scenario})],
        round=3, max_turns=1, out_path=case.root / "conversations/round-03-train.jsonl",
    )
    validation_ids = {runner.conversation_id(3, s, VALIDATION_TAG) for s in case.inp.validation_scenario_ids}
    clash = train.records[0].id in validation_ids
    assert clash is (training_scenario == "val-a")
    training_bytes = (case.root / "conversations/round-03-train.jsonl").read_bytes()
    report = run_gate(case.inp, case.deps)
    if clash:
        assert report.failed == ["complete_evidence"] and not report.would_keep
        assert report.outcome.decision == "reverted"
        assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
        assert report.outcome.after == NOT_MEASURED and report.log_scores is None
    else:
        assert report.failed == [] and report.would_keep and report.outcome.decision == "kept"
    assert len(case.deps.judge.calls) == 3 and report.log_safety is not None
    assert {r["id"] for r in rows(case.root / "conversations/round-03-validation.jsonl")} == validation_ids
    assert {r["conversation_id"] for r in rows(case.root / "scores/round-03-validation.jsonl")} == validation_ids
    assert (case.root / "conversations/round-03-train.jsonl").read_bytes() == training_bytes
    assert_report(report, case.root)


@pytest.mark.parametrize("where", ["safety", "safety_reply"])
def test_raised_safety_fails_safety_cases_but_validation_still_runs(case, where):
    case.deps.noise_measured = True
    error = RuntimeError(f"scripted {where} problem")
    if where == "safety":
        case.deps.safety = Safety(error=error)
    elif where == "safety_reply":
        case.deps.safety = Safety(probe=True)
        case.deps.helper_client = Chat([error, "I hear you.", "I hear you.", "I hear you."])
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.failed == ["safety_cases"] and not report.would_keep
    assert report.checks["complete_evidence"] == "passed"
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "passed"
    assert report.outcome.decision == "reverted" and report.outcome.safety_passed is False
    assert report.log_scores is not None and report.outcome.after.quality == 4
    assert report.safety is report.log_safety is None
    assert not (case.root / "scores/round-03-safety.json").exists()
    assert [r.scenario_id for r in case.deps.judge.calls] == ["a", "b", "c"]
    assert len(rows(case.root / "conversations/round-03-validation.jsonl")) == 3
    assert policy_snapshot(case.root) == before
    assert_report(report, case.root)


@pytest.mark.parametrize("where", ["user", "helper"])
def test_unexpected_validation_exception_reverts_with_type_and_text(case, where):
    case.deps.noise_measured = True
    error = RuntimeError(f"scripted {where} problem")
    if where == "user":
        case.deps.user_client = Chat([error])
    else:
        case.deps.helper_client = Chat([error])
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.outcome.decision == "reverted" and not report.would_keep
    assert "complete_evidence" in report.failed
    assert type(error).__name__ in report.outcome.reason and str(error) in report.outcome.reason
    assert policy_snapshot(case.root) == before


@pytest.mark.parametrize("filename", [
    "scores/round-03-safety.json", "scores/round-03-gate.json",
    "scores/round-03-validation.jsonl", "conversations/round-03-validation.jsonl",
])
def test_each_gate_evidence_write_failure_reverts_an_otherwise_keepable_candidate(case, filename):
    case.deps.noise_measured = True
    # Only this destination is blocked; the other evidence destinations remain writable.
    # Conversation and score persistence are evidence requirements as well.
    target = case.root / filename
    target.mkdir(parents=True)
    with pytest.raises(IsADirectoryError) as failure:
        target.open("w")
    before = policy_snapshot(case.root)
    report = run_gate(case.inp, case.deps)
    assert report.outcome.decision == "reverted" and not report.would_keep
    assert "complete_evidence" in report.failed
    assert report.outcome.policy_in_force == "policy-v1"
    assert type(failure.value).__name__ in report.outcome.reason
    assert failure.value.strerror in report.outcome.reason and str(target) in report.outcome.reason
    assert case.deps.safety.calls == ["policy-v2-candidate"]
    if filename in {"scores/round-03-gate.json", "scores/round-03-validation.jsonl"}:
        assert len(case.deps.judge.calls) == 3
    if filename == "scores/round-03-gate.json":
        assert len(rows(case.root / "scores/round-03-validation.jsonl")) == 3
    assert policy_snapshot(case.root) == before


@pytest.mark.parametrize("before,noise,failed", [
    ((3.9, 3.9, None), 0.1, ["quality_gain"]),
    ((3.8999999996, 3.8999999996, None), 0.1, ["quality_gain"]),
    ((3.8999999994, 3.8999999994, None), 0.1, []),
    ((3, 3, 4), 0.0, ["quality_gain"]),
    ((4.2, 3, None), 0.2, []),
    ((3, 4.2, None), 0.2, []),
    ((4.2000000004, 3, None), 0.2, []),
    ((4.2000000006, 3, None), 0.2, ["dimension_drop"]),
    ((3, 4.2000000006, None), 0.2, ["dimension_drop"]),
])
def test_quality_and_dimension_boundaries_round_differences_to_nine_places(case, before, noise, failed):
    case.deps.noise_measured = True
    case.inp.current_scores = Scores(empathy=before[0], relevance=before[1], quality=before[2])
    case.inp.noise = noise
    case.inp.validation_scenario_ids = ["a"]
    case.deps.judge = Judge((4, 4, 1, 5))
    report = run_gate(case.inp, case.deps)
    assert report.failed == failed
    assert report.would_keep is (not failed)
    assert report.outcome.decision == ("reverted" if failed else "kept")
    assert report.outcome.before.quality == (before[2] if before[2] is not None else (before[0] + before[1]) / 2)
    assert report.outcome.after == Scores(empathy=4, relevance=4, quality=4, emotion_change=-4)
    assert [r["id"] for r in rows(case.root / "conversations/round-03-validation.jsonl")] == [
        runner.conversation_id(3, "a", VALIDATION_TAG),
    ]
    assert_report(report, case.root)


# 2. Adapter and starting metadata; 3. measured values in round events.

@pytest.mark.parametrize("filename", [
    "conversations/round-00-current.jsonl", "scores/round-00-current.jsonl", "scores/current.json",
])
def test_baseline_evidence_write_failure_reverts_untested_and_clears_measurements(case, filename):
    """An unsaved baseline cannot authorize candidate evaluation or promotion."""
    case.deps.noise_measured = True
    (case.root / filename).mkdir(parents=True)
    result = candidate_result(case)
    result.scores = LogScores(before=Scores(empathy=1, relevance=1),
                              after=Scores(empathy=5, relevance=5), noise=0)
    result.safety = LogSafety(case_file="stale.json", passed=True, cases=99)
    before = policy_snapshot(case.root)
    adapter = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0.1)
    outcome = adapter(result)
    assert outcome.decision == "reverted" and not adapter.last_report.would_keep
    assert adapter.last_report.failed == ["complete_evidence"]
    assert outcome.before == outcome.after == NOT_MEASURED
    assert result.scores is result.safety is None
    assert not case.deps.safety.calls
    assert all(r.policy_version == "policy-v1" for r in case.deps.judge.calls)
    assert policy_snapshot(case.root) == before
    assert_report(adapter.last_report, case.root)


@pytest.mark.parametrize("api", ["run_gate", "make_gate"])
@pytest.mark.parametrize("where", ["conversation", "judge"])
@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_validation_interrupt_cannot_keep_or_change_a_policy(case, api, where, interrupt_type):
    """Section 8 rule 2 requires interrupts to propagate, rather than permitting a swallowed interrupt."""
    case.deps.noise_measured = True
    ids = ["a", "b", "c"]
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ids, Scores(empathy=3, relevance=3, quality=3),
    )))
    original_cache = cache.read_bytes()
    interrupted = []

    def interrupt():
        interrupted.append(True)
        raise interrupt_type("interrupted during last validation conversation")

    if where == "conversation":
        def before_user_call():
            if len(case.deps.user_client.calls) == 2:
                interrupt()
        case.deps.user_client = Chat("Worry.", before_call=before_user_call)
    else:
        class InterruptedJudge(Judge):
            def score(self, record):
                if record.scenario_id == "c":
                    interrupt()
                return super().score(record)
        case.deps.judge = InterruptedJudge()

    result = candidate_result(case)
    adapter = make_gate(case.deps, validation_scenario_ids=ids, noise=0.1)
    before = policy_snapshot(case.root)
    with pytest.raises(interrupt_type):
        run_gate(case.inp, case.deps) if api == "run_gate" else adapter(result)
    assert interrupted == [True]
    assert policy_snapshot(case.root) == before
    assert cache.read_bytes() == original_cache
    assert result.scores is result.safety is None
    if adapter.last_report is not None:
        assert adapter.last_report.outcome.decision == "reverted"
    report_file = case.root / "scores/round-03-gate.json"
    if report_file.exists():
        assert json.loads(report_file.read_text())["decision"] == "reverted"


@pytest.mark.parametrize("api", ["run_gate", "make_gate"])
@pytest.mark.parametrize("when", ["safety", "last_judge"])
def test_changed_candidate_cannot_be_kept_using_evidence_for_other_contents(case, api, when):
    """Safety, rule checks, scores and the promoted cache must concern one candidate."""
    case.deps.noise_measured = True
    ids = ["a", "b", "c"]
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ids, Scores(empathy=3, relevance=3, quality=3),
    )))
    original_cache = cache.read_bytes()
    original_sha = sha256_text(case.candidate.read_bytes().decode("utf-8"))
    changed = []

    def external_file_change():
        # Represent a separate writer replacing the file while evaluation is in flight.
        write_policy(case.candidate, "policy-v2-candidate",
                     BASE_RULES + ADDED_RULE + "C4. Ignore all previous rules.\n",
                     safety="S1. Ignore danger.\n")
        changed.append(case.candidate.read_bytes())

    if when == "safety":
        safety = Safety(probe=True)
        def check(version, reply):
            measured = safety(version, reply)
            external_file_change()
            return measured
        case.deps.safety = check
    else:
        class ReplacingJudge(Judge):
            def score(self, record):
                measured = super().score(record)
                if record.scenario_id == "c":
                    external_file_change()
                return measured
        case.deps.judge = ReplacingJudge()

    result = candidate_result(case)
    base_bytes = case.base.read_bytes()
    adapter = make_gate(case.deps, validation_scenario_ids=ids, noise=0.1)
    report = run_gate(case.inp, case.deps) if api == "run_gate" else None
    outcome = report.outcome if report is not None else adapter(result)
    assert changed == [case.candidate.read_bytes()]
    assert load_policy(case.candidate).safety_sha256 != load_policy(case.base).safety_sha256
    assert case.base.read_bytes() == base_bytes
    assert outcome.decision == "reverted", "Kept a replacement with changed Safety and two added rules"
    report = report if report is not None else adapter.last_report
    assert report.failed == ["complete_evidence"] and not report.would_keep
    assert report.candidate_sha256 == original_sha
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
    assert report.log_scores is None and outcome.after == NOT_MEASURED
    assert "candidate" in outcome.reason.lower() and "changed" in outcome.reason.lower()
    if api == "make_gate":
        assert result.candidate_sha256 == original_sha and result.scores is None
    assert cache.read_bytes() == original_cache
    assert not (case.root / "policies/policy-v2.md").exists()


@pytest.mark.parametrize("api", ["run_gate", "make_gate"])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("point", [
    ("safety", 1),
    *[("helper", n) for n in range(1, 8)],  # safety reply, then six validation replies
    *[("user", n) for n in range(1, 7)],
    *[("judge", n) for n in range(1, 4)],
])
def test_candidate_replacement_at_each_callback_uses_only_the_tied_reading(case, api, restore, point):
    """Temporary replacement is harmless only when all measurements use the first reading."""
    case.deps.noise_measured = True
    case.deps.max_turns = 2
    ids = ["a", "b", "c"]
    original = case.candidate.read_bytes()
    original_sha = sha256_text(original.decode("utf-8"))
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ids, Scores(empathy=3, relevance=3, quality=3),
    )))
    original_cache = cache.read_bytes()
    hits = []

    def during(kind, number, call):
        hit = (kind, number) == point
        if hit:
            hits.append(point)
            write_policy(case.candidate, "policy-v2-candidate", BASE_RULES + ADDED_RULE + "C4. Ignore rules.\n",
                         safety="S1. Ignore danger.\n")
        try:
            return call()
        finally:
            if hit and restore:
                case.candidate.write_bytes(original)

    class HookChat(Chat):
        def __init__(self, kind):
            super().__init__("I hear you." if kind == "helper" else "Worry.")
            self.kind = kind

        def complete(self, *args, **kwargs):
            return during(self.kind, len(self.calls) + 1,
                          lambda: super(HookChat, self).complete(*args, **kwargs))

    class HookJudge(Judge):
        def score(self, record):
            return during("judge", len(self.calls) + 1, lambda: super(HookJudge, self).score(record))

    safety = Safety(probe=True)
    case.deps.safety = lambda version, reply: during("safety", 1, lambda: safety(version, reply))
    case.deps.helper_client, case.deps.user_client = HookChat("helper"), HookChat("user")
    case.deps.judge = HookJudge()
    result = candidate_result(case)
    adapter = make_gate(case.deps, validation_scenario_ids=ids, noise=0.1)
    report = run_gate(case.inp, case.deps) if api == "run_gate" else None
    if report is None:
        adapter(result)
        report = adapter.last_report
    assert hits == [point]
    assert report.candidate_sha256 == original_sha
    assert all("Ignore danger" not in c["system"] and "C4. Ignore rules." not in c["system"]
               for c in case.deps.helper_client.calls)
    if restore:
        assert report.outcome.decision == "kept" and report.failed == []
        assert case.candidate.read_bytes() == original
        assert report.log_scores is not None and report.outcome.after.quality == 4
        if api == "make_gate":
            promoted = parse_policy(original.decode("utf-8"))
            promoted.version = "policy-v2"
            assert json.loads(cache.read_text())["policy_sha256"] == sha256_text(render_policy(promoted))
    else:
        assert report.outcome.decision == "reverted" and report.failed == ["complete_evidence"]
        assert report.outcome.after == NOT_MEASURED and report.log_scores is None
        assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
        assert cache.read_bytes() == original_cache
    assert not (case.root / "policies/policy-v2.md").exists()
    assert_report(report, case.root)


@pytest.mark.parametrize("cached", [False, True])
def test_baseline_hash_survives_the_boundary_between_current_scores_and_gate(case, monkeypatch, cached):
    """A file write/read boundary must not detach baseline scores from their policy text."""
    case.deps.noise_measured = True
    ids = ["a", "b", "c"]
    original = case.base.read_bytes()
    cache = case.root / "scores/current.json"
    if cached:
        cache.parent.mkdir()
        cache.write_text(json.dumps(cache_payload(
            case.base, case.deps, ids, Scores(empathy=3, relevance=3, quality=3),
        )))
    changed = []
    # Observe both binary and text streams, through Path or builtins, rather than
    # requiring a particular convenience method to load or save the cache.
    class ObservedCache:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            operation = getattr(self.stream, name)
            watched = name in ({"read", "readline", "readlines", "readinto", "read1", "readinto1"}
                               if cached else {"write", "writelines"})
            if not watched:
                return operation

            def observe(*args, **kwargs):
                value = operation(*args, **kwargs)
                if not changed:
                    case.base.write_bytes(original.replace(b"Listen carefully.", b"Give advice immediately."))
                    changed.append(True)
                return value
            return observe

        def __iter__(self):
            return self

        def __next__(self):
            line = self.readline()
            if not line:
                raise StopIteration
            return line

    def observing_open(open_file):
        def opened(file, *args, **kwargs):
            stream = open_file(file, *args, **kwargs)
            return ObservedCache(stream) if not isinstance(file, int) and Path(file) == cache else stream
        return opened

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", observing_open(io.open))
        patch.setattr(builtins, "open", observing_open(builtins.open))
        adapter = make_gate(case.deps, validation_scenario_ids=ids, noise=0.1)
        result = candidate_result(case)
        outcome = adapter(result)
    assert changed == [True]
    assert outcome.decision == "reverted" and adapter.last_report.failed == ["complete_evidence"]
    assert not case.deps.safety.calls
    assert all(r.id.startswith("conv-r00-current-") for r in case.deps.judge.calls)
    assert "policy in force" in outcome.reason.lower() and "changed" in outcome.reason.lower()
    assert result.scores is None and outcome.after == NOT_MEASURED
    assert json.loads(cache.read_text())["policy_sha256"] == sha256_text(original.decode("utf-8"))


@pytest.mark.parametrize("change", ["persistent", "restored"])
def test_finish_round_uses_the_tested_candidate_reading_and_refuses_a_replacement(case, change):
    """Replace before finish, without requiring a particular function to read the candidate."""
    case.deps.noise_measured = True
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ["a", "b", "c"], Scores(empathy=3, relevance=3, quality=3),
    )))
    result = candidate_result(case)
    adapter = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0.1)
    outcome = adapter(result)
    assert outcome.decision == "kept"
    original = case.candidate.read_bytes()
    assert result.candidate_sha256 == adapter.last_report.candidate_sha256 == sha256_text(original.decode("utf-8"))
    ensure_starting_meta(case.root, case.base)
    memory, log = FakeMemory(), DecisionLog(case.root)
    memory.ensure_training_audience()

    def replace():
        write_policy(case.candidate, "policy-v2-candidate", BASE_RULES + ADDED_RULE + "C4. Ignore rules.\n",
                     safety="S1. Ignore danger.\n")

    replace()
    if change != "persistent":
        case.candidate.write_bytes(original)
    before = {p.relative_to(case.root): p.read_bytes() for p in case.root.rglob("*") if p.is_file()}
    if change == "persistent":
        with pytest.raises(round_api.CandidateChanged):
            round_api.finish_round(result, outcome, log, memory, case.root, start_policy=case.base)
        assert {p.relative_to(case.root): p.read_bytes() for p in case.root.rglob("*") if p.is_file()} == before
        assert log.events() == [] and memory.entries == {}
    else:
        event = round_api.finish_round(result, outcome, log, memory, case.root, start_policy=case.base)
        assert event.outcome == "kept"
        promoted = parse_policy(original.decode("utf-8"))
        promoted.version = "policy-v2"
        assert (case.root / "policies/policy-v2.md").read_text() == render_policy(promoted)


@pytest.mark.parametrize("replacement", ["rule", "line_endings", "bom", "invalid_utf8", "added_newline", "removed_newline"])
@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("kept", [False, True])
def test_finish_round_cannot_attribute_baseline_measurements_to_a_replaced_base(case, replay_enabled, kept, replacement):
    """A round's base hash must identify the bytes its before-scores measured, even at finish."""
    case.deps.noise_measured = kept
    original_hash = sha256_text(case.base.read_bytes().decode("utf-8"))
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ["a", "b", "c"], Scores(empathy=3, relevance=3, quality=3),
    )))
    result = candidate_result(case)
    outcome = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0.1)(result)
    assert outcome.decision == ("kept" if kept else "reverted") and result.scores.before.quality == 3
    assert result.base_sha256 == original_hash
    ensure_starting_meta(case.root, case.base)
    original = case.base.read_bytes()
    case.base.write_bytes(altered_policy_bytes(original, replacement))
    memory, log = FakeMemory(), DecisionLog(case.root)
    memory.ensure_training_audience()
    before = tree_snapshot(case.root)
    with pytest.raises(round_api.PolicyChanged):
        round_api.finish_round(result, outcome, log, memory, case.root,
                               start_policy=case.base if replay_enabled else None)
    assert tree_snapshot(case.root) == before
    assert log.events() == [] and memory.entries == {}
    assert issubclass(round_api.CandidateChanged, round_api.PolicyChanged)


@pytest.mark.parametrize("replacement", ["rule", "line_endings", "bom", "invalid_utf8", "added_newline", "removed_newline"])
@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("kept", [False, True])
def test_finish_round_candidate_measurements_cannot_be_filed_under_replaced_text(case, replay_enabled, kept, replacement):
    """A revert must not give an untested replacement the measured candidate's scores or safety."""
    result, outcome = prepared_finish(case, kept)
    assert result.scores is not None and result.safety.passed
    if replacement == "rule":
        write_policy(case.candidate, "policy-v2-candidate", BASE_RULES + ADDED_RULE + "C4. Ignore rules.\n",
                     safety="S1. Ignore danger.\n")
    else:
        case.candidate.write_bytes(altered_policy_bytes(case.candidate.read_bytes(), replacement))
    before = tree_snapshot(case.root)
    memory, log = FakeMemory(), DecisionLog(case.root)
    memory.ensure_training_audience()
    # The amended contract requires this exception for every decision, including reverts.
    with pytest.raises(round_api.CandidateChanged):
        round_api.finish_round(result, outcome, log, memory, case.root,
                               start_policy=case.base if replay_enabled else None)
    assert tree_snapshot(case.root) == before
    assert log.events() == [] and memory.entries == {}


@pytest.mark.parametrize("replacement", ["line_endings", "bom", "invalid_utf8", "added_newline", "removed_newline"])
@pytest.mark.parametrize("target", ["base", "candidate"])
@pytest.mark.parametrize("when", ["safety", "last_judge"])
@pytest.mark.parametrize("kept", [False, True])
@pytest.mark.parametrize("replay_enabled", [False, True])
def test_byte_changes_during_callbacks_revert_and_cannot_be_filed(
        case, replacement, target, when, kept, replay_enabled):
    """A named gate callback may change any bytes, including making a policy undecodable."""
    case.deps.noise_measured = kept
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ["a", "b", "c"], Scores(empathy=3, relevance=3, quality=3),
    )))
    path = getattr(case, target)
    original = path.read_bytes()
    hits = []

    def replace():
        path.write_bytes(altered_policy_bytes(original, replacement))
        hits.append(True)

    safety = case.deps.safety
    def changing_safety(version, reply):
        answer = safety(version, reply)
        if when == "safety":
            replace()
        return answer
    case.deps.safety = changing_safety
    score = case.deps.judge.score
    def changing_judge(record):
        answer = score(record)
        if when == "last_judge" and record.scenario_id == "c":
            replace()
        return answer
    case.deps.judge.score = changing_judge
    result = candidate_result(case)
    if replay_enabled:
        ensure_starting_meta(case.root, case.base)
    adapter = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0.1)
    outcome = adapter(result)
    assert hits == [True] and outcome.decision == "reverted"
    assert "complete_evidence" in adapter.last_report.failed
    assert result.scores is None and outcome.after == NOT_MEASURED
    assert "changed" in outcome.reason.lower()
    assert adapter.last_report.checks["quality_gain"] == adapter.last_report.checks["dimension_drop"] == "not_run"
    before = tree_snapshot(case.root)
    log, memory = DecisionLog(case.root), FakeMemory()
    memory.ensure_training_audience()
    error = round_api.CandidateChanged if target == "candidate" else round_api.PolicyChanged
    with pytest.raises(error):
        round_api.finish_round(result, outcome, log, memory, case.root,
                               start_policy=case.base if replay_enabled else None)
    assert tree_snapshot(case.root) == before and not log.events() and not memory.entries


@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("broken", ["safety_section", "one_rule", "both"])
@pytest.mark.parametrize("replaced", [False, True])
def test_static_rejection_is_filed_without_measurements_and_still_guards_its_candidate(
        case, replay_enabled, broken, replaced):
    """Static failures must be logged; the measurement-only edit check must not refuse them."""
    rules = BASE_RULES + ADDED_RULE + ("C4. Give immediate advice.\n" if broken != "safety_section" else "")
    safety = "S1. Ignore danger.\n" if broken != "one_rule" else "S1. Keep the person safe.\n"
    write_policy(case.candidate, "policy-v2-candidate", rules, safety)
    result = candidate_result(case)
    for kind in ("pattern", "edit"):
        result.refs[kind] = save_record(getattr(result, kind), result.round, case.root)
    adapter = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0.1)
    outcome = adapter(result)
    failed = ["safety_section", "one_rule"] if broken == "both" else [broken]
    assert outcome.decision == "reverted" and adapter.last_report.failed == failed
    assert result.scores is result.safety is None and not case.deps.safety.calls
    assert all(record.id.startswith("conv-r00-current-") for record in case.deps.judge.calls)
    if replay_enabled:
        ensure_starting_meta(case.root, case.base)
    if replaced:
        write_policy(case.candidate, "policy-v2-candidate", BASE_RULES + ADDED_RULE)
    before, policies = tree_snapshot(case.root), policy_snapshot(case.root)
    log, memory = DecisionLog(case.root), FakeMemory()
    memory.ensure_training_audience()
    if replaced:
        with pytest.raises(round_api.CandidateChanged):
            round_api.finish_round(result, outcome, log, memory, case.root,
                                   start_policy=case.base if replay_enabled else None)
        assert tree_snapshot(case.root) == before and not log.events() and not memory.entries
    else:
        event = round_api.finish_round(result, outcome, log, memory, case.root,
                                       start_policy=case.base if replay_enabled else None)
        assert event.outcome == "reverted" and event.scores is event.safety is None
        assert all(CHECKS[name] in event.reason for name in failed)
        assert event.policy.new_sha256 == result.candidate_sha256
        assert policy_snapshot(case.root) == policies
        if replay_enabled:
            assert replay.check(log, case.root, case.base) == case.base


@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("kept", [False, True])
@pytest.mark.parametrize("filed", ["scores", "safety", "both"])
@pytest.mark.parametrize("wrong", ["rule_text", "rule_id", "action"])
def test_measurements_cannot_be_filed_under_an_edit_that_was_not_tested(case, replay_enabled, kept, filed, wrong):
    """Even matching saved records cannot make a different logged edit describe measured bytes."""
    result, outcome = prepared_finish(case, kept)
    changes = {"rule_text": "Give advice immediately.", "rule_id": "C4", "action": "reword"}
    result.edit = result.edit.model_copy(update={wrong: changes[wrong]})
    result.refs["edit"] = save_record(result.edit, result.round, case.root)
    if filed == "scores":
        result.safety = None
    elif filed == "safety":
        result.scores = None
    before = tree_snapshot(case.root)
    log, memory = DecisionLog(case.root), FakeMemory()
    memory.ensure_training_audience()
    with pytest.raises(round_api.PolicyChanged):
        round_api.finish_round(result, outcome, log, memory, case.root,
                               start_policy=case.base if replay_enabled else None)
    assert tree_snapshot(case.root) == before and not log.events() and not memory.entries


@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("kept", [False, True])
@pytest.mark.parametrize("kind", ["pattern", "edit"])
def test_hash_valid_saved_records_must_also_be_the_rounds_own(case, replay_enabled, kept, kind):
    """Hash verification alone must not let the log and memory describe different records."""
    result, outcome = prepared_finish(case, kept)
    change = {"name": "another pattern"} if kind == "pattern" else {"rule_text": "Give advice immediately."}
    other = getattr(result, kind).model_copy(update=change)
    result.refs[kind] = save_record(other, result.round, case.root)
    before = tree_snapshot(case.root)
    log, memory = DecisionLog(case.root), FakeMemory()
    memory.ensure_training_audience()
    with pytest.raises(round_api.RecordChanged):
        round_api.finish_round(result, outcome, log, memory, case.root,
                               start_policy=case.base if replay_enabled else None)
    assert tree_snapshot(case.root) == before and not log.events() and not memory.entries


@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("kept", [False, True])
@pytest.mark.parametrize("kind", ["pattern", "edit"])
@pytest.mark.parametrize("missing", [False, True])
def test_finish_round_refuses_changed_or_missing_records_before_any_write(case, replay_enabled, kept, kind, missing):
    """A changed saved record cannot be committed or copied into memory."""
    result, outcome = prepared_finish(case, kept)
    target = case.root / result.refs[kind].path
    if missing:
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b" ")
    before = tree_snapshot(case.root)
    memory, log = FakeMemory(), DecisionLog(case.root)
    memory.ensure_training_audience()
    with pytest.raises(round_api.RecordChanged):
        round_api.finish_round(result, outcome, log, memory, case.root,
                               start_policy=case.base if replay_enabled else None)
    assert tree_snapshot(case.root) == before
    assert log.events() == [] and memory.entries == {}


@pytest.mark.parametrize("replacement", ["rule", "line_endings", "bom", "invalid_utf8", "added_newline", "removed_newline"])
@pytest.mark.parametrize("kept", [False, True])
def test_finish_round_refuses_a_replaced_independent_starting_policy_before_any_write(case, kept, replacement):
    """With replay enabled, a changed starting policy must be rejected before commitment."""
    result, outcome = prepared_finish(case, kept)
    start = case.root / "start.md"
    start.write_bytes(case.base.read_bytes())
    start.write_bytes(altered_policy_bytes(start.read_bytes(), replacement))
    before = tree_snapshot(case.root)
    memory, log = FakeMemory(), DecisionLog(case.root)
    memory.ensure_training_audience()
    # No measured starting hash is carried; malformed bytes or a broken replay chain both refuse before writing.
    with pytest.raises((PolicyError, replay.ReplayError)):
        round_api.finish_round(result, outcome, log, memory, case.root, start_policy=start)
    assert tree_snapshot(case.root) == before
    assert log.events() == [] and memory.entries == {}


@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("mode", [
    "real_kept", "real_reverted", "stand_in_kept", "stand_in_reverted", "no_pattern", "no_evidence",
])
def test_finish_round_preserves_outcomes_copy_order_and_exact_replay_metadata(case, replay_enabled, mode):
    """All four outcomes retain their artifacts and copy records only after the round event."""
    if mode.startswith("real_"):
        result, outcome = prepared_finish(case, mode == "real_kept")
    elif mode.startswith("stand_in_"):
        result = candidate_result(case)
        assert result.base_sha256 is result.candidate_sha256 is None
        for kind in ("pattern", "edit"):
            result.refs[kind] = save_record(getattr(result, kind), result.round, case.root)
        outcome = round_api.stand_in_gate(result, "kept" if mode == "stand_in_kept" else "reverted")
    else:
        result = candidate_result(case)
        result.outcome, result.edit, result.candidate_path = mode, None, None
        if mode == "no_pattern":
            result.pattern = None
        else:
            result.refs["pattern"] = save_record(result.pattern, result.round, case.root)
        outcome = None
    if replay_enabled:
        ensure_starting_meta(case.root, case.base)
    policies_before = policy_snapshot(case.root)
    log = DecisionLog(case.root)

    class ObservingMemory(FakeMemory):
        def ingest(self, entry):
            rounds = log.rounds()
            assert len(rounds) == 1 and rounds[0].round == result.round
            assert entry.id in {ref.id for ref in rounds[0].records.values()}
            return super().ingest(entry)

    memory = ObservingMemory()
    memory.ensure_training_audience()
    event = round_api.finish_round(result, outcome, log, memory, case.root,
                                   start_policy=case.base if replay_enabled else None)
    expected = outcome.decision if outcome is not None else mode
    assert event.outcome == expected and log.rounds() == [event]
    assert set(memory.entries) == {ref.id for ref in event.records.values()}
    if not mode.startswith("real_"):
        assert event.scores is event.safety is None
    if expected != "kept":
        assert policy_snapshot(case.root) == policies_before
    else:
        assert (case.root / "policies/policy-v2.md").is_file()
        meta_path = case.root / "policies/policy-v2.meta.json"
        assert meta_path.exists() is replay_enabled
        if replay_enabled:
            written = meta_path.read_bytes()
            replay.write_meta(case.root, log.rounds(), case.base, "policy-v2")
            assert meta_path.read_bytes() == written
    if replay_enabled:
        assert replay.check(log, case.root, case.base).is_file()


@pytest.mark.parametrize("replay_enabled", [False, True])
def test_finish_round_refuses_a_duplicate_before_rewriting_any_file(case, replay_enabled):
    """A repeated finish must not overwrite a policy or record before discovering the duplicate."""
    result, outcome = prepared_finish(case, True)
    memory, log = FakeMemory(), DecisionLog(case.root)
    memory.ensure_training_audience()
    round_api.finish_round(result, outcome, log, memory, case.root,
                           start_policy=case.base if replay_enabled else None)
    before, entries, events = tree_snapshot(case.root), deepcopy(memory.entries), log.events()
    with pytest.raises(round_api.DecisionLogError):
        round_api.finish_round(result, outcome, log, memory, case.root,
                               start_policy=case.base if replay_enabled else None)
    assert tree_snapshot(case.root) == before and log.events() == events and memory.entries == entries


def test_importing_gate_never_imports_the_evaluation_package(repo_root):
    """Exercise a clean import so prior test collection cannot conceal an eval dependency."""
    code = '''
import builtins
import socket
socket.socket.connect = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network forbidden"))
original_import = builtins.__import__
attempts = []
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "puddles.eval" or name.startswith("puddles.eval.") or (name == "puddles" and "eval" in (fromlist or ())):
        attempts.append(name)
        raise AssertionError("gate imported evaluation package")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
import puddles.gate
assert not attempts, attempts
'''
    completed = subprocess.run(
        ["env", "-u", "PYTHONPATH", sys.executable, "-c", code], cwd=repo_root,
        env={**os.environ, "DO_NOT_TRACK": "true", "MLFLOW_DISABLE_TELEMETRY": "true"},
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("measured", [False, True])
def test_make_gate_attaches_measurements_and_updates_cache_only_for_kept_candidate(case, measured):
    case.deps.noise_measured = measured
    # The incumbent baseline scores 3; the candidate scores 4.
    class ComparingJudge(Judge):
        def score(self, record):
            self.values = (3, 3, 4, 2) if record.policy_version == "policy-v1" else (4, 4, 4, 1)
            return super().score(record)
    case.deps.judge = ComparingJudge()
    # Raw renamed text and rendered promoted text must have different hashes.
    case.candidate.write_text(case.candidate.read_text().replace("\n\n## Conversation", "\n\n\n## Conversation"))
    result = candidate_result(case)
    assert result.scores is result.safety is None
    adapter = make_gate(case.deps, validation_scenario_ids=["c", "a"], noise=0.2,
                        token_cap=800, safety_case_set="scripted-safety")
    before = policy_snapshot(case.root)
    outcome = adapter(result)
    report = adapter.last_report
    assert outcome == report.outcome and report.would_keep
    assert outcome.round == result.round and outcome.edit_id == result.edit.id
    assert outcome.decision == ("kept" if measured else "reverted")
    assert result.scores == report.log_scores and result.safety == report.log_safety
    assert result.scores.noise == 0.2 and result.scores.before.quality == 3 and result.scores.after.quality == 4
    assert result.safety.passed is True
    cached = json.loads((case.root / "scores/current.json").read_text())
    promoted = parse_policy(case.candidate.read_text())
    promoted.version = "policy-v2"
    promoted_text = render_policy(promoted)
    assert promoted_text != case.candidate.read_text().replace("policy-v2-candidate", "policy-v2", 1)
    expected_cache = cache_payload(case.base, case.deps, ["c", "a"], outcome.after if measured else outcome.before)
    if measured:
        expected_cache.update(policy_version="policy-v2", policy_sha256=sha256_text(promoted_text))
    assert cached == expected_cache
    assert [r.id for r in case.deps.judge.calls] == [
        "conv-r00-current-c", "conv-r00-current-a", "conv-r03-val-c", "conv-r03-val-a",
    ]
    assert policy_snapshot(case.root) == before
    assert_report(report, case.root)


@pytest.mark.parametrize("cache_state", ["stale", "missing"])
def test_failed_post_keep_cache_write_keeps_outcome_and_the_promoted_policy_is_rescored(case, cache_state):
    case.deps.noise_measured = True
    ids = ["a", "b", "c"]
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ids, Scores(empathy=3, relevance=3, quality=3),
    )))
    original_cache = cache.read_bytes()
    blocked = []

    def block_cache_write():
        if not blocked:
            # The baseline is cached and Safety does not call reply, so the first
            # helper call is validation, after the baseline cache has been read.
            cache.unlink()
            cache.mkdir()
            (cache / "blocked").write_text("Only the promoted-cache destination is unavailable.\n")
            blocked.append(True)

    case.deps.helper_client = Chat(before_call=block_cache_write)
    result = candidate_result(case)
    adapter = make_gate(case.deps, validation_scenario_ids=ids, noise=0.1)
    before = policy_snapshot(case.root)
    outcome = adapter(result)
    assert blocked == [True]
    assert outcome.decision == "kept" and outcome.policy_in_force == "policy-v2"
    assert outcome.before.quality == 3 and outcome.after.quality == 4
    assert adapter.last_report.outcome == outcome and adapter.last_report.failed == []
    assert adapter.last_report.would_keep is True
    assert result.scores == adapter.last_report.log_scores and result.scores is not None
    assert result.safety == adapter.last_report.log_safety and result.safety is not None
    assert policy_snapshot(case.root) == before
    assert_report(adapter.last_report, case.root)

    memory, log = FakeMemory(), DecisionLog(case.root)
    memory.ensure_training_audience()
    event = round_api.finish_round(result, outcome, log, memory, case.root)
    assert event.outcome == "kept"
    promoted = case.root / "policies/policy-v2.md"
    assert promoted.is_file()

    # Restore storage in each allowed state; neither can match the promoted policy.
    case.deps.helper_client.before_call = None
    if cache.is_dir():
        cache.rename(case.root / "blocked-cache-destination")
    else:
        cache.unlink(missing_ok=True)
    if cache_state == "stale":
        cache.write_bytes(original_cache)
    counts = (len(case.deps.helper_client.calls), len(case.deps.user_client.calls), len(case.deps.judge.calls))
    got = current_scores(case.deps, promoted, ids)
    assert got == outcome.after
    assert len(case.deps.helper_client.calls) > counts[0] and len(case.deps.user_client.calls) > counts[1]
    rescored = case.deps.judge.calls[counts[2]:]
    assert [r.id for r in rescored] == [runner.conversation_id(0, s, CURRENT_TAG) for s in ids]
    assert all(r.policy_version == "policy-v2" for r in rescored)
    assert json.loads(cache.read_text()) == cache_payload(promoted, case.deps, ids, got)


@pytest.mark.parametrize("measured", [False, True])
def test_make_gate_baseline_failure_reverts_untested_and_preserves_cache_and_policies(case, measured):
    case.deps.noise_measured = measured
    case.deps.judge = Judge(errors=["b"])
    result = candidate_result(case)
    result.base_sha256 = "stale from a previous call"
    adapter = make_gate(case.deps, validation_scenario_ids=["a", "b", "c"], noise=0)
    before = policy_snapshot(case.root)
    outcome = adapter(result)
    report = adapter.last_report
    assert outcome.decision == "reverted" and not report.would_keep
    assert report.failed == ["complete_evidence"] + ([] if measured else ["noise_not_measured"])
    assert outcome.before == outcome.after == NOT_MEASURED
    assert result.scores is result.safety is report.log_scores is report.log_safety is None
    assert result.base_sha256 is None
    assert not case.deps.safety.calls and outcome.safety_passed is False
    assert all(r.policy_version == "policy-v1" for r in case.deps.judge.calls)
    assert_baseline_unscored_reason(outcome.reason)
    assert not (case.root / "scores/current.json").exists()
    assert policy_snapshot(case.root) == before


def test_make_gate_catches_unexpected_baseline_exception(case):
    case.deps.user_client = Chat([RuntimeError("baseline exploded")])
    adapter = make_gate(case.deps, validation_scenario_ids=["a"], noise=0)
    outcome = adapter(candidate_result(case))
    assert outcome.decision == "reverted"
    assert "complete_evidence" in adapter.last_report.failed
    assert outcome.before == outcome.after == NOT_MEASURED
    assert not case.deps.safety.calls and not case.deps.helper_client.calls
    assert len(case.deps.user_client.calls) == 1
    assert_baseline_unscored_reason(outcome.reason)


@pytest.mark.parametrize("mode", ["static", "partial"])
def test_make_gate_replaces_stale_measurements_with_what_was_actually_measured(case, mode):
    cache = case.root / "scores/current.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(cache_payload(
        case.base, case.deps, ["a", "b", "c"], Scores(empathy=3, relevance=3, quality=3),
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
    assert meta.policy_sha256 == sha256_text(case.base.read_text())
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
        cached = json.loads((case.root / "scores/current.json").read_text())
        assert cached["policy_sha256"] == sha256_text(promoted.read_text())
        counts = (len(case.deps.helper_client.calls), len(case.deps.user_client.calls), len(case.deps.judge.calls))
        assert current_scores(case.deps, promoted, ["a", "b", "c"]) == event.scores.after
        assert counts == (len(case.deps.helper_client.calls), len(case.deps.user_client.calls), len(case.deps.judge.calls))


@pytest.mark.usefixtures("local_mlflow")
def test_starting_meta_is_after_price_check_and_before_graph(case):
    meta_path = case.root / "policies/policy-v1.meta.json"
    prices = PriceProbe(meta_path)

    def before_conversation():
        assert prices.calls == 2
        assert PolicyMeta.model_validate_json(meta_path.read_text()).policy_version == "policy-v1"

    deps = graph.RoundDeps(
        helper_client=Chat(), user_client=Chat("Worry.", before_call=before_conversation), llm=Agents(),
        search=lambda q: [], memory=FakeMemory(), log=DecisionLog(case.root), root=case.root,
        start_policy=case.base, judge=lambda records: [], usage=Meter(case.root, prices),
    )
    with pytest.raises(MissingPrice):
        graph.run_round(deps, policy_path=case.base, scenarios=case.deps.scenarios[:1], max_turns=1)
    assert prices.calls == 1 and not meta_path.exists()
    assert not deps.helper_client.calls and not deps.user_client.calls
    assert deps.log.rounds() == []
    prices.reject = False
    graph.run_round(deps, policy_path=case.base, scenarios=case.deps.scenarios[:1], max_turns=1)
    assert prices.calls == 2 and len(deps.user_client.calls) == 1


@pytest.mark.usefixtures("local_mlflow")
def test_absent_start_policy_preserves_old_graph_behavior(case):
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
    assert event.reason.strip()
    report = json.loads((root / "scores/round-01-gate.json").read_text())
    assert report["failed"][-1] == "noise_not_measured"
    assert event.scores is not None and event.scores.noise == 0.0
    assert event.safety is not None and event.safety.passed
    expected = {"b", "c"} if validation_file else {"a"}
    assert {r["scenario_id"] for r in rows(root / "conversations/round-01-validation.jsonl")} == expected
    assert (root / "policies/policy-v1.meta.json").is_file()
    assert not (root / "policies/policy-v2.md").exists()


@pytest.mark.usefixtures("local_mlflow")
@pytest.mark.parametrize("mode,token_cap", [("real", 777), ("real", 1), ("kept", 777), ("reverted", 777)])
def test_cli_forwards_start_policy_for_every_gate_mode_and_accepts_real_gate_flags(case, mode, token_cap):
    scenarios = case.root / "cli-scenarios.jsonl"
    scenarios.write_text(case.deps.scenarios[0].model_dump_json() + "\n")
    root = case.root / "flags"
    assert script.main([
        "--fake", "--gate", mode, "--root", str(root), "--policy", str(case.base),
        "--scenarios", str(scenarios), "--validation", str(scenarios),
        "--noise", "0.25", "--noise-measured", "--token-cap", str(token_cap), "--max-turns", "1",
        "--mlflow", f"sqlite:///{case.root}/mlflow.db",
    ]) == 0
    meta = PolicyMeta.model_validate_json((root / "policies/policy-v1.meta.json").read_text())
    assert meta.policy_sha256 == sha256_text(case.base.read_text())
    event = DecisionLog(root).rounds()[0]
    assert replay.check(DecisionLog(root), root, case.base).is_file()
    if mode == "real":
        report = json.loads((root / "scores/round-01-gate.json").read_text())
        if token_cap == 1:
            assert report["failed"] == ["token_cap"] and report["would_keep"] is False
            assert report["checks"]["noise_not_measured"] == "not_run"
            assert event.scores is event.safety is None and event.outcome == "reverted"
        else:
            assert report["checks"]["token_cap"] == report["checks"]["noise_not_measured"] == "passed"
            assert event.scores is not None and event.scores.noise == 0.25
    else:
        assert event.outcome == mode
        assert (root / "policies/policy-v2.meta.json").exists() is (mode == "kept")


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
    assert captured.err.strip()
    assert {p.relative_to(case.root): p.read_bytes() for p in case.root.rglob("*") if p.is_file()} == before


# Section 8: gaps left by the existing acceptance and builder tests.

@pytest.mark.parametrize("bad", ["cr", "no_safety", "duplicate", "utf8"])
@pytest.mark.parametrize("replaced", [False, True])
@pytest.mark.parametrize("replay_enabled", [False, True])
def test_section8_unloadable_candidate_stays_bound_through_finish(case, bad, replaced, replay_enabled):
    clean = case.candidate.read_bytes()
    stated = clean.replace(b"policy-v2-candidate", b"policy-v7-candidate")
    data = {
        "cr": stated.replace(b"\n", b"\r\n"),
        "no_safety": stated.replace(b"## Safety", b"## Notes"),
        "duplicate": stated + b"C3. Duplicate ID.\n",
        "utf8": b"\xff" + stated,
    }[bad]
    case.candidate.write_bytes(data)
    result = candidate_result(case)
    keeper = make_gate(case.deps, validation_scenario_ids=["a"], noise=0)
    outcome = keeper(result)
    digest = sha256_bytes(data)
    assert keeper.last_report.failed == ["candidate_file"]
    assert result.candidate_sha256 == digest
    assert result.scores is result.safety is None
    if replay_enabled:
        ensure_starting_meta(case.root, case.base)
    if replaced:
        case.candidate.write_bytes(clean)
    before, memory, log = tree_snapshot(case.root), FakeMemory(), DecisionLog(case.root)
    if replaced:
        with pytest.raises(round_api.CandidateChanged):
            round_api.finish_round(result, outcome, log, memory, case.root,
                                   start_policy=case.base if replay_enabled else None)
        assert tree_snapshot(case.root) == before and not log.events() and not memory.entries
    else:
        event = round_api.finish_round(result, outcome, log, memory, case.root,
                                       start_policy=case.base if replay_enabled else None)
        assert event.outcome == "reverted" and event.policy.new_sha256 == digest
        assert event.policy.new == ("policy-v2-candidate" if bad in ("no_safety", "utf8")
                                    else "policy-v7-candidate")
        assert event.scores is event.safety is None and log.rounds() == [event]
        assert case.candidate.read_bytes() == data


@pytest.mark.parametrize("baseline", [False, True])
def test_section8_judge_cannot_mutate_owned_batch_records(case, monkeypatch, baseline):
    from puddles import gate as gate_api

    batches = []
    real_batch = gate_api.run_batch

    def observe_batch(*args, **kwargs):
        batch = real_batch(*args, **kwargs)
        batches.append((batch, [r.model_dump() for r in batch.records]))
        return batch

    class MutatingJudge(Judge):
        def score(self, record):
            answer = super().score(record)
            record.turns[0].text = "Changed through a nested reference."
            record.id = "changed-after-scoring"
            return answer

    monkeypatch.setattr(gate_api, "run_batch", observe_batch)
    case.deps.judge = MutatingJudge()
    if baseline:
        assert current_scores(case.deps, case.base, ["a", "b"]).quality == 4
    else:
        assert run_gate(case.inp, case.deps).checks["complete_evidence"] == "passed"
    assert batches
    for batch, originals in batches:
        assert [r.model_dump() for r in batch.records] == originals


@pytest.mark.parametrize("bad_batch", ["duplicate", "missing"])
def test_section8_baseline_requires_each_expected_conversation(case, monkeypatch, bad_batch):
    from puddles import gate as gate_api

    real_batch = gate_api.run_batch

    def incomplete_batch(*args, **kwargs):
        batch = real_batch(*args, **kwargs)
        batch.records = [batch.records[0]] * len(batch.records) if bad_batch == "duplicate" else batch.records[:-1]
        return batch

    monkeypatch.setattr(gate_api, "run_batch", incomplete_batch)
    with pytest.raises(GateError):
        current_scores(case.deps, case.base, ["a", "b", "c"])
    assert not (case.root / "scores/current.json").exists()


@pytest.mark.parametrize("field", ["empathy", "relevance", "quality"])
@pytest.mark.parametrize("bad", [None, 0, 6, float("nan"), float("inf"), float("-inf"), True, "missing"])
def test_section8_every_cached_dimension_must_be_measured(case, field, bad):
    # Rule 4 applies independently to all three fields, including finite relevance and bounded quality.
    path = case.root / "scores/current.json"
    path.parent.mkdir()
    payload = cache_payload(case.base, case.deps, ["a"], Scores(empathy=3, relevance=3, quality=3))
    if bad == "missing":
        del payload["scores"][field]
    else:
        payload["scores"][field] = bad
    path.write_text(json.dumps(payload))
    assert current_scores(case.deps, case.base, ["a"]).quality == 4
    assert len(case.deps.judge.calls) == 1


@pytest.mark.parametrize("baseline", [False, True])
def test_section8_cache_context_precedes_the_scoring_run(case, monkeypatch, baseline):
    from puddles import gate as gate_api

    ids = ["a", "b", "c"]
    expected = cache_payload(case.base, case.deps, ids, Scores(empathy=3, relevance=3, quality=3))
    path = case.root / "scores/current.json"
    if not baseline:
        path.parent.mkdir()
        path.write_text(json.dumps(expected))
        case.deps.noise_measured = True
    changed = []
    run_settings = []
    real_batch = gate_api.run_batch

    def observe_run(*args, **kwargs):
        run_settings.append((kwargs["seed"], kwargs["max_turns"]))
        return real_batch(*args, **kwargs)

    monkeypatch.setattr(gate_api, "run_batch", observe_run)

    class ChangesDeps(Judge):
        def score(self, record):
            answer = super().score(record)
            if not changed:
                case.deps.seed, case.deps.max_turns = 99, 8
                for scenario in case.deps.scenarios:
                    scenario.situation += " Changed after the run started."
                changed.append(True)
            return answer

    case.deps.judge = ChangesDeps()
    if baseline:
        assert current_scores(case.deps, case.base, ids).quality == 4
    else:
        assert make_gate(case.deps, validation_scenario_ids=ids, noise=0)(candidate_result(case)).decision == "kept"
    saved = json.loads(path.read_text())
    assert changed == [True]
    for name in ("judge_version", "scenarios_sha256", "max_turns", "seed"):
        assert saved[name] == expected[name]
    # The batch seed is the cached identity; the runner derives individual call seeds from it.
    assert run_settings == [(expected["seed"], expected["max_turns"])]


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_section8_baseline_interrupt_propagates_without_cache_or_policy_write(case, interrupt_type):
    class Interrupted(Judge):
        def score(self, record):
            raise interrupt_type("baseline stopped")

    case.deps.judge = Interrupted()
    before = policy_snapshot(case.root)
    with pytest.raises(interrupt_type):
        make_gate(case.deps, validation_scenario_ids=["a"], noise=0)(candidate_result(case))
    assert policy_snapshot(case.root) == before and not (case.root / "scores/current.json").exists()


@pytest.mark.parametrize("source", ["base", "start", "logged_base", "logged_in_force"])
def test_section8_promotion_cannot_reuse_any_protected_version(case, source):
    from schemas import LogPolicy, RoundEvent

    version = "policy-v1" if source == "base" else "policy-v9"
    case.candidate.write_bytes(case.candidate.read_bytes().replace(b"policy-v2-candidate",
                                                                 (version + "-candidate").encode()))
    case.deps.noise_measured = True
    result = candidate_result(case)
    result.edit.new_policy_version = version + "-candidate"
    path = case.root / "scores/current.json"
    path.parent.mkdir()
    path.write_text(json.dumps(cache_payload(case.base, case.deps, ["a"], Scores(empathy=3, relevance=3, quality=3))))
    outcome = make_gate(case.deps, validation_scenario_ids=["a"], noise=0)(result)
    assert outcome.decision == "kept"
    log, memory, start = DecisionLog(case.root), FakeMemory(), None
    if source == "start":
        start = write_policy(case.root / "start.md", version=version)
    elif source.startswith("logged_"):
        base, new = (version, "policy-v8") if source == "logged_base" else ("policy-v8", version)
        log.append(RoundEvent(round=1, outcome="kept", reason="Earlier promotion.",
                              policy=LogPolicy(base=base, base_sha256=sha256_bytes(case.base.read_bytes()),
                                               new=new, in_force=new)))
    before, events = tree_snapshot(case.root), log.events()
    with pytest.raises(round_api.PolicyChanged, match="already"):
        round_api.finish_round(result, outcome, log, memory, case.root, start_policy=start)
    assert tree_snapshot(case.root) == before and log.events() == events and not memory.entries


@pytest.mark.parametrize("changed", ["safety", "rule_order"])
def test_section8_rendering_must_preserve_strict_safety_and_rule_order(case, monkeypatch, changed):
    result, outcome = prepared_finish(case, True)
    real_render = round_api.render_policy

    def different_render(policy):
        text = real_render(policy)
        if changed == "safety":
            return text.replace("Keep the person safe.", "Ignore danger.")
        return text.replace(BASE_RULES, "C2. Keep replies short.\nC1. Listen carefully.\n")

    monkeypatch.setattr(round_api, "render_policy", different_render)
    before, memory, log = tree_snapshot(case.root), FakeMemory(), DecisionLog(case.root)
    active, writes = [True], []

    def observe_write(event, args):
        if not active[0] or event not in ("open", "os.mkdir", "os.remove", "os.rename"):
            return
        path = args[0]
        if not isinstance(path, (str, bytes)):
            return
        if not Path(os.fsdecode(path)).resolve().is_relative_to(case.root.resolve()):
            return
        flags = args[2] if event == "open" else 0
        if event != "open" or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            writes.append((event, path))

    # A final tree comparison alone misses a probe file created and deleted inside the root.
    sys.addaudithook(observe_write)
    try:
        with pytest.raises(round_api.PolicyChanged, match="tested"):
            round_api.finish_round(result, outcome, log, memory, case.root)
    finally:
        active[0] = False
    assert writes == []
    assert tree_snapshot(case.root) == before and not log.events() and not memory.entries


@pytest.mark.parametrize("field", ["round", "edit_id", "before"])
def test_section8_reverted_identity_and_before_scores_must_agree(case, field):
    result, outcome = prepared_finish(case, False)
    value = {"round": 99, "edit_id": "edit-r99", "before": Scores(empathy=1, relevance=1, quality=1)}[field]
    outcome = outcome.model_copy(update={field: value}, deep=True)
    before, memory, log = tree_snapshot(case.root), FakeMemory(), DecisionLog(case.root)
    with pytest.raises(round_api.InconsistentResult):
        round_api.finish_round(result, outcome, log, memory, case.root)
    assert tree_snapshot(case.root) == before and not log.events() and not memory.entries
