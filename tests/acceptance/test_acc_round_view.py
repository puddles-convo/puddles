"""Acceptance tests from docs/specs/round-view.md, without the view implementation.

The existing schemas and training APIs supply files, never the expected report.
Markdown assertions allow layout choices the contract leaves open.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import socket

import mlflow
from mlflow import MlflowClient
import pytest

from loop.llm import CallUsage
from loop.runner import failures_path
from puddles import graph, tracing
from puddles.decision_log import DecisionLog
from puddles.round import RoundResult, memory_trace, stand_in_gate
from puddles.records import save_record, sha256_text
from puddles.round_view import SUMMARY_KEY, build_report, node_summary, report_path, with_summary
from puddles.usage import Meter, ModelPrice, Prices
from schemas import (
    ConversationRecord, CriticOutput, FailurePattern, GateOutcome, LogEdit,
    LogPattern, LogPolicy, LogSafety, LogScores, MemoryCopyEvent, PolicyEdit,
    ProposerOutput, ResearcherOutput, ResearchNote, RoundEvent, ScoreRecord,
    ScoredConversation, Scores, Turn,
)
from scripts import show_round


NODES = ("converse", "judge", "critic", "researcher", "proposer", "gate", "finish")
CLIENT_LINE = "PRIVATE-CLIENT-7289: I cannot tell my family about the lost job."
HELPER_LINE = "PRIVATE-HELPER-8451: What worries you most about telling them?"
URL = "https://evidence.example/listening"
QUERY = "listening before offering advice"
NEW_RULE = "Ask what happened before offering advice."
OLD_RULE = "Listen before suggesting."
STAMP = datetime(2026, 1, 2, tzinfo=timezone.utc)
REQUIRED_KEYS = {
    "converse": {"finished", "failed", "file"},
    "judge": {"scored", "unscored", "mean_empathy", "mean_relevance", "judge_version"},
    "critic": {"pattern", "seen_in", "out_of"},
    "researcher": {"queries", "sources"},
    "proposer": {"action", "rule_id", "rule_text"},
    "gate": {"decision", "quality_before", "quality_after", "noise", "safety_passed", "failed_checks"},
    "finish": {"outcome", "policy_in_force", "copies"},
}


@pytest.fixture(autouse=True)
def offline_mlflow(tmp_path, monkeypatch):
    """All generated artifacts, SQLite files and MLflow defaults stay outside the repo."""
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.endswith("API_KEY") or key in {"MLFLOW_TRACKING_URI", "MLFLOW_TRACING_DESTINATION"}:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("Round-view acceptance tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()


# Reason: attempted paid/model/memory calls must fail even if product code swallows ordinary exceptions.
@pytest.fixture(autouse=True)
def no_external_training_services(monkeypatch):
    from loop.llm import OpenAICompatClient
    from puddles import exa
    from puddles.llm import OpenAILLM
    from puddles.memory import PastClient

    def forbidden(*args, **kwargs):
        pytest.fail("The view must not call a model, Exa, or past.dev")

    monkeypatch.setattr(exa, "search", forbidden)
    for cls in (OpenAICompatClient, OpenAILLM, PastClient):
        monkeypatch.setattr(cls, "__init__", forbidden)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        (row.model_dump_json() if hasattr(row, "model_dump_json") else json.dumps(row)) + "\n"
        for row in rows
    ), encoding="utf-8")


def policy_file(root):
    path = root / "policies" / "policy-v1.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# policy-v1\n\n## Safety\nS1. Keep the person safe.\n\n"
        f"## Conversation\nC1. {OLD_RULE}\n", encoding="utf-8",
    )
    return path


def conversation(round_, suffix):
    return ConversationRecord(
        id=f"conv-r{round_:02d}-{suffix}", written_by="helper", created_at=STAMP,
        source="simulated", split="train", scenario_id=suffix, policy_version="policy-v1",
        turns=[Turn(speaker="user", text=CLIENT_LINE), Turn(speaker="helper", text=HELPER_LINE)],
    )


def score(record, empathy=2, relevance=4, version="judge-acceptance-v1"):
    return ScoreRecord(
        id=f"score-{record.id}", written_by="judge", created_at=STAMP,
        parent_ids=[record.id], conversation_id=record.id, judge_version=version,
        empathy=empathy, relevance=relevance, emotion_before=4, emotion_after=3,
        strategies=["Question"], note="Advice arrived early.",
    )


def scored(record, empathy=2, relevance=4):
    return ScoredConversation(
        conversation_id=record.id, empathy=empathy, relevance=relevance,
        note="Advice arrived early.", turns=list(record.turns),
    )


def snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def replace_logged_round(root, event):
    events = DecisionLog(root).events()
    write_jsonl(DecisionLog(root).path, [
        event if isinstance(item, RoundEvent) and item.round == event.round else item for item in events
    ])


def section(report, name):
    heading = f"## {name}"
    assert heading in report, f"Missing report section: {name}"
    return report.split(heading, 1)[1].split("\n## ", 1)[0]


def table_rows(text):
    """Only parse the pipe delimiters the contract's Markdown tables require."""
    return [
        [cell.strip().strip("`").strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines() if "|" in line
    ]


def row_for(text, identifier):
    rows = table_rows(text)
    if identifier in NODES:
        # Infer the label column: other columns can also name upstream/downstream nodes.
        width = max((len(row) for row in rows), default=0)
        assert width, "Missing node table"
        column = max(range(width), key=lambda i: len({
            row[i].strip("*` ") for row in rows if len(row) > i
        } & set(NODES)))
        rows = [row for row in rows if len(row) > column and row[column].strip("*` ") == identifier]
    else:
        rows = [row for row in rows if any(identifier in cell for cell in row)]
    assert len(rows) == 1, f"Expected one table row for {identifier}: {rows}"
    return rows[0]


def has_number(text, value):
    # Reason: the contract fixes only dollar precision; allow decimal rounding and sentence punctuation.
    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)", text)
    for number in numbers:
        places = len(number.split(".", 1)[1]) if "." in number else 0
        if float(number) == value:
            return
        if places and float(number) != 0 and abs(float(number) - value) <= 0.5 * 10 ** -places + 1e-12:
            return
    pytest.fail(f"Missing numeric value {value!r}: {text}")


class Chat:
    def __init__(self, line):
        self.line = line
        self.calls = []

    def complete(self, system, messages, *, max_tokens, temperature=0.7, seed=None):
        self.calls.append((system, messages))
        return self.line


class Agents:
    """Schema-specific answers, including two invalid proposals for an early exit."""
    def __init__(self, mode="candidate"):
        self.mode = mode
        self.calls = []

    def structured(self, system, user, schema):
        self.calls.append(schema)
        if schema is CriticOutput:
            if self.mode == "critic_error":
                return schema(found=False, reason="")
            if self.mode == "no_pattern":
                return schema(found=False, reason="No repeated failure.")
            return schema(
                found=True, name="early advice", description="Advice before understanding.",
                seen_in=1, out_of=1, example_conversation_ids=["conv-r01-a"],
            )
        if schema is ProposerOutput:
            if self.mode == "proposer_error":
                return schema(action="remove", rule_id="S1", summary="Invalid safety edit.")
            return schema(action="add", rule_text=NEW_RULE, summary="Understand before advising.")
        raise AssertionError(f"Unexpected model schema {schema}")

    def structured_with_search(self, system, user, schema, search, max_searches):
        assert schema is ResearcherOutput
        self.calls.append(schema)
        search(QUERY)
        if self.mode == "researcher_error":
            return schema(queries=[], sources=[])
        return schema(queries=[QUERY], sources=[] if self.mode == "no_evidence" else [
            {"url": URL, "title": "Listening", "technique": "Ask before advising."},
        ])


class Memory:
    def __init__(self):
        self.copies = []
        self.reads = []

    def ensure_training_audience(self):
        pass

    def ingest(self, entry):
        self.copies.append(entry)
        return entry.id

    def wait(self, ingestion_id, timeout_s=0.01):
        return True

    def recall(self, query, identity, level="low", limit=20):
        self.reads.append((query, identity))
        return []


class Judge:
    def __init__(self, empty=False):
        self.empty = empty
        self.calls = []

    def __call__(self, records):
        self.calls.append(records)
        return [] if self.empty else [scored(record) for record in records]


@pytest.fixture
def training_round(tmp_path, scenario_factory):
    def run(*, mode="candidate", decision="reverted", measured=False):
        root = tmp_path / "training"
        policy = policy_file(root)
        agents, memory = Agents(mode), Memory()
        searches = []

        def search(query):
            searches.append(query)
            return [{"url": URL, "title": "Listening", "text": "Ask before advising."}]

        def gate(result):
            outcome = stand_in_gate(result, decision)
            if measured:
                result.scores = LogScores(
                    before=Scores(empathy=2, relevance=3, quality=2.5),
                    after=Scores(empathy=4, relevance=4.5, quality=4.25), noise=0.125,
                )
                result.safety = LogSafety(case_file="offline-cases", passed=True, cases=1)
                # Deliberately distinct: gate-summary metrics must come from result.scores.
                outcome.before = Scores(empathy=1, relevance=1, quality=1)
                outcome.after = Scores(empathy=5, relevance=5, quality=5)
                outcome.safety_passed = True
                write_json(root / "scores/round-01-gate.json", {
                    "round": 1, "decision": decision, "would_keep": True,
                    "failed": [] if decision == "kept" else ["quality_gain"],
                    # Amendment: noise is measured only with an explicit passed marker.
                    "checks": {"quality_gain": "passed" if decision == "kept" else "failed",
                               "noise_not_measured": "passed"},
                    "reason": "Offline measured gate.",
                })
            return outcome

        deps = graph.RoundDeps(
            helper_client=Chat(HELPER_LINE), user_client=Chat(CLIENT_LINE), llm=agents,
            search=search, memory=memory, log=DecisionLog(root), root=root,
            judge=Judge(empty=mode == "nothing_scored"), gate=gate,
            usage=Meter(root, Prices(exa_search_usd=0)),
        )
        scenarios = [] if mode == "no_conversations" else [scenario_factory("a", split="train")]
        state = graph.run_round(deps, policy_path=policy, scenarios=scenarios, max_turns=1, seed=19)
        return root, state, deps
    return run


def archived_round(root, *, decision="reverted", action="add", measurements=True, round_=7):
    """A durable training root made with the existing record/log/usage writers."""
    base = policy_file(root)
    prefix = f"r{round_:02d}"
    records = [conversation(round_, name) for name in ("a", "b", "unscored")]
    write_jsonl(root / f"conversations/round-{round_:02d}-train.jsonl", records)
    write_jsonl(failures_path(root / f"conversations/round-{round_:02d}-train.jsonl"), [
        {"scenario_id": "failed-scenario", "reason": "offline scripted failure"},
    ])
    write_jsonl(root / f"scores/round-{round_:02d}-train.jsonl", [
        score(records[0]), score(records[1], empathy=4, relevance=2),
    ])
    pattern = FailurePattern(
        id=f"pattern-{prefix}", written_by="critic", created_at=STAMP, round=round_,
        parent_ids=[records[0].id], name="early advice", description="Advice before understanding.",
        seen_in=1, out_of=1, example_conversation_ids=[records[0].id],
    )
    research = ResearchNote(
        id=f"research-{prefix}", written_by="researcher", created_at=STAMP,
        round=round_, pattern_id=pattern.id, queries=[QUERY],
        sources=[{"url": URL, "technique": "Ask before advising."}],
    )
    rid = "C2" if action == "add" else "C1"
    text = None if action == "remove" else NEW_RULE
    candidate = root / f"policies/policy-v2-candidate-r{round_:02d}.md"
    candidate_text = base.read_text().replace("policy-v1", "policy-v2-candidate", 1)
    if action == "add":
        candidate_text += f"C2. {NEW_RULE}\n"
    else:
        candidate_text = candidate_text.replace(
            f"C1. {OLD_RULE}\n", "" if action == "remove" else f"C1. {NEW_RULE}\n")
    candidate.write_text(candidate_text, encoding="utf-8")
    edit = PolicyEdit(
        id=f"edit-{prefix}", written_by="proposer", created_at=STAMP, round=round_,
        pattern_id=pattern.id, research_id=research.id, base_policy_version="policy-v1",
        new_policy_version="policy-v2-candidate", policy_path=str(candidate.relative_to(root)),
        summary="Understand before advising.", action=action, rule_id=rid, rule_text=text,
    )
    log_scores = LogScores(
        before=Scores(empathy=2, relevance=3, quality=2.5),
        after=Scores(empathy=4, relevance=4.5, quality=4.25), noise=0.125,
    ) if measurements else None
    safety = LogSafety(
        case_file="offline-cases", passed=decision == "kept", cases=2,
        failed_case_ids=[] if decision == "kept" else ["unsafe-case-acceptance"],
    )
    in_force = "policy-v2" if decision == "kept" else "policy-v1"
    outcome = GateOutcome(
        id=f"gate-{prefix}", written_by="gatekeeper", created_at=STAMP, round=round_,
        edit_id=edit.id, decision=decision,
        before=log_scores.before if log_scores else Scores(empathy=0, relevance=0),
        after=log_scores.after if log_scores else Scores(empathy=0, relevance=0),
        safety_passed=safety.passed, policy_in_force=in_force, reason="Offline gate decision.",
    )
    objects = {"pattern": pattern, "research": research, "edit": edit, "outcome": outcome}
    refs = {kind: save_record(record, round_, root) for kind, record in objects.items()}
    if decision == "kept":
        (root / "policies/policy-v2.md").write_text(
            candidate_text.replace("policy-v2-candidate", "policy-v2", 1), encoding="utf-8")
    event = RoundEvent(
        round=round_, outcome=decision, records=refs, reason=outcome.reason,
        pattern=LogPattern(id=pattern.id, name=pattern.name, seen_in=1, out_of=1, status_before="open"),
        edit=LogEdit(id=edit.id, action=action, rule_id=rid, rule_text=text),
        policy=LogPolicy(base="policy-v1", base_sha256=sha256_text(base.read_text()),
                         new="policy-v2" if decision == "kept" else "policy-v2-candidate", in_force=in_force),
        scores=log_scores, safety=safety,
    )
    log = DecisionLog(root)
    log.append(event)
    for kind, ref in refs.items():
        log.append(MemoryCopyEvent(round=round_, record_id=ref.id,
                                   status="pending" if kind == "research" else "confirmed", attempt=1))
    # A different round's copy must not appear in this round's finish summary.
    log.append(MemoryCopyEvent(round=round_ + 1, record_id="pattern-other", status="pending", attempt=1))
    write_jsonl(root / "log/memory_reads.jsonl", [
        {"reader": "later-critic", "record_ids": [pattern.id], "query": "advice", "at": STAMP.isoformat()},
    ])
    checks = dict.fromkeys((
        "candidate_file", "safety_section", "one_rule", "token_cap", "safety_cases",
        "complete_evidence", "quality_gain", "dimension_drop", "noise_not_measured",
    ), "passed")
    if decision == "reverted":
        checks["safety_cases"] = "failed"
    if not measurements:
        checks.update(complete_evidence="failed", quality_gain="not_run", dimension_drop="not_run")
    write_json(root / f"scores/round-{round_:02d}-gate.json", {
        "round": round_, "decision": decision, "would_keep": decision == "kept",
        "failed": [name for name, status in checks.items() if status == "failed"],
        "checks": checks, "reason": outcome.reason,
    })
    prices = Prices(models={"offline-priced": ModelPrice(10000, 20000)})
    meter = Meter(root, prices, round=round_)
    meter.done(meter.sent("acceptance-agent", "offline-priced", 31, 17), CallUsage(31, 17))
    # Sent-only calls are not finished calls, even for an otherwise present agent.
    meter.sent("acceptance-agent", "offline-priced", 997, 999)
    meter.sent("unfinished-only-agent", "offline-priced", 991, 999)
    # Catch reports that sum the whole log instead of selecting the requested round.
    other = Meter(root, prices, round=round_ + 1)
    other.done(other.sent("other-round-agent", "offline-priced", 991, 999), CallUsage(991, 999))
    return event, objects, records, checks


def archived_exit(root, exit_):
    """Files for each graph exit, independent of spans or a live RoundState."""
    if exit_ in {"kept", "reverted"}:
        return archived_round(root, decision=exit_)[0]
    event, objects, _, _ = archived_round(root)
    reasons = {
        "no_conversations": "No conversation finished.",
        "nothing_scored": "The judge scored no conversation.",
        "no_pattern": "No repeated failure.",
        "critic_error": "critic failed: scripted invalid answers",
        "no_evidence": "No trustworthy source was found for this pattern.",
        "researcher_error": "researcher failed: scripted invalid answers",
        "proposer_error": "proposer failed: scripted invalid answers",
    }
    kinds = {
        "no_conversations": (), "nothing_scored": (), "no_pattern": (), "critic_error": (),
        "no_evidence": ("pattern", "research"), "researcher_error": ("pattern",),
        "proposer_error": ("pattern", "research"),
    }[exit_]
    refs = {kind: ref for kind, ref in event.records.items() if kind in kinds}
    for kind, ref in event.records.items():
        if kind not in kinds:
            (root / ref.path).unlink()
    if exit_ == "no_evidence":
        research = objects["research"].model_copy(update={"sources": []})
        refs["research"] = save_record(research, 7, root)
    (root / objects["edit"].policy_path).unlink()
    (root / "scores/round-07-gate.json").unlink()
    if exit_ == "no_conversations":
        write_jsonl(root / "conversations/round-07-train.jsonl", [])
    if exit_ in {"no_conversations", "nothing_scored"}:
        write_jsonl(root / "scores/round-07-train.jsonl", [])
    failures_path(root / "conversations/round-07-train.jsonl").unlink()
    event = RoundEvent(
        round=7, outcome="no_evidence" if exit_ in {"no_evidence", "researcher_error"} else "no_pattern",
        reason=reasons[exit_], records=refs, pattern=event.pattern if "pattern" in refs else None,
        policy=LogPolicy(base="policy-v1", base_sha256=event.policy.base_sha256, in_force="policy-v1"),
    )
    write_jsonl(DecisionLog(root).path, [event, *[
        MemoryCopyEvent(round=7, record_id=ref.id, status="confirmed", attempt=1) for ref in refs.values()
    ]])
    return event


# Section 1: privacy and the difference between measured and absent values.


def test_summary_key_and_converse_uses_the_returned_batch(tmp_path):
    assert SUMMARY_KEY == "puddles.summary"
    record = conversation(3, "a")
    summary = node_summary("converse", {"round": 3, "records": []}, {
        "records": [record], "failures": [{"scenario_id": "b", "reason": "offline failure"}],
    }, tmp_path)
    assert summary["finished"] == 1 and summary["failed"] == 1
    assert summary["file"] == "conversations/round-03-train.jsonl"
    assert CLIENT_LINE not in json.dumps(summary) and HELPER_LINE not in json.dumps(summary)


def test_judge_summary_averages_every_score_instead_of_critic_subset(tmp_path):
    records = [conversation(3, f"a-{i:02d}") for i in range(13)]
    # The default ten lowest go to the critic; two higher scores must still count.
    all_scores = [score(record, 1, 2) for record in records[:10]] + [
        score(record, 5, 4) for record in records[10:12]
    ]
    write_jsonl(tmp_path / "scores/round-03-train.jsonl", all_scores)
    # Amendment: score membership is defined by the round's conversations file.
    write_jsonl(tmp_path / "conversations/round-03-train.jsonl", records)
    summary = node_summary("judge", {"round": 3, "records": records}, {
        "scored": [scored(record, 1, 2) for record in records[:10]], "unscored": [records[12].id],
    }, tmp_path)
    assert summary["scored"] == 12 and summary["unscored"] == 1
    assert summary["mean_empathy"] == pytest.approx(20 / 12)
    assert summary["mean_relevance"] == pytest.approx(28 / 12)
    assert summary["judge_version"] == "judge-acceptance-v1"
    assert CLIENT_LINE not in json.dumps(summary) and HELPER_LINE not in json.dumps(summary)


def test_judge_summary_does_not_invent_measurements(tmp_path):
    record = conversation(1, "a")
    summary = node_summary("judge", {"round": 1, "records": [record]}, {
        "scored": [], "unscored": [record.id],
    }, tmp_path)
    assert summary["scored"] == 0 and summary["unscored"] == 1
    assert summary["mean_empathy"] is None and summary["mean_relevance"] is None
    assert summary["judge_version"] is None


@pytest.mark.parametrize("decision", ["kept", "reverted"])
def test_stand_in_gate_summary_never_converts_sentinels_to_measurements(tmp_path, decision):
    base = policy_file(tmp_path)
    result = RoundResult(round=1, outcome="candidate", base_policy_path=base)
    result.scores = result.safety = None
    outcome = GateOutcome(
        id="gate-r01", written_by="gatekeeper", created_at=STAMP, round=1, edit_id="edit-r01",
        decision=decision, before=Scores(empathy=0, relevance=0), after=Scores(empathy=0, relevance=0),
        safety_passed=decision == "kept", policy_in_force="policy-v1", reason="Stand-in gate.",
    )
    summary = node_summary("gate", {"round": 1, "result": result}, {"gate": outcome}, tmp_path)
    assert summary["decision"] == decision
    for key in ("quality_before", "quality_after", "noise", "safety_passed", "failed_checks"):
        assert summary[key] is None, f"Unmeasured {key} must stay null"


@pytest.mark.parametrize("measured_safety,outcome_safety", [
    (False, True), (True, False), (None, True), (None, False),
])
def test_gate_summary_safety_uses_result_and_never_outcome(tmp_path, measured_safety, outcome_safety):
    event, objects, _, _ = archived_round(tmp_path)
    result = RoundResult(round=7, outcome="candidate", base_policy_path=tmp_path / "policies/policy-v1.md")
    result.scores = event.scores
    result.safety = None if measured_safety is None else LogSafety(
        case_file="offline-cases", passed=measured_safety, cases=1,
        failed_case_ids=[] if measured_safety else ["unsafe-case-acceptance"],
    )
    objects["outcome"].safety_passed = outcome_safety
    summary = node_summary("gate", {"round": 7, "result": result}, {"gate": objects["outcome"]}, tmp_path)
    assert summary["safety_passed"] is measured_safety


def test_measured_gate_summary_uses_result_scores_and_all_failed_names(tmp_path):
    event, objects, _, checks = archived_round(tmp_path)
    result = RoundResult(round=7, outcome="candidate", base_policy_path=tmp_path / "policies/policy-v1.md")
    result.scores, result.safety = event.scores, event.safety
    # The persisted checks, not reason parsing, determine the failure names.
    objects["outcome"].before = Scores(empathy=1, relevance=1, quality=1)
    summary = node_summary("gate", {"round": 7, "result": result}, {"gate": objects["outcome"]}, tmp_path)
    assert summary["decision"] == "reverted"
    assert summary["quality_before"] == 2.5 and summary["quality_after"] == 4.25
    assert summary["noise"] == 0.125 and summary["safety_passed"] is False
    assert set(summary["failed_checks"]) == {name for name, status in checks.items() if status == "failed"}


def test_incomplete_gate_summary_keeps_all_failures_and_no_numbers(tmp_path):
    event, objects, _, checks = archived_round(tmp_path, measurements=False)
    result = RoundResult(round=7, outcome="candidate", base_policy_path=tmp_path / "policies/policy-v1.md")
    result.scores, result.safety = None, event.safety
    summary = node_summary("gate", {"round": 7, "result": result}, {"gate": objects["outcome"]}, tmp_path)
    assert summary["decision"] == "reverted" and summary["safety_passed"] is False
    assert summary["quality_before"] is None and summary["quality_after"] is None
    assert summary["noise"] is None
    assert set(summary["failed_checks"]) == {name for name, status in checks.items() if status == "failed"}
    assert len(summary["failed_checks"]) == 2


def test_finish_summary_reads_this_rounds_copy_events(tmp_path):
    event, _, _, _ = archived_round(tmp_path)
    log = DecisionLog(tmp_path)
    for attempt, status in ((2, "pending"), (3, "confirmed"), (4, "confirmed")):
        log.append(MemoryCopyEvent(round=7, record_id=event.records["pattern"].id,
                                   status=status, attempt=attempt))
    summary = node_summary("finish", {"round": 7}, {"event": event}, tmp_path)
    assert summary["outcome"] == "reverted" and summary["policy_in_force"] == "policy-v1"
    assert Counter((copy["record_id"], copy["status"]) for copy in summary["copies"]) == Counter(
        (item.record_id, item.status) for item in log.events()
        if isinstance(item, MemoryCopyEvent) and item.round == 7
    )


def test_all_node_summaries_leave_round_files_unchanged(tmp_path):
    root = tmp_path / "archive"
    event, objects, records, _ = archived_round(root)
    result = RoundResult(
        round=7, outcome="candidate", base_policy_path=root / "policies/policy-v1.md",
        pattern=objects["pattern"], research=objects["research"], edit=objects["edit"],
        candidate_path=root / objects["edit"].policy_path, refs=dict(event.records),
    )
    result.scores, result.safety = event.scores, event.safety
    state = {"round": 7, "policy_path": result.base_policy_path, "result": result, "records": records,
             "scored": [scored(records[0]), scored(records[1], 4, 2)], "gate": objects["outcome"]}
    updates = {
        "converse": {"records": records, "failures": []},
        "judge": {"scored": state["scored"], "unscored": [records[2].id]},
        "critic": {"result": result}, "researcher": {"result": result}, "proposer": {"result": result},
        "gate": {"gate": objects["outcome"]}, "finish": {"event": event},
    }
    before = snapshot(root)
    for name, update in updates.items():
        summary = node_summary(name, state, update, root)
        assert isinstance(summary, dict)
        encoded = json.dumps(summary)
        assert CLIENT_LINE not in encoded and HELPER_LINE not in encoded
        assert snapshot(root) == before, f"{name}'s summary changed the round's files"


@pytest.mark.parametrize("mode,node,missing", [
    ("no_pattern", "critic", "pattern"), ("proposer_error", "proposer", "edit"),
])
def test_no_pattern_or_edit_summary_has_null_and_reason(training_round, mode, node, missing):
    root, state, _ = training_round(mode=mode)
    summary = node_summary(node, {"round": 1}, {"result": state["result"]}, root)
    assert summary[missing] is None
    assert summary["reason"] == state["result"].reason
    assert summary["reason"]


@pytest.mark.parametrize("decision", ["kept", "reverted"])
def test_every_executed_node_span_has_safe_summary(training_round, decision):
    root, state, _ = training_round(decision=decision)
    mlflow.flush_trace_async_logging()
    experiment = mlflow.get_experiment_by_name("puddles")
    traces = mlflow.search_traces(experiment_ids=[experiment.experiment_id], return_type="list")
    spans = [span for trace in traces for span in trace.data.spans]
    by_id = {span.span_id: span for span in spans}
    for name in NODES:
        node_spans = [span for span in spans if span.name == name]
        assert len(node_spans) == 1, f"Expected one execution of node {name}"
        summary_spans = [span for span in spans if span.name in {name, f"summary:{name}"}
                         and SUMMARY_KEY in span.attributes]
        assert summary_spans, f"No summary on {name} or its fallback child"
        for span in summary_spans:
            if span.name.startswith("summary:"):
                assert span.parent_id in by_id and by_id[span.parent_id].name == name
                # Amendment: the round report no longer discloses fallback tracing.
            summary = span.attributes[SUMMARY_KEY]
            if isinstance(summary, str):
                summary = json.loads(summary)
            assert REQUIRED_KEYS[name] <= summary.keys()
            encoded = json.dumps(summary)
            assert CLIENT_LINE not in encoded and HELPER_LINE not in encoded
            if name == "critic":
                assert (summary["pattern"], summary["seen_in"], summary["out_of"]) == ("early advice", 1, 1)
            elif name == "researcher":
                assert summary["queries"] == 1 and summary["sources"] == [URL]
            elif name == "proposer":
                assert summary["action"] == "add" and summary["rule_text"] == NEW_RULE
                assert summary["rule_id"] == state["result"].edit.rule_id
            elif name == "finish":
                assert summary["outcome"] == decision
                assert summary["policy_in_force"] == state["event"].policy.in_force


@pytest.mark.parametrize("failure", ["no_span", "lookup", "attribute"])
def test_wrapper_returns_identical_update_even_when_mlflow_fails(tmp_path, monkeypatch, failure):
    received = {"round": 1}
    returned = {"records": [], "failures": []}
    calls = []

    def fn(state):
        assert state is received
        calls.append(state)
        return returned

    def broken(*args, **kwargs):
        raise RuntimeError("offline MLflow outage")

    class BrokenSpan:
        name = "converse"
        set_attribute = broken

    current = broken if failure == "lookup" else lambda: BrokenSpan() if failure == "attribute" else None
    monkeypatch.setattr(mlflow, "get_current_active_span", current)
    monkeypatch.setattr(mlflow, "start_span", broken)
    assert with_summary("converse", fn, tmp_path)(received) is returned
    assert calls == [received]


def test_wrapper_sets_summary_on_active_named_span_and_preserves_fn_exception(tmp_path):
    state = {"round": 1}
    update = {"records": [], "failures": []}
    with mlflow.start_span(name="converse") as span:
        returned = with_summary("converse", lambda supplied: update, tmp_path)(state)
        assert returned is update
        assert span.get_attribute(SUMMARY_KEY) == node_summary("converse", state, update, tmp_path)
    error = ValueError("the node failed")

    def fails(supplied):
        raise error

    with pytest.raises(ValueError) as caught:
        with_summary("converse", fails, tmp_path)(state)
    assert caught.value is error


# Reason: a summary-construction error must preserve the node's exact result, just like an MLflow error.
def test_wrapper_returns_identical_update_when_summarising_raises(tmp_path, monkeypatch):
    from puddles import round_view

    received = {"round": 1}
    returned = {"records": [], "failures": []}
    calls = []

    def fn(state):
        assert state is received
        calls.append(state)
        return returned

    def broken(*args, **kwargs):
        raise ValueError("offline summary failure")

    monkeypatch.setattr(round_view, "node_summary", broken)
    with mlflow.start_span(name="converse"):
        assert with_summary("converse", fn, tmp_path)(received) is returned
    assert calls == [received]


# Reason: summary failures at any node must leave a complete training round and its decision intact.
def test_round_survives_errors_in_every_node_summary(training_round, monkeypatch):
    from puddles import round_view

    attempted = []

    def broken(node, *args, **kwargs):
        attempted.append(node)
        raise ValueError("offline summary failure")

    monkeypatch.setattr(round_view, "node_summary", broken)
    root, state, _ = training_round(decision="reverted")
    assert attempted == list(NODES)
    assert state["event"].outcome == state["gate"].decision == "reverted"
    assert DecisionLog(root).rounds()[0] == state["event"]
    assert build_report(root, 1).startswith("# Round 1: reverted")


# Amendment: fallback children remain required, but are not mentioned in the round report.
def test_inaccessible_node_spans_use_summary_children_without_report_disclosure(training_round, monkeypatch):
    original = mlflow.get_current_active_span

    class InaccessibleNode:
        def __init__(self, span):
            self.span = span
            self.name = "inaccessible:" + span.name

        def __getattr__(self, name):
            return getattr(self.span, name)

    def inaccessible():
        span = original()
        return InaccessibleNode(span) if span is not None and span.name in NODES else span

    monkeypatch.setattr(mlflow, "get_current_active_span", inaccessible)
    root, _, _ = training_round()
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    spans = [span for trace in traces for span in trace.data.spans]
    by_id = {span.span_id: span for span in spans}
    for node in NODES:
        children = [span for span in spans if span.name == f"summary:{node}"]
        assert len(children) == 1 and SUMMARY_KEY in children[0].attributes
        assert by_id[children[0].parent_id].name == node
    assert "summary:" not in build_report(root, 1)


# Section 2: numbers must reflect the actual decision and measured evidence.


@pytest.mark.parametrize("decision,kept", [("kept", 1), ("reverted", 0)])
@pytest.mark.parametrize("measured", [False, True])
def test_round_metrics_omit_unmeasured_values(training_round, decision, kept, measured):
    training_round(decision=decision, measured=measured)
    experiment = mlflow.get_experiment_by_name("puddles")
    runs = MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    metrics = runs[0].data.metrics
    assert metrics["kept"] == kept
    expected = {"quality_before": 2.5, "quality_after": 4.25, "noise": 0.125}
    if measured:
        for key, value in expected.items():
            assert metrics[key] == value
    else:
        assert expected.keys().isdisjoint(metrics)


def test_round_without_gate_omits_all_four_gate_metrics(training_round):
    training_round(mode="no_pattern")
    experiment = mlflow.get_experiment_by_name("puddles")
    runs = MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert {"kept", "quality_before", "quality_after", "noise"}.isdisjoint(runs[0].data.metrics)


# Section 3: the report is durable, private, complete and observational.


def test_report_from_files_is_private_ordered_and_read_only(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    event, objects, records, _ = archived_round(root)
    # No recorded traces, models, search or memory service are needed to rebuild this root.
    def forbidden(*args, **kwargs):
        pytest.fail("The report attempted to rerun a training dependency")

    for cls, methods in ((Chat, ["complete"]), (Agents, ["structured", "structured_with_search"]),
                         (Memory, ["recall", "ingest", "wait"])):
        for method in methods:
            monkeypatch.setattr(cls, method, forbidden)
    monkeypatch.setattr(graph, "run_round", forbidden)
    # An unrelated live-session artifact must never be read into a training report.
    live = root / "conversations/live-session.jsonl"
    live_record = conversation(7, "live").model_copy(update={
        "source": "live", "split": "live",
        "turns": [Turn(speaker="user", text="LIVE-SESSION-PRIVATE-9361")],
    })
    write_jsonl(live, [live_record])
    (root / "opaque.bin").write_bytes(b"\x00\xffkeep these bytes\r\n")
    before = snapshot(root)
    original_open = Path.open

    def training_files_only(path, *args, **kwargs):
        assert path != live, "A training report must not read a live session"
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", training_files_only)
        report = build_report(root, 7)
    assert isinstance(report, str)
    headings = ["# Round 7: reverted", "## Nodes", "## Conversations", "## Memory",
                "## Policy change", "## Gate", "## Cost"]
    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert event.reason in report
    for text in (CLIENT_LINE, HELPER_LINE, "LIVE-SESSION-PRIVATE-9361"):
        assert text not in report
    for name in NODES:
        row_for(section(report, "Nodes"), name)
    assert "conversations/round-07-train.jsonl" in section(report, "Nodes")
    for record in objects.values():
        assert record.id in section(report, "Memory")
    memory = section(report, "Memory")
    for row in memory_trace(DecisionLog(root), root):
        assert row["record"] in memory and row["written_by"] in memory
        for reader in row["read_by"]:
            assert reader in memory
    assert "pending" in memory and "confirmed" in memory
    for record in records:
        row_for(section(report, "Conversations"), record.id)
    assert report_path(root, 7) == root / "reports/round-07.md"
    after = snapshot(root)
    assert after == before
    assert not report_path(root, 7).exists()


def test_report_lists_unscored_and_failed_conversations_without_zero_scores(tmp_path):
    _, _, records, _ = archived_round(tmp_path)
    conversations = section(build_report(tmp_path, 7), "Conversations")
    headers = next(row for row in table_rows(conversations)
                   if any("empathy" in cell.lower() for cell in row))
    empathy = next(i for i, cell in enumerate(headers) if "empathy" in cell.lower())
    relevance = next(i for i, cell in enumerate(headers) if "relevance" in cell.lower())
    a = row_for(conversations, records[0].id)
    b = row_for(conversations, records[1].id)
    assert float(a[empathy]) == 2 and float(a[relevance]) == 4
    assert float(b[empathy]) == 4 and float(b[relevance]) == 2
    unscored = row_for(conversations, records[2].id)
    assert unscored[empathy] == "" and unscored[relevance] == ""
    assert "unscored" in " ".join(unscored).lower()
    assert "failed-scenario" in conversations and "offline scripted failure" in conversations


def test_report_critic_column_uses_default_score_selection_and_ignores_pattern(tmp_path):
    event, objects, _, _ = archived_round(tmp_path)
    lows = [conversation(7, f"low-{i:02d}") for i in range(9)]
    tie_a, tie_b, high = [conversation(7, name) for name in ("tie-a", "tie-b", "high")]
    # Deliberately unsorted, including a tie at the default cutoff of ten.
    records = [high, tie_b, *reversed(lows), tie_a]
    write_jsonl(tmp_path / "conversations/round-07-train.jsonl", records)
    write_jsonl(tmp_path / "scores/round-07-train.jsonl", [
        score(high, 5, 5), score(tie_b, 3, 3),
        *[score(record, 1, 2) for record in reversed(lows)], score(tie_a, 3, 3),
    ])
    # This round could have used critic_conversations=11. The report must use the
    # documented default of ten even though its pattern names the eleventh item.
    pattern = objects["pattern"].model_copy(update={
        "parent_ids": [record.id for record in [*lows, tie_a, tie_b]],
        "example_conversation_ids": [tie_b.id], "out_of": 11,
    })
    refs = {**event.records, "pattern": save_record(pattern, 7, tmp_path)}
    event = event.model_copy(update={
        "records": refs, "pattern": event.pattern.model_copy(update={"out_of": 11}),
    })
    replace_logged_round(tmp_path, event)
    conversations = section(build_report(tmp_path, 7), "Conversations")
    headers = next(row for row in table_rows(conversations)
                   if any("critic" in cell.lower() for cell in row))
    critic = next(i for i, cell in enumerate(headers) if "critic" in cell.lower())
    selected = {record.id for record in [*lows, tie_a]}
    # Wording/Boolean markers are not fixed; require consistent, distinct states.
    given = {row_for(conversations, record.id)[critic] for record in records if record.id in selected}
    withheld = {row_for(conversations, record.id)[critic] for record in records if record.id not in selected}
    assert len(given) == len(withheld) == 1
    assert "" not in given and given.isdisjoint(withheld)


@pytest.mark.parametrize("with_pattern", [False, True])
def test_report_critic_column_is_empty_without_scores_file_even_with_pattern(tmp_path, with_pattern):
    if with_pattern:
        _, _, records, _ = archived_round(tmp_path)
        (tmp_path / "scores/round-07-train.jsonl").unlink()
        round_ = 7
    else:
        base = policy_file(tmp_path)
        records = [conversation(1, "unscored")]
        write_jsonl(tmp_path / "conversations/round-01-train.jsonl", records)
        DecisionLog(tmp_path).append(RoundEvent(
            round=1, outcome="no_pattern", reason="The judge scored no conversation.",
            policy=LogPolicy(base="policy-v1", base_sha256=sha256_text(base.read_text()), in_force="policy-v1"),
        ))
        round_ = 1
    conversations = section(build_report(tmp_path, round_), "Conversations")
    headers = next(row for row in table_rows(conversations)
                   if any("critic" in cell.lower() for cell in row))
    critic = next(i for i, cell in enumerate(headers) if "critic" in cell.lower())
    for record in records:
        assert row_for(conversations, record.id)[critic] == ""


def test_report_memory_lists_repeated_copy_events_separately(tmp_path):
    event, _, _, _ = archived_round(tmp_path)
    log = DecisionLog(tmp_path)
    record_id = event.records["research"].id
    for attempt, status in ((2, "pending"), (3, "confirmed"), (4, "confirmed")):
        log.append(MemoryCopyEvent(round=7, record_id=record_id, status=status, attempt=attempt))
    memory = section(build_report(tmp_path, 7), "Memory")
    for status in ("pending", "confirmed"):
        lines = [line for line in memory.splitlines()
                 if record_id in line and re.search(rf"\b{status}\b", line)]
        assert len(lines) >= 2, f"Copy events were collapsed for {record_id}: {status}"
    assert "pattern-other" not in memory


@pytest.mark.parametrize("action", ["add", "remove", "reword"])
@pytest.mark.parametrize("decision", ["kept", "reverted"])
def test_report_rule_diff_and_policy_in_force_are_not_confused(tmp_path, action, decision):
    event, _, _, _ = archived_round(tmp_path, action=action, decision=decision)
    policy = section(build_report(tmp_path, 7), "Policy change")
    if action in {"remove", "reword"}:
        assert f"- C1. {OLD_RULE}" in policy
    if action in {"add", "reword"}:
        assert f"+ {'C2' if action == 'add' else 'C1'}. {NEW_RULE}" in policy
    assert event.policy.in_force in policy
    assert build_report(tmp_path, 7).startswith(f"# Round 7: {decision}")


@pytest.mark.parametrize("measurements", [False, True])
def test_report_gate_contains_every_check_status_and_failed_safety_case(tmp_path, measurements):
    _, _, _, checks = archived_round(tmp_path, measurements=measurements)
    gate = section(build_report(tmp_path, 7), "Gate")
    for name, status in checks.items():
        # Permit either a table or a list, keeping the status tied to its check.
        lines = [line for line in gate.splitlines() if re.search(rf"\b{re.escape(name)}\b", line)]
        assert lines, f"Missing gate check {name}"
        assert any(status.replace("_", " ") in line.lower() for line in lines), (name, lines)
    assert "unsafe-case-acceptance" in gate
    if measurements:
        for value in (2.5, 4.25, 0.125):
            has_number(gate, value)
    else:
        # Reason: unmeasured values must not look measured in tables, prose, or lists.
        measurements_text = "\n".join(
            line for line in gate.splitlines()
            if not any(re.search(rf"\b{re.escape(name)}\b", line) for name in checks)
            and "unsafe-case-acceptance" not in line
        )
        assert not re.search(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)", measurements_text)
        assert not re.search(r"\b(?:passed|true)\b", measurements_text, re.I)


# Reason: absent safety evidence must never become a passed safety check in a report.
def test_report_missing_safety_is_not_reported_as_passed(tmp_path):
    event, _, _, checks = archived_round(tmp_path, measurements=False)
    event = event.model_copy(update={"safety": None})
    replace_logged_round(tmp_path, event)
    checks["safety_cases"] = "not_run"
    write_json(tmp_path / "scores/round-07-gate.json", {
        "round": 7, "decision": "reverted", "checks": checks,
        "failed": [name for name, status in checks.items() if status == "failed"],
    })
    gate = section(build_report(tmp_path, 7), "Gate")
    assert any("safety" in line.lower() and
               any(word in line.lower() for word in ("not run", "did not run", "not measured", "unmeasured"))
               for line in gate.splitlines())
    assert not any(("safety_cases" in line.lower() or "safety check" in line.lower())
                   and re.search(r"\b(?:passed|true)\b", line, re.I)
                   for line in gate.splitlines())


def test_report_cost_is_per_agent_and_only_for_requested_round(tmp_path):
    archived_round(tmp_path)
    cost = section(build_report(tmp_path, 7), "Cost")
    agent = " ".join(row_for(cost, "acceptance-agent"))
    for value in (1, 31, 17, 0.65):
        has_number(agent, value)
    assert re.search(r"(?<![\d.])0\.6500(?![\d.])", agent), "Dollars must have exactly four decimal places"
    assert "other-round-agent" not in cost
    assert "unfinished-only-agent" not in cost


@pytest.mark.parametrize("existing_report", [False, True])
def test_build_report_never_creates_or_overwrites_report_file(tmp_path, monkeypatch, existing_report):
    root = tmp_path / "archive"
    archived_round(root)
    path = root / "reports/round-07.md"
    if existing_report:
        path.parent.mkdir()
        path.write_bytes(b"A previously saved report must stay untouched.\r\n")
    before = snapshot(root)
    original_open = Path.open

    def read_only(path, mode="r", *args, **kwargs):
        # Reason: a write restored before returning still violates "build_report writes nothing".
        assert not any(flag in mode for flag in "wax+"), f"build_report attempted a write: {path}"
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", read_only)
        report = build_report(root, 7)
    assert report.startswith("# Round 7: reverted")
    assert snapshot(root) == before
    assert path.exists() is existing_report


@pytest.mark.parametrize("suffix", ["", " 2 conversations were left unscored."])
@pytest.mark.parametrize("exit_,ran", [
    ("no_conversations", {"converse", "judge", "finish"}),
    ("nothing_scored", {"converse", "judge", "finish"}),
    ("no_pattern", {"converse", "judge", "critic", "finish"}),
    ("critic_error", {"converse", "judge", "critic", "finish"}),
    ("no_evidence", {"converse", "judge", "critic", "researcher", "finish"}),
    ("researcher_error", {"converse", "judge", "critic", "researcher", "finish"}),
    ("proposer_error", {"converse", "judge", "critic", "researcher", "proposer", "finish"}),
    ("kept", set(NODES)), ("reverted", set(NODES)),
])
def test_report_infers_nodes_from_files_at_every_exit(tmp_path, exit_, ran, suffix):
    # Reason: the contract explicitly says "starts with", including appended unscored sentences.
    root = tmp_path / "archive"
    event = archived_exit(root, exit_)
    if suffix:
        event = event.model_copy(update={"reason": event.reason + suffix})
        replace_logged_round(root, event)
    before = snapshot(root)
    report = build_report(root, 7)
    assert report.startswith(f"# Round 7: {event.outcome}")
    assert event.reason in report
    nodes = section(report, "Nodes")
    for name in NODES:
        row = " ".join(row_for(nodes, name)).lower()
        unrun = "not run" in row or "did not run" in row
        assert unrun is (name not in ran), (exit_, name, row)
    assert snapshot(root) == before
    assert not (root / "reports/round-07.md").exists()


@pytest.mark.parametrize("mode,not_run", [
    ("no_pattern", ("researcher", "proposer", "gate")),
    ("critic_error", ("researcher", "proposer", "gate")),
    ("no_evidence", ("proposer", "gate")),
    ("researcher_error", ("proposer", "gate")),
    ("proposer_error", ("gate",)),
    ("no_conversations", ("critic", "researcher", "proposer", "gate")),
    ("nothing_scored", ("critic", "researcher", "proposer", "gate")),
])
def test_reports_survive_early_exits_and_mark_later_nodes_not_run(training_round, mode, not_run):
    root, state, _ = training_round(mode=mode)
    report = build_report(root, 1)
    assert report.startswith(f"# Round 1: {state['event'].outcome}")
    assert state["event"].reason in report
    nodes = section(report, "Nodes")
    for name in NODES:
        row = " ".join(row_for(nodes, name)).lower()
        if name in not_run:
            assert "not run" in row or "did not run" in row
        else:
            assert "not run" not in row and "did not run" not in row
    assert "The gate did not run." in section(report, "Gate")
    assert "No edit." in section(report, "Policy change")
    assert CLIENT_LINE not in report and HELPER_LINE not in report


# Reason: the documented every-exit guarantee also matters after an interrupted attempt reuses a round number.
def test_report_builds_after_stale_scores_from_interrupted_attempt(tmp_path, training_round):
    root = tmp_path / "training"
    write_jsonl(root / "scores/round-01-train.jsonl", [score(conversation(1, "previous-attempt"))])
    root, state, _ = training_round(mode="no_conversations")
    assert state["event"].reason.startswith("No conversation finished.")
    report = build_report(root, 1)
    assert report.startswith("# Round 1: no_pattern")
    assert "The gate did not run." in section(report, "Gate")
    # Amendment: orphan scores must be counted below the table, not silently omitted.
    assert "1 score(s) in the scores file belong to no conversation of this round" in section(report, "Conversations")


def test_stand_in_report_explicitly_says_nothing_was_measured(training_round):
    root, _, _ = training_round()
    gate = section(build_report(root, 1), "Gate").lower()
    assert len([line for line in gate.splitlines() if line.strip()]) == 1
    assert "stand-in" in gate
    assert "measur" in gate and any(word in gate for word in ("not ", "nothing", "no ", "unmeasured"))


# Section 3 CLI: artifact destination, defaults and error behavior.


def assert_not_attached(output, report):
    assert report in output.out
    notice = output.out.replace(report, "", 1).strip().splitlines()
    assert len(notice) == 1
    assert "attach" in notice[0].lower()
    assert any(word in notice[0].lower() for word in ("not ", "no ", "skip", "unattach", "couldn't"))


def test_show_round_without_matching_run_still_writes_prints_and_preserves_inputs(tmp_path, capsys):
    root = tmp_path / "archive"
    archived_round(root)
    before = snapshot(root)
    expected = build_report(root, 7)
    assert show_round.main(["--root", str(root), "--round", "7"]) == 0
    output = capsys.readouterr()
    assert_not_attached(output, expected)
    assert report_path(root, 7).read_text(encoding="utf-8") == expected
    after = snapshot(root)
    assert {key: value for key, value in after.items() if key != Path("reports/round-07.md")} == before


# Reason: a missing SQLite tracking store must not be created while displaying an archived round.
def test_show_round_missing_sqlite_store_adds_only_report(tmp_path, capsys):
    root = tmp_path / "archive"
    archived_round(root)
    before = snapshot(root)
    missing = root / "tracking/missing.db"
    expected = build_report(root, 7)
    assert show_round.main(["--root", str(root), "--round", "7",
                            "--mlflow", f"sqlite:///{missing}"]) == 0
    assert_not_attached(capsys.readouterr(), expected)
    assert snapshot(root) == {**before, Path("reports/round-07.md"): expected.encode("utf-8")}
    assert not missing.parent.exists()


@pytest.mark.parametrize("source", ["argument", "environment", "default"])
def test_show_round_attaches_to_named_run_using_tracking_address_precedence(tmp_path, monkeypatch, capsys, source):
    root = tmp_path / "archive"
    archived_round(root)
    # Same experiment/run names in three stores catch a wrong precedence decision.
    clients = {}
    for name in ("argument", "environment", "default"):
        uri = f"sqlite:///{tmp_path}/{name}.db"
        client = MlflowClient(tracking_uri=uri)
        experiment = client.create_experiment("puddles", artifact_location=(tmp_path / name).as_uri())
        # Amendment: matching run names are eligible only with this root's resolved tag.
        target = client.create_run(experiment, tags={"mlflow.runName": "round-07", "root": str(root.resolve())})
        other = client.create_run(experiment, tags={"mlflow.runName": "round-08", "root": str(root.resolve())})
        unrelated_experiment = client.create_experiment(
            "unrelated", artifact_location=(tmp_path / f"{name}-unrelated").as_uri())
        unrelated = client.create_run(unrelated_experiment,
                                      tags={"mlflow.runName": "round-07", "root": str(root.resolve())})
        client.set_terminated(target.info.run_id)
        client.set_terminated(other.info.run_id)
        client.set_terminated(unrelated.info.run_id)
        clients[name] = (uri, client, target.info.run_id, other.info.run_id, unrelated.info.run_id)
    monkeypatch.setattr(tracing, "DEFAULT_TRACKING_URI", clients["default"][0])
    # Reason: importing the documented default into the script is allowed; patch that binding too.
    monkeypatch.setattr(show_round, "DEFAULT_TRACKING_URI", clients["default"][0], raising=False)
    if source != "default":
        monkeypatch.setenv("MLFLOW_TRACKING_URI", clients["environment"][0])
    else:
        # Reason: MLflow setup itself sets this variable; a default-address test must remove it afterwards.
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    args = ["--root", str(root), "--round", "7"]
    if source == "argument":
        args.extend(["--mlflow", clients["argument"][0]])
    expected = build_report(root, 7)
    assert show_round.main(args) == 0
    assert expected in capsys.readouterr().out
    assert report_path(root, 7).read_text(encoding="utf-8") == expected
    for name, (_, client, run_id, other_id, unrelated_id) in clients.items():
        artifacts = client.list_artifacts(run_id, "reports")
        if name == source:
            assert "reports/round-07.md" in {artifact.path for artifact in artifacts}
            download = tmp_path / "download"
            download.mkdir()
            path = client.download_artifacts(run_id, "reports/round-07.md", str(download))
            assert Path(path).read_text(encoding="utf-8") == expected
        else:
            assert not artifacts
        assert not client.list_artifacts(other_id, "reports")
        assert not client.list_artifacts(unrelated_id, "reports")


@pytest.mark.parametrize("failure", ["tracking_store", "artifact_directory", "artifact_scheme"])
def test_show_round_attachment_failure_still_writes_prints_and_returns_zero(tmp_path, capsys, failure):
    root = tmp_path / "archive"
    archived_round(root)
    expected = build_report(root, 7)
    before = snapshot(root)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("This file prevents any directory being created here.", encoding="utf-8")
    if failure == "tracking_store":
        # Unsupported local scheme fails immediately, without connection retries.
        uri = "acceptance-unsupported://tracking"
    else:
        uri = f"sqlite:///{tmp_path}/attachment.db"
        client = MlflowClient(tracking_uri=uri)
        location = blocker.as_uri() if failure == "artifact_directory" else "acceptance-unsupported://artifacts"
        experiment = client.create_experiment("puddles", artifact_location=location)
        # Amendment: reach the attachment failure using an eligible root-tagged run.
        run = client.create_run(experiment, tags={"mlflow.runName": "round-07", "root": str(root.resolve())})
        client.set_terminated(run.info.run_id)
    assert show_round.main(["--root", str(root), "--round", "7", "--mlflow", uri]) == 0
    assert_not_attached(capsys.readouterr(), expected)
    assert (root / "reports/round-07.md").read_text(encoding="utf-8") == expected
    after = snapshot(root)
    assert {key: value for key, value in after.items() if key != Path("reports/round-07.md")} == before


def test_show_round_uses_most_recent_of_several_matching_runs(tmp_path, capsys):
    root = tmp_path / "archive"
    archived_round(root)
    expected = build_report(root, 7)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("puddles")
    runs = [client.create_run(experiment.experiment_id, start_time=stamp,
                              # Amendment: newest selection is among runs of this root only.
                              tags={"mlflow.runName": "round-07", "root": str(root.resolve())})
            for stamp in (1000, 2000, 3000)]
    # Start times and creation order agree; ending the oldest last must not select it.
    for index, run in enumerate(reversed(runs)):
        client.set_terminated(run.info.run_id, end_time=4000 + index)
    assert show_round.main(["--root", str(root), "--round", "7"]) == 0
    assert expected in capsys.readouterr().out
    assert (root / "reports/round-07.md").read_text(encoding="utf-8") == expected
    for run in runs[:-1]:
        assert not client.list_artifacts(run.info.run_id, "reports")
    newest = runs[-1].info.run_id
    assert "reports/round-07.md" in {item.path for item in client.list_artifacts(newest, "reports")}
    download = tmp_path / "download"
    download.mkdir()
    attached = client.download_artifacts(newest, "reports/round-07.md", str(download))
    assert Path(attached).read_text(encoding="utf-8") == expected


def test_show_round_defaults_to_last_round_in_log(tmp_path, capsys):
    root = tmp_path / "archive"
    archived_round(root, round_=7)
    archived_round(root, round_=8, decision="kept")
    assert show_round.main(["--root", str(root)]) == 0
    output = capsys.readouterr().out
    assert "# Round 8: kept" in output
    assert report_path(root, 8).read_text(encoding="utf-8").startswith("# Round 8: kept")


@pytest.mark.parametrize("missing", ["log", "round"])
def test_show_round_invalid_root_or_round_returns_two_with_one_stderr_line(tmp_path, capsys, missing):
    root = tmp_path / "archive"
    if missing == "round":
        archived_round(root)
    before = snapshot(root)
    assert show_round.main(["--root", str(root), "--round", "99"]) == 2
    output = capsys.readouterr()
    assert len(output.err.strip().splitlines()) == 1
    assert output.err.strip()
    assert snapshot(root) == before


@pytest.mark.parametrize("unreadable", ["directory", "undecodable", "malformed", "invalid_event"])
def test_show_round_unreadable_log_returns_two_with_one_stderr_line(tmp_path, capsys, unreadable):
    root = tmp_path / "archive"
    archived_round(root)
    path = DecisionLog(root).path
    if unreadable == "directory":
        path.unlink()
        path.mkdir()
    else:
        contents = {
            "undecodable": b"\xff\xfe\xfa\n",
            "malformed": b'{"event": "round",\n',
            "invalid_event": b'{"event": "round", "round": "not an integer"}\n',
        }
        path.write_bytes(contents[unreadable])
    before = snapshot(root)
    assert show_round.main(["--root", str(root), "--round", "7"]) == 2
    output = capsys.readouterr()
    assert len(output.err.strip().splitlines()) == 1 and output.err.strip()
    assert snapshot(root) == before
    assert not (root / "reports/round-07.md").exists()


# Reason: the contract joins by conversation ID, without a timestamp freshness heuristic.
@pytest.mark.parametrize("offset", [-2, -1.001, 0, 2])
def test_matching_score_is_never_dropped_because_of_clock_offset(tmp_path, offset):
    _, _, records, _ = archived_round(tmp_path)
    first = score(records[0], 5, 1).model_copy(update={"created_at": STAMP + timedelta(seconds=offset)})
    write_jsonl(tmp_path / "scores/round-07-train.jsonl", [first, score(records[1], 4, 2)])
    before = snapshot(tmp_path)
    report = build_report(tmp_path, 7)
    row = row_for(section(report, "Conversations"), records[0].id)
    assert row[1] and row[2], "A matching conversation's measured scores were discarded"
    assert float(row[1]) == 5 and float(row[2]) == 1
    assert "score(s) in the scores file belong to no conversation" not in report
    summary = node_summary("judge", {"round": 7, "records": records}, {"scored": [], "unscored": []}, tmp_path)
    assert summary["scored"] == 2
    assert summary["mean_empathy"] == 4.5 and summary["mean_relevance"] == 1.5
    assert snapshot(tmp_path) == before


# Reason: orphan scores must be counted, excluded from both means, and never joined or exposed as rows.
def test_orphan_scores_are_counted_without_changing_current_scores(tmp_path):
    _, _, records, _ = archived_round(tmp_path)
    orphan = score(conversation(7, "orphan"), 1, 1)
    write_jsonl(tmp_path / "scores/round-07-train.jsonl", [orphan, score(records[0]), score(records[1], 4, 2)])
    report = build_report(tmp_path, 7)
    conversations = section(report, "Conversations")
    assert "1 score(s) in the scores file belong to no conversation of this round" in conversations
    assert orphan.conversation_id not in conversations
    assert float(row_for(conversations, records[0].id)[1]) == 2
    summary = node_summary("judge", {"round": 7, "records": records}, {"scored": [], "unscored": []}, tmp_path)
    assert summary["scored"] == 2 and summary["mean_empathy"] == summary["mean_relevance"] == 3


# Reason: a supplied noise figure is measured only with the amended contract's explicit passed marker.
@pytest.mark.parametrize("marker", ["passed", "failed", "not_run", None, "absent", "missing_file"])
def test_noise_requires_explicit_pass_marker_in_all_three_views(tmp_path, marker):
    from puddles.round_view import gate_metrics

    event, objects, _, checks = archived_round(tmp_path)
    path = tmp_path / "scores/round-07-gate.json"
    if marker == "missing_file":
        path.unlink()
    else:
        checks.pop("noise_not_measured")
        if marker != "absent":
            checks["noise_not_measured"] = marker
        write_json(path, {"round": 7, "decision": "reverted", "checks": checks})
    result = RoundResult(round=7, outcome="candidate", base_policy_path=tmp_path / "policies/policy-v1.md")
    result.scores, result.safety = event.scores, event.safety
    state = {"round": 7, "result": result, "gate": objects["outcome"]}
    summary = node_summary("gate", state, {"gate": objects["outcome"]}, tmp_path)
    metrics = gate_metrics(state, tmp_path)
    assert summary["quality_before"] == metrics["quality_before"] == 2.5
    assert summary["quality_after"] == metrics["quality_after"] == 4.25
    assert summary["noise"] == (0.125 if marker == "passed" else None)
    assert ("noise" in metrics) is (marker == "passed")
    gate = section(build_report(tmp_path, 7), "Gate")
    if marker == "passed":
        has_number(gate, 0.125)
    elif marker == "missing_file":
        assert "Stand-in gate: nothing was measured." in gate
    else:
        assert "not measured; the figure given was" in gate


# Reason: root-tag isolation must reject both newer foreign-root runs and newer untagged legacy runs.
@pytest.mark.parametrize("matching", [False, True])
def test_attachment_rejects_foreign_and_untagged_runs(tmp_path, capsys, matching):
    root = tmp_path / "archive"
    archived_round(root)
    expected = build_report(root, 7)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("puddles")
    tags = [{"mlflow.runName": "round-07", "root": str((tmp_path / "other").resolve())},
            {"mlflow.runName": "round-07"}]
    if matching:
        tags.insert(0, {"mlflow.runName": "round-07", "root": str(root.resolve())})
    runs = [client.create_run(experiment.experiment_id, start_time=(i + 1) * 1000, tags=tag)
            for i, tag in enumerate(tags)]
    for run in runs:
        client.set_terminated(run.info.run_id)
    # Exercise resolution, not just an already canonical absolute spelling.
    assert show_round.main(["--root", str(root / ".." / root.name), "--round", "7"]) == 0
    output = capsys.readouterr()
    if not matching:
        assert_not_attached(output, expected)
    for i, run in enumerate(runs):
        artifacts = client.list_artifacts(run.info.run_id, "reports")
        assert bool(artifacts) is (matching and i == 0)


# Reason: the producer must record the same resolved root identity that report attachment matches.
def test_round_run_records_resolved_root_tag(training_round):
    root, _, deps = training_round()
    deps.root = Path("training")  # the fixture's working directory is this root's parent
    graph.run_round(deps, policy_path=root / "policies/policy-v1.md", scenarios=[])
    experiment = mlflow.get_experiment_by_name("puddles")
    runs = MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 2
    assert all(run.data.tags["root"] == str(root.resolve()) for run in runs)


# Reason: incomplete or unreadable artifacts must be disclosed without crashes, writes, or leaked turns.
@pytest.mark.parametrize("kind", ["conversations", "scores", "failures", "gate", "usage", "reads", "record", "policy"])
@pytest.mark.parametrize("damage", ["cut", "directory", "undecodable"])
def test_unreadable_artifacts_preserve_report_summary_and_cli(tmp_path, capsys, kind, damage):
    root = tmp_path / "archive"
    event, objects, records, _ = archived_round(root, action="reword")
    paths = {
        "conversations": root / "conversations/round-07-train.jsonl",
        "scores": root / "scores/round-07-train.jsonl",
        "failures": root / "conversations/round-07-train.failures.jsonl",
        "gate": root / "scores/round-07-gate.json",
        "usage": root / "log/usage.jsonl", "reads": root / "log/memory_reads.jsonl",
        "record": root / event.records["pattern"].path, "policy": root / "policies/policy-v1.md",
    }
    path = paths[kind]
    if damage == "directory":
        path.unlink()
        path.mkdir()
    elif damage == "undecodable":
        path.write_bytes(b"\xff\xfe" + CLIENT_LINE.encode())
    else:
        path.write_text('{"user": "' + CLIENT_LINE + '", "helper": "' + HELPER_LINE + '"', encoding="utf-8")
    before = snapshot(root)
    result = RoundResult(round=7, outcome="candidate", base_policy_path=root / "policies/policy-v1.md")
    result.scores, result.safety = event.scores, event.safety
    state = {"round": 7, "result": result}
    for node, update in (("judge", {}), ("gate", {"gate": objects["outcome"]}), ("finish", {"event": event})):
        summary = node_summary(node, state, update, root)
        assert isinstance(summary, dict)
        assert CLIENT_LINE not in json.dumps(summary) and HELPER_LINE not in json.dumps(summary)
    report = build_report(root, 7)
    assert report.startswith("# Round 7: reverted")
    assert CLIENT_LINE not in report and HELPER_LINE not in report
    assert any(word in report.lower() for word in ("could not", "cannot", "missing", "not in the root", "left out"))
    assert snapshot(root) == before
    assert show_round.main(["--root", str(root), "--round", "7"]) == 0
    assert report in capsys.readouterr().out
    assert snapshot(root) == {**before, Path("reports/round-07.md"): report.encode()}


# Reason: malformed check values are neither statuses nor conversation text safe to copy into a report.
@pytest.mark.parametrize("status", [{}, [], None, True, "unknown-status", CLIENT_LINE, HELPER_LINE])
def test_invalid_gate_check_values_are_unreadable_not_reported_values(tmp_path, status):
    event, objects, _, _ = archived_round(tmp_path)
    write_json(tmp_path / "scores/round-07-gate.json", {
        "round": 7, "decision": "reverted", "checks": {"safety_cases": status},
    })
    result = RoundResult(round=7, outcome="candidate", base_policy_path=tmp_path / "policies/policy-v1.md")
    result.scores, result.safety = event.scores, event.safety
    report = build_report(tmp_path, 7)
    assert CLIENT_LINE not in report and HELPER_LINE not in report
    assert any(word in section(report, "Gate").lower() for word in (
        "could not", "cannot", "invalid", "not a gate report", "unreadable", "not valid", "left out", "omitted",
    ))
    summary = node_summary("gate", {"round": 7, "result": result}, {"gate": objects["outcome"]}, tmp_path)
    assert summary["failed_checks"] is None


# Reason: missing and nonfinite usage numbers must not become zeros, nan/inf values, or a report crash.
@pytest.mark.parametrize("bad", ["missing", "nan_cost", "infinite_cost", "infinite_tokens"])
def test_unreadable_usage_numbers_are_never_shown_as_measurements(tmp_path, bad):
    archived_round(tmp_path)
    row = {"event": "done", "round": 7, "kind": "model", "agent": "invalid-usage-agent"}
    unknown = (2, 3, 4)
    if bad != "missing":
        row.update(tokens_in=10, tokens_out=5, cost_usd=0.25)
        field = "tokens_in" if bad == "infinite_tokens" else "cost_usd"
        row[field] = float("nan") if bad == "nan_cost" else float("inf")
        unknown = (2,) if field == "tokens_in" else (4,)
    path = tmp_path / "log/usage.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    report = build_report(tmp_path, 7)
    cost = section(report, "Cost")
    rows = [cells for cells in table_rows(cost) if "invalid-usage-agent" in cells]
    for cells in rows:
        for index in unknown:
            assert cells[index].lower() in ("", "null", "none", "-", "—", "n/a") or any(
                word in cells[index].lower() for word in ("not measured", "unmeasured", "unknown", "unreadable", "unavailable")
            )
    assert any(word in cost.lower() for word in (
        "could not", "cannot", "invalid", "missing", "not finite", "nonfinite", "unreadable", "left out", "omitted",
    ))


# Reason: every raw usage field must survive adversarial JSON values without inventing a measurement.
@pytest.mark.parametrize("field", ["event", "kind", "agent", "round", "tokens_in", "tokens_out", "cost_usd"])
@pytest.mark.parametrize("value", [None, True, {}, [CLIENT_LINE], {"said": HELPER_LINE},
                                  CLIENT_LINE, HELPER_LINE, float("nan"), float("inf"), 10 ** 400],
                         ids=["null", "bool", "object", "list", "nested", "user", "helper", "nan", "inf", "huge"])
def test_usage_field_boundary_matrix(tmp_path, field, value):
    archived_round(tmp_path)
    line = {"event": "done", "round": 7, "kind": "model", "agent": "boundary-agent",
            "tokens_in": 13, "tokens_out": 11, "cost_usd": 0.25}
    line[field] = value
    with (tmp_path / "log/usage.jsonl").open("a") as stream:
        stream.write(json.dumps(line) + "\n")
    before = snapshot(tmp_path)
    report = build_report(tmp_path, 7)
    assert CLIENT_LINE not in report and HELPER_LINE not in report
    assert snapshot(tmp_path) == before
    # Amendment: token counts above 10**12 are unreadable and excluded, including these two 10**400 cases.
    assert not any("boundary-agent" in row for row in table_rows(section(report, "Cost")))


def _change_raw_log(root, path, value, *, copy=False):
    rows = [json.loads(line) for line in DecisionLog(root).path.read_text().splitlines()]
    target = next(row for row in rows if row["event"] == ("memory_copy" if copy else "round"))
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    write_jsonl(DecisionLog(root).path, rows)


# Reason: an unreadable log may be refused, but its rejected input must never be echoed as a report value.
def _report_or_safe_log_refusal(root, capsys, *, allow_log_refusal=True):
    from puddles.round_view import ReportError

    try:
        return build_report(root, 7)
    except ReportError:
        # Reason: a damaged record or valid exempt string must not be mistaken for an unreadable log.
        if not allow_log_refusal:
            raise
        assert show_round.main(["--root", str(root), "--round", "7"]) == 2
        output = capsys.readouterr()
        assert CLIENT_LINE not in output.err and HELPER_LINE not in output.err
        assert len(output.err.strip().splitlines()) == 1
        return None


# Reason: ID/version/agent fields are not the contract's exempt reason, pattern, rule-text or URL fields.
@pytest.mark.parametrize("field", ["conversation_id", "failure_id", "judge_version", "reader", "writer",
                                  "copy_id", "policy_version", "rule_id", "safety_id", "ref_id", "ref_path",
                                  "policy_base", "policy_new", "edit_policy_path", "case_file"])
@pytest.mark.parametrize("value", [None, True, {}, [CLIENT_LINE], {"said": HELPER_LINE},
                                  CLIENT_LINE, HELPER_LINE, float("nan"), float("inf"), 10 ** 400],
                         ids=["null", "bool", "object", "list", "nested", "user", "helper", "nan", "inf", "huge"])
def test_identifier_boundary_matrix(tmp_path, capsys, field, value):
    # Reason: case_file is rendered on the no-failed-cases branch, including a kept round.
    event, _, _, _ = archived_round(tmp_path, decision="kept" if field == "case_file" else "reverted")
    if field in {"conversation_id", "judge_version"}:
        path = tmp_path / ("conversations/round-07-train.jsonl" if field == "conversation_id"
                           else "scores/round-07-train.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["id" if field == "conversation_id" else field] = value
        write_jsonl(path, rows)
    elif field == "failure_id":
        write_jsonl(tmp_path / "conversations/round-07-train.failures.jsonl",
                    [{"scenario_id": value, "reason": "offline failure"}])
    elif field == "reader":
        write_jsonl(tmp_path / "log/memory_reads.jsonl",
                    [{"reader": value, "record_ids": [event.records["pattern"].id]}])
    elif field in {"writer", "edit_policy_path"}:
        kind = "pattern" if field == "writer" else "edit"
        ref = event.records[kind]
        path = tmp_path / ref.path
        data = json.loads(path.read_text())
        data["written_by" if field == "writer" else "policy_path"] = value
        write_json(path, data)
        _change_raw_log(tmp_path, ["records", kind, "sha256"], sha256_text(path.read_text()))
    else:
        paths = {"copy_id": ["record_id"], "policy_version": ["policy", "in_force"],
                 "rule_id": ["edit", "rule_id"], "safety_id": ["safety", "failed_case_ids"],
                 "ref_id": ["records", "pattern", "id"], "ref_path": ["records", "pattern", "path"],
                 "policy_base": ["policy", "base"], "policy_new": ["policy", "new"],
                 "case_file": ["safety", "case_file"]}
        _change_raw_log(tmp_path, paths[field], [value] if field == "safety_id" else value,
                        copy=field == "copy_id")
    before = snapshot(tmp_path)
    report = _report_or_safe_log_refusal(tmp_path, capsys, allow_log_refusal=field not in {
        "conversation_id", "failure_id", "judge_version", "reader", "writer", "edit_policy_path",
    })
    if report is not None:
        assert CLIENT_LINE not in report and HELPER_LINE not in report
        for node in ("judge", "finish"):
            summary = node_summary(node, {"round": 7}, {}, tmp_path)
            assert CLIENT_LINE not in json.dumps(summary) and HELPER_LINE not in json.dumps(summary)
    assert snapshot(tmp_path) == before


# Reason: permitted free-text strings must remain verbatim; containers and numbers are not text.
@pytest.mark.parametrize("field", ["reason", "pattern", "rule", "url"])
@pytest.mark.parametrize("value", [CLIENT_LINE, HELPER_LINE, {}, [CLIENT_LINE], {"said": HELPER_LINE},
                                  float("nan"), float("inf"), 10 ** 400, True],
                         ids=["user", "helper", "object", "list", "nested", "nan", "inf", "huge", "bool"])
def test_permitted_free_text_boundary_matrix(tmp_path, capsys, field, value):
    event, _, _, _ = archived_round(tmp_path)
    if field == "reason":
        _change_raw_log(tmp_path, ["reason"], value)
    elif field == "rule":
        _change_raw_log(tmp_path, ["edit", "rule_text"], value)
    else:
        kind = "pattern" if field == "pattern" else "research"
        ref = event.records[kind]
        path = tmp_path / ref.path
        data = json.loads(path.read_text())
        if field == "pattern":
            data["name"] = value
            _change_raw_log(tmp_path, ["pattern", "name"], value)
        else:
            data["sources"][0]["url"] = value
        write_json(path, data)
        _change_raw_log(tmp_path, ["records", kind, "sha256"], sha256_text(path.read_text()))
    report = _report_or_safe_log_refusal(tmp_path, capsys,
                                       allow_log_refusal=not isinstance(value, str) and field != "url")
    if report is not None:
        if isinstance(value, str):
            # URL is summarized only in the researcher's span, so inspect the loaded record there.
            if field == "url":
                from puddles.records import load_record
                ref = DecisionLog(tmp_path).rounds()[0].records["research"]
                result = RoundResult(round=7, outcome="candidate", base_policy_path=tmp_path / "policies/policy-v1.md")
                result.research = load_record(ref, ResearchNote, tmp_path)
                assert value in node_summary("researcher", {"round": 7, "result": result}, {}, tmp_path)["sources"]
            else:
                assert value in report
        else:
            assert CLIENT_LINE not in report and HELPER_LINE not in report


# Reason: booleans and numeric strings must not become measured quality by Pydantic coercion.
@pytest.mark.parametrize("field", ["quality_before", "quality_after", "noise"])
@pytest.mark.parametrize("value", [True, "4.5", CLIENT_LINE, HELPER_LINE, {}, [CLIENT_LINE],
                                  {"said": HELPER_LINE}, float("nan"), float("inf"), 10 ** 400],
                         ids=["bool", "numeric-string", "user", "helper", "object", "list", "nested", "nan", "inf", "huge"])
def test_logged_quality_boundary_matrix(tmp_path, capsys, field, value):
    archived_round(tmp_path)
    paths = {"quality_before": ["scores", "before", "quality"],
             "quality_after": ["scores", "after", "quality"], "noise": ["scores", "noise"]}
    _change_raw_log(tmp_path, paths[field], value)
    report = _report_or_safe_log_refusal(tmp_path, capsys)
    if report is not None:
        gate = section(report, "Gate")
        label = {"quality_before": "Quality before", "quality_after": "Quality after", "noise": "Noise"}[field]
        assert f"{label}: unreadable" in gate or f"{label}: not measured" in gate
        assert CLIENT_LINE not in report and HELPER_LINE not in report


# Reason: the enum boundary must reject non-status JSON values, including non-finite and huge numbers.
@pytest.mark.parametrize("field", ["round", "decision", "checks", "check_name", "status", "failed"])
@pytest.mark.parametrize("value", [True, CLIENT_LINE, HELPER_LINE, {}, [CLIENT_LINE], {"said": HELPER_LINE},
                                  float("nan"), float("inf"), 10 ** 400],
                         ids=["bool", "user", "helper", "object", "list", "nested", "nan", "inf", "huge"])
def test_gate_report_boundary_matrix(tmp_path, field, value):
    archived_round(tmp_path)
    data = {"round": 7, "decision": "reverted", "checks": {"safety_cases": "failed"}}
    if field == "status":
        data["checks"]["safety_cases"] = value
    elif field == "check_name":
        data["checks"] = {json.dumps(value): "failed"}
    else:
        data[field] = value
    write_json(tmp_path / "scores/round-07-gate.json", data)
    report = build_report(tmp_path, 7)
    assert CLIENT_LINE not in report and HELPER_LINE not in report
    summary = node_summary("gate", {"round": 7}, {"gate": None}, tmp_path)
    assert CLIENT_LINE not in json.dumps(summary) and HELPER_LINE not in json.dumps(summary)


# Reason: malformed MLflow metadata and attachment exceptions must not echo rejected conversation text.
@pytest.mark.parametrize("field", ["experiment_id", "root", "run_id"])
@pytest.mark.parametrize("value", [CLIENT_LINE, {}, [HELPER_LINE], {"said": CLIENT_LINE},
                                  float("nan"), float("inf"), 10 ** 400],
                         ids=["user", "object", "list", "nested", "nan", "inf", "huge"])
def test_mlflow_metadata_boundary_matrix(tmp_path, monkeypatch, capsys, field, value):
    from types import SimpleNamespace

    root = tmp_path / "archive"
    archived_round(root)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("puddles")
    run = client.create_run(experiment.experiment_id, tags={"mlflow.runName": "round-07", "root": str(root.resolve())})
    client.set_terminated(run.info.run_id)
    if field == "experiment_id":
        monkeypatch.setattr(client, "get_experiment_by_name", lambda name: SimpleNamespace(experiment_id=value))
    else:
        candidate = SimpleNamespace(data=SimpleNamespace(tags={"root": value if field == "root" else str(root.resolve())}),
                                    info=SimpleNamespace(run_id=value if field == "run_id" else run.info.run_id))
        monkeypatch.setattr(client, "search_runs", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda **kwargs: client)
    assert show_round.main(["--root", str(root), "--round", "7"]) == 0
    output = capsys.readouterr()
    assert CLIENT_LINE not in output.out + output.err and HELPER_LINE not in output.out + output.err
    assert report_path(root, 7).exists()


# Reason: out-of-range integer token counts must not make formatting a report raise.
def test_large_token_totals_do_not_crash_report_or_cli(tmp_path, capsys):
    archived_round(tmp_path)
    with (tmp_path / "log/usage.jsonl").open("a") as stream:
        for _ in range(2):
            stream.write(json.dumps({"event": "done", "round": 7, "kind": "model", "agent": "huge-token-agent",
                                     "tokens_in": 10 ** 4300 - 1, "tokens_out": 1, "cost_usd": 0.25}) + "\n")
    report = build_report(tmp_path, 7)
    assert report.startswith("# Round 7: reverted")
    assert show_round.main(["--root", str(tmp_path), "--round", "7"]) == 0
    assert report in capsys.readouterr().out


# Reason: scores and counts must retain their JSON number type instead of accepting Boolean/text coercion.
@pytest.mark.parametrize("field", ["empathy", "relevance", "seen_in", "out_of", "cases", "attempt"])
@pytest.mark.parametrize("value", [True, "4", CLIENT_LINE, HELPER_LINE, {}, [CLIENT_LINE],
                                  {"said": HELPER_LINE}, float("nan"), float("inf"), 10 ** 400],
                         ids=["bool", "numeric-string", "user", "helper", "object", "list", "nested", "nan", "inf", "huge"])
def test_score_and_count_boundary_matrix(tmp_path, capsys, field, value):
    # Reason: the no-failed-cases branch is the one that renders the safety case count.
    event, _, _, _ = archived_round(tmp_path, decision="kept" if field == "cases" else "reverted")
    if field in {"empathy", "relevance"}:
        path = tmp_path / "scores/round-07-train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0][field] = value
        write_jsonl(path, rows)
    elif field in {"seen_in", "out_of"}:
        path = tmp_path / event.records["pattern"].path
        data = json.loads(path.read_text())
        data[field] = value
        write_json(path, data)
        _change_raw_log(tmp_path, ["records", "pattern", "sha256"], sha256_text(path.read_text()))
        _change_raw_log(tmp_path, ["pattern", field], value)
    else:
        _change_raw_log(tmp_path, ["attempt"] if field == "attempt" else ["safety", "cases"], value,
                        copy=field == "attempt")
    report = _report_or_safe_log_refusal(tmp_path, capsys,
                                       allow_log_refusal=field not in {"empathy", "relevance"})
    if report is not None:
        assert CLIENT_LINE not in report and HELPER_LINE not in report
        # Amendment: counts above 10**9 are unreadable too; every value in this matrix is invalid.
        assert any(word in report.lower() for word in ("unreadable", "could not", "cannot", "not valid", "left out"))


# Reason: the view's new root-tag write must not turn a completed round into an MLflow exception.
def test_round_survives_root_tag_logging_failure(training_round, monkeypatch):
    original = mlflow.set_tag

    def fail_root(key, value, *args, **kwargs):
        if key == "root":
            raise RuntimeError("offline root-tag write failed")
        return original(key, value, *args, **kwargs)

    monkeypatch.setattr(mlflow, "set_tag", fail_root)
    _, state, _ = training_round()
    assert state["event"].outcome == "reverted"


# Reason: rule 1 fixes the exact name alphabet and length, including the two boundaries.
@pytest.mark.parametrize("value,readable", [
    ("A", True), ("A" * 200, True), ("A._:/@+=#~-9", True),
    ("", False), ("A" * 201, False), ("_first", False), ("two words", False),
    ("line\n", False), ("é", False), ("name?query", False),
])
def test_name_shape_boundaries_reach_judge_summary(tmp_path, value, readable):
    _, _, records, _ = archived_round(tmp_path)
    write_jsonl(tmp_path / "scores/round-07-train.jsonl", [score(records[0], version=value)])
    summary = node_summary("judge", {"round": 7}, {"unscored": []}, tmp_path)
    assert summary["judge_version"] == (value if readable else "unreadable")
    assert summary["mean_empathy"] == 2 and summary["mean_relevance"] == 4


# Reason: raw numbers at the limits remain readable; immediately outside them they cannot enter a value or total.
@pytest.mark.parametrize("kind,minimum,maximum", [
    ("score", 1, 5), ("noise", 0, 5), ("count", 0, 10 ** 9),
    ("tokens", 0, 10 ** 12), ("cost", 0, 10 ** 6),
])
@pytest.mark.parametrize("edge", ["minimum", "maximum", "below", "above"])
def test_number_range_boundaries_in_reports(tmp_path, capsys, kind, minimum, maximum, edge):
    archived_round(tmp_path, decision="kept")
    value = {"minimum": minimum, "maximum": maximum, "below": minimum - 1, "above": maximum + 1}[edge]
    readable = edge in {"minimum", "maximum"}
    if kind in {"tokens", "cost"}:
        row = {"event": "done", "round": 7, "kind": "model", "agent": "range-agent",
               "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.25}
        row["tokens_in" if kind == "tokens" else "cost_usd"] = value
        with (tmp_path / "log/usage.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    else:
        paths = {"score": ["scores", "before", "quality"], "noise": ["scores", "noise"],
                 "count": ["safety", "cases"]}
        _change_raw_log(tmp_path, paths[kind], value)
    report = _report_or_safe_log_refusal(tmp_path, capsys, allow_log_refusal=kind not in {"tokens", "cost"})
    if report is None:
        assert not readable
        return
    if kind in {"tokens", "cost"}:
        cost = section(report, "Cost")
        rows = [row for row in table_rows(cost) if "range-agent" in row]
        assert bool(rows) is readable
        total = row_for(cost, "total")
        if readable:
            assert float(rows[0][2 if kind == "tokens" else 4]) == value
            assert float(total[2 if kind == "tokens" else 4]) == (31 if kind == "tokens" else 0.65) + value
        else:
            assert total[1:] == ["1", "31", "17", "0.6500"]
            assert "left out" in cost
    else:
        gate = section(report, "Gate")
        label = {"score": "Quality before", "noise": "Noise"}.get(kind)
        expected = f"{value:.2f}" if readable else "unreadable"
        if label:
            assert f"{label}: {expected}" in gate
        elif readable and value == 0:
            # Reason: rule 7 requires the explicit no-case sentence instead of the old count wording.
            assert "No safety case was run." in gate
            assert "No safety case failed" not in gate
        else:
            assert f"({value if readable else 'unreadable'} cases," in gate


# Reason: version 4's evidence, safety flags, comparison count and source quote must not add spoken text to the view.
@pytest.mark.parametrize("version", [3, 4])
def test_record_versions_preserve_report_and_hide_new_evidence_fields(tmp_path, version):
    event, _, records, _ = archived_round(tmp_path)
    expected = build_report(tmp_path, 7)
    for kind, ref in event.records.items():
        path = tmp_path / ref.path
        data = json.loads(path.read_text())
        data["schema_version"] = version
        if kind == "pattern":
            if version == 4:
                data["seen_in_high"] = 1
            else:
                data.pop("seen_in_high", None)
        if kind == "research":
            for source in data["sources"]:
                if version == 4:
                    source["quote"] = HELPER_LINE
                else:
                    source.pop("quote", None)
        write_json(path, data)
        _change_raw_log(tmp_path, ["records", kind, "sha256"], sha256_text(path.read_text()))
    path = tmp_path / "scores/round-07-train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["schema_version"] = version
        if version == 4:
            row.update(evidence=CLIENT_LINE, safety_flags=["role_break"])
        else:
            row.pop("evidence", None)
            row.pop("safety_flags", None)
    write_jsonl(path, rows)
    assert build_report(tmp_path, 7) == expected
    summary = node_summary("judge", {"round": 7, "records": records}, {"unscored": []}, tmp_path)
    assert summary["scored"] == 2 and summary["mean_empathy"] == summary["mean_relevance"] == 3
    assert CLIENT_LINE not in json.dumps(summary) and HELPER_LINE not in json.dumps(summary)


# Reason: the new gate-metric write must be best effort independently of existing graph metrics.
def test_round_survives_view_metric_logging_failure(training_round, monkeypatch):
    original = mlflow.log_metrics
    failures = []

    def fail_view(metrics, *args, **kwargs):
        if "kept" in metrics:
            failures.append(dict(metrics))
            raise RuntimeError(CLIENT_LINE)
        return original(metrics, *args, **kwargs)

    monkeypatch.setattr(mlflow, "log_metrics", fail_view)
    root, state, _ = training_round(measured=True)
    assert failures and state["event"].outcome == "reverted"
    assert DecisionLog(root).rounds()[-1] == state["event"]


# Reason: rule 2 requires a count of unreadable numbers, as well as suppressing their values.
def test_unreadable_logged_measurement_is_counted_in_report(tmp_path):
    archived_round(tmp_path)
    _change_raw_log(tmp_path, ["scores", "before", "quality"], True)
    report = build_report(tmp_path, 7)
    gate = section(report, "Gate")
    assert "Quality before: unreadable" in gate
    # Reason: the contract fixes neither the count line's location nor whether the count is spelled out.
    assert any(re.search(r"\b(?:1|one)\b", line, re.IGNORECASE)
               and re.search(r"unreadable|could not be read|not valid|left out", line, re.IGNORECASE)
               for line in report.splitlines())


# Reason: a stored question mark is an invalid name, even when it equals the missing-writer placeholder.
def test_stored_writer_placeholder_still_obeys_name_rule(tmp_path):
    event, _, _, _ = archived_round(tmp_path)
    ref = event.records["pattern"]
    path = tmp_path / ref.path
    data = json.loads(path.read_text())
    data["written_by"] = "?"
    write_json(path, data)
    _change_raw_log(tmp_path, ["records", "pattern", "sha256"], sha256_text(path.read_text()))
    memory = section(build_report(tmp_path, 7), "Memory")
    assert any(ref.id in line and "unreadable" in line for line in memory.splitlines())
    assert "?" not in memory


# Reason: an absent legacy action must not invent an add when the log records a rewording.
def test_missing_record_action_does_not_override_logged_action(tmp_path):
    event, _, _, _ = archived_round(tmp_path, action="reword")
    ref = event.records["edit"]
    path = tmp_path / ref.path
    data = json.loads(path.read_text())
    data["action"] = None
    write_json(path, data)
    _change_raw_log(tmp_path, ["records", "edit", "sha256"], sha256_text(path.read_text()))
    report = build_report(tmp_path, 7)
    proposer = row_for(section(report, "Nodes"), "proposer")
    assert "add C1" not in proposer
    assert any("reword" in cell or "unreadable" in cell for cell in proposer)
    assert f"- C1. {OLD_RULE}" in report and f"+ C1. {NEW_RULE}" in report
