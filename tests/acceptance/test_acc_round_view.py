"""Acceptance tests from docs/specs/round-view.md, without the view implementation.

The existing schemas and training APIs supply files, never the expected report.
Markdown assertions allow layout choices the contract leaves open.
"""
from datetime import datetime, timezone
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
    # Permit ordinary decimal precision/currency formatting, but not a substring of another number.
    suffix = r"0*" if "." in str(value) else r"(?:\.0+)?"
    pattern = rf"(?<![\d.,]){re.escape(str(value))}{suffix}(?![\d.,])"
    assert re.search(pattern, text), f"Missing numeric value {value!r}: {text}"


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
                    "checks": {"quality_gain": "passed" if decision == "kept" else "failed"},
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
    # Catch reports that sum the whole log instead of selecting the requested round.
    other = Meter(root, prices, round=round_ + 1)
    other.done(other.sent("other-round-agent", "offline-priced", 991, 999), CallUsage(991, 999))
    return event, objects, records, checks


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


def test_judge_summary_averages_scored_only_and_reads_file_version(tmp_path):
    records = [conversation(3, name) for name in ("a", "b", "c")]
    write_jsonl(tmp_path / "scores/round-03-train.jsonl", [score(records[0]), score(records[1], 4, 4)])
    summary = node_summary("judge", {"round": 3, "records": records}, {
        "scored": [scored(records[0]), scored(records[1], 4, 4)], "unscored": [records[2].id],
    }, tmp_path)
    assert summary["scored"] == 2 and summary["unscored"] == 1
    assert summary["mean_empathy"] == 3 and summary["mean_relevance"] == 4
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
    summary = node_summary("finish", {"round": 7}, {"event": event}, tmp_path)
    assert summary["outcome"] == "reverted" and summary["policy_in_force"] == "policy-v1"
    assert {(copy["record_id"], copy["status"]) for copy in summary["copies"]} == {
        (ref.id, "pending" if kind == "research" else "confirmed") for kind, ref in event.records.items()
    }


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
                report = build_report(root, 1).lower()
                assert "span" in report, "The fallback must be disclosed in the report"
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
    optional_report = Path("reports/round-07.md")
    assert {key: value for key, value in after.items() if key != optional_report} == before
    if optional_report in after:
        assert after[optional_report] == report.encode("utf-8")


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


def test_report_critic_column_distinguishes_named_and_unnamed_conversations(tmp_path):
    _, _, records, _ = archived_round(tmp_path)
    conversations = section(build_report(tmp_path, 7), "Conversations")
    headers = next(row for row in table_rows(conversations)
                   if any("critic" in cell.lower() for cell in row))
    critic = next(i for i, cell in enumerate(headers) if "critic" in cell.lower())
    named = row_for(conversations, records[0].id)[critic]
    unnamed = row_for(conversations, records[1].id)[critic]
    assert named and named != unnamed


def test_report_critic_column_is_empty_without_pattern_or_scores_file(tmp_path):
    base = policy_file(tmp_path)
    record = conversation(1, "unscored")
    write_jsonl(tmp_path / "conversations/round-01-train.jsonl", [record])
    DecisionLog(tmp_path).append(RoundEvent(
        round=1, outcome="no_pattern", reason="The judge scored no conversation.",
        policy=LogPolicy(base="policy-v1", base_sha256=sha256_text(base.read_text()), in_force="policy-v1"),
    ))
    conversations = section(build_report(tmp_path, 1), "Conversations")
    headers = next(row for row in table_rows(conversations)
                   if any("critic" in cell.lower() for cell in row))
    critic = next(i for i, cell in enumerate(headers) if "critic" in cell.lower())
    assert row_for(conversations, record.id)[critic] == ""


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
        # Missing scores cannot become an apparent zero gain or zero noise.
        assert not re.search(r"\b(?:quality|noise)[^\n|]*[:=]\s*0(?:\.0*)?\b", gate, re.I)


def test_report_cost_is_per_agent_and_only_for_requested_round(tmp_path):
    archived_round(tmp_path)
    cost = section(build_report(tmp_path, 7), "Cost")
    agent = " ".join(row_for(cost, "acceptance-agent"))
    for value in (1, 31, 17, 0.65):
        has_number(agent, value)
    assert "other-round-agent" not in cost


@pytest.mark.parametrize("mode,not_run", [
    ("no_pattern", ("researcher", "proposer", "gate")),
    ("no_evidence", ("proposer", "gate")),
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


def test_stand_in_report_explicitly_says_nothing_was_measured(training_round):
    root, _, _ = training_round()
    gate = section(build_report(root, 1), "Gate").lower()
    assert len([line for line in gate.splitlines() if line.strip()]) == 1
    assert "stand-in" in gate
    assert "measur" in gate and any(word in gate for word in ("not ", "nothing", "no ", "unmeasured"))


# Section 3 CLI: artifact destination, defaults and error behavior.


def test_show_round_without_matching_run_still_writes_prints_and_preserves_inputs(tmp_path, capsys):
    root = tmp_path / "archive"
    archived_round(root)
    before = snapshot(root)
    expected = build_report(root, 7)
    assert show_round.main(["--root", str(root), "--round", "7"]) == 0
    output = capsys.readouterr()
    assert expected in output.out
    assert report_path(root, 7).read_text(encoding="utf-8") == expected
    notice = output.out.replace(expected, "", 1).strip().splitlines()
    assert len(notice) == 1
    assert "attach" in notice[0].lower()
    assert any(word in notice[0].lower() for word in ("not ", "no ", "skip", "unattach", "couldn't"))
    after = snapshot(root)
    assert {key: value for key, value in after.items() if key != Path("reports/round-07.md")} == before


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
        target = client.create_run(experiment, tags={"mlflow.runName": "round-07"})
        other = client.create_run(experiment, tags={"mlflow.runName": "round-08"})
        unrelated_experiment = client.create_experiment(
            "unrelated", artifact_location=(tmp_path / f"{name}-unrelated").as_uri())
        unrelated = client.create_run(unrelated_experiment, tags={"mlflow.runName": "round-07"})
        client.set_terminated(target.info.run_id)
        client.set_terminated(other.info.run_id)
        client.set_terminated(unrelated.info.run_id)
        clients[name] = (uri, client, target.info.run_id, other.info.run_id, unrelated.info.run_id)
    monkeypatch.setattr(tracing, "DEFAULT_TRACKING_URI", clients["default"][0])
    if source != "default":
        monkeypatch.setenv("MLFLOW_TRACKING_URI", clients["environment"][0])
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
