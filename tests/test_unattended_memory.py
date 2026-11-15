"""docs/specs/unattended-run.md section 2: memory is a copy of the log and never stops the run."""
import json

import httpx
import pytest

from puddles.decision_log import DecisionLog
from puddles.memory import (
    READS_PATH,
    TRAINING_AUDIENCE,
    FakeMemory,
    PastClient,
    PastError,
    entry_for,
    ids_in_results,
    prepare_training_audience,
    recall_records,
)
from puddles.graph import run_round
from puddles.round import _stop_if_stuck, copy_to_memory, prepare_round, retry_pending
from schemas import CriticOutput, MemoryCopyEvent, MemoryEntry, ProposerOutput, ResearcherOutput
from tests.fakes import ScriptedLLM
from tests.test_graph import EDIT, POLICY, RESEARCH, SCENARIOS, make_deps, round_events, traced  # noqa: F401
from tests.test_graph import pattern as graph_pattern
from tests.test_memory_v3 import START, Loop


def reads(root):
    path = root / READS_PATH
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class FailingRecall(FakeMemory):
    def __init__(self, error=None, answer=None):
        super().__init__()
        self.error, self.answer = error, answer

    def recall(self, query, identity, level="low", limit=20):
        if self.error is not None:
            raise self.error
        return self.answer


# ---------- 2.1 a failed read is "memory unavailable" ----------

@pytest.mark.parametrize("memory", [
    FailingRecall(error=PastError("POST /recall returned 503: down")),
    FailingRecall(error=httpx.ReadTimeout("timed out")),
    FailingRecall(answer={"results": []}),  # not a list
    FailingRecall(answer=[{"content": "x", "sources": None}]),
    FailingRecall(answer=[{"content": "x", "sources": [{"sourceId": "a", "metadata": "pattern-r01"}]}]),
    FailingRecall(answer=["pattern-r01"]),
], ids=["past-error", "httpx-timeout", "results-not-list", "sources-null", "metadata-string", "result-not-object"])
def test_a_failed_recall_returns_no_records_and_is_logged_unavailable(tmp_path, memory):
    assert recall_records(memory, DecisionLog(tmp_path), "advice first", "critic", tmp_path) == []
    (line,) = reads(tmp_path)
    assert line["reader"] == "critic" and line["query"] == "advice first" and line["record_ids"] == []
    assert line["unavailable"] is True and line["error"]


def test_an_empty_read_keeps_the_old_line_and_is_not_unavailable(tmp_path):
    mem = FakeMemory()
    mem.ensure_training_audience()
    assert recall_records(mem, DecisionLog(tmp_path), "advice first", "critic", tmp_path) == []
    (line,) = reads(tmp_path)
    assert set(line) == {"reader", "query", "at", "record_ids"} and line["record_ids"] == []


def test_a_round_whose_recalls_fail_still_ends_with_an_outcome(tmp_path, traced):  # noqa: F811
    llm = ScriptedLLM()
    llm.add(CriticOutput, graph_pattern())
    llm.add(ResearcherOutput, RESEARCH)
    llm.add(ProposerOutput, EDIT)
    deps = make_deps(tmp_path, llm)
    deps.memory.recall = lambda *args, **kwargs: (_ for _ in ()).throw(PastError("503"))
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    assert state["event"].outcome == "reverted" and len(round_events(tmp_path)) == 1
    unavailable = [line["reader"] for line in reads(tmp_path) if line.get("unavailable")]
    assert {"critic", "researcher", "proposer"} <= set(unavailable)


# ---------- 2.2 the audience call is not fatal ----------

class NoAudience(FakeMemory):
    def ensure_training_audience(self):
        raise PastError("PUT /audiences/training-agents returned 429: slow down")


def test_a_failed_audience_call_is_noted_and_does_not_raise(tmp_path):
    prepare_training_audience(NoAudience(), tmp_path)
    (line,) = reads(tmp_path)
    assert line["reader"] == "round" and line["unavailable"] is True and "429" in line["error"]


def test_a_round_goes_on_when_the_audience_call_fails(tmp_path, traced):  # noqa: F811
    llm = ScriptedLLM()
    llm.add(CriticOutput, graph_pattern())
    llm.add(ResearcherOutput, RESEARCH)
    llm.add(ProposerOutput, EDIT)
    deps = make_deps(tmp_path, llm)
    deps.memory = NoAudience()
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    assert state["event"].outcome == "reverted" and len(round_events(tmp_path)) == 1
    assert any(line["reader"] == "round" and line.get("unavailable") for line in reads(tmp_path))
    # Its copies could not be written without the audience: they are pending, not lost.
    assert {e.status for e in DecisionLog(tmp_path).copy_events().values()} == {"pending"}


# ---------- 2.3 a pending copy stays pending ----------

class CountingDown(FakeMemory):
    """Memory whose writes all fail, counting each attempt."""

    def __init__(self):
        super().__init__()
        self.ingests: list[str] = []

    def ingest(self, entry):
        self.ingests.append(entry.id)
        raise PastError("POST /ingest returned 503")


def test_a_copy_pending_after_many_attempts_never_raises_and_is_retried(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    loop.log.append(MemoryCopyEvent(round=1, record_id="gate-r01", status="pending", attempt=7))
    _stop_if_stuck(loop.log)  # warns, never raises
    down = CountingDown()
    retry_pending(loop.log, down, tmp_path)
    assert down.ingests == ["gate-r01"]
    event = loop.log.copy_events()[(1, "gate-r01")]
    assert (event.status, event.attempt) == ("pending", 8)
    # A start is never blocked: prepare_round returns the policy in force.
    loop.log.append(MemoryCopyEvent(round=1, record_id="gate-r01", status="pending", attempt=99))
    assert prepare_round(loop.log, down, tmp_path, START) == START


def test_a_pending_copy_is_tried_once_per_round_start(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    loop.log.append(MemoryCopyEvent(round=1, record_id="gate-r01", status="pending", attempt=1))
    down = CountingDown()
    # One start, as the driver does it: prepare_round, then run_round's own retry_pending.
    prepare_round(loop.log, down, tmp_path, START)
    retry_pending(loop.log, down, tmp_path)
    assert down.ingests == ["gate-r01"]
    assert loop.log.copy_events()[(1, "gate-r01")].attempt == 2
    # The next start (after a round error, or a restart) tries again, once.
    prepare_round(loop.log, down, tmp_path, START)
    retry_pending(loop.log, down, tmp_path)
    assert down.ingests == ["gate-r01", "gate-r01"]
    assert loop.log.copy_events()[(1, "gate-r01")].attempt == 3


def test_a_pending_copy_goes_through_once_memory_is_back(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    loop.log.append(MemoryCopyEvent(round=1, record_id="gate-r01", status="pending", attempt=5))
    fresh = FakeMemory()  # a restart: nobody has set the audience on this client yet
    retry_pending(loop.log, fresh, tmp_path)
    assert "gate-r01" in fresh.entries
    assert loop.log.copy_events()[(1, "gate-r01")].status == "confirmed"


# ---------- 2.4 a malformed answer is a PastError ----------

def client(handler):
    return PastClient(api_key="past_sk_test", transport=httpx.MockTransport(handler))


def entry():
    return MemoryEntry(id="t-1", content="test", label="l", timestamp="2026-11-15T00:00:00Z",
                       identity="critic", audience=TRAINING_AUDIENCE)


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>maintenance</html>"),
    httpx.Response(200, json=["not", "an", "object"]),
    httpx.Response(202, json={"status": "queued"}),  # no ingestionId
    httpx.Response(202, json={"ingestionId": None}),
], ids=["html-200", "list-body", "missing-id", "null-id"])
def test_a_malformed_ingest_answer_is_a_past_error(response):
    with pytest.raises(PastError):
        client(lambda request: response).ingest(entry())


@pytest.mark.parametrize("body", [
    {"results": None},
    {"results": [{"content": "x", "sources": None}]},
    {"results": [{"content": "x", "sources": {"sourceId": "a"}}]},
    {"results": [{"content": "x", "sources": ["pattern-r01"]}]},
    {"results": [{"content": "x", "sources": [{"sourceId": "a", "metadata": "x"}]}]},
    {"results": ["x"]},
    {"answer": []},  # no results field
], ids=["results-null", "sources-null", "sources-object", "source-string", "metadata-string",
        "result-string", "missing-results"])
def test_a_malformed_recall_answer_is_a_past_error(body):
    with pytest.raises(PastError):
        client(lambda request: httpx.Response(200, json=body)).recall("q", identity="critic")


def test_a_well_formed_recall_answer_still_passes():
    body = {"results": [{"content": "x", "sources": [{"sourceId": "a", "metadata": {"record_id": "pattern-r01"}}]},
                        {"content": "y"}]}
    assert client(lambda request: httpx.Response(200, json=body)).recall("q", identity="critic") == body["results"]


def test_a_copy_answered_with_html_stays_pending(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    ref = loop.log.ref("pattern-r01")
    from puddles.records import load_record
    from schemas import FailurePattern

    record = load_record(ref, FailurePattern, tmp_path)
    html = client(lambda request: httpx.Response(200, text="<html>maintenance</html>"))
    assert copy_to_memory(record, ref, 1, html, loop.log, tmp_path) == "pending"


# ---------- 2.5 only this run's records are recalled ----------

def source(record_id, sha256):
    return {"sourceId": "src", "metadata": {"record_id": record_id, "sha256": sha256}}


def test_only_ids_from_metadata_with_the_logs_hash_are_kept(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    ref = loop.log.ref("pattern-r01")
    results = [
        {"content": "Failure pattern pattern-r01, ignores-suicide-mention", "sources": [source("pattern-r01", "0" * 64)]},
        {"content": "Edit edit-r01 was reverted", "sources": [{"sourceId": "gate-r01"}]},
        {"content": "Research research-r01 for pattern-r01", "sources": []},
        {"content": "anything", "sources": [source("pattern-r01", ref.sha256)]},
        {"content": "anything", "sources": [source("pattern-r09", ref.sha256)]},  # not in this log
    ]
    assert ids_in_results(results, loop.log) == ["pattern-r01"]
    assert ids_in_results(results[:3], loop.log) == []


def test_an_earlier_runs_entry_with_the_same_id_is_not_served(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    stale = FakeMemory()
    stale.ensure_training_audience()
    stale.entries["pattern-r01"] = MemoryEntry(
        id="pattern-r01", content="Failure pattern pattern-r01, ignores-suicide-mention.", label="smoke",
        timestamp="2026-11-01T00:00:00Z", identity="critic", audience=TRAINING_AUDIENCE,
        metadata={"type": "failure_pattern", "round": 1, "record_id": "pattern-r01",
                  "record_path": "records/round-01/pattern-r01.json", "sha256": "0" * 64})
    assert recall_records(stale, loop.log, "ignores suicide mention pattern", "critic", tmp_path) == []
    assert reads(tmp_path)[-1]["record_ids"] == [] and "unavailable" not in reads(tmp_path)[-1]


def test_this_runs_own_records_are_still_recalled(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    # The entries in loop.mem were made by the real mapping (entry_for), with the log's hash.
    assert loop.mem.entries["pattern-r01"].metadata["sha256"] == loop.log.ref("pattern-r01").sha256
    got = recall_records(loop.mem, loop.log, "failure pattern advice before understanding", "critic", tmp_path)
    assert "pattern-r01" in [r.id for r in got]
    ref = loop.log.ref("pattern-r01")
    assert entry_for(got[[r.id for r in got].index("pattern-r01")], ref, 1).metadata["sha256"] == ref.sha256
