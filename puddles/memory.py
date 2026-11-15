"""past.dev memory: turn records into entries, write them, confirm them, and recall them.

The decision log is the authority and past.dev is its searchable copy. A recall only finds
which record IDs are relevant; the content shown to an agent is loaded from the record files
and checked against the hash in the log.
"""
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import httpx

from schemas import (
    FROM_MEMORY_LIMIT,
    FailurePattern,
    GateOutcome,
    MemoryEntry,
    PolicyEdit,
    Record,
    RecordRef,
    ResearchNote,
)

BASE_URL = os.environ.get("PAST_BASE_URL", "https://api.past.dev/api/v1")
TRAINING_AUDIENCE = "training-agents"
TRAINING_IDENTITIES = ["critic", "researcher", "proposer", "gatekeeper"]
RECALL_QUERY_LIMIT = 1000  # past.dev answers 400 recall-query-too-long above this many characters
RECORD_ID_RE = re.compile(r"\b(?:pattern|research|edit|gate)-r\d+\b")

ENTRY_TYPES: dict[type, str] = {
    FailurePattern: "failure_pattern",
    ResearchNote: "research_note",
    PolicyEdit: "policy_edit",
    GateOutcome: "gate_outcome",
}
MODEL_BY_PREFIX: dict[str, type[Record]] = {
    "pattern": FailurePattern, "research": ResearchNote, "edit": PolicyEdit, "gate": GateOutcome,
}


class PastError(RuntimeError):
    pass


class Memory(Protocol):
    def ensure_training_audience(self) -> None: ...
    def ingest(self, entry: MemoryEntry) -> str: ...
    def wait(self, ingestion_id: str, timeout_s: float = 60) -> bool: ...
    def recall(self, query: str, identity: str, level: str = "low", limit: int = 20) -> list[dict]: ...


class PastClient:
    """Thin wrapper over the past.dev REST API (https://past.dev/docs/memory-api)."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL, timeout: float = 30,
                 transport: Optional[httpx.BaseTransport] = None):
        key = api_key or os.environ.get("PAST_API_KEY")
        if not key:
            raise PastError("PAST_API_KEY is not set")
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                  headers={"Authorization": f"Bearer {key}"})

    def _call(self, method: str, path: str, **kwargs: Any) -> dict:
        resp = self._http.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise PastError(f"{method} {path} returned {resp.status_code}: {resp.text[:300]}")
        if not resp.content:
            return {}
        try:
            body = resp.json()
        except ValueError as exc:
            raise PastError(f"{method} {path} returned a body that is not JSON: {resp.text[:300]}") from exc
        if not isinstance(body, dict):
            raise PastError(f"{method} {path} returned a {type(body).__name__}, not an object")
        return body

    def ensure_training_audience(self) -> None:
        self._call("PUT", f"/audiences/{TRAINING_AUDIENCE}", json={
            "kind": "fixed", "name": "Training agents", "identities": TRAINING_IDENTITIES})

    def ingest(self, entry: MemoryEntry) -> str:
        ingestion_id = self._call("POST", "/ingest", json=entry.model_dump(mode="json")).get("ingestionId")
        if not isinstance(ingestion_id, str) or not ingestion_id:
            raise PastError(f"POST /ingest returned no ingestionId for {entry.id}")
        return ingestion_id

    def ingestion_status(self, ingestion_id: str) -> dict:
        status = self._call("GET", f"/ingest/{ingestion_id}")
        if not isinstance(status.get("status"), str):
            raise PastError(f"GET /ingest/{ingestion_id} returned no status string")
        return status

    def wait(self, ingestion_id: str, timeout_s: float = 60, poll_s: float = 2) -> bool:
        """Poll until the send is completed. False if it is still processing at the deadline."""
        deadline = time.monotonic() + timeout_s
        retried = False
        while True:
            status = self.ingestion_status(ingestion_id)
            if status.get("status") == "completed":
                return True
            if status.get("status") == "failed":
                if retried:
                    raise PastError(f"ingestion {ingestion_id} failed: {status.get('failure')}")
                self._call("POST", f"/ingest/{ingestion_id}/retry")
                retried = True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_s)

    def recall(self, query: str, identity: str, level: str = "low", limit: int = 20) -> list[dict]:
        body = {"query": query[:RECALL_QUERY_LIMIT], "identity": identity, "level": level, "limit": limit}
        answer = self._call("POST", "/recall", json=body)
        if "results" not in answer:
            raise PastError("POST /recall returned no results field")
        return check_recall_results(answer["results"])


class FakeMemory:
    """In-process stand-in for tests and offline runs: word-overlap ranking, same audience rules."""

    def __init__(self) -> None:
        self.entries: dict[str, MemoryEntry] = {}
        self.audiences: dict[str, list[str]] = {}
        self.recalls: list[dict] = []

    def ensure_training_audience(self) -> None:
        self.audiences[TRAINING_AUDIENCE] = list(TRAINING_IDENTITIES)

    def ingest(self, entry: MemoryEntry) -> str:
        if entry.audience not in self.audiences and entry.audience != "project":
            raise PastError(f"400 audience-unknown: {entry.audience}")
        self.entries[entry.id] = entry
        return f"ing-{entry.id}"

    def wait(self, ingestion_id: str, timeout_s: float = 60) -> bool:
        return True

    def recall(self, query: str, identity: str, level: str = "low", limit: int = 20) -> list[dict]:
        query = query[:RECALL_QUERY_LIMIT]  # as PastClient sends it
        self.recalls.append({"query": query, "identity": identity})
        words = set(re.findall(r"\w+", query.lower()))
        visible = [e for e in self.entries.values()
                   if e.audience == "project" or identity in self.audiences.get(e.audience, [])]
        scored = sorted(((len(words & set(re.findall(r"\w+", e.content.lower()))), e) for e in visible),
                        key=lambda pair: -pair[0])
        return [{"rank": i + 1, "content": e.content, "occurredAt": e.timestamp.isoformat(),
                 "artifact": {"kind": "source"},
                 "sources": [{"sourceId": e.id, "label": e.label, "metadata": e.metadata}]}
                for i, (score, e) in enumerate(scored[:limit]) if score > 0]


def entry_content(record: Record, **extra: Any) -> str:
    """The searchable sentence, from the templates in docs/agents/t4-t7."""
    if isinstance(record, FailurePattern):
        same = f" Same pattern as {record.same_as_pattern_id}." if record.same_as_pattern_id else ""
        return (f"Failure pattern {record.id}, {record.name}, round {record.round}: {record.description} "
                f"Seen in {record.seen_in} of the {record.out_of} lowest-scoring conversations reviewed.{same}")
    if isinstance(record, ResearchNote):
        found = "; ".join(f"{s.technique} Source: {s.url}." for s in record.sources)
        name = extra.get("pattern_name", "")
        return f"Research {record.id} for {record.pattern_id}{', ' + name if name else ''}: {found}"
    if isinstance(record, PolicyEdit):
        return f"Edit {record.id} for {record.pattern_id}: {record.action} rule {record.rule_id}. {record.summary}"
    if isinstance(record, GateOutcome):
        if record.before.quality is None:
            return (f"Edit {record.edit_id} ({extra.get('summary', '')}) was {record.decision} in round "
                    f"{record.round} by a stand-in gate, not a real test. {record.policy_in_force} is in force.")
        if record.decision == "kept":
            safety = record.reason if extra.get("safety_cases") == 0 else "All safety cases passed."
            return (f"Edit {record.edit_id} was kept in round {record.round} and is rule {extra.get('rule_id')} "
                    f"of {record.policy_in_force}. Quality rose from {record.before.quality} to "
                    f"{record.after.quality}, above the noise of {extra.get('noise')}. {safety}")
        return (f"Edit {record.edit_id} ({extra.get('summary', '')}) was reverted in round {record.round}: "
                f"{record.reason} {record.policy_in_force} stays in force.")
    raise TypeError(f"no memory template for {type(record).__name__}")


def entry_for(record: Record, ref: RecordRef, round_: int, audience: str = TRAINING_AUDIENCE,
              **extra: Any) -> MemoryEntry:
    kind = ENTRY_TYPES[type(record)]
    return MemoryEntry(
        id=record.id,
        content=entry_content(record, **extra),
        label=f"{record.written_by}: {kind.replace('_', ' ')}, round {round_}",
        timestamp=record.created_at,
        identity=record.written_by,
        audience=audience,
        metadata={"type": kind, "round": round_, "record_id": record.id,
                  "record_path": ref.path, "sha256": ref.sha256},
    )


def check_recall_results(results: Any) -> list[dict]:
    """The recall results if they have the expected shape, else PastError: a list of objects, each
    with `sources` (if present) a list of objects, each with `metadata` (if present) an object."""
    if not isinstance(results, list):
        raise PastError(f"recall results are a {type(results).__name__}, not a list")
    for result in results:
        if not isinstance(result, dict):
            raise PastError(f"a recall result is a {type(result).__name__}, not an object")
        sources = result.get("sources", [])
        if not isinstance(sources, list):
            raise PastError(f"a recall result's sources are a {type(sources).__name__}, not a list")
        for source in sources:
            if not isinstance(source, dict):
                raise PastError(f"a recall source is a {type(source).__name__}, not an object")
            if not isinstance(source.get("metadata", {}), dict):
                raise PastError(f"a recall source's metadata is a {type(source['metadata']).__name__}, "
                                "not an object")
    return results


def ids_in_results(results: list[dict], log: Optional["DecisionLog"] = None) -> list[str]:
    """Record IDs named by the recall results' source metadata, in rank order, never by their text.

    With a log, an ID is kept only when the log holds it and the metadata's `sha256` is the log's hash
    for it, so an entry left by another run under the same ID is not taken for this run's record.
    """
    seen: list[str] = []
    for result in results:
        for source in result.get("sources", []):
            metadata = source.get("metadata", {})
            record_id = str(metadata.get("record_id", ""))
            if not RECORD_ID_RE.fullmatch(record_id) or record_id in seen:
                continue
            if log is not None:
                ref = log.ref(record_id)
                if ref is None or metadata.get("sha256") != ref.sha256:
                    continue
            seen.append(record_id)
    return seen


READS_PATH = "log/memory_reads.jsonl"


def _note_read(root: Path, line: dict) -> None:
    reads = Path(root) / READS_PATH
    reads.parent.mkdir(parents=True, exist_ok=True)
    with reads.open("a") as f:
        f.write(json.dumps(line) + "\n")


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def recall_records(memory: Memory, log: "DecisionLog", query: str, identity: str, root: Path,
                   max_records: int = FROM_MEMORY_LIMIT) -> list[Record]:
    """Recall, then load each named record from its file. IDs missing from the log are dropped.

    Each read is noted in log/memory_reads.jsonl so a run can show which agent read which record.
    A failed recall is "memory unavailable": no records, and the line says `"unavailable": true`.
    """
    line: dict[str, Any] = {"reader": identity, "query": query, "at": now().isoformat()}
    try:
        results = check_recall_results(memory.recall(query, identity=identity))
    except (PastError, httpx.HTTPError) as exc:
        _note_read(root, {**line, "record_ids": [], "unavailable": True, "error": _error_text(exc)})
        return []
    records: list[Record] = []
    for record_id in ids_in_results(results, log):
        ref = log.ref(record_id)
        model = MODEL_BY_PREFIX[record_id.split("-")[0]]
        record = load_record(ref, model, root) if ref else None
        if record is not None:
            records.append(record)
        if len(records) >= max_records:
            break
    _note_read(root, {**line, "record_ids": [r.id for r in records]})
    return records


def prepare_training_audience(memory: Memory, root: Path) -> None:
    """The audience call at a round's start. A failure is noted in the read log and the round goes on."""
    try:
        memory.ensure_training_audience()
    except (PastError, httpx.HTTPError) as exc:
        _note_read(root, {"reader": "round", "query": "ensure_training_audience", "at": now().isoformat(),
                          "record_ids": [], "unavailable": True, "error": _error_text(exc)})


def describe(record: Record) -> str:
    """One line per record, for an agent's <from_memory> block."""
    if isinstance(record, FailurePattern):
        same = f" Same pattern as {record.same_as_pattern_id}." if record.same_as_pattern_id else ""
        return (f"{record.id} (round {record.round}) {record.name}: {record.description} "
                f"Seen in {record.seen_in} of {record.out_of}.{same}")
    if isinstance(record, ResearchNote):
        return f"{record.id} for {record.pattern_id}: " + "; ".join(
            f"{s.technique} ({s.url})" for s in record.sources)
    if isinstance(record, PolicyEdit):
        return f"{record.id} for {record.pattern_id}: {record.action} {record.rule_id}. {record.summary}"
    if isinstance(record, GateOutcome):
        return (f"{record.id}: edit {record.edit_id} was {record.decision} in round {record.round}. "
                f"{record.reason}".strip())
    return str(record)


def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


from puddles.decision_log import DecisionLog  # noqa: E402  (type used above in annotations)
from puddles.records import load_record  # noqa: E402
