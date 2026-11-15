"""Reader and appender for the append-only decision log (log/decisions.jsonl).

A minimal version so the agents can build their complete lists from the log. The full
decision log, with the replay check, is task A5 (#13) and can replace this module.
"""
import json
from pathlib import Path
from typing import Optional, Union

from puddles.records import load_record
from schemas import (
    EditSummary,
    FailurePattern,
    MemoryCopyEvent,
    PatternBrief,
    PolicyEdit,
    RecordRef,
    ResearchNote,
    RoundEvent,
)

LOG_PATH = "log/decisions.jsonl"
Event = Union[RoundEvent, MemoryCopyEvent]


class DecisionLog:
    def __init__(self, root: Path, path: str = LOG_PATH):
        self.root = Path(root)
        self.path = self.root / path

    def events(self) -> list[Event]:
        if not self.path.exists():
            return []
        out: list[Event] = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                out.append(RoundEvent.model_validate(data) if data["event"] == "round"
                           else MemoryCopyEvent.model_validate(data))
        return out

    def rounds(self) -> list[RoundEvent]:
        return [e for e in self.events() if isinstance(e, RoundEvent)]

    def append(self, event: Event) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(event.model_dump_json() + "\n")

    def next_round(self) -> int:
        return max((r.round for r in self.rounds()), default=0) + 1

    def ref(self, record_id: str) -> Optional[RecordRef]:
        """The latest log pointer to a record file."""
        found = None
        for r in self.rounds():
            for ref in r.records.values():
                if ref.id == record_id:
                    found = ref
        return found

    def refs_of_kind(self, kind: str) -> list[RecordRef]:
        return [r.records[kind] for r in self.rounds() if kind in r.records]

    def used_rule_ids(self) -> set[str]:
        return {r.edit.rule_id for r in self.rounds() if r.edit}

    def earlier_edits(self) -> list[EditSummary]:
        out = []
        for r in self.rounds():
            if r.edit and r.outcome in ("kept", "reverted"):
                edit = load_record(r.records["edit"], PolicyEdit, self.root) if "edit" in r.records else None
                out.append(EditSummary(id=r.edit.id, action=r.edit.action, rule_id=r.edit.rule_id,
                                       outcome=r.outcome,
                                       summary=edit.summary if edit else (r.edit.rule_text or "")))
        return out

    def reverted_edits(self) -> list[RoundEvent]:
        return [r for r in self.rounds() if r.edit and r.outcome == "reverted"]

    def earlier_queries(self) -> list[str]:
        queries: list[str] = []
        for ref in self.refs_of_kind("research"):
            note = load_record(ref, ResearchNote, self.root)
            for q in (note.queries if note else []) + ([note.query] if note and note.query else []):
                if q not in queries:
                    queries.append(q)
        return queries

    def earlier_patterns(self) -> list[PatternBrief]:
        """Every pattern in the log with its status, from replaying the kept edits.

        fixed: a rule linked to it is active. recurring: named again after its rule was kept.
        open: no active rule is linked to it.
        """
        active: dict[str, str] = {}  # rule_id -> pattern_id
        kept_round: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        briefs: dict[str, PatternBrief] = {}
        for r in self.rounds():
            if r.pattern:
                last_seen[r.pattern.id] = r.round
                if r.pattern.id not in briefs:
                    record = load_record(r.records["pattern"], FailurePattern, self.root) \
                        if "pattern" in r.records else None
                    briefs[r.pattern.id] = PatternBrief(
                        id=r.pattern.id, name=r.pattern.name,
                        description=record.description if record else None)
            if r.outcome == "kept" and r.edit and r.pattern:
                if r.edit.action == "remove":
                    active.pop(r.edit.rule_id, None)
                else:
                    active[r.edit.rule_id] = r.pattern.id
                    kept_round[r.pattern.id] = r.round
        for pid, brief in briefs.items():
            rules = [rid for rid, p in active.items() if p == pid]
            if not rules:
                brief.status, brief.rule_id = "open", None
            else:
                brief.rule_id = rules[-1]
                brief.status = "recurring" if last_seen[pid] > kept_round[pid] else "fixed"
        return list(briefs.values())

    def pending_copies(self, max_attempts: int = 3) -> list[MemoryCopyEvent]:
        latest: dict[str, MemoryCopyEvent] = {}
        for e in self.events():
            if isinstance(e, MemoryCopyEvent):
                latest[e.record_id] = e
        return [e for e in latest.values() if e.status == "pending" and e.attempt < max_attempts]
