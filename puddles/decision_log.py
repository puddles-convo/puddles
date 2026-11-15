"""Reader and appender for the append-only decision log (log/decisions.jsonl).

Code builds every history list from here. Patterns are keyed by their identity, the ID of
the first sighting; a later sighting has its own ID and points to the first one.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from puddles.records import load_record
from schemas import (
    HISTORY_RECENT,
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


class DecisionLogError(RuntimeError):
    pass


@dataclass
class PatternHistory:
    brief: PatternBrief
    last_seen: int


def budgeted(relevant: list, others: list, recent: int = HISTORY_RECENT) -> tuple[list, list]:
    """Everything relevant, then the `recent` newest others (still in log order), then the rest."""
    shown = others[-recent:] if recent else []
    return relevant + shown, others[:len(others) - len(shown)]


class DecisionLog:
    def __init__(self, root: Path, path: str = LOG_PATH):
        self.root = Path(root)
        self.path = self.root / path

    def events(self) -> list[Event]:
        if not self.path.exists():
            return []
        out: list[Event] = []
        for number, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except ValueError as exc:
                raise DecisionLogError(
                    f"{self.path}:{number} is not valid JSON; the log was cut short") from exc
            out.append(RoundEvent.model_validate(data) if data["event"] == "round"
                       else MemoryCopyEvent.model_validate(data))
        return out

    def rounds(self) -> list[RoundEvent]:
        return [e for e in self.events() if isinstance(e, RoundEvent)]

    def append(self, event: Event) -> None:
        if isinstance(event, RoundEvent) and any(r.round == event.round for r in self.rounds()):
            raise DecisionLogError(f"round {event.round} is already in {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(event.model_dump_json() + "\n")

    def next_round(self) -> int:
        return max((r.round for r in self.rounds()), default=0) + 1

    def ref(self, record_id: str) -> Optional[RecordRef]:
        """The log pointer to a record file."""
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

    def pattern_identity(self, pattern_id: str) -> Optional[str]:
        """The identity of the pattern a sighting ID belongs to, or None if the log has no such sighting."""
        for r in self.rounds():
            if r.pattern and pattern_id in (r.pattern.id, r.pattern.pattern_id):
                return r.pattern.pattern_id
        return None

    def identity_by_name(self, name: str) -> Optional[str]:
        """The identity of a logged pattern with this name, ignoring case and spacing."""
        key = " ".join(name.lower().split())
        for r in self.rounds():
            if r.pattern and " ".join(r.pattern.name.lower().split()) == key:
                return r.pattern.pattern_id
        return None

    def all_edits(self) -> list[EditSummary]:
        out = []
        for r in self.rounds():
            if r.edit and r.outcome in ("kept", "reverted"):
                edit = load_record(r.records["edit"], PolicyEdit, self.root) if "edit" in r.records else None
                out.append(EditSummary(id=r.edit.id, pattern_id=r.pattern.pattern_id if r.pattern else None,
                                       action=r.edit.action, rule_id=r.edit.rule_id, outcome=r.outcome,
                                       summary=edit.summary if edit else (r.edit.rule_text or "")))
        return out

    def earlier_edits(self, pattern_id: Optional[str] = None,
                      recent: int = HISTORY_RECENT) -> tuple[list[EditSummary], str]:
        """Every edit for this pattern, then the most recent others, and a line counting the rest."""
        edits = self.all_edits()
        shown, hidden = budgeted([e for e in edits if e.pattern_id == pattern_id],
                                 [e for e in edits if e.pattern_id != pattern_id], recent)
        if not hidden:
            return shown, ""
        kept = sum(e.outcome == "kept" for e in hidden)
        return shown, (f"{len(hidden)} older edits are not shown: {kept} kept, "
                       f"{len(hidden) - kept} reverted")

    def reverted_edits(self) -> list[RoundEvent]:
        return [r for r in self.rounds() if r.edit and r.outcome == "reverted"]

    def queries_by_pattern(self) -> list[tuple[str, str]]:
        """Every (pattern identity, query) run, in log order, without repeats."""
        out: list[tuple[str, str]] = []
        for r in self.rounds():
            if "research" not in r.records:
                continue
            note = load_record(r.records["research"], ResearchNote, self.root)
            if note is None:
                continue
            pid = r.pattern.pattern_id if r.pattern else note.pattern_id
            for q in note.queries + ([note.query] if note.query else []):
                if q not in [seen for _, seen in out]:
                    out.append((pid, q))
        return out

    def all_queries(self) -> list[str]:
        return [q for _, q in self.queries_by_pattern()]

    def earlier_queries(self, pattern_id: Optional[str] = None,
                        recent: int = HISTORY_RECENT) -> tuple[list[str], str]:
        """Every query for this pattern, then the most recent others, and a line counting the rest."""
        pairs = self.queries_by_pattern()
        shown, hidden = budgeted([q for p, q in pairs if p == pattern_id],
                                 [q for p, q in pairs if p != pattern_id], recent)
        return shown, f"{len(hidden)} older queries are not shown" if hidden else ""

    def pattern_histories(self) -> list[PatternHistory]:
        """One entry per pattern identity, with its status from replaying the kept edits.

        fixed: a rule linked to it is active. recurring: seen again after that rule was kept.
        open: no active rule is linked to it.
        """
        active: dict[str, str] = {}  # rule_id -> pattern identity
        kept_round: dict[str, int] = {}
        found: dict[str, PatternHistory] = {}
        for r in self.rounds():
            if r.pattern:
                pid = r.pattern.pattern_id
                if pid in found:
                    found[pid].brief.sightings += 1
                    found[pid].last_seen = r.round
                else:
                    first = self.ref(pid)
                    record = load_record(first, FailurePattern, self.root) if first else None
                    found[pid] = PatternHistory(
                        PatternBrief(id=pid, name=r.pattern.name,
                                     description=record.description if record else None),
                        last_seen=r.round)
            if r.outcome == "kept" and r.edit and r.pattern:
                if r.edit.action == "remove":
                    active.pop(r.edit.rule_id, None)
                else:
                    active[r.edit.rule_id] = r.pattern.pattern_id
                    kept_round[r.pattern.pattern_id] = r.round
        for pid, h in found.items():
            rules = [rid for rid, p in active.items() if p == pid]
            if not rules:
                h.brief.status, h.brief.rule_id = "open", None
            else:
                h.brief.rule_id = rules[-1]
                h.brief.status = "recurring" if h.last_seen > kept_round[pid] else "fixed"
        return list(found.values())

    def earlier_patterns(self) -> list[PatternBrief]:
        """Every pattern in the log, one per identity. Use critic_patterns for a prompt."""
        return [h.brief for h in self.pattern_histories()]

    def critic_patterns(self, recalled: Optional[set[str]] = None,
                        recent: int = HISTORY_RECENT) -> tuple[list[PatternBrief], str]:
        """Every fixed or recurring pattern, every open one recall found, the most recently seen open
        ones, and a line counting the open patterns left out."""
        histories = self.pattern_histories()
        newest_open = sorted((h for h in histories if h.brief.status == "open"),
                             key=lambda h: -h.last_seen)[:recent]
        keep = {h.brief.id for h in newest_open} | (recalled or set())
        shown = [h.brief for h in histories if h.brief.status != "open" or h.brief.id in keep]
        hidden = len(histories) - len(shown)
        return shown, f"{hidden} older open patterns are not shown" if hidden else ""

    def copy_events(self) -> dict[tuple[int, str], MemoryCopyEvent]:
        latest: dict[tuple[int, str], MemoryCopyEvent] = {}
        for e in self.events():
            if isinstance(e, MemoryCopyEvent):
                latest[(e.round, e.record_id)] = e
        return latest

    def pending_copies(self) -> list[MemoryCopyEvent]:
        return [e for e in self.copy_events().values() if e.status == "pending"]

    def missing_copies(self) -> list[tuple[int, RecordRef]]:
        """Records a round event points to that have no memory-copy event: a crash cut the copies short."""
        copied = self.copy_events()
        return [(r.round, ref) for r in self.rounds() for ref in r.records.values()
                if (r.round, ref.id) not in copied]
