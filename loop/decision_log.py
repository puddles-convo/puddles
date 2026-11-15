"""The append-only decision log: a JSON Lines file of round and memory-copy events."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Union

from pydantic import Field, TypeAdapter, ValidationError

from schemas import MemoryCopyEvent, RoundEvent, Scores

Event = Union[RoundEvent, MemoryCopyEvent]

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(
    Annotated[Event, Field(discriminator="event")]
)

_COLUMNS = ["Round", "Outcome", "Pattern", "Edit", "Quality", "Safety", "Policy in force", "Reason"]


class DecisionLogError(Exception):
    """The log is malformed, or an append would break its rules."""


def append_event(path: str | os.PathLike[str], event: Event) -> None:
    """Append one event as one line; refuse a second round event for the same round."""
    path = Path(path)
    if isinstance(event, RoundEvent):
        for existing in read_events(path):
            if isinstance(existing, RoundEvent) and existing.round == event.round:
                raise DecisionLogError(f"round {event.round} is already in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = event.model_dump_json() + "\n"
    with path.open("ab") as f:
        if f.tell() > 0 and not _ends_with_newline(path):
            line = "\n" + line
        f.write(line.encode("utf-8"))


def _ends_with_newline(path: Path) -> bool:
    """Whether the last byte of a non-empty file is a newline."""
    with path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


def read_events(path: str | os.PathLike[str]) -> list[Event]:
    """Return the events in file order; a missing file gives an empty list."""
    path = Path(path)
    if not path.exists():
        return []
    events: list[Event] = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                events.append(_EVENT_ADAPTER.validate_json(line))
            except ValidationError as exc:
                raise DecisionLogError(f"{path}: line {number}: {exc}") from exc
    return events


def copy_status(events: Iterable[Event]) -> dict[str, str]:
    """Map each record ID to its status after replaying the memory_copy events."""
    status: dict[str, str] = {}
    for event in events:
        if isinstance(event, MemoryCopyEvent) and status.get(event.record_id) != "confirmed":
            status[event.record_id] = event.status
    return status


def pending_copies(events: Iterable[Event]) -> list[str]:
    """Record IDs whose status is pending, in order of first appearance."""
    return [record_id for record_id, status in copy_status(events).items() if status == "pending"]


def render_table(events: Iterable[Event]) -> str:
    """A Markdown table with one row per round event, in round order."""
    rounds = sorted((e for e in events if isinstance(e, RoundEvent)), key=lambda e: e.round)
    lines = [_row(_COLUMNS), _row(["---"] * len(_COLUMNS))]
    lines.extend(_row(_cells(e)) for e in rounds)
    return "\n".join(lines)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _cells(event: RoundEvent) -> list[str]:
    """The table cells for one round; a missing part shows as '-'."""
    pattern = f"{event.pattern.name} ({event.pattern.id})" if event.pattern else "-"
    edit = f"{event.edit.id}: {event.edit.action} {event.edit.rule_id}" if event.edit else "-"
    if event.scores:
        quality = f"{_quality(event.scores.before)} -> {_quality(event.scores.after)}"
    else:
        quality = "-"
    if event.safety is None:
        safety = "-"
    elif event.safety.passed:
        safety = f"passed ({event.safety.cases} cases)"
    else:
        failed = ", ".join(event.safety.failed_case_ids)
        safety = f"failed ({failed})" if failed else "failed"
    cells = [str(event.round), event.outcome, pattern, edit, quality, safety,
             event.policy.in_force, event.reason or "-"]
    return [_escape(c) for c in cells]


def _quality(scores: Scores) -> str:
    """Quality to two places, computed from empathy and relevance when absent."""
    value = scores.quality if scores.quality is not None else (scores.empathy + scores.relevance) / 2
    return f"{value:.2f}"


def _escape(text: str) -> str:
    """Keep a cell on one line and its pipes from splitting the row."""
    return " ".join(text.split()).replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    """Print the decision table for the log at the given path."""
    parser = argparse.ArgumentParser(prog="python -m loop.decision_log")
    parser.add_argument("path")
    args = parser.parse_args(argv)
    try:
        events = read_events(args.path)
    except (DecisionLogError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_table(events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
