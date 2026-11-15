"""The per-turn line and the session file, written only for a session the person agreed to keep.

The line is a new format. It lives here, not in `schemas.py`, until it is agreed on issue #1.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import model_validator

from puddles.inference.crisis import CheckStatus
from schemas import CheckedMessage, ConversationRecord, HelperReply, Strict, Turn

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class TurnLine(Strict):
    """One turn as it happened. A check that did not answer has no CheckedMessage, never `none`."""

    session_id: str
    turn: int
    user_text: str
    keyword_matched: bool
    check_status: CheckStatus
    checked: Optional[CheckedMessage] = None
    reply: HelperReply
    timings_ms: dict[str, int] = {}

    @model_validator(mode="after")
    def _checked_only_when_answered(self) -> "TurnLine":
        if (self.check_status == "answered") != (self.checked is not None):
            raise ValueError("checked must be set exactly when check_status is 'answered'")
        return self


def _check_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ValueError(f"invalid session ID {session_id!r}: letters, digits, '_' and '-' only")
    return session_id


def session_path(root: Path, session_id: str) -> Path:
    """<root>/sessions/<session_id>.jsonl. The ID is checked so it cannot point outside sessions/."""
    return Path(root) / "sessions" / f"{_check_id(session_id)}.jsonl"


def record_path(root: Path, session_id: str) -> Path:
    """<root>/sessions/<session_id>.record.json."""
    return Path(root) / "sessions" / f"{_check_id(session_id)}.record.json"


def append_line(root: Path, line: TurnLine) -> None:
    """Add one JSON line, creating the folder; each turn is saved as it happens."""
    path = session_path(root, line.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.model_dump_json() + "\n")


def read_lines(root: Path, session_id: str) -> list[TurnLine]:
    """The lines in order; a session with no file gives an empty list."""
    path = session_path(root, session_id)
    if not path.exists():
        return []
    return [TurnLine.model_validate_json(raw) for raw in path.read_text(encoding="utf-8").splitlines()
            if raw.strip()]


def to_conversation_record(lines: list[TurnLine], *, policy_version: str, remember_me: bool,
                           consent_for_training: bool, now: datetime | None = None) -> ConversationRecord:
    """A live record of what was spoken: the person's text, then the released reply, per line."""
    if not lines:
        raise ValueError("a conversation record needs at least one turn")
    session_id = lines[0].session_id
    if any(line.session_id != session_id for line in lines):
        raise ValueError("the lines come from more than one session")
    turns: list[Turn] = []
    for line in lines:
        turns.append(Turn(speaker="user", text=line.user_text))
        turns.append(Turn(speaker="helper", text=line.reply.text))
    return ConversationRecord(id=session_id, written_by="helper", created_at=now or datetime.now(timezone.utc),
                              source="live", scenario_id=None, policy_version=policy_version, turns=turns,
                              remember_me=remember_me, consent_for_training=consent_for_training,
                              timings_ms=[dict(line.timings_ms) for line in lines])
