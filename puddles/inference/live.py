"""The notes joined to a live session: load them at the start, run the note-taker at the end.

The note-taker always runs when the session had turns, because forgetting must work whatever the
person answered. This module opens no span: whether the note-taker's calls are traced is decided by
the model the caller passes, and that follows the person's consent. Contract: docs/specs/chat-notes.md.
"""
from __future__ import annotations

import contextvars
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from puddles.inference.note_store import NoteStore, check_user_id, expires_on
from puddles.inference.notes import NotesResult, take_notes
from puddles.inference.turn import Session, end_session
from puddles.llm import LLM
from schemas import ConversationRecord


def start_notes(store: NoteStore, user_id: str, *, today: date | None = None) -> list[str]:
    """The facts the helper is given as `user_notes`, for every session whatever the person answered."""
    today = today or date.today()
    try:
        return store.helper_notes(user_id, today)
    except Exception:
        return _on_file(store, user_id, today)


def _on_file(store: NoteStore, user_id: str, today: date) -> list[str]:
    """The live notes as the files hold them, without asking past.dev. Reading changes nothing."""
    stored = store.stored(user_id)
    pending = set(store.pending_deletes(user_id))
    return [stored[n.note_id].fact for n in store.index(user_id).notes
            if n.note_id in stored and n.note_id not in pending
            and ((last := expires_on(n)) is None or today <= last)]


def tracing_off() -> None:
    """Turn off what a session with consent may have left on for the whole process.

    `tracing.setup` turns on OpenAI and LangChain autologging process-wide, and it stays on for every
    later session. This does not import MLflow: if it was never imported, nothing can be tracing.
    A later session with consent calls `tracing.setup` again.
    """
    mlflow = sys.modules.get("mlflow")
    if mlflow is None:
        return
    for flavour in ("openai", "langchain"):
        try:
            module = __import__(f"mlflow.{flavour}", fromlist=["autolog"])
        except ImportError:
            continue  # the flavour cannot load, so it was never turned on
        module.autolog(disable=True)


def finish_session(session: Session, store: NoteStore, llm: LLM, *, today: date | None = None,
                   now: datetime | None = None) -> tuple[Optional[ConversationRecord], NotesResult]:
    """End the session, then run the note-taker on a copy of what was said. Both results are returned."""
    if session.user_id is None:
        raise ValueError("the session has no user_id, so its notes cannot be found")
    if not session.consent_for_training:
        tracing_off()
        tracing = sys.modules.get("puddles.tracing")  # not imported here, so that this module does not import MLflow
        if tracing is not None and isinstance(llm, tracing.TracedLLM):
            raise ValueError("training use is off, so the note-taker's model must not be traced")
    user_id = session.user_id
    turns = list(session.turns)  # end_session clears them
    notes = NotesResult()
    try:
        record = end_session(session, now)
    finally:  # the note-taker runs even if the record could not be written
        if turns:
            notes = _take(store, llm, session, user_id, turns, today=today, now=now)
        del turns
    return record, notes


def _take(store: NoteStore, llm: LLM, session: Session, user_id: str, turns: list, *,
          today: date | None, now: datetime | None) -> NotesResult:
    """`take_notes`, which raises nothing for a model or past.dev failure; any other failure is a failed result.

    The reason holds only the exception's type: its text could repeat the person's words.
    """
    try:
        folder = store.root / "notes" / check_user_id(user_id)
        new = _first_missing(folder)
        def run() -> NotesResult:
            return take_notes(store, llm, user_id=user_id, session_id=session.session_id, turns=turns,
                              remember_me=session.remember_me, today=today, now=now)

        if session.consent_for_training:
            result = run()
        else:
            # An empty context: no span the caller left open becomes the parent of a call, and no usage
            # meter set by the caller opens a span around the model's requests.
            result = contextvars.Context().run(run)
        if new is not None:
            _remove_if_empty(store, user_id, folder, new)
        return result
    except Exception as exc:
        return NotesResult(failed=True, reason=type(exc).__name__)


def _first_missing(folder: Path) -> Optional[Path]:
    """The highest folder on the way to `folder` that does not exist yet, or None if it exists."""
    missing = None
    for p in [folder, *folder.parents]:
        if p.exists():
            break
        missing = p
    return missing


def _remove_if_empty(store: NoteStore, user_id: str, folder: Path, new: Path) -> None:
    """A person with no notes before and none after leaves nothing behind, not even an empty index.

    `take_notes` always saves the index; for someone who had no folder and ended with no notes,
    that file holds nothing, and the session should leave nothing new on disk.
    """
    if sorted(os.listdir(folder)) != ["index.json"] or store.index(user_id).notes:
        return
    (folder / "index.json").unlink()
    for p in [folder, *folder.parents]:
        try:
            p.rmdir()
        except OSError:
            return
        if p == new:
            return
