"""Note-taker (inference step 5): after a session, decide what the robot remembers about the person.

It always runs, whatever the person answered, because forgetting must always work. With "remember
me" off it may only delete. Code works out the whole new index before saving anything, and saves
in an order that never shows the helper a note that came off. Contract: docs/specs/notes.md.

This module never traces: whether the model calls are traced is the caller's decision.
"""
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from puddles.agents import system_prompt
from puddles.inference.note_store import NOTE_TAKER, NoteStore, check_user_id, note_number
from puddles.llm import LLM, call_checked
from puddles.memory import PastError
from schemas import NOTE_CAPS, ExistingNote, LiveNote, LiveNoteIndex, NoteAction, NoteTakerInput, Strict, Turn, \
    UserNote

SYSTEM = system_prompt("i5-note-taker.md")
NEEDS_NOTE_ID = ("replace", "confirm", "delete")
WRITES = ("add", "replace")
NOT_WRITTEN_REASONS = ("consent", "forget_everything", "limit")  # and "write_failed:" + the exception's type name


def _not_stored(result: "NotesResult", reason: str, n: int = 1) -> None:
    """Count n proposed adds or replaces as not stored, under one reason. Never holds an exception's message."""
    if n <= 0:
        return
    result.not_written += n
    result.not_written_reasons[reason] = result.not_written_reasons.get(reason, 0) + n


class NoteTakerOutput(Strict):
    """A list cannot be asked for through LLM.structured, so the list sits in an object. Not yet on issue #1."""

    forget_everything: bool = False
    actions: list[NoteAction] = []


@dataclass
class NotesResult:
    added: list[str] = field(default_factory=list)  # new note IDs written, replacements included
    replaced: list[str] = field(default_factory=list)  # old note IDs that were replaced
    confirmed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # taken off by a delete or by "forget everything"
    dropped_by_limit: list[str] = field(default_factory=list)  # existing note IDs dropped by a limit
    not_written: int = 0  # adds and replaces dropped by consent, by "forget everything" or by a limit
    # why each of those was not stored; keys from NOT_WRITTEN_REASONS, or "write_failed:<type name>"; sums to not_written
    not_written_reasons: dict[str, int] = field(default_factory=dict)
    undeleted: list[str] = field(default_factory=list)  # off the index, not confirmed deleted by past.dev
    forgot_everything: bool = False
    failed: bool = False  # the model's answer failed twice; nothing was changed
    reason: str = ""


def build_input(store: NoteStore, user_id: str, turns: list[Turn], *, remember_me: bool,
                today: date) -> NoteTakerInput:
    return NoteTakerInput(today=today.isoformat(), may_add_notes=remember_me,
                          existing_notes=store.live_notes(user_id, today), turns=turns)


def render_user(inp: NoteTakerInput) -> str:
    """Every kind is shown with its count and limit, the empty ones too."""
    blocks = []
    for kind, limit in NOTE_CAPS.items():
        notes = [n for n in inp.existing_notes if n.kind == kind]
        lines = [f"- {n.note_id}, last confirmed {n.last_confirmed.isoformat()}"
                 f"{f', due {n.due_date.isoformat()}' if n.due_date else ''}: {n.fact}" for n in notes]
        blocks.append("\n".join([f'<{kind} count="{len(notes)}" limit="{limit}">', *lines, f"</{kind}>"]))
    turns = "\n".join(f"{t.speaker}: {t.text}" for t in inp.turns)
    return (f"<today>{inp.today}</today>\n"
            f"<may_add_notes>{'true' if inp.may_add_notes else 'false'}</may_add_notes>\n\n"
            f"<existing_notes>\n" + "\n".join(blocks) + "\n</existing_notes>\n\n"
            f"<conversation>\n{turns}\n</conversation>")


def check(out: NoteTakerOutput, inp: NoteTakerInput) -> list[str]:
    """A note ID that is unknown, or that belongs to another person, is refused."""
    known = {n.note_id for n in inp.existing_notes}
    problems = []
    for i, a in enumerate(out.actions, 1):
        where = f"action {i} ({a.action})"
        if a.action in NEEDS_NOTE_ID:
            if not a.note_id:
                problems.append(f"{where} needs the note_id of an existing note")
            elif a.note_id not in known:
                problems.append(f"{where} names {a.note_id}, which is not one of the existing notes")
        if a.action in WRITES and (not a.kind or not a.fact or not a.fact.strip()):
            problems.append(f"{where} needs a kind and a fact")
        if a.action == "add" and a.note_id:
            problems.append(f"{where} must not have a note_id; code gives new notes their IDs")
    named = Counter(a.note_id for a in out.actions if a.note_id)
    problems += [f"{note_id} is named by {n} actions; name each note once" for note_id, n in named.items() if n > 1]
    return problems


def run(inp: NoteTakerInput, llm: LLM) -> NoteTakerOutput:
    user = render_user(inp)
    return call_checked(lambda fb: llm.structured(SYSTEM, user + fb, NoteTakerOutput),
                        lambda out: check(out, inp), NOTE_TAKER)


def to_record(action: NoteAction, *, user_id: str, session_id: str, note_id: str, today: date,
              now: datetime, old: Optional[ExistingNote] = None) -> UserNote:
    """A due date is kept only on a follow-up; a replacing follow-up with none keeps the old note's."""
    due = None
    if action.kind == "follow_up":
        due = action.due_date or (old.due_date if old else None)
    return UserNote(id=note_id, written_by=NOTE_TAKER, created_at=now,
                    parent_ids=[session_id] + ([old.note_id] if old else []),
                    user_id=check_user_id(user_id), session_id=session_id, kind=action.kind,
                    fact=(action.fact or "").strip(), last_confirmed=today, due_date=due)


def _drop_order(note: LiveNote) -> tuple[date, int]:
    return note.last_confirmed, note_number(note.note_id)


def take_notes(store: NoteStore, llm: LLM, *, user_id: str, session_id: str, turns: list[Turn],
               remember_me: bool, today: date | None = None, now: datetime | None = None) -> NotesResult:
    """The whole step after a session. Raises nothing for a model or a past.dev failure."""
    check_user_id(user_id)
    now = now or datetime.now(timezone.utc).replace(microsecond=0)
    today = today or now.date()
    inp = build_input(store, user_id, turns, remember_me=remember_me, today=today)
    try:
        out = run(inp, llm)
    except Exception as exc:  # AgentFailed, or the model call itself failed: change nothing
        return NotesResult(failed=True, reason=str(exc), undeleted=store.pending_deletes(user_id))

    result = NotesResult(forgot_everything=out.forget_everything)
    existing = {n.note_id: n for n in inp.existing_notes}
    index = store.index(user_id)
    writes = [a for a in out.actions if a.action in WRITES]

    # 1-2. Which actions are kept.
    if out.forget_everything:
        result.deleted = [n.note_id for n in index.notes]
        _not_stored(result, "forget_everything", len(writes))
        writes, confirms = [], []
    else:
        result.deleted = [a.note_id for a in out.actions if a.action == "delete"]
        confirms = [a.note_id for a in out.actions if a.action == "confirm"]
        if not remember_me:
            _not_stored(result, "consent", len(writes))
            writes, confirms = [], []
    result.replaced = [a.note_id for a in writes if a.action == "replace"]

    # 3. The new index, worked out in full before anything is saved.
    off = set(result.deleted) | set(result.replaced)
    kept = [n.model_copy(update={"last_confirmed": today}) if n.note_id in confirms else n
            for n in index.notes if n.note_id not in off]
    new_notes: dict[str, UserNote] = {}
    for a in writes:
        note_id = store.next_note_id(user_id)
        new_notes[note_id] = to_record(a, user_id=user_id, session_id=session_id, note_id=note_id, today=today,
                                       now=now, old=existing.get(a.note_id) if a.note_id else None)
    candidates = kept + [LiveNote(note_id=n.id, kind=n.kind, last_confirmed=n.last_confirmed, due_date=n.due_date)
                         for n in new_notes.values()]
    dropped: set[str] = set()
    for kind, limit in NOTE_CAPS.items():
        of_kind = sorted((n for n in candidates if n.kind == kind), key=_drop_order)
        dropped |= {n.note_id for n in of_kind[:max(0, len(of_kind) - limit)]}
    result.dropped_by_limit = [n.note_id for n in kept if n.note_id in dropped]
    _not_stored(result, "limit", sum(1 for note_id in new_notes if note_id in dropped))
    result.confirmed = [n.note_id for n in kept if n.note_id in confirms and n.note_id not in dropped]
    came_off = result.deleted + result.replaced + result.dropped_by_limit

    # 4.1 Off the index first, with the confirms, without the new notes.
    first = [n for n in kept if n.note_id not in dropped]
    store.take_off(user_id, came_off)
    store.save_index(LiveNoteIndex(user_id=user_id, notes=first))

    # 4.2 Write the new notes; one that fails is left out and the others go on.
    written = []
    for note_id, note in new_notes.items():
        if note_id in dropped:
            continue
        try:
            store.write(note)
        except PastError as exc:
            _not_stored(result, "write_failed:" + type(exc).__name__)
            continue
        written.append(LiveNote(note_id=note_id, kind=note.kind, last_confirmed=note.last_confirmed,
                                due_date=note.due_date))
    result.added = [n.note_id for n in written]

    # 4.3 The index again, with the notes that were written.
    store.save_index(LiveNoteIndex(user_id=user_id, notes=first + written))

    # 4.4 Delete from past.dev everything that came off, and retry whatever is still pending.
    result.undeleted = store.delete(user_id, came_off)
    return result
