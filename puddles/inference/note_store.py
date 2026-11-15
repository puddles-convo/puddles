"""A person's notes: the index of live notes on disk, and their private audience on past.dev.

The index decides what the helper sees; past.dev is a searchable copy. A note comes off the
index before anything is asked of past.dev, so a note someone asked to forget is never shown
again, even while past.dev still holds it. Contract: docs/specs/notes.md.
"""
import json
import os
import re
import tempfile
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import quote

from puddles.memory import TRAINING_IDENTITIES, FakeMemory, PastClient, PastError
from schemas import FOLLOW_UP_EXPIRY_DAYS, ExistingNote, LiveNote, LiveNoteIndex, MemoryEntry, UserNote

NOTE_TAKER = "note-taker"
RECALL_QUERY = "What should the robot remember about this person?"

USER_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
USER_ID_MAX = 33  # past.dev allows an audience name 40 characters, and the audience is person-<user_id>
RESERVED_IDS = frozenset([*TRAINING_IDENTITIES, NOTE_TAKER, "project"])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def audience_for(user_id: str) -> str:
    return f"person-{check_user_id(user_id)}"


def check_user_id(user_id: str) -> str:
    """Refuse anything that could name a path outside the root, or a training identity."""
    if (not isinstance(user_id, str) or len(user_id) > USER_ID_MAX or not USER_ID_RE.fullmatch(user_id)
            or user_id in RESERVED_IDS):
        raise ValueError(f"not a usable user ID: {user_id!r}")
    return user_id


class UserMemory(Protocol):
    def ensure_audience(self, audience: str, identities: list[str]) -> None: ...
    def ingest(self, entry: MemoryEntry) -> str: ...
    def wait(self, ingestion_id: str, timeout_s: float = 60) -> bool: ...
    def recall(self, query: str, identity: str, level: str = "low", limit: int = 20) -> list[dict]: ...
    def delete(self, entry_id: str) -> bool: ...


class PastUserMemory(PastClient):
    """The past.dev client with a private audience per person and delete by our own entry ID.

    Both calls follow https://past.dev/docs/memory-api: `PUT /audiences/{slug}` (kind `fixed`)
    and `DELETE /data-points/{sourceId}`, which answers 202 with the deleted IDs.
    """

    def ensure_audience(self, audience: str, identities: list[str]) -> None:
        self._call("PUT", f"/audiences/{quote(audience, safe='')}",
                   json={"kind": "fixed", "name": audience, "identities": list(identities)})

    def delete(self, entry_id: str) -> bool:
        """True only on a 202 that names the ID. A 404 (unknown or already deleted) is not a confirmation."""
        resp = self._http.request("DELETE", f"/data-points/{quote(entry_id, safe='')}")
        if resp.status_code == 404:
            return False
        if resp.status_code >= 400:
            raise PastError(f"DELETE /data-points/{entry_id} returned {resp.status_code}: {resp.text[:300]}")
        if resp.status_code != 202:
            return False
        try:
            body = resp.json() if resp.content else {}
        except ValueError:
            return False
        return entry_id in (body.get("ids") or [])


class FakeUserMemory(FakeMemory):
    """In-process past.dev with private audiences and delete, for tests and offline runs."""

    def __init__(self, fail_deletes: int = 0) -> None:
        super().__init__()
        self.fail_deletes = fail_deletes  # the next N delete calls return False and delete nothing
        self.delete_calls: list[str] = []

    def ensure_audience(self, audience: str, identities: list[str]) -> None:
        self.audiences[audience] = list(identities)

    def delete(self, entry_id: str) -> bool:
        self.delete_calls.append(entry_id)
        if self.fail_deletes > 0:
            self.fail_deletes -= 1
            return False
        return self.entries.pop(entry_id, None) is not None


def _write_json(path: Path, data: Any) -> None:
    """Write to a temporary file in the same folder, then rename over the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def belongs_to(note_id: str, user_id: str) -> bool:
    """True only for `note-<user_id>-<NNN>`. The number has no hyphen, so no two people can share an ID."""
    return re.fullmatch(rf"note-{re.escape(user_id)}-\d{{3,}}", note_id) is not None


def note_number(note_id: str) -> int:
    m = re.search(r"-(\d+)$", note_id)
    return int(m.group(1)) if m else 0


def content_date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def expires_on(note: LiveNote) -> Optional[date]:
    """The last day a follow-up is live. Other kinds never expire."""
    if note.kind != "follow_up":
        return None
    return (note.due_date or note.last_confirmed) + timedelta(days=FOLLOW_UP_EXPIRY_DAYS)


class NoteStore:
    """One folder per person under `<root>/notes/<user_id>/`; past.dev through `UserMemory`.

    Every change that touches more than one of index.json, notes.json, pending_deletes.json and past.dev
    first writes down what it is about to do, and every public method that changes anything finishes
    what an interrupted call left (`_recover`) before doing its own work. The two intents:

    - `pending_deletes.json`: these notes are coming off. Recovery takes them off the index and out of
      notes.json; only then does `delete` ask past.dev to delete them.
    - `writing` in `counter.json`: notes being sent to past.dev that are not yet on the index, each with
      the store instance that sent it. One left by another instance (a process that stopped) is queued
      for deletion unless it reached the index, so a note past.dev may hold is never unknown to us.
    """

    def __init__(self, memory: UserMemory, root: Path, *, delete_attempts: int = 3) -> None:
        if delete_attempts < 1:
            raise ValueError("delete_attempts must be at least 1")
        self.memory = memory
        self.root = Path(root)
        self.delete_attempts = delete_attempts
        self._owner = uuid.uuid4().hex  # marks the writes this instance has in flight

    def _dir(self, user_id: str) -> Path:
        return self.root / "notes" / check_user_id(user_id)

    def ensure_person(self, user_id: str) -> None:
        self.memory.ensure_audience(audience_for(user_id), [user_id, NOTE_TAKER])

    # ---------- the files (reading them never changes anything) ----------

    def index(self, user_id: str) -> LiveNoteIndex:
        data = _read_json(self._dir(user_id) / "index.json", None)
        if data is None:
            return LiveNoteIndex(user_id=user_id)
        index = LiveNoteIndex.model_validate(data)
        if index.user_id != user_id:
            raise ValueError(f"index for {user_id} names {index.user_id}")
        return index

    def stored(self, user_id: str) -> dict[str, UserNote]:
        data = _read_json(self._dir(user_id) / "notes.json", {})
        notes = {k: UserNote.model_validate(v) for k, v in data.items()}
        for key, note in notes.items():
            if note.id != key or note.user_id != user_id or not belongs_to(key, user_id):
                raise ValueError(f"notes.json for {user_id} holds {key}, which is not one of their notes")
        return notes

    def pending_deletes(self, user_id: str) -> list[str]:
        return list(_read_json(self._dir(user_id) / "pending_deletes.json", []))

    def _save_index_file(self, index: LiveNoteIndex) -> None:
        _write_json(self._dir(index.user_id) / "index.json", index.model_dump(mode="json"))

    def _save_stored(self, user_id: str, notes: dict[str, UserNote]) -> None:
        _write_json(self._dir(user_id) / "notes.json", {k: v.model_dump(mode="json") for k, v in notes.items()})

    def _save_pending(self, user_id: str, ids: list[str]) -> None:
        _write_json(self._dir(user_id) / "pending_deletes.json", ids)

    def _add_pending(self, user_id: str, ids: list[str]) -> None:
        """Only the person's own note IDs, so a delete for one person can never reach another's entry."""
        pending = self.pending_deletes(user_id)
        added = [i for i in dict.fromkeys(ids) if i not in pending and belongs_to(i, user_id)]
        if added:
            self._save_pending(user_id, pending + added)

    def _counter(self, user_id: str) -> dict:
        return dict(_read_json(self._dir(user_id) / "counter.json", {"next": 1}))

    def _save_counter(self, user_id: str, counter: dict) -> None:
        """`{"next": N}`, plus `writing` only while a write is in flight."""
        if not counter.get("writing"):
            counter.pop("writing", None)
        _write_json(self._dir(user_id) / "counter.json", counter)

    def _writing(self, user_id: str) -> list[dict]:
        return list(self._counter(user_id).get("writing", []))

    def _set_writing(self, user_id: str, entries: list[dict]) -> None:
        counter = self._counter(user_id)
        if counter.get("writing", []) != entries:
            self._save_counter(user_id, {**counter, "writing": entries})

    def _remove_from_stored(self, user_id: str, ids: set[str]) -> None:
        stored = self.stored(user_id)
        if ids & set(stored):
            self._save_stored(user_id, {k: v for k, v in stored.items() if k not in ids})

    def _remove_from_index(self, user_id: str, ids: set[str]) -> None:
        index = self.index(user_id)
        if any(n.note_id in ids for n in index.notes):
            self._save_index_file(LiveNoteIndex(user_id=user_id,
                                                notes=[n for n in index.notes if n.note_id not in ids]))

    def _recover(self, user_id: str) -> None:
        """Finish what an interrupted call left. Local files only; each step can be run again."""
        on_index = {n.note_id for n in self.index(user_id).notes}
        writing = self._writing(user_id)
        abandoned = {w["note_id"] for w in writing if w.get("owner") != self._owner}
        # A write another instance left: kept if it reached the index, otherwise queued for deletion.
        self._add_pending(user_id, sorted(abandoned - on_index))
        pending = set(self.pending_deletes(user_id))
        self._remove_from_index(user_id, pending)
        self._remove_from_stored(user_id, pending)
        self._set_writing(user_id, [w for w in writing if w["note_id"] not in abandoned | pending])

    def save_index(self, index: LiveNoteIndex) -> None:
        """Write the index, then drop from notes.json every note not on it, queuing each for deletion.

        So a note written but never put on the index (a crash between the two) is still deleted from past.dev.
        """
        user_id = check_user_id(index.user_id)
        foreign = [n.note_id for n in index.notes if not belongs_to(n.note_id, user_id)]
        if foreign:
            raise ValueError(f"the index for {user_id} names notes that are not theirs: {foreign}")
        self._recover(user_id)
        live = {n.note_id for n in index.notes}
        forgotten = live & set(self.pending_deletes(user_id))
        if forgotten:
            raise ValueError(f"notes waiting to be deleted cannot go back on the index: {sorted(forgotten)}")
        in_flight = [w["note_id"] for w in self._writing(user_id)]
        stored = self.stored(user_id)
        self._add_pending(user_id, [i for i in [*stored, *in_flight] if i not in live])  # the intent first
        self._save_index_file(index)
        self._set_writing(user_id, [])  # each write in flight is now on the index or queued for deletion
        self._remove_from_stored(user_id, set(stored) - live)

    def next_note_id(self, user_id: str) -> str:
        """A number is never used twice, even after a delete."""
        counter = self._counter(user_id)
        n = int(counter["next"])
        self._save_counter(user_id, {**counter, "next": n + 1})
        return f"note-{user_id}-{n:03d}"  # the user ID exactly as it is

    # ---------- reading ----------

    def expire(self, user_id: str, today: date) -> list[str]:
        self._recover(user_id)
        index = self.index(user_id)
        gone = [n.note_id for n in index.notes if (last := expires_on(n)) is not None and today > last]
        if gone:
            self.take_off(user_id, gone)
            self.delete(user_id, gone)
        return gone

    def live_notes(self, user_id: str, today: date) -> list[ExistingNote]:
        """Only notes on the index and not waiting to be deleted, in the index's order; facts from notes.json."""
        self.expire(user_id, today)
        index = self.index(user_id)
        try:
            self.ensure_person(user_id)
            # Recall as the person. A result counts only if its source ID is on the index, and every
            # sentence is then taken from notes.json, so what past.dev returns adds no note and no text.
            self.memory.recall(RECALL_QUERY, identity=user_id)
        except Exception:
            pass  # the sentences are read from notes.json, so a failed recall loses nothing
        stored = self.stored(user_id)
        pending = set(self.pending_deletes(user_id))
        return [ExistingNote(note_id=n.note_id, kind=n.kind, fact=stored[n.note_id].fact,
                             last_confirmed=n.last_confirmed, due_date=n.due_date)
                for n in index.notes if n.note_id in stored and n.note_id not in pending]

    def helper_notes(self, user_id: str, today: date) -> list[str]:
        return [n.fact for n in self.live_notes(user_id, today)]

    # ---------- writing ----------

    def write(self, note: UserNote) -> None:
        """Send one note to the person's audience and wait. Only a confirmed note reaches notes.json.

        The ID is written down (`writing`) before anything is sent, so a stop at any point leaves a
        note that past.dev may hold known to us, and "forget everything" can always find it.
        """
        user_id = check_user_id(note.user_id)
        if not belongs_to(note.id, user_id):
            raise ValueError(f"note {note.id} is not an ID of {user_id}'s notes")
        self._recover(user_id)
        in_flight = {w["note_id"] for w in self._writing(user_id)}
        if note.id in self.stored(user_id) or note.id in in_flight or note.id in self.pending_deletes(user_id):
            raise ValueError(f"note {note.id} is already stored, being written or being deleted")
        entry = MemoryEntry(
            id=note.id, content=f"{content_date(note.last_confirmed)}: {note.fact}",
            label=f"{NOTE_TAKER}: user note", timestamp=note.created_at, identity=NOTE_TAKER,
            audience=audience_for(user_id),
            metadata={"type": "user_note", "kind": note.kind, "user_id": user_id, "session_id": note.session_id})
        self._set_writing(user_id, [*self._writing(user_id), {"note_id": note.id, "owner": self._owner}])
        accepted = False
        try:
            self.ensure_person(user_id)
            ingestion_id = self.memory.ingest(entry)
            accepted = True
            confirmed = self.memory.wait(ingestion_id)
        except Exception as exc:
            self._abandon_write(user_id, note.id, accepted)
            raise PastError(f"note {note.id} was not written: {exc}") from exc
        if not confirmed:
            self._abandon_write(user_id, note.id, True)
            raise PastError(f"past.dev did not confirm note {note.id} within the wait")
        stored = self.stored(user_id)
        stored[note.id] = note
        self._save_stored(user_id, stored)  # stays in `writing` until save_index puts it on the index

    def _abandon_write(self, user_id: str, note_id: str, accepted: bool) -> None:
        if accepted:
            self._add_pending(user_id, [note_id])  # past.dev took it and may still store it
        self._set_writing(user_id, [w for w in self._writing(user_id) if w["note_id"] != note_id])

    def take_off(self, user_id: str, note_ids: list[str]) -> None:
        """From the moment this returns the helper cannot see these notes.

        The IDs go into pending_deletes.json first, so a stop part-way is finished by the next call.
        """
        self._recover(user_id)
        wanted = set(note_ids)
        off = [n.note_id for n in self.index(user_id).notes if n.note_id in wanted]
        if not off:
            return
        self._add_pending(user_id, off)
        self._remove_from_index(user_id, set(off))
        self._remove_from_stored(user_id, set(off))

    def delete(self, user_id: str, note_ids: list[str]) -> list[str]:
        """Delete from past.dev, retrying, together with every ID already pending. Returns those not confirmed."""
        self.take_off(user_id, note_ids)  # off the index first, always (it recovers first)
        self._add_pending(user_id, note_ids)
        self._recover(user_id)  # from here nothing pending is on the index, in notes.json or in flight
        try:
            self.ensure_person(user_id)
        except Exception:
            pass  # a delete is by entry ID and does not need the audience
        for note_id in self.pending_deletes(user_id):
            if not belongs_to(note_id, user_id):
                continue  # never send a delete for an ID that is not this person's
            if self._delete_one(note_id):
                self._save_pending(user_id, [i for i in self.pending_deletes(user_id) if i != note_id])
        return self.pending_deletes(user_id)

    def _delete_one(self, note_id: str) -> bool:
        for _ in range(self.delete_attempts):
            try:
                if self.memory.delete(note_id):
                    return True
            except Exception:
                continue
        return False
