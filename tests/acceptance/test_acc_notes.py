"""Acceptance tests from docs/specs/notes.md, without implementation knowledge.

Safety cases come first. Temporary states during write/index commit are tested
separately from the final-state invariant, as allowed by the amended contract.
"""

from collections import Counter, deque
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket

import pytest
from pydantic import ValidationError

from puddles.agents import system_prompt
from puddles.inference import note_store as note_store_api, notes as notes_api
from puddles.inference.note_store import (
    NOTE_TAKER, RECALL_QUERY, FakeUserMemory, NoteStore, PastUserMemory,
    audience_for, check_user_id,
)
from puddles.inference.notes import (
    SYSTEM, NoteTakerOutput, NotesResult, build_input, check, render_user, run,
    take_notes, to_record,
)
from puddles.llm import AgentFailed
from puddles.memory import FakeMemory, PastClient, PastError, TRAINING_IDENTITIES
from schemas import (
    FOLLOW_UP_EXPIRY_DAYS, NOTE_CAPS, ExistingNote, LiveNote, LiveNoteIndex,
    MemoryEntry, NoteAction, NoteTakerInput, Turn, UserNote,
)


TODAY = date(2026, 11, 15)
NOW = datetime(2026, 11, 15, 12, 30, tzinfo=timezone.utc)
EARLIER = TODAY - timedelta(days=3)


class ScriptedLLM:
    """Only the documented structured-output interface; no real model."""

    def __init__(self, *answers, before_answer=None):
        self.answers = deque(answers)
        self.calls = []
        self.before_answer = before_answer

    def structured(self, system, user, schema):
        self.calls.append((system, user, schema))
        if self.before_answer is not None:
            self.before_answer()
        assert self.answers, "Unexpected extra model call"
        answer = self.answers.popleft()
        if isinstance(answer, Exception):
            raise answer
        return schema.model_validate(deepcopy(answer))


class MemorySpy(FakeMemory):
    """Instant protocol fake; scripted failures never poll or sleep."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.delete_answers = deque()
        self.ingest_error = None
        self.wait_answer = True
        self.recall_answer = None
        self.observe = None

    def ensure_audience(self, audience, identities):
        self.events.append(("ensure", audience, list(identities)))
        self.audiences[audience] = list(identities)

    def ingest(self, entry):
        self.events.append(("ingest", entry))
        if self.observe is not None:
            self.observe("ingest", entry.id)
        if self.ingest_error is not None:
            raise self.ingest_error
        return super().ingest(entry)

    def wait(self, ingestion_id, timeout_s=60):
        self.events.append(("wait", ingestion_id))
        if self.observe is not None:
            self.observe("wait", ingestion_id)
        if isinstance(self.wait_answer, Exception):
            raise self.wait_answer
        return self.wait_answer

    def recall(self, query, identity, level="low", limit=20):
        self.events.append(("recall", query, identity))
        if self.recall_answer is not None:
            return deepcopy(self.recall_answer)
        return super().recall(query, identity, level, limit)

    def delete(self, entry_id):
        self.events.append(("delete", entry_id))
        if self.observe is not None:
            self.observe("delete", entry_id)
        answer = self.delete_answers.popleft() if self.delete_answers else True
        if isinstance(answer, Exception):
            raise answer
        if answer:
            self.entries.pop(entry_id, None)
        return answer


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    for key in tuple(os.environ):
        if key.endswith("API_KEY"):
            monkeypatch.delenv(key, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Notes acceptance tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def store(tmp_path):
    return NoteStore(MemorySpy(), tmp_path, delete_attempts=3)


def note(note_id="note-alice-001", *, user_id="alice", kind="situation",
         fact="This person works nights.", last_confirmed=EARLIER, due_date=None):
    return UserNote(
        id=note_id, written_by=NOTE_TAKER, created_at=NOW,
        user_id=user_id, session_id="earlier-session", kind=kind, fact=fact,
        last_confirmed=last_confirmed, due_date=due_date,
    )


def live(record):
    return LiveNote(
        note_id=record.id, kind=record.kind,
        last_confirmed=record.last_confirmed, due_date=record.due_date,
    )


def seed(store, user_id="alice", specs=None):
    """Write through the public API; publish only after confirmation."""
    records = [note(store.next_note_id(user_id), user_id=user_id, **values)
               for values in (specs if specs is not None else [{}])]
    for record in records:
        store.write(record)
    store.save_index(LiveNoteIndex(user_id=user_id, notes=[live(r) for r in records]))
    return records


def ids(store, user_id="alice"):
    return [item.note_id for item in store.index(user_id).notes]


def act(action, **fields):
    return dict(action=action, **fields)


def session(store, llm, *, remember_me=True, user_id="alice", turns=None):
    return take_notes(
        store, llm, user_id=user_id, session_id="current-session",
        turns=turns if turns is not None else [Turn(speaker="user", text="Remember my walks.")],
        remember_me=remember_me, today=TODAY, now=NOW,
    )


def disk_snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in (root / "notes").rglob("*")
            if p.is_file()}


def assert_committed(store, root, user_id="alice"):
    """Check public reads and actual file formats after a whole step."""
    index = store.index(user_id)
    stored = store.stored(user_id)
    assert set(ids(store, user_id)) == set(stored)
    assert all(count <= NOTE_CAPS[kind]
               for kind, count in Counter(n.kind for n in index.notes).items())
    folder = root / "notes" / user_id
    assert LiveNoteIndex.model_validate_json((folder / "index.json").read_text()) == index
    raw = json.loads((folder / "notes.json").read_text())
    assert {key: UserNote.model_validate(value) for key, value in raw.items()} == stored


# Section 2: consent and forgetting take priority over proposed new memories.

def test_no_consent_drops_add_replace_confirm_but_applies_delete(store, tmp_path):
    replaced, confirmed, deleted = seed(store, specs=[
        {"fact": "This person works nights."},
        {"fact": "This person has a sister."},
        {"fact": "This person moved house."},
    ])
    store.memory.events.clear()
    llm = ScriptedLLM({"actions": [
        act("add", kind="goals", fact="Private new information."),
        act("replace", note_id=replaced.id, kind="situation", fact="Private correction."),
        act("confirm", note_id=confirmed.id),
        act("delete", note_id=deleted.id),
    ]})
    result = session(store, llm, remember_me=False, turns=[
        Turn(speaker="user", text="No, don't remember me. Forget that I moved house."),
    ])
    assert isinstance(result, NotesResult)
    assert len(llm.calls) == 1  # Declining consent must not skip the forgetting step.
    assert not result.failed and not result.forgot_everything
    assert result.added == result.replaced == result.confirmed == []
    assert result.not_written == 2
    assert result.deleted == [deleted.id] and result.undeleted == []
    assert ids(store) == [replaced.id, confirmed.id]
    assert store.stored("alice") == {r.id: r for r in (replaced, confirmed)}
    assert all(n.last_confirmed == EARLIER for n in store.index("alice").notes)
    assert set(store.memory.entries) == {replaced.id, confirmed.id}
    assert not any(e[0] == "ingest" for e in store.memory.events)
    assert_committed(store, tmp_path)


@pytest.mark.parametrize("remember_me", [False, True])
def test_forget_everything_overrides_all_retention_actions(store, tmp_path, remember_me):
    first, second, third = seed(store, specs=[{}, {}, {}])
    store.memory.events.clear()
    result = session(store, ScriptedLLM({"forget_everything": True, "actions": [
        act("replace", note_id=first.id, kind="goals", fact="Must not be kept."),
        act("confirm", note_id=second.id),
        act("delete", note_id=third.id),
        act("add", kind="people", fact="Must not be remembered."),
    ]}), remember_me=remember_me)
    assert result.forgot_everything and not result.failed
    assert set(result.deleted) == {first.id, second.id, third.id}
    assert result.added == result.replaced == result.confirmed == []
    assert result.not_written == 2 and result.undeleted == []
    assert ids(store) == [] and store.stored("alice") == {}
    assert store.memory.entries == {}
    assert not any(e[0] == "ingest" for e in store.memory.events)
    assert_committed(store, tmp_path)


def test_forget_everything_hides_notes_during_failed_remote_deletion(tmp_path):
    memory = FakeUserMemory()
    store = NoteStore(memory, tmp_path, delete_attempts=1)
    old = seed(store)[0]
    memory.fail_deletes = 10
    result = session(store, ScriptedLLM({"forget_everything": True}))
    assert result.deleted == [old.id] and result.undeleted == [old.id]
    assert old.id in memory.entries  # Visibility assertion is non-vacuous.
    restarted = NoteStore(memory, tmp_path, delete_attempts=1)
    assert restarted.pending_deletes("alice") == [old.id]
    assert restarted.live_notes("alice", TODAY) == []
    assert restarted.helper_notes("alice", TODAY) == []
    assert build_input(restarted, "alice", [], remember_me=False,
                       today=TODAY).existing_notes == []
    assert restarted.stored("alice") == {}
    assert json.loads((tmp_path / "notes/alice/pending_deletes.json").read_text()) == [old.id]
    memory.fail_deletes = 0
    assert restarted.delete("alice", []) == []
    assert old.id not in memory.entries and restarted.pending_deletes("alice") == []


# Section 1: durable deletion and identity boundaries.

def test_hyphen_distinct_people_never_share_note_ids_audiences_or_files(store, tmp_path):
    hyphenated = seed(store, "user-01", [{"fact": "This person remembers quiet walks."}])[0]
    plain = seed(store, "user01", [{"fact": "This person remembers music."}])[0]
    assert hyphenated.id == "note-user-01-001"
    assert plain.id == "note-user01-001"
    assert hyphenated.id != plain.id
    assert set(store.memory.entries) == {hyphenated.id, plain.id}
    assert store.memory.audiences["person-user-01"] == ["user-01", NOTE_TAKER]
    assert store.memory.audiences["person-user01"] == ["user01", NOTE_TAKER]
    for user_id, own, other in (("user-01", hyphenated, plain), ("user01", plain, hyphenated)):
        assert store.memory.entries[own.id].audience == audience_for(user_id)
        assert store.helper_notes(user_id, TODAY) == [own.fact]
        inp = build_input(store, user_id, [], remember_me=True, today=TODAY)
        assert [(n.note_id, n.fact) for n in inp.existing_notes] == [(own.id, own.fact)]
        assert other.fact not in render_user(inp)
        assert_committed(store, tmp_path, user_id)
    for filename in ("index.json", "notes.json", "counter.json"):
        first = tmp_path / "notes/user-01" / filename
        second = tmp_path / "notes/user01" / filename
        assert first.is_file() and second.is_file() and not first.samefile(second)

    plain_before = {p.name: p.read_bytes() for p in (tmp_path / "notes/user01").iterdir() if p.is_file()}
    assert store.delete("user-01", [hyphenated.id]) == []
    assert hyphenated.id not in store.memory.entries and plain.id in store.memory.entries
    assert store.helper_notes("user-01", TODAY) == []
    assert store.helper_notes("user01", TODAY) == [plain.fact]
    assert {p.name: p.read_bytes() for p in (tmp_path / "notes/user01").iterdir() if p.is_file()} == plain_before
    restarted = NoteStore(store.memory, tmp_path)
    assert restarted.next_note_id("user-01") == "note-user-01-002"
    assert restarted.next_note_id("user-01") == "note-user-01-003"
    assert restarted.next_note_id("user01") == "note-user01-002"


def test_save_index_queues_every_pruned_note_including_unindexed_writes_after_restart(store, tmp_path):
    kept, removed, already_pending = seed(store, specs=[{}, {}, {}])
    store.take_off("alice", [already_pending.id])
    orphan = note(store.next_note_id("alice"), kind="goals", fact="Written before a crash.")
    store.write(orphan)
    assert orphan.id in store.stored("alice") and orphan.id not in ids(store)

    restarted = NoteStore(store.memory, tmp_path)
    assert orphan.id in restarted.stored("alice") and orphan.id not in ids(restarted)
    restarted.save_index(LiveNoteIndex(user_id="alice", notes=[live(kept)]))
    expected_pending = {already_pending.id, removed.id, orphan.id}
    assert set(restarted.pending_deletes("alice")) == expected_pending
    raw_pending = json.loads((tmp_path / "notes/alice/pending_deletes.json").read_text())
    assert set(raw_pending) == expected_pending
    assert expected_pending <= set(store.memory.entries)
    assert_committed(restarted, tmp_path)
    assert restarted.helper_notes("alice", TODAY) == [kept.fact]
    assert NoteStore(store.memory, tmp_path).delete("alice", []) == []
    assert set(store.memory.entries) == {kept.id}
    assert restarted.pending_deletes("alice") == []


@pytest.mark.parametrize("failure", ["confirmed", "false", "exception"])
def test_delete_takes_live_notes_off_before_any_remote_attempt(store, tmp_path, failure):
    removed, kept = seed(store, specs=[{}, {}])
    if failure == "false":
        store.memory.delete_answers.extend([False] * 3)
    elif failure == "exception":
        store.memory.delete_answers.extend(RuntimeError("delete unavailable") for _ in range(3))
    observed = []

    def observe(stage, entry_id):
        if stage != "delete":
            return
        assert entry_id == removed.id
        assert ids(store) == [kept.id]
        assert removed.id not in store.stored("alice")
        assert removed.id in store.pending_deletes("alice")
        assert removed.id in store.memory.entries
        observed.append(entry_id)

    store.memory.observe = observe
    undeleted = store.delete("alice", [removed.id])  # No prior take_off.
    store.memory.observe = None
    assert observed
    expected = [] if failure == "confirmed" else [removed.id]
    assert undeleted == expected and store.pending_deletes("alice") == expected
    assert (removed.id in store.memory.entries) is (failure != "confirmed")
    assert store.helper_notes("alice", TODAY) == [kept.fact]
    assert_committed(store, tmp_path)


def test_take_off_hides_only_known_ids_before_remote_delete(store, tmp_path):
    removed, kept = seed(store, specs=[{}, {}])
    store.take_off("alice", [removed.id, "note-alice-999"])
    assert removed.id in store.memory.entries
    assert ids(store) == [kept.id]
    assert set(store.stored("alice")) == {kept.id}
    assert store.pending_deletes("alice") == [removed.id]
    assert [n.note_id for n in store.live_notes("alice", TODAY)] == [kept.id]
    assert_committed(store, tmp_path)


def test_delete_retries_exceptions_and_false_then_retries_after_restart(store, tmp_path):
    old = seed(store)[0]
    store.take_off("alice", [old.id])
    store.memory.events.clear()
    store.memory.delete_answers.extend([False, RuntimeError("unavailable"), False])
    assert store.delete("alice", [old.id]) == [old.id]
    assert [e[1] for e in store.memory.events if e[0] == "delete"] == [old.id] * 3
    assert old.id in store.memory.entries
    assert store.pending_deletes("alice") == [old.id]
    restarted = NoteStore(store.memory, tmp_path, delete_attempts=3)
    store.memory.events.clear()
    store.memory.delete_answers.extend([False, True])
    assert restarted.delete("alice", []) == []
    assert [e[1] for e in store.memory.events if e[0] == "delete"] == [old.id] * 2
    assert restarted.pending_deletes("alice") == [] and old.id not in store.memory.entries


def test_delete_retry_budget_is_per_id_and_pending_is_per_person(store):
    alice = seed(store, specs=[{}, {}])
    bob = seed(store, "bob")[0]
    store.take_off("alice", [n.id for n in alice])
    store.take_off("bob", [bob.id])
    store.memory.delete_answers.extend([False] * 6)
    assert set(store.delete("alice", [n.id for n in alice])) == {n.id for n in alice}
    calls = Counter(e[1] for e in store.memory.events if e[0] == "delete")
    assert calls == {n.id: 3 for n in alice}
    assert store.pending_deletes("bob") == [bob.id]
    assert bob.id in store.memory.entries


def test_fake_memory_isolates_people_and_all_training_identities(tmp_path):
    memory = FakeUserMemory()
    assert isinstance(memory, FakeMemory)
    memory.ensure_training_audience()
    store = NoteStore(memory, tmp_path)
    alice = seed(store, specs=[{"fact": "This person remembers quiet walks."}])[0]
    bob = seed(store, "bob", [{"fact": "This person remembers music."}])[0]
    assert memory.audiences["person-alice"] == ["alice", "note-taker"]
    assert memory.audiences["person-bob"] == ["bob", "note-taker"]
    for user_id, own, other in (("alice", alice, bob), ("bob", bob, alice)):
        assert store.helper_notes(user_id, TODAY) == [own.fact]
        inp = build_input(store, user_id, [], remember_me=True, today=TODAY)
        assert [(n.note_id, n.fact) for n in inp.existing_notes] == [(own.id, own.fact)]
        assert other.fact not in render_user(inp)
        recalled = memory.recall(RECALL_QUERY, identity=user_id)
        sources = {s["sourceId"] for r in recalled for s in r["sources"]}
        assert sources == {own.id}
    for identity in TRAINING_IDENTITIES:
        assert memory.recall(RECALL_QUERY, identity=identity) == []


def test_recall_is_personal_index_ordered_and_uses_only_local_facts(store):
    first, second, missing = seed(store, specs=[
        {"fact": "Authoritative first fact."},
        {"fact": "Authoritative second fact."},
        {"fact": "Not returned by search.", "kind": "follow_up", "due_date": TODAY},
    ])
    foreign = seed(store, "bob", [{"fact": "Bob's private information."}])[0]
    removed = seed_extra(store, kind="goals", fact="Forgotten information.")
    store.take_off("alice", [removed.id])
    store.memory.recall_answer = [
        {"content": "Forged second fact.", "sources": [{"sourceId": second.id}]},
        {"content": foreign.fact, "sources": [{"sourceId": foreign.id}]},
        {"content": removed.fact, "sources": [{"sourceId": removed.id}]},
        {"content": "Forged first fact.", "sources": [{"sourceId": first.id}]},
    ]
    store.memory.events.clear()
    expected = [ExistingNote(note_id=r.id, kind=r.kind, fact=r.fact,
                             last_confirmed=r.last_confirmed, due_date=r.due_date)
                for r in (first, second, missing)]
    assert store.live_notes("alice", TODAY) == expected
    assert store.helper_notes("alice", TODAY) == [r.fact for r in (first, second, missing)]
    recalls = [e for e in store.memory.events if e[0] == "recall"]
    assert recalls == [("recall", RECALL_QUERY, "alice")] * 2
    assert store.memory.events[0] == ("ensure", "person-alice", ["alice", NOTE_TAKER])


@pytest.mark.parametrize("operation", ["ensure_audience", "recall"])
@pytest.mark.parametrize("error_type", [PastError, RuntimeError, NotImplementedError])
def test_memory_read_failures_fall_back_to_local_live_notes(store, operation, error_type, monkeypatch):
    first, second, removed = seed(store, specs=[
        {"fact": "Authoritative first fact."},
        {"kind": "follow_up", "fact": "Authoritative follow up.", "due_date": TODAY},
        {"kind": "goals", "fact": "Forgotten fact still held remotely."},
    ])
    store.take_off("alice", [removed.id])
    store.save_index(LiveNoteIndex(user_id="alice", notes=[
        live(first).model_copy(update={"last_confirmed": TODAY}), live(second),
    ]))

    def unavailable(*args, **kwargs):
        raise error_type("memory unavailable")

    monkeypatch.setattr(store.memory, operation, unavailable)
    expected = [
        ExistingNote(note_id=first.id, kind=first.kind, fact=first.fact, last_confirmed=TODAY),
        ExistingNote(note_id=second.id, kind=second.kind, fact=second.fact,
                     last_confirmed=second.last_confirmed, due_date=second.due_date),
    ]
    assert removed.id in store.memory.entries
    assert store.live_notes("alice", TODAY) == expected
    assert store.helper_notes("alice", TODAY) == [first.fact, second.fact]
    inp = build_input(store, "alice", [], remember_me=False, today=TODAY)
    assert inp.existing_notes == expected
    assert removed.fact not in render_user(inp)


def seed_extra(store, **fields):
    record = note(store.next_note_id("alice"), **fields)
    index = store.index("alice")
    store.write(record)
    store.save_index(LiveNoteIndex(user_id="alice", notes=[*index.notes, live(record)]))
    return record


@pytest.mark.parametrize("bad", [
    "", ".", "..", "../outside", "/absolute", "alice/bob", "alice\\bob",
    "UPPER", "alice_1", "-alice", " alice", "alice ", "alice\n", "é", "a" * 34,
    "note-taker", "project", *TRAINING_IDENTITIES,
])
def test_invalid_user_ids_are_refused_before_files_or_memory(store, tmp_path, bad):
    with pytest.raises(ValueError):
        check_user_id(bad)
    operations = [
        # Reason: every user-ID entry point must reject reserved IDs and path escapes.
        lambda: audience_for(bad),
        lambda: store.ensure_person(bad), lambda: store.index(bad),
        lambda: store.stored(bad), lambda: store.next_note_id(bad),
        lambda: store.expire(bad, TODAY), lambda: store.live_notes(bad, TODAY),
        lambda: store.helper_notes(bad, TODAY), lambda: store.take_off(bad, []),
        lambda: store.delete(bad, []), lambda: store.pending_deletes(bad),
        lambda: store.save_index(LiveNoteIndex(user_id=bad)),
        lambda: store.write(note(user_id=bad)),
        lambda: build_input(store, bad, [], remember_me=False, today=TODAY),
        lambda: session(store, ScriptedLLM({"actions": []}), user_id=bad),
        lambda: to_record(NoteAction(action="add", kind="goals", fact="A plan."),
                          user_id=bad, session_id="s", note_id="note-alice-001",
                          today=TODAY, now=NOW),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
    assert disk_snapshot(tmp_path) == {} and store.memory.events == []


@pytest.mark.parametrize("user_id", ["a", "0", "user-01", "a--b", "a-", "a" * 33])
def test_valid_ids_and_audiences_are_exact(user_id):
    assert check_user_id(user_id) == user_id
    assert audience_for(user_id) == f"person-{user_id}"
    assert NOTE_TAKER == "note-taker"
    assert RECALL_QUERY == "What should the robot remember about this person?"


def test_maximum_length_user_id_can_store_notes_with_a_40_character_audience(store, tmp_path):
    user_id = "a" * 33
    record = seed(store, user_id)[0]
    assert record.id == f"note-{user_id}-001"
    assert len(audience_for(user_id)) == 40
    assert store.memory.audiences[audience_for(user_id)] == [user_id, NOTE_TAKER]
    assert store.helper_notes(user_id, TODAY) == [record.fact]
    assert_committed(store, tmp_path, user_id)


def test_ensure_person_uses_exact_private_audience_and_membership(store):
    store.ensure_person("alice")
    assert store.memory.events == [("ensure", "person-alice", ["alice", NOTE_TAKER])]
    assert issubclass(PastUserMemory, PastClient)


@pytest.mark.parametrize("operation", ["write", "live_notes", "delete"])
def test_remote_operations_ensure_person_before_access(store, operation):
    old = seed(store)[0]
    candidate = note(store.next_note_id("alice"))
    if operation == "delete":
        store.take_off("alice", [old.id])
    store.memory.events.clear()
    store.memory.audiences.clear()
    if operation == "write":
        store.write(candidate)
        expected = "ingest"
    elif operation == "live_notes":
        store.live_notes("alice", TODAY)
        expected = "recall"
    else:
        store.delete("alice", [old.id])
        expected = "delete"
    assert store.memory.events[0] == ("ensure", "person-alice", ["alice", NOTE_TAKER])
    assert any(e[0] == expected for e in store.memory.events)


# Section 1: writes, file formats, counters and expiry.

def test_write_has_audience_metadata_and_waits_before_local_publication(store, tmp_path):
    record = note()
    before = store.index("alice")

    def observe(stage, entry_id):
        assert record.id not in store.stored("alice")
        assert store.index("alice") == before

    store.memory.observe = observe
    store.write(record)
    store.memory.observe = None
    assert store.memory.events[0] == ("ensure", "person-alice", ["alice", NOTE_TAKER])
    assert [e[0] for e in store.memory.events if e[0] != "ensure"] == ["ingest", "wait"]
    entry = next(e[1] for e in store.memory.events if e[0] == "ingest")
    assert isinstance(entry, MemoryEntry)
    assert entry.id == record.id
    assert entry.content == "12 Nov 2026: This person works nights."
    assert entry.label == "note-taker: user note" and entry.identity == NOTE_TAKER
    assert entry.audience == "person-alice"
    assert entry.metadata == {"type": "user_note", "kind": record.kind,
                              "user_id": "alice", "session_id": "earlier-session"}
    assert store.stored("alice") == {record.id: record}
    assert store.index("alice") == before
    raw = json.loads((tmp_path / "notes/alice/notes.json").read_text())
    assert UserNote.model_validate(raw[record.id]) == record
    store.memory.events.clear()
    # The contract fixes refusal, but not the exception type for a duplicate ID.
    with pytest.raises(Exception):
        store.write(record)
    assert not any(e[0] in {"ingest", "wait"} for e in store.memory.events)
    assert store.stored("alice") == {record.id: record}


@pytest.mark.parametrize("failure", ["ingest", "wait_false", "wait_exception"])
def test_failed_write_never_enters_either_file(store, tmp_path, failure):
    kept = seed(store)[0]
    candidate = note(store.next_note_id("alice"), fact="Unconfirmed information.")
    before_index = store.index("alice")
    if failure == "ingest":
        store.memory.ingest_error = RuntimeError("transport broke")
    elif failure == "wait_false":
        store.memory.wait_answer = False
    else:
        store.memory.wait_answer = RuntimeError("wait broke")
    with pytest.raises(PastError):
        store.write(candidate)
    assert store.index("alice") == before_index
    assert store.stored("alice") == {kept.id: kept}
    expected_pending = [] if failure == "ingest" else [candidate.id]
    assert store.pending_deletes("alice") == expected_pending
    restarted = NoteStore(store.memory, tmp_path)
    assert restarted.pending_deletes("alice") == expected_pending
    if failure != "ingest":
        assert candidate.id in store.memory.entries  # Accepted remotely, never confirmed.
        raw_pending = json.loads((tmp_path / "notes/alice/pending_deletes.json").read_text())
        assert raw_pending == [candidate.id]
        assert restarted.helper_notes("alice", TODAY) == [kept.fact]
        assert restarted.delete("alice", []) == []
        assert candidate.id not in store.memory.entries
        assert restarted.pending_deletes("alice") == []
    assert_committed(store, tmp_path)


def test_empty_index_save_pruning_and_counter_survive_restart(store, tmp_path):
    assert store.index("user-01") == LiveNoteIndex(user_id="user-01", notes=[])
    assert store.next_note_id("user-01") == "note-user-01-001"
    restarted = NoteStore(store.memory, tmp_path)
    assert restarted.next_note_id("user-01") == "note-user-01-002"
    assert json.loads((tmp_path / "notes/user-01/counter.json").read_text()) == {"next": 3}
    first, second = seed(store, specs=[{}, {}])
    store.save_index(LiveNoteIndex(user_id="alice", notes=[live(second)]))
    assert ids(store) == [second.id] and store.stored("alice") == {second.id: second}
    assert first.id in store.memory.entries  # save_index prunes locally, not remotely.
    assert store.pending_deletes("alice") == [first.id]
    store.take_off("alice", [second.id])
    store.delete("alice", [second.id])
    assert NoteStore(store.memory, tmp_path).next_note_id("alice") == "note-alice-003"
    assert_committed(store, tmp_path)


def test_counter_keeps_all_digits_after_999(store, tmp_path):
    folder = tmp_path / "notes/alice"
    folder.mkdir(parents=True)
    (folder / "counter.json").write_text('{"next": 999}', encoding="utf-8")
    assert store.next_note_id("alice") == "note-alice-999"
    assert NoteStore(store.memory, tmp_path).next_note_id("alice") == "note-alice-1000"


@pytest.mark.parametrize("has_due_date", [False, True])
@pytest.mark.parametrize("days_after", [FOLLOW_UP_EXPIRY_DAYS, FOLLOW_UP_EXPIRY_DAYS + 1])
def test_follow_up_expiry_has_strict_boundary_and_uses_due_date(
    store, has_due_date, days_after,
):
    anchor = TODAY - timedelta(days=days_after)
    old = seed(store, specs=[{
        "kind": "follow_up", "due_date": anchor if has_due_date else None,
        "last_confirmed": TODAY if has_due_date else anchor,
    }])[0]
    expired = days_after > FOLLOW_UP_EXPIRY_DAYS
    assert store.expire("alice", TODAY) == ([old.id] if expired else [])
    assert ids(store) == ([] if expired else [old.id])
    assert (old.id in store.stored("alice")) is not expired
    assert (old.id in store.memory.entries) is not expired


def test_live_notes_expires_before_recall_even_if_remote_delete_fails(store):
    old, kept = seed(store, specs=[
        {"kind": "follow_up", "last_confirmed": TODAY - timedelta(days=15)},
        {"kind": "goals", "last_confirmed": TODAY - timedelta(days=100)},
    ])
    store.memory.delete_answers.extend([False] * 3)
    assert [n.note_id for n in store.live_notes("alice", TODAY)] == [kept.id]
    assert old.id in store.memory.entries
    assert store.pending_deletes("alice") == [old.id]
    assert old.id not in store.stored("alice")


# Section 2: model boundary and prompt.

def test_build_input_and_render_preserve_session_and_show_every_cap(store):
    record = seed(store, specs=[{"kind": "follow_up", "due_date": TODAY}])[0]
    turns = [Turn(speaker="user", text="Could you ask about this tomorrow?"),
             Turn(speaker="helper", text="I can ask next time.")]
    inp = build_input(store, "alice", turns, remember_me=False, today=TODAY)
    assert isinstance(inp, NoteTakerInput)
    assert inp.today == TODAY.isoformat() and inp.may_add_notes is False
    assert inp.turns == turns
    assert inp.existing_notes == store.live_notes("alice", TODAY)
    rendered = render_user(inp)
    assert f"<today>{TODAY.isoformat()}</today>" in rendered
    assert "<may_add_notes>false</may_add_notes>" in rendered.lower()
    for kind, cap in NOTE_CAPS.items():
        count = int(kind == "follow_up")
        assert f'<{kind} count="{count}" limit="{cap}">' in rendered
        assert f"</{kind}>" in rendered
    assert f"- {record.id}, last confirmed {EARLIER}, due {TODAY}: {record.fact}" in rendered
    assert "user: Could you ask about this tomorrow?" in rendered
    assert "helper: I can ask next time." in rendered
    assert "<conversation>" in rendered and "</conversation>" in rendered
    assert SYSTEM == system_prompt("i5-note-taker.md")
    assert "forget_everything" in SYSTEM and "actions" in SYSTEM


def test_agent_output_documentation_opens_with_the_object_format(repo_root):
    text = (repo_root / "docs/agents/i5-note-taker.md").read_text(encoding="utf-8")
    section = text.split("## 4. Output", 1)[1]
    opening = section.split("| Field", 1)[0]
    assert "object" in opening.lower()
    assert "forget_everything" in opening and "actions" in opening


def input_with_notes():
    return NoteTakerInput(today=TODAY.isoformat(), may_add_notes=True, turns=[], existing_notes=[
        ExistingNote(note_id="note-alice-001", kind="goals", fact="A plan.", last_confirmed=EARLIER),
        ExistingNote(note_id="note-alice-002", kind="people", fact="A sister.", last_confirmed=EARLIER),
    ])


@pytest.mark.parametrize("action", ["replace", "confirm", "delete"])
@pytest.mark.parametrize("target", [None, "note-alice-999", "note-bob-001"])
def test_check_refuses_missing_unknown_and_foreign_targets(action, target):
    fields = dict(note_id=target)
    if action == "replace":
        fields.update(kind="goals", fact="A new plan.")
    out = NoteTakerOutput(actions=[NoteAction(**act(action, **fields))])
    assert check(out, input_with_notes())


@pytest.mark.parametrize("action", ["add", "replace"])
@pytest.mark.parametrize("missing", ["kind", "fact", "empty_fact", "spaces_only_fact"])
def test_check_requires_kind_and_nonempty_fact(action, missing):
    fields = {"kind": "goals", "fact": "A plan."}
    if action == "replace":
        fields["note_id"] = "note-alice-001"
    if missing == "empty_fact":
        fields["fact"] = ""
    elif missing == "spaces_only_fact":
        fields["fact"] = "   "
    else:
        fields.pop(missing)
    assert check(NoteTakerOutput(actions=[NoteAction(**act(action, **fields))]), input_with_notes())


def test_check_refuses_id_on_add_and_duplicate_target():
    inp = input_with_notes()
    assert check(NoteTakerOutput(actions=[NoteAction(
        action="add", note_id="note-alice-001", kind="goals", fact="A plan.",
    )]), inp)
    for second in ("confirm", "delete", "replace"):
        fields = {"kind": "goals", "fact": "A new plan."} if second == "replace" else {}
        assert check(NoteTakerOutput(actions=[
            NoteAction(action="confirm", note_id="note-alice-001"),
            NoteAction(action=second, note_id="note-alice-001", **fields),
        ]), inp)


def test_check_accepts_valid_actions_and_does_not_enforce_consent_itself():
    inp = input_with_notes().model_copy(update={"may_add_notes": False})
    out = NoteTakerOutput(actions=[
        NoteAction(action="add", kind="goals", fact="A new plan."),
        NoteAction(action="replace", note_id="note-alice-001", kind="goals", fact="Changed plan."),
        NoteAction(action="delete", note_id="note-alice-002"),
    ])
    assert check(out, inp) == []  # Consent filtering belongs to take_notes.
    assert check(NoteTakerOutput(), inp) == []
    assert check(NoteTakerOutput(actions=[NoteAction(action="confirm", note_id="note-alice-001")]), inp) == []
    assert NoteTakerOutput().forget_everything is False and NoteTakerOutput().actions == []
    with pytest.raises(ValidationError):
        NoteTakerOutput.model_validate({"unexpected": True})


@pytest.mark.parametrize("bad", [
    {"actions": [{"action": "delete", "note_id": "note-bob-001"}]},
    {"actions": [{"action": "invented"}]},
])
def test_run_retries_once_with_reasons_and_exact_output_schema(bad):
    inp = input_with_notes()
    llm = ScriptedLLM(bad, {"actions": []})
    assert run(inp, llm) == NoteTakerOutput()
    assert len(llm.calls) == 2
    assert all(s == SYSTEM and schema is NoteTakerOutput for s, _, schema in llm.calls)
    first, second = [user for _, user, _ in llm.calls]
    assert first == render_user(inp) and second.startswith(first) and second != first
    # The feedback must contain the actual check/validation reason, not a generic retry.
    if bad["actions"][0]["action"] == "delete":
        problems = check(NoteTakerOutput.model_validate(bad), inp)
        assert problems and all(problem in second[len(first):] for problem in problems)
    else:
        assert "invented" in second[len(first):]


def test_run_second_invalid_answer_raises_agentfailed_without_third_call():
    bad = {"actions": [{"action": "delete", "note_id": "note-bob-001"}]}
    llm = ScriptedLLM(bad, bad, {"actions": []})
    with pytest.raises(AgentFailed):
        run(input_with_notes(), llm)
    assert len(llm.calls) == 2


@pytest.mark.parametrize("failure", ["invalid_twice", "agent_failed", "model_exception"])
def test_failed_model_step_changes_no_notes_or_files(store, tmp_path, failure):
    records = seed(store, specs=[{}, {"kind": "goals"}])
    before_files = disk_snapshot(tmp_path)
    before_entries = deepcopy(store.memory.entries)
    store.memory.events.clear()
    bad = {"forget_everything": True, "actions": [
        act("delete", note_id="note-bob-001"),
        act("add", kind="goals", fact="Must not be written."),
    ]}
    if failure == "invalid_twice":
        llm = ScriptedLLM(bad, bad)
    elif failure == "agent_failed":
        llm = ScriptedLLM(AgentFailed("refused"))
    else:
        llm = ScriptedLLM(RuntimeError("model unavailable"), RuntimeError("model unavailable"))
    result = session(store, llm)
    assert result.failed and result.reason
    assert result.added == result.replaced == result.confirmed == result.deleted == []
    assert result.dropped_by_limit == [] and result.not_written == 0
    assert disk_snapshot(tmp_path) == before_files
    assert store.memory.entries == before_entries
    assert not any(e[0] in {"ingest", "delete"} for e in store.memory.events)
    assert ids(store) == [r.id for r in records]


@pytest.mark.parametrize("failure", ["invalid_twice", "model_exception"])
@pytest.mark.parametrize("delete_confirmed", [False, True])
def test_expiry_stands_when_model_answer_fails(store, tmp_path, failure, delete_confirmed):
    expired, kept = seed(store, specs=[
        {"kind": "follow_up", "last_confirmed": TODAY - timedelta(days=FOLLOW_UP_EXPIRY_DAYS + 1)},
        {"kind": "goals", "fact": "A live plan.", "last_confirmed": EARLIER},
    ])
    if not delete_confirmed:
        store.memory.delete_answers.extend([False] * 3)
    bad = {"forget_everything": True, "actions": [
        act("delete", note_id="note-bob-001"),
        act("add", kind="goals", fact="Must not be written."),
    ]}
    llm = (ScriptedLLM(bad, bad) if failure == "invalid_twice" else
           ScriptedLLM(RuntimeError("model unavailable"), RuntimeError("model unavailable")))
    store.memory.events.clear()
    result = session(store, llm)
    assert result.failed and result.reason
    assert result.added == result.replaced == result.confirmed == []
    assert not any(e[0] == "ingest" for e in store.memory.events)
    assert ids(store) == [kept.id] and store.stored("alice") == {kept.id: kept}
    assert (expired.id in store.memory.entries) is not delete_confirmed
    expected_pending = [] if delete_confirmed else [expired.id]
    # Reason: expiry survives model failure and its unconfirmed remote deletion must be reported.
    assert result.undeleted == expected_pending
    assert result.deleted == []  # Expiry is not a delete action or "forget everything".
    assert store.pending_deletes("alice") == expected_pending
    restarted = NoteStore(store.memory, tmp_path)
    assert restarted.pending_deletes("alice") == expected_pending
    assert restarted.helper_notes("alice", TODAY) == [kept.fact]
    assert_committed(restarted, tmp_path)


@pytest.mark.parametrize("operation", ["ensure_audience", "recall"])
def test_take_notes_does_not_raise_for_past_failure_during_input(store, tmp_path, operation, monkeypatch):
    old = seed(store)[0]

    def unavailable(*args, **kwargs):
        raise PastError("memory unavailable")

    monkeypatch.setattr(store.memory, operation, unavailable)
    llm = ScriptedLLM({"actions": [act("delete", note_id=old.id)]})
    result = session(store, llm, remember_me=False)
    assert isinstance(result, NotesResult) and not result.failed
    assert len(llm.calls) == 1 and old.fact in llm.calls[0][1]
    assert result.deleted == [old.id]
    assert store.helper_notes("alice", TODAY) == [] and store.stored("alice") == {}
    assert set(result.undeleted) == set(store.pending_deletes("alice"))
    if old.id in store.memory.entries:
        assert old.id in store.pending_deletes("alice")
    assert_committed(store, tmp_path)


# Section 2: records, caps, and the ordered commit.

@pytest.mark.parametrize("kind", list(NOTE_CAPS))
def test_to_record_uses_supplied_identity_clock_and_only_follow_up_dates(kind):
    action = NoteAction(action="add", kind=kind, fact="A standalone sentence.", due_date=TODAY)
    record = to_record(action, user_id="alice", session_id="s-new", note_id="note-alice-007",
                       today=TODAY, now=NOW)
    assert isinstance(record, UserNote)
    assert (record.id, record.user_id, record.session_id) == ("note-alice-007", "alice", "s-new")
    assert record.kind == kind and record.fact == action.fact
    assert record.written_by == NOTE_TAKER and record.created_at == NOW
    assert record.last_confirmed == TODAY
    assert record.due_date == (TODAY if kind == "follow_up" else None)


@pytest.mark.parametrize("new_due", [None, TODAY + timedelta(days=4)])
def test_replace_follow_up_preserves_or_overrides_due_date(store, new_due):
    old = seed(store, specs=[{"kind": "follow_up", "due_date": TODAY}])[0]
    action = NoteAction(action="replace", note_id=old.id, kind="follow_up",
                        fact="An updated follow up.", due_date=new_due)
    existing = store.live_notes("alice", TODAY)[0]
    record = to_record(action, user_id="alice", session_id="new-session",
                       note_id="note-alice-002", today=TODAY, now=NOW, old=existing)
    assert record.due_date == (new_due if new_due is not None else TODAY)
    result = session(store, ScriptedLLM({"actions": [action.model_dump(mode="json")]}))
    assert result.replaced == [old.id] and len(result.added) == 1
    replacement = store.stored("alice")[result.added[0]]
    assert replacement.id != old.id and replacement.due_date == record.due_date
    assert old.id not in store.memory.entries


def test_confirm_updates_only_index_and_live_dates_without_past_write(store):
    old = seed(store)[0]
    store.memory.events.clear()
    result = session(store, ScriptedLLM({"actions": [act("confirm", note_id=old.id)]}))
    assert result.confirmed == [old.id] and result.added == []
    assert store.stored("alice")[old.id] == old
    existing = store.live_notes("alice", TODAY)[0]
    assert existing.last_confirmed == TODAY and existing.fact == old.fact
    assert not any(e[0] in {"ingest", "delete"} for e in store.memory.events)


def test_confirmed_follow_up_expires_using_new_index_date(store):
    old = seed(store, specs=[{
        "kind": "follow_up", "last_confirmed": TODAY - timedelta(days=FOLLOW_UP_EXPIRY_DAYS),
    }])[0]
    session(store, ScriptedLLM({"actions": [act("confirm", note_id=old.id)]}))
    assert store.stored("alice")[old.id].last_confirmed == old.last_confirmed
    assert [n.note_id for n in store.live_notes("alice", TODAY + timedelta(days=1))] == [old.id]
    assert store.expire("alice", TODAY + timedelta(days=FOLLOW_UP_EXPIRY_DAYS + 1)) == [old.id]


@pytest.mark.parametrize("kind", list(NOTE_CAPS))
def test_each_cap_drops_oldest_then_lower_number_and_new_notes_can_be_dropped(store, tmp_path, kind):
    cap = NOTE_CAPS[kind]
    old = seed(store, specs=[{"kind": kind, "last_confirmed": TODAY} for _ in range(cap)])
    store.memory.events.clear()
    # Confirmed dates tie, so numeric order, not action order, determines survivors.
    result = session(store, ScriptedLLM({"actions": [
        act("add", kind=kind, fact=f"New fact {i}.") for i in range(cap + 2)
    ]}))
    assert set(result.dropped_by_limit) == {r.id for r in old}
    assert result.not_written == 2 and len(result.added) == cap
    assert set(ids(store)) == set(result.added)
    assert not set(ids(store)) & {r.id for r in old}
    expected_ids = {f"note-alice-{n:03d}" for n in range(cap + 3, 2 * cap + 3)}
    assert set(result.added) == expected_ids
    ingested = [e[1].id for e in store.memory.events if e[0] == "ingest"]
    assert len(ingested) == cap and set(ingested) == expected_ids
    assert len(store.memory.entries) == cap
    assert_committed(store, tmp_path)


def test_caps_use_confirmation_date_before_note_number_and_keep_other_kinds(store):
    first, second, third = seed(store, specs=[
        {"last_confirmed": EARLIER},
        {"last_confirmed": TODAY - timedelta(days=1)},
        {"last_confirmed": TODAY},
    ])
    untouched = seed_extra(store, kind="people", fact="A sister.")
    result = session(store, ScriptedLLM({"actions": [
        act("confirm", note_id=first.id), act("add", kind="situation", fact="New situation."),
    ]}))
    assert result.dropped_by_limit == [second.id]
    assert first.id in ids(store) and third.id in ids(store) and untouched.id in ids(store)
    assert second.id not in store.memory.entries


def test_numeric_tie_break_at_999_is_not_lexicographic(store, tmp_path):
    folder = tmp_path / "notes/alice"
    folder.mkdir(parents=True)
    (folder / "counter.json").write_text('{"next": 999}', encoding="utf-8")
    old = seed(store, specs=[{"last_confirmed": TODAY} for _ in range(3)])
    assert old[0].id == "note-alice-999"
    result = session(store, ScriptedLLM({"actions": [
        act("add", kind="situation", fact="A later numbered note."),
    ]}))
    assert result.dropped_by_limit == ["note-alice-999"]
    assert "note-alice-1000" in ids(store)


@pytest.mark.parametrize("failure", ["ingest", "wait_false", "wait_exception"])
def test_failed_new_write_is_omitted_while_other_actions_commit(store, tmp_path, failure):
    replaced, deleted = seed(store, specs=[{}, {}])
    original_ingest = store.memory.ingest
    original_wait = store.memory.wait
    attempts = []
    failed_ids = set()
    failed_ingestions = set()

    def ingest(entry):
        attempts.append(entry.id)
        is_replacement = entry.content == "15 Nov 2026: Replacement."
        if is_replacement:
            failed_ids.add(entry.id)
            if failure == "ingest":
                raise PastError("write unavailable")
        ingestion_id = original_ingest(entry)
        if is_replacement:
            failed_ingestions.add(ingestion_id)
        return ingestion_id

    def wait(ingestion_id, timeout_s=60):
        if ingestion_id in failed_ingestions:
            if failure == "wait_false":
                return False
            if failure == "wait_exception":
                raise PastError("confirmation unavailable")
        return original_wait(ingestion_id, timeout_s)

    store.memory.ingest = ingest
    store.memory.wait = wait
    result = session(store, ScriptedLLM({"actions": [
        act("replace", note_id=replaced.id, kind="situation", fact="Replacement."),
        act("delete", note_id=deleted.id),
        act("add", kind="goals", fact="Another new note."),
    ]}))
    assert not result.failed and result.not_written == 1
    assert len(attempts) == 2 and len(result.added) == 1
    assert len(failed_ids) == 1
    assert set(result.added) == set(ids(store)) == set(attempts) - failed_ids
    assert not failed_ids & set(store.stored("alice"))
    assert result.replaced == [replaced.id]
    assert replaced.id not in ids(store) and deleted.id not in ids(store)
    assert replaced.id not in store.memory.entries and deleted.id not in store.memory.entries
    assert_committed(store, tmp_path)


def test_ordered_commit_hides_removals_before_write_and_publishes_only_confirmed_notes(store, tmp_path):
    replaced, deleted, confirmed = seed(store, specs=[{}, {}, {}])
    dropped = seed_extra(store, kind="goals", last_confirmed=EARLIER)
    seed_extra(store, kind="goals", last_confirmed=TODAY)
    seed_extra(store, kind="goals", last_confirmed=TODAY)
    removed = {replaced.id, deleted.id, dropped.id}
    before_ids = set(ids(store))
    observed = []

    def observe(stage, entry_id):
        current = store.index("alice")
        current_ids = {n.note_id for n in current.notes}
        assert not current_ids & removed
        assert next(n for n in current.notes if n.note_id == confirmed.id).last_confirmed == TODAY
        assert all(n <= NOTE_CAPS[k] for k, n in Counter(x.kind for x in current.notes).items())
        if stage in {"ingest", "wait"}:
            assert current_ids == before_ids - removed
        else:
            assert len(current_ids - before_ids) == 2
            assert current_ids <= set(store.stored("alice"))
        observed.append(stage)

    store.memory.observe = observe
    result = session(store, ScriptedLLM({"actions": [
        act("replace", note_id=replaced.id, kind="situation", fact="Replacement."),
        act("delete", note_id=deleted.id),
        act("confirm", note_id=confirmed.id),
        act("add", kind="goals", fact="A new goal."),
    ]}))
    store.memory.observe = None
    assert set(result.replaced) == {replaced.id}
    assert result.deleted == [deleted.id] and result.confirmed == [confirmed.id]
    assert result.dropped_by_limit == [dropped.id]
    assert len(result.added) == 2 and result.not_written == 0 and result.undeleted == []
    assert observed.count("ingest") == observed.count("wait") == 2
    assert observed.count("delete") == 3
    assert max(i for i, s in enumerate(observed) if s == "wait") < observed.index("delete")
    assert_committed(store, tmp_path)


# Ground rules: tracing consent belongs to the caller, never this piece.

@pytest.mark.parametrize("remember_me", [False, True])
def test_note_step_does_not_call_mlflow_or_wrap_the_model(store, tmp_path, monkeypatch, remember_me):
    import mlflow
    from puddles import tracing

    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")

    def forbidden(*args, **kwargs):
        pytest.fail("The notes piece must leave tracing to its caller")

    # Include aliases imported by either module, without assuming their names.
    forbidden_values = [value for value in vars(mlflow).values()
                        if callable(value) and str(getattr(value, "__module__", "")).startswith("mlflow")]
    forbidden_values.extend([tracing.setup, tracing.model_call,
                             tracing.TracedClient, tracing.TracedLLM])
    for module in (mlflow, note_store_api, notes_api):
        for name, value in list(vars(module).items()):
            if any(value is item for item in forbidden_values):
                monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(tracing, "setup", forbidden)
    monkeypatch.setattr(tracing, "model_call", forbidden)
    monkeypatch.setattr(tracing.TracedClient, "__init__", forbidden)
    monkeypatch.setattr(tracing.TracedLLM, "__init__", forbidden)
    old = seed(store)[0]
    llm = ScriptedLLM({"actions": [act("delete", note_id=old.id),
                                  act("add", kind="goals", fact="A new goal.")]})
    result = session(store, llm, remember_me=remember_me)
    assert not result.failed and result.deleted == [old.id] and len(llm.calls) == 1
    assert len(result.added) == int(remember_me)


# Reason: direct store calls must preserve the same person boundary as checked model actions.
@pytest.mark.parametrize("owner,other", [("alice", "bob"), ("user01", "user-01"),
                                         ("user-01-01", "user-01")])
def test_direct_store_calls_cannot_write_publish_or_delete_another_persons_id(store, owner, other):
    theirs = seed(store, owner)[0]
    mine = seed(store, other)[0]
    before = deepcopy(store.memory.entries)
    with pytest.raises(ValueError):
        store.write(note(theirs.id, user_id=other))
    with pytest.raises(ValueError):
        store.save_index(LiveNoteIndex(user_id=other, notes=[live(theirs)]))
    store.memory.events.clear()
    assert store.delete(other, [theirs.id]) == []
    assert not any(event[0] == "delete" for event in store.memory.events)
    assert store.memory.entries == before
    assert ids(store, owner) == [theirs.id] and ids(store, other) == [mine.id]
    assert store.pending_deletes(other) == []


# Reason: an interrupted local deletion must remain invisible and recoverable before remote retry.
@pytest.mark.parametrize("crash_after", ["pending_deletes.json", "index.json", "notes.json"])
def test_interrupted_take_off_is_completed_before_pending_remote_retry(store, tmp_path, monkeypatch,
                                                                      crash_after):
    removed, kept = seed(store, specs=[{"fact": "Forget this fact."}, {"fact": "Keep this fact."}])

    class ProcessStopped(BaseException):
        pass

    real_replace = os.replace

    def replace_then_stop(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if Path(dst).name == crash_after:
            raise ProcessStopped()

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace_then_stop)
        with pytest.raises(ProcessStopped):
            store.take_off("alice", [removed.id])

    restarted = NoteStore(store.memory, tmp_path)
    observations = []

    def observe(stage, entry_id):
        if stage == "delete":
            observations.append((entry_id, ids(restarted), set(restarted.stored("alice"))))

    store.memory.observe = observe
    assert restarted.delete("alice", []) == []
    store.memory.observe = None
    assert observations
    assert all(removed.id not in index_ids and removed.id not in stored_ids
               for _, index_ids, stored_ids in observations)
    assert ids(restarted) == [kept.id]
    assert restarted.stored("alice") == {kept.id: kept}
    assert removed.id not in store.memory.entries
    assert restarted.helper_notes("alice", TODAY) == [kept.fact]
    assert_committed(restarted, tmp_path)


# Reason: index pruning interrupted between files must still queue and delete the remote orphan.
@pytest.mark.parametrize("crash_after", ["index.json", "pending_deletes.json", "notes.json"])
def test_interrupted_index_pruning_recovers_on_next_session(store, tmp_path, monkeypatch, crash_after):
    removed, kept = seed(store, specs=[{}, {}])

    class ProcessStopped(BaseException):
        pass

    real_replace = os.replace

    def replace_then_stop(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if Path(dst).name == crash_after:
            raise ProcessStopped()

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace_then_stop)
        with pytest.raises(ProcessStopped):
            store.save_index(LiveNoteIndex(user_id="alice", notes=[live(kept)]))

    restarted = NoteStore(store.memory, tmp_path)
    assert restarted.helper_notes("alice", TODAY) == [kept.fact]
    result = session(restarted, ScriptedLLM({"actions": []}), remember_me=False)
    assert not result.failed and result.undeleted == []
    assert restarted.stored("alice") == {kept.id: kept}
    assert removed.id not in store.memory.entries
    assert_committed(restarted, tmp_path)


# Reason: forgetting everything must also remove a remote write interrupted before index publication.
@pytest.mark.parametrize("crash_after", ["remote_confirmation", "notes.json"])
def test_forget_everything_cleans_up_interrupted_remote_write(store, tmp_path, monkeypatch, crash_after):
    candidate = note(store.next_note_id("alice"), fact="Forget this interrupted write.")

    class ProcessStopped(BaseException):
        pass

    real_wait, real_replace = store.memory.wait, os.replace

    def wait_then_stop(*args, **kwargs):
        assert real_wait(*args, **kwargs)
        raise ProcessStopped()

    def replace_then_stop(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if Path(dst).name == "notes.json":
            raise ProcessStopped()

    with monkeypatch.context() as patch:
        if crash_after == "remote_confirmation":
            patch.setattr(store.memory, "wait", wait_then_stop)
        else:
            patch.setattr(os, "replace", replace_then_stop)
        with pytest.raises(ProcessStopped):
            store.write(candidate)

    assert candidate.id in store.memory.entries
    restarted = NoteStore(store.memory, tmp_path)
    assert restarted.helper_notes("alice", TODAY) == []
    result = session(restarted, ScriptedLLM({"forget_everything": True}), remember_me=False)
    assert result.forgot_everything and not result.failed
    assert ids(restarted) == [] and restarted.stored("alice") == {}
    assert candidate.id not in store.memory.entries
    assert result.undeleted == []


# Reason: the atomic-file rule applies to the counter and deletion queue as well as notes and index.
def test_all_store_files_use_same_folder_temporary_rename(store, monkeypatch):
    real_replace = os.replace
    destinations = set()

    def observe_replace(src, dst, *args, **kwargs):
        src, dst = Path(src), Path(dst)
        assert src != dst and src.parent == dst.parent
        json.loads(src.read_text())
        destinations.add(dst.name)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", observe_replace)
    old = seed(store)[0]
    assert store.delete("alice", [old.id]) == []
    assert destinations == {"counter.json", "notes.json", "index.json", "pending_deletes.json"}


class CutPointMemory(FakeUserMemory):
    """A remote fake with a KeyboardInterrupt checkpoint on either side of each call."""

    def __init__(self):
        super().__init__()
        self.checkpoint = lambda event: None
        self.wait_result = True

    def _around(self, name, operation):
        self.checkpoint(f"before:{name}")
        result = operation()
        self.checkpoint(f"after:{name}")
        return result

    def ensure_audience(self, audience, identities):
        return self._around("ensure", lambda: super(CutPointMemory, self).ensure_audience(audience, identities))

    def ingest(self, entry):
        return self._around("ingest", lambda: super(CutPointMemory, self).ingest(entry))

    def wait(self, ingestion_id, timeout_s=60):
        return self._around("wait", lambda: self.wait_result)

    def recall(self, query, identity, level="low", limit=20):
        return self._around("recall", lambda: super(CutPointMemory, self).recall(query, identity, level, limit))

    def delete(self, entry_id):
        return self._around("delete", lambda: super(CutPointMemory, self).delete(entry_id))


def raw_note_state(root):
    folder = root / "notes/alice"

    def read(name, default):
        path = folder / name
        return json.loads(path.read_text()) if path.exists() else default

    return {
        "index": {n["note_id"] for n in read("index.json", {"notes": []})["notes"]},
        "stored": set(read("notes.json", {})),
        "pending": set(read("pending_deletes.json", [])),
        "writing": {w["note_id"] for w in read("counter.json", {}).get("writing", [])},
    }


def stop_on_committed_file(monkeypatch, filename):
    real_replace = os.replace

    def replace_then_stop(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if Path(dst).name == filename:
            raise KeyboardInterrupt()

    monkeypatch.setattr(os, "replace", replace_then_stop)


def exercise_note_cut(root, monkeypatch, scenario, cut=None):
    memory = CutPointMemory()
    store = NoteStore(memory, root)
    reading_expiry = scenario in {"expire", "live_notes", "helper_notes", "build_input", "model_failure"}
    first, second, third = seed(store, specs=[
        {"kind": "follow_up" if reading_expiry else "situation",
         "fact": "First fact.", "last_confirmed": TODAY - timedelta(days=30) if reading_expiry else EARLIER},
        {"kind": "goals", "fact": "Second fact."},
        {"kind": "people", "fact": "Third fact."},
    ])
    candidate = note(store.next_note_id("alice"), kind="preferences", fact="New fact.")
    if scenario == "save_index":
        store.write(candidate)
    if scenario == "limit":
        seed_extra(store, fact="Another situation.")
        seed_extra(store, fact="A third situation.")
    if scenario == "recover":
        store.write(candidate)
        with monkeypatch.context() as patch:
            stop_on_committed_file(patch, "pending_deletes.json")
            with pytest.raises(KeyboardInterrupt):
                store.take_off("alice", [first.id])
        store = NoteStore(memory, root)
    if scenario == "wait_false":
        memory.wait_result = False
    if scenario == "delete_false":
        memory.fail_deletes = 100

    mixed = {"actions": [
        act("replace", note_id=first.id, kind="situation", fact="Replacement."),
        act("confirm", note_id=second.id), act("delete", note_id=third.id),
        act("add", kind="preferences", fact="New preference."),
    ]}
    calls = {
        "ensure_person": lambda: store.ensure_person("alice"),
        "next_note_id": lambda: store.next_note_id("alice"),
        "write": lambda: store.write(candidate),
        "wait_false": lambda: store.write(candidate),
        "save_index": lambda: store.save_index(LiveNoteIndex(user_id="alice", notes=[live(second), live(candidate)])),
        "take_off": lambda: store.take_off("alice", [first.id]),
        "delete": lambda: store.delete("alice", [first.id]),
        "delete_false": lambda: store.delete("alice", [first.id]),
        "expire": lambda: store.expire("alice", TODAY),
        "live_notes": lambda: store.live_notes("alice", TODAY),
        "helper_notes": lambda: store.helper_notes("alice", TODAY),
        "build_input": lambda: build_input(store, "alice", [], remember_me=False, today=TODAY),
        "session": lambda: session(store, ScriptedLLM(mixed)),
        "no_consent": lambda: session(store, ScriptedLLM(mixed), remember_me=False),
        "forget_everything": lambda: session(store, ScriptedLLM({**mixed, "forget_everything": True})),
        "limit": lambda: session(store, ScriptedLLM({"actions": [act("add", kind="situation", fact="New situation.")]})),
        "model_failure": lambda: session(store, ScriptedLLM(RuntimeError("offline model failure"))),
        "recover": lambda: store.helper_notes("alice", TODAY),
    }
    events, snapshots = [], []
    original_index = set(ids(store))

    def checkpoint(event):
        state = raw_note_state(root)
        snapshots.append((event, state, set(memory.entries)))
        events.append(event)
        if len(events) == cut:
            raise KeyboardInterrupt()

    real_replace = os.replace

    def replace(src, dst, *args, **kwargs):
        checkpoint(f"before:file:{Path(dst).name}")
        real_replace(src, dst, *args, **kwargs)
        checkpoint(f"after:file:{Path(dst).name}")

    memory.checkpoint = checkpoint
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace)
        try:
            calls[scenario]()
        except KeyboardInterrupt:
            assert cut is not None
        except PastError:
            assert scenario == "wait_false"
    memory.checkpoint = lambda event: None
    if cut is not None:
        assert len(events) == cut, (scenario, cut, events)

    forgotten = set()
    for event, state, remote in snapshots:
        known = state["index"] | state["stored"] | state["pending"] | state["writing"]
        assert remote <= known, (scenario, cut, event, "unknown remote note", remote - known)
        assert state["stored"] - state["index"] <= state["pending"] | state["writing"], (scenario, cut, event, state)
        forgotten |= state["pending"] | (original_index - state["index"])
        if event == "before:delete":
            assert not state["pending"] & (state["index"] | state["stored"] | state["writing"]), (scenario, cut, state)

    restarted = NoteStore(memory, root)
    facts = restarted.helper_notes("alice", TODAY)
    state = raw_note_state(root)
    assert not forgotten & (state["index"] | state["stored"]), (scenario, cut, state)
    assert not state["pending"] & (state["index"] | state["stored"] | state["writing"]), (scenario, cut, state)
    assert state["index"] == state["stored"]
    assert not state["writing"]
    assert set(memory.entries) <= state["index"] | state["pending"]
    assert facts == [r.fact for r in restarted.live_notes("alice", TODAY)]
    pending = restarted.delete("alice", [])
    assert set(memory.entries) <= set(ids(restarted)) | set(pending)
    assert not forgotten & set(ids(restarted))
    return events


# Reason: every file/remote boundary, including recovery itself, must survive a kill without resurrection or orphans.
@pytest.mark.parametrize("scenario", [
    "ensure_person", "next_note_id", "write", "wait_false", "save_index", "take_off", "delete", "delete_false",
    "expire", "live_notes", "helper_notes", "build_input", "session", "no_consent", "forget_everything",
    "limit", "model_failure", "recover",
])
def test_keyboard_interrupt_at_every_file_and_remote_boundary(tmp_path, monkeypatch, scenario):
    events = exercise_note_cut(tmp_path / "baseline", monkeypatch, scenario)
    assert events
    for cut in range(1, len(events) + 1):
        exercise_note_cut(tmp_path / f"cut-{cut}", monkeypatch, scenario, cut)


# Reason: the amended contract requires each mutating entry point to finish local recovery before its own work.
@pytest.mark.parametrize("operation", ["ensure_person", "next_note_id", "save_index", "write", "take_off",
                                       "delete", "expire", "live_notes", "helper_notes", "build_input", "take_notes"])
def test_every_mutating_entry_point_finishes_interrupted_local_delete(tmp_path, monkeypatch, operation):
    memory = FakeUserMemory()
    store = NoteStore(memory, tmp_path)
    gone, kept = seed(store, specs=[{}, {"kind": "goals"}])
    candidate = note(store.next_note_id("alice"), kind="preferences")
    with monkeypatch.context() as patch:
        stop_on_committed_file(patch, "pending_deletes.json")
        with pytest.raises(KeyboardInterrupt):
            store.take_off("alice", [gone.id])
    restarted = NoteStore(memory, tmp_path)
    calls = {
        "ensure_person": lambda: restarted.ensure_person("alice"),
        "next_note_id": lambda: restarted.next_note_id("alice"),
        "save_index": lambda: restarted.save_index(LiveNoteIndex(user_id="alice", notes=[live(kept)])),
        "write": lambda: restarted.write(candidate),
        "take_off": lambda: restarted.take_off("alice", []),
        "delete": lambda: restarted.delete("alice", []),
        "expire": lambda: restarted.expire("alice", TODAY),
        "live_notes": lambda: restarted.live_notes("alice", TODAY),
        "helper_notes": lambda: restarted.helper_notes("alice", TODAY),
        "build_input": lambda: build_input(restarted, "alice", [], remember_me=False, today=TODAY),
        "take_notes": lambda: session(restarted, ScriptedLLM({"actions": []}), remember_me=False),
    }
    calls[operation]()
    assert gone.id not in ids(restarted)
    assert gone.id not in restarted.stored("alice")


# Reason: a lost ingestion response cannot prove rejection; forgetting must still find any accepted remote copy.
@pytest.mark.parametrize("interrupt_cleanup", [False, True])
def test_lost_ingestion_response_cannot_erase_the_only_remote_write_intent(store, tmp_path, monkeypatch,
                                                                        interrupt_cleanup):
    candidate = note(store.next_note_id("alice"), fact="Forget the remotely accepted note.")
    original_ingest, real_replace = store.memory.ingest, os.replace
    counter_commits = []

    def accepted_without_response(entry):
        original_ingest(entry)
        raise PastError("connection lost after the server accepted ingestion")

    def replace_then_stop(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if Path(dst).name == "counter.json":
            counter_commits.append(dst)
            if len(counter_commits) == 2:
                raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(store.memory, "ingest", accepted_without_response)
        if interrupt_cleanup:
            patch.setattr(os, "replace", replace_then_stop)
        with pytest.raises(KeyboardInterrupt if interrupt_cleanup else PastError):
            store.write(candidate)
    assert candidate.id in store.memory.entries
    restarted = NoteStore(store.memory, tmp_path)
    result = session(restarted, ScriptedLLM({"forget_everything": True}), remember_me=False)
    assert result.forgot_everything and not result.failed
    assert candidate.id not in store.memory.entries
    assert ids(restarted) == [] and restarted.stored("alice") == {}
    assert result.undeleted == []
