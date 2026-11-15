"""The note store and the note-taker (docs/specs/notes.md), offline: a scripted model and FakeUserMemory."""
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from puddles.inference import notes
from puddles.inference.note_store import (
    NOTE_TAKER,
    RECALL_QUERY,
    FakeUserMemory,
    NoteStore,
    PastUserMemory,
    audience_for,
    check_user_id,
)
from puddles.inference.notes import NoteTakerOutput, build_input, check, render_user, take_notes, to_record
from puddles.llm import AgentFailed
from puddles.memory import TRAINING_IDENTITIES, PastError
from schemas import FOLLOW_UP_EXPIRY_DAYS, NOTE_CAPS, LiveNote, LiveNoteIndex, NoteAction, NoteTakerInput, Turn
from tests.fakes import ScriptedLLM

TODAY = date(2026, 11, 15)
NOW = datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc)
TURNS = [Turn(speaker="user", text="I'd rather you didn't remember me. And forget what I said about my sister."),
         Turn(speaker="helper", text="Of course. I won't keep anything new, and I'll forget that.")]


@pytest.fixture
def mem() -> FakeUserMemory:
    return FakeUserMemory()


@pytest.fixture
def store(mem: FakeUserMemory, tmp_path: Path) -> NoteStore:
    return NoteStore(mem, tmp_path)


def scripted(*outputs: dict) -> ScriptedLLM:
    llm = ScriptedLLM()
    llm.add(NoteTakerOutput, *outputs)
    return llm


def session(store: NoteStore, user_id: str, actions: list[dict], *, remember_me: bool = True,
            forget_everything: bool = False, today: date = TODAY, session_id: str = "sess-0001") -> notes.NotesResult:
    llm = scripted({"forget_everything": forget_everything, "actions": actions})
    return take_notes(store, llm, user_id=user_id, session_id=session_id, turns=TURNS, remember_me=remember_me,
                      today=today, now=NOW)


def add(kind: str, fact: str, **extra) -> dict:
    return {"action": "add", "kind": kind, "fact": fact, **extra}


def seed(store: NoteStore, user_id: str, *facts: tuple[str, str], today: date = TODAY) -> list[str]:
    """Notes written through a real note-taker session, so they are in every place a note lives."""
    result = session(store, user_id, [add(k, f) for k, f in facts], today=today, session_id="sess-seed")
    assert result.not_written == 0 and not result.failed
    return result.added


def files(store: NoteStore, user_id: str) -> dict:
    folder = store.root / "notes" / user_id
    return {p.name: json.loads(p.read_text()) for p in folder.glob("*.json")}


def index_ids(store: NoteStore, user_id: str) -> list[str]:
    return [n.note_id for n in store.index(user_id).notes]


# ---------- who is who ----------

@pytest.mark.parametrize("bad", ["", "User-01", "-user", "../etc", "a/b", "user_01", "x" * 34, "note-taker",
                                 "project", *TRAINING_IDENTITIES])
def test_refused_user_ids(store: NoteStore, bad: str) -> None:
    with pytest.raises(ValueError):
        check_user_id(bad)
    with pytest.raises(ValueError):
        store.index(bad)
    with pytest.raises(ValueError):
        store.live_notes(bad, TODAY)
    with pytest.raises(ValueError):
        take_notes(store, scripted(), user_id=bad, session_id="s", turns=TURNS, remember_me=True, today=TODAY)
    assert not (store.root / "notes").exists() or not any((store.root / "notes").iterdir())


def test_audience_and_identities(store: NoteStore, mem: FakeUserMemory) -> None:
    assert check_user_id("user-01") == "user-01" and check_user_id("x" * 33) == "x" * 33
    assert audience_for("user-01") == "person-user-01"
    store.ensure_person("user-01")
    assert mem.audiences["person-user-01"] == ["user-01", NOTE_TAKER]


# ---------- writing and the files ----------

def test_write_sends_one_entry_in_the_persons_audience(store: NoteStore, mem: FakeUserMemory) -> None:
    [note_id] = seed(store, "user-01", ("what_helps", "Slow breathing helps a bit."))
    assert note_id == "note-user-01-001"
    entry = mem.entries[note_id]
    assert entry.content == "15 Nov 2026: Slow breathing helps a bit."
    assert entry.label == "note-taker: user note" and entry.identity == NOTE_TAKER
    assert entry.audience == "person-user-01"
    assert entry.metadata == {"type": "user_note", "kind": "what_helps", "user_id": "user-01",
                              "session_id": "sess-seed"}
    assert set(files(store, "user-01")) == {"index.json", "notes.json", "counter.json"}
    assert store.next_note_id("user-01") == "note-user-01-002"


def test_write_that_past_dev_does_not_confirm_is_kept_nowhere(store: NoteStore, mem: FakeUserMemory) -> None:
    mem.wait = lambda ingestion_id, timeout_s=60: False
    note = to_record(NoteAction(action="add", kind="goals", fact="Wants to sleep earlier."), user_id="user-01",
                     session_id="s1", note_id=store.next_note_id("user-01"), today=TODAY, now=NOW)
    with pytest.raises(PastError):
        store.write(note)
    assert store.stored("user-01") == {} and index_ids(store, "user-01") == []
    # past.dev accepted the send and may still store it, so it is queued for deletion.
    assert store.pending_deletes("user-01") == [note.id]


def test_write_refuses_a_note_already_stored(store: NoteStore) -> None:
    [note_id] = seed(store, "user-01", ("goals", "Wants to sleep earlier."))
    again = to_record(NoteAction(action="add", kind="goals", fact="x"), user_id="user-01", session_id="s",
                      note_id=note_id, today=TODAY, now=NOW)
    with pytest.raises(ValueError):
        store.write(again)


def test_save_index_drops_notes_not_on_it_and_files_leave_no_temporaries(store: NoteStore) -> None:
    a, b = seed(store, "user-01", ("goals", "A."), ("goals", "B."))
    store.save_index(LiveNoteIndex(user_id="user-01", notes=[n for n in store.index("user-01").notes
                                                             if n.note_id == b]))
    assert set(store.stored("user-01")) == {b}
    assert store.pending_deletes("user-01") == [a]  # removed from notes.json, so queued for deletion
    assert not list((store.root / "notes" / "user-01").glob(".*"))


def test_note_numbers_are_never_reused(store: NoteStore) -> None:
    [first] = seed(store, "user-01", ("goals", "A."))
    session(store, "user-01", [{"action": "delete", "note_id": first}])
    [second] = seed(store, "user-01", ("goals", "B."))
    assert (first, second) == ("note-user-01-001", "note-user-01-002")


# ---------- test 12: "remember me" off, through a whole session ----------

def test_12_remember_me_off_drops_adds_and_replaces_and_applies_deletes(store: NoteStore,
                                                                         mem: FakeUserMemory) -> None:
    sister, work = seed(store, "user-01", ("people", "Has a sister they are close to."),
                        ("goals", "Planning to go back to work."))
    result = session(store, "user-01", [
        {"action": "delete", "note_id": sister},
        {"action": "replace", "note_id": work, "kind": "goals", "fact": "Not going back to work yet."},
        add("situation", "Does not want to be remembered."),
    ], remember_me=False)
    assert result.deleted == [sister] and result.not_written == 2 and result.added == [] and result.replaced == []
    assert index_ids(store, "user-01") == [work]
    assert set(store.stored("user-01")) == {work}
    assert sister not in mem.entries and work in mem.entries
    assert len(mem.entries) == 1 and result.undeleted == [] and store.pending_deletes("user-01") == []
    assert store.helper_notes("user-01", TODAY) == ["Planning to go back to work."]


def test_12_remember_me_off_drops_confirms(store: NoteStore) -> None:
    [work] = seed(store, "user-01", ("goals", "Planning to go back to work."), today=TODAY - timedelta(days=5))
    result = session(store, "user-01", [{"action": "confirm", "note_id": work}], remember_me=False)
    assert result.confirmed == []
    assert store.index("user-01").notes[0].last_confirmed == TODAY - timedelta(days=5)


def test_12_prompt_says_may_not_add(store: NoteStore) -> None:
    llm = scripted({"actions": []})
    take_notes(store, llm, user_id="user-01", session_id="s", turns=TURNS, remember_me=False, today=TODAY)
    assert "<may_add_notes>false</may_add_notes>" in llm.prompts[NoteTakerOutput][0]


# ---------- test 13: forget everything ----------

def test_13_forget_everything_empties_the_index_and_stores_nothing_new(store: NoteStore,
                                                                        mem: FakeUserMemory) -> None:
    a, b = seed(store, "user-01", ("people", "Has a sister."), ("goals", "Back to work."))
    result = session(store, "user-01", [
        add("situation", "Asked to be forgotten."),
        {"action": "replace", "note_id": b, "kind": "goals", "fact": "Not back to work."},
    ], forget_everything=True)
    assert result.forgot_everything and set(result.deleted) == {a, b}
    assert result.added == [] and result.not_written == 2
    assert store.index("user-01").notes == [] and store.stored("user-01") == {}
    assert mem.entries == {} and result.undeleted == []
    assert store.helper_notes("user-01", TODAY) == []


def test_13_forget_everything_works_with_remember_me_off(store: NoteStore, mem: FakeUserMemory) -> None:
    seed(store, "user-01", ("people", "Has a sister."))
    session(store, "user-01", [], forget_everything=True, remember_me=False)
    assert store.index("user-01").notes == [] and mem.entries == {}


# ---------- test 14: off the index means never returned ----------

def test_14_a_note_off_the_index_is_never_returned(store: NoteStore, mem: FakeUserMemory) -> None:
    keep, gone = seed(store, "user-01", ("goals", "Remember robot person goal."),
                      ("people", "Remember robot person sister."))
    mem.fail_deletes = 10
    result = session(store, "user-01", [{"action": "delete", "note_id": gone}])
    assert result.undeleted == [gone] and gone in mem.entries  # past.dev still holds it
    recalled = [s["sourceId"] for r in mem.recall(RECALL_QUERY, identity="user-01") for s in r["sources"]]
    assert gone in recalled  # the fake past.dev would hand it back to a search
    assert [n.note_id for n in store.live_notes("user-01", TODAY)] == [keep]
    assert store.helper_notes("user-01", TODAY) == ["Remember robot person goal."]
    inp = build_input(store, "user-01", TURNS, remember_me=True, today=TODAY)
    assert gone not in render_user(inp)


def test_live_notes_recalls_as_the_person_and_fills_missing_from_the_file(store: NoteStore,
                                                                         mem: FakeUserMemory) -> None:
    a, b = seed(store, "user-01", ("goals", "Zebra."), ("situation", "Works in a warehouse."))
    mem.recalls.clear()
    got = store.live_notes("user-01", TODAY)
    assert mem.recalls == [{"query": RECALL_QUERY, "identity": "user-01"}]
    assert [(n.note_id, n.kind, n.fact) for n in got] == [(a, "goals", "Zebra."),
                                                         (b, "situation", "Works in a warehouse.")]


def test_live_notes_survives_a_failed_recall(store: NoteStore, mem: FakeUserMemory) -> None:
    seed(store, "user-01", ("goals", "Zebra."))

    def broken(*args, **kwargs):
        raise PastError("503")

    mem.recall = broken
    assert store.helper_notes("user-01", TODAY) == ["Zebra."]


# ---------- test 15: unknown IDs and other people's IDs are refused ----------

def test_15_unknown_or_other_persons_note_id_is_refused(store: NoteStore, mem: FakeUserMemory) -> None:
    [theirs] = seed(store, "user-02", ("people", "Has a brother."))
    [mine] = seed(store, "user-01", ("goals", "Back to work."))
    inp = build_input(store, "user-01", TURNS, remember_me=True, today=TODAY)
    for bad in (theirs, "note-user-01-999"):
        for action in ("delete", "replace", "confirm"):
            out = NoteTakerOutput(actions=[NoteAction(action=action, note_id=bad, kind="goals", fact="x")])
            assert any(bad in p for p in check(out, inp)), (bad, action)
    bad_out = {"actions": [{"action": "delete", "note_id": theirs}]}
    result = take_notes(store, scripted(bad_out, bad_out), user_id="user-01", session_id="s", turns=TURNS,
                        remember_me=True, today=TODAY)
    assert result.failed and theirs in result.reason
    assert index_ids(store, "user-02") == [theirs] and theirs in mem.entries
    assert index_ids(store, "user-01") == [mine]


def test_check_lists_every_problem() -> None:
    inp = NoteTakerInput(today="2026-11-15", may_add_notes=True, turns=TURNS, existing_notes=[
        {"note_id": "note-user-01-001", "kind": "goals", "fact": "A.", "last_confirmed": "2026-11-01"}])
    assert check(NoteTakerOutput(actions=[]), inp) == []
    ok = NoteTakerOutput(actions=[NoteAction(action="confirm", note_id="note-user-01-001"),
                                  NoteAction(action="add", kind="goals", fact="B.")])
    assert check(ok, inp) == []
    cases = [
        NoteAction(action="delete"),
        NoteAction(action="replace", note_id="note-user-01-001", fact="no kind"),
        NoteAction(action="add", kind="goals"),
        NoteAction(action="add", kind="goals", fact="   "),
        NoteAction(action="add", kind="goals", fact="has id", note_id="note-user-01-001"),
    ]
    for action in cases:
        assert check(NoteTakerOutput(actions=[action]), inp), action
    twice = NoteTakerOutput(actions=[NoteAction(action="confirm", note_id="note-user-01-001"),
                                     NoteAction(action="delete", note_id="note-user-01-001")])
    assert check(twice, inp)


# ---------- test 16: a failed delete is retried ----------

def test_16_failed_delete_is_retried_in_the_same_call_and_the_next(store: NoteStore, mem: FakeUserMemory) -> None:
    a, b = seed(store, "user-01", ("goals", "A."), ("goals", "B."))
    mem.fail_deletes = 2  # two failed attempts, then the third succeeds within the same call
    assert store.delete("user-01", [a]) == []
    assert mem.delete_calls == [a, a, a] and a not in mem.entries

    mem.delete_calls.clear()
    mem.fail_deletes = 3  # every attempt in this call fails
    assert store.delete("user-01", [b]) == [b]
    assert mem.delete_calls == [b, b, b] and store.pending_deletes("user-01") == [b]
    assert index_ids(store, "user-01") == [] and b in mem.entries

    mem.delete_calls.clear()
    assert store.delete("user-01", []) == []  # the next call retries what is pending
    assert mem.delete_calls == [b] and b not in mem.entries and store.pending_deletes("user-01") == []


def test_16_take_notes_retries_pending_deletes_and_exceptions_count_as_failures(store: NoteStore,
                                                                                mem: FakeUserMemory) -> None:
    a, b = seed(store, "user-01", ("goals", "A."), ("goals", "B."))
    real_delete = mem.delete

    def raising(entry_id):
        raise PastError("timeout")

    mem.delete = raising
    result = session(store, "user-01", [{"action": "delete", "note_id": a}])
    assert result.undeleted == [a] and a in mem.entries and index_ids(store, "user-01") == [b]
    mem.delete = real_delete
    result = session(store, "user-01", [])
    assert result.undeleted == [] and a not in mem.entries and store.pending_deletes("user-01") == []


# ---------- test 17: one person's notes never reach another person or a training agent ----------

def test_17_notes_stay_with_their_person(store: NoteStore, mem: FakeUserMemory) -> None:
    seed(store, "user-01", ("people", "Remember robot person sister Ana."))
    seed(store, "user-02", ("people", "Remember robot person brother Ben."))
    assert store.helper_notes("user-01", TODAY) == ["Remember robot person sister Ana."]
    assert store.helper_notes("user-02", TODAY) == ["Remember robot person brother Ben."]
    llm = scripted({"actions": []})
    take_notes(store, llm, user_id="user-02", session_id="s", turns=TURNS, remember_me=True, today=TODAY)
    assert "Ana" not in llm.prompts[NoteTakerOutput][0] and "note-user01" not in llm.prompts[NoteTakerOutput][0]
    inp = build_input(store, "user-02", TURNS, remember_me=True, today=TODAY)
    assert all(n.note_id.startswith("note-user-02-") for n in inp.existing_notes)
    # Each person's own recall reaches only their own notes; a training identity reaches none.
    own = [s["sourceId"] for r in mem.recall(RECALL_QUERY, identity="user-01") for s in r["sources"]]
    assert own == ["note-user-01-001"]
    mem.ensure_training_audience()
    for identity in [*TRAINING_IDENTITIES, "project"]:
        assert mem.recall("Remember robot person sister brother Ana Ben", identity=identity) == []


# ---------- the model's answer failing ----------

def test_a_second_failed_answer_changes_nothing(store: NoteStore, mem: FakeUserMemory) -> None:
    a, b = seed(store, "user-01", ("goals", "A."), ("people", "B."))
    before_files, before_entries = files(store, "user-01"), dict(mem.entries)
    bad = {"actions": [{"action": "delete", "note_id": "note-user-01-777"}, add("goals", "New.")]}
    llm = scripted(bad, bad)
    result = take_notes(store, llm, user_id="user-01", session_id="s", turns=TURNS, remember_me=True, today=TODAY)
    assert result.failed and result.reason and result.added == [] and result.deleted == []
    assert len(llm.prompts[NoteTakerOutput]) == 2
    assert "failed these checks" in llm.prompts[NoteTakerOutput][1]
    assert files(store, "user-01") == before_files and mem.entries == before_entries


def test_a_model_call_that_raises_changes_nothing(store: NoteStore) -> None:
    seed(store, "user-01", ("goals", "A."))
    before = files(store, "user-01")

    class Broken:
        def structured(self, system, user, schema):
            raise AgentFailed("model refused")

    result = take_notes(store, Broken(), user_id="user-01", session_id="s", turns=TURNS, remember_me=True,
                        today=TODAY)
    assert result.failed and "refused" in result.reason and files(store, "user-01") == before


def test_a_first_failed_answer_is_retried(store: NoteStore) -> None:
    llm = scripted({"actions": [{"action": "delete", "note_id": "nope"}]}, {"actions": [add("goals", "A.")]})
    result = take_notes(store, llm, user_id="user-01", session_id="s", turns=TURNS, remember_me=True, today=TODAY)
    assert not result.failed and result.added == ["note-user-01-001"]


# ---------- the limits ----------

def test_limits_drop_the_oldest_confirmed_then_the_lower_number(store: NoteStore, mem: FakeUserMemory) -> None:
    old = seed(store, "user-01", ("situation", "S1."), today=TODAY - timedelta(days=9))
    mid = seed(store, "user-01", ("situation", "S2."), ("situation", "S3."), today=TODAY - timedelta(days=3))
    result = session(store, "user-01", [add("situation", "S4."), add("situation", "S5.")])
    # S1 is oldest; S2 and S3 tie, and S2 has the lower number.
    assert set(result.dropped_by_limit) == {old[0], mid[0]}
    assert set(result.added) == {"note-user-01-004", "note-user-01-005"} and result.not_written == 0
    assert index_ids(store, "user-01") == [mid[1], "note-user-01-004", "note-user-01-005"]
    assert old[0] not in mem.entries and mid[0] not in mem.entries and result.undeleted == []


def test_limits_can_drop_a_new_note_which_is_then_never_written(store: NoteStore, mem: FakeUserMemory) -> None:
    existing = seed(store, "user-01", *[("goals", f"G{i}.") for i in range(3)])
    # Seven goals, all dated today: the four lowest numbers go, three existing and the first new one.
    result = session(store, "user-01", [add("goals", "G3."), add("goals", "G4."), add("goals", "G5."),
                                        add("goals", "G6.")])
    assert set(result.dropped_by_limit) == set(existing)
    assert result.not_written == 1 and "note-user-01-004" not in mem.entries
    assert len(store.index("user-01").notes) == NOTE_CAPS["goals"]
    assert set(result.added) == {"note-user-01-005", "note-user-01-006", "note-user-01-007"}


def test_confirm_sets_last_confirmed_to_today_and_costs_no_write(store: NoteStore, mem: FakeUserMemory) -> None:
    [a] = seed(store, "user-01", ("goals", "A."), today=TODAY - timedelta(days=5))
    ingested = []
    real_ingest = mem.ingest
    mem.ingest = lambda entry: ingested.append(entry.id) or real_ingest(entry)
    result = session(store, "user-01", [{"action": "confirm", "note_id": a}])
    assert result.confirmed == [a] and ingested == []
    assert store.index("user-01").notes[0].last_confirmed == TODAY


def test_replace_writes_a_new_note_and_deletes_the_old(store: NoteStore, mem: FakeUserMemory) -> None:
    [old] = seed(store, "user-01", ("follow_up", "Ask about the interview."), )
    store.save_index(LiveNoteIndex(user_id="user-01", notes=[
        LiveNote(note_id=old, kind="follow_up", last_confirmed=TODAY, due_date=TODAY + timedelta(days=3))]))
    result = session(store, "user-01", [{"action": "replace", "note_id": old, "kind": "follow_up",
                                         "fact": "Ask how the interview went."},
                                        add("goals", "Wants a new job.", due_date="2026-12-01")])
    new, goal = result.added
    assert result.replaced == [old] and old not in mem.entries
    notes_ = {n.note_id: n for n in store.index("user-01").notes}
    assert notes_[new].due_date == TODAY + timedelta(days=3)  # a replacing follow_up keeps the old date
    assert notes_[goal].due_date is None  # a due date on any other kind is ignored
    assert store.stored("user-01")[new].parent_ids == ["sess-0001", old]


def test_a_write_that_fails_is_left_out_and_the_others_go_on(store: NoteStore, mem: FakeUserMemory) -> None:
    real_ingest = mem.ingest

    def flaky(entry):
        if entry.content.endswith("Bad."):
            raise PastError("400")
        return real_ingest(entry)

    mem.ingest = flaky
    result = session(store, "user-01", [add("goals", "Bad."), add("goals", "Good.")])
    assert result.not_written == 1 and result.added == ["note-user-01-002"]
    assert index_ids(store, "user-01") == ["note-user-01-002"] and set(store.stored("user-01")) == {"note-user-01-002"}


# ---------- the order of the four saves ----------

def test_the_four_saves_happen_in_order(mem: FakeUserMemory, tmp_path: Path) -> None:
    events: list[tuple] = []

    class Watched(NoteStore):
        def take_off(self, user_id, note_ids):
            events.append(("take_off", sorted(note_ids)))
            super().take_off(user_id, note_ids)

        def save_index(self, index):
            events.append(("save_index", [n.note_id for n in index.notes]))
            super().save_index(index)

        def write(self, note):
            events.append(("write", note.id, index_ids(self, note.user_id)))
            super().write(note)

        def delete(self, user_id, note_ids):
            events.append(("delete", sorted(note_ids), index_ids(self, user_id)))
            return super().delete(user_id, note_ids)

    store = Watched(mem, tmp_path)
    a, b, c = seed(store, "user-01", ("goals", "A."), ("goals", "B."), ("people", "C."))
    events.clear()
    session(store, "user-01", [{"action": "replace", "note_id": a, "kind": "goals", "fact": "A2."},
                               {"action": "delete", "note_id": c}, {"action": "confirm", "note_id": b}])
    new = "note-user-01-004"
    top = [e for e in events if e[0] in ("take_off", "write", "delete") or e[0] == "save_index"]
    names = [e[0] for e in top]
    first_take_off = names.index("take_off")
    first_write = names.index("write")
    last_delete = len(names) - 1 - names[::-1].index("delete")
    assert top[first_take_off] == ("take_off", sorted([a, c]))
    assert top[first_take_off + 1] == ("save_index", [b])  # confirms saved, without the new note
    assert top[first_write] == ("write", new, [b])  # the replaced and deleted notes are already off
    assert ("save_index", [b, new]) in top[first_write:last_delete]
    assert top[last_delete] == ("delete", sorted([a, c]), [b, new])


# ---------- a follow-up's expiry ----------

def test_follow_up_expiry(store: NoteStore, mem: FakeUserMemory) -> None:
    due, undated, fresh = seed(store, "user-01", ("follow_up", "Ask about the exam."),
                               ("follow_up", "Ask about the move."), ("follow_up", "Ask about the dog."))
    start = date(2026, 10, 1)
    store.save_index(LiveNoteIndex(user_id="user-01", notes=[
        LiveNote(note_id=due, kind="follow_up", last_confirmed=TODAY, due_date=start),
        LiveNote(note_id=undated, kind="follow_up", last_confirmed=start),
        LiveNote(note_id=fresh, kind="follow_up", last_confirmed=start, due_date=start + timedelta(days=10))]))
    last_day = start + timedelta(days=FOLLOW_UP_EXPIRY_DAYS)
    assert store.expire("user-01", last_day) == []  # still live on its last day
    assert store.expire("user-01", last_day + timedelta(days=1)) == [due, undated]
    assert index_ids(store, "user-01") == [fresh] and due not in mem.entries and undated not in mem.entries
    assert set(store.stored("user-01")) == {fresh}
    # live_notes expires first.
    assert store.live_notes("user-01", start + timedelta(days=10 + FOLLOW_UP_EXPIRY_DAYS + 1)) == []


# ---------- the prompt ----------

def test_render_user_shows_every_kind_with_count_and_limit(store: NoteStore) -> None:
    seed(store, "user-01", ("goals", "Back to work."))
    inp = build_input(store, "user-01", TURNS, remember_me=True, today=TODAY)
    assert inp.today == "2026-11-15" and inp.may_add_notes is True
    text = render_user(inp)
    for kind, limit in NOTE_CAPS.items():
        count = 1 if kind == "goals" else 0
        assert f'<{kind} count="{count}" limit="{limit}">' in text and f"</{kind}>" in text
    assert "- note-user-01-001, last confirmed 2026-11-15: Back to work." in text
    assert "<conversation>\nuser: I'd rather" in text and text.startswith("<today>2026-11-15</today>")


def test_system_prompt_asks_for_the_object() -> None:
    assert "forget_everything" in notes.SYSTEM and "JSON object" in notes.SYSTEM


# ---------- the real past.dev client, against a recorded transport (no network) ----------

def test_past_user_memory_follows_the_documented_calls() -> None:
    seen: list[tuple] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.method == "DELETE" and request.url.path.endswith("/missing"):
            return httpx.Response(404, json={"error": "data-point-not-found"})
        if request.method == "DELETE":
            return httpx.Response(202, json={"operationId": "op-1", "ids": [request.url.path.rsplit("/", 1)[1]]})
        return httpx.Response(201, json={})

    client = PastUserMemory(api_key="test-not-a-key", base_url="https://past.test/api/v1",
                            transport=httpx.MockTransport(handler))
    client.ensure_audience("person-user-01", ["user-01", NOTE_TAKER])
    assert client.delete("note-user-01-001") is True
    assert client.delete("missing") is False
    assert seen[0] == ("PUT", "/api/v1/audiences/person-user-01",
                       {"kind": "fixed", "name": "person-user-01", "identities": ["user-01", NOTE_TAKER]})
    assert seen[1][:2] == ("DELETE", "/api/v1/data-points/note-user-01-001")


# ---------- amended contract: two people never meet, and what survives a crash or a failure ----------

def test_note_ids_keep_the_user_id_as_it_is(store: NoteStore, mem: FakeUserMemory) -> None:
    [hyphen] = seed(store, "user-01", ("goals", "Hyphen person."))
    [plain] = seed(store, "user01", ("goals", "Plain person."))
    assert (hyphen, plain) == ("note-user-01-001", "note-user01-001")
    assert mem.entries[hyphen].audience == "person-user-01" and mem.entries[plain].audience == "person-user01"
    session(store, "user01", [{"action": "delete", "note_id": plain}])
    assert hyphen in mem.entries and store.helper_notes("user-01", TODAY) == ["Hyphen person."]
    assert store.helper_notes("user01", TODAY) == []


def test_another_persons_note_id_never_enters_their_files_or_a_delete(store: NoteStore,
                                                                     mem: FakeUserMemory) -> None:
    [theirs] = seed(store, "user01", ("goals", "Plain person."))
    note = to_record(NoteAction(action="add", kind="goals", fact="x"), user_id="user-01", session_id="s",
                     note_id=theirs, today=TODAY, now=NOW)
    with pytest.raises(ValueError):
        store.write(note)  # an ID of user01's cannot be written as user-01's note
    with pytest.raises(ValueError):
        store.save_index(LiveNoteIndex(user_id="user-01", notes=[
            LiveNote(note_id=theirs, kind="goals", last_confirmed=TODAY)]))
    assert store.delete("user-01", [theirs, "note-user-01-01-001"]) == []
    assert theirs in mem.entries and mem.delete_calls == [] and store.pending_deletes("user-01") == []
    assert index_ids(store, "user01") == [theirs]


def test_a_note_written_but_never_indexed_is_deleted_later(store: NoteStore, mem: FakeUserMemory) -> None:
    [kept] = seed(store, "user-01", ("goals", "Kept."))
    orphan = to_record(NoteAction(action="add", kind="goals", fact="Crash."), user_id="user-01", session_id="s",
                       note_id=store.next_note_id("user-01"), today=TODAY, now=NOW)
    store.write(orphan)  # then the process dies before the index is saved
    assert set(store.stored("user-01")) == {kept, orphan.id} and index_ids(store, "user-01") == [kept]
    assert store.helper_notes("user-01", TODAY) == ["Kept."]
    result = session(store, "user-01", [])
    assert orphan.id not in mem.entries and orphan.id not in store.stored("user-01") and result.undeleted == []


def test_take_notes_raises_nothing_when_the_audience_or_recall_fails(store: NoteStore,
                                                                      mem: FakeUserMemory) -> None:
    [a] = seed(store, "user-01", ("goals", "A."))

    def broken(*args, **kwargs):
        raise PastError("503")

    mem.ensure_audience = broken
    mem.recall = broken
    result = session(store, "user-01", [{"action": "delete", "note_id": a}, add("goals", "B.")])
    assert not result.failed and result.deleted == [a] and result.added == [] and result.not_written == 1
    assert index_ids(store, "user-01") == [] and a not in mem.entries and result.undeleted == []


def test_expiry_stands_when_the_answer_fails(store: NoteStore, mem: FakeUserMemory) -> None:
    [follow] = seed(store, "user-01", ("follow_up", "Ask about the exam."), today=date(2026, 10, 1))
    bad = {"actions": [{"action": "delete", "note_id": "nope"}]}
    result = take_notes(store, scripted(bad, bad), user_id="user-01", session_id="s", turns=TURNS,
                        remember_me=True, today=TODAY)
    assert result.failed and index_ids(store, "user-01") == [] and follow not in mem.entries


def test_replaced_is_listed_even_when_the_new_write_fails(store: NoteStore, mem: FakeUserMemory) -> None:
    [old] = seed(store, "user-01", ("goals", "Old."))

    def refuse(entry):
        raise PastError("400")

    mem.ingest = refuse
    result = session(store, "user-01", [{"action": "replace", "note_id": old, "kind": "goals", "fact": "New."}])
    assert result.replaced == [old] and result.added == [] and result.not_written == 1
    assert index_ids(store, "user-01") == [] and old not in mem.entries


def test_any_exception_from_the_model_is_a_failed_answer(store: NoteStore) -> None:
    seed(store, "user-01", ("goals", "A."))
    before = files(store, "user-01")

    class Down:
        def structured(self, system, user, schema):
            raise ConnectionError("no route to host")

    result = take_notes(store, Down(), user_id="user-01", session_id="s", turns=TURNS, remember_me=True,
                        today=TODAY)
    assert result.failed and "no route" in result.reason and files(store, "user-01") == before


# ---------- a process killed between two file writes: the next call finishes the job ----------

def stop_after(monkeypatch: pytest.MonkeyPatch, filename: str, nth: int = 1) -> None:
    """The process dies right after the nth rename onto `filename` (KeyboardInterrupt passes `except Exception`)."""
    real_replace, seen = os.replace, []

    def replace_then_stop(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if Path(dst).name == filename:
            seen.append(dst)
            if len(seen) == nth:
                raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", replace_then_stop)


def restart(store: NoteStore) -> NoteStore:
    return NoteStore(store.memory, store.root)


def assert_clean(store: NoteStore, user_id: str = "user-01") -> None:
    """Index and notes.json agree, nothing waits to be deleted, and no write is left in flight."""
    assert set(index_ids(store, user_id)) == set(store.stored(user_id))
    assert store.pending_deletes(user_id) == []
    assert set(json.loads((store.root / "notes" / user_id / "counter.json").read_text())) == {"next"}


@pytest.mark.parametrize("crash_after", ["pending_deletes.json", "index.json"])
def test_killed_take_off_is_finished_before_anything_else(store: NoteStore, mem: FakeUserMemory,
                                                          monkeypatch: pytest.MonkeyPatch, crash_after: str) -> None:
    gone, kept = seed(store, "user-01", ("people", "Forget me."), ("goals", "Keep me."))
    with monkeypatch.context() as patch:
        stop_after(patch, crash_after)
        with pytest.raises(KeyboardInterrupt):
            store.take_off("user-01", [gone])
    again = restart(store)
    assert again.helper_notes("user-01", TODAY) == ["Keep me."]
    assert gone not in index_ids(again, "user-01") and gone not in again.stored("user-01")
    seen = []
    real_delete = mem.delete
    mem.delete = lambda i: seen.append((i, set(index_ids(again, "user-01")), set(again.stored("user-01")))) \
        or real_delete(i)
    assert again.delete("user-01", []) == []
    assert seen and all(i not in ix and i not in st for i, ix, st in seen)
    assert gone not in mem.entries
    assert_clean(again)


@pytest.mark.parametrize("stop_at", ["before_send", "after_accept", "after_confirm", "after_notes_json"])
def test_killed_write_is_never_unknown_and_forget_everything_finds_it(store: NoteStore, mem: FakeUserMemory,
                                                                      monkeypatch: pytest.MonkeyPatch,
                                                                      stop_at: str) -> None:
    [kept] = seed(store, "user-01", ("goals", "Kept."))
    note = to_record(NoteAction(action="add", kind="people", fact="Half written."), user_id="user-01",
                     session_id="s", note_id=store.next_note_id("user-01"), today=TODAY, now=NOW)

    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        if stop_at == "before_send":
            patch.setattr(mem, "ingest", stop)
        elif stop_at == "after_accept":
            patch.setattr(mem, "wait", stop)
        elif stop_at == "after_confirm":
            real_wait = mem.wait
            patch.setattr(mem, "wait", lambda *a, **k: real_wait(*a, **k) and stop())
        else:
            stop_after(patch, "notes.json")
        with pytest.raises(KeyboardInterrupt):
            store.write(note)
    # The ID was written down before anything was sent.
    assert note.id in [w["note_id"] for w in json.loads((store.root / "notes/user-01/counter.json").read_text())
                       ["writing"]]
    again = restart(store)
    assert again.helper_notes("user-01", TODAY) == ["Kept."]
    result = session(again, "user-01", [], forget_everything=True, remember_me=False)
    assert result.forgot_everything and index_ids(again, "user-01") == [] and again.stored("user-01") == {}
    assert kept not in mem.entries and note.id not in mem.entries
    if stop_at == "before_send":
        # Never sent, so past.dev never confirms a delete; it stays queued and is never visible.
        assert result.undeleted == [note.id] and again.pending_deletes("user-01") == [note.id]
    else:
        assert result.undeleted == []
        assert_clean(again)


@pytest.mark.parametrize("crash_after", [("pending_deletes.json", 1), ("index.json", 1), ("counter.json", 1),
                                         ("notes.json", 1)])
def test_killed_save_index_is_finished_by_the_next_call(store: NoteStore, mem: FakeUserMemory,
                                                        monkeypatch: pytest.MonkeyPatch, crash_after) -> None:
    gone, kept = seed(store, "user-01", ("people", "Drop me."), ("goals", "Keep me."))
    new = to_record(NoteAction(action="add", kind="goals", fact="New."), user_id="user-01", session_id="s",
                    note_id=store.next_note_id("user-01"), today=TODAY, now=NOW)
    store.write(new)
    index = [n for n in store.index("user-01").notes if n.note_id == kept]
    index.append(LiveNote(note_id=new.id, kind="goals", last_confirmed=TODAY))
    with monkeypatch.context() as patch:
        stop_after(patch, *crash_after)
        with pytest.raises(KeyboardInterrupt):
            store.save_index(LiveNoteIndex(user_id="user-01", notes=index))
    again = restart(store)
    facts = again.helper_notes("user-01", TODAY)
    assert "Drop me." not in facts and "Keep me." in facts
    # The new note is live only if the index was saved; otherwise it is queued for deletion, never orphaned.
    published = crash_after[0] != "pending_deletes.json"
    assert ("New." in facts) is published
    result = session(again, "user-01", [])
    assert result.undeleted == [] and gone not in mem.entries and (new.id in mem.entries) is published
    assert_clean(again)


def test_killed_between_two_writes_of_one_session(store: NoteStore, mem: FakeUserMemory,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    [old] = seed(store, "user-01", ("goals", "Old goal."))
    real_ingest, calls = mem.ingest, []

    def second_dies(entry):
        calls.append(entry.id)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_ingest(entry)

    with monkeypatch.context() as patch:
        patch.setattr(mem, "ingest", second_dies)
        with pytest.raises(KeyboardInterrupt):
            session(store, "user-01", [{"action": "replace", "note_id": old, "kind": "goals", "fact": "New goal."},
                                       add("people", "Second.")])
    first = calls[0]
    again = restart(store)
    assert again.helper_notes("user-01", TODAY) == []  # the replaced note is off; the new one never published
    result = session(again, "user-01", [])
    assert set(result.undeleted) == {calls[1]}  # the second was never sent, so past.dev cannot confirm it
    assert old not in mem.entries and first not in mem.entries
    assert set(index_ids(again, "user-01")) == set(again.stored("user-01")) == set()


def test_killed_delete_after_past_dev_confirmed(store: NoteStore, mem: FakeUserMemory,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    gone, kept = seed(store, "user-01", ("people", "Forget me."), ("goals", "Keep me."))
    store.take_off("user-01", [gone])
    real_delete = mem.delete

    def confirm_then_die(entry_id):
        real_delete(entry_id)
        raise KeyboardInterrupt  # before the queue is saved

    with monkeypatch.context() as patch:
        patch.setattr(mem, "delete", confirm_then_die)
        with pytest.raises(KeyboardInterrupt):
            store.delete("user-01", [])
    assert gone not in mem.entries
    again = restart(store)
    assert again.helper_notes("user-01", TODAY) == ["Keep me."]
    # past.dev no longer has it, and a 404 is not a confirmation, so it stays queued; it is never visible.
    assert again.delete("user-01", []) == [gone] and gone not in again.stored("user-01")


def test_a_queued_note_cannot_go_back_on_the_index(store: NoteStore) -> None:
    gone, kept = seed(store, "user-01", ("people", "Forget me."), ("goals", "Keep me."))
    notes_before = store.index("user-01").notes
    store.take_off("user-01", [gone])
    with pytest.raises(ValueError):
        store.save_index(LiveNoteIndex(user_id="user-01", notes=notes_before))
    assert index_ids(store, "user-01") == [kept]
    store.write(to_record(NoteAction(action="add", kind="goals", fact="x"), user_id="user-01", session_id="s",
                          note_id=store.next_note_id("user-01"), today=TODAY, now=NOW))
    with pytest.raises(ValueError):
        store.write(to_record(NoteAction(action="add", kind="goals", fact="x"), user_id="user-01",
                              session_id="s", note_id=gone, today=TODAY, now=NOW))


# ---------- an intent is dropped only when the thing is known done ----------

def new_note(store: NoteStore, fact: str = "Sent but never answered.") -> "notes.UserNote":
    return to_record(NoteAction(action="add", kind="people", fact=fact), user_id="user-01", session_id="s",
                     note_id=store.next_note_id("user-01"), today=TODAY, now=NOW)


@pytest.mark.parametrize("killed_in_cleanup", [False, True])
def test_a_lost_ingest_response_still_queues_the_note(store: NoteStore, mem: FakeUserMemory,
                                                      monkeypatch: pytest.MonkeyPatch, killed_in_cleanup: bool) -> None:
    note = new_note(store)
    real_ingest = mem.ingest

    def stored_then_lost(entry):
        real_ingest(entry)  # past.dev stored it ...
        raise PastError("connection reset")  # ... and the answer never came back

    with monkeypatch.context() as patch:
        patch.setattr(mem, "ingest", stored_then_lost)
        if killed_in_cleanup:
            stop_after(patch, "counter.json", nth=2)  # the cleanup that drops the `writing` entry
        with pytest.raises(KeyboardInterrupt if killed_in_cleanup else PastError):
            store.write(note)
    assert note.id in mem.entries and store.pending_deletes("user-01") == [note.id]
    again = restart(store)
    result = session(again, "user-01", [], forget_everything=True, remember_me=False)
    assert note.id not in mem.entries and result.undeleted == []
    assert_clean(again)


def test_an_ingest_that_raises_is_queued_even_if_nothing_was_stored(store: NoteStore, mem: FakeUserMemory) -> None:
    note = new_note(store)

    def refused(entry):
        raise PastError("400")

    mem.ingest = refused
    with pytest.raises(PastError):
        store.write(note)
    # A refusal cannot be told apart from a lost answer, so the note is queued; it was never visible.
    assert store.pending_deletes("user-01") == [note.id] and store.stored("user-01") == {}


def test_a_failure_before_ingest_is_called_queues_nothing(store: NoteStore, mem: FakeUserMemory) -> None:
    note = new_note(store)
    sent = []
    mem.ingest = lambda entry: sent.append(entry) or "never"

    def no_audience(*args, **kwargs):
        raise PastError("503")

    mem.ensure_audience = no_audience
    with pytest.raises(PastError):
        store.write(note)
    assert sent == [] and store.pending_deletes("user-01") == []
    assert set(json.loads((store.root / "notes/user-01/counter.json").read_text())) == {"next"}


@pytest.mark.parametrize("operation", ["ensure_person", "next_note_id"])
def test_ensure_person_and_next_note_id_finish_an_interrupted_take_off(store: NoteStore, mem: FakeUserMemory,
                                                                       monkeypatch: pytest.MonkeyPatch,
                                                                       operation: str) -> None:
    gone, kept = seed(store, "user-01", ("people", "Forget me."), ("goals", "Keep me."))
    with monkeypatch.context() as patch:
        stop_after(patch, "pending_deletes.json")
        with pytest.raises(KeyboardInterrupt):
            store.take_off("user-01", [gone])
    assert gone in index_ids(store, "user-01")  # the kill left it on the index
    again = restart(store)
    getattr(again, operation)("user-01")
    assert index_ids(again, "user-01") == [kept] and gone not in again.stored("user-01")
    assert again.pending_deletes("user-01") == [gone] and gone in mem.entries  # local only; no remote call


def test_next_note_id_queues_a_write_left_by_a_stopped_store(store: NoteStore, mem: FakeUserMemory,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    note = new_note(store)

    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(mem, "wait", stop)
        with pytest.raises(KeyboardInterrupt):
            store.write(note)
    again = restart(store)
    assert again.next_note_id("user-01") == "note-user-01-002"
    assert again.pending_deletes("user-01") == [note.id]
    assert set(json.loads((store.root / "notes/user-01/counter.json").read_text())) == {"next"}
