"""The note store and the note-taker (docs/specs/notes.md), offline: a scripted model and FakeUserMemory."""
import json
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

@pytest.mark.parametrize("bad", ["", "User-01", "-user", "../etc", "a/b", "user_01", "x" * 65, "note-taker",
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
    assert check_user_id("user-01") == "user-01" and check_user_id("x" * 64) == "x" * 64
    assert audience_for("user-01") == "person-user-01"
    store.ensure_person("user-01")
    assert mem.audiences["person-user-01"] == ["user-01", NOTE_TAKER]


# ---------- writing and the files ----------

def test_write_sends_one_entry_in_the_persons_audience(store: NoteStore, mem: FakeUserMemory) -> None:
    [note_id] = seed(store, "user-01", ("what_helps", "Slow breathing helps a bit."))
    assert note_id == "note-user01-001"
    entry = mem.entries[note_id]
    assert entry.content == "15 Nov 2026: Slow breathing helps a bit."
    assert entry.label == "note-taker: user note" and entry.identity == NOTE_TAKER
    assert entry.audience == "person-user-01"
    assert entry.metadata == {"type": "user_note", "kind": "what_helps", "user_id": "user-01",
                              "session_id": "sess-seed"}
    assert set(files(store, "user-01")) == {"index.json", "notes.json", "counter.json"}
    assert store.next_note_id("user-01") == "note-user01-002"


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
    assert not list((store.root / "notes" / "user-01").glob(".*"))


def test_note_numbers_are_never_reused(store: NoteStore) -> None:
    [first] = seed(store, "user-01", ("goals", "A."))
    session(store, "user-01", [{"action": "delete", "note_id": first}])
    [second] = seed(store, "user-01", ("goals", "B."))
    assert (first, second) == ("note-user01-001", "note-user01-002")


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
    for bad in (theirs, "note-user01-999"):
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
        {"note_id": "note-user01-001", "kind": "goals", "fact": "A.", "last_confirmed": "2026-11-01"}])
    assert check(NoteTakerOutput(actions=[]), inp) == []
    ok = NoteTakerOutput(actions=[NoteAction(action="confirm", note_id="note-user01-001"),
                                  NoteAction(action="add", kind="goals", fact="B.")])
    assert check(ok, inp) == []
    cases = [
        NoteAction(action="delete"),
        NoteAction(action="replace", note_id="note-user01-001", fact="no kind"),
        NoteAction(action="add", kind="goals"),
        NoteAction(action="add", kind="goals", fact="   "),
        NoteAction(action="add", kind="goals", fact="has id", note_id="note-user01-001"),
    ]
    for action in cases:
        assert check(NoteTakerOutput(actions=[action]), inp), action
    twice = NoteTakerOutput(actions=[NoteAction(action="confirm", note_id="note-user01-001"),
                                     NoteAction(action="delete", note_id="note-user01-001")])
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
    assert all(n.note_id.startswith("note-user02-") for n in inp.existing_notes)
    # Each person's own recall reaches only their own notes; a training identity reaches none.
    own = [s["sourceId"] for r in mem.recall(RECALL_QUERY, identity="user-01") for s in r["sources"]]
    assert own == ["note-user01-001"]
    mem.ensure_training_audience()
    for identity in [*TRAINING_IDENTITIES, "project"]:
        assert mem.recall("Remember robot person sister brother Ana Ben", identity=identity) == []


# ---------- the model's answer failing ----------

def test_a_second_failed_answer_changes_nothing(store: NoteStore, mem: FakeUserMemory) -> None:
    a, b = seed(store, "user-01", ("goals", "A."), ("people", "B."))
    before_files, before_entries = files(store, "user-01"), dict(mem.entries)
    bad = {"actions": [{"action": "delete", "note_id": "note-user01-777"}, add("goals", "New.")]}
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
    assert not result.failed and result.added == ["note-user01-001"]


# ---------- the limits ----------

def test_limits_drop_the_oldest_confirmed_then_the_lower_number(store: NoteStore, mem: FakeUserMemory) -> None:
    old = seed(store, "user-01", ("situation", "S1."), today=TODAY - timedelta(days=9))
    mid = seed(store, "user-01", ("situation", "S2."), ("situation", "S3."), today=TODAY - timedelta(days=3))
    result = session(store, "user-01", [add("situation", "S4."), add("situation", "S5.")])
    # S1 is oldest; S2 and S3 tie, and S2 has the lower number.
    assert result.dropped_by_limit == [old[0], mid[0]]
    assert result.added == ["note-user01-004", "note-user01-005"] and result.not_written == 0
    assert index_ids(store, "user-01") == [mid[1], "note-user01-004", "note-user01-005"]
    assert old[0] not in mem.entries and mid[0] not in mem.entries and result.undeleted == []


def test_limits_can_drop_a_new_note_which_is_then_never_written(store: NoteStore, mem: FakeUserMemory) -> None:
    existing = seed(store, "user-01", *[("goals", f"G{i}.") for i in range(3)])
    # Seven goals, all dated today: the four lowest numbers go, three existing and the first new one.
    result = session(store, "user-01", [add("goals", "G3."), add("goals", "G4."), add("goals", "G5."),
                                        add("goals", "G6.")])
    assert result.dropped_by_limit == existing
    assert result.not_written == 1 and "note-user01-004" not in mem.entries
    assert len(store.index("user-01").notes) == NOTE_CAPS["goals"]
    assert result.added == ["note-user01-005", "note-user01-006", "note-user01-007"]


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
    assert result.not_written == 1 and result.added == ["note-user01-002"]
    assert index_ids(store, "user-01") == ["note-user01-002"] and set(store.stored("user-01")) == {"note-user01-002"}


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
    new = "note-user01-004"
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
    assert "- note-user01-001, last confirmed 2026-11-15: Back to work." in text
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
    assert client.delete("note-user01-001") is True
    assert client.delete("missing") is False
    assert seen[0] == ("PUT", "/api/v1/audiences/person-user-01",
                       {"kind": "fixed", "name": "person-user-01", "identities": ["user-01", NOTE_TAKER]})
    assert seen[1][:2] == ("DELETE", "/api/v1/data-points/note-user01-001")
