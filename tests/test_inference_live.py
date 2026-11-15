"""The notes joined to a live session (docs/specs/chat-notes.md), through `scripts.chat.main --fake`.

No network, no keys. The fake note-taker acts on markers in the messages: [note:<kind>:<fact>],
[forget:<note_id>] and [forget-everything].
"""
from __future__ import annotations

import io
import itertools
import json
from pathlib import Path

import pytest

from loop.llm import FakeClient
from puddles import tracing
from puddles.inference import crisis
from puddles.inference.live import finish_session, start_notes
from puddles.inference.note_store import FakeUserMemory, NoteStore
from puddles.inference.notes import NotesResult, NoteTakerOutput
from puddles.inference.session_log import record_path, session_path
from puddles.inference.turn import Session
from puddles.tracing import TracedLLM
from schemas import Turn
from tests.fakes import ScriptedLLM

REPO = Path(__file__).resolve().parents[1]
POLICY = str(REPO / "policies" / "policy-v1.md")
PHRASE = "violet-harbour-umbrella"            # typed by the person, never part of a note
FACT = "works night shifts at the quayside bakery"
OTHER = "has a sister called Ilse who visits on Sundays"
PERSON = "person-a"


def chat(capsys, root: Path, text: str, *extra: str, user: str = PERSON, session_id: str = "s1"):
    """Run one session; returns the exit code, standard output and standard error."""
    from scripts.chat import main

    capsys.readouterr()
    out = io.StringIO()
    code = main(["--fake", "--policy", POLICY, "--root", str(root), "--user-id", user, "--session-id", session_id,
                 *extra], stdin=io.StringIO(text), stdout=out)
    return code, out.getvalue(), capsys.readouterr().err


def notes_lines(err: str) -> list[str]:
    """The script's notes lines; MLflow may write its own lines on standard error when training is on."""
    return [ln for ln in err.splitlines() if ln.startswith("notes:")]


def files(folder: Path) -> list[Path]:
    return sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


def holding(folder: Path, phrase: str) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and phrase.encode() in p.read_bytes()]


def notes_dir(root: Path, user: str = PERSON) -> Path:
    return root / "notes" / user


def index_ids(root: Path, user: str = PERSON) -> list[str]:
    path = notes_dir(root, user) / "index.json"
    return [n["note_id"] for n in json.loads(path.read_text())["notes"]] if path.exists() else []


def stored_ids(root: Path, user: str = PERSON) -> list[str]:
    path = notes_dir(root, user) / "notes.json"
    return sorted(json.loads(path.read_text())) if path.exists() else []


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A tracking store in tmp_path, set up for every row, so a span written by mistake would be found."""
    monkeypatch.chdir(tmp_path)                  # MLflow keeps some trace files in ./mlruns
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    tracing.setup(tracking_uri=uri)
    yield uri


# ---------- 1. the four rows ----------

@pytest.mark.parametrize("remember_me,training", list(itertools.product([False, True], [False, True])))
def test_1_what_a_session_leaves_behind(remember_me, training, tmp_path, store, capsys):
    import mlflow

    root = tmp_path / "root"
    text = f"I keep thinking about the {PHRASE} again\nalso [note:situation:{FACT}]\n"
    args = ["--remember-me", "yes" if remember_me else "no", "--training", "yes" if training else "no"]
    if training:
        args += ["--mlflow", store]
    code, out, err = chat(capsys, root, text, *args)
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    assert code == 0
    assert out.splitlines()[-1] == (f"kept: {session_path(root, 's1')}" if training else "kept: nothing")
    assert notes_lines(err)[0] == "notes: 0 loaded"
    assert FACT not in err and PHRASE not in err  # the notes lines hold counts and IDs only

    expected: list[Path] = []
    if training:
        expected += [Path("sessions/s1.jsonl"), Path("sessions/s1.record.json")]
    if remember_me:
        assert index_ids(root) == [f"note-{PERSON}-001"]
        assert "notes: 1 added, 0 replaced, 0 confirmed, 0 deleted" in err
        expected += [Path(f"notes/{PERSON}/{name}") for name in ("counter.json", "index.json", "notes.json")]
    else:
        assert not (root / "notes").exists()  # no new notes
        assert "notes: 0 added, 0 replaced, 0 confirmed, 0 deleted" in err
    assert files(root) == sorted(expected)

    if training:
        assert record_path(root, "s1").exists()
        spans = {s.name for t in traces for s in t.data.spans}
        assert {"turn", "llm:helper", "llm:crisis_checker", "llm:agent"} <= spans  # llm:agent is the note-taker
        assert holding(tmp_path, PHRASE)         # the search can see the phrase when it is stored
    else:
        assert traces == []
        assert holding(tmp_path, PHRASE) == []
        assert PHRASE not in out + err
        where_fact = holding(tmp_path, FACT)
        if remember_me:
            assert where_fact == [notes_dir(root) / "notes.json"]   # by their consent, and only there
            assert FACT not in out + err
        else:
            assert where_fact == []
            assert not root.exists()


# ---------- 2. stage 4's test 12, through a whole session ----------

def test_2_remember_me_off_still_forgets(tmp_path, capsys):
    root = tmp_path / "root"
    code, _, err = chat(capsys, root, f"[note:situation:{FACT}]\n[note:people:{OTHER}]\n", "--remember-me", "yes")
    assert code == 0 and "notes: 2 added" in err
    first, second = f"note-{PERSON}-001", f"note-{PERSON}-002"
    assert index_ids(root) == [first, second] and stored_ids(root) == [first, second]
    counter = json.loads((notes_dir(root) / "counter.json").read_text())

    code, out, err = chat(capsys, root, f"please forget that [forget:{first}]\nand [note:goals:run a marathon]\n",
                          "--remember-me", "no", session_id="s2")
    assert code == 0 and out.splitlines()[-1] == "kept: nothing"
    assert notes_lines(err)[0] == "notes: 2 loaded"
    assert index_ids(root) == [second] and stored_ids(root) == [second]
    assert json.loads((notes_dir(root) / "counter.json").read_text()) == counter  # nothing new was added
    assert holding(tmp_path, FACT) == [] and holding(tmp_path, "marathon") == []
    # past.dev is in-process, so the second run's fake never held the note: its delete stays pending.
    assert notes_lines(err)[1] == ("notes: 0 added, 0 replaced, 0 confirmed, 1 deleted; "
                                   f"not yet confirmed deleted: {first}")


# ---------- 3. the notes reach the helper ----------

def test_3_loaded_notes_reach_the_helper_of_that_person_only(tmp_path, capsys, monkeypatch):
    import scripts.chat

    root = tmp_path / "root"
    chat(capsys, root, f"[note:situation:{FACT}]\n", "--remember-me", "yes")
    helpers: list[FakeClient] = []

    def recording_helper() -> FakeClient:
        helpers.append(FakeClient(lambda system, messages: scripts.chat.FAKE_HELPER_LINE))
        return helpers[-1]

    monkeypatch.setattr(scripts.chat, "fake_helper", recording_helper)
    chat(capsys, root, "hello again\n", session_id="s2")
    chat(capsys, root, "hello\n", "--remember-me", "yes", user="person-b", session_id="s3")
    mine, theirs = helpers
    assert mine.calls and FACT in mine.calls[0]["system"]
    assert theirs.calls and all(FACT not in c["system"] for c in theirs.calls)


# ---------- 4. forget everything ----------

def test_4_forget_everything_with_remember_me_on(tmp_path, capsys):
    root = tmp_path / "root"
    chat(capsys, root, f"[note:situation:{FACT}]\n[note:people:{OTHER}]\n", "--remember-me", "yes")
    code, _, err = chat(capsys, root, "[forget-everything] and [note:goals:learn the cello]\n",
                        "--remember-me", "yes", session_id="s2")
    assert code == 0
    assert index_ids(root) == [] and stored_ids(root) == []
    assert holding(tmp_path, "cello") == [] and holding(tmp_path, FACT) == []
    assert "notes: 0 added, 0 replaced, 0 confirmed, 2 deleted" in err


# ---------- 5. finish_session and consent ----------

def make_session(**kw) -> Session:
    session = Session(session_id="s-live", policy=None, helper_client=None,  # type: ignore[arg-type]
                      checker_client=None,  # type: ignore[arg-type]
                      path=crisis.load_path(), **kw)
    session.turns.extend([Turn(speaker="user", text=PHRASE), Turn(speaker="helper", text="ok")])
    return session


def test_5_finish_session_refuses_a_traced_model_without_training_consent(tmp_path):
    inner = ScriptedLLM()
    session = make_session(user_id=PERSON, remember_me=True, consent_for_training=False)
    with pytest.raises(ValueError):
        finish_session(session, NoteStore(FakeUserMemory(), tmp_path), TracedLLM(inner))
    assert inner.prompts == {} and len(session.turns) == 2 and not (tmp_path / "notes").exists()


def test_finish_session_with_no_user_id_does_nothing(tmp_path):
    session = make_session(consent_for_training=False)
    with pytest.raises(ValueError):
        finish_session(session, NoteStore(FakeUserMemory(), tmp_path), ScriptedLLM())
    assert len(session.turns) == 2


def test_finish_session_hands_the_turns_to_the_note_taker_and_clears_the_session(tmp_path):
    llm = ScriptedLLM()
    llm.add(NoteTakerOutput, {"actions": []})
    session = make_session(user_id=PERSON, consent_for_training=False)
    record, notes = finish_session(session, NoteStore(FakeUserMemory(), tmp_path), llm)
    assert record is None and notes == NotesResult()
    [prompt] = llm.prompts[NoteTakerOutput]
    assert PHRASE in prompt and "<may_add_notes>false</may_add_notes>" in prompt
    assert session.turns == []
    assert not tmp_path.joinpath("notes").exists()  # nothing on file before or after: nothing left behind


def test_finish_session_with_no_turns_skips_the_note_taker(tmp_path):
    llm = ScriptedLLM()
    session = make_session(user_id=PERSON)
    session.turns.clear()
    assert finish_session(session, NoteStore(FakeUserMemory(), tmp_path), llm) == (None, NotesResult())
    assert llm.prompts == {}


# ---------- the script's other promises ----------

def test_notes_root_is_separate_when_given(tmp_path, capsys):
    root, notes_root = tmp_path / "root", tmp_path / "people"
    chat(capsys, root, f"[note:goals:{FACT}]\n", "--remember-me", "yes", "--notes-root", str(notes_root))
    assert not root.exists()
    assert index_ids(notes_root) == [f"note-{PERSON}-001"]


def test_a_notes_failure_never_loses_the_session(tmp_path, capsys, monkeypatch):
    root = tmp_path / "root"
    monkeypatch.chdir(tmp_path)
    notes_dir(root).mkdir(parents=True)
    (notes_dir(root) / "index.json").write_text("not json")
    code, out, err = chat(capsys, root, f"the {PHRASE}\n", "--training", "yes",
                          "--mlflow", f"sqlite:///{tmp_path}/mlflow.db")
    assert code == 0
    assert out.splitlines()[-1] == f"kept: {session_path(root, 's1')}"
    assert record_path(root, "s1").exists()
    assert notes_lines(err) == ["notes: the notes step failed; none loaded",
                                  "notes: the notes step failed; not yet confirmed deleted: none"]


def test_start_notes_falls_back_to_the_files_when_past_dev_fails(tmp_path, capsys):
    chat(capsys, tmp_path, f"[note:situation:{FACT}]\n", "--remember-me", "yes")

    class Down(FakeUserMemory):
        def ensure_audience(self, audience, identities):
            raise RuntimeError("past.dev is down")

        def recall(self, *a, **k):
            raise RuntimeError("past.dev is down")

    assert start_notes(NoteStore(Down(), tmp_path), PERSON) == [FACT]


def test_missing_notes_keys_exit_2_before_any_message(tmp_path, capsys, monkeypatch):
    import loop.llm
    from scripts.chat import main

    monkeypatch.setattr(loop.llm, "client_from_env", lambda prefix: FakeClient(lambda s, m: "x"))
    for name in ("PAST_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    stdin = io.StringIO("hello\n")
    capsys.readouterr()
    assert main(["--policy", POLICY, "--root", str(tmp_path / "r")], stdin=stdin, stdout=io.StringIO()) == 2
    assert len(capsys.readouterr().err.splitlines()) == 1
    assert stdin.read() == "hello\n" and not (tmp_path / "r").exists()


def test_real_mode_without_training_uses_an_unwrapped_model_and_no_setup(tmp_path, capsys, monkeypatch):
    import loop.llm
    import scripts.chat

    seen: list[object] = []
    path = crisis.load_path()

    def answer(system, messages):
        return "none | - | ok" if system == path.system else "A reply."

    class Model(ScriptedLLM):
        def structured(self, system, user, schema):
            seen.append(self)
            return schema()

    monkeypatch.setattr(loop.llm, "client_from_env", lambda prefix: FakeClient(answer))
    monkeypatch.setattr(scripts.chat, "PastUserMemory", FakeUserMemory)
    monkeypatch.setattr(scripts.chat, "OpenAILLM", Model)
    monkeypatch.setattr(tracing, "setup", lambda *a, **k: pytest.fail("setup without training consent"))
    for name in ("PAST_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "placeholder")
    out = io.StringIO()
    assert scripts.chat.main(["--policy", POLICY, "--root", str(tmp_path / "r")],
                             stdin=io.StringIO("hello\n"), stdout=out) == 0
    assert len(seen) == 1 and type(seen[0]) is Model
    assert out.getvalue().splitlines()[-1] == "kept: nothing" and not (tmp_path / "r").exists()
