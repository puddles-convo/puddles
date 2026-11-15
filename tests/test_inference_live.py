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

ROWS = list(itertools.product([False, True], [False, True]))


def runs() -> list:
    import mlflow

    return mlflow.search_runs(search_all_experiments=True, output_format="list")


def row_args(remember_me: bool, training: bool, uri: str) -> list[str]:
    args = ["--remember-me", "yes" if remember_me else "no", "--training", "yes" if training else "no"]
    return args + (["--mlflow", uri] if training else [])


@pytest.mark.parametrize("apart", [False, True], ids=["notes-in-root", "notes-root-apart"])
@pytest.mark.parametrize("remember_me,training", ROWS)
def test_1_what_a_session_leaves_behind(remember_me, training, apart, tmp_path, store, capsys):
    """Contract tests 1 and 7: files, traces and runs, row by row, with the notes root in --root or apart."""
    import mlflow

    root = tmp_path / "root"
    notes_root = tmp_path / "people" if apart else root
    text = f"I keep thinking about the {PHRASE} again\nalso [note:situation:{FACT}]\n"
    extra = ["--notes-root", str(notes_root)] if apart else []
    code, out, err = chat(capsys, root, text, *row_args(remember_me, training, store), *extra)
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    assert code == 0
    assert out.splitlines()[-1] == (f"kept: {session_path(root, 's1')}" if training else "kept: nothing")
    assert notes_lines(err)[0] == "notes: 0 loaded"
    assert FACT not in err and PHRASE not in err  # the notes lines hold counts and IDs only
    assert runs() == []                           # no row starts an MLflow run

    in_root: list[Path] = []
    in_notes: list[Path] = []
    if training:
        in_root += [Path("sessions/s1.jsonl"), Path("sessions/s1.record.json")]
    if remember_me:
        assert index_ids(notes_root) == [f"note-{PERSON}-001"]
        assert "notes: 1 added, 0 replaced, 0 confirmed, 0 deleted" in err
        in_notes += [Path(f"notes/{PERSON}/{name}") for name in ("counter.json", "index.json", "notes.json")]
    else:
        assert not (notes_root / "notes").exists()  # no new notes
        assert "notes: 0 added, 0 replaced, 0 confirmed, 0 deleted" in err
    if apart:
        assert files(root) == sorted(in_root) and files(notes_root) == sorted(in_notes)
    else:
        assert files(root) == sorted(in_root + in_notes)

    if training:
        assert record_path(root, "s1").exists()
        spans = {s.name for t in traces for s in t.data.spans}
        assert {"turn", "llm:helper", "llm:crisis_checker", "llm:agent"} <= spans  # llm:agent is the note-taker
        assert holding(tmp_path, PHRASE)          # the search can see the phrase when it is stored
    else:
        assert traces == []
        # tmp_path holds --root, --notes-root, the MLflow store and ./mlruns.
        assert holding(tmp_path, PHRASE) == []
        assert PHRASE not in out + err
        where_fact = holding(tmp_path, FACT)
        if remember_me:
            assert where_fact == [notes_dir(notes_root) / "notes.json"]   # by their consent, and only there
            assert FACT not in out + err
        else:
            assert where_fact == []
            assert not root.exists() and not notes_root.exists()


class Failing:
    """A note-taker whose error repeats the person's words, as a real model's error text can."""

    untraced = True   # scripted: no request, no record of its own

    def structured(self, system, user, schema):
        raise RuntimeError(f"could not read the conversation: {user}")

    def structured_with_search(self, system, user, schema, search, max_searches):
        raise NotImplementedError


@pytest.mark.parametrize("remember_me,training", ROWS)
def test_6_cheap_a_note_taker_error_with_the_persons_words_goes_nowhere(remember_me, training, tmp_path, store,
                                                                       capsys, caplog, monkeypatch):
    import mlflow
    import scripts.chat

    monkeypatch.setattr(scripts.chat, "FakeNoteTaker", Failing)
    root = tmp_path / "root"
    code, out, err = chat(capsys, root, f"I keep thinking about the {PHRASE}\n",
                          *row_args(remember_me, training, store))
    mlflow.flush_trace_async_logging()
    assert code == 0
    assert notes_lines(err)[-1] == "notes: the notes step failed; not yet confirmed deleted: none"
    assert out.splitlines()[-1] == (f"kept: {session_path(root, 's1')}" if training else "kept: nothing")
    assert runs() == []
    if not training:
        assert mlflow.search_traces(return_type="list") == []
        assert holding(tmp_path, PHRASE) == []
        assert PHRASE not in out + err + caplog.text
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


def marked(llm):
    """A scripted model saying it makes no request and keeps no record, so it may run without training consent."""
    llm.untraced = True
    return llm


def test_finish_session_hands_the_turns_to_the_note_taker_and_clears_the_session(tmp_path):
    llm = marked(ScriptedLLM())
    llm.add(NoteTakerOutput, {"actions": []})
    session = make_session(user_id=PERSON, consent_for_training=False)
    record, notes = finish_session(session, NoteStore(FakeUserMemory(), tmp_path), llm)
    assert record is None and notes == NotesResult()
    [prompt] = llm.prompts[NoteTakerOutput]
    assert PHRASE in prompt and "<may_add_notes>false</may_add_notes>" in prompt
    assert session.turns == []
    assert not tmp_path.joinpath("notes").exists()  # nothing on file before or after: nothing left behind


def test_finish_session_with_no_turns_skips_the_note_taker(tmp_path):
    llm = marked(ScriptedLLM())
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


def test_finish_session_refuses_openai_llm_without_training_consent(tmp_path):
    from puddles.llm import OpenAILLM

    llm = OpenAILLM.__new__(OpenAILLM)  # no SDK client is built; the type alone is refused
    session = make_session(user_id=PERSON, remember_me=True, consent_for_training=False)
    with pytest.raises(ValueError):
        finish_session(session, NoteStore(FakeUserMemory(), tmp_path), llm)
    assert len(session.turns) == 2 and not (tmp_path / "notes").exists()


# ---------- real mode: PlainLLM over loop.llm's HTTP, OpenAILLM only with consent ----------

def completion(content: str) -> bytes:
    return json.dumps({"id": "c1", "object": "chat.completion", "created": 0, "model": "stand-in",
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": content}}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()


EMPTY_ANSWER = json.dumps({"forget_everything": False, "actions": []})


class Opener:
    """`urllib.request.urlopen` stubbed, as tests/unit/test_llm.py does: what `loop.llm` really sends, no socket.

    A request holding `pause_on` waits at the opener, before anything is sent, until `release` is set.
    `fail` answers HTTP 400 with an error whose text repeats the person's words.
    """

    def __init__(self, replies: list[str] | None = None, *, fail: bool = False, pause_on: str | None = None) -> None:
        import threading

        self.replies = list(replies or [])
        self.fail = fail
        self.pause_on = pause_on
        self.bodies: list[str] = []
        self.reached, self.release = threading.Event(), threading.Event()
        if pause_on is None:
            self.release.set()

    def __call__(self, request, timeout=None):
        import urllib.error

        body = request.data.decode()
        self.bodies.append(body)
        if self.pause_on is not None and self.pause_on in body:
            self.reached.set()
            assert self.release.wait(20)
        if self.fail:
            words = PHRASE if PHRASE in body else "?"
            error = json.dumps({"error": {"message": f"cannot use the words {words}"}}).encode()
            raise urllib.error.HTTPError(request.full_url, 400, f"Bad Request: {words}", {}, io.BytesIO(error))
        content = self.replies.pop(0) if self.replies else EMPTY_ANSWER

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        return Response(completion(content))


def stand_in_openai():
    """`OpenAILLM` itself, its SDK client on an `httpx.MockTransport`: no network, every request recorded."""
    import httpx
    from openai import OpenAI

    from puddles.llm import OpenAILLM

    bodies: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content.decode())
        return httpx.Response(200, content=completion(EMPTY_ANSWER), headers={"content-type": "application/json"})

    class StandIn(OpenAILLM):
        def __init__(self) -> None:
            super().__init__(model="stand-in", output_limits={"default": 100})
            self.client = OpenAI(api_key="placeholder", base_url="http://stand-in.invalid/v1", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(answer)))

    return StandIn, bodies


def real_mode(monkeypatch, opener: Opener, openai_llm) -> None:
    """Real mode with every outside service stood in for: helper endpoint, past.dev, both note-taker routes."""
    import urllib.request

    import loop.llm
    import scripts.chat

    path = crisis.load_path()
    monkeypatch.setattr(loop.llm, "client_from_env",
                        lambda prefix: FakeClient(lambda s, m: "none | - | ok" if s == path.system else "A reply."))
    monkeypatch.setattr(scripts.chat, "PastUserMemory", FakeUserMemory)
    monkeypatch.setattr(scripts.chat, "OpenAILLM", openai_llm)
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    for name in ("PAST_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "placeholder")
    monkeypatch.setenv("PUDDLES_AGENT_MODEL", "stand-in")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def real_session(tmp_path: Path, text: str, training: str, user: str, sid: str, uri: str) -> tuple[int, str]:
    import scripts.chat

    out = io.StringIO()
    code = scripts.chat.main(["--policy", POLICY, "--root", str(tmp_path / "root"), "--user-id", user,
                              "--session-id", sid, "--training", training, "--mlflow", uri],
                             stdin=io.StringIO(text), stdout=out)
    return code, out.getvalue()


def test_real_mode_without_training_uses_plain_http_and_no_setup(tmp_path, monkeypatch):
    class NoSDK:
        def __init__(self, *a, **k):
            pytest.fail("the OpenAI SDK was used without training consent")

    opener = Opener()
    real_mode(monkeypatch, opener, NoSDK)
    monkeypatch.setattr(tracing, "setup", lambda *a, **k: pytest.fail("setup without training consent"))
    code, out = real_session(tmp_path, f"the {PHRASE}\n", "no", PERSON, "s1", f"sqlite:///{tmp_path}/x.db")
    assert code == 0 and out.splitlines()[-1] == "kept: nothing" and not (tmp_path / "root").exists()
    [body] = opener.bodies                         # one request, the note-taker's, through loop.llm
    assert PHRASE in body and json.loads(body)["model"] == "stand-in"


def test_plain_llm_from_env_needs_key_and_model(monkeypatch):
    from loop.errors import ConfigError

    from puddles.inference.live import PlainLLM

    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.delenv("PUDDLES_AGENT_MODEL", raising=False)
    with pytest.raises(ConfigError):
        PlainLLM.from_env()
    monkeypatch.setenv("PUDDLES_AGENT_MODEL", "stand-in")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigError):
        PlainLLM.from_env()


def test_plain_llm_validates_the_answer_and_retries_once(monkeypatch):
    import urllib.request

    from loop.errors import SchemaError

    from puddles.inference.live import PlainLLM

    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.setenv("PUDDLES_AGENT_MODEL", "stand-in")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://stand-in.invalid/v1")
    opener = Opener(["not an object", EMPTY_ANSWER])
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    llm = PlainLLM.from_env()
    assert llm.structured("system", "user", NoteTakerOutput) == NoteTakerOutput()
    assert len(opener.bodies) == 2
    assert json.loads(opener.bodies[0])["response_format"]["type"] == "json_schema"
    opener.replies = ["not an object", "still not"]
    with pytest.raises(SchemaError):
        llm.structured("system", "user", NoteTakerOutput)


def plain_env(monkeypatch, **values: str) -> Opener:
    """The environment `PlainLLM.from_env` reads, with only `values` among the optional variables set."""
    import urllib.request

    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.setenv("PUDDLES_AGENT_MODEL", "stand-in")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://stand-in.invalid/v1")
    for name in ("PUDDLES_AGENT_TEMPERATURE", "PUDDLES_AGENT_MAX_TOKENS_PARAM", "PUDDLES_AGENT_TOOL_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    opener = Opener()
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    return opener


def test_plain_llm_sends_what_current_openai_models_accept_by_default(monkeypatch):
    """No temperature, the limit as `max_completion_tokens` only, as `OpenAILLM.structured` sends it."""
    from loop.llm import OpenAICompatClient
    from puddles.inference.live import PlainLLM

    opener = plain_env(monkeypatch)
    llm = PlainLLM.from_env()
    assert type(llm.client) is OpenAICompatClient
    assert llm.structured("system", "user", NoteTakerOutput) == NoteTakerOutput()
    [body] = [json.loads(b) for b in opener.bodies]
    assert "temperature" not in body
    assert body["max_completion_tokens"] == llm.max_tokens and "max_tokens" not in body
    assert "reasoning_effort" not in body and body["response_format"]["type"] == "json_schema"


def test_plain_llm_sends_the_same_body_as_openai_llm_structured(monkeypatch):
    """The same environment, both models: the same fields apart from how each sends the schema and the
    SDK's own `stream: false`, which is the server's default."""
    from puddles.inference.live import PlainLLM

    opener = plain_env(monkeypatch, PUDDLES_AGENT_TOOL_REASONING_EFFORT="none")
    StandIn, sdk_bodies = stand_in_openai()
    StandIn().structured("system", "user", NoteTakerOutput)
    PlainLLM(PlainLLM.from_env().client, max_tokens=100).structured("system", "user", NoteTakerOutput)
    [plain] = [json.loads(b) for b in opener.bodies]
    [sdk] = [json.loads(b) for b in sdk_bodies]
    for body in (plain, sdk):
        body.pop("response_format")
    assert sdk.pop("stream") is False
    assert plain == sdk


def test_plain_llm_temperature_and_limit_name_when_set(monkeypatch):
    from puddles.inference.live import PlainLLM

    opener = plain_env(monkeypatch, PUDDLES_AGENT_TEMPERATURE="0.3", PUDDLES_AGENT_MAX_TOKENS_PARAM="max_tokens")
    llm = PlainLLM.from_env()
    llm.structured("system", "user", NoteTakerOutput)
    [body] = [json.loads(b) for b in opener.bodies]
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == llm.max_tokens and "max_completion_tokens" not in body


def test_plain_llm_refuses_another_limit_name(monkeypatch):
    from loop.errors import ConfigError

    from puddles.inference.live import PlainLLM

    opener = plain_env(monkeypatch, PUDDLES_AGENT_MAX_TOKENS_PARAM="max_output_tokens")
    with pytest.raises(ConfigError, match="PUDDLES_AGENT_MAX_TOKENS_PARAM"):
        PlainLLM.from_env()
    assert opener.bodies == []


# ---------- 6. training on, then off, in the same process ----------

def test_tracing_off_does_not_import_mlflow():
    import subprocess
    import sys

    code = ("import sys, puddles.inference.live as live; live.tracing_off(); "
            "sys.exit(1 if 'mlflow' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0


EARLIER = "amber-lighthouse-kettle"          # typed in the consenting session, so traced by consent


def traced_with(words: str) -> list[str]:
    import mlflow

    mlflow.flush_trace_async_logging()
    return [s.name for t in mlflow.search_traces(return_type="list") for s in t.data.spans
            if words in json.dumps(s.inputs, default=str) + json.dumps(s.outputs, default=str)]


@pytest.mark.parametrize("fail", [False, True], ids=["answering", "failing"])
def test_6_training_on_then_off_in_one_process_leaves_no_trace(fail, tmp_path, capsys, caplog, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    model, sdk_bodies = stand_in_openai()
    opener = Opener(fail=fail)
    real_mode(monkeypatch, opener, model)
    try:
        assert real_session(tmp_path, f"before that, the {EARLIER}\n", "yes", PERSON, "s-on", uri)[0] == 0
        assert any("Completions" in name for name in traced_with(EARLIER))  # autologging traced the consenting one
        runs_before = runs()
        capsys.readouterr()
        caplog.clear()

        code, out = real_session(tmp_path, f"I keep thinking about the {PHRASE}\n", "no", PERSON, "s-off", uri)
        err = capsys.readouterr().err
        assert code == 0 and out.splitlines()[-1] == "kept: nothing"
        assert PHRASE in opener.bodies[0] and all(PHRASE not in b for b in sdk_bodies)  # asked, never via the SDK
        assert traced_with(PHRASE) == []
        assert runs() == runs_before
        assert holding(tmp_path, PHRASE) == []     # files, the sqlite store and ./mlruns
        assert PHRASE not in out + err + caplog.text
        assert ("notes: the notes step failed" in err) == fail
    finally:
        mlflow.openai.autolog(disable=True)
        mlflow.langchain.autolog(disable=True)


@pytest.mark.parametrize("fail", [False, True], ids=["answering", "failing"])
def test_6_overlap_a_consenting_session_cannot_trace_a_private_one_in_flight(fail, tmp_path, capsys, caplog,
                                                                            monkeypatch):
    """A (training off) waits just before its note-taker request; B (training on) runs to the end; A goes on."""
    import threading

    import mlflow

    import puddles.inference.notes as notes

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    model, sdk_bodies = stand_in_openai()
    opener = Opener(fail=fail)
    real_mode(monkeypatch, opener, model)
    reached, release = threading.Event(), threading.Event()
    render = notes.render_user

    def paused_render(inp):
        """The note-taker's prompt is built just before its request, whichever route it takes: A waits here."""
        if any(PHRASE in t.text for t in inp.turns):
            reached.set()
            assert release.wait(20)
        return render(inp)

    monkeypatch.setattr(notes, "render_user", paused_render)
    result: dict[str, tuple[int, str]] = {}
    person_a = threading.Thread(target=lambda: result.update(a=real_session(
        tmp_path, f"I keep thinking about the {PHRASE}\n", "no", PERSON, "s-a", uri)))
    try:
        capsys.readouterr()
        person_a.start()
        assert reached.wait(20)             # A's note-taker is about to send
        code_b, out_b = real_session(tmp_path, f"before that, the {EARLIER}\n", "yes", "person-b", "s-b", uri)
        assert code_b == 0 and out_b.splitlines()[-1].startswith("kept: ")
        assert any("Completions" in name for name in traced_with(EARLIER))  # autologging is on while A waits
        runs_before = runs()
        release.set()
        person_a.join(20)
        assert not person_a.is_alive()
        code_a, out_a = result["a"]
        err = capsys.readouterr().err
        assert code_a == 0 and out_a.splitlines()[-1] == "kept: nothing"
        assert traced_with(PHRASE) == []
        assert runs() == runs_before
        assert holding(tmp_path, PHRASE) == []     # files, the sqlite store, ./mlruns and trace artifacts
        assert PHRASE not in out_a + out_b + err + caplog.text
        assert ("notes: the notes step failed" in err) == fail
        assert any(PHRASE in b for b in opener.bodies) and all(PHRASE not in b for b in sdk_bodies)
    finally:
        release.set()
        person_a.join(20)
        mlflow.openai.autolog(disable=True)
        mlflow.langchain.autolog(disable=True)


# ---------- with consent off, only a model known to be untraced is let through ----------

STAND_IN_URL = "http://stand-in.invalid/v1"
ANSWERED = "copper-meadow-lantern"          # in the note-taker's answer only


def plain_client():
    from loop.llm import OpenAICompatClient

    return OpenAICompatClient(STAND_IN_URL, "stand-in", api_key="placeholder")


def priced_meter(root: Path):
    from puddles.usage import Meter, ModelPrice, Prices

    return Meter(root, Prices(models={"stand-in": ModelPrice(1.0, 1.0)}))


class SubClient(type(plain_client())):  # type: ignore[misc]
    """A subclass of OpenAICompatClient: it could override anything, so it is not the client itself."""


class Wrapper:
    """An object around an OpenAICompatClient that forwards every call: it could record anything."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def complete(self, *a, **k):
        return self.inner.complete(*a, **k)

    def complete_json(self, *a, **k):
        return self.inner.complete_json(*a, **k)


REFUSED = ["swapped-traced-client", "lying-plain", "subclass-client", "wrapper-client", "unmarked-scripted",
           "truthy-not-true"]


def refused_model(name: str, meter_root: Path):
    """A way round the allow-list, built without a request."""
    from puddles.inference.live import PlainLLM
    from puddles.tracing import TracedClient

    if name in ("swapped-traced-client", "lying-plain"):
        llm = PlainLLM(plain_client(), max_tokens=100)
        llm.client = TracedClient(plain_client(), "notes", meter=priced_meter(meter_root))  # after construction
        if name == "lying-plain":
            llm.untraced = True                 # a PlainLLM is judged by its client only
        return llm
    if name == "subclass-client":
        return PlainLLM(SubClient(STAND_IN_URL, "stand-in", api_key="placeholder"), max_tokens=100)
    if name == "wrapper-client":
        return PlainLLM(Wrapper(plain_client()), max_tokens=100)
    if name == "unmarked-scripted":
        return ScriptedLLM()
    llm = ScriptedLLM()
    llm.untraced = 1                            # truthy, but not True
    return llm


def test_verifiers_reproduction_is_refused_at_construction_and_nothing_is_written(tmp_path, store, monkeypatch):
    """PlainLLM(TracedClient(raw, "notes", meter=meter)): refused before any request; no trace, no usage file."""
    import urllib.request

    import mlflow

    from puddles.inference.live import PlainLLM
    from puddles.tracing import TracedClient

    opener = Opener()
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    meter_root = tmp_path / "meter"
    with pytest.raises(ValueError):
        PlainLLM(TracedClient(plain_client(), "notes", meter=priced_meter(meter_root)), max_tokens=100)
    assert opener.bodies == [] and not meter_root.exists()
    mlflow.flush_trace_async_logging()
    assert mlflow.search_traces(return_type="list") == [] and holding(tmp_path, PHRASE) == []


@pytest.mark.parametrize("name", REFUSED)
def test_without_training_consent_anything_not_known_untraced_is_refused(name, tmp_path, store, monkeypatch):
    import urllib.request

    import mlflow

    opener = Opener()
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    meter_root = tmp_path / "meter"
    llm = refused_model(name, meter_root)
    session = make_session(user_id=PERSON, remember_me=True, consent_for_training=False)
    with pytest.raises(ValueError):
        finish_session(session, NoteStore(FakeUserMemory(), tmp_path / "people"), llm)
    assert opener.bodies == [] and getattr(llm, "prompts", {}) == {}     # refused before any request
    assert len(session.turns) == 2 and not (tmp_path / "people").exists() and not meter_root.exists()
    mlflow.flush_trace_async_logging()
    assert mlflow.search_traces(return_type="list") == [] and holding(tmp_path, PHRASE) == []


def test_without_training_consent_a_marked_scripted_model_runs(tmp_path):
    llm = marked(ScriptedLLM())
    llm.add(NoteTakerOutput, {"actions": [{"action": "add", "kind": "situation", "fact": FACT}]})
    session = make_session(user_id=PERSON, remember_me=True, consent_for_training=False)
    record, notes = finish_session(session, NoteStore(FakeUserMemory(), tmp_path), llm)
    assert record is None and not notes.failed and len(notes.added) == 1
    assert PHRASE in llm.prompts[NoteTakerOutput][0]


@pytest.mark.parametrize("name", REFUSED)
def test_with_training_consent_all_of_them_are_accepted_as_before(name, tmp_path, store, monkeypatch):
    import urllib.request

    opener = Opener()
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    llm = refused_model(name, tmp_path / "meter")
    if isinstance(llm, ScriptedLLM):
        llm.add(NoteTakerOutput, {"actions": []})
    session = make_session(user_id=PERSON, remember_me=True, consent_for_training=True, root=tmp_path / "root")
    _, notes = finish_session(session, NoteStore(FakeUserMemory(), tmp_path / "people"), llm)
    assert not notes.failed                 # not refused, and the note-taker ran
    if isinstance(llm, ScriptedLLM):
        assert PHRASE in llm.prompts[NoteTakerOutput][0]
    else:
        assert len(opener.bodies) == 1 and PHRASE in opener.bodies[0]


def test_the_allowed_plain_llm_writes_nothing_of_its_own(tmp_path, store, monkeypatch, caplog):
    """Under the caller's meter and open span, with every logger listening: the request goes out and
    the answer comes back, and neither is in a usage line, a span or a log record."""
    import logging
    import urllib.request

    import mlflow

    from puddles.inference.live import PlainLLM
    from puddles.usage import metering

    caplog.set_level(logging.DEBUG)
    added = {"action": "add", "kind": "situation", "fact": f"likes the {ANSWERED}"}
    opener = Opener([json.dumps({"forget_everything": False, "actions": [added]})])
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    meter_root = tmp_path / "meter"
    people = tmp_path / "people"
    session = make_session(user_id=PERSON, remember_me=True, consent_for_training=False)
    with metering(priced_meter(meter_root), "notes"), mlflow.start_span(name="caller"):
        record, notes = finish_session(session, NoteStore(FakeUserMemory(), people),
                                       PlainLLM(plain_client(), max_tokens=100))
    mlflow.flush_trace_async_logging()
    assert record is None and not notes.failed and len(notes.added) == 1
    [body] = opener.bodies
    assert PHRASE in body                                   # it was asked
    assert not meter_root.exists()                          # no usage line
    spans = [s for t in mlflow.search_traces(return_type="list") for s in t.data.spans]
    assert [s.name for s in spans] == ["caller"]            # no span of its own under the caller's
    assert traced_with(PHRASE) == [] and traced_with(ANSWERED) == []
    assert PHRASE not in caplog.text and ANSWERED not in caplog.text
    assert holding(tmp_path, PHRASE) == []
    assert holding(tmp_path, ANSWERED) == [notes_dir(people) / "notes.json"]   # the note, by consent, only there
