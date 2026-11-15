"""The turn service (docs/specs/turn-service.md), reached in process with FastAPI's TestClient.

No network, no listening socket, no keys. The helper is scripted with `FakeClient`, the checker is
`scripts.chat.fake_checker` (it answers from markers such as [checker:crisis-self]), the note-taker is
`scripts.chat.FakeNoteTaker` (markers such as [note:<kind>:<fact>]) and past.dev is `FakeUserMemory`.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
from fastapi.testclient import TestClient

from loop.llm import FakeClient
from loop.policy import load_policy
from puddles.eval.crisis import load_keywords
from puddles.inference import crisis
from puddles.inference.note_store import FakeUserMemory, NoteStore
from puddles.inference.service import ServiceConfig, create_app
from puddles.inference.turn import Session, run_turn
from scripts.chat import FAKE_HELPER_LINE, FakeNoteTaker, fake_checker, fake_helper

REPO = Path(__file__).resolve().parents[1]
POLICY_SRC = REPO / "policies" / "policy-v1.md"
PATH = crisis.load_path()
TOKEN = "service-token-for-tests"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PERSON = "person-a"
PHRASE = "violet-harbour-umbrella"            # typed by the person
FACT = "works night shifts at the quayside bakery"
KEYWORD = load_keywords()[0]


# ---------- helpers ----------

def counting_helper() -> FakeClient:
    """A helper whose reply names how many messages it was shown, so each turn's reply differs."""
    return FakeClient(lambda system, messages: f"I hear you, and this is reply number {len(messages)}.")


def pairs(*helpers: Callable[[], FakeClient]) -> Callable[[], Any]:
    """A client factory: per session, the next helper and then a fake checker. After the list, `fake_helper`."""
    queue = list(helpers)
    state = {"n": 0}

    def factory() -> Any:
        state["n"] += 1
        if state["n"] % 2 == 0:
            return fake_checker(PATH)
        return (queue.pop(0) if queue else fake_helper)()

    return factory


def default_note_llm(consent: bool) -> Any:
    if not consent:
        return FakeNoteTaker()
    from puddles import tracing

    return tracing.TracedLLM(FakeNoteTaker())


def make_config(tmp_path: Path, *, policy: Optional[Path] = None, root: Optional[Path] = None,
                notes_root: Optional[Path] = None, factory: Optional[Callable[[], Any]] = None,
                note_llm_for: Callable[[bool], Any] = default_note_llm, **kw: Any) -> ServiceConfig:
    if policy is None:
        policy = tmp_path / "policy.md"
        policy.write_bytes(POLICY_SRC.read_bytes())
    root = root or tmp_path / "root"
    store = NoteStore(FakeUserMemory(), notes_root or root)
    return ServiceConfig(token=TOKEN, policy_path=policy, root=root, store=store,
                         client_factory=factory or pairs(), note_llm_for=note_llm_for, path=PATH, **kw)


def start(c: TestClient, user: str = PERSON, *, remember_me: bool = False, training: bool = False,
          session_id: Optional[str] = None) -> Any:
    body: dict[str, Any] = {"user_id": user, "remember_me": remember_me, "consent_for_training": training}
    if session_id is not None:
        body["session_id"] = session_id
    return c.post("/sessions", json=body, headers=AUTH)


def say(c: TestClient, sid: str, text: str) -> Any:
    return c.post(f"/sessions/{sid}/turns", json={"text": text}, headers=AUTH)


def end(c: TestClient, sid: str) -> Any:
    return c.post(f"/sessions/{sid}/end", headers=AUTH)


def session_of(app: Any, sid: str) -> Session:
    return app.state.service._entries[sid].session


def files(folder: Path) -> list[Path]:
    return sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


def holding(folder: Path, phrase: str) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and phrase.encode() in p.read_bytes()]


def wait_until(cond: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


# ---------- 1. three turns ----------

def test_1_three_turns_give_what_run_turn_gives(tmp_path):
    app = create_app(make_config(tmp_path, factory=pairs(counting_helper)))
    texts = ["Work has been heavy this week.", "I feel tired all the time.", "My manager keeps adding tasks."]
    with TestClient(app) as c:
        r = start(c, session_id="s1")
        assert r.status_code == 201
        assert r.json() == {"session_id": "s1", "policy_version": "policy-v1", "notes_loaded": 0,
                            "notes_failed": False}
        replies = [say(c, "s1", t) for t in texts]
        assert end(c, "s1").status_code == 200

    direct = Session(session_id="s1", policy=load_policy(POLICY_SRC), helper_client=counting_helper(),
                     checker_client=fake_checker(PATH), path=PATH, user_id=PERSON)
    expected = [run_turn(direct, t) for t in texts]
    for n, (r, want) in enumerate(zip(replies, expected), 1):
        assert r.status_code == 200
        body = r.json()
        assert body["turn"] == n and body["session_id"] == "s1"
        assert body["text"] == want.reply.text
        assert body["fixed_handoff_response"] is False and body["policy_version"] == "policy-v1"
        assert body["check_status"] == "answered" and body["risk"] == "none" and body["category"] is None
        assert body["keyword_matched"] is False and body["helper_failed"] is False
        assert set(body["timings_ms"]) == {"keyword", "checker", "helper", "total"}
        assert "reason" not in body


# ---------- 2. the crisis path ----------

def test_2_keyword_crisis_and_failed_check(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as c:
        start(c, session_id="s1")
        kw = say(c, "s1", f"Lately I keep thinking {KEYWORD} and nothing helps").json()
        assert kw["fixed_handoff_response"] is True and kw["keyword_matched"] is True
        assert kw["text"] in PATH.fixed_responses.values()
        assert KEYWORD not in {k: v for k, v in kw.items() if k != "text"}.values()

        scripted = say(c, "s1", "[checker:crisis-self] it is all too much").json()
        assert scripted["fixed_handoff_response"] is True and scripted["keyword_matched"] is False
        assert scripted["text"] == PATH.fixed_responses["self"]
        assert scripted["risk"] == "crisis" and scripted["category"] == "self"

        failed = say(c, "s1", "[checker:fail] what a week").json()
        assert failed["text"] == PATH.holding_line
        assert failed["check_status"] == "failed" and failed["risk"] is None and failed["category"] is None


# ---------- 3. the consent table ----------

ROWS = [(False, False), (True, False), (False, True), (True, True)]


@pytest.mark.parametrize("remember_me,training", ROWS)
def test_3_what_a_session_leaves_behind(remember_me, training, tmp_path, monkeypatch, caplog, capsys):
    import mlflow

    monkeypatch.chdir(tmp_path)                  # MLflow keeps some trace files in ./mlruns
    caplog.set_level(logging.DEBUG)
    root, notes_root = tmp_path / "root", tmp_path / "people"
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    app = create_app(make_config(tmp_path, root=root, notes_root=notes_root, mlflow_uri=uri))
    try:
        with TestClient(app) as c:
            assert start(c, session_id="s1", remember_me=remember_me, training=training).status_code == 201
            assert say(c, "s1", f"I keep thinking about the {PHRASE} again").status_code == 200
            assert say(c, "s1", f"also [note:situation:{FACT}]").status_code == 200
            ended = end(c, "s1").json()
    finally:
        mlflow.openai.autolog(disable=True)
        mlflow.langchain.autolog(disable=True)

    assert ended["kept"] is training and (ended["record"] is not None) is training
    assert ended["notes"]["added"] == (1 if remember_me else 0) and ended["notes"]["failed"] is False
    want_root = [Path("sessions/s1.jsonl"), Path("sessions/s1.record.json")] if training else []
    want_notes = [Path(f"notes/{PERSON}/{n}") for n in ("counter.json", "index.json", "notes.json")] \
        if remember_me else []
    assert files(root) == want_root
    assert files(notes_root) == want_notes
    printed = capsys.readouterr()
    if training:
        assert holding(root, PHRASE)             # the search can see the phrase when it is kept
    else:
        assert holding(root, PHRASE) == [] and holding(notes_root, PHRASE) == []
        assert holding(tmp_path, PHRASE) == []   # the policy copy, both roots, and no MLflow store at all
        assert PHRASE not in caplog.text and PHRASE not in printed.out + printed.err
        assert holding(tmp_path, FACT) == ([notes_root / "notes" / PERSON / "notes.json"] if remember_me else [])
    assert FACT not in caplog.text and PHRASE not in caplog.text  # no log line holds what was said


# ---------- 4. the token ----------

BAD_HEADERS = [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": f"bearer {TOKEN}"},
               {"Authorization": f"Bearer  {TOKEN}"}, {"Authorization": TOKEN}, {"Authorization": "Bearer"},
               {"Authorization": f"Token {TOKEN}"}]


@pytest.mark.parametrize("headers", BAD_HEADERS, ids=range(len(BAD_HEADERS)))
def test_4_every_route_but_health_needs_the_token(headers, tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}
        start(c, session_id="s1")
        say(c, "s1", "hello there")
        calls = [("/sessions", {"user_id": "person-b", "remember_me": False, "consent_for_training": False}),
                 ("/sessions", {"user_id": 7}),
                 ("/sessions/s1/turns", {"text": "another turn"}),
                 ("/sessions/s1/turns", {"text": ""}),
                 ("/sessions/nope/turns", {"text": "another turn"}),
                 ("/sessions/s1/end", None)]
        for url, body in calls:
            r = c.post(url, json=body, headers=headers)
            assert r.status_code == 401 and r.json() == {"error": "unauthorized"}, url
        r = c.post("/sessions", content=b"{not json", headers={**headers, "Content-Type": "application/json"})
        assert r.status_code == 401 and r.json() == {"error": "unauthorized"}
        assert app.state.service.live() == ["s1"]
        assert len(session_of(app, "s1").lines) == 1


# ---------- 5. consent answers ----------

@pytest.mark.parametrize("field", ["remember_me", "consent_for_training"])
@pytest.mark.parametrize("value", ["yes-ochre-answer", 1, "missing"])
def test_5_a_consent_answer_must_be_a_boolean(field, value, tmp_path):
    app = create_app(make_config(tmp_path))
    body: dict[str, Any] = {"user_id": PERSON, "remember_me": False, "consent_for_training": False}
    if value == "missing":
        del body[field]
    else:
        body[field] = value
    with TestClient(app) as c:
        r = c.post("/sessions", json=body, headers=AUTH)
    assert r.status_code == 422
    assert r.json() == {"error": "invalid_request", "fields": [field]}
    assert "ochre" not in r.text and PERSON not in r.text
    assert app.state.service.live() == []


def test_5_invalid_json_and_extra_fields(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as c:
        r = c.post("/sessions", content=b'{"user_id": "secret-words',
                   headers={**AUTH, "Content-Type": "application/json"})
        assert r.status_code == 422 and r.json() == {"error": "invalid_request", "fields": []}
        assert "secret-words" not in r.text
        r = c.post("/sessions", json=["a", "list"], headers=AUTH)
        assert r.status_code == 422 and r.json() == {"error": "invalid_request", "fields": []}
        r = c.post("/sessions", json={"user_id": "Bad/Id", "remember_me": False, "consent_for_training": False,
                                      "colour": "teal-secret"}, headers=AUTH)
        assert r.status_code == 422 and r.json() == {"error": "invalid_request", "fields": ["colour", "user_id"]}
        assert "teal-secret" not in r.text and "Bad/Id" not in r.text
    assert app.state.service.live() == []


# ---------- 6. turn text ----------

@pytest.mark.parametrize("text", ["", "   \n\t ", "longwords-" * 200 + "x"], ids=["empty", "blank", "2001"])
def test_6_a_turn_text_out_of_bounds_is_refused(text, tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as c:
        start(c, session_id="s1")
        say(c, "s1", "hello there")
        r = say(c, "s1", text)
        assert r.status_code == 422 and r.json() == {"error": "invalid_request", "fields": ["text"]}
        assert "longwords" not in r.text
        assert len(session_of(app, "s1").lines) == 1
        assert say(c, "s1", "x" * 2000).json()["turn"] == 2    # 2000 is allowed
        assert say(c, "s1", "  padded  ").json()["turn"] == 3


# ---------- 7. conflicts ----------

def test_7_conflicts(tmp_path):
    cfg = make_config(tmp_path)
    app = create_app(cfg)
    with TestClient(app) as c:
        assert start(c, session_id="s1").status_code == 201
        r = start(c, "person-b", session_id="s1")
        assert r.status_code == 409 and r.json() == {"error": "session_exists"}

        (cfg.root / "sessions").mkdir(parents=True)
        (cfg.root / "sessions" / "old.jsonl").write_text("")
        r = start(c, "person-b", session_id="old", training=True)
        assert r.status_code == 409 and r.json() == {"error": "session_exists"}
        (cfg.root / "sessions" / "old2.record.json").write_text("")
        r = start(c, "person-b", session_id="old2", training=True)
        assert r.status_code == 409 and r.json() == {"error": "session_exists"}

        r = start(c, PERSON, session_id="s2")
        assert r.status_code == 409 and r.json() == {"error": "user_busy"}
        assert app.state.service.live() == ["s1"]
        assert files(cfg.root) == [Path("sessions/old.jsonl"), Path("sessions/old2.record.json")]


def test_7_a_made_session_id_has_the_contract_form(tmp_path):
    import re

    app = create_app(make_config(tmp_path))
    with TestClient(app) as c:
        sid = start(c).json()["session_id"]
    assert re.fullmatch(r"sess-\d{14}-[0-9a-f]{8}", sid)


# ---------- 8. the session limit ----------

def test_8_too_many_sessions(tmp_path):
    app = create_app(make_config(tmp_path, max_sessions=2))
    with TestClient(app) as c:
        assert start(c, "person-a", session_id="a").status_code == 201
        assert start(c, "person-b", session_id="b").status_code == 201
        r = start(c, "person-c", session_id="c")
        assert r.status_code == 429 and r.json() == {"error": "too_many_sessions"}
        assert end(c, "a").status_code == 200
        assert start(c, "person-c", session_id="c").status_code == 201
        assert app.state.service.live() == ["b", "c"]


# ---------- 9. the policy is read per session ----------

def test_9_a_changed_policy_reaches_only_later_sessions(tmp_path):
    cfg = make_config(tmp_path)
    app = create_app(cfg)
    with TestClient(app) as c:
        assert start(c, "person-a", session_id="a").json()["policy_version"] == "policy-v1"
        text = cfg.policy_path.read_text().replace("# policy-v1", "# policy-v2", 1)
        cfg.policy_path.write_text(text)
        assert start(c, "person-b", session_id="b").json()["policy_version"] == "policy-v2"
        assert say(c, "a", "hello there").json()["policy_version"] == "policy-v1"
        assert say(c, "b", "hello there").json()["policy_version"] == "policy-v2"
        assert say(c, "a", "and again").json()["policy_version"] == "policy-v1"


# ---------- 10. no policy ----------

def test_10_a_missing_policy_file(tmp_path):
    cfg = make_config(tmp_path, policy=tmp_path / "missing.md")
    app = create_app(cfg)
    with TestClient(app) as c:
        r = start(c, session_id="s1")
    assert r.status_code == 503 and r.json() == {"error": "policy_unavailable"}
    assert app.state.service.live() == [] and not cfg.root.exists()


def test_10_a_model_that_cannot_be_built(tmp_path):
    def broken() -> Any:
        raise RuntimeError("secret-words in the message")

    app = create_app(make_config(tmp_path, factory=broken))
    with TestClient(app) as c:
        r = start(c, session_id="s1")
        assert r.status_code == 503 and r.json() == {"error": "model_unavailable"}
        assert app.state.service.live() == []
        assert start(c, session_id="s1").status_code == 503     # nothing was left reserved


# ---------- 11. after the end ----------

def test_11_a_turn_after_the_end_and_a_second_end(tmp_path):
    helper = FakeClient(lambda s, m: "a reply")
    app = create_app(make_config(tmp_path, factory=pairs(lambda: helper)))
    with TestClient(app) as c:
        start(c, session_id="s1")
        say(c, "s1", "hello there")
        assert end(c, "s1").status_code == 200
        calls = len(helper.calls)
        r = say(c, "s1", "are you there")
        assert r.status_code == 404 and r.json() == {"error": "session_not_found"}
        r = end(c, "s1")
        assert r.status_code == 404 and r.json() == {"error": "session_not_found"}
        assert say(c, "never", "hello").status_code == 404
        assert len(helper.calls) == calls


# ---------- 12. a turn and the end never overlap ----------

class Gate:
    """Blocks a call until released, and says when one is waiting."""

    def __init__(self) -> None:
        self.waiting = threading.Event()
        self.go = threading.Event()

    def wait(self) -> None:
        self.waiting.set()
        assert self.go.wait(10)


def test_12_the_end_waits_for_a_running_turn(tmp_path):
    gate = Gate()

    def slow(system: str, messages: list) -> str:
        gate.wait()
        return "a slow but full reply"

    app = create_app(make_config(tmp_path, factory=pairs(lambda: FakeClient(slow))))
    results: dict[str, Any] = {}
    with TestClient(app) as c:
        start(c, session_id="s1")
        t_turn = threading.Thread(target=lambda: results.update(turn=say(c, "s1", "hello there")))
        t_turn.start()
        assert gate.waiting.wait(10)
        t_end = threading.Thread(target=lambda: results.update(end=end(c, "s1")))
        t_end.start()
        wait_until(lambda: app.state.service._entries["s1"].ending)
        time.sleep(0.05)
        assert "end" not in results                 # the end is waiting for the turn
        gate.go.set()
        t_turn.join(10)
        t_end.join(10)
    assert results["turn"].status_code == 200 and results["turn"].json()["text"] == "a slow but full reply"
    assert results["end"].status_code == 200


class GatedNoteTaker(FakeNoteTaker):
    untraced = True

    def __init__(self, gate: Gate) -> None:
        self.gate = gate

    def structured(self, system: str, user: str, schema: Any) -> Any:
        self.gate.wait()
        return super().structured(system, user, schema)


def test_12_a_turn_sent_while_the_end_runs_reaches_no_model(tmp_path):
    gate = Gate()
    helper = FakeClient(lambda s, m: "a reply")
    app = create_app(make_config(tmp_path, factory=pairs(lambda: helper),
                                 note_llm_for=lambda consent: GatedNoteTaker(gate)))
    results: dict[str, Any] = {}
    with TestClient(app) as c:
        start(c, session_id="s1")
        say(c, "s1", "hello there")
        calls = len(helper.calls)
        t_end = threading.Thread(target=lambda: results.update(end=end(c, "s1")))
        t_end.start()
        assert gate.waiting.wait(10)                # the note-taker is running
        r = say(c, "s1", "one more thing")
        assert r.status_code == 404 and r.json() == {"error": "session_not_found"}
        assert len(helper.calls) == calls
        gate.go.set()
        t_end.join(10)
    assert results["end"].status_code == 200


# ---------- 13. idle sessions ----------

def test_13_sweep_ends_idle_sessions_with_the_note_taker(tmp_path):
    now = [0.0]
    cfg = make_config(tmp_path, idle_timeout_s=900.0, clock=lambda: now[0])
    app = create_app(cfg)
    note = f"note-{PERSON}-001"
    with TestClient(app) as c:
        start(c, PERSON, session_id="first", remember_me=True)
        say(c, "first", f"[note:situation:{FACT}]")
        assert end(c, "first").json()["notes"]["added"] == 1
        assert [n.note_id for n in cfg.store.index(PERSON).notes] == [note]

        start(c, PERSON, session_id="idle")
        say(c, "idle", f"please [forget:{note}]")
        now[0] = 500.0
        start(c, "person-b", session_id="fresh")
        now[0] = 899.0
        assert app.state.service.sweep() == []
        now[0] = 900.0
        assert app.state.service.sweep() == ["idle"]
        assert app.state.service.live() == ["fresh"]
        assert cfg.store.index(PERSON).notes == []
        assert say(c, "idle", "hello").status_code == 404


def test_13_a_session_that_fails_to_end_is_still_removed(tmp_path):
    now = [0.0]

    class Broken:
        untraced = True

        def structured(self, *a: Any) -> Any:
            raise RuntimeError("secret-words")

    cfg = make_config(tmp_path, idle_timeout_s=10.0, clock=lambda: now[0], note_llm_for=lambda consent: Broken())
    app = create_app(cfg)
    with TestClient(app) as c:
        start(c, session_id="s1")
        say(c, "s1", "hello there")
    now[0] = 10.0
    assert app.state.service.sweep() == ["s1"] and app.state.service.live() == []


# ---------- 14. sessions do not wait for each other ----------

def test_14_a_slow_helper_does_not_delay_another_session(tmp_path):
    gate = Gate()

    def slow(system: str, messages: list) -> str:
        gate.wait()
        return "the slow reply"

    app = create_app(make_config(tmp_path, factory=pairs(lambda: FakeClient(slow), fake_helper)))
    results: dict[str, Any] = {}
    with TestClient(app) as c:
        start(c, "person-a", session_id="a")
        start(c, "person-b", session_id="b")
        t = threading.Thread(target=lambda: results.update(a=say(c, "a", "hello there")))
        t.start()
        try:
            assert gate.waiting.wait(10)
            began = time.monotonic()
            r = say(c, "b", "hello there")
            assert r.status_code == 200 and r.json()["text"] == FAKE_HELPER_LINE
            assert time.monotonic() - began < 5
            assert "a" not in results
        finally:
            gate.go.set()
            t.join(10)
    assert results["a"].json()["text"] == "the slow reply"


# ---------- 15. scripts/serve.py ----------

def test_15_serve_main_with_fake_and_without_a_token(tmp_path, monkeypatch, capsys):
    from scripts.serve import main

    seen: dict[str, Any] = {}

    def stand_in(app: Any, host: str, port: int) -> None:
        seen.update(host=host, port=port)
        with TestClient(app) as c:
            r = start(c, session_id="s1")
            assert r.status_code == 201
            seen["turn"] = say(c, "s1", "hello there").json()
            seen["second"] = start(c, "person-b", session_id="s2").status_code
            seen["second_turn"] = say(c, "s2", "[checker:crisis-danger] hi").json()

    argv = ["--fake", "--policy", str(POLICY_SRC), "--root", str(tmp_path / "root")]
    monkeypatch.setenv("PUDDLES_SERVICE_TOKEN", TOKEN)
    assert main(argv, run=stand_in) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8765
    assert seen["turn"]["text"] == FAKE_HELPER_LINE and seen["turn"]["risk"] == "none"
    assert seen["second"] == 201 and seen["second_turn"]["text"] == PATH.fixed_responses["danger"]

    seen.clear()
    capsys.readouterr()
    monkeypatch.delenv("PUDDLES_SERVICE_TOKEN")
    assert main(argv, run=stand_in) == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("error: ")
    assert seen == {}


# ---------- run_turn raising ----------

def test_a_turn_that_raises_gives_the_offline_line_and_keeps_the_session(tmp_path, monkeypatch, caplog):
    import puddles.inference.service as service

    def boom(session: Any, text: str) -> Any:
        raise RuntimeError(f"failed on {text}")

    caplog.set_level(logging.DEBUG)
    app = create_app(make_config(tmp_path))
    with TestClient(app) as c:
        start(c, session_id="s1")
        monkeypatch.setattr(service, "run_turn", boom)
        r = say(c, "s1", f"about the {PHRASE}")
        assert r.status_code == 500 and r.json() == {"error": "turn_failed", "say": PATH.offline_line}
        assert app.state.service.live() == ["s1"]
        monkeypatch.undo()
        assert say(c, "s1", "hello again").status_code == 200
    assert PHRASE not in caplog.text and "RuntimeError" in caplog.text


# ---------- create_app refuses a bad config ----------

@pytest.mark.parametrize("change", [{"token": ""}, {"token": None}, {"max_sessions": 0}, {"idle_timeout_s": 0},
                                    {"idle_timeout_s": float("inf")}, {"idle_timeout_s": float("nan")}])
def test_create_app_refuses_a_bad_config(change, tmp_path):
    cfg = make_config(tmp_path)
    for k, v in change.items():
        setattr(cfg, k, v)
    with pytest.raises(ValueError):
        create_app(cfg)
