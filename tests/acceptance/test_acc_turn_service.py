"""Acceptance from docs/specs/turn-service.md and its named, existing dependencies.

Originally authored without reading the service or CLI implementation; strengthened
against the amended contract and the independent reviewer during verification.
Imports are deferred so collection also works before those modules exist.
Dependency patches replace the public function and service/CLI aliases by identity;
in particular, run_turn is patched at puddles.inference.turn.run_turn and at any
name imported from there into puddles.inference.service.
"""
from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timezone
import builtins
import importlib
import inspect
import io
import json
import logging
import os
from pathlib import Path
import re
import socket
import sys
from threading import Event, Thread, current_thread
from types import SimpleNamespace

from fastapi.testclient import TestClient
import mlflow
import pytest

from loop.errors import ConfigError, PolicyError
from loop.llm import FakeClient
from loop.policy import load_policy
from puddles import graph, tracing
from puddles.inference import crisis, live, note_store, turn
from puddles.inference.note_store import FakeUserMemory, NoteStore, RESERVED_IDS
from puddles.inference.notes import NotesResult, take_notes
from puddles.inference.session_log import read_lines, record_path, session_path
from schemas import CheckedMessage, ConversationRecord, HelperReply, Turn
from scripts import chat


TOKEN = "acceptance-token-never-disclose-7925"
PHRASE = "violet-harbour-umbrella-acceptance"
FACT = "works night shifts at the quayside bakery"
REPLY = "A scripted released reply."
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def patch_dependency(monkeypatch, owner, name, replacement):
    """Support both module-qualified calls and imported aliases, without source access."""
    original = getattr(owner, name)
    for module_name in ("puddles.inference.service", "scripts.serve"):
        module = sys.modules.get(module_name)
        if module is not None:
            for alias, value in list(vars(module).items()):
                if value is original:
                    monkeypatch.setattr(module, alias, replacement)
    monkeypatch.setattr(owner, name, replacement)


def disk(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def contains(root, words):
    return sorted(p for p in root.rglob("*") if p.is_file() and words.encode() in p.read_bytes())


def assert_error(response, status, error):
    assert response.status_code == status
    assert response.json() == {"error": error}


def assert_private(capsys, caplog, *secrets):
    captured = capsys.readouterr()
    # Inspect whole records as well as their formatted messages, including exc_info.
    direct = caplog._turn_service_records
    assert direct, "the directly attached service log handler was not exercised"
    logged = caplog.text + repr([vars(record) for record in caplog.records + direct])
    for secret in secrets:
        assert secret not in logged + captured.out + captured.err
    return captured


class Gate:
    def __init__(self, answer=REPLY):
        self.entered = Event()
        self.open = Event()
        self.answer = answer

    def __call__(self, *args, **kwargs):
        self.entered.set()
        if not self.open.wait(6.0):
            raise RuntimeError("acceptance gate timed out")
        return self.answer


class Background:
    """A bounded daemon request, with errors re-raised in the test thread."""
    def __init__(self, fn):
        self.done = Event()
        self.value = None
        self.error = None

        def work():
            try:
                self.value = fn()
            except BaseException as exc:
                self.error = exc
            finally:
                self.done.set()

        self.thread = Thread(target=work, daemon=True)
        self.thread.start()

    def result(self, timeout=2.0):
        assert self.done.wait(timeout), "request did not finish within its bound"
        if self.error is not None:
            raise self.error
        return self.value

    def join(self):
        self.thread.join(3)
        assert not self.thread.is_alive(), "acceptance request thread did not stop"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    # As in the existing acceptance suite, MLflow artifacts must stay in tmp_path.
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if (key.endswith("API_KEY") or key.startswith(("HELPER_", "OPENAI_", "MLFLOW_"))
                or key in ("PUDDLES_SERVICE_TOKEN", "PUDDLES_AGENT_MODEL")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    old_uri = mlflow.get_tracking_uri()
    from mlflow.tracking import fluent
    monkeypatch.setattr(fluent, "_active_experiment_id", fluent._active_experiment_id)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    def forbidden(*args, **kwargs):
        raise AssertionError("acceptance tests forbid network access and listening sockets")

    for name in ("connect", "connect_ex", "bind", "listen"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(old_uri)


@pytest.fixture
def api():
    return importlib.import_module("puddles.inference.service")


@pytest.fixture(autouse=True)
def capture_service_logger_directly(caplog):
    """Capture even when the service logger has its own handlers and does not propagate."""
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("puddles.inference.service")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    handler = Capture()
    logger.addHandler(handler)
    caplog._turn_service_records = records
    # The contract permits a quiet service. A harmless probe proves our handler
    # receives records without requiring the implementation to emit diagnostics.
    logger.debug("acceptance service log capture probe")
    assert records
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture
def path():
    return crisis.load_path()


@pytest.fixture
def make_service(api, path, policy_path, tmp_path):
    clients = []

    def make(*, pairs=None, note_llm=None, notes_with_sessions=False, **changes):
        made, note_flags, note_models = [], [], []
        scripted = iter(pairs) if pairs is not None else None
        pending = []

        def factory():
            if not pending:
                pending.extend(next(scripted) if scripted is not None else
                               (chat.fake_helper(), chat.fake_checker(path)))
            client = pending.pop(0)
            made.append(client)
            return client

        def note_for(training):
            note_flags.append(training)
            model = note_llm if note_llm is not None else chat.FakeNoteTaker()
            if training:
                model = tracing.TracedLLM(model)
            note_models.append(model)
            return model

        root = tmp_path / f"sessions-root-{len(clients)}"
        store = NoteStore(FakeUserMemory(), root if notes_with_sessions else
                          tmp_path / f"notes-root-{len(clients)}")
        values = dict(token=TOKEN, policy_path=policy_path, root=root, store=store,
                      client_factory=factory, note_llm_for=note_for, path=path,
                      mlflow_uri=f"sqlite:///{tmp_path}/mlflow.db", deadline_s=2.0, helper_wait_s=5.0)
        values.update(changes)
        config = api.ServiceConfig(**values)
        app = api.create_app(config)
        client = TestClient(app, raise_server_exceptions=False)
        clients.append(client)
        h = SimpleNamespace(config=config, app=app, client=client, service=app.state.service,
                            made=made, note_flags=note_flags, note_models=note_models,
                            root=Path(config.root), store=config.store)

        def start(sid="s1", user="person-a", remember=False, training=False, **extra):
            body = dict(user_id=user, remember_me=remember, consent_for_training=training,
                        session_id=sid)
            body.update(extra)
            return client.post("/sessions", headers=AUTH, json=body)

        h.start = start
        h.say = lambda sid="s1", text="A quiet day.": client.post(
            f"/sessions/{sid}/turns", headers=AUTH, json={"text": text})
        h.end = lambda sid="s1": client.post(f"/sessions/{sid}/end", headers=AUTH)
        return h

    yield make
    for client in clients:
        client.close()


@pytest.fixture
def local_tracking(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    tracing.setup(tracking_uri=uri)
    yield uri
    mlflow.flush_trace_async_logging()
    mlflow.openai.autolog(disable=True)
    mlflow.langchain.autolog(disable=True)


def seed_notes(h, *facts, user="person-a"):
    result = take_notes(h.store, chat.FakeNoteTaker(), user_id=user, session_id="seed",
                        turns=[Turn(speaker="user", text=" ".join(
                            f"[note:situation:{fact}]" for fact in facts))], remember_me=True)
    assert not result.failed and len(result.added) == len(facts)
    return result.added


# 1. Safety and the single release authority.

def test_three_turns_are_exactly_the_replies_released_by_run_turn(make_service, path, policy_path):
    messages = ["  A quiet day.  ", "Some uncertainty. [checker:concern]", "Another thought."]
    h = make_service()
    assert h.start().status_code == 201
    direct = turn.Session(session_id="s1", policy=load_policy(policy_path), path=path,
                          helper_client=chat.fake_helper(), checker_client=chat.fake_checker(path))
    for number, message in enumerate(messages, 1):
        expected = turn.run_turn(direct, message.strip())
        response = h.say(text=message)
        assert response.status_code == 200
        body = response.json()
        for field in ("text", "turn", "policy_version", "fixed_handoff_response"):
            assert body[field] == getattr(expected.reply, field)
        assert body["session_id"] == "s1" and body["turn"] == number
        assert body["risk"] == expected.checked.risk
        assert body["category"] == expected.checked.category
        assert body["check_status"] == "answered"
        assert body["keyword_matched"] is False and body["helper_failed"] is False
        assert set(body["timings_ms"]) == {"keyword", "checker", "helper", "total"}
        assert all(type(value) is int for value in body["timings_ms"].values())
    assert h.made[0].calls[0]["messages"][-1]["content"] == messages[0].strip()


def test_service_uses_the_released_reply_and_does_not_make_another_decision(make_service, monkeypatch):
    h = make_service()
    assert h.start().status_code == 201
    called = []

    def released(session, text):
        called.append((session.session_id, text))
        return turn.TurnResult(
            reply=HelperReply(session_id="s1", turn=17, text=REPLY,
                              policy_version="release-owned-version", fixed_handoff_response=False),
            keyword=None, check_status="answered",
            checked=CheckedMessage(session_id="s1", turn=17, text=text, risk="none", reason=PHRASE),
            helper_failed=False, timings_ms={"keyword": 1, "checker": 2, "helper": 3, "total": 4})

    patch_dependency(monkeypatch, turn, "run_turn", released)
    response = h.say(text="  trimmed by the service  ")
    assert response.status_code == 200
    assert called == [("s1", "trimmed by the service")]
    assert response.json() == dict(session_id="s1", turn=17, text=REPLY,
                                  policy_version="release-owned-version", fixed_handoff_response=False,
                                  check_status="answered", risk="none", category=None,
                                  keyword_matched=False, helper_failed=False,
                                  timings_ms={"keyword": 1, "checker": 2, "helper": 3, "total": 4})
    assert all(client.calls == [] for client in h.made)


@pytest.mark.parametrize("keyword,category", [(True, None), (True, "danger"),
                                               (False, "self"), (False, "danger"), (False, "unsure")])
def test_keyword_or_crisis_verdict_gives_the_fixed_response(keyword, category, make_service, path):
    h = make_service()
    assert h.start().status_code == 201
    text = path.keywords[0] if keyword else "A quiet message."
    if category:
        text += f" [checker:crisis-{category}]"
    response = h.say(text=text)
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == path.fixed_responses[category or "unsure"]
    assert body["fixed_handoff_response"] is True
    assert body["keyword_matched"] is keyword
    assert body["risk"] == ("crisis" if category else "none")
    assert body["category"] == category
    assert "keyword" not in body and "reason" not in body and "checked" not in body


@pytest.mark.parametrize("failure,status", [("fail", "failed"), ("garbage", "unreadable"), ("late", "late")])
def test_a_check_that_did_not_answer_is_null_and_uses_holding_then_offline(
        failure, status, make_service, path):
    gate = Gate("none | - | late answer")
    checker = FakeClient(gate) if failure == "late" else chat.fake_checker(path)
    h = make_service(pairs=[(chat.fake_helper(), checker)], deadline_s=0.02 if failure == "late" else 2.0)
    assert h.start().status_code == 201
    try:
        for expected in (path.holding_line, path.offline_line):
            response = h.say(text=f"A quiet message. [checker:{failure}]")
            assert response.status_code == 200
            body = response.json()
            assert body["check_status"] == status
            assert body["risk"] is None and body["category"] is None
            assert body["text"] == expected and body["fixed_handoff_response"] is True
    finally:
        gate.open.set()


@pytest.mark.parametrize("failure", ["raises", "late"])
def test_helper_failure_is_a_successful_request_with_a_safe_line(failure, make_service, path):
    gate = Gate()

    def broken(*args):
        raise RuntimeError(PHRASE)

    h = make_service(pairs=[(FakeClient(gate if failure == "late" else broken),
                            chat.fake_checker(path))], helper_wait_s=0.02 if failure == "late" else 2.0)
    assert h.start().status_code == 201
    try:
        response = h.say()
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == path.holding_line
        assert body["fixed_handoff_response"] is True and body["helper_failed"] is True
        assert body["check_status"] == "answered" and body["risk"] == "none"
        assert PHRASE not in response.text
    finally:
        gate.open.set()


# 2. Strict consent, validation, storage and privacy.

@pytest.mark.parametrize("field", ["remember_me", "consent_for_training"])
@pytest.mark.parametrize("value", ["yes", "false", 0, 1, None, [], {}])
def test_consent_answers_must_be_json_booleans(field, value, make_service):
    h = make_service()
    response = h.start(**{field: value})
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request", "fields": [field]}
    assert h.service.live() == [] and h.made == [] and disk(h.root) == {} and disk(h.store.root) == {}


@pytest.mark.parametrize("field", ["user_id", "remember_me", "consent_for_training"])
def test_required_start_answers_are_never_assumed(field, make_service):
    h = make_service()
    body = dict(user_id="person-a", remember_me=False, consent_for_training=False)
    del body[field]
    response = h.client.post("/sessions", headers=AUTH, json=body)
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request", "fields": [field]}
    assert h.service.live() == [] and h.made == []


@pytest.mark.parametrize("route", ["/sessions", "/sessions/s1/turns"])
@pytest.mark.parametrize("body", ["{", "[]", "null", '"private-scalar-value"'])
def test_a_whole_body_fault_is_reported_without_repeating_the_body(route, body, make_service, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    h = make_service()
    assert h.start().status_code == 201
    response = h.client.post(route, headers=AUTH, content=body)
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request", "fields": []}
    assert "private-scalar-value" not in response.text
    assert all(client.calls == [] for client in h.made)
    assert_private(capsys, caplog, PHRASE, "private-scalar-value")


def test_validation_reports_sorted_top_level_fields_and_never_sent_values(make_service, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    h = make_service()
    response = h.client.post("/sessions", headers=AUTH, json={
        "user_id": f"../{PHRASE}", "remember_me": PHRASE, "consent_for_training": False,
        "session_id": f"../{PHRASE}", "z_extra": PHRASE})
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request",
                               "fields": ["remember_me", "session_id", "user_id", "z_extra"]}
    assert PHRASE not in response.text
    assert h.service.live() == [] and h.made == []
    assert_private(capsys, caplog, PHRASE)


@pytest.mark.parametrize("value", ["", "../escape", "Uppercase", "a" * 34, 42, *sorted(RESERVED_IDS)])
def test_user_id_must_pass_the_existing_note_store_check(value, make_service):
    h = make_service()
    response = h.start(user=value)
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request", "fields": ["user_id"]}
    assert h.made == [] and h.service.live() == []


@pytest.mark.parametrize("value", ["", "../escape", "a.b", "-first", "a" * 65, 42])
def test_session_id_must_be_a_plain_name_of_at_most_64_characters(value, make_service):
    h = make_service()
    response = h.start(sid=value)
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request", "fields": ["session_id"]}
    assert h.made == [] and disk(h.root) == {}


def test_valid_boundary_ids_and_generated_ids_are_accepted(make_service):
    h = make_service()
    assert h.start(sid="A" + "_-0" * 21, user="a" * 33).status_code == 201
    generated = []
    for user in ("person-b", "person-c"):
        before = datetime.now(timezone.utc).replace(microsecond=0)
        response = h.client.post("/sessions", headers=AUTH, json=dict(
            user_id=user, remember_me=False, consent_for_training=False))
        after = datetime.now(timezone.utc)
        assert response.status_code == 201
        sid = response.json()["session_id"]
        assert re.fullmatch(r"sess-\d{14}-[0-9a-fA-F]{8}", sid)
        # A parseable timestamp is required, rather than just fourteen digits.
        stamp = datetime.strptime(sid.split("-")[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        assert before <= stamp <= after
        generated.append(sid)
    assert len(set(generated)) == 2


def test_null_session_id_generates_an_id_just_like_an_omitted_id(make_service):
    h = make_service()
    response = h.start(sid=None)
    assert response.status_code == 201
    sid = response.json()["session_id"]
    assert re.fullmatch(r"sess-\d{14}-[0-9a-fA-F]{8}", sid)
    assert h.service.live() == [sid]


@pytest.mark.parametrize("body,fields", [({}, ["text"]), ({"text": 7}, ["text"]),
    ({"text": None}, ["text"]), ({"text": False}, ["text"]), ({"text": []}, ["text"]),
    ({"text": ""}, ["text"]), ({"text": " \n\t "}, ["text"]),
    ({"text": PHRASE + "x" * 2001}, ["text"]),
    ({"text": "valid", "extra": PHRASE}, ["extra"])])
def test_invalid_turns_are_private_and_do_not_advance_the_turn(body, fields, make_service, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    h = make_service()
    assert h.start().status_code == 201
    response = h.client.post("/sessions/s1/turns", headers=AUTH, json=body)
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request", "fields": fields}
    assert PHRASE not in response.text
    assert all(client.calls == [] for client in h.made)
    assert h.say().json()["turn"] == 1
    assert_private(capsys, caplog, PHRASE)


@pytest.mark.parametrize("length", [1, 2000])
def test_text_limits_apply_after_stripping(length, make_service):
    h = make_service()
    assert h.start().status_code == 201
    response = h.say(text=" \n" + "q" * length + "\t ")
    assert response.status_code == 200
    assert h.made[0].calls[0]["messages"][-1]["content"] == "q" * length


@pytest.mark.parametrize("remember,training", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("apart", [False, True], ids=["shared-root", "separate-roots"])
def test_the_four_consent_rows_hold_on_disk_and_in_the_trace_store(
        remember, training, apart, make_service, local_tracking, tmp_path, caplog, capsys, monkeypatch):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="puddles.inference.service")
    setup_calls = []
    real_setup = tracing.setup

    def setup(*args, **kwargs):
        setup_calls.append((args, kwargs))
        return real_setup(*args, **kwargs)

    patch_dependency(monkeypatch, tracing, "setup", setup)
    h = make_service(notes_with_sessions=not apart)
    capsys.readouterr()
    caplog.clear()
    assert h.start(remember=remember, training=training).status_code == 201
    response = h.say(text=f"I am thinking about {PHRASE}. [note:situation:{FACT}]")
    assert response.status_code == 200
    ended = h.end()
    assert ended.status_code == 200
    body = ended.json()
    assert body["kept"] is training
    assert body["notes"] == dict(added=int(remember), replaced=0, confirmed=0,
                                  deleted=0, undeleted=[], failed=False)
    assert FACT not in ended.text or training  # consenting records may contain the conversation
    assert h.note_flags == [training]
    assert len(h.made) == 2
    assert len(setup_calls) == int(training)
    service_logs = repr([vars(record) for record in caplog.records + caplog._turn_service_records
                         if record.name == "puddles.inference.service"])
    for secret in (PHRASE, FACT, TOKEN, chat.FAKE_HELPER_LINE):
        assert secret not in service_logs
    if training:
        assert setup_calls == [((), {"tracking_uri": local_tracking})]
        record = ConversationRecord.model_validate(body["record"])
        assert record.id == "s1" and record.policy_version == "acceptance-v1"
        assert record.remember_me is remember and record.consent_for_training is True
        assert record.model_dump(mode="json") == json.loads(record_path(h.root, "s1").read_text())
        [line] = read_lines(h.root, "s1")
        assert record.turns == [Turn(speaker="user", text=line.user_text),
                                Turn(speaker="helper", text=response.json()["text"])]
    else:
        assert body["record"] is None
    expected_sessions = {"sessions/s1.jsonl", "sessions/s1.record.json"} if training else set()
    expected_notes = {f"notes/person-a/{name}" for name in ("counter.json", "index.json", "notes.json")} if (
        remember) else set()
    assert set(disk(h.root)) == expected_sessions | (set() if apart else expected_notes)
    assert set(disk(h.store.root)) == expected_notes | (set() if apart else expected_sessions)
    assert len(h.store.stored("person-a")) == int(remember)
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    assert mlflow.search_runs(search_all_experiments=True, output_format="list") == []
    if training:
        assert traces and contains(tmp_path, PHRASE)
    else:
        assert traces == [] and contains(tmp_path, PHRASE) == []
        assert contains(tmp_path, FACT) == ([h.store.root / "notes/person-a/notes.json"] if remember else [])
        captured = assert_private(capsys, caplog, PHRASE, FACT, chat.FAKE_HELPER_LINE, TOKEN)
        assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("propagate", [True, False], ids=["propagating-logger", "own-handler-only"])
def test_private_model_errors_and_note_errors_are_never_exposed(
        propagate, make_service, tmp_path, capsys, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="puddles.inference.service")
    monkeypatch.setattr(logging.getLogger("puddles.inference.service"), "propagate", propagate)

    def fail(*args, **kwargs):
        raise RuntimeError(PHRASE + FACT + TOKEN)

    class BrokenNotes(chat.FakeNoteTaker):
        structured = staticmethod(fail)

    patch_dependency(monkeypatch, tracing, "setup", fail)
    h = make_service(pairs=[(FakeClient(fail), FakeClient(fail))], note_llm=BrokenNotes())
    assert h.start(remember=True).status_code == 201
    response = h.say(text=PHRASE)
    assert response.status_code == 200 and response.json()["risk"] is None
    ended = h.end()
    assert ended.status_code == 200 and ended.json()["notes"]["failed"] is True
    for answer in (response, ended):
        assert PHRASE not in answer.text and FACT not in answer.text and TOKEN not in answer.text
    assert contains(tmp_path, PHRASE) == [] and contains(tmp_path, FACT) == []
    assert_private(capsys, caplog, PHRASE, FACT, TOKEN)


def test_tracing_setup_failure_with_consent_is_private_and_releases_every_reservation(
        make_service, monkeypatch, tmp_path, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    h = make_service(max_sessions=1)
    seed_notes(h, FACT)
    before = disk(h.root), disk(h.store.root)
    calls = []
    tracking_artifact = tmp_path / "tracking-initializer-artifact"

    def broken_setup(*args, **kwargs):
        calls.append((args, kwargs))
        tracking_artifact.write_text("initializer side effect")
        raise RuntimeError(PHRASE + FACT + TOKEN)

    patch_dependency(monkeypatch, tracing, "setup", broken_setup)
    response = h.start(training=True)
    assert_error(response, 500, "internal")
    assert calls == [((), {"tracking_uri": h.config.mlflow_uri})]
    assert h.service.live() == []
    assert (disk(h.root), disk(h.store.root)) == before
    assert tracking_artifact.read_text() == "initializer side effect"
    assert_private(capsys, caplog, PHRASE, FACT, TOKEN)
    # A failed consenting start must hold neither the ID, the person nor capacity.
    assert h.start().status_code == 201
    assert len(calls) == 1


def test_note_failure_reason_never_reaches_an_end_response_or_log(
        make_service, monkeypatch, tmp_path, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    h = make_service()
    assert h.start().status_code == 201
    assert h.say(text=PHRASE).status_code == 200

    def finish(session, store, llm, **kwargs):
        record = turn.end_session(session)
        return record, NotesResult(failed=True, reason=PHRASE + FACT + TOKEN)

    # A directly scripted result keeps the secret in reason even if live.py's
    # filesystem cleanup would otherwise replace the original model error.
    patch_dependency(monkeypatch, live, "finish_session", finish)
    response = h.end()
    assert response.status_code == 200
    assert response.json() == dict(session_id="s1", kept=False, record=None,
        notes=dict(added=0, replaced=0, confirmed=0, deleted=0, undeleted=[], failed=True))
    assert contains(tmp_path, PHRASE) == []
    assert_private(capsys, caplog, PHRASE, FACT, TOKEN)


def test_notes_load_for_every_consent_choice_and_reach_only_their_owner(make_service, local_tracking):
    h = make_service()
    seed_notes(h, FACT)
    for index, (remember, training) in enumerate(((False, False), (True, False), (False, True), (True, True))):
        sid = f"owner-{index}"
        response = h.start(sid, remember=remember, training=training)
        assert response.json() == dict(session_id=sid, policy_version="acceptance-v1",
                                       notes_loaded=1, notes_failed=False)
        assert h.say(sid).status_code == 200
        assert FACT in h.made[-2].calls[0]["system"]
        assert h.end(sid).status_code == 200
    assert h.start("other", user="person-b").json()["notes_loaded"] == 0
    assert h.say("other").status_code == 200
    assert FACT not in h.made[-2].calls[0]["system"]


def test_forgetting_works_with_both_consents_off_and_end_returns_only_counts(make_service):
    h = make_service()
    first, second = seed_notes(h, FACT, "has a sister who visits on Sundays")
    assert h.start().status_code == 201
    assert h.say(text=f"[forget:{first}] [note:situation:a new fact]").status_code == 200
    response = h.end()
    assert response.status_code == 200
    assert response.json() == dict(session_id="s1", kept=False, record=None,
                                   notes=dict(added=0, replaced=0, confirmed=0, deleted=1,
                                              undeleted=[], failed=False))
    assert list(h.store.stored("person-a")) == [second]
    assert first in h.store.memory.delete_calls
    assert FACT not in response.text and "a new fact" not in response.text


def test_end_reports_unconfirmed_delete_ids_without_note_text(make_service):
    h = make_service()
    [note_id] = seed_notes(h, FACT)
    h.store.memory.fail_deletes = 100
    assert h.start().status_code == 201
    assert h.say(text="[forget-everything]").status_code == 200
    response = h.end()
    assert response.status_code == 200
    assert response.json()["notes"] == dict(added=0, replaced=0, confirmed=0, deleted=1,
                                            undeleted=[note_id], failed=False)
    assert h.store.stored("person-a") == {} and FACT not in response.text


def test_end_counts_replacements_and_confirmations_without_returning_note_text(make_service):
    actions = []

    class ScriptedNotes(chat.FakeNoteTaker):
        def structured(self, system, user, schema):
            return schema.model_validate({"actions": actions})

    h = make_service(note_llm=ScriptedNotes())
    first, second = seed_notes(h, FACT, "has a sister who visits on Sundays")
    replacement = "now works morning shifts"
    actions.extend([dict(action="replace", note_id=first, kind="situation", fact=replacement),
                    dict(action="confirm", note_id=second)])
    assert h.start(remember=True).status_code == 201
    assert h.say().status_code == 200
    response = h.end()
    assert response.status_code == 200
    assert response.json()["notes"] == dict(added=1, replaced=1, confirmed=1, deleted=0,
                                            undeleted=[], failed=False)
    assert FACT not in response.text and replacement not in response.text
    stored = h.store.stored("person-a")
    assert first not in stored and second in stored
    assert replacement in [note.fact for note in stored.values()]


# 3. Authentication precedes validation, lookup and all side effects.

BAD_AUTH = [None, "Bearer wrong-token", "", "Basic " + TOKEN, "bearer " + TOKEN,
            "Bearer", "Bearer  " + TOKEN, "Bearer " + TOKEN + " ", "Bearer " + TOKEN + " extra",
            "Bearer\t" + TOKEN]


@pytest.mark.parametrize("header", BAD_AUTH)
@pytest.mark.parametrize("body_valid", [False, True], ids=["malformed-body", "valid-body"])
@pytest.mark.parametrize("route", ["/sessions", "/sessions/s1/turns", "/sessions/s1/end",
                                   "/sessions/unknown/turns", "/sessions/unknown/end"])
def test_auth_refusal_precedes_body_validation_and_session_lookup(
        header, body_valid, route, make_service, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="puddles.inference.service")
    h = make_service()
    assert h.start().status_code == 201
    before = disk(h.root), disk(h.store.root)
    body = dict(user_id="person-b", remember_me=False, consent_for_training=False, session_id="s2")
    if route.endswith("/turns"):
        body = {"text": PHRASE}
    response = h.client.post(route, headers={} if header is None else {"Authorization": header},
                             content=json.dumps(body) if body_valid else "{")
    assert_error(response, 401, "unauthorized")
    assert h.service.live() == ["s1"] and len(h.made) == 2
    assert all(client.calls == [] for client in h.made)
    assert (disk(h.root), disk(h.store.root)) == before
    assert h.say().json()["turn"] == 1
    assert_private(capsys, caplog, TOKEN, PHRASE)


def test_health_is_public_and_has_no_side_effects(make_service):
    h = make_service()
    for headers in ({}, {"Authorization": "malformed"}, AUTH):
        response = h.client.get("/health", headers=headers)
        assert response.status_code == 200 and response.json() == {"status": "ok"}
    assert h.service.live() == [] and h.made == [] and h.note_flags == []
    assert disk(h.root) == {} and disk(h.store.root) == {}


def test_enabled_framework_documentation_routes_are_not_an_auth_bypass(make_service):
    h = make_service()
    for route in ("/docs", "/redoc", "/openapi.json"):
        response = h.client.get(route)
        assert_error(response, 404, "not_found")


@pytest.mark.parametrize("route", ["/missing", "/sessions/s1/unknown"])
def test_unknown_paths_return_the_sanitized_not_found_body(route, make_service):
    h = make_service()
    assert_error(h.client.get(route, headers=AUTH), 404, "not_found")
    assert h.made == [] and h.service.live() == []


@pytest.mark.parametrize("method,route", [("POST", "/health"), ("GET", "/sessions"),
    ("GET", "/sessions/s1/turns"), ("DELETE", "/sessions/s1/end")])
def test_wrong_methods_return_the_sanitized_method_not_allowed_body(method, route, make_service):
    h = make_service()
    assert_error(h.client.request(method, route, headers=AUTH), 405, "method_not_allowed")
    assert h.made == [] and h.service.live() == []


@pytest.mark.parametrize("values", [(f"Bearer {TOKEN}", f"Bearer {TOKEN}"),
    (f"Bearer {TOKEN}", "Bearer wrong"), ("Bearer wrong", f"Bearer {TOKEN}")])
def test_duplicate_authorization_headers_are_refused_before_the_body(values, make_service, capsys, caplog):
    h = make_service()
    response = h.client.post("/sessions", headers=[("Authorization", value) for value in values], content="{")
    assert_error(response, 401, "unauthorized")
    assert h.made == [] and h.service.live() == [] and disk(h.root) == {} and disk(h.store.root) == {}
    assert_private(capsys, caplog, TOKEN)


@pytest.mark.parametrize("value", [b"Bearer \xff", b"Bearer " + TOKEN.encode() + b"\xc3\xa9"])
def test_non_ascii_authorization_is_a_refusal_and_never_an_error(value, make_service):
    h = make_service()
    response = h.client.post("/sessions", headers=[(b"authorization", value)], content="{")
    assert_error(response, 401, "unauthorized")
    assert h.made == [] and h.service.live() == []


def test_unauthorized_requests_are_refused_before_the_body_is_read(make_service, monkeypatch):
    from starlette.requests import Request
    h = make_service()
    reads = []
    original = Request.body

    async def observed_body(request):
        reads.append(request.url.path)
        return await original(request)

    monkeypatch.setattr(Request, "body", observed_body)
    response = h.client.post("/sessions", json=dict(user_id="person-a", remember_me=False,
                             consent_for_training=False, session_id="s1"))
    assert_error(response, 401, "unauthorized")
    assert h.made == [] and h.service.live() == []
    assert reads == [], "an unauthorized request reached the body parser"


def test_extra_field_names_are_returned_without_a_syntax_filter(make_service, capsys, caplog):
    h = make_service()
    response = h.start(**{"client-metadata": PHRASE})
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request", "fields": ["client-metadata"]}
    assert h.service.live() == [] and h.made == []
    assert_private(capsys, caplog, PHRASE)


def test_the_bearer_token_is_compared_with_compare_digest(make_service, monkeypatch):
    import hmac
    h = make_service()
    original = hmac.compare_digest
    comparisons = []

    def compare(left, right):
        comparisons.append((left, right))
        return original(left, right)

    patch_dependency(monkeypatch, hmac, "compare_digest", compare)
    assert h.start().status_code == 201
    assert_error(h.client.post("/sessions/s1/end", headers={"Authorization": "Bearer wrong"}),
                 401, "unauthorized")
    assert comparisons and all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in comparisons)
    assert any(a == TOKEN.encode("utf-8") and b == TOKEN.encode("utf-8") for a, b in comparisons)
    assert any({a, b} == {TOKEN.encode("utf-8"), b"wrong"} for a, b in comparisons)


# 4. Session startup, ordering of refusals and policy lifetime.

def test_live_session_collision_precedes_user_capacity_and_policy_checks(make_service, policy_path):
    h = make_service(max_sessions=1)
    assert h.start().status_code == 201
    policy_path.unlink()
    assert_error(h.start(), 409, "session_exists")
    assert_error(h.start("new"), 409, "user_busy")
    assert_error(h.start("new", user="person-b"), 429, "too_many_sessions")
    assert len(h.made) == 2 and h.service.live() == ["s1"]
    assert disk(h.root) == {} and disk(h.store.root) == {}


@pytest.mark.parametrize("kind", ["session", "record"])
def test_disk_collision_with_training_consent_precedes_user_and_capacity_checks(kind, make_service):
    h = make_service(max_sessions=1)
    assert h.start().status_code == 201
    target = (session_path if kind == "session" else record_path)(h.root, "old")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"earlier process sentinel\n")
    before = disk(h.root)
    assert_error(h.start("old", training=True), 409, "session_exists")
    assert disk(h.root) == before and len(h.made) == 2 and h.service.live() == ["s1"]


def test_existing_disk_files_do_not_block_a_session_without_training_consent(make_service):
    h = make_service()
    target = session_path(h.root, "s1")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"earlier process sentinel\n")
    before = disk(h.root)
    assert h.start().status_code == 201
    assert h.say().status_code == 200 and h.end().status_code == 200
    assert disk(h.root) == before


@pytest.mark.parametrize("kind", ["session", "record"])
def test_disk_lookup_errors_other_than_missing_are_internal_errors(kind, make_service, monkeypatch):
    h = make_service()
    target = (session_path if kind == "session" else record_path)(h.root, "s1")
    target.parent.mkdir(parents=True)
    faults = []

    def failing_lookup(original, *, scan=False):
        def lookup(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and (
                    Path(os.fsdecode(path)) == target or (scan and Path(os.fsdecode(path)) == target.parent)):
                faults.append(True)
                raise PermissionError(PHRASE)
            return original(path, *args, **kwargs)
        return lookup

    # Existence may be checked by stat, lstat or opening. Fault the filesystem
    # boundary, without requiring one particular implementation of that check.
    with monkeypatch.context() as patch:
        for owner, name in ((os, "stat"), (os, "lstat"), (os, "open"), (os, "access"), (os, "scandir"),
                            (builtins, "open"), (io, "open")):
            patch.setattr(owner, name, failing_lookup(getattr(owner, name), scan=name == "scandir"))
        response = h.start(training=True)
    assert faults, "the disk lookup fault was not exercised"
    assert_error(response, 500, "internal")
    assert h.service.live() == [] and h.made == []
    assert disk(h.root) == {} and disk(h.store.root) == {}


def test_capacity_is_freed_when_a_session_ends(make_service):
    h = make_service(max_sessions=1)
    assert h.start().status_code == 201
    assert_error(h.start("s2", user="person-b"), 429, "too_many_sessions")
    assert h.end().status_code == 200
    assert h.start("s2", user="person-b").status_code == 201
    assert h.service.live() == ["s2"]


@pytest.mark.parametrize("collision,status,error", [("person", 409, "user_busy"),
    ("session", 409, "session_exists"), ("capacity", 429, "too_many_sessions")])
@pytest.mark.parametrize("training", [False, True], ids=["private", "training-consent"])
def test_simultaneous_starts_reserve_the_person_session_id_and_capacity(
        collision, status, error, training, make_service, monkeypatch):
    gate = Gate()
    calls = []
    original = live.start_notes

    def slow_notes(*args, **kwargs):
        calls.append(True)
        gate()
        return original(*args, **kwargs)

    patch_dependency(monkeypatch, live, "start_notes", slow_notes)
    # Persistence and traces have their own consent matrix; this gate tests
    # reservation atomicity with either consent while startup is still pending.
    patch_dependency(monkeypatch, tracing, "setup", lambda **kwargs: None)
    h = make_service(max_sessions=1 if collision == "capacity" else 64)
    second_dispatched = Event()
    dispatches = []

    @h.app.middleware("http")
    async def observe_start(request, call_next):
        if request.method == "POST" and request.url.path == "/sessions":
            dispatches.append(True)
            if len(dispatches) == 2:
                second_dispatched.set()
        return await call_next(request)

    with h.client:
        first = Background(lambda: h.start(training=training))
        second = None
        try:
            assert gate.entered.wait(2)
            sid = "s1" if collision == "session" else "s2"
            user = "person-a" if collision == "person" else "person-b"
            second = Background(lambda: h.start(sid, user=user, training=training))
            assert second_dispatched.wait(2)
            # Allow either an immediate refusal or serialization of startup.
            # Releasing the first start must never let both requests succeed.
            second.done.wait(0.2)
            gate.open.set()
            assert first.result().status_code == 201
            assert_error(second.result(), status, error)
            assert calls == [True] and len(h.made) == 2
            assert h.service.live() == ["s1"]
        finally:
            gate.open.set()
            first.join()
            if second is not None:
                second.join()


def test_policy_is_checked_once_at_each_session_start_and_is_then_frozen(
        make_service, policy_path, monkeypatch):
    h = make_service()
    original = graph.check_policy_file
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    patch_dependency(monkeypatch, graph, "check_policy_file", read)
    assert h.start().json()["policy_version"] == "acceptance-v1"
    policy_path.write_text(policy_path.read_text().replace("acceptance-v1", "acceptance-v2"))
    assert h.start("s2", user="person-b").json()["policy_version"] == "acceptance-v2"
    policy_path.unlink()
    assert h.say().json()["policy_version"] == "acceptance-v1"
    assert h.say("s2").json()["policy_version"] == "acceptance-v2"
    assert h.end().status_code == 200 and h.end("s2").status_code == 200
    assert reads == [policy_path, policy_path]


@pytest.mark.parametrize("training", [False, True])
def test_startup_dependencies_run_in_contract_order_and_end_uses_the_same_note_model(
        training, make_service, path, monkeypatch):
    events = []
    helper, checker, notes_model = chat.fake_helper(), chat.fake_checker(path), chat.FakeNoteTaker()
    pending = iter((helper, checker))
    real_policy = graph.check_policy_file

    def policy(filename):
        events.append("policy")
        return real_policy(filename)

    def client():
        model = next(pending)
        events.append("helper" if model is helper else "checker")
        return model

    def note_for(consent):
        assert consent is training
        events.append("note-model")
        return notes_model

    def start_notes(*args, **kwargs):
        events.append("notes-load")
        return [FACT]

    real_finish = live.finish_session

    def finish(*args, **kwargs):
        bound = inspect.signature(real_finish).bind(*args, **kwargs).arguments
        session = bound["session"]
        assert bound["llm"] is notes_model
        assert session.helper_client is helper and session.checker_client is checker
        assert list(session.user_notes) == [FACT]
        events.append("finish")
        return None, NotesResult()

    patch_dependency(monkeypatch, graph, "check_policy_file", policy)
    patch_dependency(monkeypatch, live, "start_notes", start_notes)
    patch_dependency(monkeypatch, live, "finish_session", finish)
    patch_dependency(monkeypatch, tracing, "setup", lambda **kwargs: events.append("tracing"))
    h = make_service(client_factory=client, note_llm_for=note_for)
    assert events == []
    assert h.start(training=training).json()["notes_loaded"] == 1
    assert events == ["policy", "helper", "checker", "note-model"] + (
        ["tracing"] if training else []) + ["notes-load"]
    assert h.end().status_code == 200 and events[-1] == "finish"


@pytest.mark.parametrize("problem", ["missing", "refused", "unicode", "permission"])
def test_policy_unavailable_leaves_no_session_or_files(problem, make_service, policy_path, monkeypatch):
    h = make_service()
    if problem == "missing":
        policy_path.unlink()
    elif problem == "refused":
        policy_path.write_text("not a policy " + PHRASE)
    elif problem == "unicode":
        policy_path.write_bytes(b"\xff")
    else:
        def denied(path):
            raise PermissionError(PHRASE)
        patch_dependency(monkeypatch, graph, "check_policy_file", denied)
    assert_error(h.start(), 503, "policy_unavailable")
    assert h.service.live() == [] and h.made == [] and h.note_flags == []
    assert disk(h.root) == {} and disk(h.store.root) == {}


@pytest.mark.parametrize("failed_step", ["helper", "checker", "notes"])
@pytest.mark.parametrize("training", [False, True])
def test_model_construction_failure_is_private_and_leaves_nothing(
        failed_step, training, make_service, caplog, capsys, monkeypatch):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="puddles.inference.service")
    h = make_service()
    calls, setup_calls, note_loads = [], [], []
    patch_dependency(monkeypatch, tracing, "setup", lambda **kwargs: setup_calls.append(kwargs))
    patch_dependency(monkeypatch, live, "start_notes", lambda *args: note_loads.append(args) or [])

    def client():
        step = "helper" if not calls else "checker"
        calls.append(step)
        if step == failed_step:
            raise RuntimeError(PHRASE)
        return chat.fake_helper()

    def notes(training):
        calls.append("notes")
        raise RuntimeError(PHRASE)

    # Config is a public dataclass. Build a second app so no mutable-config behavior is assumed.
    config = replace(h.config, client_factory=client, note_llm_for=notes)
    app = importlib.import_module("puddles.inference.service").create_app(config)
    with TestClient(app, raise_server_exceptions=False) as client_api:
        response = client_api.post("/sessions", headers=AUTH, json=dict(
            user_id="person-a", remember_me=False, consent_for_training=training, session_id="s1"))
    assert_error(response, 503, "model_unavailable")
    assert calls == ["helper", "checker", "notes"][:["helper", "checker", "notes"].index(failed_step) + 1]
    assert app.state.service.live() == [] and disk(h.root) == {} and disk(h.store.root) == {}
    assert setup_calls == [] and note_loads == []
    assert_private(capsys, caplog, PHRASE, TOKEN)


@pytest.mark.parametrize("failure", ["policy", "model"])
def test_a_failed_start_does_not_reserve_the_user_id_session_id_or_capacity(
        failure, make_service, policy_path, path):
    fail_model = [failure == "model"]
    pending = []

    def factory():
        if fail_model[0]:
            fail_model[0] = False
            raise RuntimeError(PHRASE)
        if not pending:
            pending.extend((chat.fake_helper(), chat.fake_checker(path)))
        return pending.pop(0)

    h = make_service(max_sessions=1, client_factory=factory)
    original_policy = policy_path.read_bytes()
    if failure == "policy":
        policy_path.unlink()
    assert_error(h.start(), 503, "policy_unavailable" if failure == "policy" else "model_unavailable")
    assert h.service.live() == [] and disk(h.root) == {} and disk(h.store.root) == {}
    if failure == "policy":
        policy_path.write_bytes(original_policy)
    assert h.start().status_code == 201
    assert h.say().status_code == 200 and h.end().status_code == 200


def test_start_notes_failure_starts_with_no_notes_and_reports_failure(make_service, monkeypatch):
    h = make_service()

    def broken(*args):
        raise RuntimeError(FACT)

    patch_dependency(monkeypatch, live, "start_notes", broken)
    response = h.start()
    assert response.status_code == 201
    assert response.json() == dict(session_id="s1", policy_version="acceptance-v1",
                                   notes_loaded=0, notes_failed=True)
    assert h.say().status_code == 200
    assert FACT not in h.made[0].calls[0]["system"] and FACT not in response.text


def test_an_unnamed_handler_failure_has_only_the_internal_error(make_service, monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="puddles.inference.service")
    h = make_service()

    def unexpected(*args):
        raise RuntimeError(PHRASE)

    patch_dependency(monkeypatch, graph, "check_policy_file", unexpected)
    assert_error(h.start(), 500, "internal")
    assert h.service.live() == [] and h.made == [] and disk(h.root) == {} and disk(h.store.root) == {}
    assert_private(capsys, caplog, PHRASE)


# 5. Turn and end, including their ordering.

@pytest.mark.parametrize("state", ["never-started", "ended", "swept"])
def test_nonlive_turn_and_end_are_404_and_reach_no_model(state, make_service):
    clock = SimpleNamespace(now=0.0)
    h = make_service(clock=lambda: clock.now, idle_timeout_s=10)
    if state != "never-started":
        assert h.start().status_code == 201
        if state == "ended":
            assert h.end().status_code == 200
        else:
            clock.now = 10
            assert h.service.sweep() == ["s1"]
    assert_error(h.say(), 404, "session_not_found")
    assert_error(h.end(), 404, "session_not_found")
    assert all(client.calls == [] for client in h.made)


@pytest.mark.parametrize("cleanup", ["end", "sweep"])
def test_end_and_sweep_release_the_user_id_session_id_and_capacity(cleanup, make_service):
    clock = SimpleNamespace(now=0.0)
    h = make_service(clock=lambda: clock.now, idle_timeout_s=10, max_sessions=1)
    assert h.start().status_code == 201
    if cleanup == "end":
        assert h.end().status_code == 200
    else:
        clock.now = 10
        assert h.service.sweep() == ["s1"]
    assert h.start().status_code == 201
    assert h.say().json()["turn"] == 1


def test_run_turn_exception_returns_offline_line_and_the_session_stays_live(
        make_service, monkeypatch, path, capsys, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="puddles.inference.service")
    h = make_service()
    assert h.start().status_code == 201
    original = turn.run_turn

    def broken(session, text):
        raise RuntimeError(text + FACT + TOKEN)

    patch_dependency(monkeypatch, turn, "run_turn", broken)
    response = h.say(text=PHRASE)
    assert response.status_code == 500
    assert response.json() == {"error": "turn_failed", "say": path.offline_line}
    assert h.service.live() == ["s1"] and contains(tmp_path, PHRASE) == []
    assert_private(capsys, caplog, PHRASE, FACT, TOKEN)
    patch_dependency(monkeypatch, turn, "run_turn", original)
    assert h.say().status_code == 200 and h.end().status_code == 200


def test_end_without_turns_keeps_nothing_and_ignores_any_body(make_service, local_tracking):
    h = make_service()
    assert h.start(training=True).status_code == 201
    response = h.client.post("/sessions/s1/end", headers=AUTH, content="{not even JSON " + PHRASE)
    assert response.status_code == 200
    assert response.json() == dict(session_id="s1", kept=False, record=None,
                                   notes=dict(added=0, replaced=0, confirmed=0, deleted=0,
                                              undeleted=[], failed=False))
    assert disk(h.root) == {} and disk(h.store.root) == {} and h.service.live() == []


def test_finish_session_exception_falls_back_to_end_session_and_removes_the_session(
        make_service, monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="puddles.inference.service")
    h = make_service()
    assert h.start().status_code == 201 and h.say().status_code == 200
    ended = []
    original = turn.end_session

    def broken(*args):
        raise RuntimeError(PHRASE)

    def end(session, *args, **kwargs):
        ended.append(session)
        return original(session, *args, **kwargs)

    patch_dependency(monkeypatch, live, "finish_session", broken)
    patch_dependency(monkeypatch, turn, "end_session", end)
    response = h.end()
    assert response.status_code == 200 and response.json()["notes"]["failed"] is True
    assert len(ended) == 1 and ended[0].turns == [] and ended[0].lines == []
    assert h.service.live() == []
    assert_error(h.end(), 404, "session_not_found")
    assert_private(capsys, caplog, PHRASE)


def test_the_session_is_removed_even_when_both_end_operations_raise(make_service, monkeypatch):
    h = make_service()
    assert h.start().status_code == 201

    def broken(*args, **kwargs):
        raise RuntimeError(PHRASE)

    patch_dependency(monkeypatch, live, "finish_session", broken)
    patch_dependency(monkeypatch, turn, "end_session", broken)
    response = h.end()
    # Double cleanup failure has no unambiguous response status in the contract.
    # Removal and redaction remain required regardless of that status.
    assert PHRASE not in response.text
    assert h.service.live() == []
    assert_error(h.say(), 404, "session_not_found")
    assert all(client.calls == [] for client in h.made)


def test_end_waits_for_a_running_turn_and_rejects_turns_while_notes_are_running(make_service, path):
    helper_gate, notes_gate = Gate(), Gate()
    checker_called = Event()

    def checker(*args):
        checker_called.set()
        return "none | - | acceptance verdict"

    class SlowNotes(chat.FakeNoteTaker):
        def structured(self, *args, **kwargs):
            notes_gate()
            return super().structured(*args, **kwargs)

    h = make_service(pairs=[(FakeClient(helper_gate), FakeClient(checker))], note_llm=SlowNotes())
    [note_id] = seed_notes(h, FACT)
    end_dispatched = Event()

    @h.app.middleware("http")
    async def observe_end(request, call_next):
        if request.url.path.endswith("/end"):
            end_dispatched.set()
        return await call_next(request)

    # One TestClient context means one event loop, as in the deployed ASGI app.
    with h.client:
        assert h.start().status_code == 201
        running = Background(lambda: h.say(text=f"[forget:{note_id}]"))
        ending = None
        rejected = []
        try:
            assert helper_gate.entered.wait(2) and checker_called.wait(2)
            ending = Background(h.end)
            assert end_dispatched.wait(2)
            assert not ending.done.wait(0.1), "end overlapped the running turn"
            assert not notes_gate.entered.is_set()
            calls = [len(client.calls) for client in h.made]
            # End must mark the session before waiting for the running turn.
            rejected.append(Background(h.say))
            assert_error(rejected[-1].result(), 404, "session_not_found")
            assert [len(client.calls) for client in h.made] == calls
            assert h.service.live() == ["s1"]
            assert_error(h.start("next"), 409, "user_busy")
            assert_error(h.start("s1", user="person-b"), 409, "session_exists")
            helper_gate.open.set()
            assert running.result().status_code == 200
            assert notes_gate.entered.wait(2)
            # The ending session continues to own both reservations during notes.
            assert h.service.live() == ["s1"]
            assert_error(h.start("next"), 409, "user_busy")
            assert_error(h.start("s1", user="person-b"), 409, "session_exists")
            rejected.append(Background(h.say))
            assert_error(rejected[-1].result(), 404, "session_not_found")
            assert [len(client.calls) for client in h.made] == calls
            notes_gate.open.set()
            response = ending.result()
            assert response.status_code == 200 and response.json()["notes"]["deleted"] == 1
            assert h.store.stored("person-a") == {} and note_id in h.store.memory.delete_calls
            assert h.service.live() == []
        finally:
            helper_gate.open.set()
            notes_gate.open.set()
            running.join()
            if ending is not None:
                ending.join()
            for request in rejected:
                request.join()


# 6. Idle cleanup, live IDs and independent sessions.

def test_model_waiting_route_endpoints_are_plain_sync_functions(make_service):
    h = make_service()
    expected = {"/sessions", "/sessions/{session_id}/turns", "/sessions/{session_id}/end"}
    endpoints = {route.path: route.endpoint for route in h.app.routes
                 if getattr(route, "path", None) in expected}
    assert set(endpoints) == expected
    assert all(not inspect.iscoroutinefunction(endpoint) for endpoint in endpoints.values())


def test_loading_notes_in_one_start_does_not_block_an_existing_turn(make_service, monkeypatch):
    h = make_service()
    gate = Gate()
    original = live.start_notes

    def slow_notes(*args, **kwargs):
        gate()
        return original(*args, **kwargs)

    with h.client:
        assert h.start().status_code == 201
        patch_dependency(monkeypatch, live, "start_notes", slow_notes)
        starting = Background(lambda: h.start("s2", user="person-b"))
        answering = None
        try:
            assert gate.entered.wait(2)
            answering = Background(h.say)
            assert answering.result().status_code == 200
            assert not starting.done.is_set()
            assert h.client.get("/health").status_code == 200
        finally:
            gate.open.set()
            starting.join()
            if answering is not None:
                answering.join()


def test_live_and_sweep_are_sorted_and_sweep_applies_forget_requests(make_service):
    clock = SimpleNamespace(now=0.0)
    h = make_service(clock=lambda: clock.now, idle_timeout_s=10)
    [note_id] = seed_notes(h, FACT)
    assert h.start("z-old").status_code == 201
    assert h.say("z-old", text=f"[forget:{note_id}]").status_code == 200
    assert h.start("b-old", user="person-b").status_code == 201
    clock.now = 6
    assert h.start("a-fresh", user="person-c").status_code == 201
    assert h.service.live() == ["a-fresh", "b-old", "z-old"]
    clock.now = 9.99
    assert h.service.sweep() == []
    clock.now = 10
    assert h.service.sweep() == ["b-old", "z-old"]
    assert h.service.live() == ["a-fresh"]
    assert h.store.stored("person-a") == {} and note_id in h.store.memory.delete_calls
    assert h.service.sweep() == []


def test_running_turn_is_not_idle_and_its_finish_refreshes_activity(make_service, path):
    clock = SimpleNamespace(now=0.0)
    gate = Gate()
    h = make_service(clock=lambda: clock.now, idle_timeout_s=10,
                     pairs=[(FakeClient(gate), chat.fake_checker(path))])
    assert h.start().status_code == 201
    clock.now = 9
    request = Background(h.say)
    try:
        assert gate.entered.wait(0.7)
        clock.now = 25
        assert h.service.sweep() == [] and h.service.live() == ["s1"]
        clock.now = 30
        gate.open.set()
        assert request.result().status_code == 200
        clock.now = 39.99
        assert h.service.sweep() == []
        clock.now = 40
        assert h.service.sweep() == ["s1"]
    finally:
        gate.open.set()
        request.join()


def test_sweep_removes_failed_ends_and_continues_without_raising(make_service, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    h = make_service(clock=lambda: clock.now, idle_timeout_s=10)
    assert h.start("z").status_code == 201
    assert h.start("a", user="person-b").status_code == 201

    def broken(*args, **kwargs):
        raise RuntimeError(PHRASE)

    patch_dependency(monkeypatch, live, "finish_session", broken)
    patch_dependency(monkeypatch, turn, "end_session", broken)
    clock.now = 10
    assert h.service.sweep() == ["a", "z"]
    assert h.service.live() == [] and h.service.sweep() == []


def test_requests_never_sweep_other_sessions(make_service, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    h = make_service(clock=lambda: clock.now, idle_timeout_s=1)
    assert h.start().status_code == 201

    def forbidden():
        raise AssertionError("a request called sweep")

    monkeypatch.setattr(h.service, "sweep", forbidden)
    clock.now = 50
    assert h.client.get("/health").status_code == 200
    assert h.start("s2", user="person-b").status_code == 201
    assert h.say("s2").status_code == 200 and h.end("s2").status_code == 200
    assert h.service.live() == ["s1"]


def test_refused_requests_do_not_refresh_idle_activity(make_service):
    clock = SimpleNamespace(now=0.0)
    h = make_service(clock=lambda: clock.now, idle_timeout_s=10)
    assert h.start().status_code == 201
    clock.now = 9
    assert h.say(text=" \n ").status_code == 422
    assert_error(h.client.post("/sessions/s1/turns", json={"text": PHRASE}), 401, "unauthorized")
    clock.now = 10
    assert h.service.sweep() == ["s1"]
    assert all(client.calls == [] for client in h.made)


def test_a_slow_helper_in_one_session_does_not_delay_another_session(make_service, path):
    gate = Gate()
    h = make_service(pairs=[(FakeClient(gate), chat.fake_checker(path)),
                            (chat.fake_helper(), chat.fake_checker(path))])
    assert h.start("slow").status_code == 201
    assert h.start("fast", user="person-b").status_code == 201
    slow = Background(lambda: h.say("slow"))
    fast = None
    try:
        assert gate.entered.wait(0.7)
        fast = Background(lambda: h.say("fast"))
        assert fast.result(0.7).status_code == 200
        assert not slow.done.is_set()
    finally:
        gate.open.set()
        slow.join()
        if fast is not None:
            fast.join()


# 7. App construction and the offline command line entry point.

@pytest.mark.parametrize("token", [None, "", 0, False, b"bytes-token"])
def test_create_app_refuses_any_token_that_is_not_a_nonempty_string(token, make_service, api):
    h = make_service()
    with pytest.raises(ValueError):
        api.create_app(replace(h.config, token=token))


@pytest.mark.parametrize("capacity", [0, -1])
def test_create_app_refuses_capacity_below_one(capacity, make_service, api):
    h = make_service()
    with pytest.raises(ValueError):
        api.create_app(replace(h.config, max_sessions=capacity))


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("-inf"), float("nan"), None, "900"])
def test_create_app_requires_a_positive_finite_idle_timeout(timeout, make_service, api):
    h = make_service()
    with pytest.raises(ValueError):
        api.create_app(replace(h.config, idle_timeout_s=timeout))


def test_crisis_path_is_loaded_once_per_app_and_load_failure_propagates(make_service, api, path, monkeypatch):
    h = make_service()
    loads = []

    def load():
        loads.append(True)
        return path

    patch_dependency(monkeypatch, crisis, "load_path", load)
    app = api.create_app(replace(h.config, path=None))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/sessions", headers=AUTH, json=dict(user_id="person-a", remember_me=False,
                           consent_for_training=False)).status_code == 201
    assert loads == [True]
    marker = ValueError("crisis path failure marker")

    def fail():
        raise marker

    patch_dependency(monkeypatch, crisis, "load_path", fail)
    with pytest.raises(ValueError) as caught:
        api.create_app(replace(h.config, path=None))
    assert caught.value is marker
    # A supplied path bypasses the loader altogether.
    assert api.create_app(replace(h.config, path=path)).state.service.live() == []


def cli_args(policy_path, tmp_path, *, fake=True):
    return ["--policy", str(policy_path), "--root", str(tmp_path / "serve"),
            "--notes-root", str(tmp_path / "people")] + (["--fake"] if fake else [])


@pytest.mark.parametrize("network_args,host,port", [([], "127.0.0.1", 8765),
    (["--host", "0.0.0.0", "--port", "9876"], "0.0.0.0", 9876)])
def test_fake_main_passes_a_working_app_and_host_port_to_run(
        network_args, host, port, policy_path, tmp_path, monkeypatch, capsys, caplog, path):
    serve = importlib.import_module("scripts.serve")
    monkeypatch.setenv("PUDDLES_SERVICE_TOKEN", TOKEN)
    calls = []

    def run(app, **kwargs):
        calls.append(kwargs)
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/health").json() == {"status": "ok"}
            assert_error(client.post("/sessions", json={}), 401, "unauthorized")
            assert client.post("/sessions", headers=AUTH, json=dict(user_id="person-a", remember_me=True,
                consent_for_training=False, session_id="cli-session")).status_code == 201
            response = client.post("/sessions/cli-session/turns", headers=AUTH,
                                   json={"text": f"[checker:crisis-danger] [note:situation:{FACT}]"})
            assert response.status_code == 200
            assert response.json()["text"] == path.fixed_responses["danger"]
            assert client.post("/sessions/cli-session/end", headers=AUTH).json()["notes"]["added"] == 1

    assert serve.main(cli_args(policy_path, tmp_path) + network_args, run=run) == 0
    assert calls == [dict(host=host, port=port)]
    assert (tmp_path / "people/notes/person-a/notes.json").exists()
    assert disk(tmp_path / "serve") == {}
    assert_private(capsys, caplog, TOKEN)


def test_fake_build_config_uses_the_supplied_environment_and_existing_fakes(policy_path, tmp_path, path):
    serve = importlib.import_module("scripts.serve")
    args = Namespace(policy=str(policy_path), root=str(tmp_path / "serve"), notes_root=None,
                     host="127.0.0.1", port=8765, mlflow=None, sweep_every=30, fake=True)
    config = serve.build_config(args, {"PUDDLES_SERVICE_TOKEN": TOKEN})
    assert config.token == TOKEN and Path(config.root) == tmp_path / "serve"
    assert config.store.root == Path(config.root) and isinstance(config.store.memory, FakeUserMemory)
    for _ in range(2):
        helper, checker = config.client_factory(), config.client_factory()
        assert helper.complete("system", [{"role": "user", "content": "quiet"}], max_tokens=20) == (
            chat.fake_helper().complete("system", [{"role": "user", "content": "quiet"}], max_tokens=20))
        inp = crisis.build_input("cli", 1, [], "[checker:concern]")
        assert crisis.run(checker, inp, path).risk == "concern"
    assert isinstance(config.note_llm_for(False), chat.FakeNoteTaker)
    assert isinstance(config.note_llm_for(True), tracing.TracedLLM)


@pytest.mark.parametrize("token_only_mapping", [False, True], ids=["full-mapping", "token-only-mapping"])
def test_real_build_config_preflights_and_selects_the_consent_appropriate_models(
        token_only_mapping, policy_path, tmp_path, monkeypatch):
    serve = importlib.import_module("scripts.serve")
    from loop import llm as llm_api
    from puddles import llm as note_api
    calls = []

    def client(prefix):
        assert prefix == "HELPER"
        calls.append("helper")
        return chat.fake_helper()

    class Memory(FakeUserMemory):
        def __init__(self):
            calls.append("memory")
            super().__init__()

    class TrainingNotes(chat.FakeNoteTaker):
        def __init__(self):
            calls.append("training-notes")

    class PrivateNotes(chat.FakeNoteTaker):
        def __init__(self):
            calls.append("private-notes")

    patch_dependency(monkeypatch, llm_api, "client_from_env", client)
    patch_dependency(monkeypatch, note_store, "PastUserMemory", Memory)
    patch_dependency(monkeypatch, note_api, "OpenAILLM", TrainingNotes)
    monkeypatch.setattr(chat, "PastUserMemory", Memory)
    monkeypatch.setattr(chat, "OpenAILLM", TrainingNotes)
    monkeypatch.setattr(live.PlainLLM, "from_env", classmethod(lambda cls: PrivateNotes()))
    env = dict(PUDDLES_SERVICE_TOKEN=TOKEN, HELPER_URL="http://stand-in.invalid/v1",
               HELPER_MODEL="stand-in", PAST_API_KEY="placeholder", OPENAI_API_KEY="placeholder",
               PUDDLES_AGENT_MODEL="stand-in")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    args = Namespace(policy=str(policy_path), root=str(tmp_path / "serve"), notes_root=str(tmp_path / "people"),
                     host="127.0.0.1", port=8765, mlflow=None, sweep_every=30, fake=False)
    config = serve.build_config(args, {"PUDDLES_SERVICE_TOKEN": TOKEN} if token_only_mapping else env)
    assert {"helper", "memory", "training-notes", "private-notes"} <= set(calls)
    assert isinstance(config.store.memory, Memory) and config.store.root == tmp_path / "people"
    before = calls.count("helper")
    assert isinstance(config.client_factory(), FakeClient)
    assert isinstance(config.client_factory(), FakeClient)
    assert calls.count("helper") == before + 2
    assert isinstance(config.note_llm_for(True), TrainingNotes)
    assert isinstance(config.note_llm_for(False), PrivateNotes)


def test_main_starts_a_daemon_that_sweeps_on_the_requested_timer(api, policy_path, tmp_path, monkeypatch):
    serve = importlib.import_module("scripts.serve")
    monkeypatch.setenv("PUDDLES_SERVICE_TOKEN", TOKEN)
    swept = Event()
    daemon_flags = []

    def sweep(self):
        daemon_flags.append(current_thread().daemon)
        swept.set()
        return []

    monkeypatch.setattr(api.TurnService, "sweep", sweep)

    def run(app, **kwargs):
        assert swept.wait(0.7), "the sweep timer did not run"

    assert serve.main(cli_args(policy_path, tmp_path) + ["--sweep-every", "0.01"], run=run) == 0
    assert daemon_flags and all(daemon_flags)


@pytest.mark.parametrize("problem", ["missing-token", "empty-token", "missing-policy", "bad-policy",
                                     "crisis-load", "HELPER_URL", "HELPER_MODEL", "PAST_API_KEY",
                                     "OPENAI_API_KEY", "PUDDLES_AGENT_MODEL"])
def test_main_startup_errors_exit_two_before_run_and_never_print_the_token(
        problem, policy_path, tmp_path, monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="puddles.inference.service")
    serve = importlib.import_module("scripts.serve")
    # These are inert placeholders, never real credentials; socket access is forbidden.
    values = dict(PUDDLES_SERVICE_TOKEN=TOKEN, HELPER_URL="http://stand-in.invalid/v1",
                  HELPER_MODEL="stand-in", PAST_API_KEY="placeholder", OPENAI_API_KEY="placeholder",
                  PUDDLES_AGENT_MODEL="stand-in")
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    if problem == "missing-token":
        monkeypatch.delenv("PUDDLES_SERVICE_TOKEN")
    elif problem == "empty-token":
        monkeypatch.setenv("PUDDLES_SERVICE_TOKEN", "")
    elif problem == "missing-policy":
        policy_path.unlink()
    elif problem == "bad-policy":
        policy_path.write_text("not a policy")
    elif problem == "crisis-load":
        def broken():
            raise ValueError("cannot load crisis path " + TOKEN)
        patch_dependency(monkeypatch, crisis, "load_path", broken)
    else:
        monkeypatch.delenv(problem)
    called = []
    capsys.readouterr()
    code = serve.main(cli_args(policy_path, tmp_path, fake=problem not in {
        "HELPER_URL", "HELPER_MODEL", "PAST_API_KEY", "OPENAI_API_KEY", "PUDDLES_AGENT_MODEL"}),
        run=lambda *args, **kwargs: called.append(True))
    output = capsys.readouterr()
    assert code == 2 and called == []
    assert output.out == "" and len(output.err.splitlines()) == 1
    assert output.err.startswith("error: ")
    assert TOKEN not in output.err + output.out + caplog.text


@pytest.mark.parametrize("kind", ["config", "policy", "value", "os", "unicode"])
def test_startup_error_messages_are_shown_only_for_repo_config_and_policy_errors(
        kind, policy_path, tmp_path, monkeypatch, capsys, caplog):
    serve = importlib.import_module("scripts.serve")
    monkeypatch.setenv("PUDDLES_SERVICE_TOKEN", TOKEN)
    errors = {"config": ConfigError(PHRASE), "policy": PolicyError(PHRASE),
              "value": ValueError(PHRASE + TOKEN), "os": OSError(PHRASE + TOKEN),
              "unicode": UnicodeDecodeError("utf-8", b"\xff", 0, 1, PHRASE + TOKEN)}
    error = errors[kind]

    def broken(*args, **kwargs):
        raise error

    patch_dependency(monkeypatch, graph, "check_policy_file", broken)
    called = []
    capsys.readouterr()
    code = serve.main(cli_args(policy_path, tmp_path), run=lambda *args, **kwargs: called.append(True))
    output = capsys.readouterr()
    assert code == 2 and called == [] and output.out == ""
    assert len(output.err.splitlines()) == 1 and output.err.startswith("error: ")
    if kind in ("config", "policy"):
        assert PHRASE in output.err
    else:
        assert type(error).__name__ in output.err and PHRASE not in output.err
    assert TOKEN not in output.out + output.err + caplog.text
