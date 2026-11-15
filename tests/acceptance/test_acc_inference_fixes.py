"""Independent offline acceptance of F1–F3; no services, real sleeps, or non-temporary writes."""
from __future__ import annotations

import contextvars
import io
import json
import socket
import subprocess
import sys
import types
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from loop.errors import ConfigError
from loop.llm import OpenAICompatClient
from puddles.inference import crisis
from puddles.inference.live import PlainLLM, finish_session, known_untraced
from puddles.inference.note_store import FakeUserMemory, NoteStore, PastUserMemory
from puddles.inference.notes import NoteTakerOutput, take_notes, to_record
from puddles.inference.turn import Session
from puddles.memory import PastClient, PastError, TRAINING_IDENTITIES
from schemas import LiveNote, LiveNoteIndex, NoteAction, Turn

TODAY = date(2026, 11, 15)
NOW = datetime(2026, 11, 15, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("PUDDLES_AGENT_TEMPERATURE", "PUDDLES_AGENT_MAX_TOKENS_PARAM",
                 "PUDDLES_AGENT_TOOL_REASONING_EFFORT", "PUDDLES_NOTE_WRITE_WAIT_S",
                 "PUDDLES_NOTE_RECALL_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "acceptance-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://model.invalid/v1")
    monkeypatch.setenv("PUDDLES_AGENT_MODEL", "gpt-6-luna")

    def forbidden(*args, **kwargs):
        raise AssertionError("acceptance test attempted a real socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def completion(content):
    return {"id": "acceptance", "object": "chat.completion", "created": 0, "model": "gpt-6-luna",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}


@pytest.fixture
def wire(monkeypatch):
    class Wire:
        content = '{"forget_everything": false, "actions": []}'

        def __init__(self):
            self.requests = []
            self.on_send = lambda: None

        def __call__(self, request, timeout=None):
            self.on_send()
            self.requests.append((request, timeout))
            return io.BytesIO(json.dumps(completion(self.content)).encode())

    transport = Wire()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    return transport


@pytest.mark.parametrize("temperature", [None, "0", "0.35"])
def test_plain_request_matches_sdk_settings(monkeypatch, wire, temperature):
    """Both real serializers send the same model/messages/limit/temperature options."""
    import openai
    from puddles import usage
    from puddles.llm import OpenAILLM

    if temperature is not None:
        monkeypatch.setenv("PUDDLES_AGENT_TEMPERATURE", temperature)
    monkeypatch.setenv("PUDDLES_AGENT_TOOL_REASONING_EFFORT", "none")
    plain = PlainLLM.from_env()
    assert type(plain.client) is OpenAICompatClient and known_untraced(plain)
    assert plain.structured("system", "private words", NoteTakerOutput) == NoteTakerOutput()
    request, _ = wire.requests[0]
    body = json.loads(request.data)
    assert request.full_url == "https://model.invalid/v1/chat/completions"
    assert body["max_completion_tokens"] == plain.max_tokens
    assert "max_tokens" not in body and "reasoning_effort" not in body
    assert body.get("temperature") == (None if temperature is None else float(temperature))
    assert ("temperature" in body) == (temperature is not None)

    sent = []

    def answer(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=completion(wire.content))

    sdk = openai.OpenAI(api_key="placeholder", base_url="https://model.invalid/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(answer)))
    monkeypatch.setattr(openai, "OpenAI", lambda: sdk)
    model = OpenAILLM(output_limits={"default": plain.max_tokens})
    with usage.metering(None):
        assert model.structured("system", "private words", NoteTakerOutput) == NoteTakerOutput()
    sdk_body = sent[0]
    assert sdk_body.pop("stream") is False
    # The SDK makes schemas strict; loop's serializer uses Pydantic's schema unchanged.
    assert sdk_body.pop("response_format")["type"] == body.pop("response_format")["type"] == "json_schema"
    assert body == sdk_body
    sdk.close()


@pytest.mark.parametrize("param", ["max_tokens", "max_completion_tokens"])
def test_plain_limit_override_is_sent(monkeypatch, wire, param):
    monkeypatch.setenv("PUDDLES_AGENT_MAX_TOKENS_PARAM", param)
    llm = PlainLLM.from_env()
    llm.structured("s", "u", NoteTakerOutput)
    body = json.loads(wire.requests[0][0].data)
    assert body[param] == llm.max_tokens
    assert ({"max_tokens", "max_completion_tokens"} - {param}).isdisjoint(body)


def test_plain_empty_limit_name_sends_default(monkeypatch, wire):
    monkeypatch.setenv("PUDDLES_AGENT_MAX_TOKENS_PARAM", "")
    llm = PlainLLM.from_env()
    llm.structured("s", "u", NoteTakerOutput)
    assert len(wire.requests) == 1
    body = json.loads(wire.requests[0][0].data)
    assert body["max_completion_tokens"] == llm.max_tokens
    assert "max_tokens" not in body


@pytest.mark.parametrize("bad", ["max_output_tokens", "MAX_TOKENS", " max_tokens"])
def test_plain_rejects_every_invalid_limit_name_before_request(monkeypatch, wire, bad):
    monkeypatch.setenv("PUDDLES_AGENT_MAX_TOKENS_PARAM", bad)
    with pytest.raises(ConfigError, match="PUDDLES_AGENT_MAX_TOKENS_PARAM"):
        PlainLLM.from_env()
    assert wire.requests == []


@pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "PUDDLES_AGENT_MODEL"])
def test_plain_missing_credentials_refused_before_request(monkeypatch, wire, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ConfigError, match=missing):
        PlainLLM.from_env()
    assert wire.requests == []


def new_note(store, user="person-a", fact="Likes quiet walks."):
    return to_record(NoteAction(action="add", kind="goals", fact=fact), user_id=user,
                     session_id="acceptance-seed", note_id=store.next_note_id(user), today=TODAY, now=NOW)


def seed(store, user="person-a", fact="Likes quiet walks."):
    note = new_note(store, user, fact)
    store.write(note)
    store.save_index(LiveNoteIndex(user_id=user, notes=[
        LiveNote(note_id=note.id, kind=note.kind, last_confirmed=TODAY)]))
    return note


def session():
    return Session(session_id="acceptance-private", policy=None, helper_client=None, checker_client=None,
                   path=crisis.load_path(), user_id="person-a", remember_me=False,
                   consent_for_training=False, turns=[Turn(speaker="user", text="private-sentinel: forget it")])


def test_private_plain_forget_has_no_sdk_span_usage_or_retained_words(monkeypatch, tmp_path, wire, caplog):
    import openai
    import mlflow
    from puddles import usage
    from puddles.inference import live

    def forbidden(*args, **kwargs):
        pytest.fail("private note-taker entered SDK, tracing, or usage logging")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(mlflow, "start_span", forbidden)
    monkeypatch.setattr(usage.Meter, "_write", forbidden)
    marker = contextvars.ContextVar("acceptance_span", default=None)
    marker.set("inherited span")

    def on_send():
        assert marker.get() is None
        assert usage.current() == (None, None)

    wire.on_send = on_send
    wire.content = '{"forget_everything": true, "actions": []}'
    store = NoteStore(FakeUserMemory(), tmp_path)
    old = seed(store)
    private = session()
    # Setting an active meter exercises the empty-context guard in finish_session.
    with usage.metering(usage.Meter(tmp_path), "note-taker"):
        record, result = live.finish_session(private, store, PlainLLM.from_env(), today=TODAY, now=NOW)
    assert record is None and not result.failed and result.deleted == [old.id]
    assert not private.turns and store.helper_notes("person-a", TODAY) == []
    assert old.id not in store.memory.entries
    assert len(wire.requests) == 1 and b"private-sentinel" in wire.requests[0][0].data
    assert "private-sentinel" not in caplog.text
    assert all(b"private-sentinel" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())
    assert not (tmp_path / "log/usage.jsonl").exists()


def test_private_route_refuses_traced_wrapped_and_subclass_clients(monkeypatch, tmp_path, wire):
    from puddles import tracing

    plain = PlainLLM.from_env()
    with pytest.raises(ValueError, match="traced"):
        PlainLLM(tracing.TracedClient(plain.client, "note-taker"), max_tokens=20)

    class SubClient(OpenAICompatClient):
        pass

    class Wrapper:
        def complete_json(self, *args, **kwargs):
            pytest.fail("refused wrapper was called")

    for client in (SubClient("https://model.invalid", "test"), Wrapper()):
        llm = PlainLLM(client, max_tokens=20)
        llm.untraced = True  # cannot bypass the PlainLLM exact-client check
        assert not known_untraced(llm)
        with pytest.raises(ValueError, match="untraced"):
            finish_session(session(), NoteStore(FakeUserMemory(), tmp_path), llm)
    assert wire.requests == []


def test_existing_helper_simuser_and_json_request_bytes_unchanged(monkeypatch, wire, policy, scenario_factory):
    """Execute the parent serializer in memory and compare actual HTTP bytes for existing callers."""
    from loop.helper import helper_reply
    from loop.simulated_user import next_user_line

    source = subprocess.check_output(["git", "show", "HEAD~1:loop/llm.py"], cwd=ROOT, text=True)
    previous = types.ModuleType("loop._acceptance_parent_llm")
    monkeypatch.setitem(sys.modules, previous.__name__, previous)
    exec(compile(source, "HEAD~1:loop/llm.py", "exec"), previous.__dict__)
    scenario = scenario_factory()

    class Answer(BaseModel):
        answer: str

    batches = []
    for client_type in (previous.OpenAICompatClient, OpenAICompatClient):
        wire.requests.clear()
        client = client_type("https://model.invalid/v1", "test", extra_body={"custom": True})
        wire.content = "That sounds difficult."
        helper_reply(client, policy, [Turn(speaker="user", text="Worried.")], session_id="s", turn=1, seed=7)
        next_user_line(client, scenario, [], seed=8)
        client.request("s", [], max_tokens=23, seed=9)
        wire.content = '{"answer": "ok"}'
        client.complete_json("s", [], Answer, max_tokens=24)
        batches.append([req.data for req, _ in wire.requests])
    assert batches[0] == batches[1]
    assert [json.loads(raw)["temperature"] for raw in batches[1]] == [0.7, 0.9, 0.7, 0.0]


@pytest.mark.parametrize("settings, expected", [({}, (300.0, 5.0)),
    ({"env": True}, (401.0, 2.5)), ({"env": True, "write_wait_s": 81, "recall_timeout_s": 1}, (81.0, 1.0))])
def test_note_settings_reach_wait_and_recall(monkeypatch, tmp_path, settings, expected):
    class Memory(FakeUserMemory):
        def wait(self, ingestion_id, timeout_s=60):
            self.wait_seen = timeout_s
            return True

        def recall(self, query, identity, *, timeout_s=None):
            self.recall_seen = (identity, timeout_s)
            raise TimeoutError("offline recall timeout")

    options = dict(settings)
    if options.pop("env", False):
        monkeypatch.setenv("PUDDLES_NOTE_WRITE_WAIT_S", "401")
        monkeypatch.setenv("PUDDLES_NOTE_RECALL_TIMEOUT_S", "2.5")
    memory = Memory()
    store = NoteStore(memory, tmp_path, **options)
    note = seed(store)
    assert store.helper_notes("person-a", TODAY) == [note.fact]
    assert memory.wait_seen == expected[0] and memory.recall_seen == ("person-a", expected[1])


@pytest.mark.parametrize("setting", ["write_wait_s", "recall_timeout_s"])
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), -float("inf"), "", "later", True])
@pytest.mark.parametrize("via_env", [False, True])
def test_note_invalid_timeouts_refused(monkeypatch, tmp_path, setting, bad, via_env):
    env = "PUDDLES_NOTE_" + setting.upper()
    if via_env:
        monkeypatch.setenv(env, str(bad))
    with pytest.raises(ValueError, match=env if via_env else setting):
        NoteStore(FakeUserMemory(), tmp_path, **({} if via_env else {setting: bad}))
    assert not (tmp_path / "notes").exists()


def test_recall_http_timeout_is_per_request_and_body_unchanged(tmp_path):
    requests = []

    def answer(req):
        requests.append(req)
        return httpx.Response(200, json={"results": []})

    memory = PastClient(api_key="placeholder", base_url="https://past.invalid", timeout=30,
                        transport=httpx.MockTransport(answer))
    memory.recall("query", "person-a")
    memory.recall("query", "person-a", timeout_s=1.25)
    memory.recall("query", "person-a")
    assert [r.content for r in requests] == [requests[0].content] * 3
    assert json.loads(requests[0].content) == {"query": "query", "identity": "person-a", "level": "low", "limit": 20}
    assert [r.extensions["timeout"] for r in requests] == [
        dict.fromkeys(("connect", "read", "write", "pool"), value) for value in (30, 1.25, 30)]
    memory._http.close()


@pytest.mark.parametrize("recall_fails", [False, True])
def test_forget_stays_hidden_through_recall_timeout_and_restart(tmp_path, recall_fails):
    class Memory(FakeUserMemory):
        def recall(self, query, identity, *, timeout_s=None):
            assert timeout_s == 5
            if recall_fails:
                raise TimeoutError("offline")
            return [{"content": "remote injected text", "sources": [{"sourceId": forgotten.id}]}]

    memory = Memory(fail_deletes=100)
    store = NoteStore(memory, tmp_path)
    forgotten = seed(store)
    other = seed(store, "person-b", "Other person's private fact.")
    assert store.delete("person-a", [forgotten.id]) == [forgotten.id]
    restarted = NoteStore(memory, tmp_path)
    assert restarted.helper_notes("person-a", TODAY) == []
    assert restarted.pending_deletes("person-a") == [forgotten.id]
    assert forgotten.id in memory.entries and other.id in memory.entries
    assert restarted.helper_notes("person-b", TODAY) == [other.fact]
    for identity in TRAINING_IDENTITIES:
        assert FakeUserMemory.recall(memory, "quiet walks", identity) == []
    memory.fail_deletes = 0
    assert restarted.delete("person-a", []) == []
    assert forgotten.id not in memory.entries and other.id in memory.entries


@pytest.mark.parametrize("confirmed", [True, False])
def test_write_wait_order_and_interruption_remain_recoverable(tmp_path, confirmed):
    class Stopped(BaseException):
        pass

    class Memory(FakeUserMemory):
        def ingest(self, entry):
            assert store._writing("person-a")[0]["note_id"] == entry.id
            assert store.index("person-a").notes == [] and store.stored("person-a") == {}
            assert self.audiences[entry.audience] == ["person-a", "note-taker"]
            return super().ingest(entry)

        def wait(self, ingestion_id, timeout_s=60):
            assert timeout_s == 300
            if confirmed:
                raise Stopped()
            return False

    memory = Memory()
    store = NoteStore(memory, tmp_path)
    note = new_note(store)
    with pytest.raises(Stopped if confirmed else PastError):
        store.write(note)
    restarted = NoteStore(memory, tmp_path)
    assert restarted.helper_notes("person-a", TODAY) == []
    assert restarted.pending_deletes("person-a") == [note.id]
    assert restarted.delete("person-a", []) == []
    assert note.id not in memory.entries and restarted.stored("person-a") == {}


def test_real_past_81_second_ingestion_survives_without_real_sleep(monkeypatch, tmp_path):
    import puddles.memory as module

    clock = types.SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda duration: setattr(clock, "now", clock.now + duration)
    monkeypatch.setattr(module, "time", clock)
    seen = []

    def answer(req):
        seen.append(req)
        if req.url.path.startswith("/audiences/"):
            return httpx.Response(200, json={})
        if req.url.path == "/ingest":
            return httpx.Response(202, json={"ingestionId": "ing-1"})
        if req.url.path == "/ingest/ing-1":
            return httpx.Response(200, json={"status": "completed" if clock.now >= 81 else "processing"})
        if req.url.path == "/recall":
            assert req.extensions["timeout"]["read"] == 5
            raise httpx.ReadTimeout("offline", request=req)
        pytest.fail(f"unexpected past request: {req.method} {req.url}")

    memory = PastUserMemory(api_key="placeholder", base_url="https://past.invalid",
                            transport=httpx.MockTransport(answer))
    store = NoteStore(memory, tmp_path)
    note = seed(store)
    assert clock.now == 82 and store.helper_notes("person-a", TODAY) == [note.fact]
    assert store.pending_deletes("person-a") == []
    assert not any(req.method == "DELETE" for req in seen)
    memory._http.close()


def test_crisis_loads_five_and_runtime_uses_it_without_waiting(monkeypatch, policy):
    from puddles.inference import turn

    path = crisis.load_path()
    assert path.timeout_s == 5
    deadlines = []

    class Job:
        def __init__(self, fn):
            self.started = self.finished = 100.0
            self.error = self.value = None

        def start(self):
            pass

        def close(self, until=None):
            deadlines.append(until)
            return False

    monkeypatch.setattr(turn, "_Job", Job)
    private = session()
    private.policy = policy
    result = turn.run_turn(private, "I feel uneasy.")
    assert deadlines[0] == 105.0
    assert result.check_status == "late" and result.reply.text == path.holding_line
    result = turn.run_turn(private, "I still feel uneasy.")
    assert result.reply.text == path.offline_line
