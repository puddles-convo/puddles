"""Acceptance from docs/specs/crisis-path.md, without the implementation.

Safety coverage comes first: release matrix, stale results, helper isolation,
and consent. Fixed language and keywords are always loaded through eval.crisis.
Thread gates have short bounds and are opened in finally blocks; no real models.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from io import StringIO
import json
import logging
import os
import re
import socket
from threading import Event, Lock, Thread, current_thread

import mlflow
from mlflow.tracking import MlflowClient
import pytest
from pydantic import ValidationError

from loop import helper as helper_api
from loop import llm as llm_api
from loop.prompts import helper_messages, render_helper_system
from puddles import tracing
from puddles.eval import crisis as owned
from puddles.inference import crisis, release, session_log, turn
from schemas import CheckedMessage, ConversationRecord, CrisisCheckerInput, HelperReply, Turn
from scripts import chat


class Client:
    """Only complete(), so using JSON, retrying, or extra calls can fail."""

    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = []
        self.lock = Lock()
        self.threads = []
        self.started = Event()
        self.finished = Event()

    def complete(self, system, messages, *, max_tokens, temperature=0.7, seed=None):
        with self.lock:
            self.calls.append(deepcopy(dict(system=system, messages=messages,
                                           max_tokens=max_tokens, temperature=temperature,
                                           seed=seed)))
            self.threads.append(current_thread())
            answer = next(self.answers)  # Exhaustion exposes an unexpected call.
        self.started.set()
        try:
            if callable(answer):
                answer = answer()
            if isinstance(answer, Exception):
                raise answer
            return answer
        finally:
            self.finished.set()


class Gate:
    def __init__(self, answer):
        self.open = Event()
        self.entered = Event()
        self.answer = answer

    def __call__(self):
        self.entered.set()
        if not self.open.wait(0.3):
            raise RuntimeError("test gate was not opened")
        return self.answer


class Background:
    """Bound a synchronous API under test without an executor's blocking exit."""

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

    def result(self):
        assert self.done.wait(0.15), "turn waited for a model it must discard"
        if self.error is not None:
            raise self.error
        return self.value


def checker_line(risk="none", category=None, reason="acceptance verdict"):
    line = f"{risk} | {category or '-'} | {reason}"
    owned.parse_checker_line(line)  # The owner, not this fake, defines validity.
    return line


def checked(risk="none", category=None, **changes):
    values = dict(session_id="acceptance", turn=1, text="A message.", risk=risk,
                  category=category, reason="acceptance verdict")
    values.update(changes)
    return CheckedMessage(**values)


def draft(**changes):
    values = dict(session_id="draft-session", turn=99, policy_version="draft-policy",
                  text="A helper draft that must be gated.")
    values.update(changes)
    return HelperReply(**values)


def patch_dependency(monkeypatch, owner, name, replacement):
    """Replace the public dependency and any imported aliases by identity."""
    original = getattr(owner, name)
    for module in (crisis, turn, chat):
        for alias, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, alias, replacement)
    monkeypatch.setattr(owner, name, replacement)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.endswith("API_KEY") or key.startswith("MLFLOW_") or key.startswith("HELPER_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("crisis acceptance tests forbid network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)


@pytest.fixture
def path():
    return crisis.load_path()


@pytest.fixture
def session_factory(policy, path):
    def make(**changes):
        values = dict(session_id="acceptance", policy=policy, path=path,
                      helper_client=Client("A safe helper reply."),
                      checker_client=Client(checker_line()), deadline_s=0.1,
                      helper_wait_s=0.1)
        values.update(changes)
        return turn.Session(**values)
    return make


@pytest.fixture
def local_store(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield MlflowClient()
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()


def store_contents(client):
    mlflow.flush_trace_async_logging()
    ids = [exp.experiment_id for exp in client.search_experiments()]
    traces = mlflow.search_traces(experiment_ids=ids, return_type="list")
    runs = client.search_runs(experiment_ids=ids)
    return traces, runs


def assert_fixed(reply, text):
    assert reply.text == text
    assert reply.fixed_handoff_response is True


def assert_timings(timings):
    assert set(timings) == {"keyword", "checker", "helper", "total"}
    assert all(type(value) is int and value >= 0 for value in timings.values())


# Section 2: only rule 3 releases a helper draft. Include every section 6 row,
# missing verdicts, all crisis categories, and statuses carrying stale payloads.
RELEASE_CASES = [
    ("none", "answered", "none", None, "acceptance", 1),
    ("concern", "answered", "concern", None, "acceptance", 1),
    *[(f"crisis-{category}", "answered", "crisis", category, "acceptance", 1)
      for category in ("self", "danger", "unsure", None)],
    *[(status, status, None, None, "acceptance", 1)
      for status in ("failed", "late", "unreadable")],
    ("answered-missing", "answered", None, None, "acceptance", 1),
    *[(f"{risk}-{identity}", "answered", risk, "danger" if risk == "crisis" else None,
       "other" if identity == "session" else "acceptance", 2 if identity == "turn" else 1)
      for risk in ("none", "concern", "crisis") for identity in ("session", "turn")],
    *[(f"{status}-with-{risk}", status, risk, "self" if risk == "crisis" else None,
       "acceptance", 1)
      for status in ("failed", "late", "unreadable") for risk in ("none", "crisis")],
]


@pytest.mark.parametrize("case", RELEASE_CASES, ids=lambda case: case[0])
@pytest.mark.parametrize("has_keyword", [False, True])
@pytest.mark.parametrize("has_draft", [False, True])
@pytest.mark.parametrize("failed_before", [0, 1, 4])
def test_release_full_safety_matrix(path, case, has_keyword, has_draft, failed_before):
    _, status, risk, category, sid, number = case
    verdict = None if risk is None else checked(risk, category, session_id=sid, turn=number)
    candidate = draft() if has_draft else None
    keyword = owned.load_keywords()[0] if has_keyword else None
    reply, failures = release.release(
        session_id="acceptance", turn=1, policy_version="policy-acceptance", path=path,
        keyword=keyword, check_status=status, checked=verdict, draft=candidate,
        failed_before=failed_before,
    )
    counts = status == "answered" and verdict is not None and sid == "acceptance" and number == 1
    assert failures == (0 if counts else failed_before + 1)
    if has_keyword and counts and risk == "crisis" and category is None:
        # Rule 1 omits rule 2's explicit missing-category fallback. Only the
        # fixed-response requirement is certain for this schema-valid input.
        assert reply.fixed_handoff_response is True
        assert reply.text in path.fixed_responses.values()
    elif has_keyword or (counts and risk == "crisis"):
        expected = path.fixed_responses[(category or "unsure") if counts and risk == "crisis" else "unsure"]
        assert_fixed(reply, expected)
    elif counts and has_draft:
        assert reply.text == candidate.text
        assert reply.fixed_handoff_response is False
    else:
        assert_fixed(reply, path.offline_line if not counts and failures >= 2 else path.holding_line)
    assert (reply.session_id, reply.turn, reply.policy_version) == ("acceptance", 1, "policy-acceptance")


# Section 1: source ownership, exact checker boundary, identity from code.
def test_load_path_uses_owner_values():
    source = owned.load_crisis_path()
    value = crisis.load_path()
    assert value.version == source["version"]
    assert value.system == source["checker"]["system"]
    assert value.user_template == source["checker"]["user_template"]
    assert value.timeout_s == source["checker"]["timeout_seconds"]
    assert value.fixed_responses == source["fixed_responses"]
    assert value.holding_line == source["holding_line"]
    assert value.offline_line == source["offline_line"]
    assert value.keywords == owned.load_keywords()
    assert crisis.RECENT_TURNS == 6 and crisis.CHECKER_MAX_TOKENS == 40


def test_load_path_reloads_owned_sources(monkeypatch):
    source = deepcopy(owned.load_crisis_path())
    source["version"] = "acceptance-source-version"
    source["checker"]["timeout_seconds"] = 0.017
    source["checker"]["system"], source["checker"]["user_template"] = (
        source["checker"]["user_template"], source["checker"]["system"])
    source["holding_line"], source["offline_line"] = source["offline_line"], source["holding_line"]
    source["fixed_responses"] = dict.fromkeys(source["fixed_responses"], source["holding_line"])
    # Reassign owner-loaded text; no prompt or fixed sentence is restated here.
    patch_dependency(monkeypatch, owned, "load_crisis_path", lambda: deepcopy(source))
    keywords = list(reversed(owned.load_keywords()))
    patch_dependency(monkeypatch, owned, "load_keywords", lambda: list(keywords))
    value = crisis.load_path()
    assert (value.version, value.timeout_s, value.keywords) == (source["version"], 0.017, keywords)
    assert value.system == source["checker"]["system"]
    assert value.user_template == source["checker"]["user_template"]
    assert value.fixed_responses == source["fixed_responses"]
    assert value.holding_line == source["holding_line"] and value.offline_line == source["offline_line"]


@pytest.mark.parametrize("count", [0, 2, 6, 9])
def test_build_input_and_render_use_last_six_prior_turns(path, count):
    history = [Turn(speaker="user" if i % 2 == 0 else "helper", text=f"prior {i}") for i in range(count)]
    original = deepcopy(history)
    text = "Current message with {braces}\nand another line."
    inp = crisis.build_input("acceptance", 12, history, text)
    assert inp == CrisisCheckerInput(session_id="acceptance", turn=12, recent_turns=history[-6:], text=text)
    expected = path.user_template.format(
        recent_turns="\n".join(f"{item.speaker}: {item.text}" for item in history[-6:]), text=text)
    assert crisis.render_user(inp, path) == expected
    assert history == original


@pytest.mark.parametrize("risk,category", [("none", None), ("concern", None),
                                           ("crisis", "self"), ("crisis", "danger"), ("crisis", "unsure")])
def test_checker_complete_and_owned_parser(path, risk, category):
    raw = " \n" + checker_line(risk, category, "parser provenance") + "\t "
    inp = crisis.build_input("identity-from-code", 7, [], "Message from input.")
    client = Client(raw)
    record = crisis.run(client, inp, path)
    expected = owned.parse_checker_line(raw.strip())
    assert crisis.check(raw) == expected
    assert (record.risk, record.category, record.reason) == expected
    assert (record.session_id, record.turn, record.text) == (inp.session_id, inp.turn, inp.text)
    assert crisis.to_record(inp, *expected) == record
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["system"] == path.system
    assert call["messages"] == [{"role": "user", "content": crisis.render_user(inp, path)}]
    assert (call["max_tokens"], call["temperature"]) == (40, 0.0)


@pytest.mark.parametrize("raw", ["", "not a verdict", "none | self | invalid category",
                                 '{"risk":"none","session_id":"forged","turn":99}'])
def test_unreadable_checker_is_not_none_and_never_retried(path, raw):
    with pytest.raises(ValueError):
        owned.parse_checker_line(raw.strip())
    with pytest.raises(ValueError):
        crisis.check(raw)
    client = Client(raw, checker_line())
    with pytest.raises(crisis.CheckFailed) as error:
        crisis.run(client, crisis.build_input("acceptance", 1, [], "Message."), path)
    assert error.value.status == "unreadable"
    assert len(client.calls) == 1


@pytest.mark.parametrize("error", [RuntimeError("transport failure"), ValueError("client failure"), TimeoutError()])
def test_client_exception_is_failed_without_retry(path, error):
    client = Client(error, checker_line())
    with pytest.raises(crisis.CheckFailed) as caught:
        crisis.run(client, crisis.build_input("acceptance", 1, [], "Message."), path)
    assert caught.value.status == "failed"
    assert len(client.calls) == 1


# Section 4: threaded orchestration and irreversible release decisions.
@pytest.mark.parametrize("risk,category,keyword", [("none", None, False), ("concern", None, False),
                                                    ("crisis", "danger", False), ("none", None, True)])
def test_helper_receives_only_unchanged_helper_context(session_factory, risk, category, keyword):
    helper = Client("A helper answer.")
    checker = Client(checker_line(risk, category, "CHECKER_ONLY_SENTINEL"))
    session = session_factory(helper_client=helper, checker_client=checker, user_notes=("A remembered preference.",))
    session.turns.extend([Turn(speaker="user", text="An earlier message."),
                          Turn(speaker="helper", text="An earlier spoken answer.")])
    before = deepcopy(session.turns)
    text = owned.load_keywords()[0] if keyword else "An ordinary new message."
    expected_turns = before + [Turn(speaker="user", text=text)]
    result = turn.run_turn(session, text)
    assert helper.finished.wait(0.15)
    assert len(helper.calls) == len(checker.calls) == 1
    call = helper.calls[0]
    assert call["system"] == render_helper_system(session.policy, session.user_notes)
    assert call["messages"] == helper_messages(expected_turns)
    assert "CHECKER_ONLY_SENTINEL" not in json.dumps(call)
    assert helper.threads[0] is not checker.threads[0]
    assert helper.threads[0] is not current_thread() and checker.threads[0] is not current_thread()
    assert result.keyword == owned.keyword_match(text, session.path.keywords)
    assert result.check_status == "answered"
    assert result.checked.text == text
    if risk == "crisis" or keyword:
        assert_fixed(result.reply, session.path.fixed_responses[category if risk == "crisis" else "unsure"])
    else:
        assert result.reply.text == "A helper answer."
        assert result.helper_failed is False
    assert session.turns == expected_turns + [Turn(speaker="helper", text=result.reply.text)]
    assert len(session.lines) == 1
    line = session.lines[0]
    assert (line.session_id, line.turn, line.user_text) == (session.session_id, 1, text)
    assert line.reply == result.reply and line.checked == result.checked
    assert line.check_status == result.check_status and line.keyword_matched == keyword
    assert line.timings_ms == result.timings_ms
    assert_timings(result.timings_ms)


def test_checker_and_helper_start_before_either_finishes(session_factory):
    helper_gate, checker_gate = Gate("A helper answer."), Gate(checker_line())
    session = session_factory(helper_client=Client(helper_gate), checker_client=Client(checker_gate))
    pending = Background(lambda: turn.run_turn(session, "A message."))
    try:
        assert helper_gate.entered.wait(0.15)
        assert checker_gate.entered.wait(0.15), "checker waited for helper completion"
        helper_gate.open.set()
        checker_gate.open.set()
        assert pending.result().reply.text == "A helper answer."
    finally:
        helper_gate.open.set()
        checker_gate.open.set()
        pending.thread.join(0.05)


@pytest.mark.parametrize("outcome", ["crisis", "keyword", "failed", "unreadable", "late", "offline"])
def test_discarded_helper_never_delays_fixed_response(session_factory, outcome):
    gate = Gate("A draft that must not be spoken.")
    response = {"crisis": checker_line("crisis", "self"), "keyword": checker_line(),
                "failed": RuntimeError("checker failed"), "unreadable": "invalid answer",
                "offline": RuntimeError("checker failed")}.get(outcome)
    check_gate = Gate(checker_line())
    session = session_factory(helper_client=Client(gate), checker_client=Client(check_gate if outcome == "late" else response),
                              deadline_s=0.015, helper_wait_s=0.25)
    if outcome == "offline":
        session.failed_checks = 1
    text = owned.load_keywords()[0] if outcome == "keyword" else "An ordinary message."
    pending = Background(lambda: turn.run_turn(session, text))
    try:
        result = pending.result()  # Both gates are still shut.
        assert gate.entered.wait(0.1)
        if outcome in {"crisis", "keyword"}:
            expected = session.path.fixed_responses["self" if outcome == "crisis" else "unsure"]
        else:
            expected = session.path.offline_line if outcome == "offline" else session.path.holding_line
        assert_fixed(result.reply, expected)
        if outcome in {"failed", "unreadable", "late", "offline"}:
            assert result.checked is None
            assert result.check_status == ("failed" if outcome == "offline" else outcome)
            assert session.lines[0].checked is None
    finally:
        gate.open.set()
        check_gate.open.set()
        pending.thread.join(0.05)
        session.helper_client.finished.wait(0.05)
        session.checker_client.finished.wait(0.05)


@pytest.mark.parametrize("helper_outcome", ["raised", "late"])
def test_usable_check_with_missing_helper_holds(session_factory, helper_outcome):
    gate = Gate("Too late to use.")
    client = Client(RuntimeError("helper failure") if helper_outcome == "raised" else gate)
    session = session_factory(helper_client=client, helper_wait_s=0.015)
    try:
        result = turn.run_turn(session, "An ordinary message.")
        assert result.check_status == "answered" and result.checked.risk == "none"
        assert result.helper_failed is True
        assert_fixed(result.reply, session.path.holding_line)
        assert session.failed_checks == 0
    finally:
        gate.open.set()
        client.finished.wait(0.05)


@pytest.mark.parametrize("late_risk,late_category", [("none", None), ("crisis", "self")])
def test_late_checker_cannot_rewrite_released_or_next_turn(session_factory, late_risk, late_category):
    first_gate = Gate(checker_line(late_risk, late_category))
    second_gate = Gate(checker_line("crisis", "danger"))
    session = session_factory(checker_client=Client(first_gate, second_gate),
                              helper_client=Client("First unused draft.", "Second unused draft."), deadline_s=0.015)
    second = None
    try:
        first = turn.run_turn(session, "First message.")
        assert first.check_status == "late" and first.checked is None
        assert_fixed(first.reply, session.path.holding_line)
        snapshot = session.lines[0].model_dump(mode="json")
        session.deadline_s = 0.15
        second = Background(lambda: turn.run_turn(session, "Second message."))
        assert second_gate.entered.wait(0.1)
        first_gate.open.set()
        assert not second.done.wait(0.015), "old verdict completed the current turn"
        second_gate.open.set()
        result = second.result()
        assert_fixed(result.reply, session.path.fixed_responses["danger"])
        assert result.checked.turn == 2 and result.checked.text == "Second message."
        assert session.lines[0].model_dump(mode="json") == snapshot
        assert first.checked is None and first.check_status == "late"
        assert session.failed_checks == 0
        assert [line.turn for line in session.lines] == [1, 2]
        assert session.turns == [Turn(speaker="user", text="First message."),
                                 Turn(speaker="helper", text=first.reply.text),
                                 Turn(speaker="user", text="Second message."),
                                 Turn(speaker="helper", text=result.reply.text)]
    finally:
        first_gate.open.set()
        second_gate.open.set()
        if second is not None:
            second.thread.join(0.05)


def test_failed_check_streak_resets_only_on_answer(session_factory):
    session = session_factory(checker_client=Client(RuntimeError("one"), "unreadable",
                                                    checker_line("concern"), RuntimeError("again")),
                              helper_client=Client(*["A helper answer."] * 4))
    results = [turn.run_turn(session, f"Message {number}.") for number in range(4)]
    assert [result.check_status for result in results] == ["failed", "unreadable", "answered", "failed"]
    assert [result.checked is None for result in results] == [True, True, False, True]
    assert [result.reply.text for result in results] == [session.path.holding_line, session.path.offline_line,
                                                       "A helper answer.", session.path.holding_line]
    assert session.failed_checks == 1


def test_unexpected_checker_exception_fails_closed(monkeypatch, session_factory):
    def broken(*args, **kwargs):
        raise LookupError("unexpected checker wrapper failure")
    patch_dependency(monkeypatch, crisis, "run", broken)
    result = turn.run_turn(session_factory(), "Ordinary message.")
    assert result.check_status == "failed" and result.checked is None
    assert_fixed(result.reply, crisis.load_path().holding_line)


def test_same_session_serializes_calls(session_factory):
    gate = Gate(checker_line())
    session = session_factory(checker_client=Client(gate, checker_line()),
                              helper_client=Client("First reply.", "Second reply."), deadline_s=0.2)
    first = Background(lambda: turn.run_turn(session, "First message."))
    second = None
    try:
        assert gate.entered.wait(0.1)
        second = Background(lambda: turn.run_turn(session, "Second message."))
        assert not second.done.wait(0.015)
        assert len(session.checker_client.calls) == 1
        gate.open.set()
        assert first.result().reply.turn == 1
        assert second.result().reply.turn == 2
        assert [line.user_text for line in session.lines] == ["First message.", "Second message."]
        assert len(session.turns) == 4
    finally:
        gate.open.set()
        first.thread.join(0.05)
        if second is not None:
            second.thread.join(0.05)


# Section 3: a nonanswer must never acquire a fabricated risk of none.
@pytest.mark.parametrize("status", ["answered", "failed", "late", "unreadable"])
@pytest.mark.parametrize("has_checked", [False, True])
def test_turn_line_enforces_answer_presence(status, has_checked):
    values = dict(session_id="acceptance", turn=1, user_text="A message.", keyword_matched=False,
                  check_status=status, checked=checked() if has_checked else None, reply=draft())
    if (status == "answered") != has_checked:
        with pytest.raises(ValidationError):
            session_log.TurnLine(**values)
    else:
        line = session_log.TurnLine(**values)
        assert session_log.TurnLine.model_validate_json(line.model_dump_json()) == line


@pytest.mark.parametrize("sid", ["", "../escape", "/absolute", "a/b", "a\\b", ".hidden", "a.b",
                                 "a b", "-prefix", "_prefix", "\u00e9", "abc\n"])
def test_session_paths_reject_unsafe_ids(tmp_path, sid):
    for function in (session_log.session_path, session_log.record_path):
        with pytest.raises(ValueError):
            function(tmp_path, sid)


@pytest.mark.parametrize("sid", ["a", "A0_name-9", "123"])
def test_session_paths_exact_names(tmp_path, sid):
    assert session_log.session_path(tmp_path, sid) == tmp_path / "sessions" / f"{sid}.jsonl"
    assert session_log.record_path(tmp_path, sid) == tmp_path / "sessions" / f"{sid}.record.json"


def test_session_log_jsonl_order_and_spoken_record(tmp_path, path):
    assert session_log.read_lines(tmp_path, "acceptance") == []
    assert list(tmp_path.iterdir()) == []
    lines = [session_log.TurnLine(session_id="acceptance", turn=1, user_text="Message one.\nStill one turn.",
                                  keyword_matched=False, check_status="failed", checked=None,
                                  reply=draft(session_id="acceptance", turn=1, text=path.holding_line,
                                              fixed_handoff_response=True), timings_ms={"total": 5}),
             session_log.TurnLine(session_id="acceptance", turn=2, user_text="Message two.",
                                  keyword_matched=False, check_status="answered",
                                  checked=checked(turn=2, text="Message two."),
                                  reply=draft(session_id="acceptance", turn=2), timings_ms={"total": 8})]
    for line in lines:
        session_log.append_line(tmp_path, line)
    raw = session_log.session_path(tmp_path, "acceptance").read_text().splitlines()
    assert len(raw) == 2
    assert [session_log.TurnLine.model_validate_json(item) for item in raw] == lines
    assert json.loads(raw[0])["checked"] is None
    assert session_log.read_lines(tmp_path, "acceptance") == lines
    now = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    record = session_log.to_conversation_record(lines, policy_version="record-policy", remember_me=True,
                                                 consent_for_training=True, now=now)
    assert (record.id, record.source, record.written_by, record.scenario_id) == ("acceptance", "live", "helper", None)
    assert record.policy_version == "record-policy"
    assert record.remember_me is True and record.consent_for_training is True
    assert record.turns == [item for line in lines for item in
                            (Turn(speaker="user", text=line.user_text), Turn(speaker="helper", text=line.reply.text))]
    assert record.timings_ms == [line.timings_ms for line in lines]
    with pytest.raises(ValueError):
        session_log.to_conversation_record([], policy_version="p", remember_me=False, consent_for_training=False)


# Section 4 consent: inspect actual local traces, runs, files, and captured logs.
@pytest.mark.parametrize("remember", [False, True])
@pytest.mark.parametrize("outcome", ["safe", "failed", "unreadable", "late", "crisis", "keyword"])
def test_without_training_no_disk_trace_run_or_user_log(local_store, tmp_path, caplog, capsys,
                                                        session_factory, remember, outcome):
    marker = "PRIVATE_TYPED_MARKER_74831"
    caplog.set_level(logging.DEBUG)
    late_gate = Gate(checker_line())
    answer = {"safe": checker_line(), "failed": RuntimeError(marker), "unreadable": marker,
              "late": late_gate, "crisis": checker_line("crisis", "self"), "keyword": checker_line()}[outcome]
    session = session_factory(root=tmp_path / "session-root", remember_me=remember,
                              checker_client=Client(answer), deadline_s=0.015 if outcome == "late" else 0.1,
                              helper_client=Client(RuntimeError(marker) if outcome == "failed" else "A helper answer."))
    before_traces, before_runs = store_contents(local_store)
    before_files = {file.relative_to(tmp_path) for file in tmp_path.rglob("*") if file.is_file()}
    text = marker + (" " + owned.load_keywords()[0] if outcome == "keyword" else "")
    try:
        result = turn.run_turn(session, text)
        assert session.helper_client.finished.wait(0.1)
        assert session.lines[0].user_text == text
        assert turn.end_session(session) is None
    finally:
        late_gate.open.set()
        session.checker_client.finished.wait(0.1)
    assert session.lines == [] and session.turns == []
    traces, runs = store_contents(local_store)
    assert len(traces) == len(before_traces) == 0
    assert len(runs) == len(before_runs) == 0
    assert not session.root.exists()
    assert {file.relative_to(tmp_path) for file in tmp_path.rglob("*") if file.is_file()} == before_files
    for file in tmp_path.rglob("*"):
        if file.is_file():
            assert marker.encode() not in file.read_bytes(), file
    captured = capsys.readouterr()
    assert marker not in captured.out + captured.err + caplog.text
    assert result.reply.text != marker


def test_without_training_clients_are_passed_through_unwrapped(monkeypatch, session_factory):
    seen = []
    original_helper, original_checker = helper_api.helper_reply, crisis.run

    def helper(client, *args, **kwargs):
        seen.append(("helper", client))
        return original_helper(client, *args, **kwargs)

    def checker(client, *args, **kwargs):
        seen.append(("checker", client))
        return original_checker(client, *args, **kwargs)

    patch_dependency(monkeypatch, helper_api, "helper_reply", helper)
    patch_dependency(monkeypatch, crisis, "run", checker)
    session = session_factory()
    turn.run_turn(session, "Ordinary message.")
    assert session.helper_client.finished.wait(0.1)
    assert dict(seen) == {"helper": session.helper_client, "checker": session.checker_client}


@pytest.mark.parametrize("remember", [False, True])
def test_training_writes_spoken_lines_and_one_turn_trace(local_store, tmp_path, monkeypatch, session_factory, remember):
    session = session_factory(root=tmp_path / "session-root", consent_for_training=True, remember_me=remember,
                              deadline_s=0.4, helper_wait_s=0.4,
                              checker_client=Client(checker_line(), "bad verdict"),
                              helper_client=Client("First spoken reply.", "Must not be spoken."))
    # A caller already configured tracing; run_turn must not configure it again.
    patch_dependency(monkeypatch, tracing, "setup", lambda *a, **k: pytest.fail("run_turn called setup"))
    seen = []
    original_helper, original_checker = helper_api.helper_reply, crisis.run

    def helper(client, *args, **kwargs):
        seen.append(("helper", client))
        return original_helper(client, *args, **kwargs)

    def checker(client, *args, **kwargs):
        seen.append(("checker", client))
        return original_checker(client, *args, **kwargs)

    patch_dependency(monkeypatch, helper_api, "helper_reply", helper)
    patch_dependency(monkeypatch, crisis, "run", checker)
    first = turn.run_turn(session, "Training message one.")
    # The second helper is discarded. Use the first fully awaited turn for the
    # trace-parent assertions; late/discarded trace lifetime is unspecified.
    traces, _ = store_contents(local_store)
    assert len(traces) == 1
    spans = traces[0].data.spans
    roots = [span for span in spans if span.name == "turn"]
    assert len(roots) == 1
    for name in ("llm:helper", "llm:crisis_checker"):
        model_spans = [span for span in spans if span.name == name]
        assert len(model_spans) == 1
        # The contract fixes containment, not whether intermediate spans exist.
        by_id = {span.span_id: span for span in spans}
        parent = model_spans[0].parent_id
        visited = set()
        while parent != roots[0].span_id:
            assert parent in by_id and parent not in visited
            visited.add(parent)
            parent = by_id[parent].parent_id
    assert {name for name, _ in seen} == {"helper", "checker"}
    for name, wrapper in seen:
        assert isinstance(wrapper, tracing.TracedClient)
        assert wrapper.name == ("crisis_checker" if name == "checker" else "helper")
        assert wrapper.inner is (session.checker_client if name == "checker" else session.helper_client)
    session.helper_client.finished.clear()
    second = turn.run_turn(session, "Training message two.")
    assert session.helper_client.finished.wait(0.1)
    assert_fixed(second.reply, session.path.holding_line)
    persisted = session_log.read_lines(session.root, session.session_id)
    assert persisted == session.lines
    assert persisted[1].check_status == "unreadable" and persisted[1].checked is None
    # Deliberately change memory: finalization is required to read disk back.
    session.lines[0] = session.lines[0].model_copy(update={"user_text": "Memory-only replacement."})
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    record = turn.end_session(session, now=now)
    assert record.turns == [Turn(speaker="user", text="Training message one."),
                            Turn(speaker="helper", text=first.reply.text),
                            Turn(speaker="user", text="Training message two."),
                            Turn(speaker="helper", text=second.reply.text)]
    assert record.remember_me == remember and record.consent_for_training is True
    stored = ConversationRecord.model_validate_json(session_log.record_path(session.root, session.session_id).read_text())
    assert stored == record
    assert session.lines == [] and session.turns == []


def test_training_requires_root_before_any_model_call(session_factory):
    session = session_factory(consent_for_training=True, root=None)
    with pytest.raises(ValueError):
        turn.run_turn(session, "Must never reach a model.")
    assert session.helper_client.calls == session.checker_client.calls == []
    assert session.lines == session.turns == []


@pytest.mark.parametrize("consent", [False, True])
def test_end_empty_session_writes_nothing(tmp_path, session_factory, consent):
    session = session_factory(root=tmp_path / "empty", consent_for_training=consent)
    assert turn.end_session(session) is None
    assert not session.root.exists()
    assert session.turns == session.lines == []


# Section 5: CLI through injected streams and offline clients.
def test_chat_fake_skips_blanks_quits_and_keeps_nothing(tmp_path, policy_path):
    output = StringIO()
    code = chat.main(["--fake", "--policy", str(policy_path), "--root", str(tmp_path / "chat"),
                      "--session-id", "cli-session"],
                     stdin=StringIO("\nAn ordinary message.\n  \n/quit\nNot a message.\n"), stdout=output)
    assert code == 0
    lines = output.getvalue().splitlines()
    assert len([line for line in lines if line.startswith("robot: ")]) == 1
    diagnostic = [line for line in lines if line.startswith("  [")]
    assert len(diagnostic) == 1
    assert any(level in diagnostic[0] for level in ("none", "concern", "crisis", "failed", "late", "unreadable"))
    assert len(re.findall(r"\d+", diagnostic[0])) >= 3
    assert len(lines) == 3
    assert lines[-1].strip()
    assert not (tmp_path / "chat").exists()


@pytest.mark.parametrize("problem", ["missing-policy", "malformed-policy", "configuration"])
def test_chat_errors_return_two_and_one_stderr_line(tmp_path, policy_path, capsys, problem):
    chosen = policy_path
    if problem == "missing-policy":
        chosen = tmp_path / "missing.md"
    elif problem == "malformed-policy":
        chosen.write_text("not a policy\n")
    args = ["--policy", str(chosen), "--root", str(tmp_path / "chat")]
    if problem != "configuration":
        args.append("--fake")
    capsys.readouterr()
    assert chat.main(args, stdin=StringIO("A message.\n"), stdout=StringIO()) == 2
    assert len(capsys.readouterr().err.splitlines()) == 1
    assert not (tmp_path / "chat").exists()


def test_chat_real_mode_uses_helper_endpoint_twice_and_no_setup_without_consent(
        tmp_path, policy_path, monkeypatch):
    clients = iter([Client("A CLI helper reply."), Client(checker_line())])
    prefixes = []

    def client_from_env(prefix):
        prefixes.append(prefix)
        return next(clients)

    patch_dependency(monkeypatch, llm_api, "client_from_env", client_from_env)
    patch_dependency(monkeypatch, tracing, "setup", lambda *a, **k: pytest.fail("setup without training consent"))
    output = StringIO()
    assert chat.main(["--policy", str(policy_path), "--mlflow", f"sqlite:///{tmp_path}/unused.db",
                      "--root", str(tmp_path / "chat"), "--session-id", "cli-live"],
                     stdin=StringIO("An ordinary message.\n"), stdout=output) == 0
    assert prefixes == ["HELPER", "HELPER"]
    assert "robot: A CLI helper reply." in output.getvalue()
    assert not (tmp_path / "unused.db").exists()


def test_chat_training_setup_and_record_flags(tmp_path, policy_path, monkeypatch):
    calls = []
    setup = tracing.setup

    def record_setup(*args, **kwargs):
        calls.append((args, kwargs))
        return setup(*args, **kwargs)

    patch_dependency(monkeypatch, tracing, "setup", record_setup)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    output = StringIO()
    root = tmp_path / "chat"
    try:
        assert chat.main(["--fake", "--policy", str(policy_path), "--training", "yes", "--remember-me", "yes",
                          "--user-id", "person-01", "--session-id", "cli-training", "--root", str(root), "--mlflow", uri],
                         stdin=StringIO("A message.\n/quit\n"), stdout=output) == 0
        assert len(calls) == 1
        assert uri in calls[0][0] or calls[0][1].get("tracking_uri") == uri
        record = ConversationRecord.model_validate_json(session_log.record_path(root, "cli-training").read_text())
        assert record.remember_me is True and record.consent_for_training is True
        assert len(record.turns) == 2
        assert str(session_log.session_path(root, "cli-training")) in output.getvalue().splitlines()[-1]
    finally:
        mlflow.flush_trace_async_logging()
        if mlflow.active_run() is not None:
            mlflow.end_run()


@pytest.mark.parametrize("keyword", owned.load_keywords())
def test_every_owned_keyword_blocks_even_a_none_check(session_factory, keyword):
    # Derive variants from the owner's data instead of retyping keywords.
    text = keyword.upper().replace(" ", " - ")
    session = session_factory()
    expected = owned.keyword_match(text, session.path.keywords)
    assert expected is not None
    result = turn.run_turn(session, text)
    assert result.keyword == expected
    assert result.checked.risk == "none"
    assert_fixed(result.reply, session.path.fixed_responses["unsure"])


@pytest.mark.parametrize("kind", ["self", "danger", "unsure", "holding", "offline"])
def test_release_reads_text_from_supplied_path(path, kind):
    # Reassign existing owner-loaded lines to detect a hardcoded second copy.
    changed = replace(path, holding_line=path.offline_line, offline_line=path.holding_line,
                      fixed_responses={"self": path.fixed_responses["danger"],
                                       "danger": path.fixed_responses["unsure"],
                                       "unsure": path.fixed_responses["self"]})
    verdict = checked("crisis", kind) if kind in changed.fixed_responses else None
    reply, _ = release.release(session_id="acceptance", turn=1, policy_version="p", path=changed,
                               keyword=None, check_status="answered" if verdict else "failed",
                               checked=verdict, draft=draft(), failed_before=1 if kind == "offline" else 0)
    expected = changed.fixed_responses[kind] if verdict else getattr(changed, f"{kind}_line")
    assert_fixed(reply, expected)


def test_checker_run_does_not_apply_the_clock(path):
    inp = crisis.build_input("acceptance", 1, [], "A message.")
    result = crisis.run(Client(checker_line()), inp, replace(path, timeout_s=0.0))
    assert result.risk == "none"


def test_turn_uses_path_timeout_when_deadline_is_none(session_factory):
    gate = Gate(checker_line())
    session = session_factory(deadline_s=None, checker_client=Client(gate))
    session.path = replace(session.path, timeout_s=0.015)
    pending = Background(lambda: turn.run_turn(session, "A message."))
    try:
        result = pending.result()
        assert result.check_status == "late" and result.checked is None
        assert_fixed(result.reply, session.path.holding_line)
    finally:
        gate.open.set()
        pending.thread.join(0.05)
        session.checker_client.finished.wait(0.05)


@pytest.mark.parametrize("wrong_identity", ["session", "turn"])
def test_turn_never_releases_for_mismatched_verdict(monkeypatch, session_factory, wrong_identity):
    value = checked(session_id="another" if wrong_identity == "session" else "acceptance",
                    turn=2 if wrong_identity == "turn" else 1)
    patch_dependency(monkeypatch, crisis, "run", lambda *a, **k: value)
    session = session_factory()
    result = turn.run_turn(session, "A message.")
    assert_fixed(result.reply, session.path.holding_line)
    assert session.failed_checks == 1
    # The spec does not define whether this is logged as answered or failed.


def test_a_late_verdict_cannot_complete_another_session(session_factory):
    old_gate, new_gate = Gate(checker_line()), Gate(checker_line("crisis", "danger"))
    old = session_factory(checker_client=Client(old_gate), deadline_s=0.015)
    new = session_factory(session_id="other-session", checker_client=Client(new_gate), deadline_s=0.2)
    pending = None
    try:
        released = turn.run_turn(old, "Old session message.")
        assert released.check_status == "late"
        pending = Background(lambda: turn.run_turn(new, "New session message."))
        assert new_gate.entered.wait(0.1)
        old_gate.open.set()
        assert not pending.done.wait(0.015)
        new_gate.open.set()
        result = pending.result()
        assert_fixed(result.reply, new.path.fixed_responses["danger"])
        assert result.reply.session_id == result.checked.session_id == "other-session"
        assert old.lines[0].check_status == "late" and old.lines[0].checked is None
        assert_fixed(old.lines[0].reply, old.path.holding_line)
    finally:
        old_gate.open.set()
        new_gate.open.set()
        if pending is not None:
            pending.thread.join(0.05)


def test_late_helper_cannot_replace_a_spoken_holding_line(session_factory):
    gate = Gate("A draft arriving after its speaking deadline.")
    client = Client(gate, "A second helper reply.")
    session = session_factory(helper_client=client, helper_wait_s=0.015,
                              checker_client=Client(checker_line(), checker_line()))
    try:
        first = turn.run_turn(session, "First message.")
        assert first.helper_failed is True
        assert_fixed(first.reply, session.path.holding_line)
        gate.open.set()
        assert client.finished.wait(0.1)
        second = turn.run_turn(session, "Second message.")
        assert second.reply.text == "A second helper reply."
        assert session.lines[0].reply == first.reply
        assert session.turns[1].text == session.path.holding_line
        assert session.turns[3].text == second.reply.text
    finally:
        gate.open.set()
        client.finished.wait(0.05)


def test_fake_cli_answers_pass_through_owned_parser(tmp_path, policy_path, monkeypatch):
    seen = []
    parser = owned.parse_checker_line

    def observe(line):
        seen.append(line)
        return parser(line)

    patch_dependency(monkeypatch, owned, "parse_checker_line", observe)
    output = StringIO()
    assert chat.main(["--fake", "--policy", str(policy_path), "--root", str(tmp_path / "chat")],
                     stdin=StringIO("Ordinary message.\n"), stdout=output) == 0
    assert seen, "fake checker bypassed the owner's parser"
    assert all(parser(line)[0] in {"none", "concern", "crisis"} for line in seen)


def test_helper_wait_budget_starts_with_turn_not_checker_answer(session_factory):
    helper_started, checker_started = Event(), Event()

    def helper_answer():
        helper_started.set()
        assert checker_started.wait(0.1)
        Event().wait(0.09)
        return "A draft outside its absolute budget."

    def checker_answer():
        checker_started.set()
        assert helper_started.wait(0.1)
        Event().wait(0.07)
        return checker_line()

    session = session_factory(helper_client=Client(helper_answer), checker_client=Client(checker_answer),
                              deadline_s=0.2, helper_wait_s=0.05)
    try:
        result = turn.run_turn(session, "An ordinary message.")
        assert result.check_status == "answered"
        assert result.helper_failed is True
        assert_fixed(result.reply, session.path.holding_line)
    finally:
        session.helper_client.finished.wait(0.1)
