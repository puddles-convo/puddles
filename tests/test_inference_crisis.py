"""The crisis path, one turn, and what a session leaves behind (docs/specs/crisis-path.md).

Every keyword, prompt and fixed line is read from Ansh's files through his functions; none is typed here.
No network, no keys; deadlines are a few hundredths of a second instead of 3 seconds.
"""
from __future__ import annotations

import dataclasses
import io
import itertools
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import pytest

from loop.errors import LLMError
from loop.helper import helper_reply
from loop.llm import FakeClient
from loop.policy import load_policy
from puddles.eval.crisis import keyword_match, load_crisis_path, load_keywords, parse_checker_line
from puddles.inference import crisis
from puddles.inference.crisis import CHECKER_MAX_TOKENS, RECENT_TURNS, CheckFailed
from puddles.inference.release import release
from puddles.inference.session_log import (TurnLine, append_line, read_lines, record_path, session_path,
                                           to_conversation_record)
from puddles.inference.turn import Session, end_session, run_turn
from schemas import CheckedMessage, HelperReply, Turn

REPO = Path(__file__).resolve().parents[1]
POLICY = load_policy(REPO / "policies" / "policy-v1.md")
PATH = crisis.load_path()
FILE = load_crisis_path()            # the expected texts, straight from Ansh's file
SELF, DANGER, UNSURE = (FILE["fixed_responses"][c] for c in ("self", "danger", "unsure"))
HOLDING, OFFLINE = FILE["holding_line"], FILE["offline_line"]
KEYWORD = load_keywords()[0]
DRAFT = "That sounds really hard, and I am here with you."
SHORT = 0.05                         # the checker's deadline in these tests


def keyword_text() -> str:
    return f"Lately I keep thinking {KEYWORD} and nothing helps"


PLAIN = "Work has been heavy this week and I feel tired all the time."


# ---------- fakes ----------

@pytest.fixture
def store(tmp_path, monkeypatch):
    """A tracking store in tmp_path, for any test where consent is on and spans are written."""
    from puddles import tracing

    monkeypatch.chdir(tmp_path)                  # MLflow keeps some trace files in ./mlruns
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield tmp_path

@pytest.fixture
def gates():
    """Events that hold fake models; all are opened at the end so no thread is left waiting."""
    made: list[threading.Event] = []

    def gate() -> threading.Event:
        made.append(threading.Event())
        return made[-1]

    yield gate
    for event in made:
        event.set()


def line(text: str) -> Callable[[str, list], str]:
    return lambda system, messages: text


def boom(system: str, messages: list) -> str:
    raise LLMError("scripted failure")


def held(gate: threading.Event, text: str, returned: Optional[threading.Event] = None):
    def step(system: str, messages: list) -> str:
        gate.wait(5)
        if returned is not None:
            returned.set()
        return text
    return step


def scripted(*steps: Callable[[str, list], str]) -> FakeClient:
    """One step per call, in order."""
    queue = list(steps)
    return FakeClient(lambda system, messages: queue.pop(0)(system, messages))


def same(step: Callable[[str, list], str]) -> FakeClient:
    return FakeClient(step)


def threads_started_by(fn: Callable[[], object]) -> tuple[object, list[threading.Thread]]:
    """Run fn and return its result with the threads it started (a turn's checker and helper threads)."""
    before = set(threading.enumerate())
    result = fn()
    return result, [t for t in threading.enumerate() if t not in before]


def join_all(threads: list[threading.Thread], timeout: float = 2.0) -> None:
    """Wait, with a bound, until every thread has finished, so what they did can be asserted on."""
    until = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, until - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "a turn's thread did not finish in time"


def make_session(checker: FakeClient, helper: Optional[FakeClient] = None, **kw) -> Session:
    kw.setdefault("deadline_s", SHORT)
    kw.setdefault("helper_wait_s", 2.0)
    return Session(session_id=kw.pop("session_id", "sess-test"), policy=POLICY,
                   helper_client=helper if helper is not None else same(line(DRAFT)),
                   checker_client=checker, path=PATH, **kw)


# ---------- crisis.py ----------

def test_load_path_comes_from_ansh_files():
    assert PATH.keywords == load_keywords()
    assert PATH.system == FILE["checker"]["system"]
    assert PATH.user_template == FILE["checker"]["user_template"]
    assert PATH.timeout_s == float(FILE["checker"]["timeout_seconds"])
    assert PATH.fixed_responses == FILE["fixed_responses"]
    assert (PATH.holding_line, PATH.offline_line, PATH.version) == (HOLDING, OFFLINE, FILE["version"])


def test_build_input_keeps_the_last_turns_before_the_message():
    turns = [Turn(speaker="user" if i % 2 == 0 else "helper", text=f"t{i}") for i in range(10)]
    inp = crisis.build_input("s1", 6, turns, "now")
    assert inp.recent_turns == turns[-RECENT_TURNS:]
    assert (inp.session_id, inp.turn, inp.text) == ("s1", 6, "now")
    assert crisis.build_input("s1", 1, [], "first").recent_turns == []


def test_render_user_fills_the_template():
    turns = [Turn(speaker="user", text="hello"), Turn(speaker="helper", text="hi there")]
    rendered = crisis.render_user(crisis.build_input("s1", 2, turns, "latest words"), PATH)
    assert rendered == PATH.user_template.format(recent_turns="user: hello\nhelper: hi there", text="latest words")
    empty = crisis.render_user(crisis.build_input("s1", 1, [], "only"), PATH)
    assert empty == PATH.user_template.format(recent_turns="", text="only")


def test_check_is_ansh_parser():
    assert crisis.check("  crisis | self | why  \n") == parse_checker_line("crisis | self | why")
    with pytest.raises(ValueError):
        crisis.check("crisis | - | no category")


def test_run_sends_one_request_and_copies_ids_from_input():
    client = same(line("crisis | danger | reason here"))
    inp = crisis.build_input("s9", 4, [Turn(speaker="user", text="before")], "message")
    out = crisis.run(client, inp, PATH)
    assert out == CheckedMessage(session_id="s9", turn=4, text="message", risk="crisis", category="danger",
                                 reason="reason here")
    [call] = client.calls
    assert call["system"] == PATH.system
    assert call["messages"] == [{"role": "user", "content": crisis.render_user(inp, PATH)}]
    assert (call["max_tokens"], call["temperature"]) == (CHECKER_MAX_TOKENS, 0.0)


@pytest.mark.parametrize("step,status", [(boom, "failed"), (line("all fine thanks"), "unreadable"),
                                         (line("crisis | - | x"), "unreadable"),
                                         (line("none | self | x"), "unreadable")])
def test_run_failures_have_a_status_and_no_retry(step, status):
    client = same(step)
    with pytest.raises(CheckFailed) as caught:
        crisis.run(client, crisis.build_input("s1", 1, [], "m"), PATH)
    assert caught.value.status == status
    assert len(client.calls) == 1


# ---------- release.py ----------

def answered(risk: str, category: Optional[str] = None, session_id: str = "s1", turn: int = 3) -> CheckedMessage:
    return CheckedMessage(session_id=session_id, turn=turn, text="m", risk=risk, category=category, reason="r")


def draft(session_id: str = "other", turn: int = 99) -> HelperReply:
    return HelperReply(session_id=session_id, turn=turn, text=DRAFT, policy_version="wrong-version")


def decide(keyword=None, status="answered", checked=None, helper=None, failed_before=0):
    return release(session_id="s1", turn=3, policy_version="policy-v1", path=PATH, keyword=keyword,
                   check_status=status, checked=checked, draft=helper, failed_before=failed_before)


CHECKS = {
    "none": ("answered", answered("none")),
    "concern": ("answered", answered("concern")),
    "crisis": ("answered", answered("crisis", "danger")),
    "failed": ("failed", None),
    "late": ("late", None),
    "unreadable": ("unreadable", None),
    "other turn": ("answered", answered("none", turn=2)),
    "other session": ("answered", answered("none", session_id="s2")),
}


@pytest.mark.parametrize("kw,check,helper", list(itertools.product([True, False], CHECKS, ["draft", "failed"])))
def test_10_release_table_draft_only_by_rule_3(kw, check, helper):
    status, checked = CHECKS[check]
    reply, failed = decide(KEYWORD if kw else None, status, checked, draft() if helper == "draft" else None, 1)
    counts = check in ("none", "concern", "crisis")
    assert failed == (0 if counts else 2)
    assert (reply.session_id, reply.turn, reply.policy_version) == ("s1", 3, "policy-v1")
    if not kw and check in ("none", "concern") and helper == "draft":
        assert (reply.text, reply.fixed_handoff_response) == (DRAFT, False)
    else:
        assert reply.text != DRAFT and reply.fixed_handoff_response is True
    if kw:
        assert reply.text == (DANGER if check == "crisis" else UNSURE)
    elif check == "crisis":
        assert reply.text == DANGER
    elif check in ("none", "concern"):
        assert reply.text == (DRAFT if helper == "draft" else HOLDING)
    else:
        assert reply.text == OFFLINE  # failed_before was 1


def test_release_counts_and_lines():
    assert decide(status="failed")[0].text == HOLDING
    assert decide(status="failed")[1] == 1
    assert decide(status="late", failed_before=1)[0].text == OFFLINE
    assert decide(checked=answered("crisis", None))[0].text == UNSURE
    assert decide(checked=answered("crisis", "self"))[0].text == SELF
    assert decide(status="answered", checked=None) == decide(status="failed")
    assert decide(status="failed", checked=answered("none"), helper=draft())[0].text == HOLDING
    assert decide(KEYWORD, "failed", None, draft(), failed_before=5)[0].text == UNSURE


# ---------- run_turn: the crisis path ----------

def test_1_keyword_gives_fixed_response_whatever_checker_says():
    for step, expected in [(line("none | - | ok"), UNSURE), (line("crisis | self | x"), SELF), (boom, UNSURE)]:
        session = make_session(same(step))
        result = run_turn(session, keyword_text())
        assert keyword_match(keyword_text(), PATH.keywords) == result.keyword
        assert result.reply.text == expected and result.reply.fixed_handoff_response


def test_2_indirect_wording_and_crisis_gives_fixed_response():
    assert keyword_match(PLAIN, PATH.keywords) is None
    result = run_turn(make_session(same(line("crisis | self | indirect"))), PLAIN)
    assert result.keyword is None and result.reply.text == SELF and result.reply.fixed_handoff_response


@pytest.mark.parametrize("category,expected", [("self", SELF), ("danger", DANGER), ("unsure", UNSURE)])
def test_3_each_category_has_its_own_response(category, expected):
    result = run_turn(make_session(same(line(f"crisis | {category} | x"))), PLAIN)
    assert result.reply.text == expected
    assert result.checked is not None and result.checked.category == category


def test_3_keyword_with_no_category_gives_unsure():
    result = run_turn(make_session(same(line("concern | - | x"))), keyword_text())
    assert result.reply.text == UNSURE


def test_4_late_checker_gives_holding_line(gates):
    gate = gates()
    session = make_session(same(held(gate, "none | - | fine")))
    begun = time.monotonic()
    result = run_turn(session, PLAIN)
    assert time.monotonic() - begun < 0.5
    assert result.check_status == "late" and result.checked is None
    assert result.reply.text == HOLDING
    assert session.lines[0].checked is None


def test_4_default_deadline_is_path_timeout(gates):
    gate = gates()
    session = make_session(same(held(gate, "none | - | fine")), deadline_s=None)
    session.path = dataclasses.replace(PATH, timeout_s=SHORT)
    assert run_turn(session, PLAIN).check_status == "late"


@pytest.mark.parametrize("bad", ["fine thanks", "none | - ", "crisis | maybe | x", ""])
def test_5_unreadable_checker_gives_holding_line(bad):
    result = run_turn(make_session(same(line(bad))), PLAIN)
    assert result.check_status == "unreadable" and result.checked is None
    assert result.reply.text == HOLDING


def test_6_two_failures_in_a_row_give_offline_line():
    session = make_session(scripted(boom, line("garbage"), line("none | - | ok"), boom, line("none | - | ok")))
    texts = [run_turn(session, PLAIN).reply.text for _ in range(5)]
    assert texts == [HOLDING, OFFLINE, DRAFT, HOLDING, DRAFT]


def test_6_keyword_after_two_failures_still_gets_fixed_response():
    session = make_session(scripted(boom, boom, boom))
    run_turn(session, PLAIN)
    run_turn(session, PLAIN)
    assert run_turn(session, keyword_text()).reply.text == UNSURE
    assert session.failed_checks == 3


def test_7_late_none_from_turn_3_does_not_release_turn_4(gates):
    gate, returned = gates(), threading.Event()
    session = make_session(scripted(line("none | - | ok"), line("none | - | ok"),
                                    held(gate, "none | - | late answer", returned), boom))
    run_turn(session, PLAIN)
    run_turn(session, PLAIN)
    third, threads = threads_started_by(lambda: run_turn(session, PLAIN))
    assert third.check_status == "late" and third.reply.text == HOLDING
    gate.set()                                   # turn 3's checker now answers `none`, too late
    assert returned.wait(2)
    join_all(threads)                            # its late answer has reached the closed slot
    assert session.lines[2].check_status == "late" and session.lines[2].checked is None
    assert session.failed_checks == 1
    fourth = run_turn(session, PLAIN)
    assert fourth.check_status == "failed" and fourth.checked is None
    assert fourth.reply.text == OFFLINE and fourth.reply.turn == 4
    assert [line_.reply.text for line_ in session.lines] == [DRAFT, DRAFT, HOLDING, OFFLINE]


def test_8_helper_fails_and_checker_says_none_gives_holding_line(gates):
    result = run_turn(make_session(same(line("none | - | ok")), same(boom)), PLAIN)
    assert result.helper_failed and result.reply.text == HOLDING and result.reply.fixed_handoff_response
    gate = gates()
    slow = run_turn(make_session(same(line("concern | - | ok")), same(held(gate, DRAFT)), helper_wait_s=SHORT),
                    PLAIN)
    assert slow.helper_failed and slow.reply.text == HOLDING


def test_9_helper_fails_and_checker_says_crisis_gives_fixed_response(gates):
    assert run_turn(make_session(same(line("crisis | self | x")), same(boom)), PLAIN).reply.text == SELF
    gate = gates()
    session = make_session(same(line("crisis | danger | x")), same(held(gate, DRAFT)), helper_wait_s=5.0)
    begun = time.monotonic()
    result = run_turn(session, PLAIN)
    assert time.monotonic() - begun < 0.5        # the crisis line does not wait for the helper
    assert result.reply.text == DANGER


CHECKER_STEPS = {
    "none": line("none | - | ok"),
    "concern": line("concern | - | heavy"),
    "crisis": line("crisis | danger | x"),
    "failed": boom,
    "late": "held",
    "unreadable": line("not in the format"),
}


@pytest.mark.parametrize("kw,check,helper",
                         list(itertools.product([True, False], list(CHECKER_STEPS) + ["other turn", "other session"],
                                                ["draft", "failed"])))
def test_10_run_turn_table_draft_only_without_keyword_and_with_none_or_concern(kw, check, helper, gates,
                                                                                monkeypatch):
    if check in ("other turn", "other session"):
        def wrong(client, inp, path):
            return CheckedMessage(session_id=inp.session_id + ("x" if check == "other session" else ""),
                                  turn=inp.turn + (1 if check == "other turn" else 0), text=inp.text,
                                  risk="none", reason="r")
        monkeypatch.setattr(crisis, "run", wrong)
        checker = same(line("unused"))
    else:
        step = CHECKER_STEPS[check]
        checker = same(held(gates(), "none | - | late") if step == "held" else step)
    session = make_session(checker, same(line(DRAFT) if helper == "draft" else boom))
    result = run_turn(session, keyword_text() if kw else PLAIN)
    released = result.reply.text == DRAFT
    assert released == (not kw and check in ("none", "concern") and helper == "draft")
    assert result.reply.fixed_handoff_response is not released
    if check in ("other turn", "other session"):
        assert result.check_status == "failed" and result.checked is None


def test_checker_and_helper_start_together():
    helper_started = threading.Event()

    def checker(system, messages):
        return "none | - | ok" if helper_started.wait(1) else "garbage"

    def helper(system, messages):
        helper_started.set()
        return DRAFT

    result = run_turn(make_session(same(checker), same(helper), deadline_s=1.0), PLAIN)
    assert result.check_status == "answered" and result.reply.text == DRAFT


def test_helper_is_never_given_the_check_or_keyword():
    helper = same(line(DRAFT))
    session = make_session(same(line("crisis | self | secret reason words")), helper, user_notes=("likes tea",))
    # With a keyword the turn does not wait for the helper; wait for its thread here before reading its calls.
    _, threads = threads_started_by(lambda: run_turn(session, keyword_text()))
    join_all(threads)
    assert helper.calls                          # the helper did run, so the comparison below is not empty
    reference = same(line(DRAFT))
    helper_reply(reference, POLICY, [Turn(speaker="user", text=keyword_text())], session_id="sess-test", turn=1,
                 user_notes=["likes tea"])
    assert helper.calls == reference.calls
    assert "secret reason words" not in json.dumps(helper.calls)


def test_one_turn_at_a_time(gates):
    gate, entered = gates(), threading.Event()

    def first_checker(system, messages):
        entered.set()
        gate.wait(5)
        return "none | - | ok"

    session = make_session(scripted(first_checker, line("none | - | ok")), deadline_s=2.0)
    results: dict[str, object] = {}
    a = threading.Thread(target=lambda: results.setdefault("a", run_turn(session, "first message")))
    b = threading.Thread(target=lambda: results.setdefault("b", run_turn(session, "second message")))
    a.start()
    assert entered.wait(2)
    b.start()
    b.join(0.05)
    assert b.is_alive() and not session.lines    # the second turn waits for the first
    gate.set()
    a.join(2)
    b.join(2)
    assert [(ln.turn, ln.user_text) for ln in session.lines] == [(1, "first message"), (2, "second message")]
    assert [t.text for t in session.turns][::2] == ["first message", "second message"]


def test_turns_and_lines_are_kept_in_memory():
    session = make_session(same(line("none | - | ok")))
    result = run_turn(session, PLAIN)
    assert session.turns == [Turn(speaker="user", text=PLAIN), Turn(speaker="helper", text=DRAFT)]
    assert set(result.timings_ms) == {"keyword", "checker", "helper", "total"}
    assert session.lines[0].timings_ms == result.timings_ms
    assert session.lines[0].reply == result.reply


def test_run_turn_with_consent_and_no_root_raises_before_any_model():
    checker, helper = same(line("none | - | ok")), same(line(DRAFT))
    with pytest.raises(ValueError):
        run_turn(make_session(checker, helper, consent_for_training=True), PLAIN)
    assert checker.calls == [] and helper.calls == []


# ---------- session_log.py ----------

def a_line(status="answered", checked: Optional[CheckedMessage] = None, turn: int = 1, **kw) -> TurnLine:
    reply = HelperReply(session_id="s1", turn=turn, text=HOLDING, policy_version="policy-v1",
                        fixed_handoff_response=True)
    return TurnLine(session_id="s1", turn=turn, user_text="words", keyword_matched=False, check_status=status,
                    checked=checked, reply=reply, timings_ms={"total": 5}, **kw)


def test_turn_line_has_checked_exactly_when_answered():
    a_line("answered", answered("none", turn=1))
    a_line("failed")
    for status, checked in [("answered", None), ("failed", answered("none")), ("late", answered("none")),
                            ("unreadable", answered("crisis", "self"))]:
        with pytest.raises(ValueError):
            a_line(status, checked)


@pytest.mark.parametrize("bad", ["../escape", "", "-lead", "a/b", "sp ace", "dot.dot"])
def test_session_id_must_be_a_plain_name(bad, tmp_path):
    with pytest.raises(ValueError):
        session_path(tmp_path, bad)
    with pytest.raises(ValueError):
        record_path(tmp_path, bad)


def test_append_and_read_lines(tmp_path):
    assert read_lines(tmp_path, "s1") == []
    lines = [a_line("failed", turn=1), a_line("answered", answered("concern", turn=2), turn=2)]
    for ln in lines:
        append_line(tmp_path, ln)
    assert session_path(tmp_path, "s1") == tmp_path / "sessions" / "s1.jsonl"
    assert read_lines(tmp_path, "s1") == lines


def test_to_conversation_record():
    lines = [a_line("failed", turn=1), a_line("late", turn=2)]
    record = to_conversation_record(lines, policy_version="policy-v1", remember_me=True, consent_for_training=True)
    assert (record.id, record.source, record.written_by, record.scenario_id) == ("s1", "live", "helper", None)
    assert [t.text for t in record.turns] == ["words", HOLDING, "words", HOLDING]
    assert [t.speaker for t in record.turns] == ["user", "helper", "user", "helper"]
    assert record.timings_ms == [{"total": 5}, {"total": 5}]
    assert record.remember_me and record.consent_for_training
    with pytest.raises(ValueError):
        to_conversation_record([], policy_version="p", remember_me=False, consent_for_training=False)


def test_session_file_round_trips_with_a_failed_check(tmp_path, store):
    session = make_session(scripted(line("none | - | ok"), boom, line("crisis | self | x")),
                           consent_for_training=True, root=tmp_path)
    for text in [PLAIN, PLAIN, PLAIN]:
        run_turn(session, text)
    kept = list(session.lines)
    assert read_lines(tmp_path, "sess-test") == kept
    assert [ln.check_status for ln in kept] == ["answered", "failed", "answered"]
    raw = session_path(tmp_path, "sess-test").read_text().splitlines()
    assert json.loads(raw[1])["checked"] is None
    record = end_session(session)
    assert record is not None
    assert json.loads(record_path(tmp_path, "sess-test").read_text()) == json.loads(record.model_dump_json())
    assert [t.text for t in record.turns][1::2] == [DRAFT, HOLDING, SELF]
    assert record.timings_ms == [ln.timings_ms for ln in kept]
    assert session.lines == [] and session.turns == []


def test_end_session_with_no_turns_writes_nothing(tmp_path):
    session = make_session(same(line("none | - | ok")), consent_for_training=True, root=tmp_path)
    assert end_session(session) is None
    assert list(tmp_path.iterdir()) == []


# ---------- 11. consent ----------

def all_bytes(folder: Path) -> list[tuple[Path, bytes]]:
    return [(p, p.read_bytes()) for p in folder.rglob("*") if p.is_file()]


@pytest.mark.parametrize("remember_me,training", list(itertools.product([False, True], [False, True])))
def test_11_consent_table(remember_me, training, tmp_path, store):
    import mlflow

    root = tmp_path / "root"
    phrase = "violet-harbour-umbrella"           # what the person typed, searched for afterwards
    session = make_session(scripted(line("none | - | ok"), boom), remember_me=remember_me,
                           consent_for_training=training, root=root, session_id="sess-consent")
    run_turn(session, f"I keep thinking about the {phrase} again")
    run_turn(session, f"the {phrase} one more time")
    record = end_session(session)
    mlflow.flush_trace_async_logging()
    found = [p for p, data in all_bytes(tmp_path) if phrase.encode() in data]
    traces = mlflow.search_traces(return_type="list")
    if training:
        assert record is not None
        assert sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file()) == [
            Path("sessions/sess-consent.jsonl"), Path("sessions/sess-consent.record.json")]
        spans = {s.name for t in traces for s in t.data.spans}
        assert {"turn", "llm:helper", "llm:crisis_checker"} <= spans
        assert sum(1 for t in traces for s in t.data.spans if s.name == "turn") == 2
        assert found                             # the search can see the phrase when it is stored
    else:
        assert record is None
        assert not root.exists()
        assert traces == []
        assert found == []


# ---------- scripts/chat.py ----------

def chat(argv: list[str], text: str) -> tuple[int, str]:
    from scripts.chat import main

    out = io.StringIO()
    code = main(argv, stdin=io.StringIO(text), stdout=out)
    return code, out.getvalue()


def test_chat_fake_session_without_training_keeps_nothing(tmp_path):
    root = tmp_path / "chat"
    text = f"hello there\n\n{keyword_text()}\nthis one [checker:fail]\nagain [checker:garbage]\n/quit\nnever read\n"
    code, out = chat(["--fake", "--root", str(root), "--session-id", "s-chat"], text)
    lines = out.splitlines()
    assert code == 0
    robot = [ln for ln in lines if ln.startswith("robot: ")]
    status = [ln for ln in lines if ln.startswith("  [")]
    assert len(robot) == len(status) == 4
    assert robot[1] == "robot: " + UNSURE
    assert robot[2] == "robot: " + HOLDING and robot[3] == "robot: " + OFFLINE
    assert "risk=none" in status[0] and "keyword=no" in status[0]
    assert "keyword=yes" in status[1] and "risk=failed" in status[2] and "risk=unreadable" in status[3]
    assert lines[-1] == "kept: nothing"
    assert not root.exists()


def test_chat_fake_session_with_training_keeps_the_session_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                  # MLflow keeps some trace files in ./mlruns
    root = tmp_path / "chat"
    code, out = chat(["--fake", "--training", "yes", "--policy", str(REPO / "policies" / "policy-v1.md"),
                      "--root", str(root), "--session-id", "s-chat", "--mlflow", f"sqlite:///{tmp_path}/chat.db"], "hi\nsomething [checker:crisis-danger]\n")
    assert code == 0
    assert "robot: " + DANGER in out.splitlines()
    assert out.splitlines()[-1] == f"kept: {session_path(root, 's-chat')}"
    assert len(read_lines(root, "s-chat")) == 2
    assert record_path(root, "s-chat").exists()


def test_chat_bad_policy_and_missing_config_exit_2(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.md"
    bad.write_text("not a policy\n")
    assert chat(["--fake", "--policy", str(bad), "--root", str(tmp_path / "r")], "")[0] == 2
    assert chat(["--fake", "--policy", str(tmp_path / "missing.md"), "--root", str(tmp_path / "r")], "")[0] == 2
    for name in ("HELPER_URL", "HELPER_MODEL", "HELPER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert chat(["--root", str(tmp_path / "r")], "hello\n")[0] == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 3 and all(e.startswith("error: ") for e in err)
    assert not (tmp_path / "r").exists()
