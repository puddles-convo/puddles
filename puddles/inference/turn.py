"""One turn of a live session: the keyword check, the checker and the helper, then the release step.

The checker and the helper start together on their own threads. Each thread hands its answer to a
per-turn slot that the turn closes once it stops waiting, so an answer that comes later, even `none`,
lands nowhere: it cannot change the turn already released or any later turn.

With `consent_for_training` false nothing the person typed is written to disk or traced.
"""
from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import loop.policy
from loop.helper import helper_reply
from loop.llm import ChatClient
from puddles.eval.crisis import keyword_match
from puddles.inference import crisis
from puddles.inference.crisis import CheckFailed, CheckStatus, CrisisPath
from puddles.inference.release import release
from puddles.inference.session_log import (TurnLine, append_line, read_lines, record_path,
                                           to_conversation_record)
from schemas import CheckedMessage, ConversationRecord, HelperReply, Turn

NOT_MEASURED_MS = -1  # a step that had not answered when the turn stopped waiting for it


@dataclass
class Session:
    session_id: str
    policy: loop.policy.Policy
    helper_client: ChatClient
    checker_client: ChatClient
    path: CrisisPath
    user_id: Optional[str] = None
    remember_me: bool = False
    consent_for_training: bool = False
    user_notes: Sequence[str] = ()
    root: Optional[Path] = None           # where sessions/ goes
    deadline_s: Optional[float] = None    # None: path.timeout_s
    helper_wait_s: float = 30.0
    turns: list[Turn] = field(default_factory=list)         # what was said so far
    lines: list[TurnLine] = field(default_factory=list)     # one per turn, in memory
    failed_checks: int = 0                # failed checks in a row
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)


@dataclass
class TurnResult:
    reply: HelperReply
    keyword: Optional[str]
    check_status: CheckStatus
    checked: Optional[CheckedMessage]
    helper_failed: bool
    timings_ms: dict[str, int]            # keys: keyword, checker, helper, total


class _Job:
    """One model call on its own thread, with a slot that is closed once the turn stops waiting."""

    def __init__(self, fn: Callable[[], Any]) -> None:
        self._fn = fn
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._closed = False
        self.value: Any = None
        self.error: Optional[BaseException] = None
        self.started = 0.0
        self.finished = 0.0

    def start(self) -> None:
        self.started = time.monotonic()
        ctx = contextvars.copy_context()  # so a traced call nests under the turn's span
        threading.Thread(target=ctx.run, args=(self._work,), daemon=True).start()

    def _work(self) -> None:
        try:
            value, error = self._fn(), None
        except BaseException as exc:  # a crash in the worker is a failed call, never a hang
            value, error = None, exc
        finished = time.monotonic()
        with self._lock:
            if self._closed:
                return  # too late: the turn has moved on and nothing may change it
            self.value, self.error, self.finished = value, error, finished
            self._done.set()

    def close(self, until: Optional[float] = None) -> bool:
        """Wait until `until` (monotonic), close the slot, and say whether an answer came in time."""
        if until is not None:
            self._done.wait(max(0.0, until - time.monotonic()))
        with self._lock:
            self._closed = True
            if not self._done.is_set():
                return False
            return until is None or self.finished <= until


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def run_turn(session: Session, text: str) -> TurnResult:
    """Run one turn and return the reply. A failed model never raises; it gives a line from the path."""
    if session.consent_for_training and session.root is None:
        raise ValueError("consent_for_training is on but the session has no root to save it under")
    with session._lock:
        if not session.consent_for_training:
            return _turn(session, text, session.helper_client, session.checker_client)
        import mlflow

        from puddles.tracing import TracedClient

        with mlflow.start_span(name="turn") as span:
            span.set_attributes({"puddles.session_id": session.session_id, "puddles.turn": len(session.lines) + 1})
            result = _turn(session, text, TracedClient(session.helper_client, "helper"),
                           TracedClient(session.checker_client, "crisis_checker"))
            span.set_outputs({"check_status": result.check_status,
                              "fixed_handoff_response": result.reply.fixed_handoff_response})
            return result


def _turn(session: Session, text: str, helper_client: ChatClient, checker_client: ChatClient) -> TurnResult:
    started = time.monotonic()
    turn = len(session.lines) + 1
    path, sid, policy = session.path, session.session_id, session.policy
    keyword = keyword_match(text, path.keywords)
    keyword_ms = _ms(time.monotonic() - started)

    history = list(session.turns)
    person = Turn(speaker="user", text=text)
    helper_turns = history + [person]
    notes = list(session.user_notes)
    inp = crisis.build_input(sid, turn, history, text)

    # The helper is given the conversation only: never the keyword, the risk level or the check.
    checker = _Job(lambda: crisis.run(checker_client, inp, path))
    helper = _Job(lambda: helper_reply(helper_client, policy, helper_turns, session_id=sid, turn=turn,
                                       user_notes=notes))
    checker.start()
    helper.start()

    deadline = path.timeout_s if session.deadline_s is None else session.deadline_s
    checker_until = checker.started + deadline
    status: CheckStatus
    checked: Optional[CheckedMessage] = None
    if not checker.close(checker_until):
        status = "late"
        checker_ms = _ms(time.monotonic() - checker.started)
    else:
        checker_ms = _ms(checker.finished - checker.started)
        if isinstance(checker.error, CheckFailed):
            status = checker.error.status
        elif checker.error is not None:
            status = "failed"
        elif (isinstance(checker.value, CheckedMessage) and checker.value.session_id == sid
              and checker.value.turn == turn):
            status, checked = "answered", checker.value
        else:
            status = "failed"  # an answer for another session or turn is no answer for this one

    draft: Optional[HelperReply] = None
    usable = keyword is None and checked is not None and checked.risk in ("none", "concern")
    if usable:
        in_time = helper.close(started + session.helper_wait_s)
        if in_time and helper.error is None and isinstance(helper.value, HelperReply):
            draft = helper.value
        helper_failed = draft is None
    else:
        in_time = helper.close()  # no wait: a fixed line is spoken whether or not the helper answered
        helper_failed = in_time and (helper.error is not None or not isinstance(helper.value, HelperReply))
    helper_ms = _ms(helper.finished - helper.started) if in_time else NOT_MEASURED_MS

    reply, session.failed_checks = release(session_id=sid, turn=turn, policy_version=policy.version, path=path,
                                           keyword=keyword, check_status=status, checked=checked, draft=draft,
                                           failed_before=session.failed_checks)
    timings = {"keyword": keyword_ms, "checker": checker_ms, "helper": helper_ms,
               "total": _ms(time.monotonic() - started)}
    line = TurnLine(session_id=sid, turn=turn, user_text=text, keyword_matched=keyword is not None,
                    check_status=status, checked=checked, reply=reply, timings_ms=timings)
    session.lines.append(line)
    session.turns.extend([person, Turn(speaker="helper", text=reply.text)])
    if session.consent_for_training:
        append_line(session.root, line)  # type: ignore[arg-type]  # checked in run_turn
    return TurnResult(reply=reply, keyword=keyword, check_status=status, checked=checked,
                      helper_failed=helper_failed, timings_ms=timings)


def end_session(session: Session, now: datetime | None = None) -> Optional[ConversationRecord]:
    """With consent, write and return the record built from the session file. Then forget the session."""
    with session._lock:
        try:
            if not session.lines or not session.consent_for_training or session.root is None:
                return None
            lines = read_lines(session.root, session.session_id)
            record = to_conversation_record(lines, policy_version=session.policy.version,
                                            remember_me=session.remember_me,
                                            consent_for_training=session.consent_for_training, now=now)
            out = record_path(session.root, session.session_id)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
            return record
        finally:
            session.turns.clear()
            session.lines.clear()
