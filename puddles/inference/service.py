"""The turn service: the checked text path over HTTP, for the robot's brain. Contract: docs/specs/turn-service.md.

A FastAPI app that starts a session, runs one turn at a time through `run_turn`, and ends a session the way
`scripts/chat.py` does, with the note-taker. Every route but `GET /health` needs the bearer token. No
response, log line or printed line holds the person's text, a reply's text, a note's text, the token or an
exception's message: errors carry a fixed code, and logs carry IDs, routes, statuses, counts and type names.

The service keeps one registry of live sessions behind one lock, and that lock is never held across a model
call. Each session has its own lock that orders its turns and its end, so a slow turn in one session never
delays another session.
"""
from __future__ import annotations

import hmac
import logging
import math
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import fastapi
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import BaseRoute, Match
from starlette.types import ASGIApp, Receive, Scope, Send

from loop.errors import PolicyError
from loop.llm import ChatClient
from puddles.graph import check_policy_file
from puddles.inference import crisis
from puddles.inference.crisis import CrisisPath
from puddles.inference.live import finish_session, start_notes
from puddles.inference.note_store import NoteStore, check_user_id
from puddles.inference.notes import NotesResult
from puddles.inference.session_log import record_path, session_path
from puddles.inference.turn import Session, end_session, run_turn
from puddles.llm import LLM
from schemas import ConversationRecord

MAX_TEXT_CHARS = 2000
MAX_SESSION_ID_CHARS = 64
BEARER = "Bearer "
HEALTH = "/health"
_FIELD_NAME = re.compile(r"[A-Za-z0-9_]{1,64}")  # an unknown key is named in `fields` only if it fits this

log = logging.getLogger("puddles.inference.service")


@dataclass
class ServiceConfig:
    token: str                                   # the bearer token every request must carry
    policy_path: Path                            # read again at the start of every session
    root: Path                                   # where sessions/ goes
    store: NoteStore
    client_factory: Callable[[], ChatClient]     # called twice per session: helper, then checker
    note_llm_for: Callable[[bool], LLM]          # consent_for_training -> the note-taker's model
    path: Optional[CrisisPath] = None            # None: crisis.load_path(), once, in create_app
    mlflow_uri: Optional[str] = None
    idle_timeout_s: float = 900.0
    max_sessions: int = 64
    deadline_s: Optional[float] = None           # passed to Session
    helper_wait_s: float = 30.0                  # passed to Session
    clock: Callable[[], float] = time.monotonic


class Refused(Exception):
    """A response this contract names: a status and a body that holds only fixed codes and IDs."""

    def __init__(self, status: int, body: dict[str, Any]) -> None:
        super().__init__(status)
        self.status = status
        self.body = body


# ---------- request bodies ----------

class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StartBody(_Body):
    user_id: StrictStr
    remember_me: StrictBool
    consent_for_training: StrictBool
    session_id: Optional[StrictStr] = None

    @field_validator("user_id")
    @classmethod
    def _user_id(cls, value: str) -> str:
        return check_user_id(value)

    @field_validator("session_id")
    @classmethod
    def _session_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if len(value) > MAX_SESSION_ID_CHARS:
            raise ValueError("session_id is too long")
        session_path(Path("."), value)  # the check the session file's path makes
        return value


class TurnBody(_Body):
    text: StrictStr

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        value = value.strip()
        if not 1 <= len(value) <= MAX_TEXT_CHARS:
            raise ValueError("text must be 1 to 2000 characters once trimmed")
        return value


# ---------- the live sessions ----------

@dataclass
class _Live:
    session_id: str
    user_id: str
    session: Optional[Session] = None            # None while the session is still being started
    note_llm: Optional[LLM] = None
    last_activity: float = 0.0
    running: int = 0                             # turns admitted and not yet finished
    ending: bool = False
    turn_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _Ended:
    record: Optional[ConversationRecord]
    notes: NotesResult


def _new_session_id() -> str:
    return f"sess-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"


def _absent(path: Path) -> bool:
    """True only when the look says the file is not there; any other error looking is raised."""
    try:
        os.stat(path)
    except FileNotFoundError:
        return True
    return False


class TurnService:
    """The registry of live sessions and what each route does with it."""

    def __init__(self, config: ServiceConfig, path: CrisisPath) -> None:
        self.config = config
        self.path = path
        self._cond = threading.Condition(threading.Lock())  # guards _entries; never held across a model call
        self._entries: dict[str, _Live] = {}
        self._factory_lock = threading.Lock()  # keeps a session's two clients a pair: helper, then checker

    def live(self) -> list[str]:
        """IDs of the sessions held now, sorted. A session still being started is not held yet."""
        with self._cond:
            return sorted(sid for sid, e in self._entries.items() if e.session is not None)

    def sweep(self) -> list[str]:
        """End every idle session, exactly as the end route would; returns their IDs, sorted. Raises nothing."""
        try:
            with self._cond:
                now = self.config.clock()
                idle = [e for e in self._entries.values()
                        if e.session is not None and not e.ending and e.running == 0
                        and now - e.last_activity >= self.config.idle_timeout_s]
                for e in idle:
                    e.ending = True
            for e in idle:
                try:
                    self._finish(e)
                except Exception as exc:
                    log.warning("sweep: session %s did not end cleanly (%s)", e.session_id, type(exc).__name__)
            if idle:
                log.info("sweep: %d idle sessions ended", len(idle))
            return sorted(e.session_id for e in idle)
        except Exception as exc:  # the timer thread must keep going
            log.warning("sweep failed (%s)", type(exc).__name__)
            return []

    # ---------- start ----------

    def start(self, body: StartBody) -> dict[str, Any]:
        cfg = self.config
        sid = body.session_id or _new_session_id()
        consent = body.consent_for_training
        with self._cond:
            if sid in self._entries:
                raise Refused(409, {"error": "session_exists"})
            if consent and not (_absent(session_path(cfg.root, sid)) and _absent(record_path(cfg.root, sid))):
                raise Refused(409, {"error": "session_exists"})
            if any(e.user_id == body.user_id for e in self._entries.values()):
                raise Refused(409, {"error": "user_busy"})
            if len(self._entries) >= cfg.max_sessions:
                raise Refused(429, {"error": "too_many_sessions"})
            entry = _Live(session_id=sid, user_id=body.user_id)
            self._entries[sid] = entry  # held for this ID and person while the rest is built
        try:
            session, note_llm, notes_failed = self._build(sid, body)
        except BaseException:
            with self._cond:
                self._entries.pop(sid, None)
            raise
        with self._cond:
            entry.session, entry.note_llm = session, note_llm
            entry.last_activity = cfg.clock()
        log.info("session %s started with %d notes", sid, len(session.user_notes))
        return {"session_id": sid, "policy_version": session.policy.version,
                "notes_loaded": len(session.user_notes), "notes_failed": notes_failed}

    def _build(self, sid: str, body: StartBody) -> tuple[Session, LLM, bool]:
        cfg = self.config
        try:
            policy = check_policy_file(cfg.policy_path)
        except (PolicyError, OSError, UnicodeDecodeError) as exc:
            log.warning("session %s: policy unavailable (%s)", sid, type(exc).__name__)
            raise Refused(503, {"error": "policy_unavailable"}) from None
        try:
            with self._factory_lock:
                helper_client = cfg.client_factory()
                checker_client = cfg.client_factory()
            note_llm = cfg.note_llm_for(body.consent_for_training)
        except Exception as exc:
            log.warning("session %s: model unavailable (%s)", sid, type(exc).__name__)
            raise Refused(503, {"error": "model_unavailable"}) from None
        if body.consent_for_training:
            from puddles import tracing  # imports MLflow, so only for a session with consent

            tracing.setup(tracking_uri=cfg.mlflow_uri)
        notes_failed = False
        try:
            user_notes = start_notes(cfg.store, body.user_id)
        except Exception as exc:
            log.warning("session %s: notes not loaded (%s)", sid, type(exc).__name__)
            user_notes, notes_failed = [], True
        session = Session(session_id=sid, policy=policy, helper_client=helper_client, checker_client=checker_client,
                          path=self.path, user_id=body.user_id, remember_me=body.remember_me,
                          consent_for_training=body.consent_for_training, user_notes=user_notes, root=cfg.root,
                          deadline_s=cfg.deadline_s, helper_wait_s=cfg.helper_wait_s)
        return session, note_llm, notes_failed

    # ---------- one turn ----------

    def turn(self, sid: str, body: TurnBody) -> dict[str, Any]:
        with self._cond:
            entry = self._entries.get(sid)
            if entry is None or entry.session is None or entry.ending:
                raise Refused(404, {"error": "session_not_found"})
            entry.running += 1
            entry.last_activity = self.config.clock()
        session = entry.session
        try:
            with entry.turn_lock:
                try:
                    result = run_turn(session, body.text)
                except Exception as exc:
                    log.warning("session %s: turn failed (%s)", sid, type(exc).__name__)
                    raise Refused(500, {"error": "turn_failed", "say": self.path.offline_line}) from None
        finally:
            with self._cond:
                entry.running -= 1
                entry.last_activity = self.config.clock()
                self._cond.notify_all()
        answered = result.check_status == "answered" and result.checked is not None
        reply = result.reply
        return {"session_id": sid, "turn": reply.turn, "text": reply.text,
                "fixed_handoff_response": reply.fixed_handoff_response, "policy_version": reply.policy_version,
                "check_status": result.check_status,
                "risk": result.checked.risk if answered else None,  # type: ignore[union-attr]
                "category": result.checked.category if answered else None,  # type: ignore[union-attr]
                "keyword_matched": result.keyword is not None, "helper_failed": result.helper_failed,
                "timings_ms": dict(result.timings_ms)}

    # ---------- end ----------

    def end(self, sid: str) -> dict[str, Any]:
        with self._cond:
            entry = self._entries.get(sid)
            if entry is None or entry.session is None or entry.ending:
                raise Refused(404, {"error": "session_not_found"})
            entry.ending = True  # from here a new turn is refused and reaches no model
        ended = self._finish(entry)
        notes = ended.notes
        record = ended.record.model_dump(mode="json") if ended.record is not None else None
        log.info("session %s ended; kept=%s notes_failed=%s", sid, record is not None, notes.failed)
        return {"session_id": sid, "kept": record is not None, "record": record,
                "notes": {"added": len(notes.added), "replaced": len(notes.replaced),
                          "confirmed": len(notes.confirmed), "deleted": len(notes.deleted),
                          "undeleted": list(notes.undeleted), "failed": notes.failed}}

    def _finish(self, entry: _Live) -> _Ended:
        """Wait for a running turn, then end the session as scripts/chat.py does. It is removed whatever happens."""
        try:
            with self._cond:
                self._cond.wait_for(lambda: entry.running == 0)
            session, note_llm = entry.session, entry.note_llm
            assert session is not None and note_llm is not None
            with entry.turn_lock:
                try:
                    record, notes = finish_session(session, self.config.store, note_llm)
                except Exception as exc:
                    log.warning("session %s: notes step raised (%s)", entry.session_id, type(exc).__name__)
                    record, notes = end_session(session), NotesResult(failed=True)
            return _Ended(record=record, notes=notes)
        finally:
            with self._cond:
                self._entries.pop(entry.session_id, None)


# ---------- the app ----------

def _authorized(request: Request, token: str) -> bool:
    """Exactly one Authorization header whose raw bytes are `Bearer <token>` in UTF-8. Never raises."""
    try:
        values = [value for name, value in request.headers.raw if name.lower() == b"authorization"]
        expected = token.encode("utf-8")
    except Exception:  # a header or token that cannot be compared is a refusal, never an error
        return False
    prefix = BEARER.encode("ascii")
    if len(values) != 1 or not values[0].startswith(prefix):
        return False
    return hmac.compare_digest(values[0][len(prefix):], expected)


class _TokenGate:
    """Pure ASGI: a request for any route but GET /health needs the token before anything reads its body.

    The router's own routes decide what a route is, so a path that is no route still gets 404 and a wrong
    method 405 from the app. A refused request is answered here, and its body is never received.
    """

    def __init__(self, app: ASGIApp, *, routes: list[BaseRoute], token: str) -> None:
        self.app = app
        self.routes = routes  # the router's own list, which the app's routes are added to afterwards
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self._needs_token(scope) and not _authorized(Request(scope), self.token):
            await _error(401, {"error": "unauthorized"})(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _needs_token(self, scope: Scope) -> bool:
        for route in self.routes:
            match, _ = route.matches(scope)
            if match == Match.FULL:
                return not (isinstance(route, APIRoute) and route.path == HEALTH)
        return False


def _error(status: int, body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status, content=body)


def _fields(exc: RequestValidationError) -> list[str]:
    """The top-level field names at fault, sorted; never a value that was sent."""
    names: set[str] = set()
    for err in exc.errors():
        loc = err.get("loc") or ()
        if err.get("type") == "json_invalid" or len(loc) < 2 or loc[0] != "body":
            continue
        name = loc[1]
        if isinstance(name, str) and _FIELD_NAME.fullmatch(name):
            names.add(name)
    return sorted(names)


def _check_config(config: ServiceConfig) -> None:
    if not isinstance(config.token, str) or not config.token.strip():
        raise ValueError("token must be a non-empty string")
    m = config.max_sessions
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError("max_sessions must be an integer of at least 1")
    t = config.idle_timeout_s
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
        raise ValueError("idle_timeout_s must be a finite number above zero")


def create_app(config: ServiceConfig) -> fastapi.FastAPI:
    """Build the app. A bad config or a crisis path that cannot be loaded raises here, so nothing is served."""
    _check_config(config)
    path = config.path if config.path is not None else crisis.load_path()
    service = TurnService(config, path)
    app = fastapi.FastAPI(title="puddles turn service", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service

    app.add_middleware(_TokenGate, routes=app.router.routes, token=config.token)

    @app.exception_handler(Refused)
    async def _refused(request: Request, exc: Refused) -> JSONResponse:
        return _error(exc.status, exc.body)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, {"error": "invalid_request", "fields": _fields(exc)})

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error(exc.status_code, {"error": codes.get(exc.status_code, "internal")})

    def guarded(route: str, fn: Callable[[], dict[str, Any]], status: int) -> JSONResponse:
        try:
            return _error(status, fn())
        except Refused as exc:
            log.info("%s -> %d", route, exc.status)
            return _error(exc.status, exc.body)
        except Exception as exc:
            log.error("%s -> 500 (%s)", route, type(exc).__name__)
            return _error(500, {"error": "internal"})

    @app.get(HEALTH)
    def health() -> JSONResponse:
        return _error(200, {"status": "ok"})

    @app.post("/sessions")
    def start(body: StartBody) -> JSONResponse:
        return guarded("start", lambda: service.start(body), 201)

    @app.post("/sessions/{session_id}/turns")
    def turn(session_id: str, body: TurnBody) -> JSONResponse:
        return guarded("turn", lambda: service.turn(session_id, body), 200)

    @app.post("/sessions/{session_id}/end")
    def end(session_id: str) -> JSONResponse:
        return guarded("end", lambda: service.end(session_id), 200)

    return app
