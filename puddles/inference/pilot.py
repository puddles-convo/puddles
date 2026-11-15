"""The persona pilot: three synthetic people, three sessions each, with notes and without.

Each session runs through the live code (`run_turn`, then `finish_session`) with a simulated user
answering after a fixed opener; the existing judge scores it. The two arms differ only in the user
ID, `remember_me`, and whether the helper is given notes. Contract: docs/specs/persona-pilot.md.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import ValidationError

from loop.llm import ChatClient
from loop.policy import Policy
from loop.simulated_user import next_user_line
from puddles.inference.crisis import CrisisPath
from puddles.inference.live import finish_session, start_notes
from puddles.inference.note_store import NoteStore
from puddles.inference.notes import NotesResult
from puddles.inference.turn import Session, run_turn
from puddles.judging import Judge, score_conversations
from puddles.llm import LLM
from schemas import ConversationRecord, Scenario, ScoreRecord, Strict

ARMS = ("notes_on", "notes_off")
SESSIONS_PER_PERSONA = 3
DEFAULT_TURNS = 6
SESSION_GAP_DAYS = 7
POLICY_SENTENCE = ("The policy is the same for every person: there are no per-person policy rules. "
                   "The personal layer is the notes (situation, people, what_helps, goals, follow_up, "
                   "preferences), which the note-taker keeps and the helper is given at the start of a session.")
LIMITS_SENTENCE = ("Limits: three personas, one run each, simulated users and a model judge; "
                   "differences this small are not evidence of an effect.")
NOT_MEASURED = "not measured"
FAILED = "failed"

PERSONA_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
PERSONA_ID_MAX = 12


class PersonaSession(Strict):
    session: int
    opener: str
    scenario: Scenario
    fake_markers: list[str] = []


class Persona(Strict):
    persona_id: str
    description: str
    sessions: list[PersonaSession]


@dataclass
class Clients:
    helper: ChatClient
    checker: ChatClient
    user: ChatClient
    path: CrisisPath


@dataclass
class NoteChanges:
    added: list[dict] = field(default_factory=list)
    replaced: list[dict] = field(default_factory=list)
    confirmed: list[dict] = field(default_factory=list)
    deleted: list[dict] = field(default_factory=list)
    failed: bool = False
    reason: str = ""


@dataclass
class SessionResult:
    persona_id: str
    session: int
    arm: str
    user_id: str
    session_id: str
    notes_given: list[str]
    record: Optional[ConversationRecord]
    notes_before: list[dict]
    notes_after: list[dict]
    changes: NoteChanges
    failed: bool
    reason: str


@dataclass
class SessionScore:
    status: Literal["scored", "unscored", "failed"]
    score: Optional[ScoreRecord]
    reason: str


# ---------- the persona file ----------

def load_personas(path: Path) -> list[Persona]:
    """The personas in file order; any breach of the contract's rules is a ValueError naming where."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("personas"), list):
        raise ValueError(f"{path}: the top level must be an object with a list under 'personas'")
    if not data["personas"]:
        raise ValueError(f"{path}: 'personas' holds no persona")
    personas: list[Persona] = []
    seen_ids: set[str] = set()
    seen_scenarios: set[str] = set()
    for i, raw in enumerate(data["personas"], 1):
        name = raw.get("persona_id", f"#{i}") if isinstance(raw, dict) else f"#{i}"
        try:
            persona = Persona.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"persona {name}: {exc}") from exc
        pid = persona.persona_id
        if len(pid) > PERSONA_ID_MAX or not PERSONA_ID_RE.fullmatch(pid):
            raise ValueError(f"persona {pid!r}: the ID must match [a-z0-9][a-z0-9-]* and be at most "
                             f"{PERSONA_ID_MAX} characters")
        if pid in seen_ids:
            raise ValueError(f"persona {pid}: the ID appears more than once")
        seen_ids.add(pid)
        numbers = [s.session for s in persona.sessions]
        if numbers != list(range(1, SESSIONS_PER_PERSONA + 1)):
            raise ValueError(f"persona {pid}: sessions must be numbered 1 to {SESSIONS_PER_PERSONA} in order, "
                             f"got {numbers}")
        for s in persona.sessions:
            if not s.opener.strip():
                raise ValueError(f"persona {pid} session {s.session}: the opener is empty")
            sid = s.scenario.scenario_id
            if sid in seen_scenarios:
                raise ValueError(f"persona {pid} session {s.session}: scenario_id {sid!r} appears more than once")
            seen_scenarios.add(sid)
        personas.append(persona)
    return personas


def arm_user_id(persona_id: str, arm: str) -> str:
    if arm == "notes_on":
        return f"{persona_id}-on"
    if arm == "notes_off":
        return f"{persona_id}-off"
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def session_day(start: date, session: int) -> date:
    return start + timedelta(days=SESSION_GAP_DAYS * (session - 1))


def _noon(day: date) -> datetime:
    return datetime.combine(day, time(12, 0), tzinfo=timezone.utc)


# ---------- notes ----------

def snapshot_notes(store: NoteStore, user_id: str) -> list[dict]:
    """The live notes as the files hold them, in index order. Reads only; never asks past.dev."""
    stored = store.stored(user_id)
    out = []
    for n in store.index(user_id).notes:
        if n.note_id not in stored:
            continue
        out.append({"note_id": n.note_id, "kind": n.kind, "fact": stored[n.note_id].fact,
                    "last_confirmed": n.last_confirmed.isoformat(),
                    "due_date": n.due_date.isoformat() if n.due_date else None})
    return out


def note_changes(before: list[dict], after: list[dict], result: NotesResult) -> NoteChanges:
    facts_before = {n["note_id"]: n["fact"] for n in before}
    facts_after = {n["note_id"]: n["fact"] for n in after}

    def from_after(ids: list[str]) -> list[dict]:
        return [{"note_id": i, "fact": facts_after.get(i, facts_before.get(i, ""))} for i in ids]

    def from_before(ids: list[str]) -> list[dict]:
        return [{"note_id": i, "fact": facts_before.get(i, facts_after.get(i, ""))} for i in ids]

    return NoteChanges(added=from_after(result.added), replaced=from_before(result.replaced),
                       confirmed=from_before(result.confirmed), deleted=from_before(result.deleted),
                       failed=result.failed, reason=result.reason)


# ---------- one session ----------

def run_session(persona: Persona, session: int, arm: str, *, clients: Clients, policy: Policy,
                store: NoteStore, note_llm: LLM, root: Path, turns: int = DEFAULT_TURNS,
                start: date, fake: bool = False) -> SessionResult:
    """One session through `run_turn` and `finish_session`. A model failure is recorded, not raised."""
    if turns < 1:
        raise ValueError(f"turns must be at least 1, got {turns}")
    user_id = arm_user_id(persona.persona_id, arm)
    plan = next((s for s in persona.sessions if s.session == session), None)
    if plan is None:
        raise ValueError(f"persona {persona.persona_id} has no session {session}")
    today = session_day(start, session)
    session_id = f"{user_id}-s{session}"
    notes_before = snapshot_notes(store, user_id)
    notes_given = start_notes(store, user_id, today=today) if arm == "notes_on" else []
    live = Session(session_id=session_id, policy=policy, helper_client=clients.helper,
                   checker_client=clients.checker, path=clients.path, user_id=user_id,
                   remember_me=(arm == "notes_on"), consent_for_training=True,
                   user_notes=list(notes_given), root=Path(root))
    failed, reason = False, ""
    try:
        run_turn(live, plan.opener)
        said = 1
        while said < turns:
            line = next_user_line(clients.user, plan.scenario, live.turns)
            if line is None:
                break
            run_turn(live, line)
            said += 1
    except Exception as exc:  # KeyboardInterrupt is not an Exception
        failed, reason = True, f"{type(exc).__name__}: {exc}"

    record: Optional[ConversationRecord] = None
    try:
        record, result = finish_session(live, store, note_llm, today=today, now=_noon(today))
    except Exception as exc:
        result = NotesResult(failed=True)
        if not failed:
            failed, reason = True, f"{type(exc).__name__}: {exc}"
        record = None
    try:
        notes_after = snapshot_notes(store, user_id)
    except Exception:
        notes_after = []
    changes = note_changes(notes_before, notes_after, result)
    if failed:
        record = None
    elif record is None:
        failed, reason = True, "no record"
    return SessionResult(persona_id=persona.persona_id, session=session, arm=arm, user_id=user_id,
                         session_id=session_id, notes_given=list(notes_given), record=record,
                         notes_before=notes_before, notes_after=notes_after, changes=changes,
                         failed=failed, reason=reason)


def score_session(judge: Judge, result: SessionResult) -> SessionScore:
    if result.failed or result.record is None:
        return SessionScore("failed", None, result.reason)
    run = score_conversations(judge, [result.record])
    if run.scores:
        return SessionScore("scored", run.scores[0], "")
    return SessionScore("unscored", None, run.reasons.get(result.record.id, "unscored"))


def session_dict(result: SessionResult, score: SessionScore) -> dict:
    """The shape of `s<n>.json`, and of each session in `pilot.json`."""
    s = score.score if score.status == "scored" else None
    return {
        "persona_id": result.persona_id, "session": result.session, "arm": result.arm,
        "user_id": result.user_id, "session_id": result.session_id,
        "failed": result.failed, "reason": result.reason,
        "notes_given": list(result.notes_given),
        "record": result.record.model_dump(mode="json") if result.record is not None else None,
        "notes_before": result.notes_before,
        "notes_after": result.notes_after,
        "changes": {"added": result.changes.added, "replaced": result.changes.replaced,
                    "confirmed": result.changes.confirmed, "deleted": result.changes.deleted,
                    "failed": result.changes.failed, "reason": result.changes.reason},
        "score": {"status": score.status, "reason": score.reason,
                  "empathy": s.empathy if s else None, "relevance": s.relevance if s else None,
                  "emotion_before": s.emotion_before if s else None,
                  "emotion_after": s.emotion_after if s else None,
                  "judge_version": s.judge_version if s else None, "note": s.note if s else None},
    }


def run_pilot(personas: list[Persona], *, make_clients: Callable[[Persona, int, str], Clients],
              policy: Policy, store: NoteStore, note_llm: LLM, judge: Judge, root: Path,
              turns: int = DEFAULT_TURNS, start: date, fake: bool = False, policy_path: str = "",
              on_session: Optional[Callable[[dict], None]] = None) -> dict:
    """Every persona, both arms, sessions 1 to 3; a failed session is recorded and the pilot goes on.

    Returns the `pilot.json` object and writes nothing. `policy_path` and `on_session` (called with
    each session's dict as it finishes) are additions the command uses.
    """
    out_personas = []
    for persona in personas:
        arms: dict[str, list[dict]] = {}
        for arm in ARMS:
            arms[arm] = []
            for n in range(1, SESSIONS_PER_PERSONA + 1):
                clients = make_clients(persona, n, arm)
                result = run_session(persona, n, arm, clients=clients, policy=policy, store=store,
                                     note_llm=note_llm, root=root, turns=turns, start=start, fake=fake)
                entry = session_dict(result, score_session(judge, result))
                arms[arm].append(entry)
                if on_session is not None:
                    on_session(entry)
        out_personas.append({"persona_id": persona.persona_id, "description": persona.description,
                             "sessions": arms})
    return {"policy_version": policy.version, "policy_path": policy_path, "turns": turns, "fake": fake,
            "start": start.isoformat(), "judge_version": judge.version,
            "policy_sentence": POLICY_SENTENCE, "limits_sentence": LIMITS_SENTENCE,
            "personas": out_personas}


# ---------- the report ----------

def mean_or_not_measured(values: list[float]) -> float | str:
    if not values:
        return NOT_MEASURED
    return round(sum(values) / len(values), 2)


def _cell(text: Any) -> str:
    return " ".join(str(text).split("\n")).replace("|", "\\|")


def _score_cell(s: dict, key: str) -> str:
    status = s["score"]["status"]
    if status == "scored":
        return str(s["score"][key])
    return status  # "failed" or "unscored"


def _int_diff(a: str, b: str) -> str:
    if not (a.lstrip("-").isdigit() and b.lstrip("-").isdigit()):
        return NOT_MEASURED
    d = int(a) - int(b)
    return f"+{d}" if d > 0 else str(d)


def _fmt_mean(v: float | str) -> str:
    return v if isinstance(v, str) else f"{v:.2f}"


def _mean_diff(a: float | str, b: float | str) -> str:
    if isinstance(a, str) or isinstance(b, str):
        return NOT_MEASURED
    d = round(a - b, 2)
    return f"+{d:.2f}" if d > 0 else f"{d:.2f}" if d < 0 else "0.00"


def _scored(sessions: list[dict], key: str) -> list[float]:
    return [s["score"][key] for s in sessions if s["score"]["status"] == "scored"]


def _helper_replies(s: dict, k: int = 2) -> list[str]:
    if s is None or s.get("record") is None:
        return ["-"] * k
    replies = [t["text"] for t in s["record"]["turns"] if t["speaker"] == "helper"][:k]
    return replies + ["-"] * (k - len(replies))


def _notes_rows(s: dict) -> list[str]:
    n = s["session"]
    if s["failed"]:
        return [f"| {n} | - | (session failed) | |"]
    added = {c["note_id"] for c in s["changes"]["added"]}
    rows = [f"| {n} | {_cell(note['kind'])} | {_cell(note['fact'])} | {'added' if note['note_id'] in added else ''} |"
            for note in s["notes_after"]]
    kinds = {note["note_id"]: note["kind"] for note in s["notes_before"]}
    for label in ("replaced", "deleted"):
        for c in s["changes"][label]:
            rows.append(f"| {n} | {_cell(kinds.get(c['note_id'], '-'))} | {_cell(c['fact'])} | {label} |")
    return rows or [f"| {n} | - | (no notes) | |"]


def render_report(pilot: dict) -> str:
    """`report.md` from the `pilot.json` object alone; reads no file."""
    mode = "fake" if pilot["fake"] else "real"
    lines = ["# Persona pilot", "", pilot["policy_sentence"], "",
             f"Policy: {pilot['policy_version']} ({pilot['policy_path']}). Turns per session: {pilot['turns']}. "
             f"Mode: {mode}. Judge: {pilot['judge_version']}.", ""]
    for p in pilot["personas"]:
        on, off = p["sessions"]["notes_on"], p["sessions"]["notes_off"]
        lines += [f"## {p['persona_id']}: {_cell(p['description'])}", "",
                  "### Notes after each session (notes_on)", "",
                  "| Session | Kind | Fact | Change |", "|---|---|---|---|"]
        for s in on:
            lines += _notes_rows(s)
        lines += ["", "### Judge scores", "",
                  "| Session | notes_on empathy | notes_off empathy | Difference | notes_on relevance | "
                  "notes_off relevance | Difference |",
                  "|---|---|---|---|---|---|---|"]
        for a, b in zip(on, off):
            ea, eb = _score_cell(a, "empathy"), _score_cell(b, "empathy")
            ra, rb = _score_cell(a, "relevance"), _score_cell(b, "relevance")
            lines.append(f"| {a['session']} | {ea} | {eb} | {_int_diff(ea, eb)} | {ra} | {rb} | {_int_diff(ra, rb)} |")
        s3_on = next((s for s in on if s["session"] == 3), None)
        s3_off = next((s for s in off if s["session"] == 3), None)
        r_on, r_off = _helper_replies(s3_on), _helper_replies(s3_off)
        lines += ["", "### Session 3, first two helper replies", "",
                  "| Reply | notes_on | notes_off |", "|---|---|---|"]
        for i in range(2):
            lines.append(f"| {i + 1} | {_cell(r_on[i])} | {_cell(r_off[i])} |")
        lines.append("")

    def arm_sessions(arm: str, only3: bool = False) -> list[dict]:
        return [s for p in pilot["personas"] for s in p["sessions"][arm] if not only3 or s["session"] == 3]

    lines += ["## Overall", "", "| Measure | notes_on | notes_off | Difference |", "|---|---|---|---|"]
    for label, only3, key in (("Mean empathy, session 3", True, "empathy"),
                              ("Mean relevance, session 3", True, "relevance"),
                              ("Mean empathy, all sessions", False, "empathy"),
                              ("Mean relevance, all sessions", False, "relevance")):
        a = mean_or_not_measured(_scored(arm_sessions("notes_on", only3), key))
        b = mean_or_not_measured(_scored(arm_sessions("notes_off", only3), key))
        lines.append(f"| {label} | {_fmt_mean(a)} | {_fmt_mean(b)} | {_mean_diff(a, b)} |")
    counts = {}
    for arm in ARMS:
        ss = arm_sessions(arm)
        counts[arm] = (sum(1 for s in ss if s["score"]["status"] == "scored"), len(ss),
                       sum(1 for s in ss if s["failed"]))
    lines.append(f"| Sessions scored | {counts['notes_on'][0]} of {counts['notes_on'][1]} | "
                 f"{counts['notes_off'][0]} of {counts['notes_off'][1]} | |")
    lines.append(f"| Sessions failed | {counts['notes_on'][2]} | {counts['notes_off'][2]} | |")
    kept = sum(len(s["notes_after"]) for s in arm_sessions("notes_on", True) if not s["failed"])
    deleted = sum(len(s["changes"]["deleted"]) for s in arm_sessions("notes_on"))
    lines.append(f"| Notes kept after session 3 | {kept} | 0 | |")
    lines.append(f"| Notes deleted on request | {deleted} | 0 | |")
    lines += ["", pilot["limits_sentence"], ""]
    return "\n".join(lines)
