"""The crisis checker: one request to the helper's endpoint, answered in Ansh's one-line format.

Every keyword, the prompt, the timeout and every fixed line come from `puddles.eval.crisis`; nothing
of them is typed out here a second time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from loop.llm import ChatClient
from puddles.eval import crisis as eval_crisis
from schemas import CheckedMessage, CrisisCheckerInput, Turn

CheckStatus = Literal["answered", "failed", "late", "unreadable"]
RECENT_TURNS = 6
CHECKER_MAX_TOKENS = 40

_CATEGORIES = ("self", "danger", "unsure")


@dataclass(frozen=True)
class CrisisPath:
    """The crisis path files, loaded once: what the checker is asked and what the robot may say."""

    version: str
    system: str                      # the checker's system prompt
    user_template: str               # with {recent_turns} and {text}
    timeout_s: float
    fixed_responses: dict[str, str]  # keys: self, danger, unsure
    holding_line: str
    offline_line: str
    keywords: list[str]


class CheckFailed(Exception):
    """The checker gave no usable answer: the request failed, or its line could not be parsed."""

    status: Literal["failed", "unreadable"]

    def __init__(self, status: Literal["failed", "unreadable"], message: str = "") -> None:
        super().__init__(message or status)
        self.status = status


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"crisis path: {name} is missing or empty")
    return value


def load_path() -> CrisisPath:
    """Build the path from Ansh's files. A missing line stops here rather than at a person's turn."""
    data = eval_crisis.load_crisis_path()
    checker = data.get("checker") or {}
    fixed = data.get("fixed_responses") or {}
    responses = {category: _text(fixed.get(category), f"fixed_responses.{category}") for category in _CATEGORIES}
    timeout = checker.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("crisis path: checker.timeout_seconds is missing or not a positive number")
    template = _text(checker.get("user_template"), "checker.user_template")
    for field in ("{recent_turns}", "{text}"):
        if field not in template:
            raise ValueError(f"crisis path: checker.user_template has no {field}")
    keywords = eval_crisis.load_keywords()
    if not keywords:
        raise ValueError("crisis path: the keyword list is empty")
    return CrisisPath(version=_text(data.get("version"), "version"),
                      system=_text(checker.get("system"), "checker.system"),
                      user_template=template, timeout_s=float(timeout), fixed_responses=responses,
                      holding_line=_text(data.get("holding_line"), "holding_line"),
                      offline_line=_text(data.get("offline_line"), "offline_line"),
                      keywords=keywords)


def build_input(session_id: str, turn: int, turns: Sequence[Turn], text: str) -> CrisisCheckerInput:
    """`turns` are the turns before this message; the checker sees the last RECENT_TURNS of them."""
    recent = list(turns)[-RECENT_TURNS:] if RECENT_TURNS > 0 else []
    return CrisisCheckerInput(session_id=session_id, turn=turn, recent_turns=recent, text=text)


def render_user(inp: CrisisCheckerInput, path: CrisisPath) -> str:
    """Fill the template: one `<speaker>: <text>` line per recent turn, empty when there are none."""
    recent = "\n".join(f"{t.speaker}: {t.text}" for t in inp.recent_turns)
    return path.user_template.format(recent_turns=recent, text=inp.text)


def check(line: str) -> tuple[str, Optional[str], str]:
    """Ansh's parser; raises ValueError on a line it rejects."""
    return eval_crisis.parse_checker_line(line.strip())


def run(client: ChatClient, inp: CrisisCheckerInput, path: CrisisPath) -> CheckedMessage:
    """One request, no retry. The deadline is not kept here: `run_turn` keeps it on the clock."""
    try:
        line = client.complete(path.system, [{"role": "user", "content": render_user(inp, path)}],
                               max_tokens=CHECKER_MAX_TOKENS, temperature=0.0)
    except Exception as exc:
        raise CheckFailed("failed", f"checker request failed: {type(exc).__name__}") from exc
    if not isinstance(line, str):
        raise CheckFailed("unreadable", "checker reply is not text")
    try:
        risk, category, reason = check(line)
    except ValueError as exc:
        raise CheckFailed("unreadable", "checker line could not be parsed") from exc
    return to_record(inp, risk, category, reason)


def to_record(inp: CrisisCheckerInput, risk: str, category: Optional[str], reason: str) -> CheckedMessage:
    """The session, turn and text come from the input; the model never supplies them."""
    return CheckedMessage(session_id=inp.session_id, turn=inp.turn, text=inp.text,
                          risk=risk, category=category, reason=reason)  # type: ignore[arg-type]
