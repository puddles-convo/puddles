"""The helper: one reply under the policy in force."""
from __future__ import annotations

import re
from typing import Sequence

from loop.errors import HelperError
from loop.llm import ChatClient
from loop.policy import Policy
from loop.prompts import helper_messages, render_helper_system
from schemas import HelperReply, Turn

_CLOSERS = "\"'”’)]}"
_ENDS_COMPLETE = re.compile(rf"[.!?][{re.escape(_CLOSERS)}]?$")
_SENTENCE_END = re.compile(rf"[.!?][{re.escape(_CLOSERS)}]?(?=\s)")
# Short titles and common abbreviations whose full stop does not end a sentence (exact case).
ABBREVIATIONS = frozenset({"Dr", "Mr", "Mrs", "Ms", "Prof", "St", "vs", "etc", "approx"})
# Initialisms of two or more single letters with full stops: U.S., e.g., i.e., a.m., P.M.
_INITIALISM = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]$")
_OPENERS = "\"'“‘([{"


def _is_boundary(text: str, match: re.Match) -> bool:
    """A `!` or `?` always ends a sentence; a `.` does unless it ends an abbreviation.

    The match is already followed by whitespace, so a full stop inside a number
    (`3.` in `3.5`) never reaches here; `988.` followed by whitespace is a sentence end.
    """
    if text[match.start()] != ".":
        return True
    before = text[:match.start()]
    token = "" if not before or before[-1].isspace() else before.split()[-1].lstrip(_OPENERS)
    return not (token in ABBREVIATIONS or _INITIALISM.fullmatch(token))


def trim_to_sentence(text: str) -> str:
    """Cut text back to its last complete sentence; keep it whole if it has none.

    A sentence boundary is `.`, `!` or `?` (with any closing quote or bracket after it)
    that ends the text or is followed by whitespace, where a `.` does not end an
    initialism (`U.S.`, `e.g.`) or a word in ABBREVIATIONS. Text that ends on such a
    full stop is kept whole: the model stopped there.
    """
    if _ENDS_COMPLETE.search(text):
        return text
    ends = [m for m in _SENTENCE_END.finditer(text) if _is_boundary(text, m)]
    return text[:ends[-1].end()] if ends else text


def helper_reply(client: ChatClient, policy: Policy, turns: Sequence[Turn], *,
                 session_id: str, turn: int, user_notes: Sequence[str] = (),
                 max_tokens: int = 80, seed: int | None = None) -> HelperReply:
    """Ask the helper model for its next reply, retrying once on an empty one."""
    system = render_helper_system(policy, user_notes)
    messages = helper_messages(turns)
    for _ in range(2):
        text = client.complete(system, messages, max_tokens=max_tokens, seed=seed).strip()
        if text:
            return HelperReply(session_id=session_id, turn=turn, text=trim_to_sentence(text),
                               policy_version=policy.version, fixed_handoff_response=False)
    raise HelperError(f"empty helper reply twice (session {session_id}, turn {turn})")
