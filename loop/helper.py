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
# Abbreviations by how their full stop is read (exact case, without the stop).
# Titles precede a capitalised name, so mid-text their stop is never a sentence end.
TITLES = frozenset({"Dr", "Mr", "Mrs", "Ms", "Prof", "St"})
# These end a sentence only when an uppercase letter follows, like initialisms.
ABBREVIATIONS = frozenset({"vs", "etc", "approx"})
# Initialisms of two or more single letters with full stops: U.S., e.g., i.e., a.m., P.M.
_INITIALISM = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]$")
_OPENERS = "\"'“‘([{"


def _is_boundary(text: str, match: re.Match) -> bool:
    """Whether a `.`, `!` or `?` followed by whitespace really ends a sentence.

    `!` and `?` always do, and so does anything with only whitespace after it. A `.`
    ending a title (TITLES) does not; one ending an initialism or a word in
    ABBREVIATIONS does only when the next non-space character is an uppercase letter,
    or an opening quote or bracket followed by one. The match is already followed by
    whitespace, so a full stop inside a number (`3.` in `3.5`) never reaches here;
    `988.` followed by whitespace is a sentence end.
    """
    rest = text[match.end():].lstrip()
    if text[match.start()] != "." or not rest:
        return True
    before = text[:match.start()]
    token = "" if not before or before[-1].isspace() else before.split()[-1].lstrip(_OPENERS)
    if token in TITLES:
        return False
    if token in ABBREVIATIONS or _INITIALISM.fullmatch(token):
        following = rest[1:2] if rest[0] in _OPENERS else rest[0]
        return following.isupper()
    return True


def trim_to_sentence(text: str) -> str:
    """Cut text back to its last complete sentence; keep it whole if it has none.

    A sentence boundary is `.`, `!` or `?` (with any closing quote or bracket after it)
    that ends the text or is followed by whitespace, subject to the abbreviation rules
    in `_is_boundary`. Text that ends on an abbreviation's full stop is kept whole:
    the model stopped there.
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
