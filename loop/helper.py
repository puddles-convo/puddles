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


def trim_to_sentence(text: str) -> str:
    """Cut text back to its last complete sentence; keep it whole if it has none."""
    if _ENDS_COMPLETE.search(text):
        return text
    ends = list(_SENTENCE_END.finditer(text))
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
