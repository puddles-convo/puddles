"""The simulated user: the next line of the person in a scenario."""
from __future__ import annotations

from typing import Sequence

from loop.errors import SimulatedUserError
from loop.llm import ChatClient
from loop.prompts import render_simulated_user_system, simulated_user_messages
from schemas import Scenario, Turn

END_MARKER = "[END]"
MAX_WORDS = 80
BANNED_PHRASES = ("as an ai", "language model", "simulat")


def _invalid_reason(line: str) -> str | None:
    if not line:
        return "empty line"
    if len(line.split()) > MAX_WORDS:
        return f"more than {MAX_WORDS} words"
    lowered = line.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            return f"out of character ({phrase!r})"
    return None


def next_user_line(client: ChatClient, scenario: Scenario, turns: Sequence[Turn], *,
                   seed: int | None = None, max_tokens: int = 120) -> str | None:
    """The person's next line, or None when they end the conversation."""
    system = render_simulated_user_system(scenario)
    messages = simulated_user_messages(turns)
    reason = None
    for _ in range(2):
        line = client.complete(system, messages, max_tokens=max_tokens,
                               temperature=0.9, seed=seed).strip()
        if line == END_MARKER:
            return None
        reason = _invalid_reason(line)
        if reason is None:
            return line
    raise SimulatedUserError(f"invalid simulated user line twice: {reason}")
