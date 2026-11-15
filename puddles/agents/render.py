"""Markdown shown to a model that reads a conversation without taking part in it."""
from typing import Iterable

from schemas import Turn

SPEAKERS = {"user": "Person", "helper": "Helper"}


def transcript(turns: Iterable[Turn]) -> str:
    """One line per turn, in order: `**Person:** ...` or `**Helper:** ...`.

    Only the turns go in: no conversation ID, policy version or scenario. A line break inside a
    turn becomes a space, so a turn cannot pass for two.
    """
    return "\n".join(f"**{SPEAKERS[t.speaker]}:** {' '.join(t.text.split())}" for t in turns)
