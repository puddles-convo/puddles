"""The set_examples tool: code, not the model, picks short real exchanges for the helper's Examples section.

The proposer names a situation and how many examples it wants (1 to 3). This module picks that many
ESConv **train** conversations that the seeker rated highly and that best match the situation, and
cuts one exchange from each (what the person said, how the human supporter answered). The result is
one line, so it is edited, gated and replayed like any other policy item (`E1.`, `E2.`).

Only the train split is ever loaded: the gatekeeper's validation scenarios and the test split cannot
leak into the helper's prompt. ESConv is CC BY-NC 4.0 (Liu et al., 2021).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Optional, Sequence

MAX_EXAMPLES = 3
MAX_E_ITEMS = 2  # Examples items in a policy; at the limit set_examples rewrites the oldest one
MAX_TURN_CHARS = 170
MIN_HELPER_WORDS = 8
_WORD = re.compile(r"[a-z']+")
_STOP = frozenset("a an and are as at be but by for from has have i in is it its me my of on or so that the "
                  "their them they this to was were with you your about after when who what person "
                  "someone feels feeling".split())


class ExamplesError(ValueError):
    """No usable example for this situation."""


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _clip(text: str, limit: int = MAX_TURN_CHARS) -> str:
    text = " ".join(text.split()).replace('"', "'")
    return text if len(text) <= limit else text[: limit - 3].rsplit(" ", 1)[0] + "..."


@lru_cache(maxsize=1)
def _train() -> tuple:
    from puddles.eval.data import load_esconv

    return tuple(load_esconv("train"))


def _exchange(conv) -> Optional[tuple[str, str]]:
    """The first exchange where the person says something real and the supporter answers in full."""
    turns = conv.record.turns
    for i in range(1, len(turns)):
        user, helper = turns[i - 1], turns[i]
        if user.speaker == "user" and helper.speaker == "helper" and len(user.text.split()) >= 6 \
                and len(helper.text.split()) >= MIN_HELPER_WORDS:
            return _clip(user.text), _clip(helper.text)
    return None


def pick(situation: str, k: int, conversations: Optional[Sequence] = None) -> list[tuple[str, str, str]]:
    """Up to k (conversation id, person, supporter) exchanges for this situation, best match first.

    Only conversations rated 5 for both empathy and relevance by the seeker are used; ties are broken
    by the conversation ID so the choice is the same on every run.
    """
    if not 1 <= k <= MAX_EXAMPLES:
        raise ExamplesError(f"k must be 1 to {MAX_EXAMPLES}")
    want = _words(situation)
    if not want:
        raise ExamplesError("the situation has no words to match")
    pool = conversations if conversations is not None else _train()
    scored = []
    for conv in pool:
        r = conv.ratings
        if r.empathy != 5 or r.relevance != 5:
            continue
        s = conv.scenario
        overlap = len(want & _words(f"{s.situation} {s.problem_type} {s.emotion_type}"))
        if overlap:
            scored.append((-overlap, conv.record.id, conv))
    out = []
    for _, cid, conv in sorted(scored, key=lambda t: (t[0], t[1])):
        ex = _exchange(conv)
        if ex:
            out.append((cid, *ex))
        if len(out) == k:
            break
    if not out:
        raise ExamplesError(f"no top-rated train conversation matches {situation!r}")
    return out


def render(situation: str, picked: list[tuple[str, str, str]]) -> str:
    """One policy line holding the picked exchanges."""
    parts = [f'Person: "{person}" Supporter: "{supporter}"' for _, person, supporter in picked]
    return f"When someone is dealing with {_clip(situation, 90)}: " + " | ".join(parts)


def example_line(situation: str, k: int, picker: Callable[[str, int], list] = pick) -> tuple[str, list[str]]:
    """The Examples line and the ESConv conversation IDs behind it."""
    picked = picker(situation, k)
    return render(situation, picked), [cid for cid, _, _ in picked]
