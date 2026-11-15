"""Training data built from the loop's own conversations.

At a helper turn of a training conversation, the model in force samples several replies to the same
context, and the loop's judge scores each one as the last turn of the conversation so far. DPO takes
the best and the worst reply of a group as a pair; GRPO takes the whole group with group-normalised
advantages.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from loop.prompts import helper_messages
from schemas import ConversationRecord, Turn


@dataclass
class Candidate:
    text: str
    tokens: list[int] = field(default_factory=list)  # sampled completion tokens; empty for a scripted backend
    logprobs: list[float] = field(default_factory=list)  # the sampler's logprob of each token
    quality: Optional[float] = None  # (empathy + relevance) / 2 from the judge; None if unscored


@dataclass
class Group:
    """Several replies to one context."""

    conversation_id: str
    turn: int  # index of the helper turn in the conversation
    system: str
    messages: list[dict[str, str]]
    prompt_tokens: list[int] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)

    def scored(self) -> list[Candidate]:
        return [c for c in self.candidates if c.quality is not None]


@dataclass
class Pair:
    group: Group
    chosen: Candidate
    rejected: Candidate

    @property
    def margin(self) -> float:
        assert self.chosen.quality is not None and self.rejected.quality is not None
        return self.chosen.quality - self.rejected.quality


class Sampler(Protocol):
    def prompt_tokens(self, system: str, messages: list[dict[str, str]]) -> list[int]: ...

    def sample(self, prompt_tokens: list[int], system: str, messages: list[dict[str, str]], n: int, *,
               max_tokens: int, temperature: float) -> list[Candidate]: ...


class Judge(Protocol):
    def score(self, record: ConversationRecord) -> Any: ...


def contexts(records: Sequence[ConversationRecord], k: int, seed: int) -> list[tuple[ConversationRecord, int]]:
    """Up to k (conversation, helper-turn index) pairs, drawn the same way for the same seed."""
    spots = [(r, i) for r in records for i, t in enumerate(r.turns) if t.speaker == "helper" and i > 0]
    random.Random(seed).shuffle(spots)
    return spots[:k]


def quality_of(judge: Judge, record: ConversationRecord, prefix: list[Turn], text: str, tag: str) -> Optional[float]:
    """The judge's (empathy + relevance) / 2 for the conversation so far ending in this reply."""
    candidate = record.model_copy(deep=True, update={
        "id": f"{record.id}-{tag}", "turns": [*prefix, Turn(speaker="helper", text=text)]})
    try:
        score = judge.score(candidate).score
    except Exception:  # noqa: BLE001  an unscored candidate is left out, like an unscored conversation
        return None
    return (score.empathy + score.relevance) / 2


def build_groups(records: Sequence[ConversationRecord], system: str, sampler: Sampler, judge: Judge, *,
                 prompts: int, samples: int, seed: int, max_tokens: int = 80,
                 temperature: float = 0.9) -> list[Group]:
    groups = []
    for record, i in contexts(records, prompts, seed):
        prefix = list(record.turns[:i])
        messages = helper_messages(prefix)
        tokens = sampler.prompt_tokens(system, messages)
        group = Group(conversation_id=record.id, turn=i, system=system, messages=messages, prompt_tokens=tokens)
        group.candidates = sampler.sample(tokens, system, messages, samples, max_tokens=max_tokens,
                                          temperature=temperature)
        for j, c in enumerate(group.candidates):
            if c.text.strip():
                c.quality = quality_of(judge, record, prefix, c.text, f"t{i}-s{j}")
        groups.append(group)
    return groups


def dpo_pairs(groups: Sequence[Group], min_margin: float) -> list[Pair]:
    """Best against worst of each group, when the judge separates them by at least min_margin."""
    pairs = []
    for g in groups:
        scored = g.scored()
        if len(scored) < 2:
            continue
        best = max(scored, key=lambda c: c.quality)  # type: ignore[arg-type,return-value]
        worst = min(scored, key=lambda c: c.quality)  # type: ignore[arg-type,return-value]
        pair = Pair(g, best, worst)
        if best is not worst and pair.margin >= min_margin:
            pairs.append(pair)
    return pairs


def advantages(qualities: Sequence[float]) -> list[float]:
    """GRPO's group-normalised advantage: (q - mean) / std; all zero when the group is tied."""
    n = len(qualities)
    mean = sum(qualities) / n
    std = math.sqrt(sum((q - mean) ** 2 for q in qualities) / n)
    return [0.0] * n if std == 0 else [(q - mean) / std for q in qualities]


def grpo_groups(groups: Sequence[Group]) -> list[tuple[Group, list[Candidate], list[float]]]:
    """Groups the judge did not tie, with each scored reply's advantage. A tied group teaches nothing."""
    out = []
    for g in groups:
        scored = g.scored()
        if len(scored) < 2:
            continue
        adv = advantages([c.quality for c in scored])  # type: ignore[misc]
        if any(a != 0 for a in adv):
            out.append((g, scored, adv))
    return out


def dpo_loss(policy_chosen: float, policy_rejected: float, ref_chosen: float, ref_rejected: float,
             beta: float) -> tuple[float, float, float]:
    """DPO's loss for one pair from summed completion logprobs, and its gradient with respect to the
    policy's chosen and rejected sums. The Tinker backend computes the same thing with torch."""
    z = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    loss = math.log1p(math.exp(-z)) if z > -30 else -z  # -log sigmoid(z)
    s = 1 / (1 + math.exp(z)) if z > -30 else 1.0  # sigmoid(-z)
    return loss, -beta * s, beta * s


def write_dataset(path: Path, method: str, groups: Sequence[Group], pairs: Sequence[Pair] = ()) -> None:
    """What the adapter was trained on, one JSON line per group (and per pair for DPO)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for g in groups:
        row = {"kind": "group", "method": method, **asdict(g)}
        row.pop("prompt_tokens")
        for c in row["candidates"]:
            c.pop("tokens")
            c.pop("logprobs")
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    for p in pairs:
        lines.append(json.dumps({"kind": "pair", "conversation_id": p.group.conversation_id, "turn": p.group.turn,
                                 "chosen": p.chosen.text, "rejected": p.rejected.text, "margin": p.margin},
                                ensure_ascii=False, sort_keys=True))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(line + "\n" for line in lines))
