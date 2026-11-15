"""The adapter between the round and Ansh's judge in `puddles/eval/judge.py`.

The contract is docs/specs/judging.md. The judge is only ever called through `score(record)` and
read through `version`; nothing here chooses a model, a prompt or a temperature, and nothing here
imports `puddles.eval`, so the gate can be given a judge without importing it either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from loop.llm import CallUsage
from puddles.usage import current
from schemas import ConversationRecord, ScoredConversation, ScoreRecord

CRITIC_CONVERSATIONS = 10
# The agent name usage is logged under when a meter is current but names no agent.
DEFAULT_AGENT = "judge"


class Judge(Protocol):
    version: str

    def score(self, record: ConversationRecord) -> Any:
        """The returned object has `.score`, a ScoreRecord, and `.usage`, a list of objects each with
        `.model`, `.prompt_tokens` and `.completion_tokens` (his JudgeResult)."""
        ...


@dataclass
class JudgeRun:
    scores: list[ScoreRecord] = field(default_factory=list)  # in the order of the records, unscored left out
    unscored: list[str] = field(default_factory=list)  # conversation IDs, in the order of the records
    reasons: dict[str, str] = field(default_factory=dict)  # conversation ID -> why it was left unscored


def record_usage(usage: Iterable[Any]) -> None:
    """Write each entry to the usage log of the current meter, as counts the model reported.

    The call has already happened, so the `sent` line holds the reported counts and the `done`
    line follows at once. A model with no price raises `MissingPrice` from the meter, uncaught:
    a round whose cost cannot be counted must not go on quietly.
    """
    meter, agent = current()
    if meter is None:
        return
    for entry in usage:
        prompt_tokens, completion_tokens = int(entry.prompt_tokens), int(entry.completion_tokens)
        call = meter.sent(agent or DEFAULT_AGENT, entry.model, prompt_tokens, completion_tokens)
        meter.done(call, CallUsage(prompt_tokens, completion_tokens, estimated=False))


def score_conversations(judge: Judge, records: list[ConversationRecord]) -> JudgeRun:
    """Score each record once, in order. A conversation the judge failed on is counted and named.

    His `score` already retries each model once and raises when the second answer fails too, so
    any exception from it means that conversation is left unscored; the rest are still scored.
    """
    run = JudgeRun()
    for record in records:
        try:
            result = judge.score(record)
        except Exception as exc:  # KeyboardInterrupt is not an Exception, so it still stops the run
            run.unscored.append(record.id)
            run.reasons[record.id] = f"{type(exc).__name__}: {exc}"
            continue
        record_usage(result.usage)  # the tokens were spent even if the score is then refused
        score = result.score
        if score.conversation_id != record.id:
            run.unscored.append(record.id)
            run.reasons[record.id] = f"the judge returned a score for {score.conversation_id!r}, not {record.id!r}"
            continue
        run.scores.append(score)
    return run


def to_scored(record: ConversationRecord, score: ScoreRecord) -> ScoredConversation:
    return ScoredConversation(conversation_id=score.conversation_id, empathy=score.empathy,
                              relevance=score.relevance, note=score.note, turns=list(record.turns))


def scored_conversations(records: list[ConversationRecord], scores: list[ScoreRecord]) -> list[ScoredConversation]:
    """Each score with its record, in the order of the scores."""
    by_id = {r.id: r for r in records}
    out = []
    for score in scores:
        record = by_id.get(score.conversation_id)
        if record is None:
            raise ValueError(f"score {score.id} is for conversation {score.conversation_id!r}, which has no record")
        out.append(to_scored(record, score))
    return out


def select_for_critic(scored: list[ScoredConversation], n: int = CRITIC_CONVERSATIONS) -> list[ScoredConversation]:
    """The `n` lowest by mean of empathy and relevance, lowest first, ties by conversation ID."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return sorted(scored, key=lambda s: ((s.empathy + s.relevance) / 2, s.conversation_id))[:n]


def train_scores_path(root: Path, round_: int) -> Path:
    return Path(root) / "scores" / f"round-{round_:02d}-train.jsonl"


def save_scores(scores: list[ScoreRecord], path: Path) -> None:
    """One ScoreRecord per line; the folder is made and an existing file is replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(s.model_dump_json() + "\n" for s in scores), encoding="utf-8")


def judge_models(judge: Any) -> list[str]:
    """The model names a judge will call, for the price check before the round starts."""
    config = getattr(judge, "config", None)
    if config is None:
        return []
    if hasattr(config, "judges") and hasattr(config, "astra"):
        return [*config.judges, config.astra]
    if hasattr(config, "model"):
        return [config.model]
    return []


class MeteredJudge:
    """A judge whose calls are written to the usage log; it is what the gate is given."""

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    @property
    def version(self) -> str:
        return self.judge.version

    def score(self, record: ConversationRecord) -> Any:
        result = self.judge.score(record)
        record_usage(result.usage)
        return result
