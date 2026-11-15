"""Stage 6: the overnight loop's logic — scenario draws, spend and failure accounting, the driver.

`run_loop` runs one round at a time until a cap stops it: `max_rounds`, the spend limit, or
three rounds in a row failing on the same agent. Everything it needs survives a restart —
the decision log, `usage.jsonl` and the policy files — so a crashed run picks up where it
stopped. The round itself is `puddles.graph.run_round`; this module only decides *whether*
to start the next one and *which* scenarios it draws.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from loop.errors import ConfigError
from puddles.baselines import load_id_file
from puddles.config import LoopConfig
from puddles.decision_log import DecisionLog
from puddles.llm import AgentFailed
from puddles.memory import Memory
from puddles.round import prepare_round
from puddles.usage import Meter, read_usage, spend_by_round
from schemas import ConversationRecord, RoundEvent, Scenario, ScoredConversation

FAILURE_STREAK_LIMIT = 3
SPEND_MARGIN = 1.5  # stop when spend so far + 1.5 x the worst round so far would pass the limit
USAGE_PATH = "log/usage.jsonl"
SCORES_DIR = "scores"


class LoopError(ConfigError):
    """The loop's config, scenario files or caps cannot be used."""


# ---------- scenario draws ----------


def draw_scenarios(scenarios: Sequence[Scenario], n: int, seed: int, round_: int) -> list[Scenario]:
    """`n` scenarios drawn with `random.Random(f"{seed}:{round_}")`.

    The same seed and round always draw the same set, so re-running a round after a crash
    trains on the same conversations; a different round draws a different set.
    """
    if n < 1:
        raise LoopError(f"conversations_per_round must be at least 1, got {n}")
    pool = list(scenarios)
    if not pool:
        raise LoopError("no training scenarios to draw from")
    return random.Random(f"{seed}:{round_}").sample(pool, min(n, len(pool)))


def check_id_files(validation_path: str | Path, train_path: str | Path) -> tuple[list[str], list[str]]:
    """The validation and training ID files must share no ID. Returns both lists."""
    validation, train = load_id_file(validation_path), load_id_file(train_path)
    overlap = sorted(set(validation) & set(train))
    if overlap:
        shown = ", ".join(overlap[:5]) + (" ..." if len(overlap) > 5 else "")
        raise LoopError(f"{validation_path} and {train_path} share {len(overlap)} IDs: {shown}")
    return validation, train


# ---------- spend and failure accounting (from usage.jsonl lines) ----------


def worst_round_cost(lines: Sequence[dict], estimate: float = 0.0) -> float:
    """The most one round has cost so far; `estimate` stands in before any round has run."""
    per_round = spend_by_round(lines)
    return max(per_round.values(), default=estimate)


def agent_failure_streaks(lines: Sequence[dict]) -> dict[str, int]:
    """Per agent, how many rounds in a row — ending at the newest — logged a `failed` line.

    A round counts for an agent only if it has a `failed` line for that agent; a round with
    no `failed` line for the agent, or none at all, breaks that agent's streak. Several
    `failed` lines in one round still count as one round.
    """
    rounds_for: dict[str, set[int]] = {}
    for line in lines:
        if line.get("event") == "failed":
            agent = str(line.get("agent", "?"))
            rounds_for.setdefault(agent, set()).add(int(line.get("round") or 0))
    streaks = {}
    for agent, rounds in rounds_for.items():
        streak, round_ = 0, max(rounds)
        while round_ in rounds:
            streak += 1
            round_ -= 1
        streaks[agent] = streak
    return streaks


def repeated_failure(lines: Sequence[dict], limit: int = FAILURE_STREAK_LIMIT) -> Optional[str]:
    """The agent that failed `limit` rounds in a row, or None. Ties name the first alphabetically."""
    stuck = sorted(agent for agent, streak in agent_failure_streaks(lines).items() if streak >= limit)
    return stuck[0] if stuck else None


# ---------- the driver ----------


@dataclass
class LoopSummary:
    """What the loop did: one (round, outcome) per round event it wrote."""

    rounds: list[tuple[int, str]] = field(default_factory=list)
    stop_reason: str = ""
    spend_usd: float = 0.0


RoundRunner = Callable[[Path, list[Scenario], int], RoundEvent]


def _spend(lines: Sequence[dict]) -> float:
    return sum(spend_by_round(lines).values())


def run_loop(
    cfg: LoopConfig,
    *,
    log: DecisionLog,
    memory: Memory,
    root: Path,
    start_policy: Path,
    scenarios: Sequence[Scenario],
    run_one: RoundRunner,
    out: Callable[[str], None] = print,
) -> LoopSummary:
    """Run rounds until a cap stops the loop. `run_one(policy_path, scenarios, round)` runs
    one round and returns its RoundEvent — the script passes `graph.run_round` wrapped.

    `max_rounds` counts round events in the log, so a restarted loop keeps the same budget.
    """
    usage_file = Path(root) / USAGE_PATH
    summary = LoopSummary()
    while True:
        lines = read_usage(usage_file)
        spent = _spend(lines)
        if cfg.max_rounds is not None and len(log.rounds()) >= cfg.max_rounds:
            summary.stop_reason = f"max_rounds ({cfg.max_rounds})"
            break
        if cfg.spend_limit_usd is not None:
            per_round = spend_by_round(lines)
            estimate = cfg.round_cost_estimate_usd or 0.0
            worst = max(per_round.values(), default=estimate)
            if spent + SPEND_MARGIN * worst > cfg.spend_limit_usd:
                summary.stop_reason = (
                    f"spend limit ${cfg.spend_limit_usd:.2f}: ${spent:.2f} spent, worst round ${worst:.2f}"
                )
                break
        stuck = repeated_failure(lines)
        if stuck is not None:
            summary.stop_reason = f"{stuck} failed in {FAILURE_STREAK_LIMIT} rounds in a row"
            break
        round_ = log.next_round()
        policy_path = prepare_round(log, memory, root, start_policy)
        drawn = draw_scenarios(scenarios, cfg.conversations_per_round, cfg.seed, round_)
        event = run_one(policy_path, drawn, round_)
        summary.rounds.append((event.round, event.outcome))
        spent = _spend(read_usage(usage_file))
        summary.spend_usd = spent
        line = f"round {event.round}: {event.outcome}. {event.reason}"
        if event.scores is not None:
            line += f" quality {event.scores.before.quality:.2f} -> {event.scores.after.quality:.2f}"
        out(f"{line} — spent ${spent:.4f}")
    return summary


# ---------- the judge adapter ----------


def scoring_judge(
    judge, meter: Meter, root: Path
) -> Callable[[list[ConversationRecord], int], list[ScoredConversation]]:
    """Adapt a `puddles.eval.judge` panel/fallback to the graph's `deps.judge`.

    Each conversation is scored on its own; a judge failure leaves that conversation unscored.
    ScoreRecords go to `scores/round-NN-train.jsonl` and each reported usage entry is logged
    to the meter, so judge spend counts toward the loop's caps.
    """

    def judge_records(records: list[ConversationRecord], round_: int) -> list[ScoredConversation]:
        scored: list[ScoredConversation] = []
        score_lines: list[str] = []
        for record in records:
            try:
                result = judge.score(record)
            except (AgentFailed, ValueError):
                continue  # the conversation stays unscored
            for use in result.usage:
                meter.note_done("judge", use.model, use.prompt_tokens, use.completion_tokens)
            score = result.score
            score_lines.append(score.model_dump_json())
            scored.append(
                ScoredConversation(
                    conversation_id=score.conversation_id,
                    empathy=score.empathy,
                    relevance=score.relevance,
                    note=score.note,
                    turns=list(record.turns),
                )
            )
        if score_lines:
            path = Path(root) / SCORES_DIR / f"round-{round_:02d}-train.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(line + "\n" for line in score_lines), encoding="utf-8")
        elif records:
            meter.failed("judge", "no conversation could be scored")
        return scored

    return judge_records


def judge_models(judge) -> list[str]:
    """The model names a judge's config prices: panel judges plus Astra, or the fallback's."""
    config = getattr(judge, "config", None)
    if config is None:
        return []
    judges = getattr(config, "judges", None)
    if judges:
        return [*judges, config.astra]
    model = getattr(config, "model", None)
    return [model] if model else []
