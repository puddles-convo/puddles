"""Stage 6: the overnight loop's logic — scenario draws, spend and failure accounting, the driver.

`run_loop` runs one round at a time until a cap stops it: `max_rounds`, the spend limit, or
three rounds in a row failing on the same agent, raising an error, or scoring nothing
(docs/specs/unattended-run.md section 1). Why it stopped goes to `log/loop.jsonl`. Everything it needs survives a restart —
the decision log, `usage.jsonl` and the policy files — so a crashed run picks up where it
stopped. The round itself is `puddles.graph.run_round`; this module only decides *whether*
to start the next one and *which* scenarios it draws.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from loop.errors import ConfigError
from puddles import replay
from puddles.baselines import load_id_file
from puddles.config import LoopConfig
from puddles.decision_log import DecisionLog
from puddles.graph import NO_CONVERSATION_REASON, NOTHING_SCORED_REASON
from puddles.llm import AgentFailed
from puddles.memory import Memory
from puddles.round import prepare_round
from puddles.usage import Meter, read_usage, spend_by_round
from schemas import ConversationRecord, RoundEvent, Scenario, ScoredConversation

FAILURE_STREAK_LIMIT = 3
ROUND_ERROR_LIMIT = 3  # round errors in a row (spec 1.6)
NOTHING_SCORED_LIMIT = 3  # rounds in a row with no scored training conversation (spec 1.7)
ERROR_CHARS = 300
LOOP_LOG = "log/loop.jsonl"
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
    failed: bool = False  # stopped by failures (round errors, nothing scored, same-agent streak), not a cap
    policy_in_force: Optional[str] = None


RoundRunner = Callable[[Path, list[Scenario], int], RoundEvent]


def _spend(lines: Sequence[dict]) -> float:
    return sum(spend_by_round(lines).values())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_loop_line(root: Path, line: dict) -> None:
    """One JSON line appended to `log/loop.jsonl`."""
    path = Path(root) / LOOP_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def policy_in_force_or_none(log: DecisionLog, root: Path, start_policy: Path) -> Optional[str]:
    """The policy file in force per the log (the replay check, as `--check` gives it), or None
    when the log cannot say; the stop line is written either way."""
    try:
        return str(replay.check(log, Path(root), Path(start_policy)))
    except Exception:
        return None


def write_stopped(root: Path, summary: LoopSummary, log: DecisionLog, start_policy: Path,
                  out: Callable[[str], None] = print) -> None:
    """Spec 1.8: one `stopped` line in `log/loop.jsonl`, and the same printed."""
    summary.policy_in_force = policy_in_force_or_none(log, root, start_policy)
    append_loop_line(root, {"event": "stopped", "reason": summary.stop_reason, "rounds": len(summary.rounds),
                            "spend_usd": summary.spend_usd, "policy_in_force": summary.policy_in_force,
                            "at": _now()})
    out(f"stopped: {summary.stop_reason} — {len(summary.rounds)} rounds, ${summary.spend_usd:.4f} spent, "
        f"policy in force: {summary.policy_in_force}")


def nothing_scored(event: RoundEvent, root: Path) -> bool:
    """The round ended with no scored training conversation. Its scores file decides when the round
    left one (the round's own judge adapter always does); otherwise the reason's opening sentence,
    which finish_round may follow with the unscored IDs."""
    if event.outcome != "no_pattern":
        return False
    path = Path(root) / SCORES_DIR / f"round-{event.round:02d}-train.jsonl"
    if path.exists():
        return not any(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return event.reason.startswith((NO_CONVERSATION_REASON, NOTHING_SCORED_REASON))


def nothing_scored_streak(log: DecisionLog, root: Path) -> int:
    """Rounds in a row, ending at the newest in the log, with no scored training conversation."""
    streak = 0
    for event in reversed(log.rounds()):
        if not nothing_scored(event, root):
            break
        streak += 1
    return streak


def round_error_streak(log: DecisionLog, root: Path) -> int:
    """Round errors since the newest round that ended with an outcome, read from `log/loop.jsonl`.

    A round number advances only with an outcome, so they are the errors for the next round.
    A line that does not parse (cut short by a kill) is skipped.
    """
    path = Path(root) / LOOP_LOG
    if not path.exists():
        return 0
    next_round, count = log.next_round(), 0
    for text in path.read_text(encoding="utf-8").splitlines():
        try:
            line = json.loads(text)
        except ValueError:
            continue
        if isinstance(line, dict) and line.get("event") == "round_error" and line.get("round") == next_round:
            count += 1
    return count


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
    after_round: Optional[Callable[[int], object]] = None,
) -> LoopSummary:
    """Run rounds until a cap stops the loop. `run_one(policy_path, scenarios, round)` runs
    one round and returns its RoundEvent — the script passes `graph.run_round` wrapped.

    `max_rounds` counts round events in the log, so a restarted loop keeps the same budget.
    An `Exception` from one round (its `prepare_round` included) costs that round only: a
    `round_error` line goes to `log/loop.jsonl` and the next turn starts as a restart does.
    The round-error and nothing-scored streaks are read from disk at each turn, so they survive a restart.
    Every stop writes a `stopped` line there (`write_stopped`).
    `after_round(round)`, when given, runs after each round event (the weight-training step); an
    `Exception` from it is a `weight_error` line and the loop goes on.
    """
    usage_file = Path(root) / USAGE_PATH
    summary = LoopSummary()
    try:
        while True:
            lines = read_usage(usage_file)
            spent = _spend(lines)
            summary.spend_usd = spent
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
                summary.failed = True
                break
            # Both streaks are read from disk, so a restarted loop carries them on.
            if round_error_streak(log, root) >= ROUND_ERROR_LIMIT:
                summary.stop_reason = f"{ROUND_ERROR_LIMIT} round errors in a row"
                summary.failed = True
                break
            if nothing_scored_streak(log, root) >= NOTHING_SCORED_LIMIT:
                summary.stop_reason = f"no training conversation scored in {NOTHING_SCORED_LIMIT} rounds in a row"
                summary.failed = True
                break
            round_ = log.next_round()
            try:
                policy_path = prepare_round(log, memory, root, start_policy)
                drawn = draw_scenarios(scenarios, cfg.conversations_per_round, cfg.seed, round_)
                event = run_one(policy_path, drawn, round_)
            except Exception as exc:  # KeyboardInterrupt and SystemExit are not Exceptions: they still stop
                error = f"{type(exc).__name__}: {str(exc)[:ERROR_CHARS]}"
                append_loop_line(root, {"event": "round_error", "round": round_, "error": error, "at": _now()})
                out(f"round {round_}: error. {error}")
                continue
            summary.rounds.append((event.round, event.outcome))
            spent = _spend(read_usage(usage_file))
            summary.spend_usd = spent
            line = f"round {event.round}: {event.outcome}. {event.reason}"
            if event.scores is not None:
                line += f" quality {event.scores.before.quality:.2f} -> {event.scores.after.quality:.2f}"
            out(f"{line} — spent ${spent:.4f}")
            if after_round is not None:
                try:
                    after_round(event.round)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:ERROR_CHARS]}"
                    append_loop_line(root, {"event": "weight_error", "round": event.round, "error": error,
                                            "at": _now()})
                    out(f"round {event.round}: weight step error. {error}")
    except BaseException as exc:  # escaped outside a round, or KeyboardInterrupt: still written down
        summary.stop_reason = f"{type(exc).__name__}: {str(exc)[:ERROR_CHARS]}"
        summary.failed = True
        try:
            write_stopped(root, summary, log, start_policy, out)
        except Exception:
            pass  # the original error is the one to see
        raise
    write_stopped(root, summary, log, start_policy, out)
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
