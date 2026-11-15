"""What was tried and rejected is not tried again: docs/specs/unattended-run.md section 5.

Worked out by code from the decision log's round events only, for the policy version in force.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from puddles.decision_log import DecisionLog

SETTING = "PUDDLES_PARK_REVERTED"


def enabled() -> bool:
    """5.1: on only for `1`. scripts/run_loop.py refuses any value other than 1, 0 or unset."""
    return os.environ.get(SETTING) == "1"


@dataclass
class FailedTry:
    """5.2: a reverted round on the policy in force that has an edit."""
    round: int
    edit_id: str
    action: str
    rule_id: str
    pattern_id: Optional[str]  # the pattern's identity
    before: Optional[float]
    after: Optional[float]

    @property
    def negative(self) -> bool:
        return self.before is not None and self.after is not None and self.after < self.before

    def describe(self) -> str:
        measured = (f"quality {self.before:.3g} -> {self.after:.3g}"
                    if self.before is not None and self.after is not None else "quality not logged")
        return f"round {self.round}: {self.edit_id} ({self.action} {self.rule_id}) was reverted, {measured}"


@dataclass
class Parked:
    version: str
    rules: dict[str, list[FailedTry]] = field(default_factory=dict)  # rule_id -> the failed tries
    patterns: dict[str, list[FailedTry]] = field(default_factory=dict)  # pattern identity -> the failed tries


def _is_parked(tries: list[FailedTry]) -> bool:
    return len(tries) >= 2 or any(t.negative for t in tries)


def failed_tries(log: DecisionLog, version: str) -> list[FailedTry]:
    out = []
    for r in log.rounds():
        if r.outcome != "reverted" or r.edit is None or r.policy.base != version:
            continue
        before = r.scores.before.quality if r.scores else None
        after = r.scores.after.quality if r.scores else None
        out.append(FailedTry(round=r.round, edit_id=r.edit.id, action=r.edit.action, rule_id=r.edit.rule_id,
                             pattern_id=r.pattern.pattern_id if r.pattern else None,
                             before=before, after=after))
    return out


def parked(log: DecisionLog, version: str) -> Parked:
    """5.2: the rules and patterns parked on `version`, each with the failed tries behind it."""
    tries = failed_tries(log, version)
    by_rule: dict[str, list[FailedTry]] = {}
    by_pattern: dict[str, list[FailedTry]] = {}
    for t in tries:
        if t.action in ("reword", "remove"):
            by_rule.setdefault(t.rule_id, []).append(t)
        if t.pattern_id:
            by_pattern.setdefault(t.pattern_id, []).append(t)
    return Parked(version=version,
                  rules={k: v for k, v in by_rule.items() if _is_parked(v)},
                  patterns={k: v for k, v in by_pattern.items() if _is_parked(v)})


def tries_text(tries: list[FailedTry]) -> str:
    return "; ".join(t.describe() for t in tries)
