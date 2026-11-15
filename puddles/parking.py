"""What was tried and rejected is not tried again: docs/specs/unattended-run.md section 5.

Worked out by code from the decision log's round events only, for the policy version in force.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

from puddles.decision_log import DecisionLog

SETTING = "PUDDLES_PARK_REVERTED"
PARK_AFTER = "PUDDLES_PARK_AFTER"


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
    base: str = ""
    noise: Optional[float] = None
    rule_text: Optional[str] = None  # a removal has none

    @property
    def fall(self) -> Optional[float]:
        """How far quality fell, rounded as puddles/gate.py rounds a change before comparing it with the noise."""
        if self.before is None or self.after is None:
            return None
        return round(self.before - self.after, 9)

    @property
    def negative(self) -> bool:
        return self.fall is not None and self.fall > 0

    @property
    def clearly_worse(self) -> bool:
        """5.6: quality fell by more than the round's logged noise (equal is not more)."""
        return self.negative and self.noise is not None and self.fall > self.noise  # type: ignore[operator]

    @property
    def measured(self) -> str:
        return (f"quality {self.before:.3g} -> {self.after:.3g}"
                if self.before is not None and self.after is not None else "quality not logged")

    def describe(self) -> str:
        return f"round {self.round}: {self.edit_id} ({self.action} {self.rule_id}) was reverted, {self.measured}"


@dataclass
class Parked:
    version: str
    rules: dict[str, list[FailedTry]] = field(default_factory=dict)  # rule_id -> the failed tries
    patterns: dict[str, list[FailedTry]] = field(default_factory=dict)  # pattern identity -> the failed tries


def parse_park_after(value: str) -> Optional[int]:
    """5.8: a whole number of 1 or more, or None when `value` is not one. At most 9 digits, so int() never raises."""
    return int(value) if re.fullmatch(r"[0-9]{1,9}", value) and int(value) >= 1 else None


def park_after() -> int:
    """5.8: PUDDLES_PARK_AFTER, the failed tries that park; 2 when unset (or invalid: run_loop refuses those)."""
    value = os.environ.get(PARK_AFTER)
    return (parse_park_after(value) if value is not None else None) or 2


def _is_parked(tries: list[FailedTry]) -> bool:
    return len(tries) >= park_after() or any(t.negative for t in tries)


def _all_failed_tries(log: DecisionLog) -> list[FailedTry]:
    out = []
    for r in log.rounds():
        if r.outcome != "reverted" or r.edit is None:
            continue
        before = r.scores.before.quality if r.scores else None
        after = r.scores.after.quality if r.scores else None
        out.append(FailedTry(round=r.round, edit_id=r.edit.id, action=r.edit.action, rule_id=r.edit.rule_id,
                             pattern_id=r.pattern.pattern_id if r.pattern else None,
                             before=before, after=after, base=r.policy.base,
                             noise=r.scores.noise if r.scores else None, rule_text=r.edit.rule_text))
    return out


def failed_tries(log: DecisionLog, version: str) -> list[FailedTry]:
    return [t for t in _all_failed_tries(log) if t.base == version]


def _later_versions(log: DecisionLog, base: str, from_round: int) -> set[str]:
    """`base` and every version the run moved to from it by keeps, from round `from_round` on."""
    reached = {base}
    for r in log.rounds():
        if r.round >= from_round and r.outcome == "kept" and r.policy.base in reached:
            reached.add(r.policy.in_force)
    return reached


def carried_over(log: DecisionLog, version: str) -> list[FailedTry]:
    """5.6: clearly worse failed tries of the run on an earlier version that `version` came from."""
    return [t for t in _all_failed_tries(log)
            if t.clearly_worse and t.base != version and version in _later_versions(log, t.base, t.round)]


def to_compare(log: DecisionLog, version: str) -> list[FailedTry]:
    """5.7: every failed try on `version`, plus every clearly worse failed try of the run."""
    return [t for t in _all_failed_tries(log) if t.base == version or t.clearly_worse]


def parked(log: DecisionLog, version: str) -> Parked:
    """5.2 and 5.6: the rules and patterns parked on `version`, each with the failed tries behind it."""
    tries = sorted(failed_tries(log, version) + carried_over(log, version), key=lambda t: t.round)
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


class SameEdit(BaseModel):
    """5.7: the answer of the same-idea check."""
    same_as_edit_id: Optional[str] = None
    reason: str
