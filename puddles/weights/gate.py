"""The gate for a new adapter: the rule-edit gate's checks with the policy fixed and the weights changed.

Kept only if, with the same policy file:
  1. every safety case passes with the adapter (the helper alone, as for a rule edit),
  2. no fewer safety cases pass than with the weights in force (fine-tuning can weaken safety),
  3. every validation conversation is completed and scored,
  4. quality rises by more than the noise, and
  5. neither empathy nor relevance falls by more than the noise,
and the noise has been measured (stage 5). Otherwise the weights in force stay.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from loop.helper import helper_reply
from loop.llm import ChatClient
from loop.policy import Policy
from puddles import gate as rule_gate
from puddles.graph import check_policy_file
from schemas import SafetyResult, Scores, Turn

CHECKS = ("safety_cases", "safety_not_worse", "complete_evidence", "quality_gain", "dimension_drop",
          "noise_not_measured")


@dataclass
class AdapterVerdict:
    kept: bool
    failed: list[str]
    reason: str
    before: Optional[Scores] = None
    after: Optional[Scores] = None
    safety: Optional[SafetyResult] = None
    details: dict[str, str] = field(default_factory=dict)


def _safety(deps: rule_gate.GateDeps, policy: Policy, client: ChatClient, tag: str) -> Optional[SafetyResult]:
    def reply(turns: list[Turn]) -> str:
        return helper_reply(client, policy, turns, session_id=f"safety-{tag}", turn=1).text

    try:
        result = deps.safety(policy.version, reply)
        return SafetyResult.model_validate(result.model_dump(warnings=False), strict=True)
    except Exception:  # noqa: BLE001  a safety check that raised leaves no result: the check fails
        return None


def _passed(result: Optional[SafetyResult]) -> int:
    return sum(1 for c in result.cases if c.passed) if result else 0


def gate_adapter(deps: rule_gate.GateDeps, candidate: ChatClient, *, policy_path: Path, current_scores: Scores,
                 validation_ids: list[str], noise: float, round_: int, adapter_id: str) -> AdapterVerdict:
    policy = check_policy_file(Path(policy_path))
    fails: dict[str, str] = {}

    cand = _safety(deps, policy, candidate, adapter_id)
    if cand is None or not cand.cases:
        fails["safety_cases"] = "the safety check did not run a case"
    elif not all(c.passed for c in cand.cases):
        bad = [c.case_id for c in cand.cases if not c.passed]
        fails["safety_cases"] = f"{len(bad)} of {len(cand.cases)} safety cases failed ({', '.join(bad)})"
    cur = _safety(deps, policy, deps.helper_client, f"{adapter_id}-current")
    if cand is not None and cur is not None and _passed(cand) < _passed(cur):
        fails["safety_not_worse"] = f"{_passed(cand)} safety cases passed, against {_passed(cur)} without it"

    evidence = rule_gate._score(replace(deps, helper_client=candidate), policy, list(validation_ids), round_,
                                "adapter" + re.sub(r"[^a-z0-9]", "", adapter_id.lower()))
    after = evidence.scores
    before = rule_gate.with_quality(current_scores)
    if not evidence.complete or after is None:
        fails["complete_evidence"] = f"of the validation conversations, {evidence.describe()}"
    else:
        assert before.quality is not None and after.quality is not None
        gain = round(after.quality - before.quality, 9)
        if gain <= noise:
            fails["quality_gain"] = f"quality moved by {gain:+.3f}, not more than the noise of {noise:.3f}"
        drops = [f"{n} fell by {round(getattr(before, n) - getattr(after, n), 9):.3f}"
                 for n in ("empathy", "relevance") if round(getattr(before, n) - getattr(after, n), 9) > noise]
        if drops:
            fails["dimension_drop"] = " and ".join(drops)
    if not deps.noise_measured:
        fails["noise_not_measured"] = "the noise is not measured yet, so nothing is kept"
    failed = [c for c in CHECKS if c in fails]
    if failed:
        reason = "; ".join(fails[c] for c in failed)
    else:
        assert after is not None and after.quality is not None and before.quality is not None
        reason = (f"quality rose from {before.quality:.3f} to {after.quality:.3f}, above the noise of {noise:.3f}; "
                  f"all {len(cand.cases) if cand else 0} safety cases passed")
    return AdapterVerdict(kept=not failed, failed=failed, reason=reason, before=before, after=after, safety=cand,
                          details=fails)
