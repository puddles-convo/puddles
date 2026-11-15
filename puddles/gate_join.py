"""The gate joined to Ansh's judge and safety check (docs/specs/gate-join.md).

`puddles/gate.py` is given its judge and safety check as arguments and never imports
`puddles.eval`; this module is the one place that hands it his. It adds nothing to the decision:
the gate decides, his judge scores, his `run_safety` asks. His cases, his question prompt and his
yes-or-no rule are read from his code and file at run time and are not restated here.

His yes-or-no client logs its token counts and does not return them, so the safety questions'
tokens are not in `<root>/log/usage.jsonl`. His client is not changed to get them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loop.llm import ChatClient
from puddles import gate
from puddles.eval import safety as eval_safety
from puddles.judging import MeteredJudge
from schemas import SafetyCase, SafetyResult, Scenario


def safety_check(asker: eval_safety.YesNoLLM, cases: Optional[list[SafetyCase]] = None,
                 case_file: str = "safety-v1.json") -> gate.SafetyCheck:
    """The gate's safety check: one call to his `run_safety`. With `cases` None his loader reads his file."""

    def check(policy_version: str, reply: gate.Reply) -> SafetyResult:
        return eval_safety.run_safety(policy_version, reply, asker, cases=cases, case_file=case_file)

    return check


def real_gate_deps(*, helper_client: ChatClient, user_client: ChatClient, judge: Any, root: Path,
                   scenarios: list[Scenario], noise_measured: bool = False, max_turns: int = 10,
                   seed: Optional[int] = None, asker: Optional[eval_safety.YesNoLLM] = None) -> gate.GateDeps:
    """GateDeps on his judge and his safety check.

    The judge is metered, so the gate's scoring calls are in the usage log under the agent the graph
    sets (`gatekeeper`). Without `asker` his client is built from the environment; its ValueError
    (missing configuration) passes through.
    """
    metered = judge if isinstance(judge, MeteredJudge) else MeteredJudge(judge)
    if asker is None:
        asker = eval_safety.OpenAIYesNo.from_env()
    return gate.GateDeps(helper_client=helper_client, user_client=user_client, judge=metered,
                         safety=safety_check(asker), root=Path(root), scenarios=list(scenarios),
                         noise_measured=noise_measured, max_turns=max_turns, seed=seed)


def safety_models(asker: Any) -> list[str]:
    """The safety model's name, for the price check before the round starts."""
    return [asker.model]
