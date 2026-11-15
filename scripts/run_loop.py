"""Run the overnight loop: one training round after another until a cap stops it.

    python -m scripts.run_loop --root runs/overnight --fake --max-rounds 3
    python -m scripts.run_loop --root runs/overnight --check

--fake uses scripted chat clients, a scripted model, a fake search, in-process memory and a
scripted judge: no keys, no network. Without it the clients come from HELPER_*, SIMUSER_*,
OPENAI_API_KEY, PUDDLES_AGENT_MODEL, EXA_API_KEY, PAST_API_KEY and the PUDDLES_JUDGE_* variables,
and the judge must be frozen (calibration/frozen.json). The gate is a labelled stand-in until
lane B's puddles/gate.py lands; --gate real says so and exits.

Caps come from the [loop] table of the run's config file (puddles.config.load_loop_config);
--max-rounds overrides max_rounds for a test run. --check verifies the root, the log, the
policies and the prices, then exits without running a round.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NoReturn, Sequence

from openai import OpenAIError
from pydantic import ValidationError

from loop.errors import ConfigError, PolicyError
from loop.llm import ChatClient, FakeClient
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.baselines import load_id_file, resolve_ids
from puddles.config import LoopConfig, load_config, load_loop_config
from puddles.decision_log import DecisionLog
from puddles.eval.data import load_scenarios as load_esconv
from puddles.eval.judge import JudgeResult, Usage
from puddles.graph import RoundDeps, check_policy_file, run_round
from puddles.llm import LLM, AgentFailed
from puddles.loop import LoopError, check_id_files, judge_models, run_loop, scoring_judge
from puddles.memory import FakeMemory, Memory, PastError
from puddles.round import prepare_round, stand_in_gate
from puddles.usage import Meter, spend
from schemas import ConversationRecord, RoundEvent, Scenario, ScoreRecord


class _FakeJudge:
    """A scripted judge for --fake: fixed scores and one fake-usage entry per conversation."""

    version = "judge-fake-v0"

    def score(self, record: ConversationRecord) -> JudgeResult:
        helper_turns = sum(1 for t in record.turns if t.speaker == "helper")
        score = ScoreRecord(
            id=f"score-{record.id}",
            written_by="judge",
            created_at=datetime.now(timezone.utc),
            parent_ids=[record.id],
            conversation_id=record.id,
            judge_version=self.version,
            empathy=3,
            relevance=3,
            emotion_before=4,
            emotion_after=2,
            strategies=["reflection"] * helper_turns,
            note="fake judge",
        )
        usage = [Usage(stage="single", judge="fake", model="fake", prompt_tokens=10, completion_tokens=5)]
        return JudgeResult(score=score, verdicts=[], usage=usage)


def _print_error(message: str) -> None:
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = _Parser(prog="python -m scripts.run_loop", description=__doc__.splitlines()[0])
    p.add_argument(
        "--root", default="runs/overnight", help="where log/, records/, conversations/, policies/ and scores/ go"
    )
    p.add_argument("--policy", default="policies/policy-v1.md", help="the starting policy")
    p.add_argument(
        "--scenarios",
        default="fixtures/scenarios.jsonl",
        help="fake mode's training pool (real mode draws from ESConv train)",
    )
    p.add_argument("--max-rounds", type=int, default=None, help="override loop.max_rounds")
    p.add_argument(
        "--gate",
        choices=["kept", "reverted", "real"],
        default="reverted",
        help="what the stand-in gate decides; 'real' is not built yet (lane B)",
    )
    p.add_argument("--fake", action="store_true", help="scripted clients, model, search, memory and judge")
    p.add_argument("--check", action="store_true", help="verify the root, log, policies and prices, then exit")
    p.add_argument("--mlflow", default=None, help="MLflow tracking address")
    return p.parse_args(argv)


def _train_scenarios(cfg: LoopConfig) -> list[Scenario]:
    """Real mode: the ESConv training scenarios named by the training ID file.

    The validation IDs are resolved too, so a bad file fails before the first call.
    """
    resolve_ids(load_esconv("validation"), load_id_file(cfg.validation_ids))
    return resolve_ids(load_esconv("train"), load_id_file(cfg.train_ids))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.gate == "real":
        _print_error(
            "--gate real is not built yet: puddles/gate.py is lane B's. "
            "Use --gate kept or --gate reverted for the labelled stand-in."
        )
        return 2
    root = Path(args.root)
    try:
        cfg = load_loop_config()
        if args.max_rounds is not None:
            cfg.max_rounds = args.max_rounds
        if not args.check and cfg.max_rounds is None and cfg.spend_limit_usd is None:
            raise LoopError(
                "no cap configured: set loop.max_rounds or loop.spend_limit_usd "
                "in the config file, or pass --max-rounds"
            )
        check_id_files(cfg.validation_ids, cfg.train_ids)
        scenarios = load_scenarios(args.scenarios) if args.fake else _train_scenarios(cfg)

        helper_client: ChatClient
        user_client: ChatClient
        llm: LLM
        search: Callable[[str], list[dict]]
        memory: Memory
        meter = Meter(root, load_config().prices)
        deps_judge = None
        if args.fake:
            helper_client, user_client = FakeClient(_fake_helper), FakeClient(_fake_user)
            from scripts.run_graph_round import FakeLLM, fake_search

            llm, search, memory = FakeLLM(), fake_search, FakeMemory()
            deps_judge = scoring_judge(_FakeJudge(), meter, root)
        else:
            from loop.llm import client_from_env
            from puddles import exa
            from puddles.eval.calibrate import check_frozen
            from puddles.eval.judge import judge_from_env
            from puddles.llm import OpenAILLM
            from puddles.memory import PastClient

            helper_client, user_client = client_from_env("HELPER"), client_from_env("SIMUSER")
            if not os.environ.get("EXA_API_KEY"):
                raise ConfigError("environment variable EXA_API_KEY is not set")
            llm, search, memory = OpenAILLM(), exa.search, PastClient()
            judge = judge_from_env()
            check_frozen(judge)  # refuses an unfrozen judge before any call is made
            deps_judge = scoring_judge(judge, meter, root)
            judge_names = judge_models(judge)
            if judge_names:
                meter.prices.check(judge_names)
    except (
        ConfigError,
        LoopError,
        PolicyError,
        OSError,
        UnicodeDecodeError,
        ValidationError,
        ValueError,
        AgentFailed,
        PastError,
        OpenAIError,
    ) as exc:
        _print_error(f"{type(exc).__name__}: {exc}")
        return 2

    start_policy = Path(args.policy)
    log = DecisionLog(root)
    deps = RoundDeps(
        helper_client=helper_client,
        user_client=user_client,
        llm=llm,
        search=search,
        memory=memory,
        log=log,
        root=root,
        usage=meter,
        start_policy=start_policy,
        critic_conversations=cfg.conversations_for_critic,
    )
    if deps_judge is not None:
        deps.judge = deps_judge

    def gate(result):
        return stand_in_gate(result, args.gate)

    deps.gate = gate

    try:
        if args.check:
            check_policy_file(start_policy)
            policy_path = prepare_round(log, memory, root, start_policy)
            print(
                f"check ok: {policy_path} is in force per the log; "
                f"${spend(root / 'log' / 'usage.jsonl'):.4f} spent so far"
            )
            return 0
        tracing.setup(tracking_uri=args.mlflow)

        def run_one(policy_path: Path, drawn: list[Scenario], round_: int) -> RoundEvent:
            return run_round(deps, policy_path=policy_path, scenarios=drawn, max_turns=cfg.max_turns, seed=cfg.seed)[
                "event"
            ]

        summary = run_loop(
            cfg, log=log, memory=memory, root=root, start_policy=start_policy, scenarios=scenarios, run_one=run_one
        )
    except ConfigError as exc:
        _print_error(f"{type(exc).__name__}: {exc}")
        return 2
    print(f"stopped: {summary.stop_reason} — {len(summary.rounds)} rounds, ${summary.spend_usd:.4f} spent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
