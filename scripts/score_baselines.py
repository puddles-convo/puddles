"""Score one policy on a fixed list of scenarios, or aggregate every run under a root.

    python -m scripts.score_baselines --policy policies/policy-v1.md \
        --ids fixtures/validation_ids.txt --run 1 --seed 7 --max-turns 10
    python -m scripts.score_baselines --aggregate --root runs/baseline

A real run needs HELPER_* and SIMUSER_* for the conversation, OPENAI_API_KEY plus
PUDDLES_JUDGE_PANEL and PUDDLES_JUDGE_ASTRA for the panel judge, and a prices table in
config/run.toml that covers every model used. A panel judge must be frozen:
calibration/frozen.json must match its version (`check_frozen`). The single-model judge
(PUDDLES_JUDGE=fallback) needs no frozen file (docs/specs/unattended-run.md, rule 1.5).

--fake runs offline with scripted clients and a scripted judge: no keys, no network.
It resolves IDs against --scenarios (the fixture file by default).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Sequence

import mlflow

from loop.errors import ConfigError, PolicyError
from loop.llm import ChatClient, FakeClient, client_from_env
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.baselines import BaselineError, load_id_file, score_once, summarize
from puddles.config import load_config
from puddles.usage import Meter
from schemas import ConversationRecord, ScoreRecord


class ScriptedJudge:
    """Fixed scores for every conversation, for --fake runs and offline tests."""

    version = "judge-fake-v0"

    class _Result:
        def __init__(self, score: ScoreRecord) -> None:
            self.score = score
            self.usage: list = []

    def score(self, record: ConversationRecord) -> Any:
        helper_turns = sum(1 for turn in record.turns if turn.speaker == "helper")
        score = ScoreRecord(
            id=f"score-{record.id}", written_by="judge", created_at=datetime.now(timezone.utc),
            parent_ids=[record.id], conversation_id=record.id, judge_version=self.version,
            empathy=3, relevance=3, emotion_before=4, emotion_after=3,
            strategies=["Others"] * helper_turns, note="Scripted judge for offline runs.")
        return self._Result(score)


def _judge_models(judge: Any) -> list[str]:
    """Every model the judge can call, for the up-front prices check."""
    config = getattr(judge, "config", None)
    if config is None:
        return []
    models = list(getattr(config, "judges", None) or [getattr(config, "model", "")])
    models.append(getattr(config, "astra", ""))
    return [m for m in models if m]


def _print_error(message: str) -> None:
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = _Parser(prog="python -m scripts.score_baselines", description=__doc__.splitlines()[0])
    p.add_argument("--aggregate", action="store_true",
                   help="read every */result.json under --root and write results.json")
    p.add_argument("--policy", default="policies/policy-v1.md")
    p.add_argument("--ids", default="fixtures/validation_ids.txt",
                   help="one scenario ID per line")
    p.add_argument("--scenarios", default="fixtures/scenarios.jsonl",
                   help="scenario file used only in --fake mode")
    p.add_argument("--run", type=int, default=1, help="this run's number, for the output folder")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--root", default="runs/baseline", help="where run folders and results.json go")
    p.add_argument("--fake", action="store_true", help="scripted clients and judge: no keys")
    p.add_argument("--mlflow", default=None, help="MLflow tracking address")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    root = Path(args.root)
    if args.aggregate:
        summary = summarize(root)
        for group in summary["groups"]:
            for policy_version, entry in sorted(group["policies"].items()):
                mean = entry.get("mean_scores") or {}
                print(f"{policy_version}: runs={entry['runs']} complete={entry['complete']} "
                      f"quality={mean.get('quality')} noise={entry.get('noise', 'n/a')}")
        print(f"wrote {root / 'results.json'}")
        return 0

    current_file = None
    try:
        current_file = args.policy
        from puddles.graph import check_policy_file
        policy = check_policy_file(Path(args.policy))
        current_file = args.ids
        ids = load_id_file(args.ids)
        helper_client: ChatClient
        user_client: ChatClient
        if args.fake:
            current_file = args.scenarios
            scenarios = load_scenarios(args.scenarios)
            helper_client, user_client = FakeClient(_fake_helper), FakeClient(_fake_user)
            judge = ScriptedJudge()
            meter = Meter(root)
        else:
            from puddles.eval.calibrate import check_frozen
            from puddles.eval.data import load_scenarios as load_esconv
            from puddles.eval.judge import judge_from_env

            judge = judge_from_env()
            if getattr(judge, "kind", None) == "panel":
                check_frozen(judge)  # the fallback judge's calibration is parked: no frozen file
            scenarios = load_esconv("validation")
            helper_client, user_client = client_from_env("HELPER"), client_from_env("SIMUSER")
            meter = Meter(root, load_config().prices)
            meter.prices.check([getattr(helper_client, "model", ""), getattr(user_client, "model", ""),
                                *_judge_models(judge)])
        current_file = None
    except (ConfigError, PolicyError, OSError, ValueError, BaselineError) as exc:
        detail = str(exc)
        where = f"{current_file}: " if current_file and current_file not in detail else ""
        _print_error(f"{where}{type(exc).__name__}: {detail}")
        return 2

    tracing.setup(tracking_uri=args.mlflow)
    run_dir = root / f"{policy.version}-run-{args.run}"
    with mlflow.start_run(run_name=f"baseline-{policy.version}-run-{args.run}"):
        mlflow.set_tag("kind", "baseline")
        mlflow.log_params({"policy_version": policy.version, "run": args.run, "seed": args.seed,
                           "max_turns": args.max_turns, "scenarios": len(ids),
                           "judge_version": judge.version})
        result = score_once(policy_path=args.policy, scenarios=scenarios, ids=ids,
                            helper_client=helper_client, user_client=user_client, judge=judge,
                            meter=meter, run=args.run, seed=args.seed,
                            max_turns=args.max_turns, run_dir=run_dir)
        metrics = {"seconds": result.seconds, "cost_usd": result.cost_usd,
                   "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                   "conversations_failed": len(result.conversations_failed),
                   "unscored": len(result.unscored)}
        if result.complete:
            metrics.update({"empathy": result.empathy, "relevance": result.relevance,
                            "quality": result.quality, "emotion_change": result.emotion_change})
        mlflow.log_metrics(metrics)
    state = "complete" if result.complete else "INCOMPLETE"
    print(f"{policy.version} run {args.run}: {state}, quality={result.quality}, "
          f"cost=${result.cost_usd:.4f}, {result.seconds:.0f}s -> {run_dir}/result.json")
    return 0 if result.complete else 1


if __name__ == "__main__":
    sys.exit(main())
