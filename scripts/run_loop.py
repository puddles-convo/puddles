"""Run the overnight loop: one training round after another until a cap stops it.

    python -m scripts.run_loop --root runs/overnight --fake --max-rounds 3
    python -m scripts.run_loop --root runs/overnight --check

--fake uses scripted chat clients, a scripted model, a fake search, in-process memory and a
scripted judge: no keys, no network. Without it the clients come from HELPER_*, SIMUSER_*,
OPENAI_API_KEY, PUDDLES_AGENT_MODEL, EXA_API_KEY, PAST_API_KEY and the PUDDLES_JUDGE_* variables;
a panel judge must be frozen (calibration/frozen.json), and loop.judge_version, when set, must be
the judge's version. The rules are docs/specs/unattended-run.md section 1.

--gate real is the gatekeeper in puddles/gate.py, built as scripts/run_graph_round.py builds it, on
the validation scenarios named by loop.validation_ids (the scripted pool with --fake, where the
safety check is a stand-in). Without --fake it needs loop.noise_measured and a loop.noise above 0.
--gate kept and --gate reverted are labelled stand-ins: without --fake, kept is refused and
reverted is a pilot that keeps nothing.

Caps come from the [loop] table of the run's config file (puddles.config.load_loop_config);
--max-rounds overrides max_rounds for a test run. --check verifies the root, the log, the
policies and the prices, then exits without running a round. Every stop is written to
log/loop.jsonl; the exit code is 0 for a cap and 3 for a stop caused by failures.
"""

from __future__ import annotations

import argparse
import math
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
from puddles import gate as real_gate
from puddles import parking, tracing
from puddles.baselines import load_id_file, resolve_ids
from puddles.config import LoopConfig, load_config, load_loop_config
from puddles.decision_log import DecisionLog
from puddles.eval.data import load_scenarios as load_esconv
from puddles.eval.judge import JudgeResult, Usage
from puddles.graph import RoundDeps, check_policy_file, run_round
from puddles.judging import MeteredJudge
from puddles.llm import LLM, AgentFailed
from puddles.loop import LoopError, check_id_files, judge_models, run_loop, scoring_judge
from puddles.memory import FakeMemory, Memory, PastError
from puddles.policy import TOKEN_CAP
from puddles.round import prepare_round, stand_in_gate
from puddles.usage import Meter, metering, model_name, spend
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
        help="kept or reverted: what the stand-in gate decides (kept only with --fake). real: the gatekeeper",
    )
    p.add_argument(
        "--weights",
        choices=["off", "dpo", "grpo", "auto"],
        default="off",
        help="post-train the helper between rounds (needs --gate real): dpo, grpo, or auto (DPO first, "
        "GRPO after a reverted DPO adapter). docs/specs/weight-training.md",
    )
    p.add_argument("--fake", action="store_true", help="scripted clients, model, search, memory and judge")
    p.add_argument("--check", action="store_true", help="verify the root, log, policies and prices, then exit")
    p.add_argument("--mlflow", default=None, help="MLflow tracking address")
    return p.parse_args(argv)


def _esconv_scenarios(cfg: LoopConfig) -> tuple[list[Scenario], list[Scenario]]:
    """Real mode: the ESConv training and validation scenarios named by the two ID files.

    The validation IDs are resolved here, so a bad file fails before the first call; they are
    the real gate's validation scenarios.
    """
    validation = resolve_ids(load_esconv("validation"), load_id_file(cfg.validation_ids))
    return resolve_ids(load_esconv("train"), load_id_file(cfg.train_ids)), validation


def _check_noise(cfg: LoopConfig) -> None:
    """Spec 1.4: no improvement run on unmeasured noise."""
    noise = cfg.noise
    if not cfg.noise_measured or noise is None or not math.isfinite(noise) or noise <= 0:
        raise LoopError(
            f"--gate real without --fake needs loop.noise_measured = true and loop.noise above 0 "
            f"(noise_measured={cfg.noise_measured}, noise={noise}): the gatekeeper would revert every candidate"
        )


def _check_judge(judge, cfg: LoopConfig) -> None:
    """Spec 1.5: a panel judge must be frozen; loop.judge_version, when set, must be the judge's."""
    from puddles.eval.calibrate import check_frozen

    if getattr(judge, "kind", None) == "panel":
        check_frozen(judge)  # refuses an unfrozen panel before any call is made
    if cfg.judge_version is not None and cfg.judge_version != judge.version:
        raise LoopError(
            f"loop.judge_version is {cfg.judge_version} but the judge is {judge.version}: "
            "the baseline and the noise were measured with another judge"
        )


def _remeasure_kept() -> bool:
    """Spec 4.1: PUDDLES_GATE_REMEASURE_KEPT is 1 (on), 0 or unset (off); anything else is refused."""
    value = os.environ.get("PUDDLES_GATE_REMEASURE_KEPT")
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ConfigError(f"environment variable PUDDLES_GATE_REMEASURE_KEPT is {value!r}: it must be 1, 0 or unset")


def _check_park_reverted() -> None:
    """Spec 5.1: PUDDLES_PARK_REVERTED is 1 (on), 0 or unset (off); anything else is refused.
    The agents read it themselves (puddles.parking.enabled)."""
    value = os.environ.get("PUDDLES_PARK_REVERTED")
    if value not in (None, "0", "1"):
        raise ConfigError(f"environment variable PUDDLES_PARK_REVERTED is {value!r}: it must be 1, 0 or unset")
    after = os.environ.get("PUDDLES_PARK_AFTER")  # spec 5.8; read by puddles.parking.park_after
    if after is not None and parking.parse_park_after(after) is None:
        raise ConfigError(f"environment variable PUDDLES_PARK_AFTER is {after!r}: "
                          f"it must be a whole number of 1 or more, or unset")


def _weight_step(args, cfg: LoopConfig, deps: RoundDeps, gate_deps, meter: Meter, helper_client: ChatClient,
                 log: DecisionLog, root: Path, start_policy: Path, validation_ids: list[str]):
    """The between-rounds training step, with the adapter in force (after a restart) put back in front."""
    from puddles.config import DEFAULT_PATH
    from puddles.weights.backend import Backend, FakeBackend, TinkerBackend
    from puddles.weights.serving import AdapterLoader, FakeAdapterLoader, VLLMAdapterLoader
    from puddles.weights.step import WeightStep, load_weight_settings

    settings = load_weight_settings(Path(os.environ.get("PUDDLES_CONFIG") or DEFAULT_PATH))
    if args.weights != "auto":
        settings.method = args.weights
    factory: Callable[[], Backend]
    loader: AdapterLoader
    if args.fake:
        factory, loader = FakeBackend, FakeAdapterLoader()
    else:
        if not settings.base_model:
            raise ConfigError("weights.base_model is not set in the config file")
        if not os.environ.get("TINKER_API_KEY"):
            raise ConfigError("environment variable TINKER_API_KEY is not set")
        def factory() -> Backend:
            return TinkerBackend(settings.base_model, settings.rank)

        loader = VLLMAdapterLoader.from_env(helper_client)  # type: ignore[arg-type]
    base_price = meter.prices.models.get(model_name(helper_client))

    def use_helper(client: ChatClient, name: str) -> None:
        deps.helper_client = client
        gate_deps.helper_client = tracing.TracedClient(client, "helper")
        if base_price is not None:  # the adapter is priced as the model it was trained from
            meter.prices.models[name] = base_price

    step = WeightStep(settings=settings, backend_factory=factory, loader=loader, gate_deps=gate_deps, log=log,
                      root=root, start_policy=start_policy, validation_ids=validation_ids,
                      noise=cfg.noise if cfg.noise is not None else 0.0, seed=cfg.seed, use_helper=use_helper)
    restored = step.restore()
    if restored:
        print(f"weights: {restored} is in force (from log/adapters.jsonl)")
    return step


def _metered(step: Callable[[int], object], meter: Meter) -> Callable[[int], object]:
    """The weight step with its judge calls in the usage log (under `weights`), so the spend limit sees them."""

    def run(round_: int) -> object:
        with metering(meter, "weights"):
            return step(round_)

    return run


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.gate == "kept" and not args.fake:
        _print_error("--gate kept needs --fake: a stand-in never promotes with real models")
        return 2
    if args.gate == "reverted" and not args.fake:
        print("pilot: --gate reverted without --fake is the stand-in gate: it reverts every candidate, "
              "nothing can be kept")
    if args.weights != "off" and args.gate != "real":
        _print_error("--weights needs --gate real: an adapter is kept only by the gatekeeper")
        return 2
    root = Path(args.root)
    try:
        remeasure_kept = _remeasure_kept()  # before any model call
        _check_park_reverted()
        cfg = load_loop_config()
        if args.max_rounds is not None:
            cfg.max_rounds = args.max_rounds
        if args.gate == "real" and not args.fake:
            _check_noise(cfg)
        if not args.check and cfg.max_rounds is None and cfg.spend_limit_usd is None:
            raise LoopError(
                "no cap configured: set loop.max_rounds or loop.spend_limit_usd "
                "in the config file, or pass --max-rounds"
            )
        check_id_files(cfg.validation_ids, cfg.train_ids)
        if args.fake:
            scenarios = load_scenarios(args.scenarios)
            validation = scenarios  # the gate tests on the whole scripted pool, as run_graph_round does
        else:
            scenarios, validation = _esconv_scenarios(cfg)

        helper_client: ChatClient
        user_client: ChatClient
        llm: LLM
        search: Callable[[str], list[dict]]
        memory: Memory
        meter = Meter(root, load_config().prices)
        deps_judge = None
        judge_panel = None  # real mode: the round's own adapter scores (spec 1.2)
        asker = None  # the real safety check's yes-or-no client; None with --fake
        if args.fake:
            helper_client, user_client = FakeClient(_fake_helper), FakeClient(_fake_user)
            from scripts.run_graph_round import FakeJudge, FakeLLM, fake_search

            llm, search, memory = FakeLLM(), fake_search, FakeMemory()
            deps_judge = scoring_judge(_FakeJudge(), meter, root)
            gate_judge = FakeJudge()  # reports no usage, so nothing unpriced reaches the meter
        else:
            from loop.llm import client_from_env
            from puddles import exa
            from puddles.eval import judge as eval_judge
            from puddles.llm import OpenAILLM
            from puddles.memory import PastClient

            helper_client, user_client = client_from_env("HELPER"), client_from_env("SIMUSER")
            if not os.environ.get("EXA_API_KEY"):
                raise ConfigError("environment variable EXA_API_KEY is not set")
            llm, search, memory = OpenAILLM(), exa.search, PastClient()
            judge = eval_judge.judge_from_env()
            _check_judge(judge, cfg)
            judge_panel = gate_judge = judge
            judge_names = judge_models(judge)
            if judge_names:
                meter.prices.check(judge_names)
            if args.gate == "real":
                from puddles.eval import safety as eval_safety
                from puddles.gate_join import safety_models

                asker = eval_safety.OpenAIYesNo.from_env()  # a missing PUDDLES_SAFETY_MODEL is a ValueError
                meter.prices.check(safety_models(asker))
        # The round checks these too, but inside the loop a refusal would only cost a round: checked here.
        meter.prices.check([model_name(c) for c in (helper_client, user_client, llm)],
                           search=not args.fake)
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
        judge_panel=judge_panel,
    )
    if deps_judge is not None:
        deps.judge = deps_judge

    def gate(result):
        return stand_in_gate(result, args.gate)

    deps.gate = gate
    if args.gate == "real":  # built as scripts/run_graph_round.py builds it (spec 1.1)
        traced_helper = tracing.TracedClient(helper_client, "helper")
        traced_user = tracing.TracedClient(user_client, "simulated_user")
        if asker is None:  # --fake: the safety check is a stand-in
            from scripts.run_graph_round import fake_safety

            gate_deps = real_gate.GateDeps(
                helper_client=traced_helper, user_client=traced_user, judge=MeteredJudge(gate_judge),
                safety=fake_safety, root=root, scenarios=validation, noise_measured=cfg.noise_measured,
                max_turns=cfg.max_turns, seed=cfg.seed, remeasure_kept=remeasure_kept)
        else:
            from puddles.gate_join import real_gate_deps

            gate_deps = real_gate_deps(
                helper_client=traced_helper, user_client=traced_user, judge=gate_judge, root=root,
                scenarios=validation, noise_measured=cfg.noise_measured, max_turns=cfg.max_turns,
                seed=cfg.seed, asker=asker, remeasure_kept=remeasure_kept)
        deps.gate = real_gate.make_gate(
            gate_deps, validation_scenario_ids=[s.scenario_id for s in validation],
            noise=cfg.noise if cfg.noise is not None else 0.0, token_cap=TOKEN_CAP)

    weight_step = None
    if args.weights == "off" and not args.check:
        from puddles.weights import ledger as adapter_ledger

        try:
            kept = adapter_ledger.in_force(root)
        except (OSError, ValueError) as exc:
            _print_error(f"{type(exc).__name__}: log/adapters.jsonl cannot be read: {exc}")
            return 2
        if kept is not None:
            _print_error(f"{kept.adapter_id} is the adapter in force (log/adapters.jsonl): run with --weights "
                         "to keep using it; without it the rounds would run on other weights")
            return 2
    if args.weights != "off":
        try:
            weight_step = _weight_step(args, cfg, deps, gate_deps, meter, helper_client, log, root, start_policy,
                                       [s.scenario_id for s in validation])
        except (ConfigError, OSError, ValueError, RuntimeError) as exc:
            _print_error(f"{type(exc).__name__}: {exc}")
            return 2

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
            cfg, log=log, memory=memory, root=root, start_policy=start_policy, scenarios=scenarios, run_one=run_one,
            after_round=_metered(weight_step.after_round, meter) if weight_step is not None else None,
        )
    except ConfigError as exc:
        _print_error(f"{type(exc).__name__}: {exc}")
        return 2
    except Exception as exc:  # escaped the loop outside a round; run_loop has written the stopped line
        _print_error(f"{type(exc).__name__}: {exc}")
        return 3
    return 3 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
