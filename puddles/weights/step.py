"""One weight-training step between rounds: decide, train, export, load, gate, record.

The loop calls `WeightStep.after_round` after every round. It trains only when the trigger fires and a
training run is left in the budget, so most rounds return at once.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from loop.errors import ConfigError
from loop.llm import ChatClient
from loop.prompts import render_helper_system
from puddles import gate as rule_gate
from puddles import replay
from puddles.decision_log import DecisionLog
from puddles.graph import check_policy_file
from puddles.weights import ledger
from puddles.weights.backend import Backend
from puddles.weights.data import build_groups, dpo_pairs, grpo_groups, write_dataset
from puddles.weights.gate import gate_adapter
from puddles.weights.ledger import AdapterEvent
from puddles.weights.serving import AdapterLoader
from schemas import ConversationRecord

Method = Literal["dpo", "grpo", "auto"]


@dataclass
class WeightSettings:
    """The [weights] table of config/run.toml."""

    method: Method = "auto"  # auto: DPO first, GRPO after a DPO adapter is reverted
    base_model: str = ""  # Tinker's name for the helper's model, e.g. Qwen/Qwen3.5-9B
    rank: int = 16
    max_training_runs: int = 3
    rounds_without_keep: int = 3  # train after this many rounds in a row keep nothing
    source_rounds: int = 3  # conversations come from this many latest rounds
    prompts: int = 32  # contexts sampled per training run
    samples: int = 4  # replies per context
    min_margin: float = 0.5  # DPO: the judge must separate best and worst by this much
    dpo_beta: float = 0.1
    dpo_learning_rate: float = 1e-5
    epochs: int = 1
    grpo_learning_rate: float = 4e-5
    grpo_iterations: int = 2
    max_tokens: int = 80
    temperature: float = 0.9


def load_weight_settings(path: Path) -> WeightSettings:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8")).get("weights", {})
    s = WeightSettings()
    for key, value in data.items():
        if not hasattr(s, key):
            raise ConfigError(f"{path}: weights.{key} is not a setting")
        want = type(getattr(s, key))
        if want is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, want) or isinstance(value, bool):
            raise ConfigError(f"{path}: weights.{key} must be {want.__name__}")
        setattr(s, key, value)
    if s.method not in ("dpo", "grpo", "auto"):
        raise ConfigError(f"{path}: weights.method must be dpo, grpo or auto")
    return s


def trigger(log: DecisionLog, root: Path, s: WeightSettings) -> Optional[str]:
    """Why to train now, or None. Rounds before the last training run do not count again."""
    events = ledger.read(root)
    if len(events) >= s.max_training_runs:
        return None
    since = events[-1].after_round if events else 0
    recent = [r for r in log.rounds() if r.round > since][-s.rounds_without_keep:]
    if len(recent) == s.rounds_without_keep and all(r.outcome != "kept" for r in recent):
        return f"{s.rounds_without_keep} rounds in a row kept no rule edit"
    return None


def pick_method(s: WeightSettings, root: Path) -> Literal["dpo", "grpo"]:
    if s.method != "auto":
        return s.method
    events = ledger.read(root)
    if events and events[-1].method == "dpo" and events[-1].outcome == "reverted":
        return "grpo"
    return "dpo"


def source_records(root: Path, upto: int, n_rounds: int) -> list[ConversationRecord]:
    records = []
    for r in range(max(1, upto - n_rounds + 1), upto + 1):
        path = Path(root) / "conversations" / f"round-{r:02d}-train.jsonl"
        if path.exists():
            records += [ConversationRecord.model_validate_json(line)
                        for line in path.read_text(encoding="utf-8").splitlines() if line]
    return records


@dataclass
class WeightStep:
    settings: WeightSettings
    backend_factory: Callable[[], Backend]
    loader: AdapterLoader
    gate_deps: rule_gate.GateDeps
    log: DecisionLog
    root: Path
    start_policy: Path
    validation_ids: list[str]
    noise: float
    seed: int
    use_helper: Callable[[ChatClient, str], None]  # point the rounds and the gate at the new weights
    out: Callable[[str], None] = print
    force_method: Optional[Literal["dpo", "grpo"]] = None

    def restore(self) -> Optional[str]:
        """After a restart: put the adapter in force back in front of the helper."""
        kept = ledger.in_force(self.root)
        if kept is None or kept.adapter_dir is None:
            return None
        problems = ledger.check(self.root)
        if problems:
            raise ConfigError("adapter log check failed: " + "; ".join(problems))
        name = kept.served_as or kept.adapter_id
        self.use_helper(self.loader.load(name, self.root / kept.adapter_dir), name)
        return kept.adapter_id

    def after_round(self, round_: int) -> Optional[AdapterEvent]:
        why = trigger(self.log, self.root, self.settings)
        if why is None:
            return None
        return self.train(round_, why)

    def train(self, round_: int, why: str) -> AdapterEvent:
        s = self.settings
        method = self.force_method or pick_method(s, self.root)
        n = len(ledger.read(self.root)) + 1
        adapter_id = f"adapter-{n:02d}"
        parent = ledger.in_force(self.root)
        policy_path = replay.check(self.log, self.root, self.start_policy)
        policy = check_policy_file(policy_path)
        folder = self.root / "adapters" / adapter_id
        base: dict[str, Any] = dict(after_round=round_, adapter_id=adapter_id, method=method, trigger=why,
                    base_model=s.base_model, parent_adapter=parent.adapter_id if parent else None,
                    policy_version=policy.version)
        self.out(f"weights: {adapter_id} ({method}) after round {round_}: {why}")
        served: Optional[str] = None
        try:
            backend = self.backend_factory()
            backend.start(parent.state_path if parent else None)
            records = source_records(self.root, round_, s.source_rounds)
            system = render_helper_system(policy)
            judge = self.gate_deps.judge
            dataset = folder / "dataset.jsonl"
            groups_n = pairs_n = 0
            metrics: dict[str, float] = {}
            iterations = 1 if method == "dpo" else s.grpo_iterations
            for it in range(iterations):
                groups = build_groups(records, system, backend, judge, prompts=s.prompts, samples=s.samples,
                                      seed=self.seed * 1000 + n * 10 + it, max_tokens=s.max_tokens,
                                      temperature=s.temperature)
                groups_n += len(groups)
                if method == "dpo":
                    pairs = dpo_pairs(groups, s.min_margin)
                    write_dataset(dataset, method, groups, pairs)
                    pairs_n = len(pairs)
                    if pairs:
                        metrics = backend.train_dpo(pairs, beta=s.dpo_beta, learning_rate=s.dpo_learning_rate,
                                                    epochs=s.epochs)
                else:
                    usable = grpo_groups(groups)
                    write_dataset(dataset, method, groups)
                    pairs_n += len(usable)
                    if usable:
                        metrics = backend.train_grpo(usable, learning_rate=s.grpo_learning_rate)
            if not metrics:
                return ledger.append(self.root, AdapterEvent(
                    **base, groups=groups_n, pairs=pairs_n, outcome="no_data",
                    dataset_sha256=ledger.sha256_file(dataset) if dataset.exists() else None,
                    reason="the judge did not separate the sampled replies enough to train on"))
            saved = backend.save(adapter_id)
            peft = backend.export(saved, folder)
            adapter_sha = ledger.sha256_dir(peft)
            # Unique per adapter's bytes, so roots sharing one helper server never pick up each other's.
            served = f"{adapter_id}-{adapter_sha[:12]}"
            candidate = self.loader.load(served, peft)
            current = rule_gate.current_scores(self.gate_deps, policy_path, self.validation_ids)
            verdict = gate_adapter(self.gate_deps, candidate, policy_path=policy_path, current_scores=current,
                                   validation_ids=self.validation_ids, noise=self.noise, round_=round_,
                                   adapter_id=adapter_id)
        except Exception as exc:  # noqa: BLE001  a failed training run costs this run only
            if served is not None:
                try:
                    self.loader.unload(served)
                except Exception:  # noqa: BLE001  the original error is the one to record
                    pass
            return ledger.append(self.root, AdapterEvent(**base, outcome="failed",
                                                         reason=f"{type(exc).__name__}: {str(exc)[:300]}"))
        event = AdapterEvent(
            **base, groups=groups_n, pairs=pairs_n, dataset_sha256=ledger.sha256_file(dataset),
            sampler_path=saved.sampler_path, state_path=saved.state_path,
            adapter_dir=peft.relative_to(self.root).as_posix(), adapter_sha256=adapter_sha,
            served_as=served, metrics=metrics, outcome="kept" if verdict.kept else "reverted",
            failed_checks=verdict.failed,
            quality_before=verdict.before.quality if verdict.before else None,
            quality_after=verdict.after.quality if verdict.after else None,
            safety_passed=sum(c.passed for c in verdict.safety.cases) if verdict.safety else None,
            safety_total=len(verdict.safety.cases) if verdict.safety else None, reason=verdict.reason)
        assert served is not None
        if verdict.kept:
            self.use_helper(candidate, served)
            rule_gate.current_path(self.root).unlink(missing_ok=True)  # measured on the old weights
        else:
            self.loader.unload(served)
        self.out(f"weights: {adapter_id} {event.outcome}. {verdict.reason}")
        return ledger.append(self.root, event)
