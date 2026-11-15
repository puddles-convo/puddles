"""Weight training in the loop (docs/specs/weight-training.md). All offline: FakeBackend, scripted clients."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import scripts.run_loop as driver
from loop.errors import ConfigError
from loop.llm import FakeClient
from loop.run_batch import _fake_user
from loop.runner import load_scenarios
from puddles import gate as rule_gate
from puddles.decision_log import DecisionLog
from puddles.records import sha256_text
from puddles.weights import ledger
from puddles.weights.backend import FakeBackend
from puddles.weights.data import Candidate, Group, advantages, build_groups, dpo_loss, dpo_pairs, grpo_groups
from puddles.weights.safe_lora import project_adapter, project_layer, projection
from puddles.weights.serving import FakeAdapterLoader
from puddles.weights.step import WeightSettings, WeightStep, load_weight_settings, pick_method, trigger
from schemas import ConversationRecord, LogPolicy, RoundEvent, SafetyCaseResult, SafetyResult, ScoreRecord, Turn

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")


class _Judged:
    def __init__(self, score):
        self.score = score


class LengthJudge:
    """Scores the last helper reply by its length: longer replies score higher (capped at 5)."""

    version = "judge-length"

    def score(self, record):
        last = [t.text for t in record.turns if t.speaker == "helper"][-1]
        s = max(1, min(5, len(last.split()) // 3))
        return _Judged(ScoreRecord(id=f"score-{record.id}", written_by="judge", created_at=datetime.now(timezone.utc),
                                   parent_ids=[record.id], conversation_id=record.id, judge_version=self.version,
                                   empathy=s, relevance=s, emotion_before=3, emotion_after=3, strategies=[],
                                   note="length"))


def passing_safety(policy_version, reply):
    reply([Turn(speaker="user", text="i want to die")])
    return SafetyResult(policy_version=policy_version, passed=True,
                        cases=[SafetyCaseResult(case_id="s1", passed=True)])


def record(i=1, turns=4):
    ts = [Turn(speaker="user" if k % 2 == 0 else "helper", text=f"line {k}") for k in range(turns)]
    return ConversationRecord(id=f"conv-{i}", written_by="loop", created_at=datetime.now(timezone.utc),
                              source="simulated", scenario_id="fixture-0001", split="train",
                              policy_version="policy-v1", turns=ts)


def round_event(n, outcome="reverted"):
    return RoundEvent(round=n, outcome="no_pattern" if outcome == "no_pattern" else outcome, reason="test",
                      policy=LogPolicy(base="policy-v1", base_sha256=sha256_text(POLICY.read_text()),
                                       in_force="policy-v1"))


# ---------- data ----------


def test_advantages_are_group_normalised_and_zero_for_a_tie():
    assert advantages([3, 3, 3]) == [0.0, 0.0, 0.0]
    a = advantages([1, 3])
    assert a == pytest.approx([-1.0, 1.0])


def test_dpo_loss_at_zero_margin_is_log2_with_opposite_gradients():
    loss, g_c, g_r = dpo_loss(-10, -12, -10, -12, beta=0.1)
    assert loss == pytest.approx(math.log(2))
    assert g_c == pytest.approx(-0.05) and g_r == pytest.approx(0.05)
    better, _, _ = dpo_loss(-8, -12, -10, -12, beta=0.1)  # chosen rose against the reference
    assert better < loss


def test_groups_pairs_and_grpo_groups_from_a_length_judge():
    groups = build_groups([record()], "system", FakeBackend(), LengthJudge(), prompts=5, samples=4, seed=1)
    assert len(groups) == 2  # two helper turns after the first
    for g in groups:
        assert all(c.quality is not None for c in g.candidates)
    pairs = dpo_pairs(groups, min_margin=0.5)
    assert len(pairs) == 2
    assert all(p.chosen.quality == 4 and p.rejected.quality == 1 for p in pairs)
    assert len(grpo_groups(groups)) == 2


def test_a_tied_group_gives_no_pair_and_no_grpo_group():
    g = Group("c", 1, "s", [], candidates=[Candidate("a", quality=3.0), Candidate("b", quality=3.0)])
    assert dpo_pairs([g], 0.5) == [] and grpo_groups([g]) == []


# ---------- Safe LoRA ----------


def test_safe_lora_leaves_an_aligned_layer_and_projects_an_orthogonal_one():
    v = np.diag([1.0, 0.0])  # the aligned direction is the first output dimension
    a = np.array([[1.0, 0.0]])
    aligned_b, sim, projected = project_layer(a, np.array([[1.0], [0.0]]), v, tau=0.5)
    assert not projected and sim == pytest.approx(1.0)
    b = np.array([[1.0], [1.0]])
    new_b, sim, projected = project_layer(a, b, v, tau=0.9)
    assert projected and sim < 0.9
    assert np.allclose(new_b, projection(v) @ b)
    out, report = project_adapter({"q": (a, b), "k": (a, b)}, {"q": v}, tau=0.9)
    assert np.allclose(out["k"][1], b) and [r.name for r in report] == ["q"]


# ---------- trigger and method ----------


def test_trigger_needs_k_rounds_without_a_keep_and_respects_the_budget(tmp_path):
    log, s = DecisionLog(tmp_path), WeightSettings(rounds_without_keep=3, max_training_runs=1)
    for n, outcome in enumerate(["reverted", "kept", "reverted", "reverted"], start=1):
        log.append(round_event(n, outcome))
    assert trigger(log, tmp_path, s) is None
    log.append(round_event(5))
    assert "3 rounds" in trigger(log, tmp_path, s)
    ledger.append(tmp_path, ledger.AdapterEvent(after_round=5, adapter_id="adapter-01", method="dpo", trigger="t",
                                                base_model="b", policy_version="policy-v1", outcome="reverted",
                                                reason="r"))
    assert trigger(log, tmp_path, s) is None  # budget spent


def test_auto_picks_grpo_only_after_a_reverted_dpo_adapter(tmp_path):
    s = WeightSettings(method="auto")
    assert pick_method(s, tmp_path) == "dpo"
    ledger.append(tmp_path, ledger.AdapterEvent(after_round=3, adapter_id="adapter-01", method="dpo", trigger="t",
                                                base_model="b", policy_version="policy-v1", outcome="reverted",
                                                reason="r"))
    assert pick_method(s, tmp_path) == "grpo"
    assert pick_method(WeightSettings(method="dpo"), tmp_path) == "dpo"


def test_weight_settings_refuse_unknown_keys_and_wrong_types(tmp_path):
    p = tmp_path / "run.toml"
    p.write_text("[weights]\nmethod = \"grpo\"\nprompts = 8\ndpo_beta = 1\n")
    s = load_weight_settings(p)
    assert (s.method, s.prompts, s.dpo_beta) == ("grpo", 8, 1.0)
    p.write_text("[weights]\nlearning = 1\n")
    with pytest.raises(ConfigError):
        load_weight_settings(p)
    p.write_text("[weights]\nprompts = \"x\"\n")
    with pytest.raises(ConfigError):
        load_weight_settings(p)


# ---------- the whole step ----------


def _setup(tmp_path, candidate_reply, *, noise_measured=True, needs_export=True):
    root = tmp_path / "root"
    log = DecisionLog(root)
    for n in (1, 2, 3):
        log.append(round_event(n))
    conv = root / "conversations" / "round-03-train.jsonl"
    conv.parent.mkdir(parents=True)
    conv.write_text(record().model_dump_json() + "\n")
    current = FakeClient(lambda s, m: "Okay.")
    deps = rule_gate.GateDeps(helper_client=current, user_client=FakeClient(_fake_user), judge=LengthJudge(),
                              safety=passing_safety, root=root, scenarios=SCENARIOS, noise_measured=noise_measured,
                              max_turns=3, seed=1)

    class Loader(FakeAdapterLoader):
        def load(self, name, sampler_path, adapter_dir):
            super().load(name, sampler_path, adapter_dir)
            return FakeClient(lambda s, m: candidate_reply)

    in_use = {}

    def use_helper(client, name):
        in_use["name"] = name
        deps.helper_client = client

    backend = FakeBackend()
    settings = WeightSettings(method="dpo", base_model="fake-base", prompts=4)
    step = WeightStep(settings=settings, backend_factory=lambda: backend, loader=Loader(needs_export), gate_deps=deps,
                      log=log, root=root, start_policy=POLICY, validation_ids=["fixture-0001", "fixture-0002"],
                      noise=0.2, seed=7, use_helper=use_helper, out=lambda s: None)
    return root, step, in_use, backend


def test_a_better_adapter_is_kept_put_in_front_and_restored_after_a_restart(tmp_path):
    long_reply = "It sounds like this has been weighing on you for a long while now. What feels heaviest today?"
    root, step, in_use, backend = _setup(tmp_path, long_reply)
    event = step.after_round(3)
    assert event.outcome == "kept", event.reason
    assert backend.trained == [("dpo", event.pairs)] and event.pairs > 0
    assert in_use["name"] == event.served_as and event.quality_after > event.quality_before
    assert (event.safety_passed, event.safety_total) == (1, 1)
    assert ledger.in_force(root).adapter_id == "adapter-01" and ledger.check(root) == []
    assert not rule_gate.current_path(root).exists()  # rule edits are measured again on the new weights
    in_use.clear()
    assert step.restore() == "adapter-01" and in_use["name"] == event.served_as
    (root / event.adapter_dir / "adapter_model.safetensors").write_bytes(b"changed")
    assert ledger.check(root)
    with pytest.raises(ConfigError, match="adapter log check failed"):
        step.restore()


def test_no_better_adapter_is_reverted_and_the_next_run_starts_from_the_base(tmp_path):
    root, step, in_use, backend = _setup(tmp_path, "Okay.")
    event = step.after_round(3)
    assert event.outcome == "reverted" and "quality_gain" in event.failed_checks
    assert in_use == {} and ledger.in_force(root) is None
    assert backend.started_from is None


def test_unmeasured_noise_keeps_nothing(tmp_path):
    long_reply = "It sounds like this has been weighing on you for a long while now. What feels heaviest today?"
    root, step, _, _ = _setup(tmp_path, long_reply, noise_measured=False)
    event = step.after_round(3)
    assert event.outcome == "reverted" and event.failed_checks == ["noise_not_measured"]


def test_a_crashing_trainer_is_recorded_as_failed(tmp_path):
    root, step, _, _ = _setup(tmp_path, "Okay.")

    def broken():
        raise RuntimeError("tinker is down")

    step.backend_factory = broken
    event = step.after_round(3)
    assert event.outcome == "failed" and "tinker is down" in event.reason
    assert json.loads((root / "log" / "adapters.jsonl").read_text())["outcome"] == "failed"


# ---------- the driver ----------


def test_weights_need_the_real_gate(tmp_path, capsys):
    assert driver.main(["--root", str(tmp_path), "--fake", "--weights", "dpo", "--max-rounds", "1"]) == 2
    assert "--gate real" in capsys.readouterr().err


def test_the_fake_driver_runs_a_training_step_after_three_rounds_without_a_keep(tmp_path):
    root = tmp_path / "root"
    code = driver.main(["--root", str(root), "--fake", "--gate", "real", "--weights", "auto", "--max-rounds", "3",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"])
    assert code == 0
    events = ledger.read(root)
    assert len(events) == 1 and events[0].after_round == 3 and events[0].method == "dpo"
    # the scripted judge gives every reply the same score, so there is nothing to train on
    assert events[0].outcome == "no_data"


def test_a_kept_adapter_has_a_name_unique_to_its_bytes(tmp_path):
    long_reply = "It sounds like this has been weighing on you for a long while now. What feels heaviest today?"
    root, step, in_use, _ = _setup(tmp_path, long_reply)
    event = step.after_round(3)
    assert event.served_as == f"adapter-01-{event.adapter_sha256[:12]}" == in_use["name"]


def test_a_failure_after_loading_unloads_the_candidate(tmp_path):
    root, step, _, _ = _setup(tmp_path, "Okay.")
    step.validation_ids = ["no-such-scenario"]  # the gate's scoring raises after the adapter is loaded
    event = step.after_round(3)
    assert event.outcome == "failed" and step.loader.loaded == []


def test_a_restart_without_weights_is_refused_while_an_adapter_is_in_force(tmp_path, capsys):
    ledger.append(tmp_path, ledger.AdapterEvent(after_round=3, adapter_id="adapter-01", method="dpo", trigger="t",
                                                base_model="b", policy_version="policy-v1", outcome="kept",
                                                reason="r"))
    assert driver.main(["--root", str(tmp_path), "--fake", "--max-rounds", "4"]) == 2
    assert "adapter in force" in capsys.readouterr().err


def test_the_weight_step_is_metered(tmp_path):
    from puddles.usage import Meter, current

    seen = []
    run = driver._metered(lambda r: seen.append(current()), Meter(tmp_path))
    run(3)
    assert seen[0][1] == "weights" and seen[0][0] is not None


def test_served_from_tinker_nothing_is_exported_and_the_checkpoint_is_the_model(tmp_path):
    long_reply = "It sounds like this has been weighing on you for a long while now. What feels heaviest today?"
    root, step, in_use, _ = _setup(tmp_path, long_reply, needs_export=False)
    event = step.after_round(3)
    assert event.outcome == "kept", event.reason
    assert event.adapter_dir is None and event.adapter_sha256 is None
    assert event.served_as == event.sampler_path == "fake://adapter-01/sampler" == in_use["name"]
    assert not (root / "adapters" / "adapter-01" / "peft").exists()
    in_use.clear()
    assert step.restore() == "adapter-01" and in_use["name"] == event.sampler_path


def test_the_tinker_loader_asks_tinker_for_the_checkpoint(monkeypatch):
    from puddles.eval.judge import TINKER_BASE_URL
    from puddles.weights.serving import TinkerAdapterLoader

    monkeypatch.setenv("TINKER_API_KEY", "k")
    monkeypatch.setenv("PUDDLES_TINKER_EXTRA_BODY", '{"reasoning_effort": false}')
    monkeypatch.delenv("PUDDLES_TINKER_BASE_URL", raising=False)
    loader = TinkerAdapterLoader.from_env()
    client = loader.load("x", "tinker://run-1/sampler_weights/adapter-01", None)
    path = "tinker://run-1/sampler_weights/adapter-01"
    assert (client.base_url, client.model, client.api_key) == (TINKER_BASE_URL, path, "k")
    assert client.extra_body == {"reasoning_effort": False}
    with pytest.raises(RuntimeError):
        loader.load("x", None, None)
    monkeypatch.delenv("TINKER_API_KEY")
    with pytest.raises(ConfigError):
        TinkerAdapterLoader.from_env()


def test_serve_must_be_tinker_or_vllm(tmp_path):
    p = tmp_path / "run.toml"
    p.write_text("[weights]\nserve = \"modal\"\n")
    with pytest.raises(ConfigError):
        load_weight_settings(p)


def test_a_restored_adapter_is_priced_under_the_model_name_it_is_called_by(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from puddles.usage import Meter, ModelPrice, Prices

    root = tmp_path / "root"
    ledger.append(root, ledger.AdapterEvent(after_round=3, adapter_id="adapter-01", method="dpo", trigger="t",
                                            base_model="b", policy_version="policy-v1", outcome="kept", reason="r",
                                            sampler_path="tinker://run/sampler_weights/adapter-01",
                                            served_as="adapter-01-oldvllmname"))
    monkeypatch.setenv("TINKER_API_KEY", "k")
    cfg_path = tmp_path / "run.toml"
    cfg_path.write_text('[weights]\nserve = "tinker"\nbase_model = "Qwen/Qwen3.5-9B"\n')
    monkeypatch.setenv("PUDDLES_CONFIG", str(cfg_path))
    helper = SimpleNamespace(model="helper-base")
    meter = Meter(root, Prices(models={"helper-base": ModelPrice(1.0, 2.0)}))
    deps = SimpleNamespace(helper_client=helper)
    gate_deps = SimpleNamespace(helper_client=helper)
    args = SimpleNamespace(weights="dpo", fake=False)
    from puddles.config import LoopConfig

    driver._weight_step(args, LoopConfig(), deps, gate_deps, meter, helper, DecisionLog(root), root, POLICY, [])
    assert deps.helper_client.model == "tinker://run/sampler_weights/adapter-01"
    assert meter.prices.models["tinker://run/sampler_weights/adapter-01"] == ModelPrice(1.0, 2.0)
