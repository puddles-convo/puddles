"""Independent acceptance checks for unattended-run.md 4.1--4.5; no services."""

import json
from dataclasses import replace
from types import SimpleNamespace

import scripts.run_loop as driver
from puddles import gate
from puddles.decision_log import DecisionLog
from puddles.memory import FakeMemory
from puddles.records import sha256_text
from puddles.round import RoundResult, finish_round
from tests.test_gate import IDS, Judge, gated_round, setup
from tests.test_unattended_driver import capture_round, patch_real, write_config


def cache(deps):
    return deps.root / "scores" / "current.json"


def promote(deps, current, candidate, the_gate):
    result = gated_round(deps, current, candidate)
    outcome = the_gate(result)
    assert outcome.decision == "kept"
    event = finish_round(result, outcome, DecisionLog(deps.root), FakeMemory(), deps.root)
    return deps.root / "policies" / f"{event.policy.in_force}.md", result


def next_result(deps, current, round_=2):
    candidate = deps.root / "policies" / f"policy-v3-candidate-r{round_:02d}.md"
    candidate.write_text(current.read_text().replace("# policy-v2", "# policy-v3-candidate")
                         + "C10. Check that you understood the person's concern.\n")
    return RoundResult(round=round_, outcome="candidate", base_policy_path=current,
                       candidate_path=candidate, edit=SimpleNamespace(id=f"edit-r{round_:02d}"))


def test_off_keeps_exact_legacy_json_and_reuses_winning_scores(tmp_path):
    deps, current, candidate = setup(tmp_path)
    assert deps.remeasure_kept is False
    the_gate = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)
    promoted, result = promote(deps, current, candidate, the_gate)
    expected = dict(policy_version="policy-v2", policy_sha256=sha256_text(promoted.read_text()),
                    scenario_ids=IDS, judge_version=deps.judge.version,
                    scenarios_sha256=sha256_text("\n".join(s.model_dump_json() for s in deps.scenarios)),
                    max_turns=deps.max_turns, seed=deps.seed, scores=result.scores.after.model_dump())
    saved = cache(deps).read_bytes()
    assert saved == (json.dumps(expected, indent=2, allow_nan=False) + "\n").encode()
    n = len(deps.judge.calls)
    out = the_gate(next_result(deps, promoted))
    assert out.decision == "reverted" and out.before.quality == 4
    assert deps.judge.calls[n:] == [f"conv-r02-val-{sid}" for sid in IDS]
    assert cache(deps).read_bytes() == saved


def test_on_next_gated_round_measures_promoted_policy_and_uses_fresh_bar(tmp_path):
    def values(record):
        if record.policy_version == "policy-v1":
            return 3, 3
        if record.policy_version == "policy-v2-candidate":
            return 4, 4
        # Fresh v2 = 3.6; candidate v3 = 3.8. It beats only the fresh bar.
        high = 3 if record.policy_version == "policy-v2" else 4
        value = 4 if IDS.index(record.scenario_id) < high else 3
        return value, value

    deps, current, candidate = setup(tmp_path, judge=Judge(values))
    deps.remeasure_kept = True
    gate.current_scores(deps, current, IDS)
    before_keep = cache(deps).read_bytes()
    the_gate = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)
    promoted, first = promote(deps, current, candidate, the_gate)
    assert first.scores.before.quality == 3 and first.scores.after.quality == 4
    assert cache(deps).read_bytes() == before_keep
    n, helper_n, user_n = len(deps.judge.calls), len(deps.helper_client.calls), len(deps.user_client.calls)
    second = next_result(deps, promoted)
    out = the_gate(second)
    assert out.decision == "kept" and out.before.quality == 3.6 and out.after.quality == 3.8
    assert second.scores.before == out.before and second.scores.after == out.after
    assert deps.judge.calls[n:] == ([f"conv-r00-current-{sid}" for sid in IDS]
                                   + [f"conv-r02-val-{sid}" for sid in IDS])
    # Both baseline and validation really generated conversations.
    assert len(deps.helper_client.calls) > helper_n + len(IDS)
    assert len(deps.user_client.calls) > user_n + len(IDS)
    saved = json.loads(cache(deps).read_text())
    assert saved["measured"] == "fresh" and saved["policy_version"] == "policy-v2"
    assert saved["scores"]["quality"] == 3.6  # second keep also leaves the baseline alone
    records = [json.loads(line) for line in
               (deps.root / "conversations" / "round-00-current.jsonl").read_text().splitlines()]
    assert {r["policy_version"] for r in records} == {"policy-v2"}


def test_restart_on_distrusts_actual_legacy_keep_with_matching_key(tmp_path):
    deps, current, candidate = setup(tmp_path)
    promoted, _ = promote(deps, current, candidate,
                          gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15))
    old = json.loads(cache(deps).read_text())
    assert "measured" not in old and old["scores"]["quality"] == 4
    restarted = replace(deps, remeasure_kept=True)
    the_gate = gate.make_gate(restarted, validation_scenario_ids=IDS, noise=0.15)
    n = len(deps.judge.calls)
    out = the_gate(next_result(deps, promoted))
    assert out.decision == "kept" and out.before.quality == 3
    assert deps.judge.calls[n:n + len(IDS)] == [f"conv-r00-current-{sid}" for sid in IDS]
    fresh = json.loads(cache(deps).read_text())
    assert fresh["policy_sha256"] == old["policy_sha256"]
    assert fresh["measured"] == "fresh" and fresh["scores"]["quality"] == 3


def test_on_revert_preserves_fresh_cache_and_next_round_reuses_it(tmp_path):
    deps, current, candidate = setup(tmp_path)
    deps.max_turns = 2  # stop at the cap before the scripted user's own [END]
    deps.remeasure_kept = True
    promoted, _ = promote(deps, current, candidate,
                          gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15))
    deps.judge.rule = lambda record: (3, 3)
    the_gate = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)
    assert the_gate(next_result(deps, promoted)).decision == "reverted"
    saved = cache(deps).read_bytes()
    assert json.loads(saved)["measured"] == "fresh"
    n, helper_n, user_n = len(deps.judge.calls), len(deps.helper_client.calls), len(deps.user_client.calls)
    # Rebuilding the gate also verifies reuse after a process-style restart.
    the_gate = gate.make_gate(replace(deps), validation_scenario_ids=IDS, noise=0.15)
    assert the_gate(next_result(deps, promoted, 3)).decision == "reverted"
    assert deps.judge.calls[n:] == [f"conv-r03-val-{sid}" for sid in IDS]
    assert cache(deps).read_bytes() == saved
    assert len(deps.helper_client.calls) - helper_n == len(IDS) * deps.max_turns + 1
    assert len(deps.user_client.calls) - user_n == len(IDS) * deps.max_turns


def test_on_failed_refresh_never_certifies_legacy_winning_scores(tmp_path):
    deps, current, candidate = setup(tmp_path)
    promoted, _ = promote(deps, current, candidate,
                          gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15))
    legacy = cache(deps).read_bytes()
    deps.remeasure_kept = True

    def unavailable(record):
        raise RuntimeError("judge temporarily unavailable")

    deps.judge.rule = unavailable
    the_gate = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)
    result = next_result(deps, promoted)
    out = the_gate(result)
    assert out.decision == "reverted" and result.scores is None
    assert cache(deps).read_bytes() == legacy
    assert not (deps.root / "scores" / "round-02-validation.jsonl").exists()
    deps.judge.rule = lambda record: (3, 3)
    n = len(deps.judge.calls)
    assert the_gate(next_result(deps, promoted, 3)).before.quality == 3
    assert deps.judge.calls[n:n + len(IDS)] == [f"conv-r00-current-{sid}" for sid in IDS]
    assert json.loads(cache(deps).read_text())["measured"] == "fresh"


def test_driver_rejects_bad_env_before_model_construction(tmp_path, monkeypatch, capsys):
    write_config(tmp_path, monkeypatch)
    made = patch_real(monkeypatch)
    seen = []
    capture_round(monkeypatch, tmp_path / "root", seen)
    for value in ("", "yes", "true", "2", " 1", "1\n"):
        monkeypatch.setenv("PUDDLES_GATE_REMEASURE_KEPT", value)
        assert driver.main(["--root", str(tmp_path / "root"), "--gate", "real", "--max-rounds", "1"]) == 2
        assert "PUDDLES_GATE_REMEASURE_KEPT" in capsys.readouterr().err
    assert made == [] and seen == [] and not (tmp_path / "root").exists()


def test_driver_passes_unset_zero_and_one_through_real_gate_join(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    patch_real(monkeypatch)
    for value, expected in ((None, False), ("0", False), ("1", True)):
        if value is None:
            monkeypatch.delenv("PUDDLES_GATE_REMEASURE_KEPT", raising=False)
        else:
            monkeypatch.setenv("PUDDLES_GATE_REMEASURE_KEPT", value)
        seen = []
        root = tmp_path / f"run-{value}"
        capture_round(monkeypatch, root, seen)
        assert driver.main(["--root", str(root), "--gate", "real", "--max-rounds", "1",
                            "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]) == 0
        deps = seen[0][0].gate.deps
        assert deps.remeasure_kept is expected
        assert deps.safety.__qualname__.startswith("safety_check")
