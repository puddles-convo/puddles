"""A kept policy is measured again: docs/specs/unattended-run.md section 4 (rules 4.1 to 4.5). All offline."""

import json

import pytest

import scripts.run_loop as driver
from puddles import gate
from puddles.gate_join import real_gate_deps
from puddles.policy import parse_policy, render_policy
from puddles.records import sha256_text
from tests.test_gate import IDS, round_result, setup
from tests.test_unattended_driver import capture_round, patch_real, write_config

KEY_FIELDS = {"policy_version", "policy_sha256", "scenario_ids", "judge_version", "scenarios_sha256", "max_turns",
              "seed", "scores"}


def promoted_file(tmp_path, candidate):
    """The text finish_round writes for the promoted policy."""
    promoted = parse_policy(candidate.read_text())
    promoted.version = "policy-v2"
    path = tmp_path / "policy-v2.md"
    path.write_text(render_policy(promoted))
    return path


def calls(deps):
    return len(deps.judge.calls), len(deps.helper_client.calls)


# ---------- 4.1 the setting ----------

def test_the_setting_is_off_by_default_and_real_gate_deps_passes_it_on(tmp_path):
    deps, _, _ = setup(tmp_path)
    assert deps.remeasure_kept is False
    common = dict(helper_client=deps.helper_client, user_client=deps.user_client, judge=deps.judge,
                  root=deps.root, scenarios=deps.scenarios, asker=object())
    assert real_gate_deps(**common).remeasure_kept is False
    assert real_gate_deps(**common, remeasure_kept=True).remeasure_kept is True


# ---------- 4.2 off is today's behaviour ----------

def test_off_a_keep_saves_the_candidates_scores_under_the_promoted_version_with_no_measured_field(tmp_path):
    deps, current, candidate = setup(tmp_path)
    the_gate = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)
    assert the_gate(round_result(deps, current, candidate)).decision == "kept"

    saved = json.loads((deps.root / "scores" / "current.json").read_text())
    assert set(saved) == KEY_FIELDS  # no "measured"
    assert saved["policy_version"] == "policy-v2" and saved["scores"]["quality"] == 4.0
    assert saved["policy_sha256"] == sha256_text(promoted_file(tmp_path, candidate).read_text())


# ---------- 4.3 / 4.4 on ----------

def test_on_a_keep_leaves_current_json_and_the_next_round_measures_the_kept_policy_afresh(tmp_path):
    deps, current, candidate = setup(tmp_path)
    deps.remeasure_kept = True
    path = deps.root / "scores" / "current.json"
    the_gate = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)
    result = round_result(deps, current, candidate)
    out = the_gate(result)

    # 4.3: the decision and the logged scores are what the round measured
    assert out.decision == "kept" and out.before.quality == 3.0 and out.after.quality == 4.0
    assert result.scores == the_gate.last_report.log_scores and result.scores.after.quality == 4.0
    # current.json is still the fresh measurement of policy-v1, not the winning scores
    before_keep = json.loads(path.read_text())
    assert before_keep["policy_version"] == "policy-v1" and before_keep["measured"] == "fresh"
    assert before_keep["scores"]["quality"] == 3.0
    text_after_keep = path.read_bytes()

    # the next round needs the scores of the policy in force: policy-v2, scored again
    promoted = promoted_file(tmp_path, candidate)
    n = calls(deps)
    scores = gate.current_scores(deps, promoted, IDS)
    assert calls(deps)[0] == n[0] + len(IDS) and calls(deps)[1] > n[1]
    assert scores.quality == 3.0  # the fresh measurement, not the 4.0 the candidate won with
    saved = json.loads(path.read_text())
    assert path.read_bytes() != text_after_keep
    assert saved["measured"] == "fresh" and saved["policy_version"] == "policy-v2"
    assert saved["policy_sha256"] == sha256_text(promoted.read_text())
    assert set(saved) == KEY_FIELDS | {"measured"}

    # later rounds reuse it
    n = calls(deps)
    assert gate.current_scores(deps, promoted, IDS) == scores
    assert calls(deps) == n


def test_on_a_matching_file_without_the_field_is_scored_again_and_one_with_it_is_reused(tmp_path):
    deps, current, _ = setup(tmp_path)
    path = deps.root / "scores" / "current.json"
    gate.current_scores(deps, current, IDS)  # off: every field of the key, no "measured"
    assert "measured" not in json.loads(path.read_text())

    deps.remeasure_kept = True
    n = calls(deps)
    gate.current_scores(deps, current, IDS)
    assert calls(deps)[0] == n[0] + len(IDS)  # counted as absent
    assert json.loads(path.read_text())["measured"] == "fresh"

    n = calls(deps)
    gate.current_scores(deps, current, IDS)
    assert calls(deps) == n  # no model call

    data = json.loads(path.read_text())  # a value other than "fresh" does not match either
    data["measured"] = "kept"
    path.write_text(json.dumps(data))
    gate.current_scores(deps, current, IDS)
    assert calls(deps)[0] == n[0] + len(IDS)


# ---------- 4.1 scripts/run_loop.py ----------

def test_a_bad_setting_is_refused_before_any_call(tmp_path, monkeypatch, capsys):
    write_config(tmp_path, monkeypatch)
    made = patch_real(monkeypatch)
    seen: list = []
    capture_round(monkeypatch, tmp_path / "root", seen)
    monkeypatch.setenv("PUDDLES_GATE_REMEASURE_KEPT", "yes")
    assert driver.main(["--root", str(tmp_path / "root"), "--gate", "real", "--max-rounds", "1"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "PUDDLES_GATE_REMEASURE_KEPT" in err and err.count("\n") == 1
    assert made == [] and seen == []
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize("value, on", [(None, False), ("0", False), ("1", True)])
def test_the_setting_reaches_the_gate(tmp_path, monkeypatch, value, on):
    write_config(tmp_path, monkeypatch)
    patch_real(monkeypatch)
    if value is None:
        monkeypatch.delenv("PUDDLES_GATE_REMEASURE_KEPT", raising=False)
    else:
        monkeypatch.setenv("PUDDLES_GATE_REMEASURE_KEPT", value)
    seen: list = []
    root = tmp_path / "root"
    capture_round(monkeypatch, root, seen)
    assert driver.main(["--root", str(root), "--gate", "real", "--max-rounds", "1",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]) == 0
    assert seen[0][0].gate.deps.remeasure_kept is on


@pytest.mark.parametrize("value, on", [("0", False), ("1", True)])
def test_the_setting_reaches_the_fake_gate_too(tmp_path, monkeypatch, value, on):
    monkeypatch.setenv("PUDDLES_GATE_REMEASURE_KEPT", value)
    seen: list = []
    root = tmp_path / "root"
    capture_round(monkeypatch, root, seen)
    assert driver.main(["--root", str(root), "--fake", "--gate", "real", "--max-rounds", "1",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]) == 0
    assert seen[0][0].gate.deps.remeasure_kept is on
