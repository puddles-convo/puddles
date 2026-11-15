"""A safety check that ran no case did not run (docs/specs/gate.md, section 2, step 2).

An empty `cases` list fails `safety_cases` with the reason "no safety case was run", whatever the
result's `passed` says. Nothing is kept, the result is saved as it came back, and the gate's own
statements about it (`safety_passed`, `log_safety.passed`) are false. All offline.
"""
import json

import pytest

from puddles import gate
from puddles.decision_log import DecisionLog
from puddles.memory import FakeMemory, entry_for
from schemas import SafetyResult
from tests.test_gate import (
    IDS,
    Safety,
    gate_input,
    gated_round,
    saved_gate,
    script_args,
    setup,
    traced,  # noqa: F401  the fixture, for the script run
)


class EmptySafety(Safety):
    """A safety check that runs no case and still says it passed, as run_safety does on an empty case file."""

    def __init__(self, passed=True):
        super().__init__()
        self.passed = passed

    def __call__(self, policy_version, reply):
        self.calls.append(policy_version)
        return SafetyResult(policy_version=policy_version, case_file="safety-empty.json", passed=self.passed,
                            cases=[])


def zero_case_report(tmp_path, passed=True, noise_measured=True):
    deps, current, candidate = setup(tmp_path, safety=EmptySafety(passed), noise_measured=noise_measured)
    return deps, gate.run_gate(gate_input(current, candidate), deps)


@pytest.mark.parametrize("passed", [True, False])
def test_zero_cases_is_reverted_on_safety_cases_alone_with_everything_else_passing(tmp_path, passed):
    deps, report = zero_case_report(tmp_path, passed)
    out = report.outcome
    assert out.decision == "reverted" and report.would_keep is False
    assert report.failed == ["safety_cases"]
    assert report.checks == {**{k: "passed" for k in gate.CHECKS}, "safety_cases": "failed"}  # noise measured too
    assert out.policy_in_force == "policy-v1" and out.after.quality == 4.0  # validation still ran and gained
    assert deps.safety.calls == ["policy-v2-candidate"]


def test_the_reason_says_no_safety_case_was_run_and_never_that_safety_passed(tmp_path):
    deps, report = zero_case_report(tmp_path)
    reason = report.outcome.reason
    assert reason == f"Reverted on {gate.CHECKS['safety_cases']}: no safety case was run."
    assert "passed" not in reason.lower()
    assert saved_gate(deps)["reason"] == reason and saved_gate(deps)["failed"] == ["safety_cases"]


def test_with_unmeasured_noise_the_reason_names_both_and_does_not_say_it_would_have_been_kept(tmp_path):
    _, report = zero_case_report(tmp_path, noise_measured=False)
    assert report.failed == ["safety_cases", "noise_not_measured"] and report.would_keep is False
    assert report.outcome.reason.startswith(f"Reverted on {gate.CHECKS['safety_cases']}: no safety case was run.")
    assert "would have kept" not in report.outcome.reason and "safety check passed" not in report.outcome.reason


def test_the_gates_own_statements_about_zero_cases_are_false(tmp_path):
    _, report = zero_case_report(tmp_path)
    assert report.outcome.safety_passed is False
    assert report.log_safety.model_dump() == {"case_file": "safety-empty.json", "passed": False, "cases": 0,
                                              "failed_case_ids": []}


def test_the_saved_safety_file_is_the_result_as_it_came_back(tmp_path):
    deps, report = zero_case_report(tmp_path)
    saved = SafetyResult.model_validate_json((deps.root / "scores" / "round-01-safety.json").read_text())
    assert saved == report.safety
    assert saved.passed is True and saved.cases == []  # what the check said, not what the gate concluded


def test_a_result_with_cases_is_unchanged(tmp_path):
    deps, current, candidate = setup(tmp_path)
    report = gate.run_gate(gate_input(current, candidate), deps)
    assert report.outcome.decision == "kept" and report.failed == []
    assert report.outcome.safety_passed is True and report.log_safety.passed is True and report.log_safety.cases == 3
    assert report.outcome.reason.endswith("; the safety check passed (3 cases).")


def test_through_make_gate_and_finish_the_round_event_and_memory_never_say_safety_passed(tmp_path):
    from puddles.round import finish_round

    deps, current, candidate = setup(tmp_path, safety=EmptySafety())
    result = gated_round(deps, current, candidate)
    out = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)(result)
    assert out.decision == "reverted" and out.safety_passed is False
    assert result.safety.passed is False and result.safety.cases == 0
    memory, log = FakeMemory(), DecisionLog(deps.root)
    event = finish_round(result, out, log, memory, deps.root)
    assert event.outcome == "reverted" and event.safety == result.safety and event.safety.passed is False
    entry = entry_for(out, event.records["outcome"], 1, summary=result.edit.summary)
    assert "no safety case was run" in entry.content and "safety cases passed" not in entry.content
    current_json = json.loads((deps.root / "scores" / "current.json").read_text())
    assert current_json["policy_version"] == "policy-v1"  # nothing promoted, so current.json not rewritten


def test_the_fake_script_with_the_real_gate_names_safety_cases(tmp_path, traced, capsys):  # noqa: F811
    from scripts.run_graph_round import main

    root = tmp_path / "root"
    assert main(script_args(tmp_path, "--gate", "real", "--noise-measured")) == 0
    [event] = DecisionLog(root).rounds()
    report = json.loads((root / "scores" / "round-01-gate.json").read_text())
    assert event.outcome == "reverted" and "safety_cases" in report["failed"]
    assert report["checks"]["safety_cases"] == "failed" and report["decision"] == "reverted"
    assert "no safety case was run" in event.reason
    assert event.safety is not None and event.safety.passed is False and event.safety.cases == 0
    assert not (root / "policies" / "policy-v2.md").exists()
    assert capsys.readouterr().out == f"round 1: reverted. {event.reason}\n"
