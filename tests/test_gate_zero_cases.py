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
from schemas import SafetyResult, Scores
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

ALL_CASES = ("s1-01", "s2-01", "s3-01")  # what tests.test_gate.Safety runs


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


# ---------- a result that contradicts itself is not a pass either ----------

class ContradictorySafety(Safety):
    """Says passed while marking these cases failed."""

    def __call__(self, policy_version, reply):
        result = super().__call__(policy_version, reply)
        return result.model_copy(update={"passed": True})


@pytest.mark.parametrize("failed", [("s2-01",), ALL_CASES], ids=["one-case-failed", "all-cases-failed"])
def test_passed_true_with_a_failed_case_fails_safety_cases_and_names_the_cases(tmp_path, failed):
    deps, current, candidate = setup(tmp_path, safety=ContradictorySafety(failed=failed))
    report = gate.run_gate(gate_input(current, candidate), deps)
    out = report.outcome
    assert out.decision == "reverted" and report.would_keep is False and report.failed == ["safety_cases"]
    assert report.checks == {**{k: "passed" for k in gate.CHECKS}, "safety_cases": "failed"}  # noise measured
    assert out.reason == (f"Reverted on {gate.CHECKS['safety_cases']}: {len(failed)} of 3 safety cases failed "
                          f"({', '.join(failed)}).")
    assert out.safety_passed is False
    assert report.log_safety.model_dump() == {"case_file": "safety-test.json", "passed": False, "cases": 3,
                                              "failed_case_ids": list(failed)}
    saved = SafetyResult.model_validate_json((deps.root / "scores" / "round-01-safety.json").read_text())
    assert saved == report.safety and saved.passed is True  # saved as it came back


# ---------- a baseline outside 1 to 5 was not measured ----------

@pytest.mark.parametrize("before", [gate.NOT_MEASURED, Scores(empathy=5.5, relevance=3),
                                    Scores(empathy=3, relevance=0.5)], ids=["not-measured", "5.5", "0.5"])
def test_a_baseline_that_was_not_measured_fails_complete_evidence_and_nothing_is_kept(tmp_path, before):
    deps, current, candidate = setup(tmp_path)
    inp = gate_input(current, candidate).model_copy(update={"current_scores": before})
    report = gate.run_gate(inp, deps)
    out = report.outcome
    assert out.decision == "reverted" and report.would_keep is False and report.failed == ["complete_evidence"]
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
    assert report.checks["safety_cases"] == "passed" and deps.safety.calls  # the safety check still ran
    assert len(deps.judge.calls) == len(IDS)  # and so did validation
    assert out.reason == f"Reverted on {gate.CHECKS['complete_evidence']}: the policy in force has no measured scores."
    assert out.after == gate.NOT_MEASURED and report.log_scores is None  # as for any incomplete evidence
    assert out.before.empathy == before.empathy and out.before.relevance == before.relevance  # as given
    assert out.safety_passed is True and report.log_safety.passed is True


def test_a_matching_current_json_of_zeros_is_scored_again(tmp_path):
    deps, current, _ = setup(tmp_path)
    first = gate.current_scores(deps, current, IDS)
    path = deps.root / "scores" / "current.json"
    data = json.loads(path.read_text())
    data["scores"] = {"empathy": 0.0, "relevance": 0.0, "quality": 0.0, "emotion_change": 0.0}
    path.write_text(json.dumps(data))  # every identity field still matches
    calls = len(deps.judge.calls)
    assert gate.current_scores(deps, current, IDS) == first
    assert len(deps.judge.calls) - calls == len(IDS)  # scored again: one judge call per scenario
    assert json.loads(path.read_text())["scores"] == first.model_dump()  # and rewritten with measured scores


# ---------- quality is never taken from a file or a caller: it is always worked out ----------

def flat_judge():
    """Baseline and candidate both truly 4 and 4: no gain to find."""
    from tests.test_gate import Judge

    return Judge(rule=lambda record: (4, 4))


@pytest.mark.parametrize("quality", [0.0, -1.0, 1.0, 4.5], ids=["zero", "negative", "in-range-1", "in-range-4.5"])
def test_a_stored_quality_that_is_not_the_mean_cannot_make_a_gain_through_make_gate(tmp_path, quality):
    from puddles.round import finish_round

    deps, current, candidate = setup(tmp_path, judge=flat_judge())
    gate.current_scores(deps, current, IDS)  # truly (4, 4, quality 4)
    path = deps.root / "scores" / "current.json"
    data = json.loads(path.read_text())
    data["scores"]["quality"] = quality  # only the quality changed; every identity field still matches
    path.write_text(json.dumps(data))
    result = gated_round(deps, current, candidate)
    calls = len(deps.judge.calls)
    out = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)(result)
    baseline_calls = [c for c in deps.judge.calls[calls:] if c.startswith("conv-r00-current-")]
    assert len(baseline_calls) == len(IDS)  # the file counted as absent: the policy in force was scored again
    assert out.decision == "reverted" and out.before.quality == 4.0 == out.after.quality
    assert "quality rose by 0.00" in out.reason and "from 0" not in out.reason
    event = finish_round(result, out, DecisionLog(deps.root), FakeMemory(), deps.root)
    assert event.outcome == "reverted" and event.scores.before.quality == 4.0
    assert not (deps.root / "policies" / "policy-v2.md").exists()


@pytest.mark.parametrize("quality", [0.0, -1.0, 1.0, 3.5], ids=["zero", "negative", "in-range-1", "in-range-3.5"])
def test_current_scores_with_a_quality_that_is_not_the_mean_fail_complete_evidence(tmp_path, quality):
    deps, current, candidate = setup(tmp_path)
    inp = gate_input(current, candidate).model_copy(
        update={"current_scores": Scores(empathy=3, relevance=3, quality=quality)})
    report = gate.run_gate(inp, deps)
    out = report.outcome
    assert out.decision == "reverted" and report.failed == ["complete_evidence"]
    assert out.reason == f"Reverted on {gate.CHECKS['complete_evidence']}: the policy in force has no measured scores."
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
    assert out.before.quality == 3.0  # the record carries the worked-out quality, not the one given
    assert deps.safety.calls and len(deps.judge.calls) == len(IDS)  # safety and validation still ran


@pytest.mark.parametrize("quality", [None, 3.0, 3.0000000001], ids=["unset", "the-mean", "the-mean-to-9-places"])
def test_a_quality_that_is_unset_or_the_mean_is_measured_and_worked_out(tmp_path, quality):
    deps, current, candidate = setup(tmp_path)
    inp = gate_input(current, candidate).model_copy(
        update={"current_scores": Scores(empathy=3, relevance=3, quality=quality)})
    report = gate.run_gate(inp, deps)
    assert report.outcome.decision == "kept" and report.failed == []
    assert report.outcome.before.quality == 3.0 == report.log_scores.before.quality
    assert report.outcome.after.quality == 4.0 == report.log_scores.after.quality


def test_score_policys_own_quality_is_the_mean_of_its_two_dimensions(tmp_path):
    from tests.test_gate import Judge

    table = {sid: (e, r) for sid, (e, r) in zip(IDS, [(5, 1), (4, 2), (3, 3), (2, 5), (1, 4)])}
    deps, current, _ = setup(tmp_path, judge=Judge(rule=lambda record: table[record.scenario_id]))
    scores = gate.score_policy(deps, current, IDS, 0, gate.CURRENT_TAG).scores
    assert scores.quality == (scores.empathy + scores.relevance) / 2


# ---------- a conversation with no helper reply measures no policy ----------

def test_max_turns_zero_fails_the_candidates_evidence_and_the_judge_is_never_asked(tmp_path):
    deps, current, candidate = setup(tmp_path)
    deps.max_turns = 0
    report = gate.run_gate(gate_input(current, candidate), deps)
    out = report.outcome
    assert out.decision == "reverted" and report.failed == ["complete_evidence"]
    assert gate.NO_HELPER_REPLY in out.reason and f"{len(IDS)} of {len(IDS)} had no helper reply" in out.reason
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
    assert out.after == gate.NOT_MEASURED and report.log_scores is None
    assert deps.judge.calls == []  # nothing to judge
    assert (deps.root / "scores" / "round-01-validation.jsonl").read_text() == ""


def test_max_turns_zero_through_make_gate_reverts_untested_with_no_model_judging(tmp_path):
    deps, current, candidate = setup(tmp_path)
    deps.max_turns = 0
    result = gated_round(deps, current, candidate)
    out = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)(result)
    assert out.decision == "reverted" and out.before == out.after == gate.NOT_MEASURED
    assert "policy in force could not be scored" in out.reason and gate.NO_HELPER_REPLY in out.reason
    assert "candidate was not tested" in out.reason.lower() and deps.safety.calls == [] and deps.judge.calls == []
    assert not (deps.root / "scores" / "current.json").exists()


def test_one_conversation_without_a_helper_turn_is_unscored_whatever_the_judge_would_say(tmp_path, monkeypatch):
    real = gate.run_batch

    def drop_one_reply(*args, **kwargs):
        batch = real(*args, **kwargs)
        first = batch.records[0]
        batch.records[0] = first.model_copy(update={"turns": [t for t in first.turns if t.speaker == "user"]})
        return batch

    monkeypatch.setattr(gate, "run_batch", drop_one_reply)
    deps, current, candidate = setup(tmp_path)
    report = gate.run_gate(gate_input(current, candidate), deps)
    first_id = f"conv-r01-val-{IDS[0]}"
    assert report.outcome.decision == "reverted" and report.failed == ["complete_evidence"]
    assert f"1 of {len(IDS)} had no helper reply ({first_id})" in report.outcome.reason
    assert first_id not in deps.judge.calls and len(deps.judge.calls) == len(IDS) - 1


# ---------- the inputs the gate takes a number from ----------

@pytest.mark.parametrize("noise", [-0.1, float("nan")])
def test_a_noise_that_was_not_measured_is_refused_and_make_gate_reverts_untested(tmp_path, noise):
    deps, current, candidate = setup(tmp_path)
    with pytest.raises(ValueError):
        gate_input(current, candidate, noise=noise)  # GateInput: noise >= 0, and NaN fails it
    result = gated_round(deps, current, candidate)
    adapter = gate.make_gate(deps, validation_scenario_ids=IDS, noise=noise)
    out = adapter(result)
    assert out.decision == "reverted" and adapter.last_report.failed == ["complete_evidence"]
    assert "candidate was not tested" in out.reason.lower() and out.after == gate.NOT_MEASURED
    assert deps.safety.calls == [] and result.scores is None  # never tested


def test_an_infinite_noise_can_only_fail_quality_gain(tmp_path):
    deps, current, candidate = setup(tmp_path)
    report = gate.run_gate(gate_input(current, candidate, noise=float("inf")), deps)
    assert report.outcome.decision == "reverted" and report.failed == ["quality_gain"]
