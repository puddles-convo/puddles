"""Section 8 of docs/specs/gate.md, rules 1 to 6: evidence at the boundaries of puddles/gate.py.

The gate trusts nothing it is handed by reference. Each test here failed on the gate as it was
before section 8 (except where a parameter is marked as a guard of behaviour that was already right).
All offline; the fakes are the ones tests/test_gate.py uses.
"""
import json
import math
from types import SimpleNamespace

import pytest

from puddles import gate
from puddles.records import sha256_bytes, sha256_text
from schemas import SafetyCaseResult, SafetyResult
from tests.test_gate import (
    HIDDEN,
    IDS,
    SCENARIOS,
    Judge,
    Safety,
    candidate_text,
    gate_input,
    gated_round,
    refused,
    setup,
)


def validation_scores(deps, round_=1):
    path = deps.root / "scores" / f"round-{round_:02d}-validation.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def incomplete(report):
    out = report.outcome
    assert out.decision == "reverted" and report.failed[0] == "complete_evidence", report.failed
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"
    assert out.after == gate.NOT_MEASURED and report.log_scores is None
    return out


# ---------- rule 1: a candidate's hash is the hash of its bytes, kept when the file does not load ----------

NOT_LOADING = {
    "no_safety_section": lambda b: b.replace(b"## Safety", b"## Notes"),
    "repeated_rule_id": lambda b: b + b"C9. A second rule with the same ID.\n",
    "crlf": HIDDEN["crlf"],
    "bom": HIDDEN["bom"],
    "not_utf8": HIDDEN["not_utf8"],
}


@pytest.mark.parametrize("how", list(NOT_LOADING))
def test_a_candidate_that_does_not_load_is_rejected_under_the_hash_of_its_bytes(tmp_path, how):
    deps, current, candidate = setup(tmp_path)
    candidate.write_bytes(NOT_LOADING[how](candidate.read_bytes()))
    report = gate.run_gate(gate_input(current, candidate), deps)
    assert report.failed[0] == "candidate_file" and report.outcome.decision == "reverted"
    assert report.candidate_sha256 == sha256_bytes(candidate.read_bytes())
    assert deps.judge.calls == [] and deps.safety.calls == []


@pytest.mark.parametrize("how", ["missing", "folder", "no_safety_section", "not_utf8"])
def test_candidate_sha256_is_none_only_when_the_file_cannot_be_read(tmp_path, how):
    # "missing" and "folder" guard behaviour that was already right; the other two failed before.
    deps, current, candidate = setup(tmp_path)
    if how == "missing":
        candidate.unlink()
    elif how == "folder":
        candidate.unlink()
        candidate.mkdir()
    else:
        candidate.write_bytes(NOT_LOADING[how](candidate.read_bytes()))
    report = gate.run_gate(gate_input(current, candidate), deps)
    assert report.failed[0] == "candidate_file"
    expected = None if how in ("missing", "folder") else sha256_bytes(candidate.read_bytes())
    assert report.candidate_sha256 == expected


@pytest.mark.parametrize("how", ["no_safety_section", "repeated_rule_id"])
def test_a_rejected_candidate_replaced_before_finish_round_is_not_filed_under_the_rejection(tmp_path, how):
    from puddles.round import CandidateChanged

    deps, current, candidate = setup(tmp_path)
    clean = candidate.read_bytes()
    candidate.write_bytes(NOT_LOADING[how](clean))
    rejected = sha256_bytes(candidate.read_bytes())
    result = gated_round(deps, current, candidate)
    keeper = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)
    out = keeper(result)
    assert out.decision == "reverted" and keeper.last_report.failed[0] == "candidate_file"
    assert result.candidate_sha256 == rejected
    candidate.write_bytes(clean)  # a loadable file in its place
    refused(tmp_path, CandidateChanged, result, out, deps.root)


# ---------- rule 2: the judge's results are copied and checked as they arrive ----------

class VersionChangesAtFirstCall(Judge):
    """Changes its version on the first call and scores everything under the new one."""

    def score(self, record):
        self.version = "judge-other"
        return super().score(record)


class VersionChangesAfterLastCall(Judge):
    """Every score carries the version read before the run; the version moves after the last call."""

    def score(self, record):
        out = super().score(record)
        if len(self.calls) == len(IDS):
            self.version = "judge-other"
        return out


class WrongVersion(Judge):
    def score(self, record):
        out = super().score(record)
        return SimpleNamespace(score=out.score.model_copy(update={"judge_version": "another-judge"}))


class RenamesItsInput(Judge):
    """Changes the record it was handed, then scores the changed record."""

    def score(self, record):
        record.id = "conv-r01-val-elsewhere" if "val" in record.id else "conv-r00-current-elsewhere"
        record.turns = []
        return super().score(record)


class ChangesItsInputAfterScoring(Judge):
    """Scores the record as given, then changes it: the gate compares against the record before the call."""

    def score(self, record):
        out = super().score(record)
        record.id, record.turns = "something-else", []
        return out


class OutOfRange(Judge):
    """Hands back a ScoreRecord changed after it was validated, to a value no judge can give."""

    def score(self, record):
        out = super().score(record)
        out.score.empathy = 50
        out.score.relevance = 50
        return out


class SharedScore(Judge):
    """Hands back one object for every conversation, overwritten each time: 1s for four, a 5 last."""

    def __init__(self):
        super().__init__(lambda record: (5, 5) if len(self.calls) == len(IDS) else (1, 1))
        self.shared = None

    def score(self, record):
        fresh = super().score(record).score
        if self.shared is None:
            self.shared = fresh
        else:
            self.shared.__dict__.update(fresh.__dict__)
        return SimpleNamespace(score=self.shared)


UNUSABLE = {
    "version_changes_at_first_call": VersionChangesAtFirstCall,
    "version_changes_after_last_call": VersionChangesAfterLastCall,
    "wrong_version_in_score": WrongVersion,
    "renames_its_input": RenamesItsInput,
    "out_of_range_after_validation": OutOfRange,
}


@pytest.mark.parametrize("judge", list(UNUSABLE))
def test_validation_from_a_judge_whose_results_cannot_be_trusted_is_incomplete(tmp_path, judge):
    deps, current, candidate = setup(tmp_path, judge=UNUSABLE[judge]())
    report = gate.run_gate(gate_input(current, candidate), deps)
    incomplete(report)


@pytest.mark.parametrize("judge", list(UNUSABLE))
def test_a_baseline_from_a_judge_whose_results_cannot_be_trusted_is_not_kept(tmp_path, judge):
    # Rule 6: the baseline is scored the same way as validation.
    deps, current, _ = setup(tmp_path, judge=UNUSABLE[judge]())
    with pytest.raises(gate.GateError, match="could not be scored"):
        gate.current_scores(deps, current, IDS)
    assert not gate.current_path(deps.root).exists()


def test_the_judge_is_given_its_own_copy_and_checked_against_the_record_as_it_was(tmp_path):
    deps, current, candidate = setup(tmp_path, judge=ChangesItsInputAfterScoring())
    report = gate.run_gate(gate_input(current, candidate), deps)
    assert report.outcome.decision == "kept" and report.checks["complete_evidence"] == "passed"
    saved = [json.loads(line)["id"] for line in
             (deps.root / "conversations" / "round-01-validation.jsonl").read_text().splitlines()]
    assert sorted(s["conversation_id"] for s in validation_scores(deps)) == sorted(saved)


def test_one_score_object_reused_by_the_judge_does_not_overwrite_earlier_scores(tmp_path):
    deps, current, candidate = setup(tmp_path, judge=SharedScore())
    report = gate.run_gate(gate_input(current, candidate), deps)
    rows = validation_scores(deps)
    assert len({r["conversation_id"] for r in rows}) == len(IDS)
    assert [r["empathy"] for r in rows] == [1, 1, 1, 1, 5]
    assert report.outcome.decision == "reverted" and report.failed[0] == "quality_gain"  # 1.8 against 3


def test_a_baseline_reused_score_object_is_averaged_as_the_judge_gave_it(tmp_path):
    deps, current, _ = setup(tmp_path, judge=SharedScore())
    before = gate.current_scores(deps, current, IDS)
    assert before.empathy == pytest.approx(1.8)
    saved = json.loads(gate.current_path(deps.root).read_text())
    assert saved["scores"]["empathy"] == pytest.approx(1.8)


def test_evidence_needs_exactly_one_score_for_each_expected_conversation(tmp_path, monkeypatch):
    real = gate.run_batch

    def one_conversation_five_times(*args, **kwargs):
        batch = real(*args, **kwargs)
        batch.records = [batch.records[0]] * len(batch.records)
        return batch

    monkeypatch.setattr(gate, "run_batch", one_conversation_five_times)
    deps, current, candidate = setup(tmp_path)
    report = gate.run_gate(gate_input(current, candidate), deps)
    out = incomplete(report)
    assert "exactly one" in out.reason


def test_a_scenario_id_naming_two_scenarios_raises(tmp_path):
    deps, current, candidate = setup(tmp_path)
    deps.scenarios = list(SCENARIOS) + [SCENARIOS[0].model_copy(update={"situation": "Another story."})]
    with pytest.raises(gate.GateError, match="more than one"):
        gate.score_policy(deps, candidate, IDS, 1, gate.VALIDATION_TAG)
    with pytest.raises(gate.GateError, match="more than one"):
        gate.current_scores(deps, current, IDS)
    report = gate.run_gate(gate_input(current, candidate), deps)
    assert report.outcome.decision == "reverted" and "complete_evidence" in report.failed


# Rule 2, last bullet (KeyboardInterrupt and SystemExit are not caught) is behaviour the gate already
# had; tests/test_gate.py tests it (the judge that raises KeyboardInterrupt).


# ---------- rule 3: the safety result is copied and checked as it arrives ----------

class SafetyChangedLater(Safety):
    """Keeps a reference to the result it returned; the judge later turns it into a failure."""

    def __call__(self, policy_version, reply):
        self.result = super().__call__(policy_version, reply)
        return self.result


def test_a_safety_result_changed_after_it_was_returned_changes_nothing(tmp_path):
    safety = SafetyChangedLater()

    class Spoiler(Judge):
        def score(self, record):
            safety.result.passed = False
            for case in safety.result.cases:
                case.passed = False
            return super().score(record)

    deps, current, candidate = setup(tmp_path, judge=Spoiler(), safety=safety)
    report = gate.run_gate(gate_input(current, candidate), deps)
    disk = json.loads((deps.root / "scores" / "round-01-safety.json").read_text())
    assert report.outcome.decision == "kept" and report.checks["safety_cases"] == "passed"
    assert report.outcome.safety_passed is True and report.safety.passed is True
    assert report.log_safety.passed is True and report.log_safety.failed_case_ids == []
    assert disk["passed"] is True and all(c["passed"] for c in disk["cases"])


def _result(version, passed=True, cases=(("s1-01", True),)):
    return SafetyResult(policy_version=version, case_file="safety-test.json", passed=passed,
                        cases=[SafetyCaseResult(case_id=c, passed=p) for c, p in cases])


def _does_not_validate(version, reply):
    result = _result(version)
    result.cases = None  # changed after validation: its dumped fields are not a SafetyResult
    return result


BAD_SAFETY = {
    "does_not_validate": _does_not_validate,
    "another_policy": lambda version, reply: _result("policy-never-tested"),
    "passed_with_a_failed_case": lambda version, reply: _result(version, cases=(("s1-01", True), ("s2-01", False))),
}


@pytest.mark.parametrize("which", list(BAD_SAFETY))
def test_an_unusable_safety_result_is_treated_as_a_safety_check_that_raised(tmp_path, which):
    deps, current, candidate = setup(tmp_path, safety=BAD_SAFETY[which])
    report = gate.run_gate(gate_input(current, candidate), deps)
    assert report.outcome.decision == "reverted" and report.failed[0] == "safety_cases"
    assert report.safety is None and report.log_safety is None and report.outcome.safety_passed is False
    assert not (deps.root / "scores" / "round-01-safety.json").exists()
    assert report.checks["complete_evidence"] == "passed"  # validation still ran


@pytest.mark.parametrize("cases", [(), (("s1-01", True),)])
def test_a_safety_result_that_passed_with_no_failed_case_is_accepted(tmp_path, cases):
    # A guard: no cases at all and passed is accepted (the script's stand-in is one).
    deps, current, candidate = setup(tmp_path, safety=lambda version, reply: _result(version, cases=cases))
    report = gate.run_gate(gate_input(current, candidate), deps)
    assert report.outcome.decision == "kept" and report.log_safety.cases == len(cases)


# ---------- rule 4: current.json matches only on the same values of the same types ----------

@pytest.mark.parametrize("saved, now", [(False, 0), (0, False), (True, 1), (7.0, 7), (7, 7.0)])
def test_a_seed_of_another_type_does_not_match(tmp_path, saved, now):
    deps, current, _ = setup(tmp_path)
    deps.seed = saved
    gate.current_scores(deps, current, IDS)
    calls = len(deps.judge.calls)
    deps.seed = now
    gate.current_scores(deps, current, IDS)
    assert len(deps.judge.calls) - calls == len(IDS)
    assert json.loads(gate.current_path(deps.root).read_text())["seed"] == now
    assert type(json.loads(gate.current_path(deps.root).read_text())["seed"]) is type(now)


def test_a_turn_limit_of_another_type_does_not_match(tmp_path):
    deps, current, _ = setup(tmp_path)
    gate.current_scores(deps, current, IDS)
    path = gate.current_path(deps.root)
    data = json.loads(path.read_text())
    data["max_turns"] = float(data["max_turns"])
    path.write_text(json.dumps(data))
    calls = len(deps.judge.calls)
    gate.current_scores(deps, current, IDS)
    assert len(deps.judge.calls) - calls == len(IDS)


NOT_MEASURED_SCORES = {
    "not_measured": {"empathy": 0, "relevance": 0, "quality": 0, "emotion_change": None},
    "quality_missing": {"empathy": 3, "relevance": 3},
    "quality_none": {"empathy": 3, "relevance": 3, "quality": None},
    "above_five": {"empathy": 6, "relevance": 3, "quality": 4.5},
    "below_one": {"empathy": 3, "relevance": 0.5, "quality": 1.75},
    "nan": {"empathy": math.nan, "relevance": 3, "quality": 3},
    "infinite": {"empathy": 3, "relevance": 3, "quality": math.inf},
    "boolean": {"empathy": True, "relevance": True, "quality": 1},
}


@pytest.mark.parametrize("which", list(NOT_MEASURED_SCORES))
def test_saved_scores_that_are_not_a_measurement_count_as_absent(tmp_path, which):
    deps, current, _ = setup(tmp_path)
    first = gate.current_scores(deps, current, IDS)
    path = gate.current_path(deps.root)
    data = json.loads(path.read_text())
    data["scores"] = NOT_MEASURED_SCORES[which]
    path.write_text(json.dumps(data))
    calls = len(deps.judge.calls)
    assert gate.current_scores(deps, current, IDS) == first
    assert len(deps.judge.calls) - calls == len(IDS)


def test_a_not_measured_baseline_in_current_json_cannot_promote_a_worse_candidate(tmp_path):
    judge = Judge(lambda record: (4, 4) if "candidate" in record.policy_version else (5, 5))
    deps, current, candidate = setup(tmp_path, judge=judge)
    gate.current_scores(deps, current, IDS)
    path = gate.current_path(deps.root)
    data = json.loads(path.read_text())
    data["scores"] = gate.NOT_MEASURED.model_dump()
    path.write_text(json.dumps(data))
    out = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)(gated_round(deps, current, candidate))
    assert out.decision == "reverted" and out.before.quality == 5


# ---------- rule 5: what is written to current.json is described by the run that measured it ----------

def test_a_kept_candidates_current_json_is_described_by_its_validation_run(tmp_path, monkeypatch):
    deps, current, candidate = setup(tmp_path)
    measured = {"judge_version": deps.judge.version, "max_turns": deps.max_turns, "seed": deps.seed,
                "scenarios_sha256": gate.scenarios_sha256(deps, IDS)}
    real = gate._run_gate

    def then_change_deps(*args, **kwargs):
        report, run = real(*args, **kwargs)
        deps.judge.version = "judge-later"
        deps.max_turns, deps.seed = 3, 99
        deps.scenarios = [s.model_copy(update={"situation": s.situation + " Later."}) for s in SCENARIOS]
        return report, run

    monkeypatch.setattr(gate, "_run_gate", then_change_deps)
    out = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)(gated_round(deps, current, candidate))
    assert out.decision == "kept"
    saved = json.loads(gate.current_path(deps.root).read_text())
    assert saved["policy_version"] == "policy-v2"
    assert {k: saved[k] for k in measured} == measured


def test_a_baseline_is_written_with_the_judge_version_read_before_its_run(tmp_path):
    # Rules 5 and 6: a judge whose version moves during the baseline leaves no current.json at all,
    # so nothing scored under one version is ever filed under another.
    deps, current, _ = setup(tmp_path, judge=VersionChangesAtFirstCall())
    with pytest.raises(gate.GateError):
        gate.current_scores(deps, current, IDS)
    assert not gate.current_path(deps.root).exists()
    deps.judge.version = "judge-test"
    deps.judge.calls.clear()
    with pytest.raises(gate.GateError):  # scored again, not matched from a file
        gate.current_scores(deps, current, IDS)
    assert len(deps.judge.calls) == len(IDS)


def test_a_valid_policy_hash_is_the_hash_of_its_text(tmp_path):
    # Rule 1's last sentence, as a guard: for valid UTF-8 the bytes' hash is sha256_text of the text.
    deps, current, candidate = setup(tmp_path)
    report = gate.run_gate(gate_input(current, candidate), deps)
    assert report.candidate_sha256 == sha256_text(candidate_text()) == sha256_bytes(candidate.read_bytes())
