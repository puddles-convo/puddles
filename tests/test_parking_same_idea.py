"""docs/specs/unattended-run.md 5.6 (a clearly worse result is remembered for the whole run) and
5.7 (the same idea in other words is not tried again)."""

import pytest
from pydantic import BaseModel

from puddles import parking
from puddles.agents import proposer
from puddles.decision_log import DecisionLog
from puddles.llm import AgentFailed
from schemas import LogEdit, LogPattern, LogPolicy, LogScores, ProposerOutput, RoundEvent, Scores
from tests.fakes import ScriptedLLM
from tests.test_parking import ADD, proposer_setup

# Tonight's run (train-root2) with its real noise of 0.275, and round 8's keep of policy-v4.
# (round, outcome, same_as, action, rule, base, in_force, before, after)
TONIGHT = [
    (1, "kept", None, "reword", "C1", "policy-v1", "policy-v2", 3.325, 3.65),
    (2, "reverted", "pattern-r01", "reword", "C1", "policy-v2", "policy-v2", 3.65, 3.9),
    (3, "reverted", "pattern-r01", "reword", "C1", "policy-v2", "policy-v2", 3.65, 3.7),
    (4, "kept", None, "add", "C9", "policy-v2", "policy-v3", 3.65, 4.15),
    (5, "reverted", "pattern-r04", "reword", "C9", "policy-v3", "policy-v3", 4.15, 3.475),
    (6, "reverted", "pattern-r01", "reword", "C1", "policy-v3", "policy-v3", 4.15, 3.925),
    (7, "reverted", "pattern-r04", "reword", "C9", "policy-v3", "policy-v3", 4.15, 3.525),
    (8, "kept", None, "add", "C10", "policy-v3", "policy-v4", 4.15, 4.6),
]


def event(round_, outcome, same_as, action, rule, base, in_force, before, after, noise=0.275):
    return RoundEvent(
        round=round_, outcome=outcome,
        pattern=LogPattern(id=f"pattern-r{round_:02d}", same_as_pattern_id=same_as, name="p",
                           seen_in=2, out_of=4, status_before="open"),
        edit=LogEdit(id=f"edit-r{round_:02d}", action=action, rule_id=rule,
                     rule_text=None if action == "remove" else f"Stricter wording {round_}."),
        policy=LogPolicy(base=base, base_sha256="x", in_force=in_force),
        scores=LogScores(before=Scores(empathy=before, relevance=before, quality=before),
                         after=Scores(empathy=after, relevance=after, quality=after), noise=noise),
        reason="r")


def write_log(path, rows):
    log = DecisionLog(path)
    for row in rows:
        log.append(event(*row) if isinstance(row, tuple) else row)
    return log


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("PUDDLES_PARK_REVERTED", "1")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("PUDDLES_PARK_REVERTED", raising=False)


# ---------- 5.6 ----------

def test_tonight_v4_keeps_c9_and_pattern_r04_parked_and_v3_is_unchanged(tmp_path):
    log = write_log(tmp_path, TONIGHT)
    v4 = parking.parked(log, "policy-v4")
    assert set(v4.rules) == {"C9"} and set(v4.patterns) == {"pattern-r04"}
    assert [t.edit_id for t in v4.rules["C9"]] == ["edit-r05", "edit-r07"]
    assert [t.edit_id for t in v4.patterns["pattern-r04"]] == ["edit-r05", "edit-r07"]
    v3 = parking.parked(log, "policy-v3")
    assert set(v3.rules) == {"C9", "C1"} and set(v3.patterns) == {"pattern-r04", "pattern-r01"}
    v2 = parking.parked(log, "policy-v2")  # an earlier version is not touched by rounds 5 and 7
    assert set(v2.rules) == {"C1"} and set(v2.patterns) == {"pattern-r01"}


def test_a_fall_within_the_noise_does_not_carry_over_a_keep(tmp_path):
    log = write_log(tmp_path, [TONIGHT[5], TONIGHT[7]])  # round 6: 4.15 -> 3.925, noise 0.275
    assert parking.parked(log, "policy-v3").rules.keys() == {"C1"}
    assert parking.parked(log, "policy-v4").rules == {} and parking.parked(log, "policy-v4").patterns == {}


def test_carried_over_only_from_its_round_on(tmp_path):
    log = write_log(tmp_path, [TONIGHT[4]])  # round 5 clearly worse on v3, nothing kept yet
    assert parking.parked(log, "policy-v4").rules == {}
    log.append(event(*TONIGHT[7]))
    assert set(parking.parked(log, "policy-v4").rules) == {"C9"}


def test_a_fall_equal_to_the_noise_is_not_clearly_worse_despite_float_error(tmp_path):
    assert 4.15 - 3.875 > 0.275  # the floating-point slip the rounding guards against
    t = parking.FailedTry(round=5, edit_id="edit-r05", action="reword", rule_id="C9", pattern_id="pattern-r04",
                          before=4.15, after=3.875, base="policy-v3", noise=0.275)
    assert t.negative and not t.clearly_worse
    assert not parking.FailedTry(round=1, edit_id="e", action="reword", rule_id="C1", pattern_id=None,
                                 before=0.1 + 0.2, after=0.3).negative
    log = write_log(tmp_path, [(5, "reverted", "pattern-r04", "reword", "C9", "policy-v3", "policy-v3", 4.15, 3.875),
                               TONIGHT[7]])
    assert parking.parked(log, "policy-v4").rules == {} and parking.parked(log, "policy-v4").patterns == {}


def test_no_noise_logged_is_not_clearly_worse(tmp_path):
    t = parking.FailedTry(round=5, edit_id="edit-r05", action="reword", rule_id="C9", pattern_id="pattern-r04",
                          before=4.15, after=3.475, base="policy-v3", noise=None)
    assert t.negative and not t.clearly_worse
    no_scores = event(*TONIGHT[4])
    no_scores.scores = None
    log = write_log(tmp_path, [no_scores, TONIGHT[7]])
    assert parking.parked(log, "policy-v4").rules == {}
    assert parking.to_compare(log, "policy-v4") == []


# ---------- 5.7 ----------

def compare_setup(tmp_path):
    """Policy v4 in force after test_parking's log (noise 0.15): rounds 5, 6 and 7 were clearly worse."""
    log, p_in, policy = proposer_setup(tmp_path)
    log.append(event(*TONIGHT[7]))
    policy.version = "policy-v4"
    assert [t.edit_id for t in parking.to_compare(log, "policy-v4")] == ["edit-r05", "edit-r06", "edit-r07"]
    return log, p_in, policy


class Raising:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def structured(self, system, user, schema):
        self.calls += 1
        raise self.exc


def test_same_idea_fails_the_check_naming_the_edit(tmp_path, on):
    log, p_in, policy = compare_setup(tmp_path)
    llm = ScriptedLLM()
    llm.add(parking.SameEdit, {"same_as_edit_id": "edit-r05", "reason": "Both make C9 stricter."})
    problems = proposer.check(ADD, p_in, policy, log, llm)
    assert len(problems) == 1
    assert "edit-r05" in problems[0] and "quality 4.15 -> 3.48" in problems[0] and "different idea" in problems[0]
    user = llm.prompts[parking.SameEdit][0]
    assert "Proposed edit:\nadd a new rule: Ask one open question" in user
    assert "edit-r05: reword C9: text 5 (round 5, reverted, quality 4.15 -> 3.48)" in user and "edit-r07" in user


@pytest.mark.parametrize("answer", [{"same_as_edit_id": None, "reason": "Different situation."},
                                    {"same_as_edit_id": "edit-r99", "reason": "?"}])
def test_null_or_unknown_id_lets_the_edit_through(tmp_path, on, answer):
    log, p_in, policy = compare_setup(tmp_path)
    llm = ScriptedLLM()
    llm.add(parking.SameEdit, answer)
    assert proposer.check(ADD, p_in, policy, log, llm) == []
    assert len(llm.prompts[parking.SameEdit]) == 1


@pytest.mark.parametrize("exc", [AgentFailed("down"), RuntimeError("provider error")])
def test_a_failed_call_lets_the_edit_through(tmp_path, on, exc):
    log, p_in, policy = compare_setup(tmp_path)
    llm = Raising(exc)
    assert proposer.check(ADD, p_in, policy, log, llm) == []
    assert llm.calls == 1


def test_an_invalid_answer_lets_the_edit_through(tmp_path, on):
    log, p_in, policy = compare_setup(tmp_path)

    class Wrong(BaseModel):
        x: int

    try:
        Wrong.model_validate({})
    except Exception as e:  # a real pydantic ValidationError
        err = e
    llm = Raising(err)
    assert proposer.check(ADD, p_in, policy, log, llm) == []


def test_no_call_with_nothing_to_compare(tmp_path, on):
    _, p_in, policy = proposer_setup(tmp_path)
    p_in.pattern.status = "open"
    log2 = write_log(tmp_path / "empty", [TONIGHT[1]])  # one failed try on policy-v2, not clearly worse
    llm = Raising(AssertionError("must not be called"))
    assert proposer.check(ADD, p_in, policy, log2, llm) == []
    assert llm.calls == 0


def test_no_call_when_coded_checks_already_failed(tmp_path, on):
    log, p_in, policy = compare_setup(tmp_path)
    llm = Raising(AssertionError("must not be called"))
    bad = ProposerOutput(action="add", rule_text="Line one.\nLine two.", summary="s")
    assert any("one line" in p for p in proposer.check(bad, p_in, policy, log, llm))
    assert llm.calls == 0


def test_no_call_when_off(tmp_path, off):
    log, p_in, policy = compare_setup(tmp_path)
    llm = Raising(AssertionError("must not be called"))
    proposer.check(ADD, p_in, policy, log, llm)
    assert llm.calls == 0
    assert proposer.check(ADD, p_in, policy, log, llm) == proposer.check(ADD, p_in, policy, log)


def test_run_retries_once_after_a_same_idea_answer(tmp_path, on):
    log, p_in, policy = compare_setup(tmp_path)
    second = ProposerOutput(action="add", rule_text="Name one feeling you hear before anything else.", summary="s2")
    llm = ScriptedLLM()
    llm.add(proposer.ProposerOutput, ADD, second)
    llm.add(parking.SameEdit, {"same_as_edit_id": "edit-r07", "reason": "Same as C9's stricter wording."},
            {"same_as_edit_id": None, "reason": "A different situation."})
    out = proposer.run(p_in, policy.version, policy, log, llm)
    assert out == second
    assert len(llm.prompts[proposer.ProposerOutput]) == 2
    assert "edit-r07" in llm.prompts[proposer.ProposerOutput][1] and "different idea" in llm.prompts[proposer.ProposerOutput][1]
