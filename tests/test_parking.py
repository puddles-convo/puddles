"""What was tried and rejected is not tried again: docs/specs/unattended-run.md section 5 (rules 5.1 to 5.5)."""

import pytest

import scripts.run_loop as driver
from puddles import parking
from puddles.agents import critic, proposer, researcher
from puddles.decision_log import DecisionLog
from puddles.llm import AgentFailed
from puddles.policy import parse_policy
from schemas import (
    CriticOutput,
    LogEdit,
    LogPattern,
    LogPolicy,
    LogScores,
    ProposerOutput,
    ResearcherOutput,
    RoundEvent,
    Scores,
)
from tests.fakes import ScriptedLLM
from tests.test_agents import CONVS, GOOD, POLICY, setup
from tests.test_unattended_driver import capture_round, patch_real, write_config

# Tonight's run (train-root2): (round, outcome, same_as, action, rule, base, in_force, before, after)
TONIGHT = [
    (1, "kept", None, "reword", "C1", "policy-v1", "policy-v2", 3.325, 3.65),
    (2, "reverted", "pattern-r01", "reword", "C1", "policy-v2", "policy-v2", 3.65, 3.9),
    (3, "reverted", "pattern-r01", "reword", "C1", "policy-v2", "policy-v2", 3.65, 3.7),
    (4, "kept", None, "add", "C9", "policy-v2", "policy-v3", 3.65, 4.15),
    (5, "reverted", "pattern-r04", "reword", "C9", "policy-v3", "policy-v3", 4.15, 3.475),
    (6, "reverted", "pattern-r01", "reword", "C1", "policy-v3", "policy-v3", 4.15, 3.925),
    (7, "reverted", "pattern-r04", "reword", "C9", "policy-v3", "policy-v3", 4.15, 3.525),
]


def event(round_, outcome, same_as, action, rule, base, in_force, before, after):
    return RoundEvent(
        round=round_, outcome=outcome,
        pattern=LogPattern(id=f"pattern-r{round_:02d}", same_as_pattern_id=same_as, name=f"p{same_as or round_}",
                           seen_in=2, out_of=4, status_before="open"),
        edit=LogEdit(id=f"edit-r{round_:02d}", action=action, rule_id=rule, rule_text=f"text {round_}"),
        policy=LogPolicy(base=base, base_sha256="x", new=f"{base}-candidate", new_sha256="y", in_force=in_force),
        scores=LogScores(before=Scores(empathy=before, relevance=before, quality=before),
                         after=Scores(empathy=after, relevance=after, quality=after), noise=0.15),
        reason="r")


def write_log(tmp_path, rows):
    log = DecisionLog(tmp_path)
    for row in rows:
        log.append(event(*row))
    return log


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("PUDDLES_PARK_REVERTED", "1")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("PUDDLES_PARK_REVERTED", raising=False)


# ---------- 5.1 the setting ----------

@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True), ("yes", False)])
def test_enabled_only_for_1(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("PUDDLES_PARK_REVERTED", raising=False)
    else:
        monkeypatch.setenv("PUDDLES_PARK_REVERTED", value)
    assert parking.enabled() is expected


def test_a_bad_setting_is_refused_before_any_call(tmp_path, monkeypatch, capsys):
    write_config(tmp_path, monkeypatch)
    made = patch_real(monkeypatch)
    seen: list = []
    capture_round(monkeypatch, tmp_path / "root", seen)
    monkeypatch.delenv("PUDDLES_GATE_REMEASURE_KEPT", raising=False)
    monkeypatch.setenv("PUDDLES_PARK_REVERTED", "yes")
    assert driver.main(["--root", str(tmp_path / "root"), "--gate", "real", "--max-rounds", "1"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "PUDDLES_PARK_REVERTED" in err and err.count("\n") == 1
    assert made == [] and seen == []
    assert not (tmp_path / "root").exists()


# ---------- 5.2 what is parked ----------

def test_tonights_log_parks_c9_c1_and_their_patterns_on_v3_and_c1_on_v2_by_two_tries(tmp_path):
    log = write_log(tmp_path, TONIGHT)
    v3 = parking.parked(log, "policy-v3")
    assert set(v3.rules) == {"C9", "C1"}
    assert [t.edit_id for t in v3.rules["C9"]] == ["edit-r05", "edit-r07"]
    assert [t.edit_id for t in v3.rules["C1"]] == ["edit-r06"]
    assert set(v3.patterns) == {"pattern-r04", "pattern-r01"}
    t = v3.rules["C1"][0]
    assert (t.round, t.action, t.rule_id, t.before, t.after) == (6, "reword", "C1", 4.15, 3.925) and t.negative

    v2 = parking.parked(log, "policy-v2")  # rounds 2 and 3: quality rose both times
    assert not any(t.negative for t in v2.rules["C1"])
    assert set(v2.rules) == {"C1"} and set(v2.patterns) == {"pattern-r01"}
    assert [t.round for t in v2.rules["C1"]] == [2, 3]


def test_one_failed_try_with_rising_quality_is_not_parked(tmp_path):
    log = write_log(tmp_path, TONIGHT[:2])
    v2 = parking.parked(log, "policy-v2")
    assert v2.rules == {} and v2.patterns == {}


def test_a_negative_result_without_logged_quality_does_not_count(tmp_path):
    log = DecisionLog(tmp_path)
    e = event(*TONIGHT[1])
    e.scores.before.quality = None
    log.append(e)
    assert parking.parked(log, "policy-v2").rules == {}


def test_after_a_keep_nothing_is_parked_on_the_new_version(tmp_path):
    log = write_log(tmp_path, TONIGHT[:4])  # rounds 2 and 3 parked C1 on v2; round 4 kept policy-v3
    assert parking.parked(log, "policy-v2").rules
    v3 = parking.parked(log, "policy-v3")
    assert v3.rules == {} and v3.patterns == {}


# ---------- 5.3 the critic ----------

def critic_input(tmp_path):
    log, mem = setup(tmp_path)
    for row in TONIGHT:
        log.append(event(*row))
    return log, critic.build_input(8, POLICY, CONVS, log, mem, tmp_path)


def test_critic_on_rejects_a_parked_pattern_and_marks_it(tmp_path, on):
    log, inp = critic_input(tmp_path)
    problems = critic.check(CriticOutput.model_validate({**GOOD, "same_as_pattern_id": "pattern-r05"}), inp, log)
    assert len(problems) == 1
    assert "pattern-r05 is parked on this policy (policy-v3)" in problems[0] and "edit-r07" in problems[0]
    assert critic.check(CriticOutput.model_validate(GOOD), inp, log) == []
    text = critic.render_user(inp, log)
    assert '<parked_patterns policy="policy-v3">' in text
    assert "pattern-r04: " in text and "round 5: edit-r05 (reword C9) was reverted, quality 4.15 -> 3.48" in text
    assert "PARKED on policy-v3" in text


def test_critic_on_two_parked_answers_take_the_existing_failure_path(tmp_path, on):
    log, inp = critic_input(tmp_path)
    llm = ScriptedLLM()
    parked = {**GOOD, "same_as_pattern_id": "pattern-r04"}
    llm.add(CriticOutput, parked, parked)
    with pytest.raises(AgentFailed):
        critic.run(inp, llm, log)
    assert "parked" in llm.prompts[CriticOutput][1] and "parked_patterns" in llm.prompts[CriticOutput][0]


def test_critic_off_passes_the_parked_pattern_and_renders_as_today(tmp_path, off):
    log, inp = critic_input(tmp_path)
    assert critic.check(CriticOutput.model_validate({**GOOD, "same_as_pattern_id": "pattern-r04"}), inp, log) == []
    assert critic.render_user(inp, log) == critic.render_user(inp)
    assert "parked" not in critic.render_user(inp, log).lower()


# ---------- 5.4 the proposer ----------

def proposer_setup(tmp_path):
    """A recurrence of pattern-r01, whose rule C1 (kept in round 1) is parked on policy-v3 by round 6."""
    log, inp = critic_input(tmp_path)
    _, mem = setup(tmp_path)
    pattern = critic.to_record(CriticOutput.model_validate({**GOOD, "same_as_pattern_id": "pattern-r01"}), inp, log)
    research = researcher.to_record(ResearcherOutput.model_validate(
        {"queries": ["q"], "sources": [{"url": "https://u", "technique": "Ask first."}]}), pattern, 8)
    p_in = proposer.build_input(POLICY, pattern, research, log, mem, tmp_path)
    assert p_in.pattern.status == "recurring" and p_in.pattern.rule_id == "C1"
    return log, p_in, parse_policy(POLICY)


REWORD = ProposerOutput(action="reword", rule_id="C1", rule_text="Reflect first, briefly.", summary="s")
ADD = ProposerOutput(action="add", rule_text="Ask one open question before any suggestion.", summary="s")


def test_proposer_on_rejects_a_parked_rule_and_allows_an_add(tmp_path, on):
    log, p_in, policy = proposer_setup(tmp_path)
    problems = proposer.check(REWORD, p_in, policy, log)
    assert len(problems) == 1
    assert "C1 is parked on this policy (policy-v3)" in problems[0]
    assert "edit-r06 (reword C1) was reverted, quality 4.15 -> 3.92" in problems[0]
    remove = ProposerOutput(action="remove", rule_id="C1", summary="s")
    assert any("parked" in p for p in proposer.check(remove, p_in, policy, log))
    assert proposer.check(ADD, p_in, policy, log) == []
    text = proposer.render_user(p_in, policy.version, log)
    assert '<parked_rules policy="policy-v3">' in text and "C1: round 6: edit-r06" in text


def test_proposer_off_behaves_as_today(tmp_path, off):
    log, p_in, policy = proposer_setup(tmp_path)
    assert proposer.check(REWORD, p_in, policy, log) == []
    assert proposer.check(ADD, p_in, policy, log) == [
        "this pattern already has an active rule; reword or remove it instead of adding"]
    assert proposer.render_user(p_in, policy.version, log) == proposer.render_user(p_in, policy.version)
    assert "parked" not in proposer.render_user(p_in, policy.version, log).lower()
