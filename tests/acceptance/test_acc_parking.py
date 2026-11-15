"""Independent acceptance checks for unattended-run.md 5.1--5.4."""
import subprocess
import types

import pytest

from puddles import parking
from puddles.agents import critic, proposer
from puddles.decision_log import DecisionLog
from puddles.llm import AgentFailed
from puddles.policy import PROTECTED_RULE_IDS, parse_policy
from schemas import (CriticInput, CriticOutput, LogEdit, LogPattern, LogPolicy,
                     LogScores, PatternBrief, ProposerInput, ProposerOutput,
                     RoundEvent, ScoredConversation, Scores)
from tests.fakes import ScriptedLLM

TEXT = "# policy-v3\n\n## Safety\nS1. Be safe.\n\n## Conversation\nC1. Listen carefully.\nC2. Ask gently.\nC8. Suggest professional help when needed.\n"


def event(n, *, base="policy-v3", action="reword", rule="C1", same=None,
          outcome="reverted", qualities=(4.0, 3.0)):
    scores = None if qualities is None else LogScores(
        before=Scores(empathy=4, relevance=4, quality=qualities[0]),
        after=Scores(empathy=4, relevance=4, quality=qualities[1]), noise=0.2)
    return RoundEvent(
        round=n, outcome=outcome, reason="acceptance",
        pattern=LogPattern(id=f"pattern-r{n:02}", same_as_pattern_id=same,
                           name="Listening", seen_in=1, out_of=1, status_before="open"),
        edit=LogEdit(id=f"edit-r{n:02}", action=action, rule_id=rule,
                     rule_text=None if action == "remove" else f"Rejected wording {n}."),
        policy=LogPolicy(base=base, base_sha256="a", in_force=base), scores=scores)


def history(path, *events):
    log = DecisionLog(path)
    for e in events:
        log.append(e)
    return log


def critic_input(log, text=TEXT):
    return CriticInput(round=20, policy_text=text, earlier_patterns=log.earlier_patterns(),
                       conversations=[ScoredConversation(conversation_id="conv-a", empathy=2,
                                                          relevance=2, note="Advice too soon.", turns=[])])


def found(same=None):
    return CriticOutput(found=True, name="Listening", description="Advice too soon.",
                        seen_in=1, out_of=1, example_conversation_ids=["conv-a"],
                        same_as_pattern_id=same)


def proposer_input(*, status="recurring", rule="C1", text=TEXT):
    return ProposerInput(policy_text=text, at_length_limit=False, sources=[],
                         pattern=PatternBrief(id="pattern-r01", name="Listening",
                                              status=status, rule_id=rule))


def move(action, rule=None, text=None):
    return ProposerOutput(action=action, rule_id=rule, rule_text=text, summary="Try this.")


def test_acc_off_is_identical_to_parent_prompts_and_checks(tmp_path, monkeypatch):
    log = history(tmp_path, event(1))
    ci, pi, policy = critic_input(log), proposer_input(), parse_policy(TEXT)
    parents = []
    for module in (critic, proposer):
        source = subprocess.check_output(
            ["git", "show", f"HEAD^:puddles/agents/{module.__name__.split('.')[-1]}.py"], text=True)
        old = types.ModuleType("acceptance_parent")
        exec(compile(source, "<parent commit>", "exec"), old.__dict__)
        parents.append(old)
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv("PUDDLES_PARK_REVERTED", raising=False)
        else:
            monkeypatch.setenv("PUDDLES_PARK_REVERTED", value)
        assert critic.render_user(ci, log) == parents[0].render_user(ci)
        assert proposer.render_user(pi, policy.version, log) == parents[1].render_user(pi, policy.version)
        for out in (found("pattern-r01"), found("missing"), CriticOutput(found=False, reason="none")):
            assert critic.check(out, ci, log) == parents[0].check(out, ci, log)
        for out in (move("add", text="Fresh wording."), move("reword", "C1", "Fresh wording."),
                    move("remove", "C1"), move("reword", "C8", "x"), move("add", "C9", "x")):
            assert proposer.check(out, pi, policy, log) == parents[1].check(out, pi, policy, log)


def test_acc_only_failed_edits_on_current_version_count_and_keep_resets(tmp_path):
    log = history(tmp_path, event(1, base="policy-v2"), event(2, outcome="kept"),
                  event(3, qualities=(4, 4)), event(4, rule="C2", qualities=(4, 4.2)))
    assert parking.parked(log, "policy-v3").rules == {}
    assert parking.parked(log, "policy-v3").patterns == {}
    log.append(event(5, same="pattern-r03", qualities=(4, 4.1)))
    parked = parking.parked(log, "policy-v3")
    assert set(parked.rules) == {"C1"}
    assert set(parked.patterns) == {"pattern-r03"}
    assert [t.round for t in parked.rules["C1"]] == [3, 5]
    keep = event(6, outcome="kept")
    keep.policy.in_force = "policy-v4"
    log.append(keep)
    assert parking.parked(log, "policy-v4").rules == {}
    assert parking.parked(log, "policy-v4").patterns == {}


def test_acc_missing_scores_count_twice_but_are_never_negative(tmp_path):
    for i, qualities in enumerate((None, (None, 2), (4, None))):
        log = history(tmp_path / str(i), event(1, qualities=qualities))
        assert parking.parked(log, "policy-v3").rules == {}
        assert not parking.failed_tries(log, "policy-v3")[0].negative
        log.append(event(2, same="pattern-r01", qualities=qualities))
        parked = parking.parked(log, "policy-v3")
        assert set(parked.rules) == {"C1"} and set(parked.patterns) == {"pattern-r01"}
        assert all(not t.negative for t in parked.rules["C1"])
        assert "quality not logged" in parking.tries_text(parked.rules["C1"])


def test_acc_critic_marks_identity_rejects_later_sighting_and_retries_once(tmp_path, monkeypatch):
    monkeypatch.setenv("PUDDLES_PARK_REVERTED", "1")
    log = history(tmp_path, event(1), event(2, same="pattern-r01"))
    inp = critic_input(log)
    text = critic.render_user(inp, log)
    assert "PARKED on policy-v3" in text and "pattern-r01:" in text
    assert "edit-r01" in text and "quality 4 -> 3" in text
    for identity in ("pattern-r01", "pattern-r02"):
        problems = critic.check(found(identity), inp, log)
        assert len(problems) == 1 and "parked on this policy" in problems[0]
        assert "different pattern, or none" in problems[0]
    llm = ScriptedLLM()
    llm.add(CriticOutput, found("pattern-r02"), found("pattern-r01"))
    with pytest.raises(AgentFailed, match="parked"):
        critic.run(inp, llm, log)
    assert len(llm.prompts[CriticOutput]) == 2
    assert critic.check(CriticOutput(found=False, reason="No other weakness."), inp, log) == []


def test_acc_proposer_forbids_parked_rule_and_only_relaxes_add_for_that_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("PUDDLES_PARK_REVERTED", "1")
    log = history(tmp_path, event(1))
    inp, policy = proposer_input(), parse_policy(TEXT)
    for out in (move("reword", "C1", "A fresh approach."), move("remove", "C1")):
        problems = proposer.check(out, inp, policy, log)
        assert len(problems) == 1 and "parked" in problems[0]
        assert "edit-r01" in problems[0] and "quality 4 -> 3" in problems[0]
    assert proposer.check(move("add", text="A fresh approach."), inp, policy, log) == []
    assert "C1: round 1: edit-r01" in proposer.render_user(inp, policy.version, log)
    for status in ("fixed", "recurring"):
        unparked = proposer_input(status=status, rule="C2")
        assert any("already has an active rule" in p for p in
                   proposer.check(move("add", text="A fresh approach."), unparked, policy, log))


def test_acc_proposer_always_has_move_below_cap_including_all_editable_rules_parked(tmp_path, monkeypatch):
    monkeypatch.setenv("PUDDLES_PARK_REVERTED", "1")
    policy = parse_policy(TEXT)
    for parked_ids in ((), ("C1",), ("C1", "C2")):
        log = history(tmp_path / str(len(parked_ids)),
                      *(event(n, rule=rid) for n, rid in enumerate(parked_ids, 1)))
        # All action types and every existing rule, including protected ones. Fresh
        # text represents arbitrary one-line rewordings/additions not in the log.
        candidates = [move("add", text="An untried helpful technique.")]
        for rid in policy.rule_ids():
            candidates += [move("remove", rid), move("reword", rid, "Listen.")]
        for inp in (proposer_input(status="open", rule=None), proposer_input(rule="C1"),
                    proposer_input(rule="C2")):
            assert any(not proposer.check(out, inp, policy, log) for out in candidates)


def test_acc_length_cap_can_exhaust_moves_when_every_editable_rule_is_parked(tmp_path, monkeypatch):
    monkeypatch.setenv("PUDDLES_PARK_REVERTED", "1")
    policy = parse_policy(TEXT)
    log = history(tmp_path, event(1), event(2, rule="C2"))
    inp = proposer_input()
    inp.at_length_limit = True
    assert any("length limit" in p for p in proposer.check(move("add", text="Listen."), inp, policy, log))
    for rid in policy.rule_ids():
        for out in (move("remove", rid), move("reword", rid, "x")):
            assert proposer.check(out, inp, policy, log)
    # Below the cap, an addition is available in this same state.
    inp.at_length_limit = False
    assert proposer.check(move("add", text="Listen."), inp, policy, log) == []


def test_acc_partial_rounds_add_ids_and_titleless_input_do_not_crash_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("PUDDLES_PARK_REVERTED", "1")
    no_pattern = event(1, qualities=None)
    no_pattern.pattern = None
    no_edit = event(2)
    no_edit.edit = None
    added = event(3, action="add", rule="C9")  # logged ID is assigned by apply(), not the model
    log = history(tmp_path, no_pattern, no_edit, added)
    parked = parking.parked(log, "policy-v3")
    assert parked.rules == {} and set(parked.patterns) == {"pattern-r03"}
    for text in (TEXT, "## Conversation\nC1. Listen.\n", "", "\n"):
        ci = critic_input(log, text)
        critic.render_user(ci, log)
        assert critic.check(found(), ci, log) == []
        pi = proposer_input(status="open", rule=None, text=text)
        proposer.render_user(pi, "policy-v3", log)
        assert proposer.check(move("add", text="Try a new technique."), pi, parse_policy(TEXT), log) == []


def test_acc_driver_rejects_bad_setting_before_configuration_or_model_calls(tmp_path, monkeypatch, capsys):
    import scripts.run_loop as driver

    def premature_call(*args, **kwargs):
        pytest.fail("driver reached configuration/model setup before refusing bad parking value")

    monkeypatch.setattr(driver, "load_loop_config", premature_call)
    monkeypatch.delenv("PUDDLES_GATE_REMEASURE_KEPT", raising=False)
    for bad in ("yes", "", "true", "2", " 1"):
        monkeypatch.setenv("PUDDLES_PARK_REVERTED", bad)
        assert driver.main(["--root", str(tmp_path / "run"), "--gate", "real"]) == 2
        assert "PUDDLES_PARK_REVERTED" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
