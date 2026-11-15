"""Independent acceptance checks for unattended-run.md 5.6 and 5.7."""
from pathlib import Path
import subprocess
import types

import httpx
import openai
import pytest
from pydantic import ValidationError

from puddles import parking, usage
from puddles.agents import critic, proposer, system_prompt
from puddles.decision_log import DecisionLog
from puddles.llm import AgentFailed
from puddles.policy import parse_policy
from schemas import (CriticInput, CriticOutput, LogEdit, LogPattern, LogPolicy,
                     LogScores, PatternBrief, ProposerInput, ProposerOutput,
                     RoundEvent, Scores)


TEXT = "# policy-v1\n\n## Safety\nS1. Be safe.\n\n## Conversation\nC1. Listen.\nC2. Ask gently.\nC8. Recommend professional help.\n"


def event(n, *, base="policy-v1", kept=None, action="reword", qualities=(4.0, 3.0), noise=0.25):
    scores = None if qualities is None else LogScores(
        before=Scores(empathy=4, relevance=4, quality=qualities[0]),
        after=Scores(empathy=3, relevance=3, quality=qualities[1]), noise=noise)
    return RoundEvent(
        round=n, outcome="kept" if kept else "reverted", reason="acceptance",
        policy=LogPolicy(base=base, base_sha256="x", in_force=kept or base), scores=scores,
        pattern=LogPattern(id=f"pattern-r{n:02}", name="Listening", seen_in=1,
                           out_of=1, status_before="open"),
        edit=LogEdit(id=f"edit-r{n:02}", action=action, rule_id="C1",
                     rule_text=None if action == "remove" else "Rejected wording."))


def history(path, *events):
    log = DecisionLog(path)
    for item in events:
        log.append(item)
    return log


def inputs():
    return (ProposerInput(policy_text=TEXT, at_length_limit=False, sources=[],
                          pattern=PatternBrief(id="pattern-new", name="New situation", status="open")),
            parse_policy(TEXT))


def move(action="add", rule=None, text="Ask about practical support."):
    return ProposerOutput(action=action, rule_id=rule, rule_text=text, summary="Try this.")


class Replies:
    """One queue across schemas verifies actual call ordering and validates model JSON."""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def structured(self, system, user, schema):
        self.calls.append((system, user, schema, usage.current()[1]))
        assert self.replies, "unexpected model call"
        expected, answer = self.replies.pop(0)
        assert schema is expected
        if isinstance(answer, BaseException):
            raise answer
        return schema.model_validate(answer)


def same(edit_id):
    return {"same_as_edit_id": edit_id, "reason": "Same behavioural change."}


def test_acc_off_matches_parent_prompts_checks_and_calls(tmp_path, monkeypatch):
    log = history(tmp_path, event(1))
    inp, policy = inputs()
    ci = CriticInput(round=2, policy_text=TEXT, earlier_patterns=log.earlier_patterns(), conversations=[])
    parents = []
    for module in (critic, proposer):
        source = subprocess.check_output(
            ["git", "show", f"HEAD^:puddles/agents/{module.__name__.split('.')[-1]}.py"], text=True)
        old = types.ModuleType("parent_agent")
        exec(compile(source, "<parent>", "exec"), old.__dict__)
        parents.append(old)
    for setting in (None, "0"):
        if setting is None:
            monkeypatch.delenv(parking.SETTING, raising=False)
        else:
            monkeypatch.setenv(parking.SETTING, setting)
        assert critic.render_user(ci, log) == parents[0].render_user(ci, log)
        out = CriticOutput(found=False, reason="None.")
        assert critic.check(out, ci, log) == parents[0].check(out, ci, log)
        assert proposer.render_user(inp, policy.version, log) == parents[1].render_user(inp, policy.version, log)
        for candidate in (move(), move("remove", "C2", None), move("reword", "C1", "New wording."),
                          move("reword", "S1", "Unsafe."), move(text="Two\nlines.")):
            llm = Replies()
            assert proposer.check(candidate, inp, policy, log, llm) == parents[1].check(candidate, inp, policy, log)
            assert llm.calls == []
        llm = Replies((ProposerOutput, move()))
        assert proposer.run(inp, policy.version, policy, log, llm) == move()
        assert len(llm.calls) == 1


def test_acc_noise_boundary_and_missing_measurements_across_two_keeps(tmp_path):
    cases = [((4, 3.5), 0.25, True), ((4, 3.75), 0.25, False),
             ((4, 3.875), 0.25, False), ((None, 3), 0.25, False),
             ((4, None), 0.25, False), (None, 0.25, False),
             ((4.15, 3.875), 0.275, False)]  # decimal equality on tonight's baseline/noise
    for i, (qualities, noise, carries) in enumerate(cases):
        log = history(tmp_path / str(i), event(1, qualities=qualities, noise=noise),
                      event(2, kept="policy-v2"), event(3, base="policy-v2", kept="policy-v3"))
        assert [t.edit_id for t in parking.to_compare(log, "policy-v1")] == ["edit-r01"]
        for version in ("policy-v2", "policy-v3"):
            parked = parking.parked(log, version)
            assert set(parked.rules) == ({"C1"} if carries else set()), (qualities, noise, version)
            assert set(parked.patterns) == ({"pattern-r01"} if carries else set())
            assert [t.edit_id for t in parking.to_compare(log, version)] == (["edit-r01"] if carries else [])
    missing_noise = parking.FailedTry(1, "e", "reword", "C1", "p", 4, 3, noise=None)
    assert not missing_noise.clearly_worse


def test_acc_call_requires_setting_coded_checks_and_comparison_history(tmp_path, monkeypatch):
    monkeypatch.setenv(parking.SETTING, "1")
    inp, policy = inputs()
    empty = history(tmp_path / "empty")
    unrelated = history(tmp_path / "old", event(1, base="policy-old", qualities=(4, 4.1)))
    log = history(tmp_path / "current", event(1))
    for current_log, candidate in ((empty, move()), (unrelated, move()), (log, move(text="Two\nlines.")),
                                   (log, move("remove", "C1", None)),
                                   (log, move("add", "C9", "Rejected wording."))):
        llm = Replies()
        proposer.check(candidate, inp, policy, current_log, llm)
        assert llm.calls == []
    llm = Replies((parking.SameEdit, same(None)))
    assert proposer.check(move(), inp, policy, log, llm) == []
    assert len(llm.calls) == 1
    assert llm.calls[0][0] == system_prompt("t6b-same-edit-check.md")


def test_acc_add_remove_and_unscored_rejections_are_safe_and_known_match_names_edit(tmp_path, monkeypatch):
    monkeypatch.setenv(parking.SETTING, "1")
    inp, policy = inputs()
    log = history(tmp_path, event(1, action="remove", qualities=None))
    for candidate in (move(), move("remove", "C2", None)):
        llm = Replies((parking.SameEdit, same("edit-r01")))
        problems = proposer.check(candidate, inp, policy, log, llm)
        assert len(problems) == 1
        assert all(s in problems[0] for s in ("edit-r01", "quality not logged", "different idea"))
        prompt = llm.calls[0][1]
        assert "edit-r01: remove C1: (no text: a removal)" in prompt
        assert "quality not logged" in prompt
        assert ("add a new rule: Ask about practical support." if candidate.action == "add"
                else "remove C2: (no text: a removal)") in prompt


def test_acc_fail_open_answers_validation_failure_agent_failure_and_provider_error(tmp_path, monkeypatch):
    monkeypatch.setenv(parking.SETTING, "1")
    inp, policy = inputs()
    log = history(tmp_path, event(1))
    try:
        parking.SameEdit.model_validate({"reason": 123})
    except ValidationError as exc:
        invalid = exc
    answers = (same(None), same("edit-unknown"), {"reason": 123}, invalid,
               AgentFailed("No answer"), openai.APIConnectionError(request=httpx.Request("POST", "https://example.test")))
    for answer in answers:
        llm = Replies((parking.SameEdit, answer))
        assert proposer.check(move(), inp, policy, log, llm) == []
        assert len(llm.calls) == 1


def test_acc_run_retries_once_then_returns_second_or_fails_twice(tmp_path, monkeypatch):
    monkeypatch.setenv(parking.SETTING, "1")
    inp, policy = inputs()
    log = history(tmp_path / "history", event(1))
    first, second = move(), move(text="Explore a different situation.")
    for repeated in (False, True):
        llm = Replies((ProposerOutput, first), (parking.SameEdit, same("edit-r01")),
                      (ProposerOutput, second), (parking.SameEdit, same("edit-r01" if repeated else None)))
        meter = usage.Meter(tmp_path / str(repeated), round=2)
        with usage.metering(meter, "proposer"):
            if repeated:
                with pytest.raises(AgentFailed, match="proposer failed its checks twice:.*edit-r01"):
                    proposer.run(inp, policy.version, policy, log, llm)
            else:
                assert proposer.run(inp, policy.version, policy, log, llm) == second
        assert len(llm.calls) == 4 and not llm.replies
        assert all(call[3] == "proposer" for call in llm.calls)
        assert "edit-r01" in llm.calls[2][1] and "different idea" in llm.calls[2][1]
        assert "Explore a different situation." in llm.calls[3][1]


def test_acc_interrupt_and_exit_escape_fail_open(tmp_path, monkeypatch):
    monkeypatch.setenv(parking.SETTING, "1")
    inp, policy = inputs()
    log = history(tmp_path, event(1))
    for error in (KeyboardInterrupt(), SystemExit(7)):
        with pytest.raises(type(error)) as caught:
            proposer.run(inp, policy.version, policy, log,
                         Replies((ProposerOutput, move()), (parking.SameEdit, error)))
        assert caught.value is error
