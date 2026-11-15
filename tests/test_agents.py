import json
from pathlib import Path

import pytest

from puddles.agents import critic, proposer, researcher
from puddles.decision_log import DecisionLog
from puddles.llm import AgentFailed
from puddles.memory import FakeMemory
from puddles.policy import parse_policy
from schemas import CriticOutput, ProposerOutput, ResearcherOutput, ScoredConversation
from tests.fakes import FakeSearch, ScriptedLLM

FIXTURES = Path(__file__).parents[1] / "fixtures"
CONVS = [ScoredConversation.model_validate(c) for c in json.loads((FIXTURES / "bad_conversations.json").read_text())]
POLICY = (FIXTURES / "policy-v3.md").read_text()
GOOD = {"found": True, "name": "advice-before-understanding", "description": "Advises before asking.",
        "seen_in": 3, "out_of": 4, "example_conversation_ids": ["conv-hand-001", "conv-hand-002"]}


def setup(tmp_path):
    mem = FakeMemory()
    mem.ensure_training_audience()
    return DecisionLog(tmp_path), mem


def test_critic_retries_once_with_the_failed_checks(tmp_path):
    log, mem = setup(tmp_path)
    inp = critic.build_input(1, POLICY, CONVS, log, mem, tmp_path)
    llm = ScriptedLLM()
    llm.add(CriticOutput, {**GOOD, "out_of": 10, "example_conversation_ids": ["conv-made-up"]}, GOOD)
    out = critic.run(inp, llm)
    assert out.seen_in == 3
    assert "out_of must be 4" in llm.prompts[CriticOutput][1] and "conv-made-up" in llm.prompts[CriticOutput][1]
    record = critic.to_record(out, inp)
    assert record.id == "pattern-r01" and record.out_of == 4


def test_critic_gives_up_after_two_failures_and_allows_none_found(tmp_path):
    log, mem = setup(tmp_path)
    inp = critic.build_input(1, POLICY, CONVS, log, mem, tmp_path)
    llm = ScriptedLLM()
    llm.add(CriticOutput, {**GOOD, "seen_in": 9}, {**GOOD, "seen_in": 9})
    with pytest.raises(AgentFailed):
        critic.run(inp, llm)
    llm.add(CriticOutput, {"found": False, "reason": "No pattern in more than one conversation."})
    assert critic.to_record(critic.run(inp, llm), inp) is None


def test_researcher_rejects_urls_exa_did_not_return(tmp_path):
    log, mem = setup(tmp_path)
    inp = critic.build_input(1, POLICY, CONVS, log, mem, tmp_path)
    pattern = critic.to_record(CriticOutput.model_validate(GOOD), inp)
    search = FakeSearch({"exploring before advice": [{"url": "https://uni.example/skills", "title": "t", "text": "x"}]})
    llm = ScriptedLLM()
    bad = {"queries": ["exploring before advice"],
           "sources": [{"url": "https://invented.example", "technique": "Ask first."}]}
    good = {"queries": ["exploring before advice"],
            "sources": [{"url": "https://uni.example/skills", "title": "t", "technique": "Ask one question."}]}
    llm.add(ResearcherOutput, bad, good)
    out, searches = researcher.run(researcher.build_input(pattern, log, mem, tmp_path), llm, search)
    assert [s.url for s in out.sources] == ["https://uni.example/skills"]
    assert "https://invented.example" in llm.prompts[ResearcherOutput][1]
    assert searches[0]["query"] == "exploring before advice"


def test_proposer_cannot_touch_safety_and_code_applies_the_edit(tmp_path):
    log, mem = setup(tmp_path)
    inp = critic.build_input(1, POLICY, CONVS, log, mem, tmp_path)
    pattern = critic.to_record(CriticOutput.model_validate(GOOD), inp)
    research = researcher.to_record(ResearcherOutput.model_validate(
        {"queries": ["q"], "sources": [{"url": "https://u", "technique": "Ask first."}]}), pattern, 1)
    p_in = proposer.build_input(POLICY, pattern, research, log, mem, tmp_path)
    llm = ScriptedLLM()
    llm.add(ProposerOutput, {"action": "reword", "rule_id": "S1", "rule_text": "x", "summary": "s"},
            {"action": "add", "rule_text": "Ask one open question before any suggestion.", "summary": "Ask first."})
    policy = parse_policy(POLICY)
    out = proposer.run(p_in, policy.version, policy, log, llm)
    assert "safety rule" in llm.prompts[ProposerOutput][1]
    edit, path = proposer.apply(out, POLICY, log, pattern, research, 1, tmp_path)
    assert edit.rule_id == "C9" and edit.new_policy_version == "policy-v4-candidate"
    candidate = path.read_text()
    assert candidate.startswith("# policy-v4-candidate\n")
    assert candidate.replace("# policy-v4-candidate", "# policy-v3").replace(
        "C9. Ask one open question before any suggestion.\n", "") == POLICY
