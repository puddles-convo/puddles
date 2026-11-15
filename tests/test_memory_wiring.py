"""Task B5: two rounds where each agent reads, from memory, records other agents wrote."""
import json
from pathlib import Path

import pytest

from puddles.decision_log import DecisionLog
from puddles.memory import FakeMemory, PastError
from puddles.round import finish_round, memory_trace, policy_in_force, retry_pending, run_agents, stand_in_gate
from schemas import CriticOutput, MemoryCopyEvent, ProposerOutput, ResearcherOutput, ScoredConversation
from tests.fakes import FakeSearch, ScriptedLLM

FIXTURES = Path(__file__).parents[1] / "fixtures"
CONVS = [ScoredConversation.model_validate(c) for c in json.loads((FIXTURES / "bad_conversations.json").read_text())]
START = FIXTURES / "policy-v3.md"
PATTERN = {"found": True, "name": "advice-before-understanding",
           "description": "Gives advice in its first reply before asking about the situation.",
           "seen_in": 3, "out_of": 4, "example_conversation_ids": ["conv-hand-001", "conv-hand-003"]}
URL1, URL2 = "https://counselling.example.ac.uk/skills", "https://health.example.gov/listening"
REVERTED_RULE = "Offer advice only after the person has asked for it."


def test_two_rounds_wire_memory_between_agents(tmp_path):
    root, log, mem, llm = tmp_path, DecisionLog(tmp_path), FakeMemory(), ScriptedLLM()
    mem.ensure_training_audience()
    search = FakeSearch({
        "counselling skills exploring before giving advice": [{"url": URL1, "title": "Skills", "text": "..."}],
        "open questions before advice in counselling": [{"url": URL2, "title": "Listening", "text": "..."}]})

    # Round 1: the edit is reverted.
    llm.add(CriticOutput, PATTERN)
    llm.add(ResearcherOutput, {"queries": ["counselling skills exploring before giving advice"],
                               "sources": [{"url": URL1, "title": "Skills", "technique": "Explore before advising."}]})
    llm.add(ProposerOutput, {"action": "add", "rule_text": REVERTED_RULE,
                             "summary": "Hold advice until asked, for advice-before-understanding."})
    r1 = run_agents(1, CONVS, START, log, mem, llm, search, root)
    finish_round(r1, stand_in_gate(r1, "reverted"), log, mem, root)
    assert {e.identity for e in mem.entries.values()} == {"critic", "researcher", "proposer", "gatekeeper"}

    # Round 2: the same pattern recurs. Each agent first gets a bad answer rejected.
    llm.add(CriticOutput, {**PATTERN, "same_as_pattern_id": "pattern-r01"})
    llm.add(ResearcherOutput,
            {"queries": ["counselling skills exploring before giving advice"], "sources": []},
            {"queries": ["open questions before advice in counselling"],
             "sources": [{"url": URL2, "title": "Listening", "technique": "Ask one open question first."}]})
    llm.add(ProposerOutput,
            {"action": "add", "rule_text": REVERTED_RULE, "summary": "Same again."},
            {"action": "add", "rule_text": "In your first two replies, ask one open question and hold suggestions.",
             "summary": "Ask first instead of withholding advice; unlike edit-r01 it gives the helper a step."})
    r2 = run_agents(2, CONVS, policy_in_force(root, START), log, mem, llm, search, root)
    finish_round(r2, stand_in_gate(r2, "kept"), log, mem, root)

    critic_prompt = llm.prompts[CriticOutput][-1]
    assert "pattern-r01: advice-before-understanding, open" in critic_prompt  # from the log
    assert "gate-r01: edit edit-r01 was reverted" in critic_prompt  # from memory, written by the gatekeeper
    assert "counselling skills exploring before giving advice" in llm.prompts[ResearcherOutput][-2]
    assert "were run in earlier rounds" in llm.prompts[ResearcherOutput][-1]
    proposer_prompt = llm.prompts[ProposerOutput][-1]
    assert "edit-r01: add C9, reverted." in proposer_prompt and "Gate: Stand-in gate" in proposer_prompt
    assert "same edit as edit-r01, which was reverted" in proposer_prompt

    assert r2.pattern.id == "pattern-r01" and r2.edit.rule_id == "C10"
    assert (root / "policies" / "policy-v4.md").read_text().startswith("# policy-v4\n")
    assert policy_in_force(root, START).name == "policy-v4.md"
    assert "pattern-r01-seen-r02" in mem.entries
    assert [e.status for e in log.events() if e.event == "memory_copy"] == ["confirmed"] * 8

    trace = {(row["record"], row["round"]): row for row in memory_trace(log, root)}
    assert trace[("gate-r01", 1)]["written_by"] == "gatekeeper"
    assert {"critic", "proposer"} <= set(trace[("gate-r01", 1)]["read_by"])
    assert "researcher" in trace[("research-r01", 1)]["read_by"]
    assert any(w != r for row in trace.values() for w in [row["written_by"]] for r in row["read_by"])

    # The simulated user can see none of it.
    assert mem.recall("advice before understanding edit pattern", identity="simulated-user") == []


def test_copy_still_pending_after_three_attempts_stops_the_loop(tmp_path):
    log = DecisionLog(tmp_path)
    log.append(MemoryCopyEvent(round=1, record_id="pattern-r01", status="pending", attempt=3))
    with pytest.raises(PastError, match="pattern-r01"):
        retry_pending(log, FakeMemory(), tmp_path)
