"""Memory schema version 3: per-sighting IDs, budgeted history, the replay check and restart safety."""
import json
from pathlib import Path

import pytest

from puddles import replay
from puddles.agents import researcher
from puddles.decision_log import DecisionLog, DecisionLogError
from puddles.memory import FakeMemory
from puddles.round import finish_round, prepare_round, retry_pending, run_agents, stand_in_gate
from schemas import (
    FROM_MEMORY_LIMIT,
    CriticOutput,
    LogPolicy,
    MemoryCopyEvent,
    ProposerOutput,
    ResearcherOutput,
    RoundEvent,
    ScoredConversation,
)
from tests.fakes import FakeSearch, ScriptedLLM

FIXTURES = Path(__file__).parents[1] / "fixtures"
CONVS = [ScoredConversation.model_validate(c) for c in json.loads((FIXTURES / "bad_conversations.json").read_text())]
START = FIXTURES / "policy-v3.md"
URL = "https://counselling.example.ac.uk/skills"


def pattern(name: str, **extra) -> dict:
    return {"found": True, "name": name, "description": f"{name} description.", "seen_in": 2, "out_of": 4,
            "example_conversation_ids": ["conv-hand-001"], **extra}


class Loop:
    def __init__(self, root: Path):
        self.root, self.log, self.mem, self.llm = root, DecisionLog(root), FakeMemory(), ScriptedLLM()
        self.mem.ensure_training_audience()
        self.search = FakeSearch({})

    def round(self, name: str, gate: str = "reverted", same_as=None, rule_text=None, action="add", rule_id=None):
        n = self.log.next_round()
        query = f"query {n} for {name}"
        self.search.results_by_query[query] = [{"url": URL, "title": "t", "text": "x"}]
        self.llm.add(CriticOutput, pattern(name, same_as_pattern_id=same_as))
        self.llm.add(ResearcherOutput, {"queries": [query], "sources": [{"url": URL, "technique": "Ask first."}]})
        text = None if action == "remove" else (rule_text or f"Rule for {name} in round {n}.")
        self.llm.add(ProposerOutput, {"action": action, "rule_id": rule_id, "rule_text": text,
                                      "summary": f"Edit for {name} in round {n}."})
        policy = prepare_round(self.log, self.mem, self.root, START)
        result = run_agents(n, CONVS, policy, self.log, self.mem, self.llm, self.search, self.root)
        return result, finish_round(result, stand_in_gate(result, gate), self.log, self.mem, self.root, START)


def test_recurrence_gets_its_own_id_and_matches_by_name(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding", gate="kept", rule_text="Ask before advising.")
    r2, _ = loop.round("Advice-Before-Understanding", rule_text="Ask two questions before advising.",
                       action="reword", rule_id="C9")  # no same_as_pattern_id: code matches the name
    assert r2.pattern.id == "pattern-r02" and r2.pattern.same_as_pattern_id == "pattern-r01"
    assert r2.status_before == "fixed"
    (brief,) = loop.log.earlier_patterns()
    assert (brief.id, brief.status, brief.sightings) == ("pattern-r01", "recurring", 2)
    assert {"pattern-r01", "pattern-r02"} <= set(loop.mem.entries)


def test_history_lists_are_budgeted(tmp_path):
    loop = Loop(tmp_path)
    for i in range(1, 14):
        loop.round(f"pattern-name-{i}", rule_text=f"Rule number {i}.")
    shown, summary = loop.log.earlier_edits("pattern-r01")
    assert [e.id for e in shown] == ["edit-r01"] + [f"edit-r{i:02d}" for i in range(4, 14)]
    assert summary == "2 older edits are not shown: 0 kept, 2 reverted"
    queries, qsummary = loop.log.earlier_queries("pattern-r01")
    assert queries[0] == "query 1 for pattern-name-1" and len(queries) == 11
    assert qsummary == "2 older queries are not shown"
    patterns, psummary = loop.log.critic_patterns(recalled={"pattern-r02"})
    assert [p.id for p in patterns][:2] == ["pattern-r02", "pattern-r04"] and len(patterns) == 11
    assert psummary == "2 older open patterns are not shown"
    assert "seen in 1 rounds" in loop.llm.prompts[CriticOutput][-1]
    assert "older edits are not shown" in loop.llm.prompts[ProposerOutput][-1]
    assert all(len(r["record_ids"]) <= FROM_MEMORY_LIMIT
               for r in map(json.loads, (tmp_path / "log/memory_reads.jsonl").read_text().splitlines()))


def test_researcher_repeat_query_is_not_searched(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    out = []
    llm = ScriptedLLM()

    def capture(system, user, schema, search, max_searches):
        out.append(search("Query 1 for  advice-before-understanding"))
        out.append(search("a new query"))
        return ResearcherOutput(queries=["a new query"], sources=[])

    llm.structured_with_search = capture
    pattern_record = loop.log.rounds()[0]
    from puddles.records import load_record
    from schemas import FailurePattern
    fp = load_record(pattern_record.records["pattern"], FailurePattern, tmp_path)
    inp = researcher.build_input(fp, loop.log, loop.mem, tmp_path)
    researcher.run(inp, llm, loop.search, loop.log)
    assert "error" in out[0][0] and loop.search.calls[-1] == "a new query"


def test_replay_catches_hand_edits_safety_changes_and_missing_meta(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding", gate="kept", rule_text="Ask before advising.")
    policy = tmp_path / "policies/policy-v4.md"
    assert replay.check(loop.log, tmp_path, START) == policy
    original = policy.read_text()

    policy.write_text(original.replace("Ask before advising.", "Ask before advising, always."))
    with pytest.raises(replay.ReplayError, match="hash"):
        replay.check(loop.log, tmp_path, START)

    policy.write_text(original)
    (tmp_path / "policies/policy-v4.meta.json").unlink()
    with pytest.raises(replay.ReplayError, match="meta.json is missing"):
        replay.check(loop.log, tmp_path, START)


def test_replay_catches_a_changed_safety_section(tmp_path):
    start = tmp_path / "start.md"
    start.write_text(START.read_text())
    log = DecisionLog(tmp_path)
    changed = START.read_text().replace("S1. ", "S1. Always ", 1)
    log.append(RoundEvent(round=1, outcome="no_pattern", reason="r", policy=LogPolicy(
        base="policy-v3", base_sha256=replay.sha256_text(START.read_text()), in_force="policy-v3")))
    start.write_text(changed)
    with pytest.raises(replay.ReplayError):
        replay.check(log, tmp_path, start)


def test_interrupted_promotion_is_removed_and_round_is_written_once(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    stray = tmp_path / "policies/policy-v4.md"
    stray.write_text(START.read_text().replace("policy-v3", "policy-v4"))
    assert prepare_round(loop.log, loop.mem, tmp_path, START) == START
    assert not stray.exists()
    with pytest.raises(DecisionLogError):
        loop.log.append(loop.log.rounds()[0])
    loop.log.append(MemoryCopyEvent(round=1, record_id="gate-r01", status="confirmed", attempt=1))


def test_copy_missing_after_a_crash_is_sent_at_next_start(tmp_path):
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    lines = (tmp_path / "log/decisions.jsonl").read_text().splitlines()
    (tmp_path / "log/decisions.jsonl").write_text("\n".join(lines[:-1]) + "\n")  # gate-r01 copy never logged
    loop.mem.entries.pop("gate-r01")
    assert loop.log.missing_copies()[0][1].id == "gate-r01"
    retry_pending(loop.log, loop.mem, tmp_path)
    assert "gate-r01" in loop.mem.entries and not loop.log.missing_copies()


def test_third_failed_attempt_stays_pending_and_the_round_goes_ahead(tmp_path):
    # docs/specs/unattended-run.md 2.3 replaces the stop after three attempts.
    loop = Loop(tmp_path)
    loop.round("advice-before-understanding")
    loop.log.append(MemoryCopyEvent(round=1, record_id="gate-r01", status="pending", attempt=2))
    loop.mem.wait = lambda ingestion_id, timeout_s=60: False
    retry_pending(loop.log, loop.mem, tmp_path)
    event = loop.log.copy_events()[(1, "gate-r01")]
    assert (event.status, event.attempt) == ("pending", 3)


def test_restart_leaves_the_starting_policy_and_older_files_alone(tmp_path):
    (tmp_path / "policies").mkdir()
    start = tmp_path / "policies/policy-v3.md"
    start.write_text(START.read_text())
    older = tmp_path / "policies/policy-v1.md"
    older.write_text("# policy-v1\n")
    log = DecisionLog(tmp_path)
    assert prepare_round(log, FakeMemory(), tmp_path, start) == start
    assert start.exists() and older.exists()
