"""Critic (step 4): names one failure pattern from the lowest-scoring conversations."""
from pathlib import Path
from typing import Optional

from puddles.agents import bullet_lines, system_prompt
from puddles.decision_log import DecisionLog
from puddles.llm import LLM, call_checked
from puddles.memory import Memory, describe, now, recall_records
from schemas import CriticInput, CriticOutput, FailurePattern, ScoredConversation

SYSTEM = system_prompt("t4-critic.md")
IDENTITY = "critic"


def memory_query(conversations: list[ScoredConversation]) -> str:
    notes = " ".join(c.note for c in conversations[:5])
    return f"Earlier failure patterns and what happened to them, related to: {notes}"


def build_input(round_: int, policy_text: str, conversations: list[ScoredConversation],
                log: DecisionLog, memory: Memory, root: Path) -> CriticInput:
    recalled = recall_records(memory, log, memory_query(conversations), IDENTITY, root)
    return CriticInput(round=round_, policy_text=policy_text, earlier_patterns=log.earlier_patterns(),
                       from_memory=[describe(r) for r in recalled], conversations=conversations)


def render_user(inp: CriticInput) -> str:
    patterns = [f"{p.id}: {p.name}, {p.status}, rule {p.rule_id}" for p in inp.earlier_patterns]
    convs = []
    for c in inp.conversations:
        turns = "\n".join(f"{i}. {t.speaker}: {t.text}" for i, t in enumerate(c.turns, 1))
        convs.append(f'<conversation id="{c.conversation_id}" empathy="{c.empathy}" '
                     f'relevance="{c.relevance}">\n<judge_note>{c.note}</judge_note>\n{turns}\n</conversation>')
    return (f"<current_policy>\n{inp.policy_text}\n</current_policy>\n\n"
            f"<earlier_patterns>\n{bullet_lines(patterns)}\n</earlier_patterns>\n\n"
            f"<from_memory>\n{bullet_lines(inp.from_memory)}\n</from_memory>\n\n"
            f"<conversations>\n" + "\n".join(convs) + "\n</conversations>")


def check(out: CriticOutput, inp: CriticInput) -> list[str]:
    if not out.found:
        return [] if out.reason else ["found is false but no reason is given"]
    given = {c.conversation_id for c in inp.conversations}
    known = {p.id for p in inp.earlier_patterns}
    problems = []
    if not out.name or not out.description:
        problems.append("a found pattern needs a name and a description")
    if out.out_of != len(inp.conversations):
        problems.append(f"out_of must be {len(inp.conversations)}, the number of conversations given")
    if not 1 <= out.seen_in <= len(inp.conversations):
        problems.append("seen_in must be between 1 and the number of conversations given")
    unknown = [i for i in out.example_conversation_ids if i not in given]
    if unknown:
        problems.append(f"example IDs not in the input: {unknown}")
    if not out.example_conversation_ids:
        problems.append("give at least one example conversation ID")
    if out.same_as_pattern_id and out.same_as_pattern_id not in known:
        problems.append(f"same_as_pattern_id {out.same_as_pattern_id} is not an earlier pattern")
    return problems


def run(inp: CriticInput, llm: LLM) -> CriticOutput:
    user = render_user(inp)
    return call_checked(lambda fb: llm.structured(SYSTEM, user + fb, CriticOutput),
                        lambda out: check(out, inp), "critic")


def to_record(out: CriticOutput, inp: CriticInput) -> Optional[FailurePattern]:
    if not out.found:
        return None
    return FailurePattern(
        id=out.same_as_pattern_id or f"pattern-r{inp.round:02d}", written_by=IDENTITY,
        created_at=now(), parent_ids=[c.conversation_id for c in inp.conversations],
        round=inp.round, name=out.name or "", description=out.description or "", seen_in=out.seen_in,
        out_of=out.out_of, example_conversation_ids=out.example_conversation_ids,
        same_as_pattern_id=out.same_as_pattern_id)
