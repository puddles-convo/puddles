"""Critic (step 4): names one failure pattern from the lowest-scoring conversations."""
from pathlib import Path
from typing import Optional

from puddles.agents import bullet_lines, system_prompt
from puddles.decision_log import DecisionLog
from puddles.llm import LLM, call_checked
from puddles.memory import Memory, describe, now, recall_records
from puddles import parking
from schemas import CriticInput, CriticOutput, FailurePattern, PolicyEdit, ResearchNote, ScoredConversation

SYSTEM = system_prompt("t4-critic.md")
IDENTITY = "critic"


def select_for_critic(scored: list[ScoredConversation], n: int) -> list[ScoredConversation]:
    """The `n` lowest-scoring conversations by (empathy + relevance) / 2, ties by conversation ID.
    All of them when there are fewer than `n`."""
    if n < 1:
        raise ValueError(f"conversations_for_critic must be at least 1, got {n}")
    ranked = sorted(scored, key=lambda c: ((c.empathy + c.relevance) / 2, c.conversation_id))
    return ranked[:n]


def memory_query(conversations: list[ScoredConversation]) -> str:
    notes = " ".join(c.note for c in conversations[:5])
    return f"Earlier failure patterns and what happened to them, related to: {notes}"


def build_input(round_: int, policy_text: str, conversations: list[ScoredConversation],
                log: DecisionLog, memory: Memory, root: Path) -> CriticInput:
    recalled = recall_records(memory, log, memory_query(conversations), IDENTITY, root)
    found = {r.pattern_id for r in recalled if isinstance(r, (FailurePattern, ResearchNote, PolicyEdit))}
    found = {pid for p in found if (pid := log.pattern_identity(p))}
    patterns, summary = log.critic_patterns(recalled=found)
    return CriticInput(round=round_, policy_text=policy_text, earlier_patterns=patterns,
                       older_patterns_summary=summary, from_memory=[describe(r) for r in recalled],
                       conversations=conversations)


def policy_version(policy_text: str) -> Optional[str]:
    first = policy_text.splitlines()[0] if policy_text else ""
    return first[2:].strip() if first.startswith("# ") else None


def parked_patterns(inp: CriticInput, log: Optional[DecisionLog]) -> Optional[parking.Parked]:
    """Spec 5.3: the patterns parked on the policy in force; None when the setting is off."""
    version = policy_version(inp.policy_text)
    if log is None or version is None or not parking.enabled():
        return None
    return parking.parked(log, version)


def render_user(inp: CriticInput, log: Optional[DecisionLog] = None) -> str:
    park = parked_patterns(inp, log)
    marks = park.patterns if park else {}
    patterns = [f"{p.id}: {p.name}, {p.status}, rule {p.rule_id}, seen in {p.sightings} rounds"
                + (f", PARKED on {park.version}" if park and p.id in marks else "")
                for p in inp.earlier_patterns]
    parked_block = ""
    if park and marks:
        lines = [f"{pid}: {parking.tries_text(tries)}" for pid, tries in marks.items()]
        parked_block = (f'<parked_patterns policy="{park.version}">\n'
                        f"These patterns were tried and rejected on this policy. Do not name one as "
                        f"same_as_pattern_id; name a different pattern, or none.\n"
                        f"{bullet_lines(lines)}\n</parked_patterns>\n\n")
    convs = []
    for c in inp.conversations:
        turns = "\n".join(f"{i}. {t.speaker}: {t.text}" for i, t in enumerate(c.turns, 1))
        convs.append(f'<conversation id="{c.conversation_id}" empathy="{c.empathy}" '
                     f'relevance="{c.relevance}">\n<judge_note>{c.note}</judge_note>\n{turns}\n</conversation>')
    summary = f"\n{inp.older_patterns_summary}" if inp.older_patterns_summary else ""
    return (f"<current_policy>\n{inp.policy_text}\n</current_policy>\n\n"
            f"<earlier_patterns>\n{bullet_lines(patterns)}{summary}\n</earlier_patterns>\n\n"
            f"{parked_block}"
            f"<from_memory>\n{bullet_lines(inp.from_memory)}\n</from_memory>\n\n"
            f"<conversations>\n" + "\n".join(convs) + "\n</conversations>")


def check(out: CriticOutput, inp: CriticInput, log: DecisionLog) -> list[str]:
    if not out.found:
        return [] if out.reason else ["found is false but no reason is given"]
    given = {c.conversation_id for c in inp.conversations}
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
    if out.same_as_pattern_id and log.pattern_identity(out.same_as_pattern_id) is None:
        problems.append(f"same_as_pattern_id {out.same_as_pattern_id} is not an earlier pattern")
    park = parked_patterns(inp, log) if out.same_as_pattern_id else None
    identity = log.pattern_identity(out.same_as_pattern_id) if park and out.same_as_pattern_id else None
    if park and identity in park.patterns:
        problems.append(f"same_as_pattern_id {out.same_as_pattern_id} is parked on this policy ({park.version}): "
                        f"{parking.tries_text(park.patterns[identity])}; name a different pattern, or none")
    return problems


def run(inp: CriticInput, llm: LLM, log: DecisionLog) -> CriticOutput:
    user = render_user(inp, log)
    return call_checked(lambda fb: llm.structured(SYSTEM, user + fb, CriticOutput),
                        lambda out: check(out, inp, log), "critic")


def to_record(out: CriticOutput, inp: CriticInput, log: DecisionLog) -> Optional[FailurePattern]:
    """Every sighting gets its own ID. A recurrence points to the pattern's first sighting."""
    if not out.found:
        return None
    same_as = (log.pattern_identity(out.same_as_pattern_id) if out.same_as_pattern_id
               else log.identity_by_name(out.name or ""))
    return FailurePattern(
        id=f"pattern-r{inp.round:02d}", written_by=IDENTITY,
        created_at=now(), parent_ids=[c.conversation_id for c in inp.conversations],
        round=inp.round, name=out.name or "", description=out.description or "", seen_in=out.seen_in,
        out_of=out.out_of, example_conversation_ids=out.example_conversation_ids,
        same_as_pattern_id=same_as)
