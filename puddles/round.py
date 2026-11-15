"""Steps 4 to 6 of a round, then the log entry and the copies into past.dev.

Within a round the critic, researcher and proposer hand records straight to each other.
Only after the round is logged are its records copied into memory, for later rounds to recall.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import httpx

from puddles.agents import critic, proposer, researcher
from puddles.decision_log import DecisionLog
from puddles.llm import LLM
from puddles.memory import READS_PATH, Memory, PastError, entry_for, now
from puddles.policy import parse_policy, render_policy
from puddles.records import load_record, save_record, sha256_text
from schemas import (
    FailurePattern,
    GateOutcome,
    LogEdit,
    LogPattern,
    LogPolicy,
    MemoryCopyEvent,
    PolicyEdit,
    Record,
    RecordRef,
    ResearchNote,
    RoundEvent,
    ScoredConversation,
    Scores,
)

MODELS: dict[str, type[Record]] = {
    "pattern": FailurePattern, "research": ResearchNote, "edit": PolicyEdit, "outcome": GateOutcome,
}


@dataclass
class RoundResult:
    round: int
    outcome: Literal["candidate", "no_pattern", "no_evidence"]
    base_policy_path: Path
    reason: str = ""
    pattern: Optional[FailurePattern] = None
    status_before: Literal["open", "fixed", "recurring"] = "open"
    research: Optional[ResearchNote] = None
    edit: Optional[PolicyEdit] = None
    candidate_path: Optional[Path] = None
    refs: dict[str, RecordRef] = field(default_factory=dict)


def policy_in_force(root: Path, fallback: Path) -> Path:
    """The newest promoted policy under root/policies, or the starting policy."""
    promoted = sorted((root / "policies").glob("policy-v*.md"), key=lambda p: p.stat().st_mtime)
    promoted = [p for p in promoted if "candidate" not in p.name]
    return promoted[-1] if promoted else fallback


NO_EVIDENCE_REASON = "No trustworthy source was found for this pattern."


def run_critic(round_: int, conversations: list[ScoredConversation], policy_path: Path,
               log: DecisionLog, memory: Memory, llm: LLM, root: Path) -> RoundResult:
    """Step 4. A new result, outcome no_pattern; with the pattern set if the critic found one."""
    policy_text = policy_path.read_text()
    result = RoundResult(round=round_, outcome="no_pattern", base_policy_path=policy_path)
    c_in = critic.build_input(round_, policy_text, conversations, log, memory, root)
    c_out = critic.run(c_in, llm)
    pattern = critic.to_record(c_out, c_in)
    if pattern is None:
        result.reason = c_out.reason
        return result
    result.pattern = pattern
    statuses = [p.status for p in c_in.earlier_patterns if p.id == pattern.id and p.status]
    result.status_before = statuses[0] if statuses else "open"
    result.refs["pattern"] = save_record(pattern, round_, root)
    return result


def run_researcher(result: RoundResult, log: DecisionLog, memory: Memory, llm: LLM,
                   search: Callable[[str], list[dict]], root: Path) -> RoundResult:
    """Step 5. Sets the research note; with no sources the outcome becomes no_evidence."""
    assert result.pattern is not None
    pattern, round_ = result.pattern, result.round
    r_in = researcher.build_input(pattern, log, memory, root)
    r_out, searches = researcher.run(r_in, llm, search)
    research = researcher.to_record(r_out, pattern, round_)
    researcher.save_search_log(searches, round_, root)
    result.research = research
    result.refs["research"] = save_record(research, round_, root)
    if not research.sources:
        result.outcome, result.reason = "no_evidence", NO_EVIDENCE_REASON
    return result


def run_proposer(result: RoundResult, log: DecisionLog, memory: Memory, llm: LLM,
                 root: Path) -> RoundResult:
    """Step 6. Writes the candidate policy; the outcome becomes candidate."""
    assert result.pattern is not None and result.research is not None
    pattern, research, round_ = result.pattern, result.research, result.round
    policy_text = result.base_policy_path.read_text()
    p_in = proposer.build_input(policy_text, pattern, research, log, memory, root)
    policy = parse_policy(policy_text)
    p_out = proposer.run(p_in, policy.version, policy, log, llm)
    edit, candidate = proposer.apply(p_out, policy_text, log, pattern, research, round_, root)
    result.edit, result.candidate_path, result.outcome = edit, candidate, "candidate"
    result.refs["edit"] = save_record(edit, round_, root)
    return result


def run_agents(round_: int, conversations: list[ScoredConversation], policy_path: Path,
               log: DecisionLog, memory: Memory, llm: LLM, search: Callable[[str], list[dict]],
               root: Path) -> RoundResult:
    result = run_critic(round_, conversations, policy_path, log, memory, llm, root)
    if result.pattern is None:
        return result
    run_researcher(result, log, memory, llm, search, root)
    if result.outcome == "no_evidence":
        return result
    return run_proposer(result, log, memory, llm, root)


def stand_in_gate(result: RoundResult, decision: Literal["kept", "reverted"]) -> GateOutcome:
    """For development until the gatekeeper (J7) exists. It is labelled so in memory and the log."""
    assert result.edit is not None
    base = parse_policy(result.base_policy_path.read_text()).version
    kept_version = result.edit.new_policy_version.replace("-candidate", "")
    return GateOutcome(
        id=f"gate-r{result.round:02d}", written_by="gatekeeper", created_at=now(),
        parent_ids=[result.edit.id], round=result.round, edit_id=result.edit.id, decision=decision,
        before=Scores(empathy=0, relevance=0), after=Scores(empathy=0, relevance=0),
        safety_passed=decision == "kept", policy_in_force=kept_version if decision == "kept" else base,
        reason="Stand-in gate for development; no conversations were scored.")


def extras_for(record: Record, log: DecisionLog, root: Path) -> dict:
    """Values a memory sentence needs that live in other records."""
    if isinstance(record, ResearchNote):
        names = {r.pattern.id: r.pattern.name for r in log.rounds() if r.pattern}
        return {"pattern_name": names.get(record.pattern_id, "")}
    if isinstance(record, GateOutcome):
        ref = log.ref(record.edit_id)
        edit = load_record(ref, PolicyEdit, root) if ref else None
        event = next((r for r in log.rounds() if r.round == record.round), None)
        return {"rule_id": edit.rule_id if edit else None, "summary": edit.summary if edit else "",
                "noise": event.scores.noise if event and event.scores else None}
    return {}


def copy_to_memory(record: Record, ref: RecordRef, round_: int, memory: Memory, log: DecisionLog,
                   root: Path, attempt: int = 1, wait_s: float = 60) -> str:
    entry = entry_for(record, ref, round_, **extras_for(record, log, root))
    try:
        confirmed = memory.wait(memory.ingest(entry), timeout_s=wait_s)
    except (PastError, httpx.HTTPError):
        confirmed = False  # logged as pending, so retry_pending resends it
    status: Literal["confirmed", "pending"] = "confirmed" if confirmed else "pending"
    log.append(MemoryCopyEvent(round=round_, record_id=record.id, status=status, attempt=attempt))
    return status


MAX_COPY_ATTEMPTS = 3


def retry_pending(log: DecisionLog, memory: Memory, root: Path) -> None:
    """Run at the start of a round: resend copies that were still pending; stop the loop after three attempts."""
    pending = log.pending_copies()
    stuck = [f"{e.record_id} (round {e.round})" for e in pending if e.attempt >= MAX_COPY_ATTEMPTS]
    if stuck:
        raise PastError(f"memory copies still pending after {MAX_COPY_ATTEMPTS} attempts: {', '.join(stuck)}")
    for event in pending:
        for kind, model in MODELS.items():
            round_event = next((r for r in log.rounds() if r.round == event.round), None)
            ref = round_event.records.get(kind) if round_event else None
            if ref and ref.id == event.record_id:
                record = load_record(ref, model, root)
                if record:
                    copy_to_memory(record, ref, event.round, memory, log, root, attempt=event.attempt + 1)


def finish_round(result: RoundResult, gate: Optional[GateOutcome], log: DecisionLog, memory: Memory,
                 root: Path) -> RoundEvent:
    base_text = result.base_policy_path.read_text()
    base_version = parse_policy(base_text).version
    policy = LogPolicy(base=base_version, base_sha256=sha256_text(base_text), in_force=base_version)
    outcome: Literal["kept", "reverted", "no_pattern", "no_evidence"]
    outcome = "no_evidence" if result.outcome == "no_evidence" else "no_pattern"
    reason = result.reason
    if gate is not None:
        assert result.candidate_path is not None and result.edit is not None
        result.refs["outcome"] = save_record(gate, result.round, root)
        outcome, reason = gate.decision, gate.reason
        candidate = parse_policy(result.candidate_path.read_text())
        if gate.decision == "kept":
            candidate.version = gate.policy_in_force
            text = render_policy(candidate)
            (root / "policies" / f"{candidate.version}.md").write_text(text)
            policy.new, policy.new_sha256, policy.in_force = candidate.version, sha256_text(text), candidate.version
        else:
            policy.new, policy.new_sha256 = candidate.version, sha256_text(result.candidate_path.read_text())
    p = result.pattern
    event = RoundEvent(
        round=result.round, outcome=outcome, records=result.refs, policy=policy, reason=reason or outcome,
        pattern=LogPattern(id=p.id, name=p.name, seen_in=p.seen_in, out_of=p.out_of,
                           status_before=result.status_before) if p else None,
        edit=LogEdit(id=result.edit.id, action=result.edit.action or "add", rule_id=result.edit.rule_id or "",
                     rule_text=result.edit.rule_text) if result.edit and gate else None)
    log.append(event)
    for kind, model in MODELS.items():
        if kind in result.refs:
            record = load_record(result.refs[kind], model, root)
            assert record is not None, f"{kind} record changed on disk"
            copy_to_memory(record, result.refs[kind], result.round, memory, log, root)
    return event


def memory_trace(log: DecisionLog, root: Path) -> list[dict]:
    """For each record copied into memory: who wrote it and which agents later read it."""
    import json

    reads_path = root / READS_PATH
    reads = [json.loads(line) for line in reads_path.read_text().splitlines()] if reads_path.exists() else []
    rows = []
    for event in log.rounds():
        for kind, ref in event.records.items():
            record = load_record(ref, MODELS[kind], root)
            readers = sorted({r["reader"] for r in reads if ref.id in r["record_ids"]})
            rows.append({"record": ref.id, "round": event.round,
                         "written_by": record.written_by if record else "?", "read_by": readers})
    return rows
