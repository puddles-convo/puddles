"""Proposer (step 6): turns the pattern and research into exactly one policy edit."""
from pathlib import Path

from puddles import parking
from puddles.agents import bullet_lines, system_prompt
from puddles.decision_log import DecisionLog
from puddles.llm import LLM, call_checked
from puddles.memory import Memory, now, recall_records
from puddles.policy import PROTECTED_RULE_IDS, Policy, apply_edit, at_length_limit, candidate_version, parse_policy, render_policy
from schemas import (
    EditSummary,
    FailurePattern,
    GateOutcome,
    PatternBrief,
    PolicyEdit,
    ProposerInput,
    ProposerOutput,
    ResearchNote,
)

SYSTEM = system_prompt("t6-proposer.md")
IDENTITY = "proposer"


def norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def build_input(policy_text: str, pattern: FailurePattern, research: ResearchNote,
                log: DecisionLog, memory: Memory, root: Path) -> ProposerInput:
    edits, summary = log.earlier_edits(pattern.pattern_id)
    history = next((p for p in log.earlier_patterns() if p.id == pattern.pattern_id), None)
    status = history.status if history else "open"
    if status == "fixed":  # this round's sighting is not logged yet: named again while its rule is active
        status = "recurring"
    # Memory adds the detail: why the gatekeeper kept or reverted each earlier edit.
    query = f"Edits tried for {pattern.name} and what happened to them."
    reasons = {r.edit_id: r.reason for r in recall_records(memory, log, query, IDENTITY, root)
               if isinstance(r, GateOutcome) and r.reason}
    edits = [EditSummary(**{**e.model_dump(), "summary": f"{e.summary} Gate: {reasons[e.id]}"})
             if e.id in reasons else e for e in edits]
    return ProposerInput(
        policy_text=policy_text, at_length_limit=at_length_limit(policy_text),
        pattern=PatternBrief(id=pattern.pattern_id, name=pattern.name, description=pattern.description,
                             status=status, rule_id=history.rule_id if history else None,
                             sightings=history.sightings + 1 if history else 1),
        sources=research.sources, earlier_edits=edits, older_edits_summary=summary)


def parked_rules(version: str, log: DecisionLog | None) -> dict[str, list[parking.FailedTry]] | None:
    """Spec 5.4: the rules parked on the policy in force; None when the setting is off."""
    if log is None or not parking.enabled():
        return None
    return parking.parked(log, version).rules


def render_user(inp: ProposerInput, base_version: str, log: DecisionLog | None = None) -> str:
    rules = parked_rules(base_version, log)
    parked_block = ""
    if rules:
        lines = [f"{rid}: {parking.tries_text(tries)}" for rid, tries in rules.items()]
        parked_block = (f'\n\n<parked_rules policy="{base_version}">\n'
                        f"These rules were edited and rejected on this policy. Do not reword or remove one; "
                        f"edit a different rule or add a new one.\n"
                        f"{bullet_lines(lines)}\n</parked_rules>")
    sources = [f"{s.technique} ({s.url})" for s in inp.sources]
    edits = [f"{e.id} for {e.pattern_id}: {e.action} {e.rule_id}, {e.outcome}. {e.summary}"
             for e in inp.earlier_edits]
    summary = f"\n{inp.older_edits_summary}" if inp.older_edits_summary else ""
    return (f'<current_policy version="{base_version}" at_length_limit="{str(inp.at_length_limit).lower()}">\n'
            f"{inp.policy_text}\n</current_policy>\n\n"
            f'<failure_pattern id="{inp.pattern.id}" status="{inp.pattern.status}">\n'
            f"{inp.pattern.name}: {inp.pattern.description}\n</failure_pattern>\n\n"
            f"<sources>\n{bullet_lines(sources)}\n</sources>\n\n"
            f"<earlier_edits>\n{bullet_lines(edits)}{summary}\n</earlier_edits>{parked_block}")


def check(out: ProposerOutput, inp: ProposerInput, policy: Policy, log: DecisionLog) -> list[str]:
    problems = []
    rules = parked_rules(policy.version, log) or {}
    if out.action in ("reword", "remove") and out.rule_id in rules:
        problems.append(f"{out.rule_id} is parked on this policy ({policy.version}): "
                        f"{parking.tries_text(rules[out.rule_id])}; edit a different rule or add a new one")
    if out.rule_id and out.rule_id.startswith("S"):
        problems.append(f"{out.rule_id} is a safety rule and cannot be edited")
    if out.rule_id in PROTECTED_RULE_IDS:
        problems.append(f"{out.rule_id} is protected; choose another rule")
    if out.action == "add" and (out.rule_id or not out.rule_text):
        problems.append("an addition has rule_text and no rule_id")
    if out.action in ("reword", "remove") and out.rule_id not in policy.conversation:
        problems.append(f"{out.rule_id} is not a Conversation rule in the current policy")
    if out.action == "reword" and not out.rule_text:
        problems.append("a rewording needs rule_text")
    if out.action == "remove" and out.rule_text:
        problems.append("a removal has no rule_text")
    if out.rule_text and "\n" in out.rule_text.strip():
        problems.append("rule_text must be one line")
    if inp.pattern.status in ("fixed", "recurring") and out.action == "add" and inp.pattern.rule_id not in rules:
        problems.append("this pattern already has an active rule; reword or remove it instead of adding")
    if inp.at_length_limit:
        if out.action == "add":
            problems.append("the policy is at its length limit; remove a rule or shorten one")
        if out.action == "reword" and len(out.rule_text or "") >= len(policy.conversation.get(out.rule_id or "", "")):
            problems.append("at the length limit a rewording must be shorter than the rule it replaces")
    for r in log.reverted_edits():
        assert r.edit is not None
        if (r.edit.action, r.edit.rule_id if out.action != "add" else None, norm(r.edit.rule_text)) == \
                (out.action, out.rule_id, norm(out.rule_text)):
            problems.append(f"this is the same edit as {r.edit.id}, which was reverted")
    return problems


def run(inp: ProposerInput, base_version: str, policy: Policy, log: DecisionLog, llm: LLM) -> ProposerOutput:
    user = render_user(inp, base_version, log)
    return call_checked(lambda fb: llm.structured(SYSTEM, user + fb, ProposerOutput),
                        lambda out: check(out, inp, policy, log), "proposer")


def apply(out: ProposerOutput, policy_text: str, log: DecisionLog, pattern: FailurePattern,
          research: ResearchNote, round_: int, root: Path) -> tuple[PolicyEdit, Path]:
    """Code applies the one edit and writes the candidate file; the model never writes the policy."""
    policy = parse_policy(policy_text)
    new_version = candidate_version(policy.version)
    candidate, rule_id = apply_edit(policy, out.action, out.rule_id, out.rule_text, new_version,
                                    used_ids=log.used_rule_ids())
    rel = f"policies/{new_version}-r{round_:02d}.md"  # one file per round, so logged hashes stay valid
    (root / "policies").mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(render_policy(candidate))
    edit = PolicyEdit(id=f"edit-r{round_:02d}", written_by=IDENTITY, created_at=now(),
                      parent_ids=[pattern.id, research.id], round=round_, pattern_id=pattern.pattern_id,
                      research_id=research.id, base_policy_version=policy.version,
                      new_policy_version=new_version, policy_path=rel, summary=out.summary,
                      action=out.action, rule_id=rule_id,
                      rule_text=out.rule_text.strip() if out.rule_text else None)
    return edit, root / rel
