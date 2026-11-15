"""Proposer with the toolbox (step 6, PUDDLES_PROPOSER=toolbox): one action, chosen from three tools.

- edit_rule: add, reword or remove a Conversation rule (C), as the policy-only proposer does.
- add_skill_card: add, rewrite or retire a skill card (K): a named technique with when to use it,
  when not to, and its steps, grounded in the researcher's sources.
- set_examples: name a situation; code picks 1 to 3 top-rated human exchanges from the ESConv train
  split and writes them as one Examples item (E). The model never writes the examples.

Every action becomes exactly one item edit on the policy file, so the gate ("exactly one item
changed", the Safety section byte for byte, the token cap), the decision log, the replay check,
parking and memory all work unchanged. Which tool an edit used is its item's ID prefix.
`train_adapter`, `merge_adapters` and `no_change` from schemas v4 are not offered yet.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, Literal, Optional

from puddles import examples_picker
from puddles.agents import bullet_lines, proposer, system_prompt
from puddles.decision_log import DecisionLog
from puddles.llm import LLM, call_checked
from puddles.policy import Policy
from schemas import FailurePattern, PolicyEdit, ProposerInput, ProposerOutput, ResearchNote, Strict

SYSTEM = system_prompt("t6c-proposer-toolbox.md")
TOOLS = ("edit_rule", "add_skill_card", "set_examples")
KIND = {"edit_rule": "C", "add_skill_card": "K", "set_examples": "E"}
TOOL_OF = {v: k for k, v in KIND.items()}
MAX_CARD_CHARS = 700
MAX_STEPS = 5


class ConsideredTool(Strict):
    tool: Literal["edit_rule", "add_skill_card", "set_examples"]
    rejected_because: str


class ToolboxOutput(Strict):
    """The toolbox proposer's answer. Flat and typed so it works with strict structured output."""

    considered: list[ConsideredTool]  # the two tools not chosen, and why (checked by code)
    tool: Literal["edit_rule", "add_skill_card", "set_examples"]
    # edit_rule
    rule_op: Optional[Literal["add", "reword", "remove"]] = None
    rule_id: Optional[str] = None
    rule_text: Optional[str] = None
    # add_skill_card
    card_op: Optional[Literal["add", "rewrite", "retire"]] = None
    card_id: Optional[str] = None
    card_name: Optional[str] = None
    when_to_use: Optional[str] = None
    when_not_to_use: Optional[str] = None
    steps: list[str] = []
    source_urls: list[str] = []
    # set_examples
    situation: Optional[str] = None
    k: Optional[int] = None
    hypothesis: str
    prediction: str


_LEADS = re.compile(r"^(?:(?:use|not|only|do not use(?: it)?|don't use(?: it)?)\s+)?(?:when|if)\s+|^do not use(?: it)?\s+(?=to\b|for\b)",
                    re.I)


def clause(text: Optional[str]) -> str:
    """A 'when' phrase as it reads after 'Use when' / 'Not when': no repeated lead-in, lower-case start."""
    t = " ".join((text or "").split()).rstrip(".")
    t = _LEADS.sub("", t, count=1)
    return t[:1].lower() + t[1:] if t[:2] != t[:2].upper() else t  # keeps acronyms such as "ADHD"


def card_text(out: ToolboxOutput) -> str:
    steps = " ".join(f"{i}) {s.strip().rstrip('.')}." for i, s in enumerate(out.steps, 1))
    return (f"{(out.card_name or '').strip().rstrip('.')}. Use when {clause(out.when_to_use)}. "
            f"Not when {clause(out.when_not_to_use)}. Steps: {steps}")


def to_edit(out: ToolboxOutput, policy: Policy,
            example_line: Callable[[str, int], tuple[str, list[str]]] = examples_picker.example_line
            ) -> tuple[ProposerOutput, str, list[str]]:
    """The one item edit this action makes, its kind, and the ESConv IDs behind an Examples item.

    Raises ValueError with the reason when the action's fields do not fit it.
    """
    summary = f"[{out.tool}] {out.hypothesis.strip()} Expect: {out.prediction.strip()}"
    if out.tool == "edit_rule":
        if out.rule_op is None:
            raise ValueError("edit_rule needs rule_op")
        if out.rule_id and not out.rule_id.startswith("C"):
            raise ValueError("edit_rule edits C rules only; use add_skill_card for K and set_examples for E")
        return ProposerOutput(action=out.rule_op, rule_id=out.rule_id, rule_text=out.rule_text,
                              summary=summary), "C", []
    if out.tool == "add_skill_card":
        if out.card_op is None:
            raise ValueError("add_skill_card needs card_op")
        action = {"add": "add", "rewrite": "reword", "retire": "remove"}[out.card_op]
        if action != "add" and not (out.card_id or "").startswith("K"):
            raise ValueError("rewrite and retire need the card_id of an existing card (K...)")
        if action == "add" and out.card_id:
            raise ValueError("a new card has no card_id")
        text = None
        if action != "remove":
            if not (out.card_name and out.when_to_use and out.when_not_to_use):
                raise ValueError("a card needs card_name, when_to_use and when_not_to_use")
            if not 2 <= len(out.steps) <= MAX_STEPS:
                raise ValueError(f"a card needs 2 to {MAX_STEPS} steps")
            text = card_text(out)
            if "\n" in text:
                raise ValueError("card fields must be one line each")
            if len(text) > MAX_CARD_CHARS:
                raise ValueError(f"the card is {len(text)} characters; keep it under {MAX_CARD_CHARS}")
        return ProposerOutput(action=action, rule_id=out.card_id if action != "add" else None, rule_text=text,
                              summary=summary), "K", []
    # set_examples
    if not out.situation or not out.k:
        raise ValueError("set_examples needs situation and k (1 to 3)")
    text, ids = example_line(out.situation, out.k)
    existing = sorted((rid for rid in policy.conversation if rid.startswith("E")), key=lambda r: int(r[1:]))
    if len(existing) < examples_picker.MAX_E_ITEMS:
        return ProposerOutput(action="add", rule_id=None, rule_text=text, summary=summary), "E", ids
    return ProposerOutput(action="reword", rule_id=existing[0], rule_text=text, summary=summary), "E", ids


def scoreboard(log: DecisionLog) -> list[str]:
    """One line per tool: tried and kept so far, from the log (an item's ID prefix names its tool)."""
    rows = {t: [0, 0] for t in TOOLS}
    for r in log.rounds():
        if r.edit is None or r.outcome not in ("kept", "reverted"):
            continue
        tool = TOOL_OF.get(r.edit.rule_id[:1])
        if tool:
            rows[tool][0] += 1
            rows[tool][1] += r.outcome == "kept"
    return [f"{t}: tried {n}, kept {k}" for t, (n, k) in rows.items()]


def render_user(inp: ProposerInput, base_version: str, log: DecisionLog, round_: int) -> str:
    tools = list(TOOLS)
    random.Random(round_).shuffle(tools)  # no tool is favoured by its place in the list
    research_urls = [s.url for s in inp.sources]
    return (proposer.render_user(inp, base_version, log)
            + f"\n\n<tools>\n{bullet_lines(tools)}\n</tools>"
            + f"\n\n<scoreboard>\n{bullet_lines(scoreboard(log))}\n</scoreboard>"
            + f"\n\n<source_urls>\n{bullet_lines(research_urls) if research_urls else '(none this round)'}\n</source_urls>")


def check(out: ToolboxOutput, inp: ProposerInput, policy: Policy, log: DecisionLog, llm: Optional[LLM],
          example_line=examples_picker.example_line) -> list[str]:
    problems = []
    if len({c.tool for c in out.considered}) < 2:
        problems.append("considered must name both tools you did not choose, each with why")
    if out.tool in {c.tool for c in out.considered}:
        problems.append("considered lists the tools you did not choose; the chosen tool is not one of them")
    if out.tool == "add_skill_card":
        allowed = {s.url for s in inp.sources}
        if not out.source_urls and out.card_op != "retire":
            problems.append("a card needs at least one source_url from <source_urls>")
        stray = [u for u in out.source_urls if u not in allowed]
        if stray:
            problems.append(f"these source_urls are not from this round's research: {stray}")
    try:
        edit, _, _ = to_edit(out, policy, example_line)
    except (ValueError, examples_picker.ExamplesError) as exc:
        return problems + [str(exc)]
    return problems + proposer.check(edit, inp, policy, log, llm)


def run(inp: ProposerInput, base_version: str, policy: Policy, log: DecisionLog, llm: LLM,
        round_: int) -> ToolboxOutput:
    user = render_user(inp, base_version, log, round_)
    return call_checked(lambda fb: llm.structured(SYSTEM, user + fb, ToolboxOutput),
                        lambda out: check(out, inp, policy, log, llm), "proposer")


def apply(out: ToolboxOutput, policy_text: str, log: DecisionLog, pattern: FailurePattern,
          research: ResearchNote, round_: int, root: Path) -> tuple[PolicyEdit, Path]:
    from puddles.policy import parse_policy

    edit_out, kind, example_ids = to_edit(out, parse_policy(policy_text))
    if example_ids:
        edit_out = edit_out.model_copy(update={"summary": f"{edit_out.summary} ESConv train: {', '.join(example_ids)}"})
    return proposer.apply(edit_out, policy_text, log, pattern, research, round_, root, kind=kind)
