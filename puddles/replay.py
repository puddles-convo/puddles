"""The check that proves the policy and the log agree, and where the policy in force comes from.

The log is replayed from the starting policy: every kept addition, rewording and removal in
order. The policy file in force, its Safety section and its .meta.json must match the result.
Any mismatch raises ReplayError, and the loop stops.
"""
import json
import re
from pathlib import Path
from typing import Optional

from puddles.decision_log import DecisionLog
from puddles.policy import Policy, parse_policy, safety_section
from puddles.records import sha256_text
from schemas import LearnedRule, PolicyMeta, RoundEvent


class ReplayError(RuntimeError):
    pass


def policy_path(root: Path, version: str) -> Path:
    return root / "policies" / f"{version}.md"


def meta_path(root: Path, version: str) -> Path:
    return root / "policies" / f"{version}.meta.json"


def expected_rules(rounds: list[RoundEvent], start: Policy) -> dict[str, str]:
    """The Conversation rules that should be active after the kept edits, with their exact text."""
    rules = dict(start.conversation)
    for r in rounds:
        if r.outcome != "kept" or r.edit is None:
            continue
        rid, text = r.edit.rule_id, (r.edit.rule_text or "").strip()
        if r.edit.action == "add":
            if rid in rules:
                raise ReplayError(f"round {r.round} adds {rid}, which is already active")
            rules[rid] = text
        elif rid not in rules:
            raise ReplayError(f"round {r.round} edits {rid}, which is not active")
        elif r.edit.action == "reword":
            rules[rid] = text
        else:
            del rules[rid]
    return rules


def expected_meta(rounds: list[RoundEvent], start: Policy, policy: Policy, text: str) -> PolicyMeta:
    """Where each rule of `policy` came from: a base rule, or the last kept edit that touched it."""
    base = [rid for rid in start.conversation]
    learned: dict[str, LearnedRule] = {}
    parent = start.version
    for r in rounds:
        if r.outcome != "kept" or r.edit is None or r.pattern is None:
            continue
        rid = r.edit.rule_id
        base = [b for b in base if b != rid]
        learned.pop(rid, None)
        if r.edit.action != "remove":
            learned[rid] = LearnedRule(edit=r.edit.id, action=r.edit.action, pattern=r.pattern.pattern_id,
                                       round=r.round, text_sha256=sha256_text((r.edit.rule_text or "").strip()))
        if r.policy.in_force == policy.version:
            parent = r.policy.base
    return PolicyMeta(policy_version=policy.version, parent=parent, policy_sha256=sha256_text(text),
                      safety_sha256=sha256_text(safety_section(start)), base_rules=base, learned_rules=learned)


def write_meta(root: Path, rounds: list[RoundEvent], start_path: Path, version: str) -> Path:
    """Write <version>.meta.json beside a promoted policy, from the log plus the round being finished."""
    text = policy_path(root, version).read_bytes().decode("utf-8")
    meta = expected_meta(rounds, parse_policy(start_path.read_bytes().decode("utf-8")), parse_policy(text), text)
    path = meta_path(root, version)
    path.write_text(meta.model_dump_json(indent=2) + "\n")
    return path


def in_force(log: DecisionLog, root: Path, start_path: Path) -> tuple[str, Path, Optional[str]]:
    """The version in force per the last round event, its file, and the hash the log gave it."""
    rounds = log.rounds()
    start_text = start_path.read_bytes().decode("utf-8")
    if not rounds:
        return parse_policy(start_text).version, start_path, sha256_text(start_text)
    version = rounds[-1].policy.in_force
    if version == parse_policy(start_text).version:
        return version, start_path, sha256_text(start_text)
    promoted = [r for r in rounds if r.outcome == "kept" and r.policy.new == version]
    return version, policy_path(root, version), promoted[-1].policy.new_sha256 if promoted else None


def policy_in_force(log: DecisionLog, root: Path, start_path: Path) -> Path:
    """The policy in force comes from the log, and its file must have the hash the log recorded."""
    version, path, digest = in_force(log, root, start_path)
    if not path.exists():
        raise ReplayError(f"{version} is in force per the log, but {path} does not exist")
    if digest is None or sha256_text(path.read_bytes().decode("utf-8")) != digest:
        raise ReplayError(f"{path} does not have the hash the log recorded for {version}")
    return path


def check(log: DecisionLog, root: Path, start_path: Path) -> Path:
    """Replay the log and compare it with the policy in force. Returns the policy file in force."""
    rounds = log.rounds()
    start_text = start_path.read_bytes().decode("utf-8")
    start = parse_policy(start_text)
    previous, previous_sha = start.version, sha256_text(start_text)
    for r in rounds:
        if (r.policy.base, r.policy.base_sha256) != (previous, previous_sha):
            raise ReplayError(f"round {r.round} starts from {r.policy.base}, but {previous} was in force")
        if r.outcome == "kept":
            if not (r.policy.new and r.policy.new_sha256 and r.policy.in_force == r.policy.new):
                raise ReplayError(f"round {r.round} was kept but names no new policy in force")
            previous, previous_sha = r.policy.new, r.policy.new_sha256
        elif r.policy.in_force != r.policy.base:
            raise ReplayError(f"round {r.round} was {r.outcome} but changed the policy in force")

    path = policy_in_force(log, root, start_path)
    text = path.read_bytes().decode("utf-8")
    policy = parse_policy(text)
    expected = expected_rules(rounds, start)
    if set(policy.conversation) != set(expected):
        raise ReplayError(f"{path} has rules {sorted(policy.conversation)}; the log gives {sorted(expected)}")
    changed = [rid for rid, t in expected.items() if sha256_text(policy.conversation[rid]) != sha256_text(t)]
    if changed:
        raise ReplayError(f"{path}: the text of {changed} differs from the log")
    if sha256_text(safety_section(policy)) != sha256_text(safety_section(start)):
        raise ReplayError(f"{path}: the Safety section differs from the starting policy's")
    if path != start_path:
        meta_file = meta_path(root, policy.version)
        if not meta_file.exists():
            raise ReplayError(f"{meta_file} is missing")
        meta = PolicyMeta.model_validate(json.loads(meta_file.read_text()))
        if meta != expected_meta(rounds, start, policy, text):
            raise ReplayError(f"{meta_file} does not match the log")
    return path


def _number(version: str) -> int:
    match = re.fullmatch(r"policy-v(\d+)", version)
    return int(match.group(1)) if match else -1


def undo_interrupted_promotion(log: DecisionLog, root: Path, start_path: Path) -> list[Path]:
    """Remove a promoted policy no round event knows about: a crash between the write and the log.

    Only versions newer than the one in force can come from such a crash; older or unrelated
    files in policies/, and the starting policy itself, are left alone.
    """
    current, _, _ = in_force(log, root, start_path)
    known = {r.policy.new for r in log.rounds() if r.outcome == "kept" and r.policy.new}
    removed = []
    for path in sorted((root / "policies").glob("policy-v*.md")):
        if path.resolve() == start_path.resolve() or path.stem in known or _number(path.stem) <= _number(current):
            continue
        for stray in (path, meta_path(root, path.stem)):
            if stray.exists():
                stray.unlink()
                removed.append(stray)
    return removed
