"""policies/<version>.meta.json: where each rule of a policy in force came from.

Kept out of the policy text so it costs no prompt tokens. The starting policy gets one when a
run's root folder is first set up; each promoted policy gets one when its edit is kept. A
reverted round writes none.
"""
from pathlib import Path
from typing import Optional

from puddles.policy import parse_policy, safety_section
from puddles.records import sha256_text
from schemas import LearnedRule, PolicyEdit, PolicyMeta


class MetaError(RuntimeError):
    pass


def meta_path(root: Path, version: str) -> Path:
    return Path(root) / "policies" / f"{version}.meta.json"


def safety_sha256(policy_text: str) -> str:
    """The hash of the Safety section. It is the same in every version."""
    return sha256_text(safety_section(parse_policy(policy_text)))


def starting_meta(policy_text: str) -> PolicyMeta:
    """No parent, every Conversation rule a base rule, nothing learned."""
    policy = parse_policy(policy_text)
    return PolicyMeta(policy_version=policy.version, parent=None, policy_sha256=sha256_text(policy_text),
                      safety_sha256=safety_sha256(policy_text), base_rules=list(policy.conversation),
                      learned_rules={})


def promoted_meta(parent: PolicyMeta, edit: PolicyEdit, policy_text: str) -> PolicyMeta:
    """The parent's provenance with one kept edit applied. Refuses a changed Safety section."""
    policy = parse_policy(policy_text)
    if safety_sha256(policy_text) != parent.safety_sha256:
        raise MetaError(f"{policy.version}: the Safety section differs from {parent.policy_version}'s")
    if not edit.action or not edit.rule_id:
        raise MetaError(f"{edit.id} names no action or rule, so its rule cannot be traced")
    base = [rule for rule in parent.base_rules if rule != edit.rule_id]
    learned = {rule: where for rule, where in parent.learned_rules.items() if rule != edit.rule_id}
    if edit.action != "remove":
        learned[edit.rule_id] = LearnedRule(
            edit=edit.id, action=edit.action, pattern=edit.pattern_id, round=edit.round,
            text_sha256=sha256_text((edit.rule_text or "").strip()))
    if set(base) | set(learned) != set(policy.conversation):
        raise MetaError(f"{policy.version}: its rules {sorted(policy.conversation)} are not the parent's "
                        f"rules with {edit.id} applied")
    return PolicyMeta(policy_version=policy.version, parent=parent.policy_version,
                      policy_sha256=sha256_text(policy_text), safety_sha256=parent.safety_sha256,
                      base_rules=base, learned_rules=learned)


def load_meta(root: Path, version: str) -> Optional[PolicyMeta]:
    path = meta_path(root, version)
    return PolicyMeta.model_validate_json(path.read_text()) if path.exists() else None


def write_meta(root: Path, meta: PolicyMeta) -> Path:
    path = meta_path(root, meta.policy_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(meta.model_dump_json(indent=2) + "\n")
    return path


def ensure_starting_meta(root: Path, policy_path: Path) -> Path:
    """Write the starting policy's file if the root does not have it yet. Never overwrites."""
    text = Path(policy_path).read_text()
    path = meta_path(root, parse_policy(text).version)
    return path if path.exists() else write_meta(root, starting_meta(text))
