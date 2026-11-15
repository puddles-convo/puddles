"""The policy file: parse it, apply exactly one edit, and render it back.

Code applies the proposer's edit, so the Safety section and every other rule stay untouched.
"""
import os
import re
from dataclasses import dataclass, field

RULE_RE = re.compile(r"^([SC])(\d+)\.\s+(.*)$")
SAFETY_HEADER_RE = re.compile(r"^## Safety\b")
CONVERSATION_HEADER_RE = re.compile(r"^## Conversation\b")
TOKEN_CAP = 800
TYPICAL_RULE_TOKENS = 60
# Conversation rules the setting PROTECT_SETTING protects like the Safety section: C8 recommends a
# counsellor or doctor. Off by default; it was on during the run of 2026-11-15 (spec 3.3).
PROTECTED_RULE_IDS = frozenset({"C8"})
PROTECT_SETTING = "PUDDLES_PROTECT_C8"


def protected_rule_ids() -> frozenset[str]:
    """Spec 3.3: the conversation rules protected now. Read at each check; on only for `1`."""
    return PROTECTED_RULE_IDS if os.environ.get(PROTECT_SETTING) == "1" else frozenset()


class PolicyEditError(ValueError):
    pass


@dataclass
class Policy:
    version: str
    safety_header: str
    safety: dict[str, str] = field(default_factory=dict)
    conversation_header: str = "## Conversation"
    conversation: dict[str, str] = field(default_factory=dict)
    safety_lines: list[str] = field(default_factory=list)  # the section exactly as written

    def rule_ids(self) -> set[str]:
        return set(self.safety) | set(self.conversation)


def parse_policy(text: str) -> Policy:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise PolicyEditError("policy must start with '# <version>'")
    policy = Policy(version=lines[0][2:].strip(), safety_header="")
    section: dict[str, str] | None = None
    last_id = None
    for line in lines[1:]:
        if section is policy.safety and not CONVERSATION_HEADER_RE.match(line):
            policy.safety_lines.append(line)
        if SAFETY_HEADER_RE.match(line):
            policy.safety_header, section = line, policy.safety
        elif CONVERSATION_HEADER_RE.match(line):
            policy.conversation_header, section = line, policy.conversation
        elif (m := RULE_RE.match(line)) and section is not None:
            last_id = m.group(1) + m.group(2)
            section[last_id] = m.group(3).strip()
        elif line.strip() and section is not None and last_id in section:
            section[last_id] += " " + line.strip()
    if not policy.safety_header:
        raise PolicyEditError("policy has no '## Safety' section")
    while policy.safety_lines and not policy.safety_lines[-1].strip():
        policy.safety_lines.pop()
    return policy


def _safety_block(policy: Policy) -> list[str]:
    if policy.safety_lines:
        return [policy.safety_header] + policy.safety_lines
    return [policy.safety_header] + [f"{rid}. {text}" for rid, text in policy.safety.items()]


def render_policy(policy: Policy) -> str:
    out = [f"# {policy.version}", ""] + _safety_block(policy)
    out += ["", policy.conversation_header]
    out += [f"{rid}. {text}" for rid, text in policy.conversation.items()]
    return "\n".join(out) + "\n"


def safety_section(policy: Policy) -> str:
    return "\n".join(_safety_block(policy))


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def at_length_limit(policy_text: str, cap: int = TOKEN_CAP) -> bool:
    return estimate_tokens(policy_text) + TYPICAL_RULE_TOKENS > cap


def next_rule_id(used_ids: set[str]) -> str:
    numbers = [int(r[1:]) for r in used_ids if r.startswith("C") and r[1:].isdigit()]
    return f"C{max(numbers, default=0) + 1}"


def apply_edit(policy: Policy, action: str, rule_id: str | None, rule_text: str | None,
               new_version: str, used_ids: frozenset[str] | set[str] = frozenset()) -> tuple[Policy, str]:
    """Return the candidate policy and the rule ID the edit touched.

    `used_ids` holds every rule ID ever used, so a removed rule's ID is never reused.
    """
    if rule_id and rule_id.startswith("S"):
        raise PolicyEditError(f"rule {rule_id} is a protected safety rule")
    if rule_id in protected_rule_ids():
        raise PolicyEditError(f"rule {rule_id} is protected")
    conversation = dict(policy.conversation)
    if action == "add":
        if rule_id:
            raise PolicyEditError("an addition must not name a rule_id")
        if not rule_text:
            raise PolicyEditError("an addition needs rule_text")
        rule_id = next_rule_id(set(used_ids) | policy.rule_ids())
        conversation[rule_id] = rule_text.strip()
    elif action in ("reword", "remove"):
        if rule_id not in conversation:
            raise PolicyEditError(f"rule {rule_id} is not in the Conversation section")
        if action == "reword":
            if not rule_text:
                raise PolicyEditError("a rewording needs rule_text")
            conversation[rule_id] = rule_text.strip()
        else:
            if rule_text:
                raise PolicyEditError("a removal must not carry rule_text")
            del conversation[rule_id]
    else:
        raise PolicyEditError(f"unknown action {action!r}")
    candidate = Policy(version=new_version, safety_header=policy.safety_header,
                       safety=dict(policy.safety), conversation_header=policy.conversation_header,
                       conversation=conversation, safety_lines=list(policy.safety_lines))
    return candidate, rule_id


def candidate_version(base_version: str) -> str:
    m = re.fullmatch(r"policy-v(\d+)", base_version)
    if not m:
        raise PolicyEditError(f"cannot work out the next version after {base_version!r}")
    return f"policy-v{int(m.group(1)) + 1}-candidate"
