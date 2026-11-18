"""Loading and hashing policy files."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from loop.errors import PolicyError

RULE_RE = re.compile(r"^([SCKE]\d+)\. (.+)$")  # K skill cards, E examples: puddles/policy.py


def sha256_text(text: str) -> str:
    """Hex SHA-256 digest of the UTF-8 bytes of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Policy:
    """A parsed policy file."""

    version: str
    text: str
    safety_text: str
    rules: dict[str, str]

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)

    @property
    def safety_sha256(self) -> str:
        return sha256_text(self.safety_text)


def load_policy(path: str | PathLike[str]) -> Policy:
    """Read and validate a policy file."""
    text = Path(path).read_bytes().decode("utf-8")
    # Each line as (offset in text, content without its line ending).
    lines: list[tuple[int, str]] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        lines.append((offset, raw.splitlines()[0]))
        offset += len(raw)
    if not lines or not lines[0][1].startswith("# "):
        raise PolicyError(f"{path}: first line must be a '# ' title")
    version = lines[0][1][2:].strip()

    rules: dict[str, str] = {}
    safety_start: int | None = None
    safety_end = 0
    safety_rules = 0
    in_safety = False
    in_section = False
    for number, (start, line) in enumerate(lines[1:], start=2):
        if line.startswith("## "):
            in_section = True
            in_safety = line == "## Safety" or line.startswith("## Safety ")
            if in_safety:
                if safety_start is not None:
                    raise PolicyError(f"{path}: more than one Safety section")
                safety_start, safety_end = start, start + len(line)
            continue
        if not line.strip():
            continue
        if in_safety:
            safety_end = start + len(line)
        match = RULE_RE.match(line)
        if not in_section or not match:
            raise PolicyError(f"{path}: line {number} is not a rule: {line!r}")
        rule_id, rule_text = match.groups()
        if in_safety != rule_id.startswith("S"):
            raise PolicyError(f"{path}: line {number}: rule {rule_id} is in the wrong section; "
                              "only S rules belong in Safety, and S rules nowhere else")
        if rule_id in rules:
            raise PolicyError(f"{path}: rule {rule_id} appears twice")
        rules[rule_id] = rule_text
        safety_rules += in_safety

    if safety_start is None:
        raise PolicyError(f"{path}: no '## Safety' section")
    if not safety_rules:
        raise PolicyError(f"{path}: the Safety section has no rules")
    return Policy(version=version, text=text, safety_text=text[safety_start:safety_end],
                  rules=rules)
