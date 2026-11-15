import hashlib
import re
from pathlib import Path

import pytest

from loop.errors import PolicyError
from loop.policy import load_policy, sha256_text

ROOT = Path(__file__).resolve().parents[2]

GOOD = """# policy-test

## Safety (protected: no edit may change this section)
S1. First safety rule.
S2. Second safety rule.


## Conversation
C1. First conversation rule.
C3. Another rule.
"""


def _write(tmp_path, text):
    path = tmp_path / "p.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_policy_parses_parts(tmp_path):
    policy = load_policy(_write(tmp_path, GOOD))
    assert policy.version == "policy-test"
    assert policy.text == GOOD
    assert policy.safety_text == ("## Safety (protected: no edit may change this section)\n"
                                  "S1. First safety rule.\nS2. Second safety rule.")
    assert list(policy.rules.items()) == [
        ("S1", "First safety rule."), ("S2", "Second safety rule."),
        ("C1", "First conversation rule."), ("C3", "Another rule."),
    ]
    assert policy.sha256 == hashlib.sha256(GOOD.encode()).hexdigest()
    assert policy.safety_sha256 == sha256_text(policy.safety_text)


def test_safety_text_keeps_crlf_bytes(tmp_path):
    path = tmp_path / "p.md"
    path.write_bytes(b"# p\r\n## Safety\r\nS1. a\r\nS2. b\r\n")
    policy = load_policy(path)
    assert policy.text == "# p\r\n## Safety\r\nS1. a\r\nS2. b\r\n"
    assert policy.safety_text == "## Safety\r\nS1. a\r\nS2. b"
    assert policy.rules == {"S1": "a", "S2": "b"}


def test_safety_hash_differs_when_only_line_endings_differ(tmp_path):
    lf, crlf = tmp_path / "lf.md", tmp_path / "crlf.md"
    lf.write_bytes(b"# p\n## Safety\nS1. a\nS2. b\n")
    crlf.write_bytes(b"# p\n## Safety\r\nS1. a\r\nS2. b\n")
    assert load_policy(lf).safety_sha256 != load_policy(crlf).safety_sha256


@pytest.mark.parametrize("raw", [
    b"# p\n## Safety\nS1. a\n",
    b"# p\n## Safety\nS1. a",                                         # no final newline
    b"# p\r\n\r\n## Safety (protected)\r\nS1. a  \r\n\r\nS2. b\r\n \t\r\n## Conversation\r\nC1. c\r\n",
    b"# p\r## Safety\rS1. a\r## Conversation\rC1. c\r",                # old Mac endings
    b"# p\n\n## Conversation\nC1. c\n\n## Safety\nS1. a\n\n\n",
])
def test_safety_text_is_exact_substring_of_text(tmp_path, raw):
    path = tmp_path / "p.md"
    path.write_bytes(raw)
    policy = load_policy(path)
    assert policy.safety_text in policy.text
    assert policy.safety_text.startswith("## Safety")
    assert policy.safety_text == policy.safety_text.rstrip("\r\n")


def test_safety_text_keeps_inner_blank_lines_and_trailing_spaces(tmp_path):
    raw = "# p\r\n## Safety\r\nS1. a\r\n\r\nS2. b  \r\n\r\n## Conversation\r\n"
    policy = load_policy(_write_bytes(tmp_path, raw))
    assert policy.safety_text == "## Safety\r\nS1. a\r\n\r\nS2. b  "


def _write_bytes(tmp_path, text):
    path = tmp_path / "p.md"
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.mark.parametrize("heading", ["## Safety", "## Safety (protected)", "## Safety "])
def test_safety_heading_forms_accepted(tmp_path, heading):
    policy = load_policy(_write(tmp_path, f"# p\n{heading}\nS1. a\n"))
    assert policy.safety_text == f"{heading}\nS1. a"


@pytest.mark.parametrize("heading", ["## SafetyNet", "## Safety:", "## safety"])
def test_lookalike_safety_headings_are_not_safety(tmp_path, heading):
    with pytest.raises(PolicyError):
        load_policy(_write(tmp_path, f"# p\n{heading}\nS1. a\n"))


def test_sha256_text_hashes_utf8():
    assert sha256_text("café") == hashlib.sha256("café".encode("utf-8")).hexdigest()


def test_conversation_section_is_optional(tmp_path):
    text = "# p\n\n## Safety\nS1. Rule.\n"
    assert load_policy(_write(tmp_path, text)).rules == {"S1": "Rule."}
    text_empty = "# p\n\n## Safety\nS1. Rule.\n\n## Conversation\n"
    assert load_policy(_write(tmp_path, text_empty)).rules == {"S1": "Rule."}


@pytest.mark.parametrize("text", [
    "policy\n\n## Safety\nS1. Rule.\n",                       # no title
    "# p\n\n## Conversation\nC1. Rule.\n",                     # no Safety section
    "# p\n\n## Safety\n\n## Conversation\nC1. Rule.\n",        # Safety without rules
    "# p\n\n## Safety\nS1. Rule.\n\n## Conversation\nC1. A.\nC1. B.\n",  # duplicate ID
    "# p\n\n## Safety\nS1. Rule.\nThis is not a rule.\n",      # stray line
    "# p\n\n## Safety\nS1.Rule without space.\n",               # malformed rule
    "# p\n\n## Safety\nS1. Rule.\nC1. Learned rule hiding in Safety.\n",  # C rule in Safety
    "# p\n\n## Safety\nS1. Rule.\n\n## Conversation\nS2. Safety rule outside.\n",  # S rule outside
    "# p\n\n## Safety\nS1. Rule.\n\n## Notes\nS2. Safety rule in another section.\n",
])
def test_load_policy_rejects_bad_files(tmp_path, text):
    with pytest.raises(PolicyError):
        load_policy(_write(tmp_path, text))


def _example_rules():
    agents = (ROOT / "docs" / "agents.md").read_text(encoding="utf-8")
    block = agents.split("# policy-v4", 1)[1].split("```", 1)[0]
    return dict(re.findall(r"^([SC]\d+)\. (.+)$", block, re.M))


def test_policy_v1_matches_example_policy():
    policy = load_policy(ROOT / "policies" / "policy-v1.md")
    example = _example_rules()
    assert policy.version == "policy-v1"
    assert list(policy.rules) == ["S1", "S2", "S3", "C1", "C3", "C4", "C6", "C7", "C8"]
    for rule_id in ["S1", "S2", "S3", "C1", "C3", "C6", "C7", "C8"]:
        assert policy.rules[rule_id] == example[rule_id]
    assert policy.rules["C4"] == "Keep replies short."


def test_baseline_plain_shares_safety_section_and_has_no_conversation_rules():
    v1 = load_policy(ROOT / "policies" / "policy-v1.md")
    plain = load_policy(ROOT / "policies" / "baseline-plain.md")
    assert plain.version == "baseline-plain"
    assert plain.safety_text == v1.safety_text
    assert plain.safety_sha256 == v1.safety_sha256
    assert list(plain.rules) == ["S1", "S2", "S3"]
    assert "## Conversation" in plain.text
