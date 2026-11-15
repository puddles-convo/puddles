"""Policy parsing, safety fingerprints, and the shipped policy contract."""

from dataclasses import FrozenInstanceError
import hashlib
import importlib
import re

import pytest


@pytest.fixture
def api():
    return importlib.import_module("loop.policy")


def write_policy(tmp_path, text):
    path = tmp_path / "input.md"
    path.write_bytes(text.encode("utf-8"))
    return path


def test_exact_text_rules_order_and_frozen_policy(api, tmp_path):
    text = (
        '# policy-é\n\n## Safety (protected: no edit may change this section)\n'
        'S1. Stay safe.\nS9. Ask first.\n\n## Conversation\n'
        'C1. Listen.\nC3. One question.\n\n'
    )
    policy = api.load_policy(write_policy(tmp_path, text))
    assert policy.version == "policy-é"
    assert policy.text == text
    assert list(policy.rules.items()) == [
        ("S1", "Stay safe."), ("S9", "Ask first."),
        ("C1", "Listen."), ("C3", "One question."),
    ]
    with pytest.raises(FrozenInstanceError):
        policy.version = "changed"


def test_version_is_title_line_without_hash_space_and_surrounding_whitespace(api, tmp_path):
    policy = api.load_policy(write_policy(tmp_path, "#   policy-v1  \n## Safety\nS1. Safe.\n"))
    assert policy.version == "policy-v1"


@pytest.mark.parametrize("ending", ["", "\n", "\n\n", "\n\n\n"])
def test_safety_at_eof_has_heading_internal_blanks_and_no_trailing_newline(api, tmp_path, ending):
    safety = '## Safety (protected: no edit may change this section)\nS1. Café.\n\nS3. Ask.'
    text = "# p\n\n" + safety + ending
    policy = api.load_policy(write_policy(tmp_path, text))
    assert policy.safety_text == safety
    assert policy.safety_sha256 == hashlib.sha256(safety.encode("utf-8")).hexdigest()
    assert policy.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_safety_hash_excludes_next_heading_and_separator_blank_lines(api, tmp_path):
    safety = "## Safety (protected: no edit may change this section)\nS1. Do no harm."
    first = api.load_policy(write_policy(tmp_path, "# a\n\n" + safety + "\n\n## Conversation\nC1. Listen.\n"))
    second = api.load_policy(write_policy(tmp_path, "# b\n\n" + safety + "\n\n\n## Conversation\nC8. Ask.\n"))
    assert first.safety_text == second.safety_text == safety
    assert first.safety_sha256 == second.safety_sha256
    assert first.sha256 != second.sha256


def test_safety_heading_and_rule_bytes_are_protected(api, tmp_path):
    texts = [
        "## Safety (protected: no edit may change this section)\nS1. Safe.",
        "## Safety\nS1. Safe.",
        "## Safety (protected: no edit may change this section)\nS1. Safer.",
    ]
    policies = [api.load_policy(write_policy(tmp_path, "# p\n\n" + text)) for text in texts]
    assert len({p.safety_sha256 for p in policies}) == 3


def test_safety_preserves_trailing_spaces_on_nonblank_lines(api, tmp_path):
    safety = "## Safety (protected: no edit may change this section)\nS1. Safe.  \n\nS3. Ask.\t"
    policy = api.load_policy(write_policy(tmp_path, "# p\n\n" + safety + "\n \t\n## Conversation\n"))
    assert policy.safety_text == safety
    assert policy.safety_sha256 == hashlib.sha256(safety.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("text", ["", "abc", "café\n", "\n spaced \n"])
def test_sha256_uses_utf8_without_normalization(api, text):
    assert api.sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("tail", ["", "\n\n## Conversation\n\n"])
def test_conversation_rules_are_optional(api, tmp_path, tail):
    policy = api.load_policy(write_policy(tmp_path, "# p\n\n## Safety\nS1. Safe." + tail))
    assert policy.rules == {"S1": "Safe."}


@pytest.mark.parametrize("text", [
    "policy\n## Safety\nS1. Safe.\n",
    "\n# p\n## Safety\nS1. Safe.\n",
    "# p\n## Conversation\nC1. Listen.\n",
    "# p\n## Safety\n\n## Conversation\nC1. Listen.\n",
    "# p\n## Safety\nS1. First.\nS1. Again.\n",
    "# p\n## Safety\nS1. Safe.\n## Conversation\nC1. First.\nC1. Again.\n",
    "# p\n## Safety\nS1. Safe.\n## Conversation\nS1. Duplicate across sections.\n",
    "# p\n## Safety\nS1. a\nC1. b\n",
    "# p\n## Safety\nS1. a\n## Conversation\nS2. b\n",
    "# p\n## Safety\nS1. a\n## Notes\nS2. b\n",
    "# p\n## SafetyNet\nS1. a\n",
    "# p\n## Safety\nnot a rule\n",
    "# p\n## Safety\nS1. Safe.\n## Conversation\ncontinuation prose\n",
    "# p\n## Safety\nS1.Safe.\n",
    "# p\n## Safety\nX1. Safe.\n",
])
def test_invalid_policy_raises_policy_error(api, tmp_path, text):
    from loop.errors import PolicyError
    with pytest.raises(PolicyError):
        api.load_policy(write_policy(tmp_path, text))


@pytest.mark.parametrize("ending", ["", "\r\n", "\r\n \t\r\n", "\r\n\r\n## Conversation\r\nC1. Listen.\r\n"])
def test_windows_policy_preserves_text_and_safety_bytes(api, tmp_path, ending):
    safety = "## Safety (protected: no edit may change this section)\r\nS1. Café.\r\n\r\nS3. Ask.  "
    text = "# p\r\n\r\n" + safety + ending
    policy = api.load_policy(write_policy(tmp_path, text))
    assert policy.text == text
    assert policy.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert policy.safety_text == safety
    assert policy.safety_sha256 == hashlib.sha256(safety.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("eol", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("final_newline", [False, True])
@pytest.mark.parametrize("safety_first", [False, True])
def test_exact_safety_and_file_order_with_each_line_ending(
    api, tmp_path, eol, final_newline, safety_first,
):
    safety = eol.join(["## Safety ", "S1. Café.  ", "", "S3. Ask.\t"])
    conversation = "## Conversation" + eol + "C1. Listen."
    sections = [safety, conversation] if safety_first else [conversation, safety]
    text = "# p" + eol + (eol + eol + " \t" + eol).join(sections)
    text += eol if final_newline else ""
    policy = api.load_policy(write_policy(tmp_path, text))
    expected_rules = [("S1", "Café.  "), ("S3", "Ask.\t")]
    if safety_first:
        expected_rules.append(("C1", "Listen."))
    else:
        expected_rules.insert(0, ("C1", "Listen."))
    assert list(policy.rules.items()) == expected_rules
    assert policy.text == text
    assert policy.safety_text == safety
    assert policy.safety_text in policy.text
    assert policy.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert policy.safety_sha256 == hashlib.sha256(safety.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("heading", ["## Safety:", "## Safety\t(protected)", "## safety"])
def test_safety_heading_requires_exact_name_and_literal_space(api, tmp_path, heading):
    from loop.errors import PolicyError
    with pytest.raises(PolicyError):
        api.load_policy(write_policy(tmp_path, "# p\n" + heading + "\nS1. Safe.\n"))


@pytest.mark.parametrize("separator", [
    "\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029",
])
@pytest.mark.parametrize("section", ["Safety", "Conversation"])
def test_rule_continuation_after_any_splitlines_separator_is_rejected(api, tmp_path, separator, section):
    from loop.errors import PolicyError
    if section == "Safety":
        text = "# p\n## Safety\nS1. First part." + separator + "Unnumbered continuation.\n"
    else:
        text = ("# p\n## Safety\nS1. Safe.\n## Conversation\nC1. First part."
                + separator + "Unnumbered continuation.\n")
    with pytest.raises(PolicyError):
        api.load_policy(write_policy(tmp_path, text))


@pytest.mark.parametrize("separator", ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"])
def test_unusual_line_boundaries_preserve_exact_safety_substring_and_hash(api, tmp_path, separator):
    safety = separator.join(["## Safety", "S1. Café.  ", "", "S2. Ask."])
    text = "# p\n" + safety + separator * 2 + "## Conversation\nC1. Listen.\n"
    policy = api.load_policy(write_policy(tmp_path, text))
    assert policy.text == text
    assert policy.safety_text == safety
    assert list(policy.rules.items()) == [("S1", "Café.  "), ("S2", "Ask."), ("C1", "Listen.")]
    assert policy.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert policy.safety_sha256 == hashlib.sha256(safety.encode("utf-8")).hexdigest()


def test_shipped_policies_match_documented_rules_and_share_exact_safety(api, repo_root):
    section = (repo_root / "docs/agents.md").read_text(encoding="utf-8").split("## 2. ", 1)[1]
    example = section.split("```markdown\n", 1)[1].split("```", 1)[0]
    example_rules = dict(re.findall(r"^([SC]\d+)\. (.+)$", example, re.MULTILINE))
    policy = api.load_policy(repo_root / "policies/policy-v1.md")
    plain = api.load_policy(repo_root / "policies/baseline-plain.md")
    assert policy.version == "policy-v1"
    assert plain.version == "baseline-plain"
    assert list(policy.rules) == ["S1", "S2", "S3", "C1", "C3", "C4", "C6", "C7", "C8"]
    for rule_id in ("S1", "S2", "S3", "C1", "C3", "C6", "C7", "C8"):
        assert policy.rules[rule_id] == example_rules[rule_id]
    assert policy.rules["C4"] == "Keep replies short."
    assert plain.rules == {rule_id: policy.rules[rule_id] for rule_id in ("S1", "S2", "S3")}
    assert "## Conversation" in plain.text
    # Independently slice the shipped bytes; equal parser hashes alone could
    # conceal normalization of different source sections.
    def safety_bytes(path):
        raw = path.read_bytes()
        start = raw.index(b"## Safety")
        stop = raw.index(b"## Conversation", start)
        return raw[start:stop].rstrip(b"\r\n")
    expected = safety_bytes(repo_root / "policies/policy-v1.md")
    assert safety_bytes(repo_root / "policies/baseline-plain.md") == expected
    assert plain.safety_text.encode("utf-8") == policy.safety_text.encode("utf-8") == expected
    assert plain.safety_sha256 == policy.safety_sha256 == hashlib.sha256(expected).hexdigest()
