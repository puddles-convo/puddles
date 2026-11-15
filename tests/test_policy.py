from pathlib import Path

import pytest

from puddles.policy import (
    PolicyEditError,
    apply_edit,
    at_length_limit,
    candidate_version,
    parse_policy,
    render_policy,
    safety_section,
)

POLICY = (Path(__file__).parents[1] / "fixtures" / "policy-v3.md").read_text()


def test_round_trip_is_byte_for_byte():
    assert render_policy(parse_policy(POLICY)) == POLICY


def test_add_takes_next_unused_id_and_keeps_safety():
    policy = parse_policy(POLICY)
    new, rule_id = apply_edit(policy, "add", None, "Ask one open question first.", "policy-v4-candidate",
                              used_ids={"C2", "C9"})
    assert rule_id == "C10"
    assert safety_section(new) == safety_section(policy)
    assert {k: v for k, v in new.conversation.items() if k != "C10"} == policy.conversation


def test_reword_and_remove_touch_only_one_rule():
    policy = parse_policy(POLICY)
    reworded, _ = apply_edit(policy, "reword", "C4", "Keep replies short.", "v")
    assert reworded.conversation["C4"] == "Keep replies short."
    removed, _ = apply_edit(policy, "remove", "C5", None, "v")
    assert "C5" not in removed.conversation and len(removed.conversation) == len(policy.conversation) - 1


@pytest.mark.parametrize("action,rule_id,text", [
    ("reword", "S1", "anything"), ("remove", "S2", None), ("remove", "C99", None),
    ("add", "C1", "x"), ("reword", "C1", None), ("remove", "C1", "x"), ("rewrite", "C1", "x")])
def test_bad_edits_are_refused(action, rule_id, text):
    with pytest.raises(PolicyEditError):
        apply_edit(parse_policy(POLICY), action, rule_id, text, "v")


def test_multiline_safety_rule_is_kept_byte_for_byte():
    start = "S1. If the person mentions wanting to die,"
    wrapped = POLICY.replace(start, start + "\n   ")
    policy = parse_policy(wrapped)
    new, _ = apply_edit(policy, "remove", "C5", None, "v")
    assert safety_section(new) == safety_section(policy)
    def safety(text):
        return text.split("## Safety")[1].split("## Conversation")[0]

    assert safety(render_policy(new)) == safety(wrapped)


def test_versions_and_length_limit():
    assert candidate_version("policy-v3") == "policy-v4-candidate"
    assert not at_length_limit(POLICY)
    assert at_length_limit(POLICY, cap=100)
