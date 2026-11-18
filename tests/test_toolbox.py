"""The proposer's toolbox (PUDDLES_PROPOSER=toolbox): skill cards and examples as policy items. Offline."""
import glob
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.policy import load_policy
from puddles import examples_picker, gate
from puddles.agents import proposer_toolbox as tb
from puddles.decision_log import DecisionLog
from puddles.policy import PolicyEditError, apply_edit, parse_policy, render_policy
from schemas import PatternBrief, ProposerInput, Source

START = Path("policies/training-2026-11-15/policy-v1.md")
URL = "https://example.org/reflective-listening"


def _inp(policy_text: str) -> ProposerInput:
    return ProposerInput(policy_text=policy_text, at_length_limit=False,
                         pattern=PatternBrief(id="pattern-r01", name="advice-first", description="d",
                                              status="open", sightings=1),
                         sources=[Source(url=URL, technique="Reflect, then ask.")], earlier_edits=[])


def _out(**kw) -> tb.ToolboxOutput:
    base = {"considered": [{"tool": t, "rejected_because": "x"} for t in tb.TOOLS if t != kw["tool"]],
            "hypothesis": "h", "prediction": "p"}
    return tb.ToolboxOutput.model_validate({**base, **kw})


CARD = dict(tool="add_skill_card", card_op="add", card_name="Reflect before advice",
            when_to_use="the person describes a problem", when_not_to_use="they ask a direct question",
            steps=["Name the feeling", "Ask one open question"], source_urls=[URL])


def _fake_examples(situation, k):
    return f"When someone is dealing with {situation}: Person: \"a\" Supporter: \"b\"", ["esconv-train-0001"]


def test_policies_without_cards_or_examples_render_byte_for_byte():
    """The policy-only arm must be unchanged by the toolbox code."""
    files = glob.glob("policies/**/*.md", recursive=True) + glob.glob("results/training-2026-11-15/run/policies/*.md")
    assert files
    for f in files:
        text = Path(f).read_text()
        assert render_policy(parse_policy(text)) == text, f


def test_a_card_is_one_item_in_its_own_section_and_loads_everywhere(tmp_path):
    policy = parse_policy(START.read_text())
    edit, kind, _ = tb.to_edit(_out(**CARD), policy)
    cand, rid = apply_edit(policy, edit.action, edit.rule_id, edit.rule_text, "policy-v2", kind=kind)
    assert rid == "K1"
    text = render_policy(cand)
    assert "\n## Skills" in text and "K1. Reflect before advice. Use when" in text
    assert parse_policy(text).conversation["K1"] == cand.conversation["K1"]
    p = tmp_path / "p.md"
    p.write_text(text)
    loaded = load_policy(p)  # what the helper and the gate read
    assert gate._changed_rules(load_policy(START), loaded) == ["K1"]
    assert loaded.safety_sha256 == load_policy(START).safety_sha256


def test_examples_are_added_then_the_oldest_is_replaced():
    policy = parse_policy(START.read_text())
    ids = []
    for _ in range(3):
        edit, kind, src = tb.to_edit(_out(tool="set_examples", situation="a breakup", k=2), policy, _fake_examples)
        policy, rid = apply_edit(policy, edit.action, edit.rule_id, edit.rule_text, policy.version,
                                 used_ids=set(ids), kind=kind)
        ids.append(rid)
        assert src == ["esconv-train-0001"]
    assert ids == ["E1", "E2", "E1"]
    assert "## Examples" in render_policy(policy)


@pytest.mark.parametrize("fields, problem", [
    ({"tool": "edit_rule", "rule_op": "reword", "rule_id": "K1", "rule_text": "x"}, "C rules only"),
    ({**CARD, "steps": ["only one"]}, "2 to 5 steps"),
    ({**CARD, "source_urls": ["https://elsewhere.example"]}, "not from this round's research"),
    ({**CARD, "source_urls": []}, "at least one source_url"),
    ({"tool": "set_examples", "situation": "a breakup"}, "needs situation and k"),
])
def test_malformed_actions_are_refused(tmp_path, fields, problem):
    text = START.read_text()
    problems = tb.check(_out(**fields), _inp(text), parse_policy(text), DecisionLog(tmp_path), None, _fake_examples)
    assert any(problem in p for p in problems), problems


def test_safety_rules_cannot_be_reached_through_any_tool(tmp_path):
    text = START.read_text()
    policy = parse_policy(text)
    for fields in ({"tool": "edit_rule", "rule_op": "remove", "rule_id": "S1"},
                   {**CARD, "card_op": "retire", "card_id": "S1"}):
        problems = tb.check(_out(**fields), _inp(text), policy, DecisionLog(tmp_path), None, _fake_examples)
        assert problems
    with pytest.raises(PolicyEditError):
        apply_edit(policy, "remove", "S1", None, "v", kind="K")


def test_a_clean_card_passes_the_checks(tmp_path):
    text = START.read_text()
    assert tb.check(_out(**CARD), _inp(text), parse_policy(text), DecisionLog(tmp_path), None, _fake_examples) == []


def _conv(cid, situation, emp, rel, turns):
    return SimpleNamespace(
        ratings=SimpleNamespace(empathy=emp, relevance=rel),
        scenario=SimpleNamespace(situation=situation, problem_type="breakup with partner", emotion_type="sadness"),
        record=SimpleNamespace(id=cid, turns=[SimpleNamespace(speaker=s, text=t) for s, t in turns]))


TURNS = [("helper", "Hi, how are you?"),
         ("user", "My girlfriend left me last week and I feel so alone now"),
         ("helper", "That sounds really painful, losing someone close leaves a big empty space. What has been hardest?")]


def test_the_picker_uses_only_top_rated_matches_in_a_fixed_order():
    pool = [_conv("esconv-train-0002", "girlfriend left me", 5, 5, TURNS),
            _conv("esconv-train-0001", "girlfriend left me alone", 5, 5, TURNS),
            _conv("esconv-train-0003", "girlfriend left me alone", 4, 5, TURNS)]  # not 5/5: never used
    picked = examples_picker.pick("my girlfriend left me and I am alone", 3, pool)
    assert [p[0] for p in picked] == ["esconv-train-0001", "esconv-train-0002"]
    line = examples_picker.render("a breakup", picked)
    assert "\n" not in line and line.count("Supporter:") == 2
    with pytest.raises(examples_picker.ExamplesError):
        examples_picker.pick("anything", 4, pool)


def test_the_picker_loads_only_the_train_split(monkeypatch):
    seen = []
    monkeypatch.setattr("puddles.eval.data.load_esconv", lambda split, **kw: seen.append(split) or [])
    examples_picker._train.cache_clear()
    with pytest.raises(examples_picker.ExamplesError):
        examples_picker.pick("a breakup", 1)
    examples_picker._train.cache_clear()
    assert seen == ["train"]


def test_scoreboard_reads_the_tool_from_the_item_id(tmp_path):
    lines = tb.scoreboard(DecisionLog(tmp_path))
    assert lines == ["edit_rule: tried 0, kept 0", "add_skill_card: tried 0, kept 0", "set_examples: tried 0, kept 0"]


def test_the_round_accepts_a_card_candidate_as_base_plus_its_logged_edit(tmp_path):
    """Regression: the re-application check assumed every addition was a C rule (PolicyChanged)."""
    from puddles.round import _check_logged_edit
    from schemas import PolicyEdit
    from puddles.memory import now

    base_text = START.read_text()
    policy = parse_policy(base_text)
    out, kind, _ = tb.to_edit(_out(**CARD), policy)
    cand, rid = apply_edit(policy, out.action, None, out.rule_text, "policy-v2-candidate", kind=kind)
    edit = PolicyEdit(id="edit-r01", written_by="proposer", created_at=now(), round=1, pattern_id="pattern-r01",
                      research_id="research-r01", base_policy_version=policy.version,
                      new_policy_version="policy-v2-candidate", policy_path="p.md", summary="s",
                      action="add", rule_id=rid, rule_text=out.rule_text)
    _check_logged_edit(base_text, render_policy(cand), edit, DecisionLog(tmp_path), None)  # no PolicyChanged


@pytest.mark.parametrize("raw, expected", [
    ("After the person answers", "after the person answers"),
    ("When they ask for advice.", "they ask for advice"),
    ("Do not use to override safety rules", "to override safety rules"),
    ("Not when they are in crisis", "they are in crisis"),
    ("ADHD is mentioned", "ADHD is mentioned"),
])
def test_card_phrases_read_after_use_when(raw, expected):
    assert tb.clause(raw) == expected
