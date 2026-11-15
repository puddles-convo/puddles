import json
from datetime import datetime, timezone

import pytest

from puddles.eval import judge_skill as js
from puddles.eval.data import RealConversation, SeekerRatings
from puddles.llm import AgentFailed
from schemas import ConversationRecord, Scenario, Turn

PROBLEMS = ("job crisis", "breakup with partner", "academic pressure", "ongoing depression")


def _conv(i, empathy, relevance, split="train", n_words=40, problem=None):
    sid = f"esconv-{split}-{i:04d}"
    text = " ".join(["word"] * (n_words // 4))
    turns = [Turn(speaker="user", text=f"secret-user-text-{i} " + text),
             Turn(speaker="helper", text="helper " + text),
             Turn(speaker="user", text="user " + text),
             Turn(speaker="helper", text="helper " + text)]
    scenario = Scenario(scenario_id=sid, split=split, emotion_type="sadness",
                        problem_type=problem or PROBLEMS[i % len(PROBLEMS)], situation="s", initial_intensity=4)
    record = ConversationRecord(id=f"conv-{sid}", written_by="esconv-loader", created_at=datetime.now(timezone.utc),
                                source="real", scenario_id=sid, split=split, policy_version="esconv-human", turns=turns)
    return RealConversation(scenario=scenario, record=record,
                            ratings=SeekerRatings(empathy=empathy, relevance=relevance, emotion_before=4, emotion_after=2),
                            strategies=["Question", "Reflection of feelings"])


def _dataset():
    convs, i = [], 0
    for e in range(1, 6):
        for r in range(1, 6):
            for _ in range(3):
                convs.append(_conv(i, e, r))
                i += 1
    return convs


GOOD_GUIDE = js.DimensionGuide(raises=["names the feeling", "asks before advising"],
                               lowers=["generic comfort", "advice in the first reply"],
                               anchor_1="ignores the feeling", anchor_3="acknowledges it generally",
                               anchor_5="names this person's feeling accurately", traps=["long polite replies"])


class FakeLLM:
    def __init__(self, guides=None):
        self.calls = []
        self.guides = list(guides) if guides else None

    def structured(self, system, user, schema):
        self.calls.append((system, user, schema))
        if schema is js.DimensionGuide:
            return self.guides.pop(0) if self.guides else GOOD_GUIDE
        return js.ExampleReason(reason="Turn 2 reflects the worry, so the person felt heard.")


def test_refuses_anything_but_the_training_split():
    convs = _dataset() + [_conv(999, 3, 3, split="validation")]
    with pytest.raises(ValueError, match="training split only"):
        js.build(convs, FakeLLM(), builder_model="m")
    with pytest.raises(ValueError, match="training split only"):
        js.plan(convs, per_level=2, seed=0, max_words=450, example_count=3, example_words=250)


def test_sampling_is_deterministic_capped_and_spread_over_problem_types():
    convs = _dataset()
    a = js.sample_levels(convs, "empathy", 4, seed=1, max_words=450)
    b = js.sample_levels(convs, "empathy", 4, seed=1, max_words=450)
    assert {k: [c.scenario.scenario_id for c in v] for k, v in a.items()} == \
           {k: [c.scenario.scenario_id for c in v] for k, v in b.items()}
    for level, chosen in a.items():
        assert len(chosen) == 4
        assert all(c.ratings.empathy == level for c in chosen)
        assert len({c.scenario.problem_type for c in chosen}) == 4


def test_sampling_respects_the_word_cap_and_skips_unrated():
    convs = [_conv(0, 5, 5, n_words=2000), _conv(1, 5, 5), _conv(2, None, 5)]
    chosen = js.sample_levels(convs, "empathy", 5, seed=0, max_words=450)
    assert [c.scenario.scenario_id for c in chosen[5]] == ["esconv-train-0001"]


def test_examples_cover_the_target_combinations_within_the_cap():
    convs = _dataset()
    examples = js.choose_examples(convs, 5, max_words=250)
    pairs = [(c.ratings.empathy, c.ratings.relevance) for c in examples]
    assert pairs[:4] == [(5, 5), (3, 3), (5, 2), (2, 5)]
    assert pairs[4] == (1, 1)
    assert all(js.words(c) <= 250 for c in examples)
    assert len({c.scenario.scenario_id for c in examples}) == 5
    assert js.choose_examples(convs, 5, max_words=250) == examples


def test_build_renders_the_skill_and_meta_holds_ids_only():
    convs = _dataset()
    llm = FakeLLM()
    text, meta = js.build(convs, llm, builder_model="builder-x", per_level=2, example_count=4)
    assert text.startswith("---\nname: grade-support-conversations\n")
    for heading in ("## Empathy:", "## Relevance:", "## Always", "## Examples", "### Example 1:"):
        assert heading in text
    assert "- **5:** names this person's feeling accurately" in text
    assert "CC BY-NC 4.0" in text
    assert meta["sha256"] == js.sha256_text(text)
    assert meta["esconv"]["split"] == "train"
    dumped = json.dumps(meta)
    assert "secret-user-text" not in dumped
    assert all(sid.startswith("esconv-train-") for d in meta["guideline_sources"].values()
               for ids in d.values() for sid in ids)
    assert len(llm.calls) == 2 + 4


def test_model_prompts_carry_no_scenario_ids():
    llm = FakeLLM()
    js.build(_dataset(), llm, builder_model="m", per_level=2, example_count=2)
    for _, user, _ in llm.calls:
        assert "esconv-train-" not in user
        assert "conv-" not in user


def test_an_invalid_guide_is_retried_once_then_fails():
    bad = js.DimensionGuide(raises=["a", "b"], lowers=["c", "d"], anchor_1="", anchor_3="x", anchor_5="y")
    llm = FakeLLM(guides=[bad, GOOD_GUIDE, GOOD_GUIDE])
    js.build(_dataset(), llm, builder_model="m", per_level=1, example_count=1)
    first, second = llm.calls[0][1], llm.calls[1][1]
    assert "anchor_1 is empty" in second and "anchor_1 is empty" not in first
    with pytest.raises(AgentFailed):
        js.build(_dataset(), FakeLLM(guides=[bad, bad]), builder_model="m", per_level=1, example_count=1)


def test_write_is_atomic_and_complete(tmp_path):
    js.write(tmp_path / "skill", "text", {"sha256": "x"})
    assert (tmp_path / "skill" / "SKILL.md").read_text() == "text"
    assert json.loads((tmp_path / "skill" / "skill.meta.json").read_text()) == {"sha256": "x"}
    assert not list((tmp_path / "skill").glob(".*.tmp"))
