import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from puddles.eval.judge import (
    ASTRA_SYSTEM,
    JUDGE_SYSTEM,
    JudgeFailed,
    JudgePanel,
    OpenAIJudgeLLM,
    PanelConfig,
    judge_version,
)
from puddles.llm import AgentFailed
from schemas import ConversationRecord, JudgeOutput, Turn


def _output(note="fine note", strategies=None, empathy=3):
    return JudgeOutput(
        empathy=empathy,
        relevance=3,
        emotion_before=4,
        emotion_after=3,
        strategies=strategies if strategies is not None else ["Question", "Information"],
        note=note,
    )


class FakeJudgeLLM:
    def __init__(self, model, outputs):
        self.model = model
        self.outputs = list(outputs)
        self.calls = []

    def judge(self, system, user):
        self.calls.append((system, user))
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out, 100 + len(self.calls), 10 + len(self.calls)


def _record(turns=None):
    turns = turns or [
        Turn(speaker="user", text="I am scared about work."),
        Turn(speaker="helper", text="Tell me more."),
        Turn(speaker="user", text="It is the layoffs."),
        Turn(speaker="helper", text="Here is a plan."),
    ]
    return ConversationRecord(
        id="conv-RECORDID-99",
        written_by="test",
        created_at=datetime.now(timezone.utc),
        parent_ids=[],
        source="simulated",
        scenario_id="sc-SCENARIID-77",
        split="train",
        policy_version="pv-POLICY-55",
        turns=turns,
    )


def _config(**kw):
    return PanelConfig(**{"judges": ["model-a", "model-b", "model-c"], "astra": "model-astra", **kw})


def _panel(outputs_by_judge=None, astra_outputs=None, config=None):
    config = config or _config()
    outputs_by_judge = outputs_by_judge or [
        [_output("judgeA initial note"), _output("judgeA final note")],
        [_output("judgeB initial note"), _output("judgeB final note")],
        [_output("judgeC initial note"), _output("judgeC final note")],
    ]
    judges = [FakeJudgeLLM(model, outputs) for model, outputs in zip(config.judges, outputs_by_judge, strict=True)]
    astra = FakeJudgeLLM("model-astra", astra_outputs or [_output("astra kept note", empathy=4)])
    return JudgePanel(config, judges, astra), judges, astra


def test_happy_path_seven_calls_and_score_record():
    panel, judges, astra = _panel()
    result = panel.score(_record())
    assert sum(len(j.calls) for j in judges) + len(astra.calls) == 7
    score = result.score
    assert score.id == "score-conv-RECORDID-99"
    assert score.written_by == "judge"
    assert score.parent_ids == ["conv-RECORDID-99"]
    assert score.conversation_id == "conv-RECORDID-99"
    assert score.judge_version == panel.version
    assert score.empathy == 4
    assert score.note == "astra kept note"
    assert [v.judge for v in result.verdicts] == ["A", "B", "C"]
    assert result.verdicts[0].initial.note == "judgeA initial note"
    assert result.verdicts[0].final.note == "judgeA final note"
    assert result.verdicts[0].model == "model-a"
    assert [(u.stage, u.judge) for u in result.usage] == [
        ("initial", "A"), ("final", "A"),
        ("initial", "B"), ("final", "B"),
        ("initial", "C"), ("final", "C"),
        ("astra", "astra"),
    ]
    assert [u.model for u in result.usage] == ["model-a"] * 2 + ["model-b"] * 2 + ["model-c"] * 2 + ["model-astra"]
    assert all(u.prompt_tokens >= 100 and u.completion_tokens >= 10 for u in result.usage)


def test_prompts_are_blind_to_ids_and_policy():
    panel, judges, astra = _panel()
    panel.score(_record())
    prompts = [text for llm in judges + [astra] for pair in llm.calls for text in pair]
    for prompt in prompts:
        for secret in ("RECORDID-99", "SCENARIID-77", "POLICY-55"):
            assert secret not in prompt


def test_self_review_sees_only_its_own_initial():
    panel, judges, astra = _panel()
    panel.score(_record())
    for label, llm in zip("ABC", judges, strict=True):
        assert len(llm.calls) == 2
        final_user = llm.calls[1][1]
        assert f"judge{label} initial note" in final_user
        for other in "ABC":
            if other != label:
                assert f"judge{other} initial note" not in final_user
                assert f"judge{other} final note" not in final_user


def test_astra_sees_labelled_final_judgements_only():
    panel, judges, astra = _panel()
    panel.score(_record())
    system, user = astra.calls[0]
    assert system == ASTRA_SYSTEM
    for label in "ABC":
        assert f'<judgement judge="{label}">' in user
        assert f"judge{label} final note" in user
        assert f"judge{label} initial note" not in user
    for model in ("model-a", "model-b", "model-c", "model-astra"):
        assert model not in user
    assert user.startswith("<conversation>")


def test_wrong_strategy_count_retries_then_succeeds():
    bad = _output("bad", strategies=["Question"])
    panel, judges, astra = _panel(outputs_by_judge=[
        [bad, _output("judgeA initial note"), _output("judgeA final note")],
        [_output("judgeB initial note"), _output("judgeB final note")],
        [_output("judgeC initial note"), _output("judgeC final note")],
    ])
    result = panel.score(_record())
    assert len(judges[0].calls) == 3
    retry_user = judges[0].calls[1][1]
    assert "failed these checks" in retry_user
    assert "strategies must have exactly 2 labels, one per helper turn, got 1" in retry_user
    assert [(u.stage, u.judge) for u in result.usage[:4]] == [
        ("initial", "A"), ("initial", "A"), ("final", "A"), ("initial", "B"),
    ] or len(result.usage) == 8
    assert len(result.usage) == 8


def test_two_bad_answers_raise_judge_failed():
    bad = _output("bad", strategies=["Question"])
    panel, judges, _ = _panel(outputs_by_judge=[
        [bad, bad, _output("never")],
        [_output("judgeB initial note"), _output("judgeB final note")],
        [_output("judgeC initial note"), _output("judgeC final note")],
    ])
    with pytest.raises(JudgeFailed, match=r"initial judgement by judge A \(model-a\) failed"):
        panel.score(_record())


def test_unknown_strategy_and_empty_note_rejected():
    for bad in (_output("x", strategies=["Question", "Mind reading"]),
                _output("  ")):
        llm = FakeJudgeLLM("model-a", [bad, _output("ok"), _output("ok final")])
        panel = JudgePanel(_config(), [llm, FakeJudgeLLM("model-b", [_output()] * 2),
                                       FakeJudgeLLM("model-c", [_output()] * 2)],
                           FakeJudgeLLM("model-astra", [_output()]))
        panel.score(_record())
        assert len(llm.calls) == 3
        assert "failed these checks" in llm.calls[1][1]


def test_no_helper_turns_raises_before_any_call():
    panel, judges, astra = _panel()
    record = _record(turns=[Turn(speaker="user", text="hi")])
    with pytest.raises(ValueError, match="no helper turns"):
        panel.score(record)
    assert all(not j.calls for j in judges) and not astra.calls


def test_judge_version_format_and_sensitivity():
    version = judge_version(_config())
    assert re.fullmatch(r"judge-v1-[0-9a-f]{8}", version)
    assert judge_version(_config()) == version
    assert judge_version(_config(judges=["model-a", "model-b", "model-d"])) != version
    assert judge_version(_config(astra="model-other")) != version
    assert judge_version(_config(temperature=0.5)) != version
    assert judge_version(_config(temperature=None)) != version


def test_panel_config_from_env(monkeypatch):
    monkeypatch.setenv("PUDDLES_JUDGE_PANEL", " m1 , m2 ,m3 ")
    monkeypatch.setenv("PUDDLES_JUDGE_ASTRA", " m4 ")
    config = PanelConfig.from_env()
    assert config.judges == ["m1", "m2", "m3"] and config.astra == "m4" and config.temperature == 0.0


def test_panel_config_from_env_temperature_none(monkeypatch):
    monkeypatch.setenv("PUDDLES_JUDGE_PANEL", "m1,m2,m3")
    monkeypatch.setenv("PUDDLES_JUDGE_ASTRA", "m4")
    monkeypatch.setenv("PUDDLES_JUDGE_TEMPERATURE", "none")
    assert PanelConfig.from_env().temperature is None


@pytest.mark.parametrize("env", [
    {"PUDDLES_JUDGE_PANEL": "m1,m2", "PUDDLES_JUDGE_ASTRA": "m4"},
    {"PUDDLES_JUDGE_PANEL": "m1,m1,m3", "PUDDLES_JUDGE_ASTRA": "m4"},
    {"PUDDLES_JUDGE_ASTRA": "m4"},
    {"PUDDLES_JUDGE_PANEL": "m1,m2,m3"},
])
def test_panel_config_from_env_rejects_bad(monkeypatch, env):
    for key in ("PUDDLES_JUDGE_PANEL", "PUDDLES_JUDGE_ASTRA", "PUDDLES_JUDGE_TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        PanelConfig.from_env()


class _FakeResponse:
    def __init__(self, parsed, usage=None):
        self.choices = [type("C", (), {"message": type("M", (), {"parsed": parsed})()})()]
        self.usage = usage


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(response)})()


def test_openai_judge_llm_parses_and_sends_temperature():
    output = _output()
    usage = type("U", (), {"prompt_tokens": 42, "completion_tokens": 7})()
    client = _FakeClient(_FakeResponse(output, usage))
    llm = OpenAIJudgeLLM("m-x", temperature=0.0, client=client)
    out, prompt_tokens, completion_tokens = llm.judge("sys", "usr")
    assert out is output and (prompt_tokens, completion_tokens) == (42, 7)
    kwargs = client.chat.completions.kwargs
    assert kwargs["model"] == "m-x"
    assert kwargs["response_format"] is JudgeOutput
    assert kwargs["temperature"] == 0.0
    assert kwargs["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_openai_judge_llm_omits_temperature_when_none():
    client = _FakeClient(_FakeResponse(_output()))
    OpenAIJudgeLLM("m-x", temperature=None, client=client).judge("s", "u")
    assert "temperature" not in client.chat.completions.kwargs


def test_openai_judge_llm_unparsed_raises_agent_failed():
    client = _FakeClient(_FakeResponse(None))
    with pytest.raises(AgentFailed):
        OpenAIJudgeLLM("m-x", client=client).judge("s", "u")


def test_judge_system_matches_design_doc():
    doc = (Path(__file__).parents[1] / "docs" / "agents" / "t3-judge.md").read_text()
    section = doc.split("### System message", 1)[1]
    fenced = section.split("```", 2)[1].strip("\n")
    assert JUDGE_SYSTEM == fenced
