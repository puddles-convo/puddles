import json
import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from puddles.eval.judge import (
    JUDGE_SYSTEM,
    FallbackConfig,
    JudgeFailed,
    JudgePanel,
    PanelConfig,
    SingleJudge,
    TinkerJudgeLLM,
    extract_json,
    fallback_judge_version,
    judge_from_env,
    judge_version,
)
from schemas import ConversationRecord, JudgeOutput, Turn


def _output(strategies=("Question",), note="a note"):
    return {"empathy": 3, "relevance": 4, "emotion_before": 4, "emotion_after": 3,
            "strategies": list(strategies), "note": note}


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    ('Here you go: {"a": 1} hope that helps', '{"a": 1}'),
    ('no json at all', 'no json at all'),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


class _FakeClient:
    def __init__(self, content, usage=None):
        self.kwargs = None
        message = type("M", (), {"content": content, "parsed": None})()
        self.response = type("R", (), {"choices": [type("C", (), {"message": message})()],
                                     "usage": usage})()
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_tinker_judge_llm_create_not_parse():
    usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 3})()
    client = _FakeClient(json.dumps(_output()), usage)
    llm = TinkerJudgeLLM("m-tink", temperature=0.0, client=client)
    out, prompt_tokens, completion_tokens = llm.judge("sys", "usr")
    assert isinstance(out, JudgeOutput) and out.empathy == 3
    assert (prompt_tokens, completion_tokens) == (11, 3)
    assert "response_format" not in client.kwargs
    assert client.kwargs["temperature"] == 0.0
    assert client.kwargs["model"] == "m-tink"


def test_tinker_judge_llm_fenced_and_no_temperature():
    client = _FakeClient("```json\n" + json.dumps(_output()) + "\n```")
    llm = TinkerJudgeLLM("m", temperature=None, client=client)
    out, _, _ = llm.judge("s", "u")
    assert out.relevance == 4
    assert "temperature" not in client.kwargs


def test_tinker_judge_llm_invalid_json_raises_validation():
    client = _FakeClient("I cannot help with that")
    with pytest.raises(ValidationError):
        TinkerJudgeLLM("m", client=client).judge("s", "u")


def test_tinker_judge_llm_needs_api_key(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TINKER_API_KEY"):
        TinkerJudgeLLM("m")
    monkeypatch.setenv("TINKER_API_KEY", "tk-test")
    assert TinkerJudgeLLM("m").model == "m"


def test_fallback_config_from_env(monkeypatch):
    for key in ("PUDDLES_FALLBACK_JUDGE_MODEL", "PUDDLES_TINKER_BASE_URL", "PUDDLES_JUDGE_TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="PUDDLES_FALLBACK_JUDGE_MODEL"):
        FallbackConfig.from_env()
    monkeypatch.setenv("PUDDLES_FALLBACK_JUDGE_MODEL", "open/model")
    config = FallbackConfig.from_env()
    assert config.model == "open/model" and config.temperature == 0.0
    assert config.base_url.startswith("https://tinker.")
    monkeypatch.setenv("PUDDLES_TINKER_BASE_URL", "http://local")
    monkeypatch.setenv("PUDDLES_JUDGE_TEMPERATURE", "none")
    config = FallbackConfig.from_env()
    assert config.base_url == "http://local" and config.temperature is None


def test_fallback_judge_version():
    config = FallbackConfig(model="m1")
    version = fallback_judge_version(config)
    assert re.fullmatch(r"judge-fallback-v1-[0-9a-f]{8}", version)
    assert fallback_judge_version(FallbackConfig(model="m1")) == version
    assert fallback_judge_version(FallbackConfig(model="m2")) != version
    assert fallback_judge_version(FallbackConfig(model="m1", temperature=None)) != version
    assert fallback_judge_version(FallbackConfig(model="m1", base_url="http://x")) == version
    assert version != judge_version(PanelConfig(judges=["a", "b", "c"], astra="d"))


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
        return out, 20, 5


def _record(turns=None):
    turns = turns or [Turn(speaker="user", text="hi"), Turn(speaker="helper", text="hello")]
    return ConversationRecord(id="conv-RECID-1", written_by="t", created_at=datetime.now(timezone.utc),
                              parent_ids=[], source="real", scenario_id="sc-SCID-2", split="validation",
                              policy_version="pv-PV-3", turns=turns)


def _judge(outputs):
    return SingleJudge(FallbackConfig(model="m-fb"), FakeJudgeLLM("m-fb", outputs))


def test_single_judge_happy_path():
    judge = _judge([JudgeOutput.model_validate(_output())])
    result = judge.score(_record())
    llm = judge.llm
    assert len(llm.calls) == 1
    assert llm.calls[0][0] == JUDGE_SYSTEM
    assert result.verdicts == []
    score = result.score
    assert score.id == "score-conv-RECID-1" and score.written_by == "judge"
    assert score.parent_ids == ["conv-RECID-1"] and score.conversation_id == "conv-RECID-1"
    assert score.judge_version == judge.version
    assert score.judge_version.startswith("judge-fallback-v1-")
    assert score.empathy == 3 and score.strategies == ["Question"]
    assert [(u.stage, u.judge, u.model) for u in result.usage] == [("single", "fallback", "m-fb")]


def test_single_judge_prompt_blind():
    judge = _judge([JudgeOutput.model_validate(_output())])
    judge.score(_record())
    prompt = judge.llm.calls[0][1]
    for secret in ("RECID-1", "SCID-2", "PV-3"):
        assert secret not in prompt


def test_single_judge_retry_then_success():
    bad = JudgeOutput.model_validate(_output(strategies=("Question", "Extra"), note="bad"))
    judge = _judge([bad, JudgeOutput.model_validate(_output())])
    result = judge.score(_record())
    assert len(judge.llm.calls) == 2
    assert "exactly 1 labels" in judge.llm.calls[1][1]
    assert len(result.usage) == 2


def test_single_judge_double_failure():
    bad = JudgeOutput.model_validate(_output(strategies=("Question", "Extra")))
    judge = _judge([bad, bad])
    with pytest.raises(JudgeFailed, match=r"single judgement by judge fallback \(m-fb\)"):
        judge.score(_record())


def test_single_judge_no_helper_turn():
    judge = _judge([JudgeOutput.model_validate(_output())])
    record = _record(turns=[Turn(speaker="user", text="hi")])
    with pytest.raises(ValueError, match="no helper turns"):
        judge.score(record)
    assert not judge.llm.calls


def test_judge_from_env(monkeypatch):
    monkeypatch.delenv("PUDDLES_JUDGE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PUDDLES_JUDGE_PANEL", "a,b,c")
    monkeypatch.setenv("PUDDLES_JUDGE_ASTRA", "d")
    judge = judge_from_env()
    assert isinstance(judge, JudgePanel) and judge.kind == "panel"
    monkeypatch.setenv("PUDDLES_JUDGE", "panel")
    assert isinstance(judge_from_env(), JudgePanel)
    monkeypatch.setenv("PUDDLES_JUDGE", "fallback")
    monkeypatch.setenv("PUDDLES_FALLBACK_JUDGE_MODEL", "m-fb")
    monkeypatch.setenv("TINKER_API_KEY", "tk")
    judge = judge_from_env()
    assert isinstance(judge, SingleJudge) and judge.kind == "fallback"
    monkeypatch.setenv("PUDDLES_JUDGE", "other")
    with pytest.raises(ValueError, match="PUDDLES_JUDGE"):
        judge_from_env()
