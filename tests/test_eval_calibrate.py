import math
from datetime import datetime, timezone

import pytest

import puddles.eval.calibrate as calibrate_module
from puddles.eval.calibrate import (
    agreement,
    calibrate,
    check_frozen,
    freeze,
    qwk,
    spearman,
)
from puddles.eval.data import RealConversation, SeekerRatings
from puddles.eval.judge import JudgeFailed, JudgeResult, PanelConfig, PanelVerdict, Usage
from schemas import ConversationRecord, JudgeOutput, Scenario, ScoreRecord, Turn


def _output(empathy=3, strategies=("Question",), note="a note"):
    return JudgeOutput(empathy=empathy, relevance=3, emotion_before=4, emotion_after=2,
                       strategies=list(strategies), note=note)


def _result(record_id, empathy=3, model_verdicts=True):
    now = datetime.now(timezone.utc)
    verdicts = [
        PanelVerdict(judge=label, model=f"model-{label.lower()}",
                     initial=_output(note=f"{label} initial"), final=_output(note=f"{label} final"))
        for label in "ABC"
    ]
    score = ScoreRecord(id=f"score-{record_id}", written_by="judge", created_at=now,
                        parent_ids=[record_id], conversation_id=record_id, judge_version="judge-v1-test",
                        empathy=empathy, relevance=3, emotion_before=4, emotion_after=2,
                        strategies=["Question"], note="astra note")
    usage = [Usage(stage="initial", judge="A", model="model-a", prompt_tokens=10, completion_tokens=5),
             Usage(stage="final", judge="A", model="model-a", prompt_tokens=20, completion_tokens=6),
             Usage(stage="astra", judge="astra", model="model-astra", prompt_tokens=30, completion_tokens=7)]
    return JudgeResult(score=score, verdicts=verdicts, usage=usage)


def _conv(index, ratings=None, speakers=("usr", "sys"), strategies=(None,)):
    scenario_id = f"esconv-validation-{index:04d}"
    ratings = SeekerRatings(**(ratings if ratings is not None else
                             {"empathy": 3, "relevance": 3, "emotion_before": 4, "emotion_after": 2}))
    scenario = Scenario(scenario_id=scenario_id, split="validation", emotion_type="e",
                        problem_type="p", situation="SITUATION-SECRET", initial_intensity=4)
    record = ConversationRecord(id=f"conv-{scenario_id}", written_by="t",
                                created_at=datetime.now(timezone.utc), parent_ids=[],
                                source="real", scenario_id=scenario_id, split="validation",
                                policy_version="esconv-human",
                                turns=[Turn(speaker={"usr": "user", "sys": "helper"}[s], text=f"turn {i}")
                                       for i, s in enumerate(speakers)])
    return RealConversation(scenario=scenario, record=record, ratings=ratings, strategies=list(strategies))


class FakePanel:
    kind = "panel"

    def __init__(self, results, config=None):
        self.config = config or PanelConfig(judges=["model-a", "model-b", "model-c"], astra="model-astra")
        self.version = "judge-v1-test"
        self.results = results

    def score(self, record):
        out = self.results[record.id]
        if isinstance(out, Exception):
            raise out
        return out


def _use(monkeypatch, convs):
    monkeypatch.setattr(calibrate_module, "load_esconv", lambda split: convs)


def test_spearman():
    assert spearman([(1, 1), (2, 2), (3, 3)]) == 1.0
    assert spearman([(1, 3), (2, 2), (3, 1)]) == -1.0
    assert math.isclose(spearman([(1, 1), (1, 2), (2, 3)]), 0.8660254, abs_tol=1e-6)
    assert spearman([(2, 1), (2, 2), (2, 3)]) is None
    assert spearman([(1, 1)]) is None


def test_qwk():
    assert qwk([(1, 1), (3, 3), (5, 5)]) == 1
    assert qwk([(1, 5), (5, 1)]) == -1
    assert qwk([(3, 3), (3, 3)]) is None
    assert qwk([]) is None


def test_agreement():
    a = agreement([(5, 3), (4, 3), (3, 3)])
    assert a.n == 3 and a.mean_abs_error == 1.0 and a.exact_match == pytest.approx(1 / 3)
    assert a.spearman is None  # human side constant
    empty = agreement([])
    assert empty.n == 0 and empty.mean_abs_error is None and empty.qwk is None


def test_calibrate_sources_and_agreement(monkeypatch):
    convs = [_conv(i) for i in range(3)]
    _use(monkeypatch, convs)
    panel = FakePanel({c.record.id: _result(c.record.id, empathy=c.ratings.empathy) for c in convs})
    report, details = calibrate("validation", panel)
    assert report.n == 3 and report.failed == [] and report.judge_version == "judge-v1-test"
    assert set(report.sources) == {"astra", "A/initial", "A/final", "B/initial",
                                   "B/final", "C/initial", "C/final"}
    assert report.sources["astra"].model == "model-astra"
    assert report.sources["A/initial"].model == "model-a"
    assert report.sources["B/final"].model == "model-b"
    assert report.agreement["empathy"] == report.sources["astra"].empathy
    assert report.agreement["empathy"].n == 3 and report.agreement["empathy"].mean_abs_error == 0
    assert len(details) == 3 and details[0]["conversation_id"] == "conv-esconv-validation-0000"
    assert details[0]["score"]["note"] == "astra note"
    assert len(details[0]["verdicts"]) == 3


def test_missing_seeker_ratings_excluded_per_dim(monkeypatch):
    convs = [_conv(0, ratings={"empathy": None, "relevance": 4, "emotion_before": None,
                               "emotion_after": 2})]
    _use(monkeypatch, convs)
    panel = FakePanel({convs[0].record.id: _result(convs[0].record.id)})
    report, _ = calibrate("validation", panel)
    astra = report.sources["astra"]
    assert astra.empathy.n == 0 and astra.relevance.n == 1
    assert astra.emotion_before_mae is None
    assert astra.emotion_after_mae is not None
    assert astra.emotion_change_mae is None  # needs both intensities


def test_emotion_change_and_strategies(monkeypatch):
    convs = [_conv(0, ratings={"empathy": 3, "relevance": 3, "emotion_before": 4, "emotion_after": 2},
                   strategies=["Question", None], speakers=("sys", "usr", "sys"))]
    _use(monkeypatch, convs)
    result = _result(convs[0].record.id)
    result.score = result.score.model_copy(update={
        "emotion_before": 5, "emotion_after": 1, "strategies": ["Question", "Information"]})
    panel = FakePanel({convs[0].record.id: result})
    report, _ = calibrate("validation", panel)
    astra = report.sources["astra"]
    # judge change 4 vs seeker change 2 -> mae 2
    assert astra.emotion_change_mae == 2.0
    # only the first helper turn has a gold label
    assert astra.strategy_n == 1 and astra.strategy_exact_match == 1.0


def test_sampling_deterministic_and_sorted(monkeypatch):
    convs = [_conv(i) for i in range(4)]
    _use(monkeypatch, convs)
    panel = FakePanel({c.record.id: _result(c.record.id) for c in convs})
    r1, d1 = calibrate("validation", panel, n=2, seed=7)
    r2, d2 = calibrate("validation", panel, n=2, seed=7)
    assert r1.n == 2 and r1.n_requested == 2
    assert [d["conversation_id"] for d in d1] == [d["conversation_id"] for d in d2]
    assert [d["conversation_id"] for d in d1] == sorted(d["conversation_id"] for d in d1)


def test_ineligible_excluded(monkeypatch):
    convs = [_conv(0),
             _conv(1, speakers=("usr", "usr")),  # no helper turn
             _conv(2, ratings={"empathy": None, "relevance": None, "emotion_before": None,
                               "emotion_after": None})]
    _use(monkeypatch, convs)
    panel = FakePanel({convs[0].record.id: _result(convs[0].record.id)})
    report, _ = calibrate("validation", panel)
    assert report.n == 1


def test_failure_recorded_and_others_propagate(monkeypatch):
    convs = [_conv(0), _conv(1)]
    _use(monkeypatch, convs)
    panel = FakePanel({convs[0].record.id: _result(convs[0].record.id),
                       convs[1].record.id: JudgeFailed("judge A failed")})
    report, details = calibrate("validation", panel)
    assert report.failed == [convs[1].record.id] and report.n == 1
    assert len(details) == 1
    panel = FakePanel({convs[0].record.id: ValueError("boom")})
    with pytest.raises(ValueError):
        calibrate("validation", panel)


def test_test_split_rejected_before_loading(monkeypatch):
    called = []
    monkeypatch.setattr(calibrate_module, "load_esconv", lambda s: called.append(s))
    with pytest.raises(ValueError, match="train.*validation"):
        calibrate("test", FakePanel({}))
    assert not called


def test_usage_aggregated_per_model(monkeypatch):
    convs = [_conv(0), _conv(1)]
    _use(monkeypatch, convs)
    panel = FakePanel({c.record.id: _result(c.record.id) for c in convs})
    report, _ = calibrate("validation", panel)
    assert report.usage["model-a"] == {"calls": 4, "prompt_tokens": 60, "completion_tokens": 22}
    assert report.usage["model-astra"] == {"calls": 2, "prompt_tokens": 60, "completion_tokens": 14}


def test_freeze_and_check(tmp_path, monkeypatch):
    convs = [_conv(0)]
    _use(monkeypatch, convs)
    panel = FakePanel({convs[0].record.id: _result(convs[0].record.id)})
    train_report, _ = calibrate("train", panel)
    path = tmp_path / "cal" / "frozen.json"
    with pytest.raises(ValueError, match="validation"):
        freeze(train_report, path)
    val_report, _ = calibrate("validation", panel)
    frozen = freeze(val_report, path)
    assert frozen.frozen and path.is_file()
    assert check_frozen(panel, path).judge_version == "judge-v1-test"
    other = FakePanel({})
    other.version = "judge-v1-other"
    with pytest.raises(ValueError, match="judge-v1-test"):
        check_frozen(other, path)
    with pytest.raises(ValueError, match="not frozen"):
        check_frozen(panel, tmp_path / "missing.json")


def test_report_json_has_no_turn_or_situation_text(monkeypatch):
    convs = [_conv(0)]
    _use(monkeypatch, convs)
    panel = FakePanel({convs[0].record.id: _result(convs[0].record.id)})
    report, details = calibrate("validation", panel)
    blob = report.model_dump_json() + str(details)
    assert "SITUATION-SECRET" not in blob and "turn 0" not in blob and "turn 1" not in blob


def test_calibrate_fallback_judge(monkeypatch, tmp_path):
    from puddles.eval.judge import FallbackConfig

    class FakeFallback:
        kind = "fallback"
        config = FallbackConfig(model="m-fb")
        version = "judge-fallback-v1-deadbeef"

        def __init__(self, results):
            self.results = results

        def score(self, record):
            out = self.results[record.id]
            if isinstance(out, Exception):
                raise out
            return out

    convs = [_conv(0), _conv(1)]
    _use(monkeypatch, convs)
    judge = FakeFallback({c.record.id: _result(c.record.id, empathy=c.ratings.empathy) for c in convs})
    report, details = calibrate("validation", judge)
    assert report.judge_kind == "fallback"
    assert report.judge_config == judge.config
    assert set(report.sources) == {"fallback"}
    assert report.sources["fallback"].model == "m-fb"
    assert report.agreement["empathy"] == report.sources["fallback"].empathy
    assert report.agreement["empathy"].mean_abs_error == 0

    from puddles.eval.calibrate import frozen_path
    assert frozen_path("panel").name == "frozen.json"
    assert frozen_path("fallback").name == "frozen-fallback.json"
    path = tmp_path / "frozen-fallback.json"
    frozen = freeze(report, path)
    assert frozen.frozen and path.is_file()
    assert check_frozen(judge, path).judge_version == "judge-fallback-v1-deadbeef"
    panel = FakePanel({})
    with pytest.raises(ValueError, match="judge-fallback-v1-deadbeef"):
        check_frozen(panel, path)  # the panel's version is not the fallback's
