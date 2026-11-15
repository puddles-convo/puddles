import io
import json

import pytest

import puddles.eval.data as data
from puddles.eval.data import download, load_esconv, load_scenarios
from schemas import Scenario


def _row(speakers, seeker=None):
    seeker = {
        "empathy": "4",
        "relevance": "5",
        "initial_emotion_intensity": "3",
        "final_emotion_intensity": "2",
        **(seeker or {}),
    }
    return {
        "dialog": speakers,
        "emotion_type": "anxiety",
        "experience_type": "Previous Experience",
        "problem_type": "academic pressure",
        "seeker_question1": "q1",
        "seeker_question2": "q2",
        "situation": "I am stressed about exams.",
        "supporter_question1": "a1",
        "supporter_question2": "a2",
        "survey_score": {"seeker": seeker, "supporter": {}},
    }


def _use_rows(monkeypatch, rows, split="validation"):
    text = "\n".join(json.dumps(row) for row in rows)
    monkeypatch.setattr(data, "download", lambda *args: text)
    monkeypatch.setitem(data.SPLITS, split, ("fake.txt", len(rows)))


def test_ids_are_zero_based_and_stable(monkeypatch):
    _use_rows(monkeypatch, [_row([{"speaker": "usr", "text": "hi"}]), _row([{"speaker": "usr", "text": "yo"}])])
    conversations = load_esconv("validation")
    assert [c.scenario.scenario_id for c in conversations] == ["esconv-validation-0000", "esconv-validation-0001"]
    assert [c.record.id for c in conversations] == ["conv-esconv-validation-0000", "conv-esconv-validation-0001"]


def test_speakers_map_and_order_is_preserved(monkeypatch):
    dialog = [
        {"speaker": "sys", "text": "hello", "strategy": "Question"},
        {"speaker": "usr", "text": "hi"},
        {"speaker": "sys", "text": "how?", "strategy": "Reflection of feelings"},
    ]
    _use_rows(monkeypatch, [_row(dialog)])
    (conversation,) = load_esconv("validation")
    assert [(t.speaker, t.text) for t in conversation.record.turns] == [
        ("helper", "hello"),
        ("user", "hi"),
        ("helper", "how?"),
    ]


def test_ratings_become_ints(monkeypatch):
    _use_rows(monkeypatch, [_row([{"speaker": "usr", "text": "hi"}])])
    (conversation,) = load_esconv("validation")
    assert conversation.ratings.empathy == 4
    assert conversation.ratings.relevance == 5
    assert conversation.ratings.emotion_before == 3
    assert conversation.ratings.emotion_after == 2


@pytest.mark.parametrize("bad", ["", " ", "0", "7", "high"])
def test_bad_rating_becomes_none_and_others_stay(monkeypatch, bad):
    _use_rows(monkeypatch, [_row([{"speaker": "usr", "text": "hi"}], seeker={"empathy": bad})])
    (conversation,) = load_esconv("validation")
    assert conversation.ratings.empathy is None
    assert conversation.ratings.relevance == 5
    assert conversation.ratings.emotion_before == 3
    assert conversation.ratings.emotion_after == 2


def test_strategies_one_per_helper_turn(monkeypatch):
    dialog = [
        {"speaker": "usr", "text": "hi"},
        {"speaker": "sys", "text": "a", "strategy": "Question"},
        {"speaker": "sys", "text": "b"},
        {"speaker": "usr", "text": "bye"},
    ]
    _use_rows(monkeypatch, [_row(dialog)])
    (conversation,) = load_esconv("validation")
    assert conversation.strategies == ["Question", None]


def test_record_fields(monkeypatch):
    _use_rows(monkeypatch, [_row([{"speaker": "usr", "text": "hi"}])])
    (conversation,) = load_esconv("validation")
    record = conversation.record
    assert record.source == "real"
    assert record.split == "validation"
    assert record.scenario_id == "esconv-validation-0000"
    assert record.policy_version == data.REAL_POLICY_VERSION
    scenario = conversation.scenario
    assert scenario.split == "validation"
    assert scenario.emotion_type == "anxiety"
    assert scenario.problem_type == "academic pressure"
    assert scenario.initial_intensity == 3


def test_row_count_mismatch_raises(monkeypatch):
    _use_rows(monkeypatch, [_row([{"speaker": "usr", "text": "hi"}])])
    monkeypatch.setitem(data.SPLITS, "validation", ("fake.txt", 2))
    with pytest.raises(ValueError, match="expected 2 rows, got 1"):
        load_esconv("validation")


def test_test_split_requires_allow_test(monkeypatch):
    _use_rows(monkeypatch, [_row([{"speaker": "usr", "text": "hi"}])], split="test")
    with pytest.raises(ValueError, match="reserved for the final evaluation"):
        load_esconv("test")
    assert len(load_esconv("test", allow_test=True)) == 1


def test_unknown_split_raises():
    with pytest.raises(ValueError, match="Unknown ESConv split"):
        load_esconv("dev")


def test_unknown_speaker_raises(monkeypatch):
    _use_rows(monkeypatch, [_row([{"speaker": "bot", "text": "hi"}])])
    with pytest.raises(ValueError, match="esconv-validation-0000.*'bot'"):
        load_esconv("validation")


def test_load_scenarios_returns_scenarios(monkeypatch):
    _use_rows(monkeypatch, [_row([{"speaker": "usr", "text": "hi"}])])
    scenarios = load_scenarios("validation")
    assert isinstance(scenarios[0], Scenario)
    assert scenarios[0].scenario_id == "esconv-validation-0000"


def test_download_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUDDLES_CACHE", str(tmp_path))
    payload = b"hello esconv"

    class _Response:
        def __enter__(self):
            return io.BytesIO(payload)

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(data, "urlopen", lambda url, timeout: _Response())
    assert download("some/repo", "rev1", "f.txt") == "hello esconv"

    def _boom(url, timeout):
        raise AssertionError("network should not be hit")

    monkeypatch.setattr(data, "urlopen", _boom)
    assert download("some/repo", "rev1", "f.txt") == "hello esconv"
    assert (tmp_path / "some" / "repo" / "rev1" / "f.txt").is_file()
