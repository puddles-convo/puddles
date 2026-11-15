"""Stage 2, lane A: the ESConv loader and the calibration script. Offline: the rows are made up here."""
import json

import pytest

from puddles import esconv
from schemas import JudgeOutput
from scripts import calibrate_judge
from tests.fakes import ScriptedLLM


def row(i, *, ratings=True, initial="4"):
    seeker = {"initial_emotion_intensity": initial}
    if ratings:
        seeker.update({"empathy": str(1 + i % 5), "relevance": str(5 - i % 5), "final_emotion_intensity": "2"})
    return {"experience_type": "Current Experience", "emotion_type": "anxiety", "problem_type": "job crisis",
            "situation": f" Made-up situation number {i}. ", "survey_score": {"seeker": seeker, "supporter": {}},
            "dialog": [{"speaker": "sys", "text": "Hello, how are you today? ", "strategy": "Question"},
                       {"speaker": "usr", "text": f"Made-up line {i}."},
                       {"speaker": "usr", "text": "  "},
                       {"speaker": "sys", "text": "That sounds hard.", "strategy": "Reflection of feelings"}]}


class Hub:
    """Stands in for Hugging Face: one made-up file per split, of the published size."""

    def __init__(self, unrated=(1, 7)):
        self.calls = []
        self.unrated = set(unrated)

    def __call__(self, url):
        self.calls.append(url)
        name = url.rsplit("/", 1)[1]
        split = {v: k for k, v in esconv.FILES.items()}[name]
        return "".join(json.dumps(row(i, ratings=i not in self.unrated)) + "\n"
                       for i in range(esconv.SIZES[split]))


def test_scenarios_come_by_split_with_stable_ids(tmp_path):
    hub = Hub()
    train = esconv.scenarios("train", cache_dir=tmp_path, fetch=hub)
    validation = esconv.scenarios("validation", cache_dir=tmp_path, fetch=hub)

    assert (len(train), len(validation)) == (910, 195)
    assert train[0].scenario_id == "esconv-train-0000" and train[-1].scenario_id == "esconv-train-0909"
    assert validation[42].scenario_id == "esconv-validation-0042" and validation[42].split == "validation"
    first = validation[0]
    assert (first.emotion_type, first.problem_type, first.initial_intensity) == ("anxiety", "job crisis", 4)
    assert first.situation == "Made-up situation number 0."
    assert hub.calls == [f"{esconv.BASE_URL}/train.txt", f"{esconv.BASE_URL}/valid.txt"]


def test_the_test_split_needs_an_explicit_argument(tmp_path):
    hub = Hub()
    for load in (esconv.scenarios, esconv.real_conversations, esconv.load_rows):
        with pytest.raises(esconv.EsconvError, match="allow_test"):
            load("test", cache_dir=tmp_path, fetch=hub)
    assert hub.calls == [] and list(tmp_path.iterdir()) == []
    assert len(esconv.scenarios("test", allow_test=True, cache_dir=tmp_path, fetch=hub)) == 195
    with pytest.raises(esconv.EsconvError, match="unknown split"):
        esconv.scenarios("live", cache_dir=tmp_path, fetch=hub)  # type: ignore[arg-type]


def test_the_download_happens_once_and_the_cache_is_used_after(tmp_path):
    hub = Hub()
    first = esconv.scenarios("validation", cache_dir=tmp_path, fetch=hub)

    def no_fetch(url):
        raise AssertionError("the cache should have been used")

    assert esconv.scenarios("validation", cache_dir=tmp_path, fetch=no_fetch) == first
    assert len(hub.calls) == 1 and (tmp_path / "valid.txt").exists()


def test_the_cache_folder_can_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(esconv.CACHE_VARIABLE, str(tmp_path / "elsewhere"))
    esconv.load_rows("validation", fetch=Hub())
    assert (tmp_path / "elsewhere" / "valid.txt").exists()


def test_a_file_of_the_wrong_size_is_refused(tmp_path):
    (tmp_path / "valid.txt").write_text(json.dumps(row(0)) + "\n")
    with pytest.raises(esconv.EsconvError, match="195"):
        esconv.load_rows("validation", cache_dir=tmp_path)


def test_real_conversations_carry_the_users_own_ratings(tmp_path):
    rated = esconv.real_conversations("validation", cache_dir=tmp_path, fetch=Hub(unrated=(1, 7)))

    assert len(rated) == 193  # rows 1 and 7 gave no ratings; the others keep their row's ID
    assert [c.record.id for c in rated[:3]] == ["esconv-validation-0000", "esconv-validation-0002",
                                                "esconv-validation-0003"]
    third = rated[2]
    assert third.ratings == esconv.UserRatings(empathy=4, relevance=2, emotion_before=4, emotion_after=2)
    record = third.record
    assert (record.source, record.split, record.scenario_id) == ("real", "validation", record.id)
    assert [(t.speaker, t.text) for t in record.turns] == [
        ("helper", "Hello, how are you today?"), ("user", "Made-up line 3."), ("helper", "That sounds hard.")]


def test_a_row_without_a_starting_intensity_gives_no_scenario(tmp_path):
    rows = [row(i, initial="" if i == 5 else "3") for i in range(195)]
    (tmp_path / "valid.txt").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ids = [s.scenario_id for s in esconv.scenarios("validation", cache_dir=tmp_path)]
    assert len(ids) == 194 and "esconv-validation-0005" not in ids and "esconv-validation-0006" in ids


# ---- the calibration script ----

def test_rank_correlation_and_mean_absolute_error():
    assert calibrate_judge.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert calibrate_judge.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # with ties: ranks 1, 2.5, 2.5, 4 against 1, 2, 3, 4
    assert calibrate_judge.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)
    assert calibrate_judge.spearman([3, 3, 3], [1, 2, 3]) is None
    assert calibrate_judge.spearman([1], [1]) is None
    assert calibrate_judge.mean_abs_error([1, 2, 5], [2, 2, 3]) == pytest.approx(1.0)
    assert calibrate_judge.mean_abs_error([], []) is None
    assert calibrate_judge.ranks([10, 30, 10, 20]) == [1.5, 4.0, 1.5, 3.0]


def test_pick_is_spread_over_the_split_and_repeatable():
    items = list(range(193))
    picked = calibrate_judge.pick(items, 100)  # type: ignore[arg-type]
    assert len(picked) == len(set(picked)) == 100 and picked == sorted(picked)
    assert picked == calibrate_judge.pick(items, 100)  # type: ignore[arg-type]
    assert picked[0] == 0 and picked[-1] > 180
    assert calibrate_judge.pick(items, 500) == items  # type: ignore[arg-type]


def test_calibrate_saves_pairs_without_text_and_reports_agreement(tmp_path):
    conversations = esconv.real_conversations("validation", cache_dir=tmp_path / "cache", fetch=Hub())[:6]
    llm = ScriptedLLM()
    good = [{"empathy": c.ratings.empathy, "relevance": 3, "emotion_before": 4, "emotion_after": 2,
             "strategies": ["Question", "Others"], "note": "Private note about the conversation."}
            for c in conversations[:5]]
    bad = {**good[0], "strategies": []}
    llm.add(JudgeOutput, *good, bad, bad)  # the sixth conversation is left unscored
    report = calibrate_judge.calibrate(conversations, llm, "judge-test", split="validation",
                                       out_dir=tmp_path / "calibration")

    pairs_text = (tmp_path / "calibration" / "pairs.jsonl").read_text()
    pairs = [json.loads(line) for line in pairs_text.splitlines()]
    assert len(pairs) == 5
    assert set(pairs[0]) == {"conversation_id", "judge_empathy", "user_empathy", "judge_relevance",
                             "user_relevance"}
    assert "Made-up" not in pairs_text and "Private note" not in pairs_text
    assert pairs[1] == {"conversation_id": "esconv-validation-0002", "judge_empathy": 3, "user_empathy": 3,
                        "judge_relevance": 3, "user_relevance": 3}
    assert report == json.loads((tmp_path / "calibration" / "report.json").read_text())
    assert (report["judge_version"], report["n"], report["unscored"], report["split"]) == (
        "judge-test", 5, 1, "validation")
    assert report["frozen"] is False
    assert report["agreement"]["empathy"] == {"spearman": pytest.approx(1.0), "mean_abs_error": 0.0}
    assert report["agreement"]["relevance"]["spearman"] is None  # the judge gave every one a 3
    assert report["agreement"]["relevance"]["mean_abs_error"] == pytest.approx(1.4)


def test_the_script_cannot_be_pointed_at_the_test_split(capsys):
    with pytest.raises(SystemExit):
        calibrate_judge.main(["--split", "test"])
    assert "invalid choice" in capsys.readouterr().err


def test_the_script_stops_with_one_line_when_the_judge_model_is_not_set(monkeypatch, capsys):
    monkeypatch.delenv("PUDDLES_JUDGE_MODEL", raising=False)
    assert calibrate_judge.main([]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ConfigError") and err.count("\n") == 1
