"""Offline tests for puddles.baselines and scripts.make_id_files: no keys, no network."""
import json
from datetime import datetime, timezone

import pytest

from loop.errors import LLMError
from loop.llm import FakeClient
from loop.run_batch import _fake_helper, _fake_user
from puddles.baselines import (
    BaselineError,
    BaselineResult,
    average,
    load_id_file,
    mean_scores,
    noise,
    read_results,
    resolve_ids,
    score_once,
    summarize,
)
from puddles.usage import Meter
from schemas import Scenario, ScoreRecord
from scripts.make_id_files import pick_ids
from scripts.score_baselines import ScriptedJudge

POLICY = """# test-policy-v1

## Safety (protected: no edit may change this section)
S1. Keep the person safe.

## Conversation
C1. Listen before suggesting.
"""


@pytest.fixture
def policy_path(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(POLICY, encoding="utf-8")
    return path


@pytest.fixture
def scenarios():
    return [
        Scenario(scenario_id=f"esconv-validation-{i:04d}", split="validation",
                 emotion_type="worry", problem_type="work", situation="My hours were cut.",
                 initial_intensity=4)
        for i in range(3)
    ]


def make_result(**changes):
    values = dict(policy_version="policy-v1", run=1, n=3, seconds=1.0, complete=True,
                  empathy=3.0, relevance=3.0, quality=3.0, emotion_change=1.0, seed=7,
                  max_turns=10, judge_version="judge-v1", scenario_ids_sha256="abc")
    values.update(changes)
    return BaselineResult(**values)


def score_record(conversation_id, empathy, relevance, before=4, after=3):
    return ScoreRecord(id=f"score-{conversation_id}", written_by="judge",
                       created_at=datetime.now(timezone.utc), parent_ids=[conversation_id],
                       conversation_id=conversation_id, judge_version="judge-v1",
                       empathy=empathy, relevance=relevance, emotion_before=before,
                       emotion_after=after, strategies=["Others"], note="test")


class QueueJudge:
    """Scripted judge returning the queued scores, or raising the queued exceptions."""

    version = "judge-test-v1"

    def __init__(self, results):
        self.results = list(results)

    class _Result:
        def __init__(self, score):
            self.score = score
            self.usage = []

    def score(self, record):
        out = self.results.pop(0)
        if isinstance(out, Exception):
            raise out
        return self._Result(out)


class FailingClient:
    def complete(self, system, messages, *, max_tokens, temperature=0.7, seed=None):
        raise LLMError("scripted failure")


class TestLoadIdFile:
    def test_reads_one_id_per_line(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("a\n\nb\n", encoding="utf-8")
        assert load_id_file(path) == ["a", "b"]

    def test_empty_file_is_an_error(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(BaselineError):
            load_id_file(path)

    def test_repeated_id_is_an_error(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("a\nb\na\n", encoding="utf-8")
        with pytest.raises(BaselineError, match="repeated"):
            load_id_file(path)


class TestResolveIds:
    def test_order_of_the_id_list(self, scenarios):
        ids = [s.scenario_id for s in reversed(scenarios)]
        assert [s.scenario_id for s in resolve_ids(scenarios, ids)] == ids

    def test_unknown_id(self, scenarios):
        with pytest.raises(BaselineError, match="unknown"):
            resolve_ids(scenarios, ["esconv-validation-9999"])

    def test_repeated_id(self, scenarios):
        with pytest.raises(BaselineError, match="repeated"):
            resolve_ids(scenarios, [scenarios[0].scenario_id, scenarios[0].scenario_id])

    def test_empty_list(self, scenarios):
        with pytest.raises(BaselineError):
            resolve_ids(scenarios, [])

    def test_test_split_is_refused(self, scenarios):
        test = scenarios[0].model_copy(update={"scenario_id": "esconv-test-0000",
                                             "split": "test"})
        with pytest.raises(BaselineError, match="test split"):
            resolve_ids([*scenarios, test], ["esconv-test-0000"])


class TestScoreOnce:
    def test_complete_run_writes_its_evidence(self, policy_path, scenarios, tmp_path):
        ids = [s.scenario_id for s in scenarios[:2]]
        run_dir = tmp_path / "run-1"
        result = score_once(policy_path, scenarios, ids,
                            helper_client=FakeClient(_fake_helper),
                            user_client=FakeClient(_fake_user),
                            judge=ScriptedJudge(), meter=Meter(tmp_path / "root"),
                            seed=7, max_turns=3, run=1, run_dir=run_dir)
        assert result.complete is True
        assert result.quality == 3.0
        assert result.empathy == 3.0 and result.relevance == 3.0
        assert result.emotion_change == 1.0
        assert result.judge_version == "judge-fake-v0"
        assert result.input_tokens > 0 and result.output_tokens > 0
        assert (run_dir / "conversations.jsonl").exists()
        scores = (run_dir / "scores.jsonl").read_text().splitlines()
        assert len(scores) == 2
        reloaded = json.loads((run_dir / "result.json").read_text())
        assert BaselineResult(**reloaded) == result

    def test_failed_conversation_leaves_scores_unset(self, policy_path, scenarios, tmp_path):
        result = score_once(policy_path, scenarios, [scenarios[0].scenario_id],
                            helper_client=FakeClient(_fake_helper),
                            user_client=FailingClient(),
                            judge=ScriptedJudge(), meter=Meter(tmp_path / "root"),
                            run=1, run_dir=tmp_path / "run-1")
        assert result.complete is False
        assert result.quality is None and result.empathy is None
        assert result.conversations_failed == [scenarios[0].scenario_id]

    def test_unscored_conversation_is_counted(self, policy_path, scenarios, tmp_path):
        judge = QueueJudge([RuntimeError("judge broke"), score_record("other-conv", 3, 3)])
        result = score_once(policy_path, scenarios, [s.scenario_id for s in scenarios[:2]],
                            helper_client=FakeClient(_fake_helper),
                            user_client=FakeClient(_fake_user),
                            judge=judge, meter=Meter(tmp_path / "root"),
                            run=1, run_dir=tmp_path / "run-1")
        assert result.complete is False
        assert len(result.unscored) == 2  # the crash, and the record whose ID did not match
        assert result.quality is None


class TestAverage:
    def test_gate_formula(self):
        records = [score_record("a", 2, 4, before=5, after=3),
                   score_record("b", 4, 4, before=4, after=2)]
        scores = average(records)
        assert scores.empathy == 3.0
        assert scores.relevance == 4.0
        assert scores.quality == 3.5  # mean of (2+4)/2 and (4+4)/2
        assert scores.emotion_change == 2.0  # mean of 5-3 and 4-2

    def test_empty_is_an_error(self):
        with pytest.raises(BaselineError):
            average([])


class TestNoise:
    def test_max_minus_min(self):
        results = [make_result(run=1, quality=3.0), make_result(run=2, quality=3.5),
                   make_result(run=3, quality=3.2)]
        assert noise(results) == pytest.approx(0.5)

    def test_one_run_is_an_error(self):
        with pytest.raises(BaselineError):
            noise([make_result()])

    def test_incomplete_runs_do_not_count(self):
        results = [make_result(run=1, quality=3.0), make_result(run=2, quality=3.5),
                   make_result(run=3, complete=False, quality=None)]
        assert noise(results) == pytest.approx(0.5)

    def test_different_seed_is_refused(self):
        with pytest.raises(BaselineError, match="cannot be compared"):
            noise([make_result(run=1, quality=3.0), make_result(run=2, quality=3.5, seed=8)])

    def test_different_judge_is_refused(self):
        with pytest.raises(BaselineError, match="cannot be compared"):
            noise([make_result(run=1, quality=3.0),
                   make_result(run=2, quality=3.5, judge_version="judge-v2")])

    def test_different_scenarios_are_refused(self):
        with pytest.raises(BaselineError, match="cannot be compared"):
            noise([make_result(run=1, quality=3.0),
                   make_result(run=2, quality=3.5, scenario_ids_sha256="other")])


class TestMeanScores:
    def test_means_over_complete_runs(self):
        results = [make_result(run=1, empathy=2.0, relevance=4.0, quality=3.0,
                               emotion_change=0.0),
                   make_result(run=2, empathy=4.0, relevance=4.0, quality=4.0,
                               emotion_change=1.0)]
        scores = mean_scores(results)
        assert scores.empathy == 3.0 and scores.relevance == 4.0
        assert scores.quality == 3.5 and scores.emotion_change == 0.5


class TestSummarize:
    def test_groups_runs_and_writes_results(self, tmp_path):
        for run, quality in ((1, 3.0), (2, 3.4)):
            path = tmp_path / f"policy-v1-run-{run}"
            path.mkdir()
            result = make_result(run=run, quality=quality)
            (path / "result.json").write_text(result.model_dump_json())
        summary = summarize(tmp_path)
        group = summary["groups"][0]["policies"]["policy-v1"]
        assert group["noise"] == pytest.approx(0.4)
        assert group["mean_scores"]["quality"] == pytest.approx(3.2)
        assert (tmp_path / "results.json").exists()
        assert len(read_results(tmp_path)) == 2


class TestPickIds:
    def test_deterministic_sorted_sample(self):
        ids = [f"id-{i:04d}" for i in range(50)]
        first = pick_ids(ids, 20, seed=7)
        assert first == pick_ids(ids, 20, seed=7)
        assert len(first) == 20 and first == sorted(first)
        assert pick_ids(ids, 20, seed=8) != first

    def test_asking_for_more_than_exists_is_an_error(self):
        with pytest.raises(BaselineError):
            pick_ids(["a"], 2, seed=7)
