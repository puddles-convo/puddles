"""Stage 2, lane A: the judge, the selection for the critic, and the judge inside a round. All offline."""
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pytest

from loop.errors import ConfigError
from loop.llm import FakeClient
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.agents import judge
from puddles.decision_log import DecisionLog
from puddles.graph import RoundDeps, run_round
from puddles.llm import AgentFailed
from puddles.memory import FakeMemory
from schemas import ConversationRecord, CriticOutput, JudgeOutput, ScoredConversation, ScoreRecord, Turn
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")
VERSION = "judge-test"
TURNS = [("user", "I'm feeling anxious that I'm going to lose my job."),
         ("helper", "That sounds really stressful. Have you tried talking to HR?"),
         ("user", "Not yet. I'm not ready to go back."),
         ("helper", "It might help to make a list of your concerns.")]


def conversation(conv_id="conv-r03-0017", turns=TURNS):
    return ConversationRecord(
        id=conv_id, written_by="helper", created_at=datetime(2026, 11, 14, tzinfo=timezone.utc),
        source="simulated", scenario_id="scenario-secret-0017", split="train",
        policy_version="policy-v9", turns=[Turn(speaker=s, text=t) for s, t in turns])


def answer(empathy=3, relevance=2, strategies=2, **over):
    return {"empathy": empathy, "relevance": relevance, "emotion_before": 4, "emotion_after": 4,
            "strategies": ["Providing Suggestions"] * strategies,
            "note": "In turn 2 the helper suggested HR before asking anything.", **over}


def scripted(*answers):
    llm = ScriptedLLM()
    llm.add(JudgeOutput, *answers)
    return llm


# ---- test 1: the judge on its own ----

def test_a_scripted_answer_becomes_a_valid_score_record():
    record = conversation()
    judged = judge.score_conversations([record], scripted(answer()), VERSION)

    assert judged.unscored == []
    [score] = judged.scores
    assert ScoreRecord.model_validate_json(score.model_dump_json()) == score
    assert (score.id, score.conversation_id, score.judge_version) == ("score-r03-0017", record.id, VERSION)
    assert (score.written_by, score.parent_ids) == ("judge", [record.id])
    assert (score.empathy, score.relevance, score.emotion_before, score.emotion_after) == (3, 2, 4, 4)
    assert score.strategies == ["Providing Suggestions"] * 2


def test_an_out_of_range_score_is_retried_once_with_the_reason():
    llm = scripted(answer(empathy=6), answer())
    judged = judge.score_conversations([conversation()], llm, VERSION)

    assert len(judged.scores) == 1 and judged.unscored == []
    first, second = llm.prompts[JudgeOutput]
    assert "failed these checks" not in first and "failed these checks" in second


def test_the_wrong_number_of_strategy_labels_is_retried_once():
    llm = scripted(answer(strategies=1), answer())
    judged = judge.score_conversations([conversation()], llm, VERSION)

    assert len(judged.scores) == 1
    assert "exactly 2 labels" in llm.prompts[JudgeOutput][1]


def test_check_catches_a_score_the_schema_let_through():
    out = JudgeOutput.model_construct(**answer(relevance=0))
    assert judge.check(out, judge.build_input(conversation())) == ["relevance must be between 1 and 5"]


def test_a_second_failure_leaves_the_conversation_unscored_and_the_batch_goes_on():
    records = [conversation("conv-r03-0001"), conversation("conv-r03-0002"), conversation("conv-r03-0003")]
    llm = scripted(answer(), answer(empathy=9), answer(strategies=5), answer())
    judged = judge.score_conversations(records, llm, VERSION)

    assert [s.conversation_id for s in judged.scores] == ["conv-r03-0001", "conv-r03-0003"]
    assert judged.unscored == ["conv-r03-0002"]
    assert "failed its checks twice" in judged.reasons["conv-r03-0002"]
    assert len(llm.prompts[JudgeOutput]) == 4


def test_run_raises_after_the_second_failure():
    with pytest.raises(AgentFailed):
        judge.run(judge.build_input(conversation()), scripted(answer(strategies=0), answer(strategies=0)))


def test_the_rendered_input_holds_the_turns_and_no_ids():
    record = conversation()
    inp = judge.build_input(record)
    text = judge.render_user(inp)

    assert set(inp.model_dump()) == {"turns"}
    for hidden in (record.id, "r03", "0017", record.scenario_id, record.policy_version, "policy", "train",
                   "simulated", "2026"):
        assert hidden not in text
    assert text.splitlines() == ["<conversation>", *[f"{i}. {s}: {t}" for i, (s, t) in enumerate(TURNS, 1)],
                                 "</conversation>"]


def test_the_system_prompt_comes_from_the_agent_doc():
    assert judge.SYSTEM.startswith("You are rating a support conversation")
    assert "empathy, relevance, emotion_before, emotion_after, strategies, note" in judge.SYSTEM
    assert judge.SYSTEM in (REPO / "docs" / "agents" / "t3-judge.md").read_text()


def test_the_judge_version_changes_with_the_model_and_with_the_prompt(monkeypatch):
    first = judge.judge_version("model-2026-01-01")
    assert first.startswith("judge-v1-") and first.endswith("-model-2026-01-01")
    assert judge.judge_version("model-2026-02-02") != first
    monkeypatch.setattr(judge, "SYSTEM", judge.SYSTEM + " Be strict.")
    assert judge.judge_version("model-2026-01-01") != first


def test_the_judge_model_has_its_own_variable_and_temperature_zero(monkeypatch):
    monkeypatch.delenv(judge.MODEL_VARIABLE, raising=False)
    with pytest.raises(ConfigError, match="PUDDLES_JUDGE_MODEL"):
        judge.llm_from_env()

    monkeypatch.setenv(judge.MODEL_VARIABLE, "judge-model-2026-01-01")
    monkeypatch.setenv("PUDDLES_AGENT_MODEL", "some-other-model")
    monkeypatch.setenv("PUDDLES_AGENT_TEMPERATURE", "0.8")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-not-a-real-key")  # the client is built, never called
    llm, version = judge.llm_from_env()
    assert llm.model == "judge-model-2026-01-01" and llm.extra == {"temperature": 0.0}
    assert version == judge.judge_version("judge-model-2026-01-01")


def test_scores_are_saved_one_record_per_line(tmp_path):
    records = [conversation("conv-r03-0001"), conversation("conv-r03-0002")]
    judged = judge.score_conversations(records, scripted(answer(), answer(empathy=5)), VERSION)
    path = judge.save_scores(judged.scores, judge.train_scores_path(tmp_path, 3))

    assert path == tmp_path / "scores" / "round-03-train.jsonl"
    assert len(path.read_text().splitlines()) == 2
    assert judge.load_scores(path) == judged.scores


def test_a_conversation_and_its_score_become_what_the_critic_takes():
    record = conversation()
    [score] = judge.score_conversations([record], scripted(answer()), VERSION).scores
    scored = judge.to_scored(record, score)

    assert scored == ScoredConversation(conversation_id=record.id, empathy=3, relevance=2,
                                        note=score.note, turns=record.turns)
    with pytest.raises(ValueError):
        judge.to_scored(conversation("conv-r03-9999"), score)


# ---- test 2: select_for_critic ----

def sc(conv_id, empathy, relevance):
    return ScoredConversation(conversation_id=conv_id, empathy=empathy, relevance=relevance, note="n", turns=[])


def test_select_for_critic_returns_the_lowest_n_in_a_fixed_order():
    scored = [sc("conv-e", 5, 5), sc("conv-d", 2, 3), sc("conv-b", 3, 2), sc("conv-a", 4, 1),
              sc("conv-c", 1, 1), sc("conv-f", 3, 3)]
    picked = judge.select_for_critic(scored, 4)

    # conv-c is lowest at 1.0; a, b and d tie at 2.5 and go by ID
    assert [c.conversation_id for c in picked] == ["conv-c", "conv-a", "conv-b", "conv-d"]
    assert judge.select_for_critic(list(reversed(scored)), 4) == picked


def test_select_for_critic_returns_all_of_them_when_there_are_fewer_than_n():
    scored = [sc("conv-b", 4, 4), sc("conv-a", 2, 2)]
    assert [c.conversation_id for c in judge.select_for_critic(scored, 10)] == ["conv-a", "conv-b"]
    assert judge.select_for_critic([], 10) == []
    assert judge.select_for_critic(scored, 0) == []
    assert len(judge.select_for_critic([sc(f"conv-{i:02d}", 3, 3) for i in range(15)])) == 10


# ---- the judge inside a round ----

NOT_FOUND = {"found": False, "reason": "The conversations went well."}


@pytest.fixture
def traced(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


def deps_with_judge(root, llm, **over):
    return RoundDeps(helper_client=FakeClient(_fake_helper), user_client=FakeClient(_fake_user), llm=llm,
                     search=FakeSearch({}), memory=FakeMemory(), log=DecisionLog(root), root=root,
                     judge_llm=llm, judge_version=VERSION, **over)


def round_answer(empathy):
    return answer(empathy=empathy, relevance=3, strategies=3)  # the fake helper speaks three times


def test_a_round_judges_every_conversation_and_saves_the_scores(tmp_path, traced):
    llm = scripted(*[round_answer(e) for e in (4, 2, 5, 1, 3)])
    llm.add(CriticOutput, NOT_FOUND)
    state = run_round(deps_with_judge(tmp_path, llm, critic_conversations=3),
                      policy_path=POLICY, scenarios=SCENARIOS)

    saved = judge.load_scores(tmp_path / "scores" / "round-01-train.jsonl")
    assert [s.conversation_id for s in saved] == [r.id for r in state["records"]]
    assert {s.judge_version for s in saved} == {VERSION}
    assert state["unscored"] == []
    # the critic is given the three lowest, lowest first
    assert [c.conversation_id for c in state["scored"]] == [
        "conv-r01-fixture-0004", "conv-r01-fixture-0002", "conv-r01-fixture-0005"]
    [critic_prompt] = llm.prompts[CriticOutput]
    assert critic_prompt.count("<conversation id=") == 3
    for prompt in llm.prompts[JudgeOutput]:
        assert "fixture" not in prompt and "policy-v1" not in prompt and "conv-" not in prompt


def test_an_unscored_conversation_is_counted_and_the_round_goes_on(tmp_path, traced):
    bad = round_answer(7)
    llm = scripted(round_answer(3), bad, bad, round_answer(3), round_answer(3), round_answer(3))
    llm.add(CriticOutput, NOT_FOUND)
    state = run_round(deps_with_judge(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)

    assert state["unscored"] == ["conv-r01-fixture-0002"]
    assert len(state["scored"]) == len(SCENARIOS) - 1
    event = state["event"]
    assert event.outcome == "no_pattern"
    assert event.reason == (f"{NOT_FOUND['reason']} The judge left 1 conversation unscored: "
                            "conv-r01-fixture-0002.")
    [logged] = [json.loads(line) for line in DecisionLog(tmp_path).path.read_text().splitlines()]
    assert "1 conversation unscored" in logged["reason"]
    assert len((tmp_path / "scores" / "round-01-train.jsonl").read_text().splitlines()) == len(SCENARIOS) - 1
    [run] = mlflow.search_runs(output_format="list")
    assert run.data.metrics["conversations_unscored"] == 1


def test_a_round_in_which_nothing_is_scored_ends_without_the_critic(tmp_path, traced):
    bad = round_answer(0)
    llm = scripted(*[bad] * (2 * len(SCENARIOS)))
    state = run_round(deps_with_judge(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)

    assert state["scored"] == [] and len(state["unscored"]) == len(SCENARIOS)
    assert state["event"].outcome == "no_pattern"
    assert state["event"].reason.startswith("The judge scored no conversation. The judge left 5 conversations")
    assert not llm.prompts[CriticOutput]
    assert (tmp_path / "scores" / "round-01-train.jsonl").read_text() == ""


def test_without_a_judge_model_the_stub_still_runs(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND)
    deps = RoundDeps(helper_client=FakeClient(_fake_helper), user_client=FakeClient(_fake_user), llm=llm,
                     search=FakeSearch({}), memory=FakeMemory(), log=DecisionLog(tmp_path), root=tmp_path)
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)

    assert len(state["scored"]) == len(SCENARIOS) and "unscored" not in state
    assert not (tmp_path / "scores").exists()
    assert state["event"].reason == NOT_FOUND["reason"]


def test_the_fake_script_round_runs_the_real_judge_code(tmp_path, traced, capsys):
    from scripts.run_graph_round import FAKE_JUDGE_VERSION, main

    root = tmp_path / "root"
    argv = ["--fake", "--root", str(root), "--mlflow", f"sqlite:///{tmp_path}/script.db",
            "--policy", str(POLICY), "--scenarios", str(REPO / "fixtures" / "scenarios.jsonl")]
    assert main(argv) == 0
    saved = judge.load_scores(root / "scores" / "round-01-train.jsonl")
    assert len(saved) == len(SCENARIOS) and {s.judge_version for s in saved} == {FAKE_JUDGE_VERSION}
    assert all(len(s.strategies) == 3 for s in saved)
