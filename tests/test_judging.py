"""Ansh's judge in the graph, through puddles/judging.py. All offline: every judge here is a fake."""
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest

import puddles.eval.judge as eval_judge
from loop.llm import FakeClient
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.decision_log import DecisionLog
from puddles.eval.judge import JudgeFailed, JudgePanel, PanelConfig
from puddles.graph import NOTHING_SCORED_REASON, RoundDeps, run_round
from puddles.judging import (
    CRITIC_CONVERSATIONS,
    MeteredJudge,
    judge_models,
    record_usage,
    save_scores,
    score_conversations,
    scored_conversations,
    select_for_critic,
    to_scored,
    train_scores_path,
)
from puddles.memory import FakeMemory
from puddles.round import unscored_sentence
from puddles.usage import Meter, MissingPrice, ModelPrice, Prices, metering, read_usage
from schemas import ConversationRecord, CriticOutput, JudgeOutput, ScoredConversation, ScoreRecord, Turn
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")
NOT_FOUND = {"found": False, "reason": "The conversations went well."}
PRICES = Prices(models={"judge-model": ModelPrice(1.0, 2.0)})


def record(cid="conv-r01-a", turns=None):
    turns = turns or [Turn(speaker="user", text="I feel low."), Turn(speaker="helper", text="Tell me more."),
                      Turn(speaker="user", text="Work."), Turn(speaker="helper", text="That sounds hard.")]
    return ConversationRecord(id=cid, written_by="test", created_at=datetime.now(timezone.utc), parent_ids=[],
                              source="simulated", scenario_id="a", split="train", policy_version="v1", turns=turns)


def score(cid, empathy=3, relevance=3, note="A note."):
    return ScoreRecord(id=f"score-{cid}", written_by="judge", created_at=datetime.now(timezone.utc),
                       parent_ids=[cid], conversation_id=cid, judge_version="judge-test", empathy=empathy,
                       relevance=relevance, emotion_before=3, emotion_after=2, strategies=["Question"], note=note)


def usage(model="judge-model", prompt_tokens=100, completion_tokens=20):
    return SimpleNamespace(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class FakeJudge:
    """Scores from a table by conversation ID; raises for the IDs in `fail`; reports `per_call` usage."""

    version = "judge-test"

    def __init__(self, scores=None, fail=(), per_call=(), wrong_id=()):
        self.scores, self.fail, self.per_call, self.wrong_id = scores or {}, set(fail), list(per_call), set(wrong_id)
        self.calls: list[str] = []

    def score(self, rec):
        self.calls.append(rec.id)
        if rec.id in self.fail:
            raise JudgeFailed(f"astra judgement by judge astra failed for {rec.id}")
        cid = "conv-somebody-else" if rec.id in self.wrong_id else rec.id
        s = self.scores.get(rec.id) or score(cid)
        return SimpleNamespace(score=s, usage=list(self.per_call))


@pytest.fixture
def traced(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


def make_deps(root, judge, llm, prices=None):
    return RoundDeps(helper_client=FakeClient(_fake_helper), user_client=FakeClient(_fake_user), llm=llm,
                     search=FakeSearch({}), memory=FakeMemory(), log=DecisionLog(root), root=root,
                     judge_panel=judge, usage=Meter(root, prices or Prices()))


def round_events(root):
    return [e for e in DecisionLog(root).events() if e.event == "round"]


# 1
def test_a_scored_record_becomes_a_scored_conversation_with_its_turns():
    rec = record()
    judge = FakeJudge(scores={rec.id: score(rec.id, empathy=2, relevance=4, note="Turn 2 asked well.")})
    run = score_conversations(judge, [rec])
    assert run.unscored == [] and run.reasons == {}
    [scored] = scored_conversations([rec], run.scores)
    assert scored == ScoredConversation(conversation_id=rec.id, empathy=2, relevance=4, note="Turn 2 asked well.",
                                        turns=rec.turns)
    assert to_scored(rec, run.scores[0]) == scored


def test_a_score_with_no_record_is_refused():
    with pytest.raises(ValueError, match="conv-r01-z"):
        scored_conversations([record("conv-r01-a")], [score("conv-r01-z")])


def test_scores_are_paired_in_the_order_of_the_scores():
    recs = [record("conv-r01-a"), record("conv-r01-b")]
    assert [s.conversation_id for s in scored_conversations(recs, [score("conv-r01-b"), score("conv-r01-a")])] \
        == ["conv-r01-b", "conv-r01-a"]


# 2
def test_a_judge_that_raises_leaves_that_conversation_unscored_and_scores_the_rest():
    recs = [record("conv-r01-a"), record("conv-r01-b"), record("conv-r01-c")]
    judge = FakeJudge(fail={"conv-r01-b"})
    run = score_conversations(judge, recs)
    assert judge.calls == ["conv-r01-a", "conv-r01-b", "conv-r01-c"]
    assert [s.conversation_id for s in run.scores] == ["conv-r01-a", "conv-r01-c"]
    assert run.unscored == ["conv-r01-b"]
    assert run.reasons["conv-r01-b"].startswith("JudgeFailed: ")
    assert "conv-r01-b" in run.reasons["conv-r01-b"]


def test_a_score_for_another_conversation_is_unscored_but_its_tokens_are_counted(tmp_path):
    meter = Meter(tmp_path, PRICES)
    judge = FakeJudge(wrong_id={"conv-r01-a"}, per_call=[usage()])
    with metering(meter, "judge"):
        run = score_conversations(judge, [record("conv-r01-a")])
    assert run.scores == [] and run.unscored == ["conv-r01-a"]
    assert "conv-somebody-else" in run.reasons["conv-r01-a"]
    assert [line["tokens_in"] for line in read_usage(tmp_path / "log/usage.jsonl") if line["event"] == "done"] \
        == [100]


def test_an_interrupt_is_not_taken_for_an_unscored_conversation():
    class Interrupted(FakeJudge):
        def score(self, rec):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        score_conversations(Interrupted(), [record()])


def test_a_round_with_an_unscored_conversation_reaches_finish_and_names_it(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND)
    judge = FakeJudge(fail={"conv-r01-fixture-0002"})
    state = run_round(make_deps(tmp_path, judge, llm), policy_path=POLICY, scenarios=SCENARIOS)
    assert state["unscored"] == ["conv-r01-fixture-0002"]
    assert state["result"].unscored == ["conv-r01-fixture-0002"]
    event = state["event"]
    assert event.outcome == "no_pattern"
    assert event.reason == "The conversations went well. The judge left 1 conversation unscored: conv-r01-fixture-0002."
    assert round_events(tmp_path)[-1].reason == event.reason
    # The critic read the four that were scored, and the scores file holds those four.
    assert len(state["scored"]) == len(SCENARIOS) - 1
    lines = train_scores_path(tmp_path, 1).read_text().splitlines()
    assert sorted(json.loads(line)["conversation_id"] for line in lines) == \
        sorted(f"conv-r01-{s.scenario_id}" for s in SCENARIOS if s.scenario_id != "fixture-0002")
    run = mlflow.search_runs(experiment_names=["puddles"]).iloc[-1]
    assert run["metrics.conversations_unscored"] == 1


def test_a_round_whose_judge_scores_nothing_names_every_conversation(tmp_path, traced):
    ids = [f"conv-r01-{s.scenario_id}" for s in SCENARIOS]
    llm = ScriptedLLM()  # the critic is never asked
    state = run_round(make_deps(tmp_path, FakeJudge(fail=set(ids)), llm), policy_path=POLICY, scenarios=SCENARIOS)
    assert state["event"].outcome == "no_pattern"
    assert state["event"].reason == f"{NOTHING_SCORED_REASON} {unscored_sentence(ids)}"
    assert f"The judge left {len(ids)} conversations unscored: " + ", ".join(ids) + "." in state["event"].reason
    assert train_scores_path(tmp_path, 1).read_text() == ""
    assert llm.prompts[CriticOutput] == []


def test_without_a_judge_panel_the_old_judge_is_called(tmp_path, traced):
    calls = []

    def old_judge(records):
        calls.append(len(records))
        return []

    deps = make_deps(tmp_path, None, ScriptedLLM())
    deps.judge = old_judge
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    assert calls == [len(SCENARIOS)]
    assert state["event"].reason == NOTHING_SCORED_REASON
    assert not (tmp_path / "scores").exists()


def test_the_sentence_counts_one_and_many():
    assert unscored_sentence(["a"]) == "The judge left 1 conversation unscored: a."
    assert unscored_sentence(["a", "b"]) == "The judge left 2 conversations unscored: a, b."


# 3
def scored(cid, empathy, relevance):
    return ScoredConversation(conversation_id=cid, empathy=empathy, relevance=relevance, note="n", turns=[])


def test_select_for_critic_takes_the_lowest_n_in_a_fixed_order():
    items = [scored("c", 3, 3), scored("a", 4, 2), scored("b", 1, 2), scored("d", 5, 5), scored("e", 2, 1)]
    assert [s.conversation_id for s in select_for_critic(items, 3)] == ["b", "e", "a"]
    assert [s.conversation_id for s in select_for_critic(items, 10)] == ["b", "e", "a", "c", "d"]
    assert select_for_critic([], 3) == []
    assert CRITIC_CONVERSATIONS == 10
    with pytest.raises(ValueError):
        select_for_critic(items, 0)


def test_the_round_passes_the_critic_only_the_lowest(tmp_path, traced):
    ids = [f"conv-r01-{s.scenario_id}" for s in SCENARIOS]
    table = {cid: score(cid, empathy=1 + i % 5, relevance=5 - i % 3) for i, cid in enumerate(ids)}
    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND)
    deps = make_deps(tmp_path, FakeJudge(scores=table), llm)
    deps.critic_conversations = 2
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    expected = sorted(ids, key=lambda c: ((table[c].empathy + table[c].relevance) / 2, c))[:2]
    assert [s.conversation_id for s in state["scored"]] == expected
    assert len(train_scores_path(tmp_path, 1).read_text().splitlines()) == len(SCENARIOS)


# 4
def test_the_scores_file_has_one_valid_score_record_per_line(tmp_path):
    path = train_scores_path(tmp_path, 3)
    assert path == tmp_path / "scores" / "round-03-train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("old content\n")
    scores = [score("conv-r03-a"), score("conv-r03-b", empathy=5)]
    save_scores(scores, path)
    lines = path.read_text().splitlines()
    assert [ScoreRecord.model_validate_json(line) for line in lines] == scores
    save_scores([], path)
    assert path.read_text() == ""


# 5
def test_reported_tokens_reach_the_usage_log_under_the_judge(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND)
    judge = FakeJudge(per_call=[usage(prompt_tokens=100, completion_tokens=20),
                                usage(prompt_tokens=7, completion_tokens=3)])
    deps = make_deps(tmp_path, judge, llm, prices=PRICES)
    run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    lines = [line for line in read_usage(tmp_path / "log/usage.jsonl") if line.get("agent") == "judge"]
    done = [line for line in lines if line["event"] == "done"]
    assert len(done) == 2 * len(SCENARIOS)
    assert all(line["model"] == "judge-model" and line["estimated"] is False for line in done)
    assert sorted({(line["tokens_in"], line["tokens_out"]) for line in done}) == [(7, 3), (100, 20)]
    assert sum(line["event"] == "sent" for line in lines) == 2 * len(SCENARIOS)
    assert deps.usage.usage["judge"].tokens_in == 107 * len(SCENARIOS)


def test_the_metered_judge_counts_under_the_current_agent(tmp_path):
    inner = FakeJudge(per_call=[usage(prompt_tokens=11, completion_tokens=5)])
    metered = MeteredJudge(inner)
    assert metered.version == inner.version
    meter = Meter(tmp_path, PRICES)
    with metering(meter, "gatekeeper"):
        result = metered.score(record())
    assert result.score.conversation_id == "conv-r01-a"
    [done] = [line for line in read_usage(tmp_path / "log/usage.jsonl") if line["event"] == "done"]
    assert (done["agent"], done["tokens_in"], done["tokens_out"]) == ("gatekeeper", 11, 5)


def test_the_metered_judge_passes_an_exception_through(tmp_path):
    with pytest.raises(JudgeFailed):
        MeteredJudge(FakeJudge(fail={"conv-r01-a"})).score(record())


def test_with_no_meter_nothing_is_written(tmp_path):
    record_usage([usage()])
    MeteredJudge(FakeJudge(per_call=[usage()])).score(record())
    assert not (tmp_path / "log").exists()


def test_a_judge_model_with_no_price_stops_the_scoring(tmp_path):
    judge = FakeJudge(per_call=[usage(model="unpriced-model")])
    with metering(Meter(tmp_path, PRICES), "judge"), pytest.raises(MissingPrice):
        score_conversations(judge, [record()])


def test_the_round_checks_the_judge_models_prices_before_it_starts(tmp_path, traced):
    class Priced(FakeJudge):
        config = SimpleNamespace(judges=["judge-model", "m-b", "m-c"], astra="m-astra")

    with pytest.raises(MissingPrice, match="m-astra"):
        run_round(make_deps(tmp_path, Priced(), ScriptedLLM(), prices=PRICES), policy_path=POLICY,
                  scenarios=SCENARIOS)
    assert not (tmp_path / "log" / "decisions.jsonl").exists()


def test_judge_models():
    config = PanelConfig(judges=["a", "b", "c"], astra="astra")
    assert judge_models(JudgePanel(config, [object()] * 3, object())) == ["a", "b", "c", "astra"]
    assert judge_models(SimpleNamespace(config=SimpleNamespace(model="fallback"))) == ["fallback"]
    assert judge_models(FakeJudge()) == []
    assert judge_models(None) == []


# 6
def output(helper_turns=2, empathy=3, relevance=3, note="Turn 2 asked an open question."):
    return JudgeOutput(empathy=empathy, relevance=relevance, emotion_before=4, emotion_after=3,
                       strategies=["Question"] * helper_turns, note=note)


class FakeJudgeLLM:
    """His JudgeLLM protocol: `model` and `judge(system, user) -> (JudgeOutput, prompt, completion)`."""

    def __init__(self, model, out):
        self.model, self.out, self.calls = model, out, 0

    def judge(self, system, user):
        self.calls += 1
        if isinstance(self.out, Exception):
            raise self.out
        return self.out, 200 + self.calls, 30


def test_his_real_panel_runs_through_the_adapter(tmp_path):
    config = PanelConfig(judges=["judge-model", "m-b", "m-c"], astra="m-astra")
    judges = [FakeJudgeLLM(m, output()) for m in config.judges]
    astra = FakeJudgeLLM("m-astra", output(empathy=2, relevance=4, note="Turn 4 moved to advice early."))
    panel = JudgePanel(config, judges, astra)
    no_helper = record("conv-r01-b", turns=[Turn(speaker="user", text="Hello?")])
    prices = Prices(models={m: ModelPrice(1.0, 1.0) for m in judge_models(panel)})
    with metering(Meter(tmp_path, prices), "judge"):
        run = score_conversations(panel, [record("conv-r01-a"), no_helper])
    [s] = run.scores
    assert (s.conversation_id, s.empathy, s.relevance, s.judge_version) == ("conv-r01-a", 2, 4, panel.version)
    assert run.unscored == ["conv-r01-b"] and run.reasons["conv-r01-b"].startswith("ValueError: ")
    [scored_] = scored_conversations([record("conv-r01-a")], run.scores)
    assert scored_.note == "Turn 4 moved to advice early."
    done = [line for line in read_usage(tmp_path / "log/usage.jsonl") if line["event"] == "done"]
    assert len(done) == 7  # three judges twice each, then Astra
    assert {line["model"] for line in done} == {"judge-model", "m-b", "m-c", "m-astra"}


def test_his_panel_failing_twice_leaves_the_conversation_unscored():
    config = PanelConfig(judges=["m-a", "m-b", "m-c"], astra="m-astra")
    bad = output(helper_turns=5)  # wrong number of strategy labels, so both answers fail his checks
    judges = [FakeJudgeLLM("m-a", bad), FakeJudgeLLM("m-b", output()), FakeJudgeLLM("m-c", output())]
    panel = JudgePanel(config, judges, FakeJudgeLLM("m-astra", output()))
    run = score_conversations(panel, [record("conv-r01-a")])
    assert run.scores == [] and run.unscored == ["conv-r01-a"]
    assert run.reasons["conv-r01-a"].startswith("JudgeFailed: ")
    assert judges[0].calls == 2


# The script
def test_the_fake_script_goes_through_the_adapter_and_writes_the_scores(tmp_path, traced):
    from scripts.run_graph_round import main

    root = tmp_path / "run"
    code = main(["--fake", "--root", str(root), "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"])
    assert code == 0
    lines = train_scores_path(root, 1).read_text().splitlines()
    assert len(lines) == len(SCENARIOS)
    assert {ScoreRecord.model_validate_json(line).judge_version for line in lines} == {"judge-fake"}


def test_the_fake_script_with_the_real_gate_gives_the_gate_the_same_judge(tmp_path, traced, monkeypatch):
    import scripts.run_graph_round as script

    made = []
    real_make_gate = script.real_gate.make_gate

    def spy(gate_deps, **kw):
        made.append(gate_deps.judge)
        return real_make_gate(gate_deps, **kw)

    monkeypatch.setattr(script.real_gate, "make_gate", spy)
    seen = []
    real_deps = script.RoundDeps

    def deps_spy(**kw):
        seen.append(kw["judge_panel"])
        return real_deps(**kw)

    monkeypatch.setattr(script, "RoundDeps", deps_spy)
    code = script.main(["--fake", "--gate", "real", "--root", str(tmp_path / "run"),
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"])
    assert code == 0
    [gate_judge], [panel] = made, seen
    assert isinstance(gate_judge, MeteredJudge) and gate_judge.judge is panel
    assert panel.version == "judge-fake"


def test_the_real_script_stops_on_missing_judge_configuration(tmp_path, monkeypatch, capsys):
    import loop.llm
    import puddles.llm
    import puddles.memory
    from scripts.run_graph_round import main

    monkeypatch.setattr(loop.llm, "client_from_env", lambda prefix: FakeClient(_fake_helper))
    monkeypatch.setattr(puddles.llm, "OpenAILLM", lambda: ScriptedLLM())
    monkeypatch.setattr(puddles.memory, "PastClient", lambda: FakeMemory())
    monkeypatch.setenv("EXA_API_KEY", "not-a-key")

    def missing():
        raise ValueError("PUDDLES_JUDGE_PANEL is not set")

    monkeypatch.setattr(eval_judge, "judge_from_env", missing)
    root = tmp_path / "run"
    assert main(["--root", str(root)]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert err == ["error: ValueError: PUDDLES_JUDGE_PANEL is not set"]
    assert not (root / "log").exists()
