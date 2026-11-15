"""Acceptance tests from docs/specs/judging.md, written without its implementation.

Deliberately unspecified: duplicate IDs, malformed judge results/configs, exact
mismatch reasons. A judge that raises returns no usage: a stated accounting limit.
"""

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import os
import socket
from types import SimpleNamespace

import mlflow
import pytest

from loop import llm as chat_api
from loop.llm import FakeClient
from puddles import exa, graph, judging, tracing
from puddles import llm as agent_api
from puddles import memory as memory_api
from puddles import round as round_api
from puddles.decision_log import DecisionLog
from puddles.eval import judge as judge_api
from puddles.eval.judge import JudgePanel, PanelConfig
from puddles.judging import (
    CRITIC_CONVERSATIONS,
    JudgeRun,
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
from puddles.llm import AgentFailed
from puddles.memory import FakeMemory
from puddles.usage import Meter, MissingPrice, ModelPrice, Prices, metering, read_usage
from schemas import (
    ConversationRecord,
    CriticOutput,
    JudgeOutput,
    PolicyEdit,
    ProposerOutput,
    ResearcherOutput,
    ScoreRecord,
    ScoredConversation,
    Turn,
)
from scripts import run_graph_round as script


STAMP = datetime(2026, 1, 2, tzinfo=timezone.utc)
MODEL_NAMES = ["acceptance-a", "acceptance-b", "acceptance-c", "acceptance-astra"]
QUERY = "acceptance listening before advice"
URL = "https://evidence.example/listening"


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.endswith("API_KEY") or key.startswith("PUDDLES_JUDGE") or key in {
            "PUDDLES_FALLBACK_JUDGE_MODEL", "MLFLOW_TRACKING_URI", "MLFLOW_TRACING_DESTINATION",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("Judging acceptance tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)


@pytest.fixture
def local_mlflow(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def round_policy(tmp_path):
    # The existing proposer advances titles of the form policy-vN.
    path = tmp_path / "starting.md"
    path.write_text(
        "# policy-v1\n\n## Safety (protected: no edit may change this section)\n"
        "S1. Keep the person safe.\n\n## Conversation\nC1. Listen before suggesting.\n",
        encoding="utf-8",
    )
    return path


def record(name):
    return ConversationRecord(
        id=name, written_by="helper", created_at=STAMP, source="simulated",
        scenario_id=name, split="train", policy_version="policy-v1",
        turns=[Turn(speaker="user", text=f"My situation is {name}."),
               Turn(speaker="helper", text=f"Tell me about {name}.")],
    )


def score(name, empathy=2, relevance=4):
    return ScoreRecord(
        id=f"score-{name}", written_by="judge", created_at=STAMP, parent_ids=[name],
        conversation_id=name, judge_version="acceptance-judge", empathy=empathy,
        relevance=relevance, emotion_before=5, emotion_after=2,
        strategies=["Question"], note=f"Judge note for {name}.",
    )


def usage(model=MODEL_NAMES[0], prompt=101, completion=7):
    return SimpleNamespace(model=model, prompt_tokens=prompt, completion_tokens=completion)


def answer(name, empathy=2, relevance=4, entries=()):
    return SimpleNamespace(score=score(name, empathy, relevance), usage=list(entries))


class ScriptedJudge:
    version = "acceptance-judge"

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def score(self, item):
        self.calls.append(item)
        outcome = self.outcomes[item.id] if isinstance(self.outcomes, dict) else self.outcomes(item)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class JudgeLLM:
    """The public JudgeLLM interface, with no client or network behind it."""

    def __init__(self, model, empathy=2, relevance=3):
        self.model = model
        self.output = JudgeOutput(
            empathy=empathy, relevance=relevance, emotion_before=5, emotion_after=2,
            strategies=["Question"], note="The helper asks about the user's situation.",
        )
        self.calls = []

    def judge(self, system, user):
        counts = (self.model, 100 + len(self.calls), 9 + len(self.calls))
        self.calls.append(counts)
        return self.output.model_copy(deep=True), counts[1], counts[2]


class RoundLLM:
    """Small scripted critic/researcher/proposer, including their failure paths."""

    def __init__(self, mode="no_pattern", ids=()):
        self.mode = mode
        self.ids = list(ids)
        self.calls = []

    def structured(self, system, user, schema):
        self.calls.append(schema)
        if self.mode == "critic_failed" and schema is CriticOutput:
            raise AgentFailed("scripted critic failed")
        if self.mode == "proposer_failed" and schema is ProposerOutput:
            raise AgentFailed("scripted proposer failed")
        if schema is CriticOutput:
            if self.mode == "no_pattern":
                return CriticOutput(found=False, reason="No repeated failure.")
            return CriticOutput(
                found=True, name="early advice", description="Advice before understanding.",
                seen_in=1, out_of=len(self.ids), example_conversation_ids=self.ids[:1],
            )
        if schema is ProposerOutput:
            return ProposerOutput(
                action="add", rule_text="Ask about the situation before suggesting a solution.",
                summary="Understand before advising.",
            )
        raise AssertionError(f"Unexpected schema {schema}")

    def structured_with_search(self, system, user, schema, search, max_searches):
        self.calls.append(schema)
        assert schema is ResearcherOutput
        if self.mode == "researcher_failed":
            raise AgentFailed("scripted researcher failed")
        search(QUERY)
        return ResearcherOutput(queries=[QUERY], sources=[] if self.mode == "no_evidence" else [
            {"url": URL, "title": "Listening", "technique": "Ask before advising."},
        ])


def prices(names=MODEL_NAMES):
    return Prices(models={name: ModelPrice(2.0, 5.0) for name in names})


def deps_for(root, **overrides):
    values = dict(
        helper_client=FakeClient(lambda system, messages: "What happened?"),
        user_client=FakeClient(lambda system, messages: "My work hours were cut."),
        llm=RoundLLM(), search=lambda query: [{"url": URL, "title": "Listening", "text": "Ask first."}],
        memory=FakeMemory(), log=DecisionLog(root), root=root, usage=Meter(root, prices()),
    )
    values.update(overrides)
    return graph.RoundDeps(**values)


def usage_rows(root):
    return [row for row in read_usage(root / "log/usage.jsonl") if row.get("event") == "done"]


def reported_rows(rows):
    return Counter((row["agent"], row["model"], row["tokens_in"], row["tokens_out"], row["estimated"])
                   for row in rows)


@contextmanager
def patch_public(monkeypatch, owner, name, replacement, *consumers):
    """Patch a public API and any already-imported aliases of that same API."""
    with monkeypatch.context() as patch:
        original = getattr(owner, name)
        patch.setattr(owner, name, replacement)
        for consumer in consumers:
            for alias, value in tuple(vars(consumer).items()):
                if value is original:
                    patch.setattr(consumer, alias, replacement)
        yield


# Section 1: isolation, attribution, usage, conversions, selection and persistence.


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError, AgentFailed, OSError])
def test_score_exceptions_leave_ordered_unscored_ids_and_continue_without_retry(exception_type):
    """Catches scoring a failed record, retrying it here, dropping its ID, or aborting the batch."""
    records = [record(name) for name in ("z-good", "b-failed", "a-good", "d-failed", "c-good")]
    failures = {"b-failed": exception_type("bad judgement: first failure"),
                "d-failed": exception_type("bad judgement: second failure")}
    outcomes = {item.id: failures[item.id] if item.id in failures else answer(item.id) for item in records}
    judge = ScriptedJudge(outcomes)
    run = score_conversations(judge, records)
    assert isinstance(run, JudgeRun)
    assert judge.calls == records
    assert run.scores == [outcomes[name].score for name in ("z-good", "a-good", "c-good")]
    assert run.unscored == ["b-failed", "d-failed"]
    for name, error in failures.items():
        assert type(error).__name__ in run.reasons[name]
        assert str(error) in run.reasons[name]


@pytest.mark.parametrize("wrong_id", ["b", "not-in-this-batch"])
def test_wrong_conversation_id_is_unscored_and_never_reassigned(wrong_id):
    """Catches accepting, relabelling, or attaching a returned score to another record."""
    records = [record("a"), record("b"), record("c")]
    good_b, good_c = answer("b", 5, 4), answer("c", 3, 2)
    judge = ScriptedJudge({"a": answer(wrong_id, 1, 1), "b": good_b, "c": good_c})
    run = score_conversations(judge, records)
    assert judge.calls == records
    assert run.scores == [good_b.score, good_c.score]
    assert run.unscored == ["a"]
    assert isinstance(run.reasons["a"], str) and run.reasons["a"].strip()


@pytest.mark.parametrize("wrong_id", ["b", "not-in-this-batch"])
def test_wrong_id_results_still_record_all_reported_usage_and_continue(tmp_path, wrong_id):
    """Catches discarding spent tokens along with a mismatched score, or recording them twice."""
    records = [record("a"), record("b"), record("c")]
    wrong = answer(wrong_id, entries=[usage(), usage(MODEL_NAMES[1], 0, 3)])
    good_b = answer("b", 5, 4, [usage(MODEL_NAMES[2], 17, 0)])
    good_c = answer("c", 3, 2)
    judge = ScriptedJudge({"a": wrong, "b": good_b, "c": good_c})
    with metering(Meter(tmp_path, prices()), "judge"):
        run = score_conversations(judge, records)
    assert judge.calls == records
    assert run.unscored == ["a"] and run.scores == [good_b.score, good_c.score]
    assert reported_rows(usage_rows(tmp_path)) == Counter([
        ("judge", MODEL_NAMES[0], 101, 7, False), ("judge", MODEL_NAMES[1], 0, 3, False),
        ("judge", MODEL_NAMES[2], 17, 0, False),
    ])


@pytest.mark.parametrize("wrong_id", [False, True])
@pytest.mark.parametrize("exception_type", [MissingPrice, RuntimeError])
def test_score_conversations_propagates_record_usage_errors(monkeypatch, wrong_id, exception_type):
    """Catches treating a metering error as an unscored conversation and quietly continuing."""
    records = [record("a"), record("b")]
    result = answer("some-other-id" if wrong_id else "a", entries=[usage()])
    judge = ScriptedJudge({"a": result, "b": answer("b")})
    error = exception_type("the returned usage cannot be counted")
    observed = []

    def fail_usage(entries):
        observed.append(list(entries))
        raise error

    monkeypatch.setattr(judging, "record_usage", fail_usage)
    with pytest.raises(exception_type) as caught:
        score_conversations(judge, records)
    assert caught.value is error
    assert observed == [result.usage]
    assert judge.calls == records[:1]


def test_successes_pass_each_usage_list_to_record_usage_once_in_record_order(monkeypatch):
    """Catches omitting usage, forwarding only the last result, or recording an entry twice."""
    records = [record("second"), record("first")]
    results = [answer("second", entries=[usage(), usage(MODEL_NAMES[1], 0, 3)]),
               answer("first", entries=[usage(MODEL_NAMES[2], 17, 0)])]
    observed = []
    monkeypatch.setattr(judging, "record_usage", lambda entries: observed.append(list(entries)))
    run = score_conversations(ScriptedJudge(dict(zip((r.id for r in records), results, strict=True))), records)
    assert observed == [result.usage for result in results]
    assert run.scores == [result.score for result in results]
    assert run.unscored == []


def test_empty_batch_does_not_call_judge():
    """Catches fabricated scores/failures or a judge call for an empty batch."""
    judge = ScriptedJudge({})
    run = score_conversations(judge, [])
    assert judge.calls == []
    assert run.scores == [] and run.unscored == [] and run.reasons == {}


def test_record_usage_uses_current_meter_agent_and_reported_counts(tmp_path):
    """Catches hardcoded judge attribution, estimates, omitted zero counts, or wrong token fields."""
    first, second = Meter(tmp_path / "first", prices(), round=7), Meter(tmp_path / "second", prices(), round=8)
    entries = [usage(), usage(MODEL_NAMES[1], 0, 3), usage(MODEL_NAMES[0], 17, 0)]
    snapshots = [vars(entry).copy() for entry in entries]
    with metering(first, "judge"):
        record_usage(iter(entries))
        with metering(second, "gatekeeper"):
            record_usage([usage(MODEL_NAMES[2], 29, 31)])
        record_usage([usage(MODEL_NAMES[3], 1, 2)])
    assert reported_rows(usage_rows(tmp_path / "first")) == Counter([
        ("judge", MODEL_NAMES[0], 101, 7, False), ("judge", MODEL_NAMES[1], 0, 3, False),
        ("judge", MODEL_NAMES[0], 17, 0, False), ("judge", MODEL_NAMES[3], 1, 2, False),
    ])
    assert reported_rows(usage_rows(tmp_path / "second")) == Counter([
        ("gatekeeper", MODEL_NAMES[2], 29, 31, False),
    ])
    assert all(row["round"] == 7 for row in usage_rows(tmp_path / "first"))
    assert [vars(entry) for entry in entries] == snapshots


def test_record_usage_without_current_meter_does_nothing(tmp_path):
    """Catches constructing an implicit meter or pricing an entry when there is no current meter."""
    with metering(None, "judge"):
        record_usage([usage("deliberately-unpriced")])
    assert not (tmp_path / "log/usage.jsonl").exists()


def test_record_usage_propagates_missing_price(tmp_path):
    """Catches swallowing the stage 1 meter's missing-price exception."""
    with metering(Meter(tmp_path, Prices()), "judge"), pytest.raises(MissingPrice):
        record_usage([usage("unpriced")])


def test_to_scored_and_id_join_preserve_scores_and_each_records_own_turns():
    """Catches positional zipping, dropped notes, invented numbers or another record's turns."""
    records = [record("unused"), record("b"), record("a")]
    scores = [score("a", 1, 5), score("b", 4, 2)]
    expected = [ScoredConversation(
        conversation_id=item.conversation_id, empathy=item.empathy, relevance=item.relevance,
        note=item.note, turns=next(r.turns for r in records if r.id == item.conversation_id),
    ) for item in scores]
    assert to_scored(records[2], scores[0]) == expected[0]
    assert to_scored(record("different-record-id"), scores[0]) == expected[0].model_copy(
        update={"turns": record("different-record-id").turns},
    )
    assert scored_conversations(records, scores) == expected
    assert scored_conversations(records, []) == []


def test_score_without_matching_record_raises_valueerror():
    """Catches silently dropping an orphan score or attaching it to an unrelated record."""
    with pytest.raises(ValueError):
        scored_conversations([record("a")], [score("a"), score("missing")])


def test_critic_selection_ranks_mean_then_id_and_caps_default_at_ten():
    """Catches taking the best, sorting only empathy, unstable ties, or ignoring the default cap."""
    assert CRITIC_CONVERSATIONS == 10
    values = [("top", 5, 5), ("z-tie", 1, 5), ("a-tie", 4, 2), ("worst", 1, 1), ("middle", 2, 3)]
    items = [to_scored(record(name), score(name, e, r)) for name, e, r in values]
    assert select_for_critic(items, 4) == [items[i] for i in (3, 4, 2, 1)]
    assert select_for_critic(items, 20) == [items[i] for i in (3, 4, 2, 1, 0)]
    assert select_for_critic(items, 1) == [items[3]]
    assert select_for_critic([], 2) == []
    many = [to_scored(record(f"c-{i:02d}"), score(f"c-{i:02d}", 3, 3)) for i in range(12, -1, -1)]
    assert [item.conversation_id for item in select_for_critic(many)] == [f"c-{i:02d}" for i in range(10)]


@pytest.mark.parametrize("n", [0, -1, -10])
def test_critic_selection_rejects_nonpositive_n_even_with_no_conversations(n):
    """Catches Python slicing behavior being used instead of the required validation."""
    for items in ([], [to_scored(record("a"), score("a"))]):
        with pytest.raises(ValueError):
            select_for_critic(items, n)


@pytest.mark.parametrize("number", [1, 9, 12, 123])
def test_score_path_and_jsonl_persistence_create_folders_and_preserve_full_records(tmp_path, number):
    """Catches the wrong round/split filename, omitted fields, reordered rows or missing folders."""
    path = train_scores_path(tmp_path, number)
    assert path == tmp_path / "scores" / f"round-{number:02d}-train.jsonl"
    scores = [score("b", 1, 4), score("a", 5, 2)]
    save_scores(scores, path)
    assert [ScoreRecord.model_validate_json(line) for line in path.read_text().splitlines()] == scores


def test_save_empty_scores_creates_empty_file(tmp_path):
    """Catches skipping the file or writing an empty JSON array instead of empty JSONL."""
    path = tmp_path / "new/nested/empty.jsonl"
    save_scores([], path)
    assert path.is_file() and path.read_bytes() == b""


@pytest.mark.parametrize("empty", [False, True])
def test_save_scores_replaces_existing_file_including_with_empty_list(tmp_path, empty):
    """Catches appending, retaining trailing old rows, or skipping an empty replacement."""
    path = tmp_path / "scores.jsonl"
    old = [score("old-one"), score("old-two")]
    path.write_text("".join(item.model_dump_json() + "\n" for item in old), encoding="utf-8")
    replacement = [] if empty else [score("replacement", 5, 1)]
    save_scores(replacement, path)
    assert [ScoreRecord.model_validate_json(line) for line in path.read_text().splitlines()] == replacement
    if empty:
        assert path.read_bytes() == b""


@pytest.mark.parametrize("kind", ["panel", "fallback", "unconfigured"])
def test_judge_models_discovers_panel_astra_fallback_and_unconfigured_judges(kind):
    """Catches missing Astra, missing the fallback model, or inspecting a nonexistent config."""
    config = {"panel": SimpleNamespace(judges=MODEL_NAMES[:3], astra=MODEL_NAMES[3]),
              "fallback": SimpleNamespace(model="fallback-model")}
    judge = SimpleNamespace(config=config[kind]) if kind in config else ScriptedJudge({})
    expected = {"panel": MODEL_NAMES, "fallback": ["fallback-model"], "unconfigured": []}[kind]
    found = judge_models(judge)
    assert isinstance(found, list) and Counter(found) == Counter(expected)
    if kind == "unconfigured":
        assert judge_models(None) == []


def test_metered_judge_preserves_version_result_and_current_agent(tmp_path):
    """Catches returning only a score, omitting gate usage, or hardcoding the judge agent."""
    item = record("gate-conversation")
    result = answer(item.id, entries=[usage(), usage(MODEL_NAMES[1], 0, 3)])
    raw = ScriptedJudge({item.id: result})
    wrapped = MeteredJudge(raw)
    assert wrapped.version == raw.version
    with metering(Meter(tmp_path, prices(), round=3), "gatekeeper"):
        assert wrapped.score(item) is result
    assert raw.calls == [item]
    assert reported_rows(usage_rows(tmp_path)) == Counter([
        ("gatekeeper", MODEL_NAMES[0], 101, 7, False), ("gatekeeper", MODEL_NAMES[1], 0, 3, False),
    ])


def test_metered_judge_propagates_original_exception_without_retry():
    """Catches swallowing, replacing or retrying an exception from the wrapped judge."""
    item, error = record("bad"), RuntimeError("judge exhausted its checks")
    raw = ScriptedJudge({item.id: error})
    with pytest.raises(RuntimeError) as caught:
        MeteredJudge(raw).score(item)
    assert caught.value is error
    assert raw.calls == [item]


def test_real_judge_panel_fits_adapter_and_all_its_reported_usage_reaches_meter(tmp_path):
    """Catches incompatibility with JudgeResult, taking a panel vote, or losing panel/Astra usage."""
    clients = [JudgeLLM(name) for name in MODEL_NAMES[:3]]
    astra = JudgeLLM(MODEL_NAMES[3], empathy=5, relevance=4)
    panel = JudgePanel(PanelConfig(judges=MODEL_NAMES[:3], astra=MODEL_NAMES[3]), clients, astra)
    item = record("real-panel-with-fakes")
    with metering(Meter(tmp_path, prices()), "judge"):
        run = score_conversations(panel, [item])
    assert run.unscored == [] and len(run.scores) == 1
    kept = run.scores[0]
    assert kept.conversation_id == item.id and kept.judge_version == panel.version
    assert (kept.empathy, kept.relevance, kept.note) == (5, 4, astra.output.note)
    assert to_scored(item, kept).turns == item.turns
    assert all(client.calls for client in [*clients, astra])
    expected = Counter(("judge", model, inp, out, False)
                       for client in [*clients, astra] for model, inp, out in client.calls)
    assert reported_rows(usage_rows(tmp_path)) == expected


# Section 2: full graph rounds, including the most harmful failure cases.


def test_rounddeps_defaults_allow_legacy_judge_without_panel(tmp_path):
    """Catches making a panel mandatory or changing the critic cap default."""
    deps = deps_for(tmp_path)
    assert deps.judge_panel is None
    assert deps.critic_conversations == CRITIC_CONVERSATIONS == 10


@pytest.mark.parametrize("mode", [
    "no_pattern", "critic_failed", "no_evidence", "researcher_failed", "proposer_failed", "kept", "reverted",
])
def test_round_excludes_failed_and_wrong_id_scores_caps_critic_and_finishes(
    tmp_path, monkeypatch, local_mlflow, round_policy, scenario_factory, mode,
):
    """Catches stub fallback, lost unscored IDs on any exit, bad critic inputs, or partial persistence."""
    names = ["high", "broken", "z-low", "wrong-id", "a-low", "middle"]
    scenarios = [scenario_factory(name, split="train") for name in names]
    rubric = {"high": (5, 5), "z-low": (3, 1), "a-low": (1, 3), "middle": (2, 3)}
    expected_ids = ["conv-r01-a-low", "conv-r01-z-low"]
    entries = [usage(), usage(MODEL_NAMES[1], 0, 3)]

    def outcome(item):
        if item.scenario_id == "broken":
            return ValueError("both judge answers failed")
        if item.scenario_id == "wrong-id":
            return answer("conv-r01-high", 1, 1)
        e, r = rubric[item.scenario_id]
        return answer(item.id, e, r, entries)

    panel = ScriptedJudge(outcome)
    llm = RoundLLM(mode, expected_ids)
    gate_calls, critic_inputs = [], []

    def gate(result):
        gate_calls.append(result)
        return round_api.stand_in_gate(result, mode)

    deps = deps_for(
        tmp_path, llm=llm, judge_panel=panel, critic_conversations=2, gate=gate,
        judge=lambda records: pytest.fail("legacy/stub judge called with a panel configured"),
    )
    original = round_api.run_critic

    def observe_critic(round_, conversations, *args, **kwargs):
        critic_inputs.append(list(conversations))
        return original(round_, conversations, *args, **kwargs)

    with patch_public(monkeypatch, round_api, "run_critic", observe_critic, graph):
        state = graph.run_round(deps, policy_path=round_policy, scenarios=scenarios, max_turns=1)
    records = state["records"]
    assert len(records) == 6 and panel.calls == records
    unscored = [item.id for item in records if item.scenario_id in {"broken", "wrong-id"}]
    assert state["unscored"] == state["result"].unscored == unscored
    assert [item.conversation_id for item in state["scored"]] == expected_ids
    assert critic_inputs == [state["scored"]]
    by_id = {item.id: item for item in records}
    for item in critic_inputs[0]:
        assert item.turns == by_id[item.conversation_id].turns
        assert (item.empathy, item.relevance) == rubric[by_id[item.conversation_id].scenario_id]
        assert item.note == f"Judge note for {item.conversation_id}."
    saved = [ScoreRecord.model_validate_json(line) for line in (
        tmp_path / "scores/round-01-train.jsonl").read_text().splitlines()]
    good = [item for item in records if item.scenario_id in rubric]
    assert saved == [score(item.id, *rubric[item.scenario_id]) for item in good]
    event = state["event"]
    assert deps.log.rounds() == [event]
    assert event.reason.endswith(f"The judge left 2 conversations unscored: {', '.join(unscored)}.")
    assert event.outcome == (mode if mode in {"kept", "reverted"} else
                             "no_evidence" if mode in {"no_evidence", "researcher_failed"} else "no_pattern")
    assert len(gate_calls) == int(mode in {"kept", "reverted"})
    judge_rows = [row for row in usage_rows(tmp_path) if row["model"] in MODEL_NAMES]
    assert reported_rows(judge_rows) == Counter({
        ("judge", MODEL_NAMES[0], 101, 7, False): 4,
        ("judge", MODEL_NAMES[1], 0, 3, False): 4,
    })
    experiment = mlflow.get_experiment_by_name("puddles")
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert len(runs) == 1
    assert runs[0].data.metrics["conversations_unscored"] == 2


@pytest.mark.parametrize("kind", ["no_records", "all_raise", "all_wrong_ids"])
def test_empty_scoring_finishes_without_critic_and_counts_every_failed_record(
    tmp_path, local_mlflow, round_policy, scenario_factory, kind,
):
    """Catches a crashed round, critic invocation with no scores, or uncounted all-failed batches."""
    scenarios = [] if kind == "no_records" else [scenario_factory("only", split="train")]
    panel = ScriptedJudge(lambda item: RuntimeError("judge failed twice") if kind == "all_raise"
                          else answer("some-other-conversation"))
    llm = RoundLLM()
    deps = deps_for(tmp_path, judge_panel=panel, llm=llm,
                    judge=lambda records: pytest.fail("stub fallback"),
                    gate=lambda result: pytest.fail("gate ran with no scores"))
    state = graph.run_round(deps, policy_path=round_policy, scenarios=scenarios, max_turns=1)
    assert panel.calls == state["records"]
    assert state["scored"] == [] and llm.calls == []
    expected = [item.id for item in state["records"]]
    assert state["unscored"] == state["result"].unscored == expected
    assert state["event"].outcome == "no_pattern"
    reason = state["event"].reason
    if kind == "no_records":
        assert reason == "No conversation finished."
    else:
        assert reason.startswith("The judge scored no conversation.")
        assert reason.endswith(f"The judge left 1 conversation unscored: {expected[0]}.")
    assert deps.log.rounds() == [state["event"]]
    assert (tmp_path / "scores/round-01-train.jsonl").read_bytes() == b""
    experiment = mlflow.get_experiment_by_name("puddles")
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert len(runs) == 1 and runs[0].data.metrics["conversations_unscored"] == len(expected)


def test_none_panel_calls_legacy_judge(tmp_path, local_mlflow, round_policy, scenario_factory):
    """Catches ignoring deps.judge when the new optional panel is absent."""
    calls = []

    def legacy(records):
        calls.append(list(records))
        return [ScoredConversation(conversation_id=item.id, empathy=4, relevance=1,
                                   note="Legacy custom judge.", turns=item.turns) for item in records]

    deps = deps_for(tmp_path, judge=legacy, judge_panel=None)
    state = graph.run_round(deps, policy_path=round_policy,
                            scenarios=[scenario_factory("legacy", split="train")], max_turns=1)
    assert len(calls) == 1 and calls[0] == state["records"]
    assert len(state["scored"]) == 1
    assert state["scored"][0].note == "Legacy custom judge."
    assert state["result"].unscored == []


@pytest.mark.parametrize("missing", MODEL_NAMES + ["fallback-model"])
def test_every_judge_model_is_price_checked_before_memory_log_or_any_model(
    tmp_path, local_mlflow, round_policy, scenario_factory, missing,
):
    """Catches checking only chat models, skipping Astra/fallback, or checking after round work."""
    panel = ScriptedJudge(lambda item: answer(item.id))
    names = ["fallback-model"] if missing == "fallback-model" else MODEL_NAMES
    panel.config = (SimpleNamespace(model="fallback-model") if missing == "fallback-model" else
                    SimpleNamespace(judges=MODEL_NAMES[:3], astra=MODEL_NAMES[3]))
    deps = deps_for(tmp_path, judge_panel=panel, usage=Meter(tmp_path, prices([n for n in names if n != missing])))
    memory_calls = []
    deps.memory.ensure_training_audience = lambda: memory_calls.append("ensure")
    with pytest.raises(MissingPrice) as caught:
        graph.run_round(deps, policy_path=round_policy,
                        scenarios=[scenario_factory("priced", split="train")], max_turns=1)
    assert missing in str(caught.value)
    assert memory_calls == [] and panel.calls == []
    assert deps.helper_client.calls == deps.user_client.calls == deps.llm.calls == []
    assert deps.log.events() == [] and read_usage(tmp_path / "log/usage.jsonl") == []
    assert not (tmp_path / "conversations").exists()


# Section 3: the durable round event names unscored conversations on both gate paths.


@pytest.mark.parametrize("ids", [[], ["c-one"], ["c-z", "c-a"]])
@pytest.mark.parametrize("decision", [None, "kept", "reverted"])
def test_finish_round_appends_exact_unscored_sentence_to_existing_reason(
    tmp_path, round_policy, ids, decision,
):
    """Catches missing/default-shared IDs, bad grammar/order, or using the result reason over the gate's."""
    result = round_api.RoundResult(round=1, outcome="no_pattern", base_policy_path=round_policy,
                                   reason="Original round reason.")
    other = round_api.RoundResult(round=2, outcome="no_pattern", base_policy_path=round_policy)
    assert result.unscored == other.unscored == []
    result.unscored.extend(ids)
    assert other.unscored == []
    gate = None
    if decision is not None:
        candidate = tmp_path / "candidate.md"
        candidate.write_text(round_policy.read_text().replace("# policy-v1", "# policy-v2-candidate"))
        result.candidate_path = candidate
        result.edit = PolicyEdit(
            id="edit-r01", written_by="proposer", created_at=STAMP, round=1, pattern_id="pattern-r01",
            research_id="research-r01", base_policy_version="policy-v1",
            new_policy_version="policy-v2-candidate", policy_path="candidate.md", summary="Acceptance edit.",
        )
        gate = round_api.stand_in_gate(result, decision)
        gate.reason = "Original gate reason."
        (tmp_path / "policies").mkdir()
    log, memory = DecisionLog(tmp_path), FakeMemory()
    memory.ensure_training_audience()
    event = round_api.finish_round(result, gate, log, memory, tmp_path)
    original = gate.reason if gate is not None else result.reason
    if ids:
        noun = "conversation" if len(ids) == 1 else "conversations"
        suffix = f"The judge left {len(ids)} {noun} unscored: {', '.join(ids)}."
        assert event.reason.startswith(original) and event.reason.endswith(suffix)
        assert event.reason.count(suffix) == 1
    else:
        assert event.reason == original
    assert log.rounds() == [event]


# Section 4: runner construction, fake persistence, and configuration failures.


@pytest.fixture
def script_args(tmp_path, round_policy, scenario_factory):
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(scenario_factory("script", split="train").model_dump_json() + "\n")
    return ["--root", str(tmp_path / "script-root"), "--policy", str(round_policy),
            "--scenarios", str(scenarios), "--max-turns", "1", "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]


def test_fake_script_uses_panel_adapter_writes_scores_and_never_constructs_real_judge(
    tmp_path, monkeypatch, local_mlflow, script_args,
):
    """Catches leaving --fake on the stub path, constructing a real judge, or failing to save scores."""
    captured = []
    original_run = graph.run_round

    def run(deps, **kwargs):
        captured.append(deps)
        deps.judge = lambda records: pytest.fail("--fake bypassed the panel adapter")
        return original_run(deps, **kwargs)

    def real_factory():
        pytest.fail("--fake constructed the real judge")

    with patch_public(monkeypatch, judge_api, "judge_from_env", real_factory, script):
        with patch_public(monkeypatch, graph, "run_round", run, script):
            assert script.main(["--fake", *script_args]) == 0
    assert len(captured) == 1 and captured[0].judge_panel.version == "judge-fake"
    root = tmp_path / "script-root"
    records = [ConversationRecord.model_validate_json(line) for line in (
        root / "conversations/round-01-train.jsonl").read_text().splitlines()]
    scores = [ScoreRecord.model_validate_json(line) for line in (
        root / "scores/round-01-train.jsonl").read_text().splitlines()]
    assert records and [s.conversation_id for s in scores] == [r.id for r in records]
    assert all(s.judge_version == "judge-fake" for s in scores)
    fake = captured[0].judge_panel
    different = record("unrelated-content")
    first, second = fake.score(records[0]), fake.score(different)
    assert first.usage == second.usage == []
    fields = ("empathy", "relevance", "emotion_before", "emotion_after")
    assert tuple(getattr(first.score, f) for f in fields) == tuple(getattr(second.score, f) for f in fields)


def test_fake_real_gate_saves_training_scores_without_inventing_judge_tokens(
    tmp_path, monkeypatch, local_mlflow, script_args,
):
    """Catches bypassing the adapter or estimating tokens for the usage-free scripted judge."""
    def real_factory():
        pytest.fail("--fake --gate real constructed the real judge")

    monkeypatch.setattr(judge_api, "judge_from_env", real_factory)
    root = tmp_path / "script-root"
    assert not root.exists()
    assert script.main(["--fake", "--gate", "real", *script_args]) == 0
    records = [ConversationRecord.model_validate_json(line) for line in (
        root / "conversations/round-01-train.jsonl").read_text().splitlines()]
    scores = [ScoreRecord.model_validate_json(line) for line in (
        root / "scores/round-01-train.jsonl").read_text().splitlines()]
    assert records and [item.conversation_id for item in scores] == [item.id for item in records]
    assert all(item.judge_version == "judge-fake" for item in scores)
    assert (root / "scores/round-01-gate.json").is_file()
    assert DecisionLog(root).rounds()[0].outcome == "reverted"
    usage_path = root / "log/usage.jsonl"
    assert usage_path.is_file()
    for row in read_usage(usage_path):
        if row.get("agent") in {"judge", "gatekeeper"}:
            assert row.get("tokens_in", 0) == row.get("tokens_out", 0) == 0


def test_fake_real_gate_wraps_the_rounds_same_judge_and_records_reported_tokens(
    tmp_path, monkeypatch, local_mlflow, script_args,
):
    """Reason: usage-free fake results cannot detect a separate or unmetered gate judge."""
    from puddles import gate as gate_api

    item = record("gate-wiring")
    result = answer(item.id, entries=[usage()])
    panel = ScriptedJudge({item.id: result})
    gates = []

    def make_gate(deps, **kwargs):
        gates.append(deps)
        return lambda result: pytest.fail("the mocked round must not run a gate")

    def run(deps, **kwargs):
        assert deps.judge_panel is panel
        assert len(gates) == 1 and isinstance(gates[0].judge, MeteredJudge)
        assert gates[0].judge.version == panel.version
        with metering(Meter(tmp_path / "gate-usage", prices()), "gatekeeper"):
            assert gates[0].judge.score(item) is result
        return {"event": SimpleNamespace(round=1, outcome="no_pattern", reason="Finished.")}

    monkeypatch.setattr(script, "FakeJudge", lambda: panel)
    with patch_public(monkeypatch, gate_api, "make_gate", make_gate, script), \
         patch_public(monkeypatch, graph, "run_round", run, script):
        assert script.main(["--fake", "--gate", "real", *script_args]) == 0
    assert panel.calls == [item]
    assert reported_rows(usage_rows(tmp_path / "gate-usage")) == Counter([
        ("gatekeeper", MODEL_NAMES[0], 101, 7, False),
    ])


def test_fake_judge_reads_only_the_conversation_id(monkeypatch, local_mlflow, script_args):
    """Reason: equal scores on two records do not prove the fake avoids reading their content."""
    class IdOnly:
        id = "content-must-not-be-read"

        def __getattr__(self, name):
            pytest.fail(f"fake judge read conversation field {name}")

    def run(deps, **kwargs):
        result = deps.judge_panel.score(IdOnly())
        assert result.score.conversation_id == IdOnly.id and result.usage == []
        return {"event": SimpleNamespace(round=1, outcome="no_pattern", reason="Finished.")}

    with patch_public(monkeypatch, graph, "run_round", run, script):
        assert script.main(["--fake", *script_args]) == 0


@pytest.mark.parametrize("missing_config", [False, True])
def test_real_script_constructs_judge_from_env_and_handles_its_valueerror_as_configuration(
    tmp_path, monkeypatch, local_mlflow, script_args, capsys, missing_config,
):
    """Catches ignoring judge_from_env, not passing its judge_panel, or starting after missing config."""
    from puddles.eval import safety as safety_api

    # A real gate now also needs a safety asker; this test never runs its gate.
    monkeypatch.setattr(safety_api.OpenAIYesNo, "from_env",
                        classmethod(lambda cls: SimpleNamespace(model="fake")))
    # Preflight also prices the replaced Exa boundary before the mocked round.
    config = tmp_path / "prices.toml"
    config.write_text("[output_limits]\ndefault = 80\n\n[prices]\nexa_search_usd = 0\n",
                      encoding="utf-8")
    monkeypatch.setenv("PUDDLES_CONFIG", str(config))
    calls, round_calls = [], []
    panel = ScriptedJudge({})

    def factory():
        calls.append("judge_from_env")
        if missing_config:
            raise ValueError("PUDDLES_JUDGE_PANEL is not set")
        return panel

    def run(deps, **kwargs):
        round_calls.append(deps)
        assert deps.judge_panel is panel
        return {"event": SimpleNamespace(round=1, outcome="no_pattern", reason="Finished.")}

    # This placeholder only satisfies the script's presence check; no API client is built.
    monkeypatch.setenv("EXA_API_KEY", "offline-test-placeholder")
    # The amended contract requires the factory to be looked up here when main runs.
    monkeypatch.setattr(judge_api, "judge_from_env", factory)
    with patch_public(monkeypatch, chat_api, "client_from_env", lambda prefix: FakeClient([]), script), \
         patch_public(monkeypatch, agent_api, "OpenAILLM", lambda: RoundLLM(), script), \
         patch_public(monkeypatch, memory_api, "PastClient", FakeMemory, script), \
         patch_public(monkeypatch, exa, "search", lambda query: [], script), \
         patch_public(monkeypatch, graph, "run_round", run, script):
        capsys.readouterr()
        code = script.main(script_args)
    captured = capsys.readouterr()
    assert calls == ["judge_from_env"]
    if missing_config:
        assert code == 2 and round_calls == []
        assert len(captured.err.splitlines()) == 1 and captured.err.strip()
        # Reason: an empty decision log alone misses usage/tracing writes on configuration failure.
        assert captured.out == "" and not (tmp_path / "script-root").exists()
        assert DecisionLog(tmp_path / "script-root").events() == []
    else:
        assert code == 0 and len(round_calls) == 1
