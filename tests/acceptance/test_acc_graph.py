"""Acceptance checks derived from docs/specs/graph.md, including its amendments."""
from contextlib import contextmanager
import json
import os
import socket

import mlflow
import pytest
from langgraph.graph import END, START
from langgraph.pregel import Pregel

from loop.errors import PolicyError
from loop.llm import FakeClient
from loop.policy import load_policy
from loop.runner import failures_path, scenario_seed
from puddles import graph, tracing
from puddles import round as round_api
from puddles.decision_log import DecisionLog
from puddles.llm import AgentFailed
from puddles.memory import FakeMemory
from puddles.records import load_record
from schemas import (
    ConversationRecord, CriticOutput, FailurePattern, GateOutcome, MemoryCopyEvent,
    PolicyEdit, ProposerOutput, ResearcherOutput, ResearchNote, RoundEvent,
    ScoredConversation,
)
from tests.fakes import FakeSearch, ScriptedLLM


URL = "https://evidence.example/listening"
QUERY = "listening before offering advice"
NODES = {"converse", "judge", "critic", "researcher", "proposer", "gate", "finish"}
NO_EVIDENCE = "No trustworthy source was found for this pattern."


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.endswith("API_KEY") or key in {"MLFLOW_TRACKING_URI", "MLFLOW_TRACING_DESTINATION"}:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("Acceptance tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def inputs(tmp_path, scenario_factory):
    policy = tmp_path / "starting.md"
    policy.write_text(
        "# policy-v1\n\n## Safety\nS1. Keep the person safe.\n\n"
        "## Conversation\nC1. Listen carefully.\n", encoding="utf-8",
    )
    scenario = scenario_factory("acceptance-01", split="train")
    scored = ScoredConversation(
        conversation_id="conv-r01-acceptance-01", empathy=3, relevance=3,
        note="Advice before understanding", turns=[
            {"speaker": "user", "text": "My work hours were cut."},
            {"speaker": "helper", "text": "Try finding another job."},
        ],
    )
    return policy, [scenario], [scored]


def model(mode="candidate", *, conversation_id="conv-r01-acceptance-01"):
    llm = ScriptedLLM()
    pattern = dict(
        found=True, name="early advice", description="Advice before understanding.",
        seen_in=1, out_of=1, example_conversation_ids=[conversation_id],
    )
    llm.add(CriticOutput, {"found": False, "reason": "No repeated failure."}
            if mode == "no_pattern" else pattern)
    llm.add(ResearcherOutput, {
        "queries": [QUERY], "sources": [] if mode == "no_evidence" else [
            {"url": URL, "title": "Listening", "technique": "Ask before advising."}],
    })
    llm.add(ProposerOutput, {
        "action": "add", "rule_text": "Ask what happened before offering advice.",
        "summary": "Understand the situation before advice.",
    })
    return llm


def dependencies(tmp_path, llm=None, **overrides):
    values = dict(
        helper_client=FakeClient(lambda system, messages: "That sounds difficult."),
        user_client=FakeClient(lambda system, messages: "My work hours were cut."),
        llm=llm if llm is not None else model(),
        search=FakeSearch({QUERY: [{"url": URL, "title": "Listening", "text": "Ask first."}]}),
        memory=FakeMemory(), log=DecisionLog(tmp_path), root=tmp_path,
    )
    values.update(overrides)
    return graph.RoundDeps(**values)


def critic(deps, policy, scored):
    return round_api.run_critic(1, scored, policy, deps.log, deps.memory, deps.llm, deps.root)


def researcher(deps, result):
    return round_api.run_researcher(
        result, deps.log, deps.memory, deps.llm, deps.search, deps.root)


def proposer(deps, result):
    return round_api.run_proposer(result, deps.log, deps.memory, deps.llm, deps.root)


@contextmanager
def replace_round_function(monkeypatch, name, replacement):
    with monkeypatch.context() as patch:
        original = getattr(round_api, name)
        patch.setattr(round_api, name, replacement)
        for alias, value in tuple(vars(graph).items()):
            if value is original:
                patch.setattr(graph, alias, replacement)
        yield


def traces():
    mlflow.flush_trace_async_logging()
    experiment = mlflow.get_experiment_by_name("puddles")
    return mlflow.search_traces(experiment_ids=[experiment.experiment_id], return_type="list")


def assert_trace_path(expected, deps):
    found = traces()
    assert len(found) == 1
    spans = found[0].data.spans
    assert {s.name for s in spans if s.name in NODES} == set(expected)
    by_id = {s.span_id: s for s in spans}
    for name, client in (("llm:helper", deps.helper_client),
                         ("llm:simulated_user", deps.user_client)):
        calls = [s for s in spans if s.name == name]
        assert len(calls) == len(client.calls) > 0
        unmatched = list(client.calls)
        for span in calls:
            assert span.span_type == "CHAT_MODEL"
            matched = next((call for call in unmatched
                            if all(span.inputs.get(key) == value for key, value in call.items())), None)
            assert matched is not None, f"{name} does not match the named client's calls"
            unmatched.remove(matched)
            ancestors = []
            parent = span.parent_id
            while parent is not None:
                assert parent in by_id
                ancestor = by_id[parent]
                ancestors.append(ancestor.name)
                parent = ancestor.parent_id
            assert "converse" in ancestors


def test_rounddeps_defaults(tmp_path):
    deps = dependencies(tmp_path)
    assert deps.judge is graph.stub_judge
    assert deps.gate is graph.stand_in_reverted


def test_stub_judge_preserves_records_and_marks_stub(inputs):
    _, _, scored = inputs
    from datetime import datetime, timezone

    records = [ConversationRecord(
        id=name, written_by="helper", created_at=datetime.now(timezone.utc),
        source="simulated", policy_version="policy-v1", turns=scored[0].turns,
    ) for name in ("second", "first")]
    got = graph.stub_judge(records)
    assert all(isinstance(item, ScoredConversation) for item in got)
    assert [item.conversation_id for item in got] == ["second", "first"]
    for record, item in zip(records, got, strict=True):
        assert item.turns == record.turns
        assert item.empathy == item.relevance == 3
        assert "stub" in item.note.lower()
    assert graph.stub_judge([]) == []


@pytest.mark.parametrize("kind", ["good", "carriage_return", "bare_carriage_return", "wrapped", "malformed"])
def test_check_policy_file(tmp_path, inputs, kind):
    policy, _, _ = inputs
    good = policy.read_bytes()
    if kind == "good":
        assert graph.check_policy_file(policy) == load_policy(policy)
        return
    content = {
        "carriage_return": good.replace(b"\n", b"\r\n"),
        "bare_carriage_return": good.replace(b"\n", b"\r", 1),
        "wrapped": good.replace(b"C1. Listen carefully.", b"C1. Listen\ncarefully."),
        "malformed": b"This is not a policy.\n",
    }[kind]
    policy.write_bytes(content)
    with pytest.raises(PolicyError):
        graph.check_policy_file(policy)


@pytest.mark.parametrize("mode", ["no_pattern", "no_evidence", "candidate"])
def test_split_functions_store_records_and_mutate_same_result(tmp_path, inputs, mode):
    policy, _, scored = inputs
    deps = dependencies(tmp_path, model(mode))
    result = critic(deps, policy, scored)
    assert isinstance(result, round_api.RoundResult)
    assert result.round == 1 and result.base_policy_path == policy
    assert result.outcome == "no_pattern"
    if mode == "no_pattern":
        assert result.pattern is None and result.reason == "No repeated failure."
        assert "pattern" not in result.refs
        return
    assert result.pattern.name == "early advice"
    assert result.status_before == "open"
    assert load_record(result.refs["pattern"], FailurePattern, tmp_path) == result.pattern
    assert researcher(deps, result) is result
    assert result.research.pattern_id == result.pattern.id
    assert load_record(result.refs["research"], ResearchNote, tmp_path) == result.research
    # The contract requires a saved search log, without prescribing its path or format.
    record_paths = {tmp_path / ref.path for ref in result.refs.values()}
    searches = [path for path in (tmp_path / "records").rglob("*")
                if path.is_file() and path not in record_paths
                and QUERY in path.read_text() and URL in path.read_text()]
    assert searches, "No saved search log contains the query and returned evidence"
    if mode == "no_evidence":
        assert result.research.sources == []
        assert result.outcome == "no_evidence" and result.reason == NO_EVIDENCE
        return
    assert proposer(deps, result) is result
    assert result.outcome == "candidate"
    assert result.candidate_path.is_file()
    assert result.candidate_path == tmp_path / result.edit.policy_path
    assert load_record(result.refs["edit"], PolicyEdit, tmp_path) == result.edit
    assert result.edit.rule_text in result.candidate_path.read_text()


@pytest.mark.parametrize("mode,expected", [
    ("no_pattern", ["run_critic"]),
    ("no_evidence", ["run_critic", "run_researcher"]),
    ("candidate", ["run_critic", "run_researcher", "run_proposer"]),
])
def test_run_agents_delegates_in_order_and_stops_early(tmp_path, inputs, monkeypatch, mode, expected):
    policy, _, scored = inputs
    deps = dependencies(tmp_path, model(mode))
    calls, returned = [], []
    for name in ("run_critic", "run_researcher", "run_proposer"):
        original = getattr(round_api, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            result = _original(*args, **kwargs)
            returned.append(result)
            return result

        monkeypatch.setattr(round_api, name, spy)
    result = round_api.run_agents(
        1, scored, policy, deps.log, deps.memory, deps.llm, deps.search, tmp_path)
    assert calls == expected
    assert all(item is result for item in returned)
    assert result.outcome == mode


def bad_model(agent):
    llm = model()
    schema, bad = {
        "critic": (CriticOutput, {"found": False, "reason": ""}),
        "researcher": (ResearcherOutput, {"queries": [], "sources": []}),
        "proposer": (ProposerOutput, {"action": "remove", "rule_id": "S1", "summary": "Invalid."}),
    }[agent]
    # Public queues belong to the supplied test fake, not to production code.
    llm.queue[schema] = [bad, bad]
    return llm, schema


@pytest.mark.parametrize("agent", ["critic", "researcher", "proposer"])
def test_split_functions_propagate_agentfailed(tmp_path, inputs, agent):
    policy, _, scored = inputs
    llm, schema = bad_model(agent)
    deps = dependencies(tmp_path, llm)
    with pytest.raises(AgentFailed):
        result = critic(deps, policy, scored)
        assert agent != "critic", "critic swallowed AgentFailed"
        researcher(deps, result)
        assert agent != "researcher", "researcher swallowed AgentFailed"
        proposer(deps, result)
    assert len(llm.prompts[schema]) == 2


def test_stand_in_reverted_matches_existing_gate(tmp_path, inputs):
    policy, _, scored = inputs
    deps = dependencies(tmp_path)
    result = proposer(deps, researcher(deps, critic(deps, policy, scored)))
    actual = graph.stand_in_reverted(result)
    expected = round_api.stand_in_gate(result, "reverted")
    # Each call timestamps a new record; the decision and all other fields match.
    assert actual.model_dump(exclude={"created_at"}) == expected.model_dump(exclude={"created_at"})


def test_run_critic_sets_existing_pattern_status_before(tmp_path, inputs):
    policy, _, scored = inputs
    deps = dependencies(tmp_path)
    deps.memory.ensure_training_audience()
    first = proposer(deps, researcher(deps, critic(deps, policy, scored)))
    round_api.finish_round(first, round_api.stand_in_gate(first, "kept"),
                           deps.log, deps.memory, tmp_path)
    llm = ScriptedLLM()
    llm.add(CriticOutput, dict(
        found=True, name="early advice", description="Advice is still premature.",
        seen_in=1, out_of=1, example_conversation_ids=[scored[0].conversation_id],
        same_as_pattern_id=first.pattern.id,
    ))
    status_before = next(p.status for p in deps.log.earlier_patterns() if p.id == first.pattern.id)
    second = round_api.run_critic(2, scored, policy, deps.log, deps.memory, llm, tmp_path)
    assert second is not first and second.outcome == "no_pattern"
    assert second.pattern is not None and second.status_before == status_before
    assert load_record(second.refs["pattern"], FailurePattern, tmp_path) == second.pattern


def test_graph_has_exactly_the_seven_named_nodes(tmp_path):
    compiled = graph.build_round_graph(dependencies(tmp_path))
    drawn = compiled.get_graph()
    assert set(drawn.nodes) - {START, END} == NODES
    assert {(edge.source, edge.target) for edge in drawn.edges} == {
        (START, "converse"), ("converse", "judge"),
        ("judge", "critic"), ("judge", "finish"),
        ("critic", "researcher"), ("critic", "finish"),
        ("researcher", "proposer"), ("researcher", "finish"),
        ("proposer", "gate"), ("proposer", "finish"),
        ("gate", "finish"), ("finish", END),
    }


@pytest.mark.parametrize("mode,decision,expected_nodes,outcome", [
    ("no_pattern", "reverted", ["converse", "judge", "critic", "finish"], "no_pattern"),
    ("no_evidence", "reverted", ["converse", "judge", "critic", "researcher", "finish"], "no_evidence"),
    ("candidate", "kept", NODES, "kept"),
    ("candidate", "reverted", NODES, "reverted"),
])
def test_graph_routes_and_traces_each_normal_path(tmp_path, inputs, mode, decision, expected_nodes, outcome):
    policy, scenarios, _ = inputs
    gates = []

    def gate(result):
        assert result.edit is not None
        gates.append(result)
        return round_api.stand_in_gate(result, decision)

    deps = dependencies(tmp_path, model(mode), gate=gate)
    state = graph.run_round(deps, policy_path=policy, scenarios=scenarios, max_turns=1, seed=42)
    assert state["event"].outcome == outcome
    assert state["result"].outcome == mode
    assert bool(gates) == (mode == "candidate")
    if mode == "candidate":
        assert state["gate"].decision == decision
    else:
        assert state.get("gate") is None
    assert deps.log.rounds() == [state["event"]]
    assert state["records"][0].id == "conv-r01-acceptance-01"
    assert state["failures"] == []
    assert state["scored"] == graph.stub_judge(state["records"])
    assert deps.helper_client.calls[0]["seed"] == scenario_seed(42, scenarios[0].scenario_id)
    assert len(state["records"][0].turns) == 2
    assert_trace_path(expected_nodes, deps)


@pytest.mark.parametrize("kind", ["no_scenarios", "all_failed", "nothing_scored"])
def test_graph_finishes_when_nothing_is_scored(tmp_path, inputs, kind):
    policy, scenarios, _ = inputs
    judge_calls, gate_calls = [], []

    def judge(records):
        judge_calls.append(records)
        return []

    deps = dependencies(tmp_path, judge=judge, gate=lambda result: gate_calls.append(result))
    if kind == "no_scenarios":
        scenarios = []
    elif kind == "all_failed":
        deps.user_client = FakeClient(["[END]"])
    state = graph.run_round(deps, policy_path=policy, scenarios=scenarios, max_turns=1)
    assert judge_calls == [state["records"]]
    assert state["scored"] == []
    assert state["result"].outcome == state["event"].outcome == "no_pattern"
    assert state["result"].pattern is None
    assert not gate_calls and not deps.llm.prompts
    if kind != "nothing_scored":
        assert state["records"] == []
        assert state["event"].reason == "No conversation finished."
    else:
        assert len(state["records"]) == 1
        assert state["event"].reason == "The judge scored no conversation."
    assert len(state["failures"]) == (1 if kind == "all_failed" else 0)
    found = traces()
    assert len(found) == 1
    assert {s.name for s in found[0].data.spans if s.name in NODES} == {"converse", "judge", "finish"}


@pytest.mark.parametrize("agent", ["critic", "researcher", "proposer"])
def test_graph_agent_fails_twice_but_finishes_without_gate(tmp_path, inputs, agent):
    policy, scenarios, scored = inputs
    # Obtain the complete exception text independently through the documented split API.
    expected_llm, _ = bad_model(agent)
    expected_deps = dependencies(tmp_path / "expected", expected_llm)
    with pytest.raises(AgentFailed) as failure:
        result = critic(expected_deps, policy, scored)
        researcher(expected_deps, result)
        proposer(expected_deps, result)
    llm, schema = bad_model(agent)
    deps = dependencies(tmp_path / "actual", llm,
                        gate=lambda result: pytest.fail("gate ran after AgentFailed"))
    state = graph.run_round(deps, policy_path=policy, scenarios=scenarios, max_turns=1)
    result, event = state["result"], state["event"]
    assert len(llm.prompts[schema]) == 2
    assert event.reason.startswith(agent)
    assert str(failure.value) in event.reason
    assert event.outcome == ("no_evidence" if agent == "researcher" else "no_pattern")
    assert result.outcome == ("no_evidence" if agent == "researcher" else "no_pattern")
    assert (result.pattern is not None) == (agent != "critic")
    assert (event.pattern is not None) == (agent != "critic")
    assert (result.research is not None) == (agent == "proposer")
    if agent != "critic":
        assert result.pattern.name == "early advice"
        assert result.pattern.example_conversation_ids == [scored[0].conversation_id]
        assert event.pattern.id == result.pattern.id
        assert load_record(result.refs["pattern"], FailurePattern, deps.root) == result.pattern
    if agent == "proposer":
        assert result.research.queries == [QUERY]
        assert [source.url for source in result.research.sources] == [URL]
        assert load_record(result.refs["research"], ResearchNote, deps.root) == result.research
    assert result.edit is None and state.get("gate") is None
    assert deps.log.rounds() == [event]
    assert event.records == result.refs
    expected_records = set() if agent == "critic" else {"pattern"}
    if agent == "proposer":
        expected_records.add("research")
    assert set(event.records) == expected_records
    copied_ids = {ref.id for ref in event.records.values()}
    copies = [item for item in deps.log.events() if isinstance(item, MemoryCopyEvent)]
    assert {item.record_id for item in copies} == copied_ids
    assert all(item.status == "confirmed" for item in copies)
    assert set(deps.memory.entries) == copied_ids
    expected_nodes = {"converse", "judge", "critic", "finish"}
    if agent != "critic":
        expected_nodes.add("researcher")
    if agent == "proposer":
        expected_nodes.add("proposer")
    assert_trace_path(expected_nodes, deps)


@pytest.mark.parametrize("where", ["converse", "critic", "researcher", "proposer", "judge", "gate", "finish"])
def test_graph_does_not_catch_other_exceptions(tmp_path, inputs, monkeypatch, where):
    policy, scenarios, _ = inputs
    sentinel = RuntimeError(f"unexpected {where} error")

    def fail(*args, **kwargs):
        raise sentinel

    if where in {"critic", "researcher", "proposer", "finish"}:
        function = "finish_round" if where == "finish" else f"run_{where}"
        context = replace_round_function(monkeypatch, function, fail)
    else:
        from contextlib import nullcontext
        context = nullcontext()
    with context:
        overrides = {where: fail} if where in {"judge", "gate"} else {}
        if where == "converse":
            overrides["helper_client"] = FakeClient(fail)
        deps = dependencies(tmp_path, **overrides)
        with pytest.raises(RuntimeError) as error:
            graph.run_round(deps, policy_path=policy, scenarios=scenarios, max_turns=1)
        assert error.value is sentinel
        assert deps.log.rounds() == []


def test_run_round_order_run_context_defaults_and_no_setup(tmp_path, inputs, monkeypatch):
    policy, scenarios, _ = inputs
    deps = dependencies(tmp_path, model("no_pattern"))
    order = []
    original_ensure = deps.memory.ensure_training_audience
    original_next = deps.log.next_round
    original_retry = round_api.retry_pending
    original_invoke = Pregel.invoke

    def ensure():
        order.append("ensure")
        original_ensure()

    def retry(log, memory, root):
        assert (log, memory, root) == (deps.log, deps.memory, deps.root)
        order.append("retry")
        return original_retry(log, memory, root)

    def next_round():
        order.append("next")
        return original_next()

    def invoke(compiled, state, *args, **kwargs):
        order.append("invoke")
        assert state["round"] == 1 and state["policy_path"] == policy
        assert state["scenarios"] == scenarios
        assert state["max_turns"] == 10 and state["seed"] is None
        run = mlflow.active_run()
        assert run is not None and run.info.run_name == "round-01"
        return original_invoke(compiled, state, *args, **kwargs)

    monkeypatch.setattr(deps.memory, "ensure_training_audience", ensure)
    monkeypatch.setattr(deps.log, "next_round", next_round)
    monkeypatch.setattr(Pregel, "invoke", invoke)
    monkeypatch.setattr(tracing, "setup", lambda *a, **k: pytest.fail("run_round called setup"))
    with replace_round_function(monkeypatch, "retry_pending", retry):
        state = graph.run_round(deps, policy_path=policy, scenarios=scenarios)
    assert order == ["ensure", "retry", "next", "invoke"]
    assert state["event"].outcome == "no_pattern"


@pytest.mark.parametrize("bad", ["malformed", "carriage_return"])
def test_bad_policy_is_checked_before_memory_and_round_allocation(tmp_path, inputs, monkeypatch, bad):
    policy, scenarios, _ = inputs
    if bad == "carriage_return":
        policy.write_bytes(policy.read_bytes().replace(b"\n", b"\r\n"))
    else:
        policy.write_text("not a policy\n")
    deps = dependencies(tmp_path)
    order = []
    monkeypatch.setattr(deps.memory, "ensure_training_audience", lambda: order.append("ensure"))
    monkeypatch.setattr(deps.log, "next_round", lambda: pytest.fail("allocated round for invalid policy"))
    with replace_round_function(monkeypatch, "retry_pending", lambda *args: order.append("retry")):
        with pytest.raises(PolicyError):
            graph.run_round(deps, policy_path=policy, scenarios=scenarios)
    assert order == []
    assert deps.log.events() == []
    assert not deps.helper_client.calls and not deps.user_client.calls
    experiment = mlflow.get_experiment_by_name("puddles")
    assert mlflow.search_runs([experiment.experiment_id], output_format="list") == []


@pytest.mark.parametrize("bad", ["malformed", "carriage_return"])
def test_bad_policy_leaves_pending_log_bytes_and_memory_untouched(tmp_path, inputs, monkeypatch, bad):
    policy, scenarios, scored = inputs
    deps = dependencies(tmp_path)
    deps.memory.ensure_training_audience()
    result = critic(deps, policy, scored)
    # A genuine pending copy, backed by a record that retry_pending could ingest.
    monkeypatch.setattr(deps.memory, "wait", lambda *a, **k: False)
    round_api.finish_round(result, None, deps.log, deps.memory, tmp_path)
    assert deps.log.pending_copies()
    before = deps.log.path.read_bytes()
    memory = FakeMemory()
    deps.memory = memory
    if bad == "carriage_return":
        policy.write_bytes(policy.read_bytes().replace(b"\n", b"\r\n"))
    else:
        policy.write_text("not a policy\n")
    with pytest.raises(PolicyError):
        graph.run_round(deps, policy_path=policy, scenarios=scenarios)
    assert deps.log.path.read_bytes() == before
    assert memory.entries == {} and memory.audiences == {} and memory.recalls == []
    assert not deps.helper_client.calls and not deps.user_client.calls


def test_compiled_graph_rejects_carriage_return_before_conversation(tmp_path, inputs):
    policy, scenarios, _ = inputs
    policy.write_bytes(policy.read_bytes().replace(b"\n", b"\r\n"))
    deps = dependencies(tmp_path)
    compiled = graph.build_round_graph(deps)
    with pytest.raises(PolicyError):
        compiled.invoke(dict(round=1, policy_path=policy, scenarios=scenarios, max_turns=1, seed=None))
    assert not deps.helper_client.calls and not deps.user_client.calls
    assert deps.log.events() == [] and deps.memory.entries == {}
    assert not (tmp_path / "conversations").exists()


def test_two_rounds_number_files_memory_mlflow_parameters_tag_and_metrics(tmp_path, inputs):
    policy, scenarios, _ = inputs
    # A failed second conversation makes both metrics nontrivial.
    failed = scenarios[0].model_copy(update={"scenario_id": "fails"})

    # Use a queue instead of relying on how scenario IDs appear in prompts.
    deps = dependencies(tmp_path, user_client=FakeClient(["My work hours were cut.", "[END]"]))
    first = graph.run_round(deps, policy_path=policy, scenarios=scenarios + [failed], max_turns=1, seed=17)
    deps.llm = model("no_pattern", conversation_id="conv-r02-acceptance-01")
    deps.user_client = FakeClient(["My work hours were cut."])
    second = graph.run_round(deps, policy_path=policy, scenarios=scenarios, max_turns=1)
    assert [first["event"].round, second["event"].round] == [1, 2]
    assert [event.round for event in deps.log.rounds()] == [1, 2]
    assert first["event"].outcome == "reverted" and first["result"].outcome == "candidate"
    assert second["event"].outcome == "no_pattern"
    for number, state in ((1, first), (2, second)):
        path = tmp_path / f"conversations/round-{number:02d}-train.jsonl"
        records = [ConversationRecord.model_validate_json(line) for line in path.read_text().splitlines()]
        assert records == state["records"]
        assert records[0].id.startswith(f"conv-r{number:02d}-")
    failure_rows = [json.loads(line) for line in failures_path(
        tmp_path / "conversations/round-01-train.jsonl").read_text().splitlines()]
    assert failure_rows == [vars(item) for item in first["failures"]]
    assert not failures_path(tmp_path / "conversations/round-02-train.jsonl").exists()
    models = {"pattern": FailurePattern, "research": ResearchNote, "edit": PolicyEdit, "outcome": GateOutcome}
    assert set(first["event"].records) == set(models)
    for kind, ref in first["event"].records.items():
        assert (tmp_path / ref.path).is_relative_to(tmp_path / "records")
        assert load_record(ref, models[kind], tmp_path) is not None
    copies = [event for event in deps.log.events() if isinstance(event, MemoryCopyEvent)]
    assert isinstance(deps.log.events()[0], RoundEvent)
    assert {event.record_id for event in copies} == {ref.id for ref in first["event"].records.values()}
    assert all(event.status == "confirmed" for event in copies)
    assert set(deps.memory.entries) == {event.record_id for event in copies}
    assert first["result"].candidate_path.is_file()
    experiment = mlflow.get_experiment_by_name("puddles")
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert len(runs) == 2
    by_name = {run.info.run_name: run for run in runs}
    for number, state, failed_count in ((1, first, 1), (2, second, 0)):
        run = by_name[f"round-{number:02d}"]
        assert run.data.params["round"] == str(number)
        assert run.data.params["policy_version"] == "policy-v1"
        assert run.data.tags["outcome"] == state["event"].outcome
        assert run.data.metrics["conversations"] == 1
        assert run.data.metrics["conversations_failed"] == failed_count
    assert len(traces()) == 2
