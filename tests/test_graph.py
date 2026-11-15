"""Stage 0: one round as a LangGraph graph, with MLflow tracing. All offline."""
import json
import shutil
from pathlib import Path

import mlflow
import pytest

from loop.errors import LLMError, PolicyError
from loop.llm import FakeClient
from loop.policy import load_policy
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.decision_log import DecisionLog
from puddles.graph import RoundDeps, run_round
from puddles.memory import FakeMemory
from puddles.policy import parse_policy
from puddles.round import stand_in_gate
from schemas import CriticOutput, ProposerOutput, ResearcherOutput
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")
URL = "https://counselling.example.ac.uk/skills"
QUERY = "counselling skills exploring before giving advice"
NODES = {"converse", "judge", "critic", "researcher", "proposer", "gate", "finish"}


def pattern(out_of=len(SCENARIOS), example="conv-r01-fixture-0001"):
    return {"found": True, "name": "advice-before-understanding",
            "description": "Gives advice in its first reply before asking about the situation.",
            "seen_in": 1, "out_of": out_of, "example_conversation_ids": [example]}


NOT_FOUND = {"found": False, "reason": "The conversations went well."}
RESEARCH = {"queries": [QUERY], "sources": [{"url": URL, "title": "Skills", "technique": "Explore first."}]}
NO_SOURCES = {"queries": [QUERY], "sources": []}
EDIT = {"action": "add", "rule_text": "Ask one open question before any suggestion.", "summary": "Ask first."}


class Gate:
    """The stand-in gate, counting its calls."""

    def __init__(self, decision="reverted"):
        self.decision, self.calls = decision, 0

    def __call__(self, result):
        self.calls += 1
        return stand_in_gate(result, self.decision)


@pytest.fixture
def traced(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


def make_deps(root, llm, *, gate=None, user=None, helper=None):
    search = FakeSearch({QUERY: [{"url": URL, "title": "Skills", "text": "..."}]})
    return RoundDeps(helper_client=helper or FakeClient(_fake_helper), user_client=user or FakeClient(_fake_user),
                     llm=llm, search=search, memory=FakeMemory(), log=DecisionLog(root), root=root,
                     gate=gate or Gate())


def round_events(root):
    return [e for e in DecisionLog(root).events() if e.event == "round"]


# 1
@pytest.mark.parametrize("decision", ["kept", "reverted"])
def test_round_that_reaches_the_proposer_ends_as_the_gate_decides(tmp_path, traced, decision):
    llm = ScriptedLLM()
    llm.add(CriticOutput, pattern())
    llm.add(ResearcherOutput, RESEARCH)
    llm.add(ProposerOutput, EDIT)
    gate = Gate(decision)
    state = run_round(make_deps(tmp_path, llm, gate=gate), policy_path=POLICY, scenarios=SCENARIOS)

    assert state["result"].outcome == "candidate"
    assert state["gate"].decision == decision and gate.calls == 1
    assert state["event"].outcome == decision
    events = round_events(tmp_path)
    assert len(events) == 1 and events[0].outcome == decision and events[0].round == 1


# 2
def test_no_pattern_skips_researcher_and_proposer(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND)
    deps = make_deps(tmp_path, llm)
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)

    assert state["event"].outcome == "no_pattern"
    assert state["event"].reason == NOT_FOUND["reason"]
    assert not llm.prompts[ResearcherOutput] and not llm.prompts[ProposerOutput]
    assert deps.search.calls == [] and deps.gate.calls == 0
    assert state["gate"] is None


# 3
def test_no_sources_skips_the_proposer(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, pattern())
    llm.add(ResearcherOutput, NO_SOURCES)
    deps = make_deps(tmp_path, llm)
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)

    assert state["result"].outcome == "no_evidence"
    assert state["event"].outcome == "no_evidence"
    assert not llm.prompts[ProposerOutput] and deps.gate.calls == 0


# 4
@pytest.mark.parametrize("answers, ran", [
    ([(CriticOutput, NOT_FOUND)], {"converse", "judge", "critic", "finish"}),
    ([(CriticOutput, pattern()), (ResearcherOutput, NO_SOURCES)],
     {"converse", "judge", "critic", "researcher", "finish"}),
    ([(CriticOutput, pattern()), (ResearcherOutput, RESEARCH), (ProposerOutput, EDIT)], NODES),
], ids=["no_pattern", "no_evidence", "gate"])
def test_each_path_leaves_one_trace_with_a_span_per_node(tmp_path, traced, answers, ran):
    llm = ScriptedLLM()
    for schema, out in answers:
        llm.add(schema, out)
    run_round(make_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)
    mlflow.flush_trace_async_logging()

    traces = mlflow.search_traces(return_type="list")
    assert len(traces) == 1
    spans = traces[0].data.spans
    names = {s.name for s in spans}
    assert ran <= names and not (NODES - ran) & names

    by_id = {s.span_id: s for s in spans}

    def ancestors(span):
        while span.parent_id is not None and span.parent_id in by_id:
            span = by_id[span.parent_id]
            yield span

    llm_spans = [s for s in spans if s.name in ("llm:helper", "llm:simulated_user")]
    assert {s.name for s in llm_spans} == {"llm:helper", "llm:simulated_user"}
    for s in llm_spans:
        assert s.span_type == "CHAT_MODEL"
        assert "converse" in {a.name for a in ancestors(s)}


# 5
def test_two_rounds_are_numbered_one_and_two(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND, NOT_FOUND)
    first = run_round(make_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)
    second = run_round(make_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)

    assert (first["round"], second["round"]) == (1, 2)
    assert [e.round for e in round_events(tmp_path)] == [1, 2]
    assert (tmp_path / "conversations" / "round-02-train.jsonl").exists()


# 6
def test_a_failed_conversation_still_finishes_the_round(tmp_path, traced):
    def user(system, messages):
        if "two mugs" in system:  # fixture-0002
            raise LLMError("simulated user server down")
        return _fake_user(system, messages)

    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND)
    state = run_round(make_deps(tmp_path, llm, user=FakeClient(user)), policy_path=POLICY, scenarios=SCENARIOS)

    assert state["event"].outcome == "no_pattern"
    assert [f.scenario_id for f in state["failures"]] == ["fixture-0002"]
    assert len(state["records"]) == len(SCENARIOS) - 1
    failures_file = tmp_path / "conversations" / "round-01-train.failures.jsonl"
    assert [json.loads(line)["scenario_id"] for line in failures_file.read_text().splitlines()] == ["fixture-0002"]

    [run] = mlflow.search_runs(output_format="list")
    assert run.info.run_name == "round-01"
    assert run.data.metrics["conversations_failed"] == 1
    assert run.data.metrics["conversations"] == len(SCENARIOS) - 1
    assert run.data.params == {"round": "1", "policy_version": "policy-v1"}
    assert run.data.tags["outcome"] == "no_pattern"


# 7
def test_the_conversations_file_holds_one_line_per_finished_conversation(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND)
    state = run_round(make_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)

    lines = (tmp_path / "conversations" / "round-01-train.jsonl").read_text().splitlines()
    assert len(lines) == len(state["records"]) == len(SCENARIOS)
    assert [json.loads(line)["id"] for line in lines] == [r.id for r in state["records"]]
    assert not (tmp_path / "conversations" / "round-01-train.failures.jsonl").exists()


# 8
BAD_CRITIC = {"found": True, "seen_in": 1, "out_of": 1, "example_conversation_ids": []}
BAD_RESEARCH = {"queries": [QUERY], "sources": [{"url": "https://invented.example", "title": "x", "technique": "y"}]}
BAD_EDIT = {"action": "remove", "rule_id": "S1", "summary": "Drop a safety rule."}


@pytest.mark.parametrize("agent", ["critic", "researcher", "proposer"])
def test_an_agent_failing_twice_still_finishes_the_round(tmp_path, traced, agent):
    llm = ScriptedLLM()
    if agent == "critic":
        llm.add(CriticOutput, BAD_CRITIC, BAD_CRITIC)
    else:
        llm.add(CriticOutput, pattern())
        if agent == "researcher":
            llm.add(ResearcherOutput, BAD_RESEARCH, BAD_RESEARCH)
        else:
            llm.add(ResearcherOutput, RESEARCH)
            llm.add(ProposerOutput, BAD_EDIT, BAD_EDIT)
    deps = make_deps(tmp_path, llm)
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    result, event = state["result"], state["event"]

    assert event.reason.startswith(agent) and "failed its checks twice" in event.reason
    assert deps.gate.calls == 0 and state["gate"] is None and result.edit is None
    assert len(round_events(tmp_path)) == 1
    if agent == "critic":
        assert result.pattern is None and result.outcome == "no_pattern" and event.outcome == "no_pattern"
        assert not llm.prompts[ResearcherOutput]
    elif agent == "researcher":
        assert result.outcome == "no_evidence" and event.outcome == "no_evidence"
        assert result.research is None and result.pattern is not None
        assert not llm.prompts[ProposerOutput]
    else:
        assert result.outcome == "no_pattern" and event.outcome == "no_pattern"
        assert result.pattern is not None and result.research is not None
        assert event.pattern is not None and event.pattern.id == result.pattern.id


# 9
@pytest.mark.parametrize("name", ["policy-v1.md", "baseline-plain.md"])
def test_both_policy_loaders_agree_on_the_shipped_policies(name):
    path = REPO / "policies" / name
    loop_policy, puddles_policy = load_policy(path), parse_policy(path.read_text())
    assert loop_policy.version == puddles_policy.version
    assert loop_policy.rules == {**puddles_policy.safety, **puddles_policy.conversation}


def test_the_candidate_policy_loads_and_keeps_the_safety_text(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, pattern())
    llm.add(ResearcherOutput, RESEARCH)
    llm.add(ProposerOutput, EDIT)
    state = run_round(make_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS)

    candidate = load_policy(state["result"].candidate_path)
    assert candidate.version == "policy-v2-candidate"
    assert candidate.safety_text.encode() == load_policy(POLICY).safety_text.encode()
    assert EDIT["rule_text"] in candidate.rules.values()


# 10
def wrapped(path):
    text = POLICY.read_text().replace("C4. Keep replies short.", "C4. Keep replies\nshort.")
    path.write_text(text)
    return path


def crlf(path):
    path.write_bytes(POLICY.read_bytes().replace(b"\n", b"\r\n"))
    return path


@pytest.mark.parametrize("make_bad", [wrapped, crlf], ids=["wrapped-rule", "windows-line-endings"])
def test_a_malformed_starting_policy_is_refused_before_the_round(tmp_path, traced, make_bad):
    root = tmp_path / "root"
    bad = make_bad(tmp_path / "policy-v1.md")
    helper, user = FakeClient(_fake_helper), FakeClient(_fake_user)
    deps = make_deps(root, ScriptedLLM(), helper=helper, user=user)

    with pytest.raises(PolicyError):
        run_round(deps, policy_path=bad, scenarios=SCENARIOS)
    assert not (root / "log" / "decisions.jsonl").exists()
    assert not (root / "conversations").exists()
    assert helper.calls == [] and user.calls == []
    assert mlflow.search_runs(output_format="list") == []


class UnconfirmedMemory(FakeMemory):
    """Takes every copy but never confirms one, so the log ends with pending copies."""

    def wait(self, ingestion_id, timeout_s=60):
        return False


def leave_pending_copies(root):
    llm = ScriptedLLM()
    llm.add(CriticOutput, pattern())
    llm.add(ResearcherOutput, NO_SOURCES)
    deps = make_deps(root, llm)
    deps.memory = UnconfirmedMemory()
    run_round(deps, policy_path=POLICY, scenarios=SCENARIOS)
    log = DecisionLog(root)
    assert {e.record_id for e in log.pending_copies()} == {"pattern-r01", "research-r01"}
    return log.path.read_bytes()


@pytest.mark.parametrize("make_bad", [wrapped, crlf], ids=["wrapped-rule", "windows-line-endings"])
def test_a_bad_policy_leaves_a_log_with_pending_copies_unchanged(tmp_path, traced, make_bad):
    root = tmp_path / "root"
    before = leave_pending_copies(root)
    memory = FakeMemory()
    deps = make_deps(root, ScriptedLLM())
    deps.memory = memory

    with pytest.raises(PolicyError):
        run_round(deps, policy_path=make_bad(tmp_path / "policy-v1.md"), scenarios=SCENARIOS)
    assert DecisionLog(root).path.read_bytes() == before
    assert memory.entries == {} and memory.audiences == {}


@pytest.mark.parametrize("bad", ["policy", "scenarios"])
def test_the_script_refuses_a_bad_file_before_writing_to_the_log(tmp_path, traced, capsys, bad):
    from scripts.run_graph_round import main

    root = tmp_path / "root"
    before = leave_pending_copies(root)
    bad_policy, bad_scenarios = wrapped(tmp_path / "policy-v1.md"), tmp_path / "scenarios.jsonl"
    bad_scenarios.write_text('{"scenario_id": "broken"}\n')
    argv = ["--fake", "--root", str(root), "--mlflow", f"sqlite:///{tmp_path}/script.db",
            "--policy", str(bad_policy if bad == "policy" else POLICY),
            "--scenarios", str(bad_scenarios if bad == "scenarios" else REPO / "fixtures" / "scenarios.jsonl")]

    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1
    assert DecisionLog(root).path.read_bytes() == before


def test_the_shipped_policy_copied_to_another_folder_is_accepted(tmp_path, traced):
    # Guards test 10: the refusal comes from the defect, not from the folder.
    copy = tmp_path / "policy-v1.md"
    shutil.copy(POLICY, copy)
    llm = ScriptedLLM()
    llm.add(CriticOutput, NOT_FOUND)
    state = run_round(make_deps(tmp_path / "root", llm), policy_path=copy, scenarios=SCENARIOS)
    assert state["event"].outcome == "no_pattern"
