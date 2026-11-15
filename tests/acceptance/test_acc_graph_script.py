"""Command-line contract checks with temporary roots and offline dependencies."""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
from types import SimpleNamespace

import mlflow
import pytest

from loop import llm as chat_api
from loop.llm import FakeClient
from loop.run_batch import FAKE_HELPER_LINES, FAKE_USER_LINES
from loop.runner import load_scenarios, scenario_seed
from puddles import graph, tracing
from puddles import llm as agent_api
from puddles import memory as memory_api
from puddles.agents import judge as judge_api
from puddles.decision_log import DecisionLog
from puddles.memory import FakeMemory
from schemas import ConversationRecord
from scripts import run_graph_round as script
from tests.fakes import FakeSearch, ScriptedLLM


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch, repo_root):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.endswith("API_KEY") or key in {"MLFLOW_TRACKING_URI", "MLFLOW_TRACING_DESTINATION"}:
            monkeypatch.delenv(key, raising=False)
    for prefix in ("HELPER", "SIMUSER"):
        for suffix in ("URL", "MODEL"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    monkeypatch.delenv("PUDDLES_AGENT_MODEL", raising=False)
    monkeypatch.delenv("PUDDLES_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("Acceptance tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    # Relative defaults are exercised from a temporary replica of their inputs.
    for relative in ("policies/policy-v1.md", "fixtures/scenarios.jsonl"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo_root / relative).read_bytes())
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()


@contextmanager
def patched_script(monkeypatch, patches):
    """Patch public source APIs and their existing imports without reloading modules."""
    with monkeypatch.context() as patch:
        for owner, name, value in patches:
            original = getattr(owner, name)
            patch.setattr(owner, name, value)
            for alias, imported in tuple(vars(script).items()):
                if imported is original:
                    patch.setattr(script, alias, value)
        yield script


def tracking_uri(tmp_path):
    return f"sqlite:///{tmp_path}/mlflow.db"


def event_state(outcome="no_pattern", number=1):
    return {
        "event": SimpleNamespace(round=number, outcome=outcome, reason="Acceptance reason."),
        "result": SimpleNamespace(outcome="candidate"),
    }


def assert_printed(capsys, event):
    captured = capsys.readouterr()
    assert captured.out == f"round {event.round}: {event.outcome}. {event.reason}\n"


@pytest.mark.parametrize("decision", ["kept", "reverted"])
def test_fake_reaches_gate_prints_event_and_second_round_finishes(tmp_path, capsys, decision):
    root = tmp_path / "experiment"
    argv = ["--fake", "--root", str(root), "--mlflow", tracking_uri(tmp_path)]
    if decision == "kept":
        argv += ["--gate", "kept"]
    capsys.readouterr()
    assert script.main(argv) == 0
    first = DecisionLog(root).rounds()
    assert len(first) == 1 and first[0].round == 1 and first[0].outcome == decision
    assert_printed(capsys, first[0])
    conversations = root / "conversations/round-01-train.jsonl"
    records = [ConversationRecord.model_validate_json(line) for line in conversations.read_text().splitlines()]
    assert len(records) == len(load_scenarios(tmp_path / "fixtures/scenarios.jsonl"))
    for record in records:
        assert [turn.text for turn in record.turns if turn.speaker == "user"] == FAKE_USER_LINES
        assert [turn.text for turn in record.turns if turn.speaker == "helper"] == FAKE_HELPER_LINES
    assert (root / "records").is_dir() and (root / "policies").is_dir()
    assert script.main(argv) == 0
    rounds = DecisionLog(root).rounds()
    assert [event.round for event in rounds] == [1, 2]
    assert rounds[1].outcome in {"kept", "reverted", "no_pattern", "no_evidence"}
    assert rounds[1].policy.base == rounds[0].policy.in_force
    assert_printed(capsys, rounds[1])
    second = root / "conversations/round-02-train.jsonl"
    assert second.is_file()
    assert all(ConversationRecord.model_validate_json(line).id.startswith("conv-r02-")
               for line in second.read_text().splitlines())


def test_explicit_policy_scenarios_max_turns_seed_and_mlflow(tmp_path, capsys, monkeypatch):
    root = tmp_path / "custom-root"
    policy = tmp_path / "custom-policy.md"
    policy.write_bytes((tmp_path / "policies/policy-v1.md").read_bytes())
    scenarios = tmp_path / "one-scenario.jsonl"
    scenario = load_scenarios(tmp_path / "fixtures/scenarios.jsonl")[0]
    scenarios.write_text(scenario.model_dump_json() + "\n")
    calls, round_deps = [], []
    original_setup = tracing.setup
    original_run = graph.run_round

    def setup(tracking_uri=None, experiment="puddles"):
        calls.append(tracking_uri)
        return original_setup(tracking_uri=tracking_uri, experiment=experiment)

    def run_round(deps, **kwargs):
        round_deps.append(deps)
        return original_run(deps, **kwargs)

    capsys.readouterr()
    with patched_script(monkeypatch, [(tracing, "setup", setup), (graph, "run_round", run_round)]):
        assert script.main([
            "--fake", "--root", str(root), "--policy", str(policy),
            "--scenarios", str(scenarios), "--max-turns", "1", "--seed", "91",
            "--gate", "kept", "--mlflow", tracking_uri(tmp_path),
        ]) == 0
    assert calls == [tracking_uri(tmp_path)]
    event = DecisionLog(root).rounds()[0]
    assert event.outcome == "kept" and event.policy.base == "policy-v1"
    assert_printed(capsys, event)
    records = [ConversationRecord.model_validate_json(line) for line in (
        root / "conversations/round-01-train.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0].scenario_id == scenario.scenario_id
    assert len(records[0].turns) == 2
    mlflow.flush_trace_async_logging()
    experiment = mlflow.get_experiment_by_name("puddles")
    found = mlflow.search_traces(experiment_ids=[experiment.experiment_id], return_type="list")
    chat_spans = [span for trace in found for span in trace.data.spans
                  if span.name in {"llm:helper", "llm:simulated_user"}]
    assert {span.name for span in chat_spans} == {"llm:helper", "llm:simulated_user"}
    assert all(span.inputs["seed"] == scenario_seed(91, scenario.scenario_id) for span in chat_spans)
    assert len(round_deps) == 1
    for name, client in (("llm:helper", round_deps[0].helper_client),
                         ("llm:simulated_user", round_deps[0].user_client)):
        named_spans = [span for span in chat_spans if span.name == name]
        assert len(named_spans) == len(client.calls) > 0
        for span in named_spans:
            assert any(all(span.inputs.get(key) == value for key, value in call.items())
                       for call in client.calls), f"{name} belongs to the wrong client"


@pytest.mark.parametrize("promoted", [False, True])
def test_default_arguments_and_policy_in_force(tmp_path, capsys, monkeypatch, promoted):
    root = tmp_path / "runs/dev"
    chosen = tmp_path / "policies/policy-v1.md"
    if promoted:
        chosen = root / "policies/policy-v2.md"
        chosen.parent.mkdir(parents=True)
        chosen.write_text((tmp_path / "policies/policy-v1.md").read_text().replace("# policy-v1", "# policy-v2"))
    calls, setup_calls = [], []
    original_setup = tracing.setup

    def setup(tracking_uri=None, experiment="puddles"):
        setup_calls.append(tracking_uri)
        return original_setup(tracking_uri=tracking_uri, experiment=experiment)

    def run_round(deps, *, policy_path, scenarios, max_turns=10, seed=None):
        assert setup_calls == [None], "setup must precede the round"
        calls.append(deps)
        assert Path(deps.root).resolve() == root
        assert Path(policy_path).resolve() == chosen
        assert scenarios == load_scenarios(tmp_path / "fixtures/scenarios.jsonl")
        assert max_turns == 10 and seed is None
        assert isinstance(deps.memory, FakeMemory)
        return event_state()

    capsys.readouterr()
    with patched_script(monkeypatch, [(tracing, "setup", setup), (graph, "run_round", run_round)]):
        # No --root, --policy, --scenarios, --max-turns, --seed, --gate or --mlflow.
        assert script.main(["--fake"]) == 0
    assert len(calls) == 1 and setup_calls == [None]
    assert_printed(capsys, event_state()["event"])


@pytest.mark.parametrize("outcome", ["kept", "reverted", "no_pattern", "no_evidence"])
def test_main_returns_zero_and_prints_event_outcome_for_every_finished_round(tmp_path, capsys, monkeypatch, outcome):
    state = event_state(outcome, 7)
    capsys.readouterr()
    with patched_script(monkeypatch, [(graph, "run_round", lambda *a, **k: state)]):
        assert script.main(["--fake", "--root", str(tmp_path / "root"),
                            "--mlflow", tracking_uri(tmp_path)]) == 0
    assert_printed(capsys, state["event"])


@pytest.mark.parametrize("problem", [
    "missing_policy", "malformed_policy", "carriage_return", "wrapped_rule",
    "missing_scenarios", "malformed_json", "invalid_scenario", "non_utf8_policy", "non_utf8_scenarios",
])
def test_file_errors_return_two_one_stderr_line_and_empty_log(tmp_path, capsys, problem):
    root = tmp_path / "bad-input-root"
    policy = tmp_path / "policies/policy-v1.md"
    scenarios = tmp_path / "fixtures/scenarios.jsonl"
    if problem == "missing_policy":
        policy = tmp_path / "missing-policy.md"
    elif problem == "malformed_policy":
        policy.write_text("This has no title or rules.\n")
    elif problem == "carriage_return":
        policy.write_bytes(policy.read_bytes().replace(b"\n", b"\r\n"))
    elif problem == "wrapped_rule":
        policy.write_bytes(policy.read_bytes().replace(b"C4. Keep replies short.", b"C4. Keep replies\nshort."))
    elif problem == "missing_scenarios":
        scenarios = tmp_path / "missing-scenarios.jsonl"
    elif problem == "malformed_json":
        scenarios.write_text("{not JSON}\n")
    elif problem == "invalid_scenario":
        scenarios.write_text('{"scenario_id": "missing-required-fields"}\n')
    elif problem == "non_utf8_policy":
        policy.write_bytes(b"\xff\xfe\n")
    elif problem == "non_utf8_scenarios":
        scenarios.write_bytes(b"\xff\xfe\n")
    capsys.readouterr()
    code = script.main(["--fake", "--root", str(root), "--policy", str(policy),
                        "--scenarios", str(scenarios), "--mlflow", tracking_uri(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert len(captured.err.splitlines()) == 1 and captured.err.strip()
    assert DecisionLog(root).events() == []


def test_missing_configuration_returns_two_one_stderr_line_and_empty_log(tmp_path, capsys):
    root = tmp_path / "unconfigured"
    capsys.readouterr()
    assert script.main(["--root", str(root), "--mlflow", tracking_uri(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert len(captured.err.splitlines()) == 1 and captured.err.strip()
    assert DecisionLog(root).events() == []


def test_without_fake_constructs_documented_dependencies(tmp_path, capsys, monkeypatch):
    # Constructors are replaced before use, so neither keys nor a server are needed.
    monkeypatch.setenv("EXA_API_KEY", "acceptance-placeholder-not-a-real-key")
    exa = importlib.import_module("puddles.exa")
    helper, user = FakeClient([]), FakeClient([])
    probe = "acceptance search probe"
    expected = [{"url": "https://evidence.example/probe"}]
    llm, search, memory = ScriptedLLM(), FakeSearch({probe: expected}), FakeMemory()
    calls = []

    def client(prefix):
        calls.append(prefix)
        return {"HELPER": helper, "SIMUSER": user}[prefix]

    def agent():
        calls.append("OpenAILLM")
        return llm

    def past():
        calls.append("PastClient")
        return memory

    judge_llm = ScriptedLLM()

    def judge_from_env():
        calls.append("judge")
        return judge_llm, "judge-acceptance"

    def run_round(deps, **kwargs):
        assert deps.helper_client is helper and deps.user_client is user
        assert deps.llm is llm and deps.memory is memory
        assert deps.judge_llm is judge_llm and deps.judge_version == "judge-acceptance"
        assert deps.search(probe) == expected
        return event_state()

    capsys.readouterr()
    with patched_script(monkeypatch, [
        (chat_api, "client_from_env", client), (agent_api, "OpenAILLM", agent),
        (exa, "search", search), (memory_api, "PastClient", past),
        (judge_api, "llm_from_env", judge_from_env),
        (graph, "run_round", run_round),
    ]):
        assert script.main(["--root", str(tmp_path / "root"), "--mlflow", tracking_uri(tmp_path)]) == 0
    assert sorted(calls) == sorted(["HELPER", "SIMUSER", "OpenAILLM", "PastClient", "judge"])
    assert search.calls == [probe]
    assert_printed(capsys, event_state()["event"])


def test_chat_config_without_exa_key_returns_two_before_round(tmp_path, capsys, monkeypatch):
    root = tmp_path / "missing-exa"
    for prefix in ("HELPER", "SIMUSER"):
        monkeypatch.setenv(f"{prefix}_URL", "http://127.0.0.1:1/v1")
        monkeypatch.setenv(f"{prefix}_MODEL", "acceptance-model")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    capsys.readouterr()
    with patched_script(monkeypatch, [
        (graph, "run_round", lambda *a, **k: pytest.fail("round started without EXA_API_KEY")),
        (agent_api, "OpenAILLM", lambda: ScriptedLLM()),
        (memory_api, "PastClient", lambda: FakeMemory()),
    ]):
        code = script.main(["--root", str(root), "--mlflow", tracking_uri(tmp_path)])
    assert code == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and error.strip()
    assert DecisionLog(root).events() == []


def test_chat_and_exa_config_without_agent_model_returns_two(tmp_path, capsys, monkeypatch):
    root = tmp_path / "missing-agent"
    for prefix in ("HELPER", "SIMUSER"):
        monkeypatch.setenv(f"{prefix}_URL", "http://127.0.0.1:1/v1")
        monkeypatch.setenv(f"{prefix}_MODEL", "acceptance-model")
    monkeypatch.setenv("EXA_API_KEY", "acceptance-placeholder-not-a-real-key")
    monkeypatch.delenv("PUDDLES_AGENT_MODEL", raising=False)
    capsys.readouterr()
    with patched_script(monkeypatch, [
        (graph, "run_round", lambda *a, **k: pytest.fail("round started without agent configuration")),
        (memory_api, "PastClient", lambda: FakeMemory()),
    ]):
        code = script.main(["--root", str(root), "--mlflow", tracking_uri(tmp_path)])
    assert code == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and error.strip()
    assert DecisionLog(root).events() == []


@pytest.mark.parametrize("bad", ["malformed", "carriage_return"])
def test_bad_promoted_policy_returns_two_and_leaves_log_empty(tmp_path, capsys, monkeypatch, bad):
    root = tmp_path / "bad-promoted"
    promoted = root / "policies/policy-v2.md"
    promoted.parent.mkdir(parents=True)
    if bad == "carriage_return":
        promoted.write_bytes((tmp_path / "policies/policy-v1.md").read_bytes().replace(b"\n", b"\r\n"))
    else:
        promoted.write_text("not a policy\n")
    capsys.readouterr()
    with patched_script(monkeypatch, [
        (graph, "run_round", lambda *a, **k: pytest.fail("round started with a bad promoted policy")),
    ]):
        code = script.main(["--fake", "--root", str(root), "--mlflow", tracking_uri(tmp_path)])
    assert code == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and error.strip()
    assert DecisionLog(root).events() == []
    assert not (root / "conversations").exists()


def test_main_accepts_none_argv(tmp_path, monkeypatch, capsys):
    state = event_state()
    monkeypatch.setattr(sys, "argv", ["run_graph_round", "--fake", "--root", str(tmp_path / "root"),
                                      "--mlflow", tracking_uri(tmp_path)])
    capsys.readouterr()
    with patched_script(monkeypatch, [(graph, "run_round", lambda *a, **k: state)]):
        assert script.main(None) == 0
    assert_printed(capsys, state["event"])


@pytest.mark.parametrize("bad", [False, True])
def test_module_entry_point_propagates_zero_or_two_exit_code(tmp_path, monkeypatch, capsys, bad):
    root = tmp_path / "module-root"
    argv = ["run_graph_round", "--fake", "--root", str(root), "--mlflow", tracking_uri(tmp_path)]
    if bad:
        argv += ["--policy", str(tmp_path / "missing.md")]
    monkeypatch.setattr(sys, "argv", argv)
    # Execute the actual module entry point in-process under the network guard.
    monkeypatch.delitem(sys.modules, "scripts.run_graph_round", raising=False)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("scripts.run_graph_round", run_name="__main__")
    assert exited.value.code == (2 if bad else 0)
    if bad:
        captured = capsys.readouterr()
        assert len(captured.err.splitlines()) == 1
        assert DecisionLog(root).events() == []
    else:
        assert_printed(capsys, DecisionLog(root).rounds()[0])
