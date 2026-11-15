"""The round view: node summaries on the spans, gate numbers on the run, the report and show_round.

All offline: rounds are run with the fakes of scripts/run_graph_round.py and tests/fakes.py.
"""
import json
import re
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

import puddles.round_view as round_view
from loop.llm import FakeClient
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.decision_log import DecisionLog
from puddles.eval.judge import JudgePanel, PanelConfig
from puddles.graph import RoundDeps, run_round
from puddles.memory import FakeMemory
from puddles.round_view import SUMMARY_KEY, build_report, node_summary, report_path, with_summary
from schemas import ConversationRecord, CriticOutput, JudgeOutput, ResearcherOutput
from scripts import run_graph_round, show_round
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS_FILE = REPO / "fixtures" / "scenarios.jsonl"
SCENARIOS = load_scenarios(SCENARIOS_FILE)
NODES = ("converse", "judge", "critic", "researcher", "proposer", "gate", "finish")
KEYS = {
    "converse": {"finished", "failed", "file"},
    "judge": {"scored", "unscored", "mean_empathy", "mean_relevance", "judge_version"},
    "critic": {"pattern", "seen_in", "out_of"},
    "researcher": {"queries", "sources"},
    "proposer": {"action", "rule_id", "rule_text"},
    "gate": {"decision", "quality_before", "quality_after", "noise", "safety_passed", "failed_checks"},
    "finish": {"outcome", "policy_in_force", "copies"},
}
GATE_NUMBERS = ("quality_before", "quality_after", "noise")
QUERY = "counselling skills exploring before giving advice"
URL = "https://counselling.example.ac.uk/skills"
PATTERN = {"found": True, "name": "advice-before-understanding",
           "description": "Gives advice in its first reply before asking about the situation.",
           "seen_in": 1, "out_of": len(SCENARIOS), "example_conversation_ids": ["conv-r01-fixture-0001"]}


@pytest.fixture
def mlflow_db(tmp_path, monkeypatch):
    """Tracking in the test's folder; the working folder too, so no artifact lands in the repo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    tracing.setup(tracking_uri=uri)
    yield uri
    mlflow.flush_trace_async_logging()


def fake_round(root: Path, uri: str, gate: str) -> int:
    argv = ["--fake", "--root", str(root), "--policy", str(POLICY), "--scenarios", str(SCENARIOS_FILE),
            "--gate", gate, "--mlflow", uri, "--max-turns", "4"]
    return run_graph_round.main(argv)


def spoken_lines(root: Path, round_: int = 1) -> list[str]:
    path = root / "conversations" / f"round-{round_:02d}-train.jsonl"
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(ConversationRecord.model_validate_json(line))
        except ValueError:  # a line cut short by a test that damages the file
            pass
    return [t.text for r in records for t in r.turns if len(t.text.strip()) > 3]


def only_trace():
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    assert len(traces) == 1
    return traces[0]


def summaries(trace) -> dict[str, dict]:
    out = {}
    for span in trace.data.spans:
        name = span.name.removeprefix("summary:")
        if name in NODES and span.attributes.get(SUMMARY_KEY) is not None:
            out[name] = span.attributes[SUMMARY_KEY]
    return out


def run_of(round_: int):
    [run] = mlflow.search_runs(filter_string=f"attributes.run_name = 'round-{round_:02d}'", output_format="list")
    return run


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def scripted_deps(root: Path, llm) -> RoundDeps:
    search = FakeSearch({QUERY: [{"url": URL, "title": "Skills", "text": "..."}]})
    return RoundDeps(helper_client=FakeClient(_fake_helper), user_client=FakeClient(_fake_user), llm=llm,
                     search=search, memory=FakeMemory(), log=DecisionLog(root), root=root,
                     judge_panel=run_graph_round.FakeJudge())


# 1
def test_every_node_span_has_a_summary_and_none_holds_a_spoken_line(tmp_path, mlflow_db):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "real") == 0
    found = summaries(only_trace())
    assert set(found) == set(NODES)
    for node in NODES:
        assert KEYS[node] <= set(found[node]), node
    text = json.dumps(found)
    lines = spoken_lines(root)
    assert lines
    for line in lines:
        assert line not in text
    assert "Scripted judge" not in text  # the judge's note is not copied either
    assert found["converse"]["file"] == "conversations/round-01-train.jsonl"
    assert found["judge"]["judge_version"] == "judge-fake"
    assert found["judge"]["scored"] == found["converse"]["finished"] and found["judge"]["unscored"] == 0
    assert found["judge"]["mean_empathy"] == 3 and found["judge"]["mean_relevance"] == 3
    assert found["critic"]["pattern"] == "advice-before-understanding"
    assert found["researcher"]["queries"] == 1 and found["researcher"]["sources"][0].startswith("https://")
    assert found["proposer"]["action"] == "add" and found["proposer"]["rule_id"]
    assert found["gate"]["decision"] == "reverted"
    assert "noise_not_measured" in found["gate"]["failed_checks"]
    # The fake safety check runs no case: rule 7 says that is not a pass, as the gate itself now says.
    assert found["gate"]["safety_passed"] is False
    assert "safety_cases" in found["gate"]["failed_checks"]
    assert found["finish"]["outcome"] == "reverted"
    event = DecisionLog(root).rounds()[0]
    assert found["finish"]["policy_in_force"] == event.policy.in_force
    assert sorted(c["record_id"] for c in found["finish"]["copies"]) == sorted(r.id for r in event.records.values())
    assert {c["status"] for c in found["finish"]["copies"]} == {"confirmed"}


# 2
@pytest.mark.parametrize("answers, outcome, not_run", [
    ({CriticOutput: [{"found": False, "reason": "The conversations went well."}]}, "no_pattern",
     {"researcher", "proposer", "gate"}),
    ({CriticOutput: [PATTERN], ResearcherOutput: [{"queries": [QUERY], "sources": []}]}, "no_evidence",
     {"proposer", "gate"}),
])
def test_rounds_that_end_early_give_a_report_with_the_later_nodes_not_run(tmp_path, mlflow_db, answers, outcome,
                                                                          not_run):
    llm = ScriptedLLM()
    for schema, outs in answers.items():
        llm.add(schema, *outs)
    state = run_round(scripted_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS, max_turns=4)
    assert state["event"].outcome == outcome
    report = build_report(tmp_path, 1)
    assert report.startswith(f"# Round 1: {outcome}\n")
    rows = {line.split("|")[1].strip(): line for line in report.splitlines()
            if line.startswith("| ") and line.split("|")[1].strip() in NODES}
    assert set(rows) == set(NODES)
    for node in NODES:
        assert ("did not run" in rows[node]) == (node in not_run), node
    assert "The gate did not run." in report and "No edit." in report
    sections = [line for line in report.splitlines() if line.startswith("## ")]
    assert sections == ["## Nodes", "## Conversations", "## Memory", "## Policy change", "## Gate", "## Cost"]
    for line in spoken_lines(tmp_path):
        assert line not in report


def test_a_round_with_nothing_scored_marks_the_critic_not_run(tmp_path, mlflow_db):
    deps = scripted_deps(tmp_path, ScriptedLLM())
    deps.judge_panel = None
    deps.judge = lambda records: []
    assert run_round(deps, policy_path=POLICY, scenarios=SCENARIOS, max_turns=4)["event"].outcome == "no_pattern"
    report = build_report(tmp_path, 1)
    rows = [line for line in report.splitlines() if line.startswith("| critic ")]
    assert rows and "did not run" in rows[0]


# 3
def test_a_kept_round_shows_the_rule_diff(tmp_path, mlflow_db):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "kept") == 0
    event = DecisionLog(root).rounds()[0]
    report = build_report(root, 1)
    assert f"+ {event.edit.rule_id}. {event.edit.rule_text}" in report
    assert f"Policy in force after the round: {event.policy.in_force}" in report
    assert "Stand-in gate: nothing was measured." in report


def test_a_kept_rewording_shows_both_lines(tmp_path, mlflow_db):
    from puddles.round import policy_in_force, stand_in_gate
    from schemas import ProposerOutput

    def kept(result):
        return stand_in_gate(result, "kept")

    first = ScriptedLLM()
    first.add(CriticOutput, PATTERN)
    first.add(ResearcherOutput, {"queries": [QUERY], "sources": [{"url": URL, "title": "S", "technique": "Ask."}]})
    first.add(ProposerOutput, {"action": "add", "rule_text": "Ask one open question first.", "summary": "Ask."})
    deps = scripted_deps(tmp_path, first)
    deps.gate = kept
    run_round(deps, policy_path=POLICY, scenarios=SCENARIOS, max_turns=4)
    rule_id = DecisionLog(tmp_path).rounds()[0].edit.rule_id

    second = ScriptedLLM()
    second.add(CriticOutput, {**PATTERN, "example_conversation_ids": ["conv-r02-fixture-0001"]})
    second.add(ResearcherOutput, {"queries": [QUERY + " again"],
                                  "sources": [{"url": URL, "title": "S", "technique": "Ask."}]})
    second.add(ProposerOutput, {"action": "reword", "rule_id": rule_id,
                                "rule_text": "Ask one open question before any suggestion.", "summary": "Reword."})
    deps = scripted_deps(tmp_path, second)
    deps.search = lambda q: [{"url": URL, "title": "Skills", "text": "..."}]
    deps.gate = kept
    state = run_round(deps, policy_path=policy_in_force(tmp_path, POLICY), scenarios=SCENARIOS, max_turns=4)
    assert state["event"].outcome == "kept", state["event"].reason
    report = build_report(tmp_path, 2)
    assert f"- {rule_id}. Ask one open question first." in report
    assert f"+ {rule_id}. Ask one open question before any suggestion." in report


def test_failed_and_unscored_conversations_are_listed_without_text(tmp_path, mlflow_db):
    from loop.errors import LLMError

    def user(system, messages):
        if "two mugs" in system:  # fixture-0002
            raise LLMError("simulated user server down")
        return _fake_user(system, messages)

    class SometimesJudge(run_graph_round.FakeJudge):
        def score(self, record):
            if record.id.endswith("0003"):
                raise ValueError("the judge failed twice")
            return super().score(record)

    llm = ScriptedLLM()
    llm.add(CriticOutput, {"found": False, "reason": "The conversations went well."})
    deps = scripted_deps(tmp_path, llm)
    deps.user_client = FakeClient(user)
    deps.judge_panel = SometimesJudge()
    run_round(deps, policy_path=POLICY, scenarios=SCENARIOS, max_turns=4)
    report = build_report(tmp_path, 1)
    assert "| conv-r01-fixture-0003 |  |  | unscored |" in report
    assert "fixture-0002 (failed)" in report and "simulated user server down" in report
    for line in spoken_lines(tmp_path):
        assert line not in report


def test_a_reverted_round_with_the_real_gate_names_the_failed_checks(tmp_path, mlflow_db):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "real") == 0
    report = build_report(root, 1)
    gate = json.loads((root / "scores" / "round-01-gate.json").read_text())
    for name, status in gate["checks"].items():
        assert f"- {name}: {round_view.CHECK_WORDS[status]}" in report
    assert "- noise_not_measured: failed" in report
    unscored_or_scored = [line for line in report.splitlines() if line.startswith("| conv-r01-")]
    assert unscored_or_scored and all("| 3 | 3 |" in line for line in unscored_or_scored)


# 4
def test_a_stand_in_gate_leaves_the_numbers_off_the_run_and_null_in_the_summary(tmp_path, mlflow_db):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "kept") == 0
    metrics = run_of(1).data.metrics
    for name in GATE_NUMBERS:
        assert name not in metrics
    assert metrics["kept"] == 1
    gate = summaries(only_trace())["gate"]
    for name in (*GATE_NUMBERS, "safety_passed", "failed_checks"):
        assert gate[name] is None, name


def test_unmeasured_noise_is_never_shown_as_a_measured_zero(tmp_path, mlflow_db):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "real") == 0  # --fake --gate real: the noise was not measured
    gate_file = json.loads((root / "scores" / "round-01-gate.json").read_text())
    assert gate_file["checks"]["noise_not_measured"] != "passed"
    scores = DecisionLog(root).rounds()[0].scores
    assert scores is not None and scores.noise == 0
    metrics = run_of(1).data.metrics
    assert metrics["kept"] == 0
    assert metrics["quality_before"] == pytest.approx(scores.before.quality)
    assert "noise" not in metrics
    gate = summaries(only_trace())["gate"]
    assert gate["noise"] is None and gate["quality_before"] == pytest.approx(scores.before.quality)
    report = build_report(root, 1)
    assert "Noise: not measured; the figure given was 0.00." in report
    assert "Noise: 0.00" not in report


def test_measured_noise_is_shown_logged_and_summarised(tmp_path):
    from puddles.round import RoundResult
    from schemas import LogScores, Scores

    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "round-01-gate.json").write_text(json.dumps(
        {"round": 1, "decision": "reverted", "checks": {"noise_not_measured": "passed", "quality_gain": "failed"}}))
    result = RoundResult(round=1, outcome="candidate", base_policy_path=POLICY)
    result.scores = LogScores(before=Scores(empathy=3, relevance=3, quality=3.0),
                              after=Scores(empathy=3, relevance=3, quality=3.0), noise=0.125)

    class Outcome:
        decision = "reverted"

    state = {"round": 1, "result": result, "gate": Outcome()}
    assert round_view.gate_metrics(state, tmp_path)["noise"] == 0.125
    assert node_summary("gate", state, {"gate": Outcome()}, tmp_path)["noise"] == 0.125


def test_a_round_without_a_gate_logs_no_gate_numbers(tmp_path, mlflow_db):
    llm = ScriptedLLM()
    llm.add(CriticOutput, {"found": False, "reason": "The conversations went well."})
    run_round(scripted_deps(tmp_path, llm), policy_path=POLICY, scenarios=SCENARIOS, max_turns=4)
    metrics = run_of(1).data.metrics
    assert all(name not in metrics for name in (*GATE_NUMBERS, "kept"))


# 5
def test_show_round_attaches_the_report_to_the_rounds_run(tmp_path, mlflow_db, capsys):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "kept") == 0
    capsys.readouterr()
    assert show_round.main(["--root", str(root), "--mlflow", mlflow_db]) == 0
    out = capsys.readouterr().out
    text = report_path(root, 1).read_text()
    assert text in out and "not attached" not in out
    artifacts = MlflowClient(tracking_uri=mlflow_db).list_artifacts(run_of(1).info.run_id, "reports")
    assert [a.path for a in artifacts] == ["reports/round-01.md"]


def test_show_round_with_no_such_run_still_writes_and_prints(tmp_path, mlflow_db, capsys):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "kept") == 0
    other = f"sqlite:///{tmp_path}/other.db"
    capsys.readouterr()
    assert show_round.main(["--root", str(root), "--mlflow", other]) == 0
    out = capsys.readouterr().out
    assert report_path(root, 1).read_text() in out
    assert len([line for line in out.splitlines() if "not attached" in line]) == 1
    assert not (tmp_path / "other.db").exists()  # looking for the run made no tracking store


def test_show_round_uses_only_the_run_of_its_own_root(tmp_path, mlflow_db, capsys):
    first, second = tmp_path / "first", tmp_path / "second"
    assert fake_round(first, mlflow_db, "kept") == 0
    assert fake_round(second, mlflow_db, "reverted") == 0  # the newer round-01 in the same store
    client = MlflowClient(tracking_uri=mlflow_db)
    experiment = client.get_experiment_by_name("puddles")
    untagged = client.create_run(experiment.experiment_id, tags={"mlflow.runName": "round-01"})  # newest of all
    client.set_terminated(untagged.info.run_id)
    runs = {r.data.tags.get("root"): r.info.run_id for r in client.search_runs([experiment.experiment_id])}
    assert set(runs) == {str(first.resolve()), str(second.resolve()), None}
    capsys.readouterr()
    assert show_round.main(["--root", str(first), "--mlflow", mlflow_db]) == 0
    assert "not attached" not in capsys.readouterr().out
    assert [a.path for a in client.list_artifacts(runs[str(first.resolve())], "reports")] == ["reports/round-01.md"]
    assert not client.list_artifacts(runs[str(second.resolve())], "reports")
    assert not client.list_artifacts(runs[None], "reports")
    third = tmp_path / "third"  # a root whose round has no run of its own: never another root's
    third.mkdir()
    for name in ("log", "conversations"):
        for path in (first / name).iterdir():
            (third / name).mkdir(exist_ok=True)
            (third / name / path.name).write_bytes(path.read_bytes())
    assert show_round.main(["--root", str(third), "--mlflow", mlflow_db]) == 0
    assert len([line for line in capsys.readouterr().out.splitlines() if "not attached" in line]) == 1


def test_show_round_refuses_a_missing_log_or_round(tmp_path, mlflow_db, capsys):
    assert show_round.main(["--root", str(tmp_path / "empty"), "--mlflow", mlflow_db]) == 2
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "kept") == 0
    capsys.readouterr()
    assert show_round.main(["--root", str(root), "--round", "7", "--mlflow", mlflow_db]) == 2
    assert len(capsys.readouterr().err.strip().splitlines()) == 1
    bad = tmp_path / "bad"
    (bad / "log").mkdir(parents=True)
    (bad / "log" / "decisions.jsonl").write_text("{not json\n")
    assert show_round.main(["--root", str(bad), "--mlflow", mlflow_db]) == 2
    assert not (root / "reports" / "round-07.md").exists()


# 6
def test_build_report_writes_nothing_and_show_round_adds_only_the_report(tmp_path, mlflow_db, capsys):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "real") == 0
    assert fake_round(root, mlflow_db, "kept") == 0
    before = snapshot(root)
    build_report(root, 1)
    build_report(root, 2)
    assert snapshot(root) == before
    assert show_round.main(["--root", str(root), "--round", "1", "--mlflow", mlflow_db]) == 0
    after = snapshot(root)
    assert set(after) - set(before) == {"reports/round-01.md"}
    assert {k: v for k, v in after.items() if k in before} == before


# stale or half-written files never stop the view
def test_report_builds_after_stale_scores_from_an_interrupted_attempt(tmp_path, mlflow_db):
    stale = tmp_path / "scores" / "round-01-train.jsonl"
    stale.parent.mkdir(parents=True)
    old = run_graph_round.FakeJudge().score(ConversationRecord(
        id="conv-r01-fixture-0001", written_by="helper", created_at="2020-01-01T00:00:00Z", parent_ids=[],
        source="simulated", scenario_id="fixture-0001", split="train", policy_version="v1", turns=[])).score
    stale.write_text(old.model_dump_json() + "\n")
    deps = scripted_deps(tmp_path, ScriptedLLM())
    deps.judge_panel = None  # the stub judge saves no scores, so the stale file stays
    state = run_round(deps, policy_path=POLICY, scenarios=[], max_turns=4)
    assert state["event"].reason.startswith("No conversation finished.")
    report = build_report(tmp_path, 1)
    assert report.startswith("# Round 1: no_pattern")
    assert "1 score(s) in the scores file belong to no conversation of this round." in report
    assert "| conv-r01-fixture-0001 |" not in report
    judged = summaries(only_trace())["judge"]
    assert judged["scored"] == 0 and judged["mean_empathy"] is None and judged["judge_version"] is None


def _cut(path: Path) -> None:
    """A crash in the middle of writing the last line."""
    text = path.read_text()
    path.write_text(text + text.splitlines()[-1][: len(text.splitlines()[-1]) // 2])


def _stale_scores(root: Path) -> None:
    path = root / "scores" / "round-01-train.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    gone = {**first, "id": "score-conv-r01-gone", "conversation_id": "conv-r01-gone"}
    other = {**first, "id": "score-conv-r09-x", "conversation_id": "conv-r09-x"}
    path.write_text(path.read_text() + json.dumps(gone) + "\n" + json.dumps(other) + "\n")


def _bad_usage(root: Path) -> None:
    path = root / "log" / "usage.jsonl"
    line = {"event": "done", "kind": "model", "round": 1, "agent": "critic", "tokens_in": "lots", "cost_usd": 1}
    path.write_text(path.read_text() + json.dumps(line) + "\n")
    _cut(path)


def _failures(root: Path) -> None:
    path = root / "conversations" / "round-01-train.failures.jsonl"
    path.write_text(json.dumps({"scenario_id": "fixture-0009", "reason": "server down"}) + "\n{\"scenario_")


def _stale_gate(root: Path) -> None:
    path = root / "scores" / "round-01-gate.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "decision": "kept"}))


def _cut_record(root: Path) -> None:
    path = next((root / "records" / "round-01").glob("pattern-*.json"))
    path.write_text(path.read_text()[:40])


def _cut_reads(root: Path) -> None:
    path = root / "log" / "memory_reads.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((path.read_text() if path.exists() else "") + "{\"reader\": \"cri")


@pytest.mark.parametrize("damage, says", [
    (_stale_scores, "2 score(s) in the scores file belong to no conversation of this round."),
    (lambda root: _cut(root / "scores" / "round-01-train.jsonl"), "1 line(s) of scores/round-01-train.jsonl"),
    (lambda root: _cut(root / "conversations" / "round-01-train.jsonl"),
     "1 line(s) of conversations/round-01-train.jsonl"),
    (_failures, "1 line(s) of conversations/round-01-train.failures.jsonl"),
    (_bad_usage, "1 finished call(s) in log/usage.jsonl, of this round or of a round that cannot be read"),
    (lambda root: (root / "scores" / "round-01-gate.json").write_text(
        (root / "scores" / "round-01-gate.json").read_text()[:60]), "scores/round-01-gate.json could not be read"),
    (_stale_gate, "does not match this round's gate decision"),
    (_cut_record, "is missing or no longer matches the log"),
    (_cut_reads, "Who read the round's records could not be worked out"),
])
def test_stale_or_half_written_files_never_stop_the_view(tmp_path, mlflow_db, capsys, damage, says):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "real") == 0
    clean = build_report(root, 1)
    damage(root)
    before = snapshot(root)
    report = build_report(root, 1)
    assert says in report and says not in clean
    assert snapshot(root) == before
    for line in spoken_lines(root) if (root / "conversations" / "round-01-train.jsonl").exists() else []:
        assert line not in report
    state = {"round": 1, "records": []}
    for node in ("judge", "gate", "finish"):
        assert "error" not in node_summary(node, state, {}, root), node
    capsys.readouterr()
    assert show_round.main(["--root", str(root), "--mlflow", mlflow_db]) == 0
    assert says in capsys.readouterr().out


def test_scores_left_by_an_earlier_attempt_stay_out_of_the_judges_means(tmp_path):
    from schemas import ScoredConversation

    path = tmp_path / "scores" / "round-01-train.jsonl"
    path.parent.mkdir(parents=True)
    record = ConversationRecord(id="conv-r01-a", written_by="helper", created_at="2026-01-02T00:00:00Z",
                                parent_ids=[], source="simulated", scenario_id="a", split="train",
                                policy_version="v1", turns=[])
    now_score = run_graph_round.FakeJudge().score(record).score.model_copy(update={"empathy": 4, "relevance": 2})
    stray = now_score.model_copy(update={"id": "score-x", "conversation_id": "conv-r01-x", "empathy": 1,
                                         "relevance": 1, "judge_version": "old-judge"})
    path.write_text(stray.model_dump_json() + "\n" + now_score.model_dump_json() + "\n")
    scored = [ScoredConversation(conversation_id="conv-r01-a", empathy=4, relevance=2, note="n", turns=[])]
    summary = node_summary("judge", {"round": 1, "records": [record]}, {"scored": scored, "unscored": []}, tmp_path)
    assert summary == {"scored": 1, "unscored": 0, "mean_empathy": 4, "mean_relevance": 2,
                       "judge_version": "judge-fake"}


# a value read from a file is shown or used only when it is of its kind
@pytest.fixture
def gated_round(tmp_path, mlflow_db):
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "real") == 0
    return root


def test_a_score_of_this_rounds_conversation_is_kept_whatever_its_stamp(gated_round):
    path = gated_round / "scores" / "round-01-train.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[0].update(created_at="2001-01-01T00:00:00Z", empathy=5, relevance=1)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    report = build_report(gated_round, 1)
    assert f"| {lines[0]['conversation_id']} | 5 | 1 |" in report
    assert "belong to no conversation" not in report
    judged = node_summary("judge", {"round": 1}, {"unscored": []}, gated_round)
    assert judged["scored"] == len(lines) and judged["mean_empathy"] == pytest.approx((5 + 3 * (len(lines) - 1)) / len(lines))


SPOKEN = "I just feel like nobody would notice if I was gone."


@pytest.mark.parametrize("value", [{}, [], SPOKEN, 1, True, None, "PASSED"])
def test_a_check_value_that_is_not_a_status_is_unreadable_and_never_printed(gated_round, value):
    path = gated_round / "scores" / "round-01-gate.json"
    data = json.loads(path.read_text())
    data["checks"]["safety_cases"] = value
    path.write_text(json.dumps(data))
    report = build_report(gated_round, 1)
    assert "- safety_cases: unreadable" in report and SPOKEN not in report
    assert "are not passed, failed or not_run" in report
    assert "Noise: not measured; the figure given was 0.00." in report
    gate = node_summary("gate", {"round": 1}, {"gate": None}, gated_round)
    assert gate["failed_checks"] is None


@pytest.mark.parametrize("change", [
    {"checks": {SPOKEN: "failed"}}, {"round": "1"}, {"round": True}, {"decision": SPOKEN}, {"checks": [SPOKEN]},
])
def test_gate_report_fields_that_are_not_of_their_kind_are_never_printed(gated_round, change):
    path = gated_round / "scores" / "round-01-gate.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    report = build_report(gated_round, 1)
    assert SPOKEN not in report
    assert "not a gate check" in report or "not a gate report" in report
    assert node_summary("gate", {"round": 1}, {"gate": None}, gated_round)["failed_checks"] is None


@pytest.mark.parametrize("numbers", [
    {}, {"tokens_in": 10, "tokens_out": 5}, {"tokens_in": 10, "tokens_out": 5, "cost_usd": float("nan")},
    {"tokens_in": 10, "tokens_out": 5, "cost_usd": float("inf")},
    {"tokens_in": float("inf"), "tokens_out": 5, "cost_usd": 0.25},
    {"tokens_in": "10", "tokens_out": 5, "cost_usd": 0.25}, {"tokens_in": True, "tokens_out": 5, "cost_usd": 0.25},
    {"tokens_in": 10, "tokens_out": 5, "cost_usd": 0.25, "agent": ["x"]},
    {"tokens_in": 10, "tokens_out": 5, "cost_usd": 0.25, "kind": SPOKEN},
])
def test_usage_numbers_that_cannot_be_read_are_left_out_never_zero(gated_round, numbers):
    clean = build_report(gated_round, 1).split("## Cost", 1)[1]
    line = {"event": "done", "round": 1, "kind": "model", "agent": "odd-agent", **numbers}
    with (gated_round / "log" / "usage.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")
    cost = build_report(gated_round, 1).split("## Cost", 1)[1]
    assert "odd-agent" not in cost and "nan" not in cost.lower() and "inf" not in cost.lower()
    assert "left out of the totals" in cost
    assert [l for l in cost.splitlines() if l.startswith("| total")] == \
        [l for l in clean.splitlines() if l.startswith("| total")]


def test_failure_and_reads_lines_that_are_not_text_are_unreadable(gated_round):
    (gated_round / "conversations" / "round-01-train.failures.jsonl").write_text(
        json.dumps({"scenario_id": {"said": SPOKEN}, "reason": [SPOKEN]}) + "\n")
    reads = gated_round / "log" / "memory_reads.jsonl"
    reads.parent.mkdir(parents=True, exist_ok=True)
    reads.write_text(json.dumps({"reader": {"x": SPOKEN}, "record_ids": "pattern-r01"}) + "\n")
    report = build_report(gated_round, 1)
    assert SPOKEN not in report
    assert "| unreadable (failed) |  |  | failed: unreadable |" in report
    assert "Who read the round's records could not be worked out" in report
    assert "read later by unknown" in report


def test_non_finite_gate_figures_are_unreadable_and_never_measurements(tmp_path):
    from puddles.round import RoundResult
    from schemas import LogScores, Scores

    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "round-01-gate.json").write_text(json.dumps(
        {"round": 1, "decision": "reverted", "checks": {"noise_not_measured": "passed"}}))
    result = RoundResult(round=1, outcome="candidate", base_policy_path=POLICY)
    result.scores = LogScores(before=Scores(empathy=3, relevance=3, quality=float("nan")),
                              after=Scores(empathy=3, relevance=3, quality=float("inf")), noise=float("inf"))

    class Outcome:
        decision = "reverted"

    state = {"round": 1, "result": result, "gate": Outcome()}
    assert round_view.gate_metrics(state, tmp_path) == {"kept": 0}
    summary = node_summary("gate", state, {"gate": Outcome()}, tmp_path)
    assert summary["quality_before"] is None and summary["quality_after"] is None and summary["noise"] is None


# the view never breaks a round
def test_a_failing_mlflow_does_not_break_the_round(tmp_path, mlflow_db, monkeypatch):
    def boom():
        raise RuntimeError("mlflow is down")

    monkeypatch.setattr(round_view.mlflow, "get_current_active_span", boom)
    root = tmp_path / "root"
    assert fake_round(root, mlflow_db, "kept") == 0
    assert DecisionLog(root).rounds()[0].outcome == "kept"


def test_with_summary_without_a_span_returns_the_result_and_lets_errors_through(tmp_path):
    update = {"records": [], "failures": []}
    wrapped = with_summary("converse", lambda state: update, tmp_path)
    assert wrapped({"round": 1}) is update

    def fails(state):
        raise KeyError("inside the node")

    with pytest.raises(KeyError):
        with_summary("converse", fails, tmp_path)({"round": 1})

    def interrupted(state):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with_summary("judge", interrupted, tmp_path)({"round": 1})


def test_a_span_not_named_after_the_node_gets_a_summary_child(tmp_path, mlflow_db):
    wrapped = with_summary("converse", lambda state: {"records": [], "failures": []}, tmp_path)
    with mlflow.start_span(name="outer"):
        wrapped({"round": 1})
    spans = {s.name: s for s in only_trace().data.spans}
    assert spans["summary:converse"].attributes[SUMMARY_KEY]["finished"] == 0
    assert SUMMARY_KEY not in spans["outer"].attributes


def test_not_measured_is_null_never_zero(tmp_path):
    judged = node_summary("judge", {"round": 1}, {"scored": [], "unscored": ["conv-r01-a"]}, tmp_path)
    assert judged == {"scored": 0, "unscored": 1, "mean_empathy": None, "mean_relevance": None,
                      "judge_version": None}


# where his judge's model calls land in the trace
class SpanningJudgeLLM:
    """His JudgeLLM protocol, opening one MLflow span per call, as a real traced client would."""

    def __init__(self, model):
        self.model = model

    def judge(self, system, user):
        conversation = user.split("</conversation>")[0]
        helpers = len(re.findall(r"^\d+\. helper: ", conversation, re.M))
        with mlflow.start_span(name=f"llm:judge:{self.model}"):
            out = JudgeOutput(empathy=3, relevance=3, emotion_before=4, emotion_after=3,
                              strategies=["Question"] * helpers, note="A fake note.")
        return out, 10, 5


def test_where_the_panels_threaded_judge_calls_land(tmp_path, mlflow_db, capsys):
    config = PanelConfig(judges=["m-a", "m-b", "m-c"], astra="m-astra")
    panel = JudgePanel(config, [SpanningJudgeLLM(m) for m in config.judges], SpanningJudgeLLM("m-astra"))
    llm = ScriptedLLM()
    llm.add(CriticOutput, {"found": False, "reason": "The conversations went well."})
    deps = scripted_deps(tmp_path, llm)
    deps.judge_panel = panel
    from puddles.usage import Meter, ModelPrice, Prices

    deps.usage = Meter(tmp_path, Prices(models={m: ModelPrice(1.0, 1.0) for m in [*config.judges, config.astra]}))
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS, max_turns=4)
    assert state["event"].outcome == "no_pattern"
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    round_trace = next(t for t in traces if any(s.name == "judge" for s in t.data.spans))
    by_id = {s.span_id: s for s in round_trace.data.spans}
    judge_span = next(s for s in round_trace.data.spans if s.name == "judge")

    def under_judge(span):
        while span.parent_id in by_id:
            span = by_id[span.parent_id]
            if span is judge_span:
                return True
        return False

    in_round = [s for s in round_trace.data.spans if s.name.startswith("llm:judge:")]
    astra = [s for s in in_round if s.name == "llm:judge:m-astra"]
    assert astra and all(under_judge(s) for s in astra)  # Astra runs on the node's own thread
    elsewhere = [t for t in traces if t is not round_trace]
    panel_spans = [s for s in in_round if s not in astra] + [
        s for t in elsewhere for s in t.data.spans if s.name.startswith("llm:judge:m-")]
    conversations = len(state["records"])
    assert len(astra) == conversations and len(panel_spans) == 6 * conversations  # 3 judges, 2 calls each
    with capsys.disabled():
        print(f"\n[judge threads] round trace {round_trace.info.trace_id}: "
              f"{len(astra)} Astra spans under judge, "
              f"{len([s for s in in_round if s not in astra])} panel-judge spans in the round trace; "
              f"{len(elsewhere)} separate traces: "
              f"{sorted({t.info.trace_id for t in elsewhere})[:3]}...")
