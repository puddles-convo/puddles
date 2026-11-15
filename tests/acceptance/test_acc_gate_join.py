"""Acceptance tests from gate-join.md; original cases authored before the join existed.

Includes the amended fail-closed rules for empty/contradictory safety evidence
and unmeasured baselines. Cases, prompts and the yes/no interpretation belong
to the public safety implementation alone. The builder's report is not output
that these tests inspect; price preflight may live in the script or graph.
"""

import ast
from collections import Counter
from copy import deepcopy
import inspect
import json
import os
import re
import socket
import subprocess
import sys
from threading import Lock
from types import SimpleNamespace

import mlflow
import pytest

from loop import llm as chat_api
from puddles import exa, gate, gate_join, graph, tracing
from puddles import llm as agent_api
from puddles import memory as memory_api
from puddles import round as round_api
from puddles.decision_log import DecisionLog
from puddles.eval import judge as judge_api
from puddles.eval import safety as safety_api
from puddles.eval.judge import FallbackConfig, JudgePanel, PanelConfig, SingleJudge
from puddles.gate_join import real_gate_deps, safety_check, safety_models
from puddles.judging import MeteredJudge
from puddles.memory import FakeMemory, now
from puddles.policy import apply_edit, parse_policy, render_policy
from puddles.usage import Meter, ModelPrice, Prices, metering, read_usage
from schemas import (
    ConversationRecord, FailurePattern, GateInput, JudgeOutput, PolicyEdit, ProposerOutput,
    SafetyCaseResult, SafetyResult, Scores,
)
from scripts import run_graph_round as script


MODELS = ["acc-join-a", "acc-join-b", "acc-join-c", "acc-join-astra"]
SAFETY_MODEL = "acc-join-safety"
RULE = "Ask one open question before offering advice."
BASE_REPLY = "Baseline response."
CANDIDATE_REPLY = "Candidate response."
SECOND_VALIDATION = "second-validation-marker"


def patch_public(monkeypatch, owner, name, replacement):
    """Replace a public boundary, including aliases imported before the patch."""
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, replacement)
    for consumer in (gate_join, script):
        for alias, value in tuple(vars(consumer).items()):
            if value is original:
                monkeypatch.setattr(consumer, alias, replacement)


def protected_sources(repo_root):
    paths = [repo_root / "schemas.py", repo_root / "puddles/gate.py"]
    for folder in (repo_root / "puddles/eval", repo_root / "safety"):
        paths.extend(p for p in folder.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    paths.extend((repo_root / "tests").glob("test_eval_*.py"))
    return {str(p.relative_to(repo_root)): p.read_bytes() for p in paths}


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch, repo_root):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if (key.endswith("API_KEY") or key.startswith("MLFLOW_")
                or key.startswith("PUDDLES_JUDGE") or key.startswith("PUDDLES_SAFETY")
                or key == "PUDDLES_FALLBACK_JUDGE_MODEL"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("DO_NOT_TRACK", "true")

    def no_network(*args, **kwargs):
        raise AssertionError("Gate-join acceptance tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    before = protected_sources(repo_root)
    yield
    # Catches runtime attempts to persist cases, schemas or rules in protected sources.
    assert protected_sources(repo_root) == before


@pytest.fixture
def local_mlflow(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()


def case_reply(case):
    """Only supply required substrings; the real runner evaluates assertions."""
    return " ".join([a.value for a in case.must if a.type == "contains"] + ["Case reply."])


class CaseAsker:
    """Answers derived at runtime using the safety module's public templates.

    Keys are entire public prompts, so a changed question, opener or helper reply
    cannot accidentally get a passing default. No parser or yes/no rule is copied.
    """

    model = SAFETY_MODEL

    def __init__(self, cases, *, unclear=None):
        self.answers = {}
        self.expected = Counter()
        self.target = None
        self.calls = []
        self.lock = Lock()
        for case in cases:
            for polarity, assertions, answer in (
                ("must", case.must, "yes"), ("must_not", case.must_not, "no"),
            ):
                for assertion in assertions:
                    if assertion.type != "judged":
                        continue
                    key = (safety_api.YES_NO_SYSTEM, safety_api.YES_NO_USER.format(
                        opener=case.opener, reply=case_reply(case), question=assertion.question,
                    ))
                    self.answers[key] = answer
                    self.expected[key] += 1
                    if unclear == (case.case_id, polarity) and self.target is None:
                        self.target = key
        if unclear is not None:
            assert self.target is not None, "The selected real case needs a judged assertion"
            self.expected[self.target] += 1  # the real runner retries an unclear answer once

    def ask(self, system, user):
        key = (system, user)
        with self.lock:
            self.calls.append(key)
        if key == self.target:
            return "maybe"
        return self.answers[key]


class Helper:
    def __init__(self, cases):
        self.replies = {case.opener: case_reply(case) for case in cases}
        self.calls = []

    def complete(self, system, messages, *, max_tokens, temperature=0.7, seed=None):
        self.calls.append(deepcopy(dict(system=system, messages=messages,
                                       max_tokens=max_tokens, seed=seed)))
        opener = messages[-1]["content"]
        if opener in self.replies:
            return self.replies[opener]
        return CANDIDATE_REPLY if RULE in system else BASE_REPLY


class User:
    def __init__(self):
        self.calls = []

    def complete(self, system, messages, *, max_tokens, temperature=0.7, seed=None):
        self.calls.append(deepcopy(dict(system=system, messages=messages,
                                       max_tokens=max_tokens, seed=seed)))
        return SECOND_VALIDATION + "." if SECOND_VALIDATION in system else "My hours were cut."


class PanelLLM:
    """Public JudgeLLM fake; the real panel performs all review and retry steps."""

    def __init__(self, model, *, fail_validation=False):
        self.model = model
        self.fail_validation = fail_validation
        self.calls = []

    def judge(self, system, user):
        self.calls.append((system, user, 17, 5))
        candidate = f"helper: {CANDIDATE_REPLY}" in user
        invalid = self.fail_validation and candidate and SECOND_VALIDATION in user
        value = 4 if candidate else 2
        return JudgeOutput(
            empathy=value, relevance=value, emotion_before=4, emotion_after=2,
            strategies=[] if invalid else ["Question"], note="Scripted rating of the helper turn.",
        ), 17, 5


class ObservedPanel(JudgePanel):
    def __init__(self, *, fail_validation=False):
        self.clients = [PanelLLM(model) for model in MODELS[:3]]
        astra = PanelLLM(MODELS[3], fail_validation=fail_validation)
        super().__init__(PanelConfig(judges=MODELS[:3], astra=MODELS[3]), self.clients, astra)
        self.records = []
        self.reported = []

    def score(self, record):
        self.records.append(record)
        result = super().score(record)
        self.reported.append((record, result.usage))
        return result


class ObservedSingle(SingleJudge):
    """The real fallback judge, observing its public score boundary and returned usage."""

    def __init__(self, *, fail_validation=False):
        self.client = PanelLLM(MODELS[0], fail_validation=fail_validation)
        super().__init__(FallbackConfig(model=self.client.model), self.client)
        self.records, self.reported = [], []

    def score(self, record):
        self.records.append(record)
        result = super().score(record)
        self.reported.append((record, result.usage))
        return result


class UntouchedMemory:
    """Any memory method is a preflight violation, including unexpected methods."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def forbidden(*args, **kwargs):
            self.calls.append(name)
            pytest.fail(f"Memory method {name} ran before safety-model price preflight")
        return forbidden


class Agents(script.FakeLLM):
    """Existing offline agents, with a predictable proposed rule for the helper fake."""

    def structured(self, system, user, schema):
        if schema is ProposerOutput:
            return schema(action="add", summary="Understand before advising.", rule_text=RULE)
        return super().structured(system, user, schema)


def prices():
    return Prices(models={name: ModelPrice(1.0, 2.0) for name in [*MODELS, SAFETY_MODEL]})


@pytest.fixture
def case(tmp_path, scenario_factory):
    cases = safety_api.load_cases()
    base = tmp_path / "starting.md"
    base.write_text(
        "# policy-v1\n\n## Safety\nS1. Keep the person safe.\n\n"
        "## Conversation\nC1. Listen carefully.\n", encoding="utf-8",
    )
    candidate_policy, rule_id = apply_edit(parse_policy(base.read_text()), "add", None,
                                           RULE, "policy-v2-candidate")
    candidate = tmp_path / "candidate.md"
    candidate.write_text(render_policy(candidate_policy), encoding="utf-8")
    scenarios = [scenario_factory("val-a"), scenario_factory("val-b", situation=SECOND_VALIDATION)]
    return SimpleNamespace(root=tmp_path / "run", base=base, candidate=candidate,
                           cases=cases, scenarios=scenarios, rule_id=rule_id,
                           helper=Helper(cases), user=User(), panel=ObservedPanel(),
                           asker=CaseAsker(cases))


def joined(case, **overrides):
    values = dict(helper_client=case.helper, user_client=case.user, judge=case.panel,
                  root=case.root, scenarios=case.scenarios, noise_measured=True,
                  max_turns=1, seed=31, asker=case.asker)
    values.update(overrides)
    return real_gate_deps(**values)


def candidate_result(case):
    edit = PolicyEdit(
        id="edit-r01", written_by="proposer", created_at=now(), round=1,
        pattern_id="pattern-r01", research_id="research-r01", base_policy_version="policy-v1",
        new_policy_version="policy-v2-candidate", policy_path=str(case.candidate),
        summary="Understand before advising.", action="add", rule_id=case.rule_id, rule_text=RULE,
    )
    pattern = FailurePattern(
        id="pattern-r01", written_by="critic", created_at=now(), round=1,
        name="early advice", description="Advice before understanding.", seen_in=1, out_of=1,
        example_conversation_ids=["conv-r01-training"],
    )
    return round_api.RoundResult(round=1, outcome="candidate", base_policy_path=case.base,
                                 edit=edit, pattern=pattern, candidate_path=case.candidate)


def finish(case, result, outcome):
    memory = FakeMemory()
    memory.ensure_training_audience()
    # finish_round expects the round's policies directory, normally made by prepare_round.
    (case.root / "policies").mkdir(parents=True, exist_ok=True)
    log = DecisionLog(case.root)
    event = round_api.finish_round(result, outcome, log, memory, case.root)
    assert log.rounds() == [event]
    return event


def gate_report(root):
    return json.loads((root / "scores/round-01-gate.json").read_text())


def assert_would_keep_reason(reason):
    """Check the counterfactual meaning without fixing the contract's prose."""
    text = reason.casefold()
    keep = r"(?:keep|kept|retain\w*|promot\w*|accept\w*|qualif\w*|eligible)"
    positive = r"(?:(?!\b(?:not|never|fail\w*|reject\w*|revert\w*)\b)[^.!?;]){0,120}"
    assert re.search(rf"\b(?:would|could|otherwise|except)\b{positive}\b{keep}\b", text)


def scored_records(root):
    return [ConversationRecord.model_validate_json(line)
            for path in sorted((root / "conversations").glob("*.jsonl"))
            if not path.name.endswith(".failures.jsonl")
            for line in path.read_text().splitlines() if line.strip()]


# Section 1: the join delegates; it neither rewrites safety evidence nor decides.


def test_gate_import_has_no_eval_dependency_in_source_or_fresh_process(repo_root):
    """Catches adding static or executed dynamic eval imports to the independent gate."""
    source = (repo_root / "puddles/gate.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert not any(a.name == "puddles.eval" or a.name.startswith("puddles.eval.")
                           for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module != "puddles.eval" and not module.startswith("puddles.eval.")
            assert not (module == "puddles" and any(a.name == "eval" for a in node.names))
            assert not (node.level and (module == "eval" or module.startswith("eval.")
                                        or any(a.name == "eval" for a in node.names)))
    program = """
import socket
import sys
def no_network(*args, **kwargs):
    raise AssertionError('Gate import must stay offline')
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
import puddles.gate
assert not any(name == 'puddles.eval' or name.startswith('puddles.eval.') for name in sys.modules)
"""
    result = subprocess.run(["env", "-u", "PYTHONPATH", sys.executable, "-B", "-c", program],
                            cwd=repo_root, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("selection", ["default", "empty", "explicit"])
def test_safety_adapter_calls_public_runner_once_with_unchanged_inputs_and_result(
    case, monkeypatch, selection,
):
    """Catches copied checks, dropped cases/file names, eager calls, retries or a new verdict."""
    cases = {"default": None, "empty": [], "explicit": list(reversed(case.cases[:2]))}[selection]
    calls = []
    answer = SafetyResult(policy_version="unmodified-result", case_file="returned-file.json",
                          passed=False, cases=[SafetyCaseResult(case_id="returned-id", passed=False)])

    def run(policy_version, reply, asker, cases=None, case_file="safety-v1.json"):
        calls.append((policy_version, reply, asker, cases, case_file))
        return answer

    patch_public(monkeypatch, safety_api, "run_safety", run)
    check = safety_check(case.asker, cases=cases, case_file="custom-case-set.json")
    assert calls == []
    reply = lambda turns: "Unused by the boundary spy."
    assert check("candidate-under-test", reply) is answer
    assert len(calls) == 1
    assert calls[0][0] == "candidate-under-test" and calls[0][1] is reply
    assert calls[0][2] is case.asker and calls[0][3] == cases
    assert calls[0][4] == "custom-case-set.json"
    answer = answer.model_copy(update={"policy_version": "second-result"})
    second_reply = lambda turns: "A different callback."
    assert check("another-candidate", second_reply) is answer
    assert len(calls) == 2 and calls[1][0] == "another-candidate"
    assert calls[1][1] is second_reply and calls[1][2] is case.asker and calls[1][3] == cases


def test_safety_adapter_runner_failure_is_not_retried_or_reported_as_pass(case, monkeypatch):
    """Catches converting a broken real safety check into a synthetic pass."""
    calls = []
    error = RuntimeError("safety runner failed")

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        raise error

    patch_public(monkeypatch, safety_api, "run_safety", run)
    with pytest.raises(RuntimeError) as caught:
        safety_check(case.asker)("candidate", lambda turns: "Reply.")
    assert caught.value is error and len(calls) == 1


@pytest.mark.parametrize("already_metered", [False, True])
def test_real_gate_deps_preserves_inputs_wraps_same_judge_once_and_meters_gate_scores(
    case, already_metered,
):
    """Catches a substitute judge, double metering, lost settings, or uncounted gate tokens."""
    judge = MeteredJudge(case.panel) if already_metered else case.panel
    deps = joined(case, judge=judge)
    assert isinstance(deps, gate.GateDeps)
    assert untrace(deps.helper_client) is case.helper and untrace(deps.user_client) is case.user
    assert deps.root == case.root and all(s in deps.scenarios for s in case.scenarios)
    assert (deps.noise_measured, deps.max_turns, deps.seed) == (True, 1, 31)
    assert isinstance(deps.judge, MeteredJudge)
    assert deps.judge.judge is case.panel and deps.judge.version == case.panel.version
    if already_metered:
        assert deps.judge is judge
    result = candidate_result(case)
    with metering(Meter(case.root, prices()), "gatekeeper"):
        outcome = gate.make_gate(deps, validation_scenario_ids=[s.scenario_id for s in case.scenarios],
                                 noise=0.1)(result)
    assert outcome.decision == "kept"
    records = scored_records(case.root)
    assert records and Counter(r.id for r in case.panel.records) == Counter(r.id for r in records)
    assert all(count == 1 for count in Counter(r.id for r in case.panel.records).values())
    expected = Counter((client.model, inp, out) for client in [*case.panel.clients, case.panel.astra]
                       for _, _, inp, out in client.calls)
    done = [row for row in read_usage(case.root / "log/usage.jsonl") if row.get("event") == "done"]
    judge_done = [row for row in done if row["model"] in MODELS]
    assert Counter((r["model"], r["tokens_in"], r["tokens_out"]) for r in judge_done) == expected
    assert all(r["agent"] == "gatekeeper" and r["estimated"] is False for r in judge_done)
    # The safety client returns text only: no invented safety-question usage entries.
    assert case.asker.calls and SAFETY_MODEL not in {r["model"] for r in done}


def test_real_gate_deps_defaults_and_env_asker_are_lazy(case, monkeypatch):
    """Catches bypassing from_env, changing defaults or making model calls during construction."""
    factories = []

    def factory(cls):
        factories.append("from_env")
        return case.asker

    monkeypatch.setattr(safety_api.OpenAIYesNo, "from_env", classmethod(factory))
    deps = real_gate_deps(helper_client=case.helper, user_client=case.user, judge=case.panel,
                          root=case.root, scenarios=case.scenarios)
    assert factories == ["from_env"]
    assert (deps.noise_measured, deps.max_turns, deps.seed) == (False, 10, None)
    assert not case.helper.calls and not case.user.calls
    assert not case.panel.records and not case.asker.calls
    result = deps.safety("candidate", lambda turns: case.helper.replies[turns[-1].text])
    assert result.passed and len(result.cases) == len(case.cases) > 0
    assert Counter(case.asker.calls) == case.asker.expected


def test_explicit_asker_avoids_env_factory_and_safety_models_names_only_that_model(case, monkeypatch):
    """Catches constructing an unused paid client or pricing a different safety model."""
    monkeypatch.setattr(safety_api.OpenAIYesNo, "from_env",
                        classmethod(lambda cls: pytest.fail("Explicit asker must bypass from_env")))
    deps = joined(case)
    assert safety_models(case.asker) == [SAFETY_MODEL]
    other = SimpleNamespace(model="another-safety-model")
    assert safety_models(other) == [other.model]
    assert not case.asker.calls and not case.panel.records
    assert deps.safety("candidate", lambda turns: case.helper.replies[turns[-1].text]).passed


def test_missing_safety_configuration_valueerror_propagates_before_work(case):
    """Catches deferring configuration until candidate evaluation or fabricating a pass."""
    with pytest.raises(ValueError):
        joined(case, asker=None)
    assert not case.helper.calls and not case.user.calls and not case.panel.records


# Section 3.1--3.4: real safety cases and real panel, including harmful failures.


@pytest.mark.parametrize("mode", ["kept", "noise_unmeasured", "unclear", "judge_failure"])
def test_single_judge_real_safety_round_keeps_or_reverts_and_meters_once(case, mode):
    """Catches panel-only wiring, unmetered fallback scoring, and failed fallback/safety keeps."""
    judge = ObservedSingle(fail_validation=mode == "judge_failure")
    target = next(c for c in case.cases if any(a.type == "judged" for a in c.must))
    asker = (CaseAsker(case.cases, unclear=(target.case_id, "must"))
             if mode == "unclear" else case.asker)
    result = candidate_result(case)
    check = gate.make_gate(joined(case, judge=judge, asker=asker,
                                  noise_measured=mode != "noise_unmeasured"),
                           validation_scenario_ids=[s.scenario_id for s in case.scenarios], noise=0.1)
    with metering(Meter(case.root, prices()), "gatekeeper"):
        outcome = check(result)
    assert outcome.decision == ("kept" if mode == "kept" else "reverted")
    if mode != "kept":
        failed = {"noise_unmeasured": "noise_not_measured", "unclear": "safety_cases",
                  "judge_failure": "complete_evidence"}[mode]
        assert failed in check.last_report.failed
    event = finish(case, result, outcome)
    assert event.outcome == outcome.decision and event.safety.cases == len(case.cases) > 0
    assert event.safety.passed is (mode != "unclear")
    assert Counter(asker.calls) == asker.expected
    records = scored_records(case.root)
    assert Counter(r.id for r in judge.records) == Counter(r.id for r in records)
    assert all(n == 1 for n in Counter(r.id for r in judge.records).values())
    expected = Counter((entry.model, entry.prompt_tokens, entry.completion_tokens)
                       for _, entries in judge.reported for entry in entries)
    done = [row for row in read_usage(case.root / "log/usage.jsonl")
            if row.get("event") == "done" and row.get("model") == judge.client.model]
    assert Counter((row["model"], row["tokens_in"], row["tokens_out"]) for row in done) == expected
    assert all(row["agent"] == "gatekeeper" and row["estimated"] is False for row in done)
    if mode == "judge_failure":
        failed_calls = [user for _, user, _, _ in judge.client.calls
                        if f"helper: {CANDIDATE_REPLY}" in user and SECOND_VALIDATION in user]
        assert len(failed_calls) == 2
    else:
        assert len(judge.client.calls) == len(records)


def test_empty_real_loader_reverts_despite_clean_measured_gain_and_records_no_pass(case, monkeypatch):
    """Catches vacuous safety passes leaking through the real join into a keep or event."""
    loads, returned = [], []
    real_run = safety_api.run_safety

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return []

    def run(policy_version, reply, asker, cases=None, case_file="safety-v1.json"):
        answer = real_run(policy_version, reply, asker, cases=cases, case_file=case_file)
        returned.append(answer.model_copy(deep=True))
        return answer

    patch_public(monkeypatch, safety_api, "load_cases", load)
    patch_public(monkeypatch, safety_api, "run_safety", run)
    result = candidate_result(case)
    check = gate.make_gate(joined(case),
                           validation_scenario_ids=[s.scenario_id for s in case.scenarios], noise=0.1)
    outcome = check(result)
    report = check.last_report
    assert len(loads) == len(returned) == 1
    assert returned[0].passed is True and returned[0].cases == []
    assert outcome.decision == "reverted" and outcome.safety_passed is False
    assert report.failed == ["safety_cases"] and report.checks["safety_cases"] == "failed"
    assert "no safety case was run" in outcome.reason
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "passed"
    assert report.checks["noise_not_measured"] == "passed"
    assert report.log_safety.passed is False and report.log_safety.cases == 0
    saved = SafetyResult.model_validate_json((case.root / "scores/round-01-safety.json").read_text())
    assert saved == returned[0]  # Save the real runner's evidence without rewriting its verdict.
    event = finish(case, result, outcome)
    assert event.outcome == "reverted" and event.safety.passed is False and event.safety.cases == 0
    assert not case.asker.calls


@pytest.mark.parametrize("reported_pass", [False, True])
def test_failed_case_overrides_result_pass_flag_in_gate_and_event(case, monkeypatch, reported_pass):
    """Catches trusting an aggregate pass while a reported case failed."""
    failed_id = case.cases[0].case_id
    answer = SafetyResult(
        policy_version="policy-v2-candidate", passed=reported_pass,
        cases=[SafetyCaseResult(case_id=failed_id, passed=False)],
    )
    patch_public(monkeypatch, safety_api, "run_safety", lambda *args, **kwargs: answer)
    result = candidate_result(case)
    check = gate.make_gate(joined(case),
                           validation_scenario_ids=[s.scenario_id for s in case.scenarios], noise=0.1)
    outcome = check(result)
    report = check.last_report
    assert outcome.decision == "reverted" and outcome.safety_passed is False
    assert report.failed == ["safety_cases"] and failed_id in outcome.reason
    assert report.log_safety.passed is False and report.log_safety.failed_case_ids == [failed_id]
    saved = SafetyResult.model_validate_json((case.root / "scores/round-01-safety.json").read_text())
    assert saved.passed is reported_pass and saved.cases[0].passed is False
    event = finish(case, result, outcome)
    assert event.safety.passed is False and event.safety.failed_case_ids == [failed_id]


@pytest.mark.parametrize("dimension", ["empathy", "relevance"])
@pytest.mark.parametrize("value", [0, 0.5, 5.5, 6])
def test_out_of_range_baseline_is_unmeasured_and_cannot_enable_keep(case, dimension, value):
    """Catches comparing quality against an unmeasured baseline in either dimension."""
    before = Scores(empathy=2, relevance=2).model_copy(update={dimension: value})
    inp = GateInput(round=1, edit_id="edit-r01", current_policy=str(case.base),
                    candidate_policy=str(case.candidate),
                    validation_scenario_ids=[s.scenario_id for s in case.scenarios],
                    current_scores=before, noise=0.1, token_cap=800, safety_case_set="safety-v1")
    report = gate.run_gate(inp, joined(case))
    assert report.outcome.decision == "reverted" and "complete_evidence" in report.failed
    assert "the policy in force has no measured scores" in report.outcome.reason
    assert report.checks["quality_gain"] == report.checks["dimension_drop"] == "not_run"


@pytest.mark.parametrize("dimension", ["empathy", "relevance"])
@pytest.mark.parametrize("value", [0, 0.5, 5.5, 6])
def test_out_of_range_cached_baseline_is_rescored(case, dimension, value):
    """Catches reusing an otherwise matching cache whose baseline was never measured."""
    deps = joined(case)
    ids = [s.scenario_id for s in case.scenarios]
    assert gate.current_scores(deps, case.base, ids).empathy == 2
    path = case.root / "scores/current.json"
    cached = json.loads(path.read_text())
    cached["scores"][dimension] = value
    path.write_text(json.dumps(cached), encoding="utf-8")
    case.panel.records.clear()
    measured = gate.current_scores(deps, case.base, ids)
    assert measured.empathy == measured.relevance == 2
    assert Counter(r.scenario_id for r in case.panel.records) == Counter(ids)
    assert all(r.policy_version == "policy-v1" for r in case.panel.records)
    assert json.loads(path.read_text())["scores"][dimension] == 2


@pytest.mark.parametrize("measured", [True, False])
def test_real_cases_run_on_candidate_and_round_event_records_real_nonempty_evidence(case, measured):
    """Catches fake/empty safety, incumbent replies, omitted cases, or a new promotion rule."""
    result = candidate_result(case)
    outcome = gate.make_gate(joined(case, noise_measured=measured),
                             validation_scenario_ids=[s.scenario_id for s in case.scenarios],
                             noise=0.1)(result)
    assert outcome.decision == ("kept" if measured else "reverted")
    report = gate_report(case.root)
    assert report["checks"]["safety_cases"] == "passed"
    assert report["would_keep"] is True
    if not measured:
        assert report["failed"] == ["noise_not_measured"]
        assert_would_keep_reason(outcome.reason)
    event = finish(case, result, outcome)
    assert event.safety.passed is True and event.safety.case_file == "safety-v1.json"
    assert event.safety.cases == len(case.cases) > 0 and event.safety.failed_case_ids == []
    saved = SafetyResult.model_validate_json((case.root / "scores/round-01-safety.json").read_text())
    assert saved.policy_version == "policy-v2-candidate" and saved.passed
    assert Counter(c.case_id for c in saved.cases) == Counter(c.case_id for c in case.cases)
    assert all(c.passed for c in saved.cases)
    assert Counter(case.asker.calls) == case.asker.expected
    safety_calls = [c for c in case.helper.calls if c["messages"][-1]["content"] in case.helper.replies]
    assert Counter(c["messages"][-1]["content"] for c in safety_calls) == Counter(c.opener for c in case.cases)
    assert all(RULE in c["system"] for c in safety_calls)
    assert all(client.calls for client in [*case.panel.clients, case.panel.astra])


@pytest.mark.parametrize("polarity", ["must", "must_not"])
def test_unclear_twice_on_real_question_reverts_and_names_failed_case(case, polarity):
    """Catches truthy 'maybe', treating unclear as no, or losing the failing case in the event."""
    target = next(c for c in case.cases if any(a.type == "judged" for a in getattr(c, polarity)))
    asker = CaseAsker(case.cases, unclear=(target.case_id, polarity))
    result = candidate_result(case)
    outcome = gate.make_gate(joined(case, asker=asker),
                             validation_scenario_ids=[s.scenario_id for s in case.scenarios],
                             noise=0.1)(result)
    assert outcome.decision == "reverted" and outcome.safety_passed is False
    assert "safety_cases" in gate_report(case.root)["failed"]
    event = finish(case, result, outcome)
    assert event.safety.passed is False and event.safety.failed_case_ids == [target.case_id]
    assert event.safety.cases == len(case.cases) > 0
    assert Counter(asker.calls) == asker.expected


def test_astra_failure_twice_on_one_validation_conversation_stops_keep(case):
    """Catches replacing the failed judge, retrying a whole score, or averaging survivors."""
    panel = ObservedPanel(fail_validation=True)
    result = candidate_result(case)
    outcome = gate.make_gate(joined(case, judge=panel),
                             validation_scenario_ids=[s.scenario_id for s in case.scenarios],
                             noise=0.1)(result)
    assert outcome.decision == "reverted"
    assert "complete_evidence" in gate_report(case.root)["failed"]
    failed_calls = [user for _, user, _, _ in panel.astra.calls
                    if f"helper: {CANDIDATE_REPLY}" in user and SECOND_VALIDATION in user]
    assert len(failed_calls) == 2
    assert Counter(r.scenario_id for r in panel.records if r.policy_version == "policy-v2-candidate") == {
        s.scenario_id: 1 for s in case.scenarios
    }
    assert finish(case, result, outcome).outcome == "reverted"


def test_safety_runner_exception_cannot_become_a_keep(case, monkeypatch):
    """Catches suppressing a failed public safety runner while the judge gives a clean gain."""
    calls = []

    def broken(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("real safety runner unavailable")

    patch_public(monkeypatch, safety_api, "run_safety", broken)
    inp = GateInput(round=1, edit_id="edit-r01", current_policy=str(case.base),
                    candidate_policy=str(case.candidate),
                    validation_scenario_ids=[s.scenario_id for s in case.scenarios],
                    current_scores=Scores(empathy=2, relevance=2), noise=0.1, token_cap=800,
                    safety_case_set="safety-v1")
    report = gate.run_gate(inp, joined(case))
    assert report.outcome.decision == "reverted" and "safety_cases" in report.failed
    assert report.outcome.safety_passed is False and len(calls) == 1
    assert report.safety is None and report.log_safety is None
    assert not (case.root / "scores/round-01-safety.json").exists()
    validation = [r for r in case.panel.records if r.policy_version == "policy-v2-candidate"]
    assert Counter(r.scenario_id for r in validation) == Counter(s.scenario_id for s in case.scenarios)


# Section 2 and 3.6: script defaults, shared dependencies, and preflight ordering.


@pytest.fixture
def script_case(case, tmp_path, scenario_factory, monkeypatch):
    training = [scenario_factory("train-only", split="train")]
    training_file, validation_file = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    training_file.write_text("".join(s.model_dump_json() + "\n" for s in training))
    validation_file.write_text("".join(s.model_dump_json() + "\n" for s in reversed(case.scenarios)))
    config = tmp_path / "run.toml"

    def configure(*, safety_price=True, missing=()):
        names = MODELS + ([SAFETY_MODEL] if safety_price else [])
        config.write_text("[output_limits]\ndefault = 80\n\n[prices]\nexa_search_usd = 0\n" + "".join(
            f'\n[prices.models."{name}"]\ninput_usd_per_million = 1\noutput_usd_per_million = 2\n'
            for name in names if name not in missing
        ))

    configure()
    monkeypatch.setenv("PUDDLES_CONFIG", str(config))
    # Only a presence sentinel; every client/search factory is replaced before use.
    monkeypatch.setenv("EXA_API_KEY", "offline-unused-placeholder")
    patch_public(monkeypatch, chat_api, "client_from_env",
                 lambda prefix: case.helper if prefix == "HELPER" else case.user)
    patch_public(monkeypatch, agent_api, "OpenAILLM", Agents)
    patch_public(monkeypatch, memory_api, "PastClient", lambda: FakeMemory())
    patch_public(monkeypatch, exa, "search", script.fake_search)
    args = ["--root", str(case.root), "--policy", str(case.base),
            "--scenarios", str(training_file), "--validation", str(validation_file),
            "--max-turns", "1", "--seed", "31", "--noise", "0.1", "--noise-measured",
            "--token-cap", "777", "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]
    return SimpleNamespace(case=case, args=args, configure=configure, training=training)


def install_real_factories(monkeypatch, case):
    patch_public(monkeypatch, judge_api, "judge_from_env", lambda: case.panel)
    monkeypatch.setattr(safety_api.OpenAIYesNo, "from_env", classmethod(lambda cls: case.asker))


def untrace(client):
    while isinstance(client, tracing.TracedClient):
        client = client.inner
    return client


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("measured", [False, True])
def test_nonfake_script_runs_real_gate_with_rounds_clients_judge_and_all_validation_ids(
    script_case, monkeypatch, local_mlflow, explicit, measured,
):
    """Catches old refusal/default, train/validation mixups, a different judge or missing safety price."""
    case = script_case.case
    install_real_factories(monkeypatch, case)
    gates, rounds, price_checks = [], [], []
    real_make, real_run, real_check = gate.make_gate, graph.run_round, Prices.check

    def make(*args, **kwargs):
        bound = inspect.signature(real_make).bind(*args, **kwargs).arguments
        gates.append((bound["deps"], {name: value for name, value in bound.items() if name != "deps"}))
        return real_make(*args, **kwargs)

    def run(*args, **kwargs):
        bound = inspect.signature(real_run).bind(*args, **kwargs).arguments
        rounds.append((bound["deps"], {name: value for name, value in bound.items() if name != "deps"}))
        return real_run(*args, **kwargs)

    def check(self, models, search=False):
        models = list(models)
        price_checks.append(models)
        if SAFETY_MODEL in models:
            assert not case.helper.calls and not case.user.calls and not case.panel.records
            assert not case.asker.calls
        return real_check(self, models, search=search)

    patch_public(monkeypatch, gate, "make_gate", make)
    patch_public(monkeypatch, graph, "run_round", run)
    monkeypatch.setattr(Prices, "check", check)
    args = (["--gate", "real"] if explicit else []) + script_case.args
    if not measured:
        args = [arg for arg in args if arg != "--noise-measured"]
    assert script.main(args) == 0
    assert len(gates) == len(rounds) == 1
    gdeps, options = gates[0]
    rdeps, run_options = rounds[0]
    assert rdeps.judge_panel is case.panel
    assert isinstance(gdeps.judge, MeteredJudge) and gdeps.judge.judge is case.panel
    assert untrace(gdeps.helper_client) is untrace(rdeps.helper_client) is case.helper
    assert untrace(gdeps.user_client) is untrace(rdeps.user_client) is case.user
    assert all(s in gdeps.scenarios for s in case.scenarios)
    assert Counter(options["validation_scenario_ids"]) == Counter(s.scenario_id for s in case.scenarios)
    assert options["noise"] == 0.1 and options["token_cap"] == 777
    assert gdeps.noise_measured is measured and (gdeps.max_turns, gdeps.seed) == (1, 31)
    assert run_options["scenarios"] == script_case.training
    assert any(SAFETY_MODEL in names for names in price_checks)
    event = DecisionLog(case.root).rounds()[0]
    assert event.outcome == ("kept" if measured else "reverted") and event.safety.passed
    if not measured:
        assert gate_report(case.root)["would_keep"] is True
        assert_would_keep_reason(event.reason)
    assert event.safety.case_file == "safety-v1.json" and event.safety.cases == len(case.cases) > 0
    assert Counter(case.asker.calls) == case.asker.expected
    records = scored_records(case.root)
    assert Counter(r.id for r in case.panel.records) == Counter(r.id for r in records)
    assert all(n == 1 for n in Counter(r.id for r in case.panel.records).values())
    validation_ids = {s.scenario_id for s in case.scenarios}
    expected = Counter(
        ("gatekeeper" if record.scenario_id in validation_ids else "judge",
         entry.model, entry.prompt_tokens, entry.completion_tokens)
        for record, entries in case.panel.reported for entry in entries
    )
    done = [row for row in read_usage(case.root / "log/usage.jsonl")
            if row.get("event") == "done" and row.get("model") in MODELS]
    assert Counter((row["agent"], row["model"], row["tokens_in"], row["tokens_out"])
                   for row in done) == expected
    assert all(row["estimated"] is False for row in done)


@pytest.mark.parametrize("missing", ["safety", "judge"])
@pytest.mark.parametrize("explicit", [False, True])
def test_missing_real_configuration_exits_two_one_stderr_line_before_any_write_or_call(
    script_case, monkeypatch, capsys, missing, explicit,
):
    """Catches late configuration discovery, old refusal masking wiring, or a traceback."""
    case = script_case.case
    calls = []
    if missing == "safety":
        # A valid injected round judge need not use an Astra environment variable
        # (the public factory also supports fallback judges).
        patch_public(monkeypatch, judge_api, "judge_from_env", lambda: case.panel)
        real_factory = safety_api.OpenAIYesNo.from_env

        def safety_factory(cls):
            calls.append("safety")
            return real_factory()

        monkeypatch.setattr(safety_api.OpenAIYesNo, "from_env", classmethod(safety_factory))
    else:
        # Keep the real judge factory: no judge variables or credentials are set.
        monkeypatch.setattr(safety_api.OpenAIYesNo, "from_env",
                            classmethod(lambda cls: case.asker))
        real_factory = judge_api.judge_from_env

        def judge_factory():
            calls.append("judge")
            return real_factory()

        patch_public(monkeypatch, judge_api, "judge_from_env", judge_factory)

    patch_public(monkeypatch, graph, "run_round",
                 lambda *args, **kwargs: pytest.fail("Round started with missing configuration"))
    patch_public(monkeypatch, tracing, "setup",
                 lambda *args, **kwargs: pytest.fail("Tracing wrote before configuration was checked"))
    before = {p: p.read_bytes() for p in case.base.parent.rglob("*") if p.is_file()}
    capsys.readouterr()
    assert script.main((["--gate", "real"] if explicit else []) + script_case.args) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    assert captured.err.strip()
    assert {p: p.read_bytes() for p in case.base.parent.rglob("*") if p.is_file()} == before
    assert not case.root.exists()
    assert not case.helper.calls and not case.user.calls and not case.panel.records and not case.asker.calls
    assert calls == [missing]


def test_unpriced_safety_model_stops_before_round_writes_memory_or_model_calls(
    script_case, monkeypatch, local_mlflow, capsys,
):
    """Catches checking only helper/panel prices, or checking safety after round work."""
    case = script_case.case
    install_real_factories(monkeypatch, case)
    script_case.configure(safety_price=False)
    memory = UntouchedMemory()
    patch_public(monkeypatch, memory_api, "PastClient", lambda: memory)
    # The configured MLflow store is outside root; section 2 forbids root writes.
    capsys.readouterr()
    assert script.main(script_case.args) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    assert SAFETY_MODEL in captured.err
    assert not case.root.exists() or not list(case.root.rglob("*"))
    assert not memory.calls and not case.helper.calls and not case.user.calls
    assert not case.panel.records and not case.asker.calls


@pytest.mark.parametrize("missing", [MODELS[0], MODELS[3]])
def test_unpriced_round_judge_cannot_write_mlflow_under_root_before_price_error(
    script_case, monkeypatch, capsys, missing,
):
    """Catches checking a judge price only after tracing has already written under root."""
    case = script_case.case
    install_real_factories(monkeypatch, case)
    script_case.configure(missing=[missing])
    memory = UntouchedMemory()
    patch_public(monkeypatch, memory_api, "PastClient", lambda: memory)
    args = list(script_case.args)
    args[args.index("--mlflow") + 1] = f"sqlite:///{case.root}/mlflow.db"
    capsys.readouterr()
    assert script.main(args) == 2
    captured = capsys.readouterr()
    assert not memory.calls and not case.helper.calls and not case.user.calls
    assert not case.panel.records and not case.asker.calls
    assert not case.root.exists() or not list(case.root.rglob("*")), "Price preflight followed a root write"
    assert captured.out == "" and len(captured.err.splitlines()) == 1 and missing in captured.err


def test_missing_safety_api_credentials_exit_two_before_work(script_case, monkeypatch, capsys):
    """Catches a missing OpenAI credential escaping the script's configuration boundary."""
    case = script_case.case
    patch_public(monkeypatch, judge_api, "judge_from_env", lambda: case.panel)
    monkeypatch.setenv("PUDDLES_SAFETY_MODEL", SAFETY_MODEL)
    memory = UntouchedMemory()
    patch_public(monkeypatch, memory_api, "PastClient", lambda: memory)
    capsys.readouterr()
    assert script.main(script_case.args) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1 and captured.err.strip()
    assert not memory.calls and not case.root.exists()
    assert not case.helper.calls and not case.user.calls and not case.panel.records


def test_script_prices_astra_safety_fallback_when_safety_model_is_unset(
    script_case, monkeypatch, local_mlflow,
):
    """Catches missing the public safety factory's Astra fallback during price preflight."""
    case = script_case.case
    monkeypatch.setenv("PUDDLES_JUDGE_ASTRA", SAFETY_MODEL)
    patch_public(monkeypatch, judge_api, "judge_from_env", lambda: case.panel)
    created, requests, checked = [], [], []

    def create(**request):
        requests.append(request["model"])
        messages = request["messages"]
        content = case.asker.ask(messages[0]["content"], messages[1]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def init(self, model, temperature=0.0, client=None):
        self.model, self.temperature = model, temperature
        self.client = offline_client

    offline_client = client
    monkeypatch.setattr(safety_api.OpenAIYesNo, "__init__", init)
    real_factory, real_check = safety_api.OpenAIYesNo.from_env, Prices.check

    def factory(cls):
        asker = real_factory()  # Keep the public environment-selection logic unchanged.
        created.append(asker)
        return asker

    def check(self, models, search=False):
        models = list(models)
        checked.append(models)
        return real_check(self, models, search=search)

    monkeypatch.setattr(safety_api.OpenAIYesNo, "from_env", classmethod(factory))
    monkeypatch.setattr(Prices, "check", check)
    assert script.main(script_case.args) == 0
    assert created and requests and set(requests) == {SAFETY_MODEL}
    for asker in created:
        assert safety_models(asker) == [SAFETY_MODEL]
        assert any(set(safety_models(asker)) <= set(models) for models in checked)
    assert Counter(case.asker.calls) == case.asker.expected
    assert DecisionLog(case.root).rounds()[0].outcome == "kept"


@pytest.mark.parametrize("mode", [None, "kept", "reverted", "real"])
def test_fake_script_retains_each_gate_mode_without_real_factories(
    script_case, monkeypatch, local_mlflow, mode,
):
    """Catches changing the fake default/stand-ins or constructing any real client in fake mode."""
    def forbidden(*args, **kwargs):
        pytest.fail("--fake reached a real service factory")

    patch_public(monkeypatch, judge_api, "judge_from_env", forbidden)
    patch_public(monkeypatch, chat_api, "client_from_env", forbidden)
    patch_public(monkeypatch, agent_api, "OpenAILLM", forbidden)
    patch_public(monkeypatch, memory_api, "PastClient", forbidden)
    monkeypatch.setattr(safety_api.OpenAIYesNo, "from_env", classmethod(forbidden))
    args = ["--fake", *script_case.args]
    if mode is not None:
        args.extend(["--gate", mode])
    assert script.main(args) == 0
    event = DecisionLog(script_case.case.root).rounds()[0]
    assert event.outcome == ("kept" if mode == "kept" else "reverted")
    if mode == "real":
        report = gate_report(script_case.case.root)
        assert "safety_cases" in report["failed"] and report["checks"]["safety_cases"] == "failed"
        assert "no safety case was run" in report["reason"]
        assert event.safety.cases == 0 and event.safety.passed is False
    assert not script_case.case.helper.calls and not script_case.case.asker.calls
