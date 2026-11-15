"""Acceptance tests from docs/specs/gate-join.md, without sight of the join.

Unspecified (and deliberately not invented here): empty safety suites are not
forbidden, the location/shape of the safety-token accounting note is undefined,
and the price preflight may live in the script or graph. Cases, prompts and the
yes/no interpretation belong to the public safety implementation alone.
"""

from collections import Counter
from copy import deepcopy
import json
import os
import re
import socket
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
from puddles.eval.judge import JudgePanel, PanelConfig
from puddles.gate_join import real_gate_deps, safety_check, safety_models
from puddles.judging import MeteredJudge
from puddles.memory import FakeMemory, now
from puddles.policy import apply_edit, parse_policy, render_policy
from puddles.usage import Meter, MissingPrice, ModelPrice, Prices, metering, read_usage
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

    def score(self, record):
        self.records.append(record)
        return super().score(record)


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
    assert re.search(r"\b(would|could|otherwise|except|if)\b|but for|were it", text)
    assert re.search(r"\b(keep|kept|retain\w*|promot\w*|accept\w*|qualif\w*|eligible)\b", text)


def scored_records(root):
    return [ConversationRecord.model_validate_json(line)
            for path in sorted((root / "conversations").glob("*.jsonl"))
            if not path.name.endswith(".failures.jsonl")
            for line in path.read_text().splitlines() if line.strip()]


# Section 1: the join delegates; it neither rewrites safety evidence nor decides.


@pytest.mark.parametrize("selection", ["default", "empty", "explicit"])
def test_safety_adapter_calls_public_runner_once_with_unchanged_inputs_and_result(
    case, monkeypatch, selection,
):
    """Catches copied checks, dropped cases/file names, eager calls, retries or a new verdict."""
    cases = {"default": None, "empty": [], "explicit": list(reversed(case.cases[:2]))}[selection]
    calls = []
    answer = SafetyResult(policy_version="unmodified-result", case_file="returned-file.json",
                          passed=False, cases=[SafetyCaseResult(case_id="returned-id", passed=False)])

    def run(version, reply, asker, *, cases=None, case_file="safety-v1.json"):
        calls.append((version, reply, asker, cases, case_file))
        return answer

    patch_public(monkeypatch, safety_api, "run_safety", run)
    check = safety_check(case.asker, cases=cases, case_file="custom-case-set.json")
    assert calls == []
    reply = lambda turns: "Unused by the boundary spy."
    assert check("candidate-under-test", reply) is answer
    assert len(calls) == 1
    assert calls[0][0] == "candidate-under-test" and calls[0][1] is reply
    assert calls[0][2] is case.asker and calls[0][3] is cases
    assert calls[0][4] == "custom-case-set.json"
    answer = answer.model_copy(update={"policy_version": "second-result"})
    second_reply = lambda turns: "A different callback."
    assert check("another-candidate", second_reply) is answer
    assert len(calls) == 2 and calls[1][0] == "another-candidate"
    assert calls[1][1] is second_reply and calls[1][2] is case.asker and calls[1][3] is cases


def test_safety_adapter_runner_failure_is_not_retried_or_reported_as_pass(case, monkeypatch):
    """Catches converting a broken real safety check into a synthetic pass."""
    calls = []
    error = RuntimeError("safety runner failed")

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        raise error

    patch_public(monkeypatch, safety_api, "run_safety", run)
    try:
        result = safety_check(case.asker)("candidate", lambda turns: "Reply.")
    except Exception:
        pass  # The contract does not prescribe the presentation of a runner exception.
    else:
        assert isinstance(result, SafetyResult) and result.passed is False
    assert len(calls) == 1


@pytest.mark.parametrize("already_metered", [False, True])
def test_real_gate_deps_preserves_inputs_wraps_same_judge_once_and_meters_gate_scores(
    case, already_metered,
):
    """Catches a substitute judge, double metering, lost settings, or uncounted gate tokens."""
    judge = MeteredJudge(case.panel) if already_metered else case.panel
    deps = joined(case, judge=judge)
    assert isinstance(deps, gate.GateDeps)
    assert untrace(deps.helper_client) is case.helper and untrace(deps.user_client) is case.user
    assert deps.root == case.root and deps.scenarios == case.scenarios
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
    assert not case.root.exists() and not case.helper.calls and not case.user.calls
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
    assert not case.root.exists()
    assert not case.helper.calls and not case.user.calls and not case.panel.records


# Section 3.1--3.4: real safety cases and real panel, including harmful failures.


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


# Section 2 and 3.6: script defaults, shared dependencies, and preflight ordering.


@pytest.fixture
def script_case(case, tmp_path, scenario_factory, monkeypatch):
    training = [scenario_factory("train-only", split="train")]
    training_file, validation_file = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    training_file.write_text("".join(s.model_dump_json() + "\n" for s in training))
    validation_file.write_text("".join(s.model_dump_json() + "\n" for s in reversed(case.scenarios)))
    config = tmp_path / "run.toml"

    def configure(*, safety_price=True):
        names = MODELS + ([SAFETY_MODEL] if safety_price else [])
        config.write_text("[output_limits]\ndefault = 80\n\n[prices]\nexa_search_usd = 0\n" + "".join(
            f'\n[prices.models."{name}"]\ninput_usd_per_million = 1\noutput_usd_per_million = 2\n'
            for name in names
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

    def make(deps, **kwargs):
        gates.append((deps, kwargs))
        return real_make(deps, **kwargs)

    def run(deps, **kwargs):
        rounds.append((deps, kwargs))
        return real_run(deps, **kwargs)

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
    assert gdeps.scenarios == list(reversed(case.scenarios))
    assert options["validation_scenario_ids"] == [s.scenario_id for s in reversed(case.scenarios)]
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
    script_case, monkeypatch, local_mlflow,
):
    """Catches checking only helper/panel prices, or checking safety after round work."""
    case = script_case.case
    install_real_factories(monkeypatch, case)
    script_case.configure(safety_price=False)
    memory_calls = []
    monkeypatch.setattr(FakeMemory, "ensure_training_audience", lambda self: memory_calls.append("ensure"))
    # The join contract fixes the preflight, but not price-error presentation.
    try:
        code = script.main(script_case.args)
    except MissingPrice:
        pass
    else:
        assert isinstance(code, int) and code != 0
    assert not case.root.exists() or not list(case.root.rglob("*"))
    assert not memory_calls and not case.helper.calls and not case.user.calls
    assert not case.panel.records and not case.asker.calls


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
        assert (script_case.case.root / "scores/round-01-gate.json").is_file()
        assert event.safety.cases == 0
    assert not script_case.case.helper.calls and not script_case.case.asker.calls
