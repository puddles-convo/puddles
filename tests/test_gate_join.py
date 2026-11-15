"""The gate joined to Ansh's judge and safety check (docs/specs/gate-join.md). All offline.

His `run_safety`, his loader, his `JudgePanel` and his `SingleJudge` run for real; only the model
calls are scripted.
No safety case, question or yes-or-no rule is typed here: the scripted asker and helper read his
cases through his loader and work out the passing answers from them.
"""
import ast
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest

from loop.llm import FakeClient
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import gate, gate_join, tracing
from puddles.decision_log import DecisionLog
from puddles.eval import safety as eval_safety
from puddles.eval.judge import FallbackConfig, JudgePanel, PanelConfig, SingleJudge
from puddles.graph import RoundDeps, run_round
from puddles.judging import MeteredJudge
from puddles.memory import FakeMemory
from puddles.round import RoundResult
from puddles.usage import Meter, ModelPrice, Prices, metering
from schemas import CriticOutput, JudgeOutput, ProposerOutput, ResearcherOutput
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")
IDS = [s.scenario_id for s in SCENARIOS]
NEW_RULE = "Ask one open question before any suggestion."
MARK = "What has been the hardest part of it for you?"  # the candidate's helper says this; the judge rewards it
PANEL = PanelConfig(judges=["fake-judge-a", "fake-judge-b", "fake-judge-c"], astra="fake-astra")
FALLBACK = FallbackConfig(model="fake-fallback")
PRICES = Prices(models={m: ModelPrice(1.0, 2.0) for m in [*PANEL.judges, PANEL.astra, FALLBACK.model]},
                exa_search_usd=0.0)
URL = "https://counselling.example.ac.uk/skills"
QUERY = "counselling skills exploring before giving advice"


# ---------- fakes worked out from his file ----------

def his_cases():
    return eval_safety.load_cases()


def contains_values(assertions):
    return [a.value for a in assertions if a.type == "contains"]


class Asker:
    """His YesNoLLM, scripted: yes to a `must` question, no to a `must_not` one, read from his cases.

    `override` maps (case_id, question) to a list of answers given in turn. Thread-safe: his
    run_safety asks from several threads.
    """

    model = "fake-safety"

    def __init__(self, override=None):
        self.cases = his_cases()
        self.override = {k: list(v) for k, v in (override or {}).items()}
        self.lock = threading.Lock()
        self.asked: list[tuple[str, str]] = []

    def _find(self, user):
        openers = [c for c in self.cases if c.opener in user]
        case = max(openers, key=lambda c: len(c.opener))
        need = {a.question: True for a in case.must if a.type == "judged"}
        need.update({a.question: False for a in case.must_not if a.type == "judged"})
        question = max((q for q in need if q in user), key=len)
        return case, question, need[question]

    def ask(self, system, user):
        case, question, wanted = self._find(user)
        with self.lock:
            self.asked.append((case.case_id, question))
            queued = self.override.get((case.case_id, question))
            if queued:
                return queued.pop(0)
        return "yes" if wanted else "no"


def helper(system, messages):
    """The helper: under the candidate it says MARK; to a safety opener it gives his `must` values."""
    opener = messages[0]["content"]
    for case in his_cases():
        if opener == case.opener:
            values = contains_values(case.must)
            assert not set(values) & set(contains_values(case.must_not))
            return " ".join(f"{v}." for v in values) + " I am here with you."
    if NEW_RULE in system:
        return MARK
    return _fake_helper(system, messages)


def helper_turns(user):
    """The number of helper turns in the first conversation of a judge prompt."""
    block = user.split("</conversation>")[0]
    return len(re.findall(r"^\d+\. helper: ", block, re.M))


class FakeJudgeLLM:
    """His JudgeLLM, scripted: 4 and 4 for a conversation with MARK in it, 3 and 3 otherwise."""

    def __init__(self, model, fails=lambda conversation: False):
        self.model, self.fails = model, fails
        self.calls = 0

    def judge(self, system, user):
        self.calls += 1
        conversation = user.split("</conversation>")[0]
        n = helper_turns(user)
        if self.fails(conversation):
            n += 1  # a label too many: it fails his checks, every time
        good = MARK in conversation
        out = JudgeOutput(empathy=4 if good else 3, relevance=4 if good else 3, emotion_before=4, emotion_after=3,
                          strategies=["Question"] * n, note="Scripted: the helper asked at turn 2.")
        return out, 100, 10


def panel(astra_fails=lambda conversation: False):
    return JudgePanel(PANEL, [FakeJudgeLLM(m) for m in PANEL.judges], FakeJudgeLLM(PANEL.astra, astra_fails))


def candidate_text():
    return POLICY.read_text().replace("# policy-v1", "# policy-v2-candidate") + f"C9. {NEW_RULE}\n"


def setup(tmp_path, *, judge=None, asker=None, noise_measured=True):
    current = tmp_path / "policy-v1.md"
    current.write_bytes(POLICY.read_bytes())
    root = tmp_path / "root"
    candidate = root / "policies" / "policy-v2-candidate-r01.md"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(candidate_text())
    deps = gate_join.real_gate_deps(helper_client=FakeClient(helper), user_client=FakeClient(_fake_user),
                                    judge=judge or panel(), root=root, scenarios=SCENARIOS,
                                    noise_measured=noise_measured, seed=7, asker=asker or Asker())
    return deps, current, candidate


def gated(deps, current, candidate):
    the_gate = gate.make_gate(deps, validation_scenario_ids=IDS, noise=0.15)
    result = RoundResult(round=1, outcome="candidate", base_policy_path=current, candidate_path=candidate,
                         edit=SimpleNamespace(id="edit-r01"))
    out = the_gate(result)
    return out, the_gate.last_report, result


@pytest.fixture
def traced(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


def round_deps(root, start, judge, asker):
    llm = ScriptedLLM()
    llm.add(CriticOutput, {"found": True, "name": "advice-before-understanding",
                           "description": "Gives advice in its first reply before asking about the situation.",
                           "seen_in": 1, "out_of": len(SCENARIOS),
                           "example_conversation_ids": [f"conv-r01-{IDS[0]}"]})
    llm.add(ResearcherOutput, {"queries": [QUERY], "sources": [{"url": URL, "title": "Skills",
                                                                "technique": "Explore first."}]})
    llm.add(ProposerOutput, {"action": "add", "rule_text": NEW_RULE, "summary": "Ask first."})
    helper_client, user_client = FakeClient(helper), FakeClient(_fake_user)
    gate_deps = gate_join.real_gate_deps(helper_client=helper_client, user_client=user_client, judge=judge,
                                         root=root, scenarios=SCENARIOS, noise_measured=True, asker=asker)
    search = FakeSearch({QUERY: [{"url": URL, "title": "Skills", "text": "..."}]})
    return RoundDeps(helper_client=helper_client, user_client=user_client, llm=llm, search=search,
                     memory=FakeMemory(), log=DecisionLog(root), root=root, start_policy=start,
                     judge_panel=judge, usage=Meter(root, PRICES),
                     gate=gate.make_gate(gate_deps, validation_scenario_ids=IDS, noise=0.15))


def start_policy(tmp_path):
    path = tmp_path / "policy-v1.md"
    path.write_bytes(POLICY.read_bytes())
    return path


# ---------- the module ----------

def test_safety_check_is_one_call_to_his_run_safety(monkeypatch):
    calls = []

    def spy(policy_version, reply, asker, cases=None, case_file="x"):
        calls.append((policy_version, reply, asker, cases, case_file))
        return "result"

    monkeypatch.setattr(eval_safety, "run_safety", spy)
    asker, reply = Asker(), (lambda turns: "r")
    assert gate_join.safety_check(asker)("policy-v2-candidate", reply) == "result"
    assert calls == [("policy-v2-candidate", reply, asker, None, "safety-v1.json")]


def test_real_gate_deps_meters_the_judge_once_and_passes_missing_configuration_through(tmp_path, monkeypatch):
    judge = panel()
    deps = gate_join.real_gate_deps(helper_client=FakeClient([]), user_client=FakeClient([]), judge=judge,
                                    root=tmp_path, scenarios=SCENARIOS, asker=Asker())
    assert isinstance(deps.judge, MeteredJudge) and deps.judge.judge is judge
    assert deps.noise_measured is False and deps.max_turns == 10 and deps.seed is None
    metered = MeteredJudge(judge)
    again = gate_join.real_gate_deps(helper_client=FakeClient([]), user_client=FakeClient([]), judge=metered,
                                     root=tmp_path, scenarios=SCENARIOS, asker=Asker())
    assert again.judge is metered
    assert gate_join.safety_models(Asker()) == ["fake-safety"]

    monkeypatch.delenv("PUDDLES_SAFETY_MODEL", raising=False)
    monkeypatch.delenv("PUDDLES_JUDGE_ASTRA", raising=False)
    with pytest.raises(ValueError, match="PUDDLES_SAFETY_MODEL"):
        gate_join.real_gate_deps(helper_client=FakeClient([]), user_client=FakeClient([]), judge=judge,
                                 root=tmp_path, scenarios=SCENARIOS)


def test_gate_py_still_imports_nothing_from_puddles_eval():
    tree = ast.parse((REPO / "puddles" / "gate.py").read_text())
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert names and not [n for n in names if n.startswith("puddles.eval")]


# ---------- test 1: his real safety check, passing ----------

def test_the_round_event_names_his_case_file_and_his_number_of_cases(tmp_path, traced):
    root, start = tmp_path / "root", start_policy(tmp_path)
    asker = Asker()
    state = run_round(round_deps(root, start, panel(), asker), policy_path=start, scenarios=SCENARIOS)
    event = state["event"]

    cases = his_cases()
    judged = sum(a.type == "judged" for c in cases for a in c.must + c.must_not)
    assert event.safety.model_dump() == {"case_file": "safety-v1.json", "passed": True, "cases": len(cases),
                                         "failed_case_ids": []}
    assert len(asker.asked) == judged  # every judged assertion asked once, nothing asked twice
    assert event.outcome == "kept" and event.policy.in_force == "policy-v2"
    saved = json.loads((root / "scores" / "round-01-safety.json").read_text())
    assert [c["case_id"] for c in saved["cases"]] == [c.case_id for c in cases]


# ---------- test 2: an answer that is neither word, twice ----------

def test_an_answer_that_is_neither_word_twice_fails_that_assertion(tmp_path):
    case = next(c for c in his_cases() if any(a.type == "judged" for a in c.must + c.must_not))
    question = next(a.question for a in case.must + case.must_not if a.type == "judged")
    asker = Asker({(case.case_id, question): ["maybe", "maybe"]})
    deps, current, candidate = setup(tmp_path, asker=asker)
    out, report, result = gated(deps, current, candidate)

    assert out.decision == "reverted" and out.safety_passed is False
    assert "safety_cases" in report.failed
    assert report.log_safety.failed_case_ids == [case.case_id]
    assert result.safety.failed_case_ids == [case.case_id]
    assert asker.asked.count((case.case_id, question)) == 2
    [failed] = [c for c in report.safety.cases if not c.passed]
    assert question in failed.reason


# ---------- test 3: his real JudgePanel ----------

def test_a_clean_candidate_is_kept_with_the_noise_measured(tmp_path):
    deps, current, candidate = setup(tmp_path, noise_measured=True)
    out, report, _ = gated(deps, current, candidate)
    assert (out.decision, report.failed) == ("kept", [])
    assert out.before.quality == 3.0 and out.after.quality == 4.0
    assert out.policy_in_force == "policy-v2" and out.safety_passed is True


def test_a_clean_candidate_is_reverted_with_the_noise_unmeasured_and_says_the_checks_would_keep_it(tmp_path):
    deps, current, candidate = setup(tmp_path, noise_measured=False)
    out, report, _ = gated(deps, current, candidate)
    assert out.decision == "reverted" and report.failed == ["noise_not_measured"] and report.would_keep
    assert "would have kept the candidate" in out.reason
    assert out.policy_in_force == "policy-v1"


# ---------- test 4: Astra failing twice ----------

def test_astra_failing_twice_on_one_validation_conversation_reverts_on_incomplete_evidence(tmp_path):
    # Only the candidate's conversations hold MARK. Astra fails on the first of them, at both tries;
    # the gate scores in scenario order, so that is the first validation conversation.
    first: list[str] = []

    def fails(conversation):
        if MARK in conversation and not first:
            first.append(conversation)
        return conversation in first

    judge = panel(astra_fails=fails)
    deps, current, candidate = setup(tmp_path, judge=judge)
    out, report, _ = gated(deps, current, candidate)
    assert out.decision == "reverted" and report.failed == ["complete_evidence"]
    assert f"conv-r01-val-{IDS[0]}" in out.reason
    assert out.after == gate.NOT_MEASURED


# ---------- test 5: his judge's tokens in the usage log ----------

def test_the_gates_judge_tokens_are_logged_under_gatekeeper(tmp_path, traced):
    root, start = tmp_path / "root", start_policy(tmp_path)
    run_round(round_deps(root, start, panel(), Asker()), policy_path=start, scenarios=SCENARIOS)
    lines = [json.loads(line) for line in (root / "log" / "usage.jsonl").read_text().splitlines()]
    done = [d for d in lines if d["agent"] == "gatekeeper" and d["event"] == "done"]

    per_conversation = 3 * 2 + 1  # three judges, each rating then re-checking, then Astra
    assert len(done) == 2 * len(SCENARIOS) * per_conversation  # the policy in force, then the candidate
    assert {d["model"] for d in done} == {*PANEL.judges, PANEL.astra}
    assert {(d["tokens_in"], d["tokens_out"]) for d in done} == {(100, 10)}
    assert not [d for d in lines if d.get("model") == Asker.model]  # his asker's tokens are not logged


# ---------- test 6: the script without --fake ----------

def real_script_env(monkeypatch, judge_from_env=None):
    import loop.llm
    import puddles.llm
    import puddles.memory
    from puddles.eval import judge as eval_judge

    monkeypatch.setattr(loop.llm, "client_from_env", lambda prefix: FakeClient(_fake_helper))
    monkeypatch.setattr(puddles.llm, "OpenAILLM", lambda: ScriptedLLM())
    monkeypatch.setattr(puddles.memory, "PastClient", lambda: FakeMemory())
    monkeypatch.setenv("EXA_API_KEY", "not-a-key")
    for name in ("PUDDLES_JUDGE", "PUDDLES_JUDGE_PANEL", "PUDDLES_JUDGE_ASTRA", "PUDDLES_SAFETY_MODEL"):
        monkeypatch.delenv(name, raising=False)
    if judge_from_env is not None:
        monkeypatch.setattr(eval_judge, "judge_from_env", judge_from_env)


@pytest.mark.parametrize("judge_set", [False, True], ids=["judge-unset", "only-safety-unset"])
def test_the_script_without_fake_and_the_judge_variables_unset_exits_2_and_writes_nothing(
        tmp_path, monkeypatch, capsys, judge_set):
    from scripts import run_graph_round as script

    real_script_env(monkeypatch, judge_from_env=(lambda: panel()) if judge_set else None)
    monkeypatch.setattr(script, "run_round", lambda *a, **k: pytest.fail("the round started"))
    root = tmp_path / "root"
    assert script.main(["--root", str(root), "--policy", str(POLICY), "--max-turns", "6"]) == 2
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: ValueError: PUDDLES_")
    assert ("PUDDLES_SAFETY_MODEL" in lines[0]) is judge_set
    assert not captured.out
    assert not root.exists()


def test_the_script_without_fake_builds_the_real_gate_on_the_rounds_judge(tmp_path, monkeypatch):
    from scripts import run_graph_round as script

    judge = panel()
    real_script_env(monkeypatch, judge_from_env=lambda: judge)
    monkeypatch.setattr(eval_safety.OpenAIYesNo, "from_env", classmethod(lambda cls: Asker()))
    made, checked = [], []
    real_make_gate = script.real_gate.make_gate

    def spy_gate(deps, **kw):
        made.append((deps, kw))
        return real_make_gate(deps, **kw)

    def fake_round(deps, **kw):
        checked.append(deps)
        return {"event": SimpleNamespace(round=1, outcome="reverted", reason="Scripted.")}

    class SpyPrices:
        def check(self, models, search=False):
            checked.append(list(models))

    monkeypatch.setattr(script.real_gate, "make_gate", spy_gate)
    monkeypatch.setattr(script, "run_round", fake_round)
    monkeypatch.setattr(script, "round_meter", lambda deps: SimpleNamespace(prices=SpyPrices()))
    monkeypatch.setattr(script.tracing, "setup", lambda tracking_uri=None: None)
    validation = tmp_path / "validation.jsonl"
    validation.write_text("".join(s.model_dump_json() + "\n" for s in SCENARIOS[:2]))
    code = script.main(["--root", str(tmp_path / "root"), "--policy", str(POLICY), "--max-turns", "6",
                        "--validation", str(validation), "--noise", "0.2", "--noise-measured", "--token-cap", "700"])
    assert code == 0
    [(deps, kw)] = made
    assert kw == {"validation_scenario_ids": IDS[:2], "noise": 0.2, "token_cap": 700}
    assert isinstance(deps.judge, MeteredJudge) and deps.judge.judge is judge
    assert deps.noise_measured is True and deps.max_turns == 6
    assert checked[0] == ["fake-safety"] and checked[1].judge_panel is judge


def test_with_the_fallback_judge_and_no_safety_model_the_line_names_puddles_safety_model(
        tmp_path, monkeypatch, capsys):
    from scripts import run_graph_round as script

    real_script_env(monkeypatch, judge_from_env=lambda: SingleJudge(FALLBACK, FakeJudgeLLM(FALLBACK.model)))
    monkeypatch.setenv("PUDDLES_JUDGE", "fallback")  # and PUDDLES_JUDGE_ASTRA is unset, as it usually is then
    monkeypatch.setattr(script, "run_round", lambda *a, **k: pytest.fail("the round started"))
    root = tmp_path / "root"
    assert script.main(["--root", str(root), "--policy", str(POLICY)]) == 2
    captured = capsys.readouterr()
    assert captured.err.splitlines() == ["error: ValueError: PUDDLES_SAFETY_MODEL (or PUDDLES_JUDGE_ASTRA) is not set"]
    assert not captured.out and not root.exists()


def test_an_unpriced_safety_model_stops_the_script_before_memory_any_write_or_any_call(tmp_path, monkeypatch, capsys):
    from scripts import run_graph_round as script

    judge = panel()
    real_script_env(monkeypatch, judge_from_env=lambda: judge)
    monkeypatch.setattr(eval_safety.OpenAIYesNo, "from_env", classmethod(lambda cls: Asker()))
    touched = []

    class Memory(FakeMemory):
        def __getattribute__(self, name):
            if not name.startswith("_"):
                touched.append(name)
            return super().__getattribute__(name)

    import puddles.memory
    monkeypatch.setattr(puddles.memory, "PastClient", lambda: Memory())
    config = tmp_path / "run.toml"  # every model priced but the safety model
    rows = "".join(f'[prices.models."{m}"]\ninput_usd_per_million = 1\noutput_usd_per_million = 2\n'
                   for m in [*PANEL.judges, PANEL.astra])
    config.write_text(f"[prices]\nexa_search_usd = 0\n{rows}[output_limits]\ndefault = 100\n")
    monkeypatch.setenv("PUDDLES_CONFIG", str(config))
    root = tmp_path / "root"
    code = script.main(["--root", str(root), "--policy", str(POLICY), "--mlflow", f"sqlite:///{tmp_path}/m.db"])
    captured = capsys.readouterr()
    assert code == 2 and not captured.out
    [line] = captured.err.splitlines()
    assert line.startswith("error: MissingPrice: ") and "fake-safety" in line
    assert not root.exists() and touched == []
    assert not (tmp_path / "m.db").exists()  # tracing was not even set up
    assert judge.astra.calls == 0 and not [j for j in judge.judges if j.calls]


# ---------- test 7: the judge a real round uses now ----------

def test_the_single_model_judge_keeps_a_clean_candidate_once_per_conversation_with_its_tokens_logged(tmp_path):
    llm = FakeJudgeLLM(FALLBACK.model)
    seen = []
    judge_once = llm.judge

    def judge(system, user):
        seen.append(MARK in user.split("</conversation>")[0])
        return judge_once(system, user)

    llm.judge = judge
    deps, current, candidate = setup(tmp_path, judge=SingleJudge(FALLBACK, llm), noise_measured=True)
    with metering(Meter(deps.root, PRICES), "gatekeeper"):
        out, report, _ = gated(deps, current, candidate)

    assert (out.decision, report.failed) == ("kept", [])
    assert out.before.quality == 3.0 and out.after.quality == 4.0
    assert seen.count(True) == len(IDS)  # once per validation conversation, no retry
    assert seen.count(False) == len(IDS)  # and once per conversation of the policy in force
    lines = [json.loads(x) for x in (deps.root / "log" / "usage.jsonl").read_text().splitlines()]
    done = [d for d in lines if d["event"] == "done"]
    assert len(done) == 2 * len(IDS)
    assert {(d["agent"], d["model"], d["tokens_in"], d["tokens_out"]) for d in done} == {
        ("gatekeeper", FALLBACK.model, 100, 10)}


# ---------- a round the real gate was started on and killed ----------

def test_a_round_killed_inside_the_real_gate_is_simply_run_again(tmp_path, traced):
    """gate.md section 9 refuses caller-given measurements for such a round; the real gate gives none."""
    root, start = tmp_path / "root", start_policy(tmp_path)
    judge = panel()
    astra = judge.astra

    class Killed:
        model = astra.model

        def judge(self, system, user):
            if MARK in user.split("</conversation>")[0]:
                raise KeyboardInterrupt  # killed while the candidate's validation is judged
            return astra.judge(system, user)

    judge.astra = Killed()
    with pytest.raises(KeyboardInterrupt):
        run_round(round_deps(root, start, judge, Asker()), policy_path=start, scenarios=SCENARIOS)
    assert (root / "conversations" / "round-01-validation.jsonl").exists()
    assert (root / "scores" / "round-01-safety.json").exists()
    assert DecisionLog(root).rounds() == []

    state = run_round(round_deps(root, start, panel(), Asker()), policy_path=start, scenarios=SCENARIOS)
    event = state["event"]
    assert event.round == 1 and event.outcome == "kept"
    assert event.safety.passed is True and event.scores.after.quality == 4.0
    assert DecisionLog(root).rounds() == [event]
