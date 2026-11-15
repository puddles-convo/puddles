"""The overnight driver's unattended-run rules: docs/specs/unattended-run.md section 1. All offline."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

import scripts.run_loop as driver
from loop.errors import ConfigError
from loop.llm import FakeClient
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles.config import LoopConfig
from puddles.decision_log import DecisionLog
from puddles.graph import NO_CONVERSATION_REASON, NOTHING_SCORED_REASON, RoundDeps, run_round, stub_judge
from puddles.loop import run_loop
from puddles.memory import FakeMemory, now
from puddles.records import sha256_text
from schemas import CriticOutput, LogPolicy, RoundEvent, ScoreRecord
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")


# ---------- helpers ----------


def event(round_: int, reason: str = "test", outcome: str = "no_pattern") -> RoundEvent:
    return RoundEvent(round=round_, outcome=outcome, reason=reason,
                      policy=LogPolicy(base="policy-v1", base_sha256=sha256_text(POLICY.read_text()),
                                       in_force="policy-v1"))


def logged(root: Path, round_: int, reason: str = "test") -> RoundEvent:
    e = event(round_, reason)
    DecisionLog(root).append(e)
    return e


def loop_lines(root: Path) -> list[dict]:
    path = root / "log" / "loop.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run(root: Path, run_one, **cfg) -> object:
    return run_loop(LoopConfig(conversations_per_round=1, **cfg), log=DecisionLog(root), memory=FakeMemory(),
                    root=root, start_policy=POLICY, scenarios=SCENARIOS, run_one=run_one, out=lambda s: None)


def write_config(tmp_path: Path, monkeypatch, *, noise="0.2", noise_measured="true", judge_version=None,
                 safety_priced=True) -> Path:
    """A config file of the test's own, with ID files over the fixture scenarios."""
    val = tmp_path / "val_ids.txt"
    train = tmp_path / "train_ids.txt"
    val.write_text("fixture-0001\nfixture-0002\n")
    train.write_text("fixture-0003\nfixture-0004\nfixture-0005\n")
    price = ('[prices.models."safety-model"]\ninput_usd_per_million = 1.0\noutput_usd_per_million = 1.0\n'
             if safety_priced else "")
    lines = [f"noise = {noise}" if noise is not None else "", f"noise_measured = {noise_measured}",
             f'judge_version = "{judge_version}"' if judge_version else "",
             f'validation_ids = "{val}"', f'train_ids = "{train}"', "max_turns = 3", "seed = 11",
             "conversations_per_round = 2"]
    path = tmp_path / "run.toml"
    path.write_text("[prices]\nexa_search_usd = 0.005\n" + price + "\n[output_limits]\ndefault = 1000\n\n[loop]\n"
                    + "\n".join(x for x in lines if x) + "\n")
    monkeypatch.setenv("PUDDLES_CONFIG", str(path))
    return path


class _Judge:
    def __init__(self, kind="panel", version="judge-v9"):
        self.kind, self.version = kind, version

    def score(self, record):
        raise AssertionError("not called in these tests")


def patch_real(monkeypatch, *, judge=None, frozen_calls=None, calls=None):
    """Replace every real-mode constructor, so no key and no server is needed."""
    import loop.llm
    import puddles.eval.calibrate
    import puddles.eval.judge
    import puddles.eval.safety
    import puddles.llm
    import puddles.memory

    calls = calls if calls is not None else []
    judge = judge or _Judge()
    monkeypatch.setenv("EXA_API_KEY", "placeholder-not-a-key")

    def client(prefix):
        calls.append(prefix)
        return FakeClient(_fake_helper if prefix == "HELPER" else _fake_user)

    def frozen(j):
        if frozen_calls is not None:
            frozen_calls.append(j)

    monkeypatch.setattr(loop.llm, "client_from_env", client)
    monkeypatch.setattr(puddles.llm, "OpenAILLM", lambda: calls.append("OpenAILLM") or ScriptedLLM())
    monkeypatch.setattr(puddles.memory, "PastClient", lambda: calls.append("PastClient") or FakeMemory())
    monkeypatch.setattr(puddles.eval.judge, "judge_from_env", lambda: calls.append("judge") or judge)
    monkeypatch.setattr(puddles.eval.calibrate, "check_frozen", frozen)
    monkeypatch.setattr(puddles.eval.safety.OpenAIYesNo, "from_env",
                        classmethod(lambda cls: calls.append("asker") or SimpleNamespace(model="safety-model")))
    monkeypatch.setattr(driver, "load_esconv", lambda split: SCENARIOS)
    return calls


def capture_round(monkeypatch, root: Path, seen: list):
    def fake_run_round(deps, **kwargs):
        seen.append((deps, kwargs))
        return {"event": logged(root, deps.log.next_round())}

    monkeypatch.setattr(driver, "run_round", fake_run_round)


# ---------- 1.1 the real gatekeeper / 1.2 the judge through the round's adapter ----------


def test_gate_real_builds_the_gatekeeper_on_the_validation_ids_and_the_loop_table(tmp_path, monkeypatch):
    from puddles import gate as real_gate

    write_config(tmp_path, monkeypatch)
    judge = _Judge()
    patch_real(monkeypatch, judge=judge)
    seen: list = []
    root = tmp_path / "root"
    capture_round(monkeypatch, root, seen)
    code = driver.main(["--root", str(root), "--gate", "real", "--max-rounds", "1",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"])
    assert code == 0
    deps, kwargs = seen[0]
    gate = deps.gate
    assert isinstance(gate, real_gate._Gate)
    assert gate.ids == ["fixture-0001", "fixture-0002"] and gate.noise == 0.2
    assert gate.deps.noise_measured is True and gate.deps.max_turns == 3 and gate.deps.seed == 11
    assert [s.scenario_id for s in gate.deps.scenarios] == ["fixture-0001", "fixture-0002"]
    assert gate.deps.judge.judge is judge  # metered, as real_gate_deps does
    assert gate.deps.safety.__qualname__.startswith("safety_check")  # the real safety check, not the stand-in
    # 1.2: the judge goes to the round as judge_panel, and the old adapter is not installed
    assert deps.judge_panel is judge and deps.judge is stub_judge
    assert kwargs["max_turns"] == 3 and kwargs["seed"] == 11


def test_an_unpriced_safety_model_is_refused_before_any_round(tmp_path, monkeypatch, capsys):
    write_config(tmp_path, monkeypatch, safety_priced=False)
    patch_real(monkeypatch)
    seen: list = []
    capture_round(monkeypatch, tmp_path / "root", seen)
    assert driver.main(["--root", str(tmp_path / "root"), "--gate", "real", "--max-rounds", "1"]) == 2
    assert "safety-model" in capsys.readouterr().err and seen == []


def test_fake_gate_real_runs_the_gatekeeper_with_the_stand_in_safety_check(tmp_path, capsys):
    root = tmp_path / "root"
    code = driver.main(["--root", str(root), "--fake", "--gate", "real", "--max-rounds", "1",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"])
    assert code == 0
    events = [e for e in DecisionLog(root).events() if e.event == "round"]
    assert [e.round for e in events] == [1]
    assert events[0].outcome != "kept"  # the [loop] table's noise is unmeasured: the real gate keeps nothing
    assert loop_lines(root)[-1]["event"] == "stopped"


def test_a_provider_error_in_one_judgment_leaves_it_unscored_and_the_round_goes_on(tmp_path):
    class OneFails:
        version = "judge-t"

        def __init__(self):
            self.n = 0

        def score(self, record):
            self.n += 1
            if self.n == 1:
                raise openai.APIConnectionError(request=httpx.Request("POST", "https://judge.invalid"))
            return SimpleNamespace(usage=[], score=ScoreRecord(
                id=f"score-{record.id}", written_by="judge", created_at=now(), parent_ids=[record.id],
                conversation_id=record.id, judge_version=self.version, empathy=3, relevance=3,
                emotion_before=3, emotion_after=3, strategies=[], note="n"))

    import puddles.tracing as tracing

    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    llm = ScriptedLLM()
    llm.add(CriticOutput, {"found": False, "reason": "fine"})
    deps = RoundDeps(helper_client=FakeClient(_fake_helper), user_client=FakeClient(_fake_user), llm=llm,
                     search=FakeSearch({}), memory=FakeMemory(), log=DecisionLog(tmp_path), root=tmp_path,
                     judge_panel=OneFails())
    state = run_round(deps, policy_path=POLICY, scenarios=SCENARIOS[:3], max_turns=3, seed=1)
    assert state["event"].round == 1
    assert len(state["unscored"]) == 1 and len(state["scored"]) == 2


# ---------- 1.3 a stand-in never promotes with real models ----------


def test_gate_kept_without_fake_is_refused_before_any_call(tmp_path, monkeypatch, capsys):
    calls = patch_real(monkeypatch)
    assert driver.main(["--root", str(tmp_path), "--gate", "kept", "--max-rounds", "1"]) == 2
    assert "--fake" in capsys.readouterr().err and calls == []


def test_gate_reverted_without_fake_says_pilot_in_its_first_line(tmp_path, monkeypatch, capsys):
    import loop.llm

    write_config(tmp_path, monkeypatch)

    def no_client(prefix):
        raise ConfigError("stop here")

    monkeypatch.setattr(loop.llm, "client_from_env", no_client)
    monkeypatch.setattr(driver, "load_esconv", lambda split: SCENARIOS)
    driver.main(["--root", str(tmp_path / "root"), "--gate", "reverted", "--max-rounds", "1"])
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("pilot:") and "nothing can be kept" in first


# ---------- 1.4 no improvement run on unmeasured noise ----------


@pytest.mark.parametrize("noise,measured", [("0.2", "false"), ("0.0", "true"), ("nan", "true"), (None, "true")])
def test_gate_real_without_fake_needs_measured_positive_noise(tmp_path, monkeypatch, capsys, noise, measured):
    write_config(tmp_path, monkeypatch, noise=noise, noise_measured=measured)
    calls = patch_real(monkeypatch)
    assert driver.main(["--root", str(tmp_path / "root"), "--gate", "real", "--max-rounds", "1"]) == 2
    assert "noise" in capsys.readouterr().err and calls == []


# ---------- 1.5 which judge must be frozen ----------


def test_a_panel_judge_must_be_frozen_and_a_fallback_judge_need_not(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    for kind, expected in (("panel", 1), ("fallback", 0)):
        frozen: list = []
        patch_real(monkeypatch, judge=_Judge(kind=kind), frozen_calls=frozen)
        seen: list = []
        root = tmp_path / f"root-{kind}"
        capture_round(monkeypatch, root, seen)
        assert driver.main(["--root", str(root), "--gate", "real", "--max-rounds", "1",
                            "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]) == 0
        assert len(frozen) == expected and len(seen) == 1


@pytest.mark.parametrize("kind", ["panel", "fallback"])
def test_a_judge_other_than_loop_judge_version_is_refused(tmp_path, monkeypatch, capsys, kind):
    write_config(tmp_path, monkeypatch, judge_version="judge-v1")
    patch_real(monkeypatch, judge=_Judge(kind=kind, version="judge-v2"))
    seen: list = []
    capture_round(monkeypatch, tmp_path / "root", seen)
    assert driver.main(["--root", str(tmp_path / "root"), "--gate", "real", "--max-rounds", "1"]) == 2
    assert "judge_version" in capsys.readouterr().err and seen == []


@pytest.mark.parametrize("kind,frozen_checked", [("panel", True), ("fallback", False)])
def test_score_baselines_freezes_only_a_panel_judge(tmp_path, monkeypatch, capsys, kind, frozen_checked):
    import loop.llm
    import puddles.eval.calibrate
    import puddles.eval.data
    import puddles.eval.judge
    import scripts.score_baselines as baselines

    frozen: list = []
    monkeypatch.setattr(puddles.eval.judge, "judge_from_env", lambda: _Judge(kind=kind))
    monkeypatch.setattr(puddles.eval.calibrate, "check_frozen", lambda j: frozen.append(j))

    def stop(split):
        raise OSError("past the judge checks")

    monkeypatch.setattr(puddles.eval.data, "load_scenarios", stop)
    monkeypatch.setattr(loop.llm, "client_from_env", lambda prefix: FakeClient(_fake_helper))
    assert baselines.main(["--root", str(tmp_path), "--policy", str(POLICY)]) == 2
    assert "past the judge checks" in capsys.readouterr().err
    assert bool(frozen) is frozen_checked


# ---------- 1.6 an unexpected error costs one round ----------


def test_a_round_error_is_logged_and_the_loop_goes_on(tmp_path):
    calls = []

    def run_one(policy_path, drawn, round_):
        calls.append(round_)
        if len(calls) == 1:
            raise httpx.ReadTimeout("x" * 1000)
        return logged(tmp_path, round_)

    summary = run(tmp_path, run_one, max_rounds=2)
    assert calls == [1, 1, 2] and summary.stop_reason == "max_rounds (2)" and not summary.failed
    error = loop_lines(tmp_path)[0]
    assert error["event"] == "round_error" and error["round"] == 1
    assert error["error"] == "ReadTimeout: " + "x" * 300 and "at" in error


def test_an_error_in_prepare_round_is_a_round_error(tmp_path, monkeypatch):
    import puddles.loop

    def broken(*a):
        raise RuntimeError("memory said no")

    monkeypatch.setattr(puddles.loop, "prepare_round", broken)
    summary = run(tmp_path, lambda *a: pytest.fail("no round runs"), max_rounds=5)
    errors = [line for line in loop_lines(tmp_path) if line["event"] == "round_error"]
    assert len(errors) == 3 and errors[0]["error"] == "RuntimeError: memory said no"
    assert summary.failed and summary.stop_reason == "3 round errors in a row"


def test_an_outcome_resets_the_round_error_count(tmp_path):
    script = iter(["err", "err", "ok", "err", "err", "ok"])

    def run_one(policy_path, drawn, round_):
        if next(script) == "err":
            raise ValueError("boom")
        return logged(tmp_path, round_)

    summary = run(tmp_path, run_one, max_rounds=2)
    assert summary.stop_reason == "max_rounds (2)" and not summary.failed


def test_keyboard_interrupt_is_not_caught_but_is_written_down(tmp_path):
    def run_one(policy_path, drawn, round_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, run_one, max_rounds=3)
    lines = loop_lines(tmp_path)
    assert [line["event"] for line in lines] == ["stopped"] and "KeyboardInterrupt" in lines[0]["reason"]


# ---------- 1.7 three rounds with nothing scored ----------


def test_three_rounds_with_nothing_scored_stop_the_loop(tmp_path):
    reasons = iter([NOTHING_SCORED_REASON, NO_CONVERSATION_REASON, NOTHING_SCORED_REASON])
    calls = []

    def run_one(policy_path, drawn, round_):
        calls.append(round_)
        return logged(tmp_path, round_, next(reasons))

    summary = run(tmp_path, run_one, max_rounds=10)
    assert calls == [1, 2, 3] and summary.failed
    assert "scored" in summary.stop_reason


def test_a_scored_round_resets_the_nothing_scored_count(tmp_path):
    reasons = iter([NOTHING_SCORED_REASON, NOTHING_SCORED_REASON, "found nothing", NOTHING_SCORED_REASON,
                    NOTHING_SCORED_REASON])

    summary = run(tmp_path, lambda p, d, r: logged(tmp_path, r, next(reasons)), max_rounds=5)
    assert summary.stop_reason == "max_rounds (5)" and not summary.failed


# ---------- 1.8 why it stopped is written down ----------


def test_a_cap_stop_writes_the_stopped_line_and_prints_it(tmp_path):
    out: list[str] = []
    summary = run_loop(LoopConfig(max_rounds=1, conversations_per_round=1), log=DecisionLog(tmp_path),
                       memory=FakeMemory(), root=tmp_path, start_policy=POLICY, scenarios=SCENARIOS,
                       run_one=lambda p, d, r: logged(tmp_path, r), out=out.append)
    line = loop_lines(tmp_path)[-1]
    assert line["event"] == "stopped" and line["reason"] == "max_rounds (1)" and line["rounds"] == 1
    assert line["policy_in_force"] == str(POLICY) and line["spend_usd"] == 0.0 and "at" in line
    assert out[-1].startswith("stopped: max_rounds (1)") and str(POLICY) in out[-1]
    assert summary.policy_in_force == str(POLICY)


def test_the_driver_exits_0_for_a_cap_and_3_for_failures(tmp_path, capsys, monkeypatch):
    root = tmp_path / "ok"
    assert driver.main(["--root", str(root), "--fake", "--max-rounds", "1",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]) == 0
    assert loop_lines(root)[-1]["reason"] == "max_rounds (1)"

    def broken(deps, **kwargs):
        raise RuntimeError("server gone")

    monkeypatch.setattr(driver, "run_round", broken)
    root = tmp_path / "bad"
    assert driver.main(["--root", str(root), "--fake", "--max-rounds", "5",
                        "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]) == 3
    lines = loop_lines(root)
    assert [line["event"] for line in lines] == ["round_error"] * 3 + ["stopped"]
    assert lines[-1]["reason"] == "3 round errors in a row" and lines[-1]["rounds"] == 0
    assert "stopped: 3 round errors in a row" in capsys.readouterr().out


def test_a_same_agent_streak_stop_exits_3(tmp_path):
    def run_one(policy_path, drawn, round_):
        path = tmp_path / "log" / "usage.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"event": "failed", "round": round_, "agent": "critic"}) + "\n")
        return logged(tmp_path, round_)

    (tmp_path / "log").mkdir()
    summary = run(tmp_path, run_one, max_rounds=10)
    assert summary.failed and loop_lines(tmp_path)[-1]["reason"] == "critic failed in 3 rounds in a row"
