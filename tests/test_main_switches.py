"""The switches main reads from the environment, and the weight step's place in the loop. All offline."""

import io
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest

from loop import llm as chat_api
from loop.runner import conversation_workers
from puddles.config import LoopConfig
from puddles.decision_log import DecisionLog
from puddles.loop import LOOP_LOG, run_loop
from scripts import run_graph_round
from tests.test_loop import POLICY, loop_kwargs, make_run_one

REPO = Path(__file__).parents[1]
SKILLS = "## Skills"


def _round(root: Path, tmp_path: Path) -> str:
    """One scripted round from the starting policy; returns the candidate policy it wrote."""
    code = run_graph_round.main([
        "--root", str(root), "--fake", "--gate", "kept", "--policy", str(POLICY),
        "--scenarios", str(REPO / "fixtures" / "scenarios.jsonl"), "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"])
    mlflow.flush_trace_async_logging()
    assert code == 0
    return (root / "policies" / "policy-v2-candidate-r01.md").read_text()


def _edit(root: Path) -> dict:
    return json.loads((root / "records" / "round-01" / "edit-r01.json").read_text())


# ---------- PUDDLES_PROPOSER ----------


@pytest.mark.parametrize("value", ["toolbox", " toolbox "])
def test_the_toolbox_proposer_runs_only_when_asked_for(tmp_path, monkeypatch, value):
    monkeypatch.setenv("PUDDLES_PROPOSER", value)
    candidate = _round(tmp_path / "run", tmp_path)
    assert SKILLS in candidate and "\nK1. " in candidate
    assert _edit(tmp_path / "run")["rule_id"] == "K1"
    assert DecisionLog(tmp_path / "run").rounds()[0].outcome == "kept"


@pytest.mark.parametrize("value", [None, "", "v1", "Toolbox", "TOOLBOX", "on", "1"])
def test_any_other_value_runs_the_rules_only_proposer(tmp_path, monkeypatch, value):
    monkeypatch.delenv("PUDDLES_PROPOSER", raising=False)
    plain = _round(tmp_path / "plain", tmp_path)
    assert SKILLS not in plain and "## Examples" not in plain
    assert _edit(tmp_path / "plain")["rule_id"].startswith("C")
    if value is not None:
        monkeypatch.setenv("PUDDLES_PROPOSER", value)
        assert _round(tmp_path / "other", tmp_path) == plain


# ---------- PUDDLES_CONVERSATION_WORKERS ----------


@pytest.mark.parametrize("value, workers", [(None, 1), ("", 1), ("1", 1), (" 4 ", 4), ("32", 32)])
def test_conversation_workers_defaults_to_one_at_a_time(monkeypatch, value, workers):
    monkeypatch.delenv("PUDDLES_CONVERSATION_WORKERS", raising=False)
    if value is not None:
        monkeypatch.setenv("PUDDLES_CONVERSATION_WORKERS", value)
    assert conversation_workers() == workers


@pytest.mark.parametrize("value", ["0", "33", "-1", "2.5", "four"])
def test_conversation_workers_refuses_anything_but_one_to_32(monkeypatch, value):
    monkeypatch.setenv("PUDDLES_CONVERSATION_WORKERS", value)
    with pytest.raises(ValueError, match="PUDDLES_CONVERSATION_WORKERS"):
        conversation_workers()


# ---------- PUDDLES_TOKEN_CAP ----------


@pytest.mark.parametrize("value, cap", [(None, 800), ("1200", 1200)])
def test_token_cap_is_800_unless_set(value, cap):
    import os

    env = {k: v for k, v in os.environ.items() if k != "PUDDLES_TOKEN_CAP"}
    if value is not None:
        env["PUDDLES_TOKEN_CAP"] = value
    proc = subprocess.run([sys.executable, "-c", "from puddles.policy import TOKEN_CAP; print(TOKEN_CAP)"],
                          capture_output=True, text=True, cwd=REPO, env=env)
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout) == cap


# ---------- {prefix}_USER_AGENT ----------


@pytest.mark.parametrize("value, sent", [(None, None), ("", None), ("puddles/0.1", "puddles/0.1")])
def test_user_agent_is_sent_only_when_set(monkeypatch, value, sent):
    monkeypatch.setenv("HELPER_URL", "http://helper.test/v1")
    monkeypatch.setenv("HELPER_MODEL", "m")
    for name in ("HELPER_API_KEY", "HELPER_EXTRA_BODY", "HELPER_MAX_TOKENS_PARAM", "HELPER_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv("HELPER_USER_AGENT", value)
    requests = []

    def urlopen(request, timeout=None):
        requests.append(request)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode())

    monkeypatch.setattr(chat_api.urllib.request, "urlopen", urlopen)
    client = chat_api.client_from_env("HELPER")
    assert client.complete("system", [{"role": "user", "content": "hi"}], max_tokens=5, temperature=None,
                           seed=None) == "hello"
    assert requests[0].get_header("User-agent") == sent


# ---------- the weight step after each round ----------


def test_the_weight_step_runs_after_every_round(tmp_path):
    seen = []
    summary = run_loop(LoopConfig(max_rounds=3, conversations_per_round=2), after_round=seen.append,
                       run_one=make_run_one(tmp_path, POLICY.read_text()), **loop_kwargs(tmp_path))
    assert seen == [1, 2, 3] == [r for r, _ in summary.rounds]
    assert not (tmp_path / LOOP_LOG).read_text().count("weight_error")


def test_a_failing_weight_step_is_written_down_and_the_loop_goes_on(tmp_path):
    def after_round(round_):
        if round_ == 1:
            raise RuntimeError("trainer unreachable")

    said = []
    summary = run_loop(LoopConfig(max_rounds=3, conversations_per_round=2), after_round=after_round,
                       run_one=make_run_one(tmp_path, POLICY.read_text()),
                       **loop_kwargs(tmp_path, out=said.append))
    assert [r for r, _ in summary.rounds] == [1, 2, 3]
    assert summary.stop_reason == "max_rounds (3)" and not summary.failed
    lines = [json.loads(line) for line in (tmp_path / LOOP_LOG).read_text().splitlines()]
    errors = [line for line in lines if line["event"] == "weight_error"]
    assert [(e["round"], e["error"]) for e in errors] == [(1, "RuntimeError: trainer unreachable")]
    assert any("weight step error" in line for line in said)
