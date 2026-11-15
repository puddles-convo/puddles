import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(*args, env=None):
    return subprocess.run([sys.executable, "-m", "loop.run_batch", *args], cwd=ROOT,
                          capture_output=True, text=True, env=env or {})


def _args(out, policy="policies/policy-v1.md", scenarios="fixtures/scenarios.jsonl"):
    return ["--policy", policy, "--scenarios", scenarios, "--round", "1", "--out", str(out)]


def test_fake_run_prints_summary_and_writes_records(tmp_path):
    out = tmp_path / "round-01" / "conversations.jsonl"
    proc = _run(*_args(out), "--seed", "7", "--fake")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().count("\n") == 0
    assert json.loads(proc.stdout) == {"round": 1, "conversations": 5, "failed": 0}
    assert len(out.read_text().splitlines()) == 5


def test_max_turns_is_passed_through(tmp_path):
    out = tmp_path / "c.jsonl"
    proc = _run(*_args(out), "--max-turns", "1", "--fake")
    assert proc.returncode == 0, proc.stderr
    first = json.loads(out.read_text().splitlines()[0])
    assert [t["speaker"] for t in first["turns"]] == ["user", "helper"]


def test_missing_env_is_a_config_error(tmp_path):
    proc = _run(*_args(tmp_path / "c.jsonl"))
    assert proc.returncode == 2
    assert "HELPER_URL" in proc.stderr
    assert len(proc.stderr.strip().splitlines()) == 1
    assert proc.stdout == ""


def test_missing_policy_file_is_exit_2(tmp_path):
    proc = _run(*_args(tmp_path / "c.jsonl", policy="policies/nope.md"), "--fake")
    assert proc.returncode == 2
    assert len(proc.stderr.strip().splitlines()) == 1


def test_bad_scenario_file_is_exit_2(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"scenario_id": "x"}\n')
    proc = _run(*_args(tmp_path / "c.jsonl", scenarios=str(bad)), "--fake")
    assert proc.returncode == 2
    assert len(proc.stderr.strip().splitlines()) == 1


@pytest.mark.parametrize("which", ["policy", "scenarios"])
def test_non_utf8_input_file_is_exit_2_without_traceback(tmp_path, which):
    policy = tmp_path / "policy.md"
    policy.write_bytes((ROOT / "policies" / "policy-v1.md").read_bytes())
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_bytes((ROOT / "fixtures" / "scenarios.jsonl").read_bytes())
    bad = policy if which == "policy" else scenarios
    bad.write_bytes(bad.read_bytes() + b"\xff\n")
    proc = _run(*_args(tmp_path / "c.jsonl", policy=str(policy), scenarios=str(scenarios)),
                "--fake")
    assert proc.returncode == 2
    assert len(proc.stderr.splitlines()) == 1
    assert "Traceback" not in proc.stderr
    assert str(bad) in proc.stderr
    assert proc.stdout == ""


@pytest.mark.parametrize("which", ["policy", "scenarios"])
def test_file_name_with_line_breaks_still_gives_one_line_error(tmp_path, which):
    bad = tmp_path / "oddly\nnamed file"
    bad.write_bytes(b"\xff\n")
    policy = bad if which == "policy" else "policies/policy-v1.md"
    scenarios = bad if which == "scenarios" else "fixtures/scenarios.jsonl"
    proc = _run(*_args(tmp_path / "c.jsonl", policy=str(policy), scenarios=str(scenarios)),
                "--fake")
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "Traceback" not in proc.stderr
    assert len(proc.stderr.splitlines()) == 1, repr(proc.stderr)
    assert proc.stderr.count("oddly") == 1


def test_policy_error_with_line_break_in_name_names_file_once(tmp_path):
    bad = tmp_path / "bad\npolicy.md"
    bad.write_text("not a policy\n")
    proc = _run(*_args(tmp_path / "c.jsonl", policy=str(bad)), "--fake")
    assert proc.returncode == 2
    assert len(proc.stderr.splitlines()) == 1, repr(proc.stderr)
    assert proc.stderr.count("policy.md") == 1


def test_usage_error_with_line_break_in_value_gives_one_line(tmp_path):
    proc = _run("--policy", "p", "--scenarios", "s", "--out", "o", "--round", "1\n2")
    assert proc.returncode == 2
    assert len(proc.stderr.splitlines()) == 1, repr(proc.stderr)


def test_print_error_flattens_every_line_break(capsys):
    from loop.run_batch import _print_error
    _print_error("a\nb\r\nc\rd\x0be\x0cf\x1cg\x1dh\x1ei\x85j k l")
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1
    assert err == "error: a b c d e f g h i j k l\n"


def test_unwritable_output_is_exit_2_without_traceback(tmp_path):
    (tmp_path / "blocker").write_text("a file where a folder should be")
    proc = _run(*_args(tmp_path / "blocker" / "c.jsonl"), "--fake")
    assert proc.returncode == 2
    assert len(proc.stderr.splitlines()) == 1
    assert "Traceback" not in proc.stderr


def test_missing_argument_is_exit_2_with_one_line():
    proc = _run("--fake")
    assert proc.returncode == 2
    assert len(proc.stderr.strip().splitlines()) == 1


def test_no_successful_conversation_is_exit_1(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    proc = _run(*_args(tmp_path / "c.jsonl", scenarios=str(empty)), "--fake")
    assert proc.returncode == 1
    assert json.loads(proc.stdout) == {"round": 1, "conversations": 0, "failed": 0}
