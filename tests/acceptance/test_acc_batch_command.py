"""CLI checks use tmp_path inputs and outputs, and no real service credentials."""

import json
import os
import runpy
import subprocess
import sys

import pytest


def command(repo_root, tmp_path, policy_path, scenarios, *extra):
    env = os.environ.copy()
    for prefix in ("HELPER", "SIMUSER"):
        for suffix in ("URL", "MODEL", "API_KEY"):
            env.pop(f"{prefix}_{suffix}", None)
    env.update(PYTHONPATH=str(repo_root), PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(tmp_path))
    return subprocess.run([
        sys.executable, "-B", "-m", "loop.run_batch", "--policy", str(policy_path),
        "--scenarios", str(scenarios), "--round", "4", "--out", str(tmp_path / "out" / "records.jsonl"),
        *extra,
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)


def test_fake_command_needs_no_env_and_prints_one_json_summary(repo_root, tmp_path, policy_path, scenario_factory):
    from schemas import ConversationRecord
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(scenario_factory().model_dump_json() + "\n", encoding="utf-8")
    result = command(repo_root, tmp_path, policy_path, scenarios, "--fake", "--seed", "7", "--max-turns", "2")
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 1
    assert json.loads(result.stdout) == {"round": 4, "conversations": 1, "failed": 0}
    records = [ConversationRecord.model_validate_json(line) for line in
               (tmp_path / "out" / "records.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0].id == "conv-r04-acceptance-01"
    assert sum(t.speaker == "helper" for t in records[0].turns) <= 2


def test_no_conversations_exits_one_and_prints_summary(repo_root, tmp_path, policy_path):
    scenarios = tmp_path / "empty.jsonl"
    scenarios.write_text("\n", encoding="utf-8")
    result = command(repo_root, tmp_path, policy_path, scenarios, "--fake")
    assert result.returncode == 1, result.stderr
    assert len(result.stdout.splitlines()) == 1
    assert json.loads(result.stdout) == {"round": 4, "conversations": 0, "failed": 0}


@pytest.mark.parametrize("problem", ["missing_scenarios", "missing_policy", "invalid_policy", "malformed_json", "invalid_scenario", "blocked_output_parent", "non_utf8_policy", "non_utf8_scenarios"])
def test_file_errors_exit_two_with_one_line_stderr(repo_root, tmp_path, policy_path, scenario_factory, problem):
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(scenario_factory().model_dump_json() + "\n", encoding="utf-8")
    if problem == "missing_scenarios":
        scenarios.unlink()
    elif problem == "missing_policy":
        policy_path.unlink()
    elif problem == "invalid_policy":
        policy_path.write_text("This is not a policy.\n", encoding="utf-8")
    elif problem == "malformed_json":
        scenarios.write_text("{broken\n", encoding="utf-8")
    elif problem == "invalid_scenario":
        scenarios.write_text('{"scenario_id":"incomplete"}\n', encoding="utf-8")
    elif problem == "blocked_output_parent":
        (tmp_path / "out").write_text("A file blocks the directory.", encoding="utf-8")
    elif problem == "non_utf8_policy":
        policy_path.write_bytes(b"# p\n## Safety\nS1. \xff\n")
    elif problem == "non_utf8_scenarios":
        scenarios.write_bytes(b"\xff\n")
    result = command(repo_root, tmp_path, policy_path, scenarios, "--fake")
    assert result.returncode == 2
    assert len(result.stderr.splitlines()) == 1
    assert result.stderr.strip()


@pytest.mark.parametrize("which", ["policy", "scenarios"])
def test_non_utf8_file_with_newline_in_name_has_one_line_error(
    repo_root, tmp_path, policy_path, scenario_factory, which,
):
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(scenario_factory().model_dump_json() + "\n", encoding="utf-8")
    bad_path = tmp_path / "invalid\ninput"
    bad_path.write_bytes(b"\xff\n")
    if which == "policy":
        policy_path = bad_path
    else:
        scenarios = bad_path
    result = command(repo_root, tmp_path, policy_path, scenarios, "--fake")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert len(result.stderr.splitlines()) == 1, repr(result.stderr)


def test_missing_environment_is_configuration_exit_two(repo_root, tmp_path, policy_path, scenario_factory):
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(scenario_factory().model_dump_json() + "\n", encoding="utf-8")
    result = command(repo_root, tmp_path, policy_path, scenarios)
    assert result.returncode == 2
    assert len(result.stderr.splitlines()) == 1
    assert any(f"{prefix}_{suffix}" in result.stderr
               for prefix in ("HELPER", "SIMUSER") for suffix in ("URL", "MODEL"))


@pytest.mark.parametrize("some_success", [True, False])
def test_nonfake_command_uses_both_env_factories_and_exit_follows_success(
    monkeypatch, capsys, tmp_path, policy_path, scenario_factory, client_factory, some_success,
):
    from loop import llm
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text("\n".join(scenario_factory(sid).model_dump_json() for sid in ("a", "b")) + "\n",
                         encoding="utf-8")
    requested = []
    helper = client_factory(["I hear you."] if some_success else [])
    user = client_factory(["Worry", "[END]", "[END]"] if some_success else ["[END]", "[END]"])

    def factory(prefix):
        requested.append(prefix)
        return {"HELPER": helper, "SIMUSER": user}[prefix]

    monkeypatch.setattr(llm, "client_from_env", factory)
    monkeypatch.setattr(sys, "argv", ["loop.run_batch", "--policy", str(policy_path),
                        "--scenarios", str(scenarios), "--round", "5", "--seed", "13", "--max-turns", "1",
                        "--out", str(tmp_path / "records.jsonl")])
    monkeypatch.chdir(tmp_path)
    try:
        runpy.run_module("loop.run_batch", run_name="__main__")
    except SystemExit as caught:
        exit_code = caught.code
    else:
        exit_code = 0
    output = capsys.readouterr()
    assert exit_code == (0 if some_success else 1)
    assert sorted(requested) == ["HELPER", "SIMUSER"]
    assert len(output.out.splitlines()) == 1
    assert json.loads(output.out) == dict(round=5, conversations=int(some_success), failed=2 - int(some_success))
    from loop.runner import scenario_seed
    assert [call["seed"] for call in user.calls] == [scenario_seed(13, sid) for sid in ("a", "b")]
    if some_success:
        assert [call["seed"] for call in helper.calls] == [scenario_seed(13, "a")]
