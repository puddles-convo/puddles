"""End-to-end runner tests use independent scripted chat clients."""

from datetime import datetime, timezone
import importlib
import json
import os
import subprocess
import sys

import pytest


@pytest.fixture
def api():
    return importlib.import_module("loop.runner")


def test_scenario_seed_none_range_and_repeatability_for_multiple_ids(api):
    assert api.scenario_seed(None, "a") is None
    values = [api.scenario_seed(7, name) for name in ("a", "b", "café", "a:b")]
    assert all(isinstance(value, int) and 0 <= value < 2**31 for value in values)
    # The amended contract explicitly permits chance collisions.
    assert api.scenario_seed(7, "a") == values[0]


def test_scenario_seed_is_stable_across_process_hash_randomization(api, repo_root, tmp_path):
    code = "from loop.runner import scenario_seed; print(scenario_seed(42, 'café-id'))"
    values = []
    for hash_seed in ("1", "987"):
        env = os.environ.copy()
        env.update(PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(tmp_path))
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=repo_root, env=env,
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        values.append(int(result.stdout.strip()))
    assert values == [api.scenario_seed(42, "café-id")] * 2


def test_scenario_jsonl_skips_blanks_preserves_order_and_validates(api, tmp_path, scenario_factory):
    scenarios = [scenario_factory("two"), scenario_factory("one", openness="resistant", from_session="live-origin")]
    path = tmp_path / "scenarios.jsonl"
    path.write_text("\n" + scenarios[0].model_dump_json() + "\n  \n" + scenarios[1].model_dump_json() + "\n",
                    encoding="utf-8")
    loaded = api.load_scenarios(path)
    assert loaded == scenarios


def test_shipped_scenarios_meet_fixture_contract(api, repo_root):
    scenarios = api.load_scenarios(repo_root / "fixtures/scenarios.jsonl")
    assert [s.scenario_id for s in scenarios] == [f"fixture-{i:04d}" for i in range(1, 6)]
    assert {s.split for s in scenarios} == {"validation"}
    assert {"guarded", "resistant"} <= {s.openness for s in scenarios}


def test_scenario_loader_uses_schema_defaults(api, tmp_path, scenario_factory):
    data = scenario_factory().model_dump()
    del data["openness"]
    del data["from_session"]
    path = tmp_path / "defaults.jsonl"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    loaded = api.load_scenarios(path)
    assert len(loaded) == 1
    assert loaded[0].openness == "open"
    assert loaded[0].from_session is None


@pytest.mark.parametrize("changes", [{"initial_intensity": 6}, {"openness": "eager"},
                                      {"split": "unknown"}, {"unexpected": True}])
def test_invalid_scenario_is_not_silently_accepted(api, tmp_path, scenario_factory, changes):
    data = scenario_factory().model_dump()
    data.update(changes)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    # The spec does not prescribe the exception class at this boundary.
    try:
        api.load_scenarios(path)
    except Exception:
        pass
    else:
        pytest.fail("Invalid Scenario was accepted")


@pytest.mark.parametrize("split", ["train", "validation", "test", "live"])
def test_conversation_order_histories_seeds_and_record_metadata(api, policy, scenario_factory, client_factory, split):
    from schemas import ConversationRecord
    order = []
    user = client_factory(["My hours were cut.", "I am worried about rent.", "[END]"], name="user", order=order)
    helper = client_factory(["That sounds hard.", "What worries you most?"], name="helper", order=order)
    now = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    scenario = scenario_factory("sample", split=split)
    record = api.run_conversation(helper, user, policy, scenario, round=3, seed=17, now=now)
    assert isinstance(record, ConversationRecord)
    assert (record.id, record.written_by, record.created_at, record.source, record.scenario_id,
            record.split, record.policy_version) == (
        "conv-r03-sample", "helper", now, "simulated", "sample", split, policy.version,
    )
    assert [(t.speaker, t.text) for t in record.turns] == [
        ("user", "My hours were cut."), ("helper", "That sounds hard."),
        ("user", "I am worried about rent."), ("helper", "What worries you most?"),
    ]
    assert order == ["user", "helper", "user", "helper", "user"]
    assert helper.calls[1]["messages"] == [
        {"role": "user", "content": "My hours were cut."},
        {"role": "assistant", "content": "That sounds hard."},
        {"role": "user", "content": "I am worried about rent."},
    ]
    assert user.calls[1]["messages"] == [
        {"role": "user", "content": "(The helper is listening. Begin.)"},
        {"role": "assistant", "content": "My hours were cut."},
        {"role": "user", "content": "That sounds hard."},
    ]
    assert {call["seed"] for call in helper.calls + user.calls} == {api.scenario_seed(17, "sample")}


@pytest.mark.parametrize("max_turns", [1, 3, 10])
def test_stops_immediately_after_helper_limit_without_extra_user_call(api, policy, scenario_factory, client_factory, max_turns):
    user = client_factory([f"Worry {i}" for i in range(max_turns)])
    helper = client_factory(["I hear you."] * max_turns)
    record = api.run_conversation(helper, user, policy, scenario_factory(), round=1, max_turns=max_turns)
    assert [t.speaker for t in record.turns] == ["user", "helper"] * max_turns
    assert len(helper.calls) == len(user.calls) == max_turns


def test_default_limit_is_ten_helper_replies(api, policy, scenario_factory, client_factory):
    user = client_factory(["Still worried"] * 10)
    helper = client_factory(["I hear you."] * 10)
    record = api.run_conversation(helper, user, policy, scenario_factory(), round=123)
    assert record.id == "conv-r123-acceptance-01"
    assert len(record.turns) == 20
    assert len(helper.calls) == len(user.calls) == 10
    assert all(call["seed"] is None for call in helper.calls + user.calls)


def test_zero_helper_limit_never_calls_helper(api, policy, scenario_factory, client_factory):
    helper = client_factory([])
    user = client_factory(["Worried"])
    record = api.run_conversation(helper, user, policy, scenario_factory(), round=1, max_turns=0)
    assert helper.calls == []
    # The spec permits an unanswered user turn here but does not require one.
    assert len(user.calls) <= 1
    assert all(turn.speaker == "user" for turn in record.turns)
    assert len(record.turns) <= 1


def test_runner_passes_record_id_and_one_based_helper_turn_numbers(repo_root, tmp_path, policy_path, scenario_factory):
    # Import the runner after spying on the specified public helper function in
    # a fresh process. This avoids depending on a private runner import alias.
    # This requires loop.__init__ not to import loop.runner before the patch.
    code = '''
import json
import sys
from unittest.mock import patch
from loop.helper import helper_reply
from loop.policy import load_policy
from loop.llm import FakeClient
from schemas import Scenario
with patch("loop.helper.helper_reply", wraps=helper_reply) as observed:
    from loop.runner import run_conversation
    record = run_conversation(
        FakeClient(["I hear you.", "Tell me more."]),
        FakeClient(["Worried", "Still worried"]),
        load_policy(sys.argv[1]), Scenario.model_validate_json(sys.argv[2]),
        round=6, max_turns=2,
    )
    print(json.dumps({"id": record.id, "calls": [
        {"session_id": call.kwargs["session_id"], "turn": call.kwargs["turn"]}
        for call in observed.call_args_list
    ]}))
'''
    env = os.environ.copy()
    env.update(PYTHONPATH=str(repo_root), PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-B", "-c", code, str(policy_path), scenario_factory().model_dump_json()],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.splitlines()[-1])
    assert data["calls"] == [{"session_id": data["id"], "turn": 1}, {"session_id": data["id"], "turn": 2}]


def test_omitted_now_is_current_utc(api, policy, scenario_factory, client_factory):
    before = datetime.now(timezone.utc)
    record = api.run_conversation(client_factory(["I hear you."]), client_factory(["Worried", "[END]"]),
                                  policy, scenario_factory(), round=1)
    after = datetime.now(timezone.utc)
    assert record.created_at.utcoffset() == timezone.utc.utcoffset(record.created_at)
    assert before.replace(microsecond=0) <= record.created_at <= after


@pytest.mark.parametrize("failure", ["initial_end", "user_transport", "helper_transport", "user_invalid", "helper_empty"])
def test_conversation_failures_are_attributed_and_never_return_partial_record(api, policy, scenario_factory, client_factory, failure):
    from loop.errors import ConversationFailed, LLMError
    user_replies = {
        "initial_end": ["[END]"], "user_transport": ["Worried", LLMError("offline")],
        "helper_transport": ["Worried"], "user_invalid": ["", "As an AI"],
        "helper_empty": ["Worried"],
    }[failure]
    helper_replies = {"helper_transport": [LLMError("offline")], "helper_empty": ["", ""]}.get(failure, ["I hear you."])
    with pytest.raises(ConversationFailed) as caught:
        api.run_conversation(client_factory(helper_replies), client_factory(user_replies), policy,
                             scenario_factory("failed-id"), round=1)
    assert caught.value.scenario_id == "failed-id"
    assert isinstance(caught.value.reason, str) and caught.value.reason.strip()


def test_batch_continues_in_order_writes_only_successes_and_separate_failures(api, tmp_path, policy, scenario_factory, client_factory):
    from schemas import ConversationRecord
    scenarios = [scenario_factory("a"), scenario_factory("b"), scenario_factory("c")]
    user = client_factory(["Worry A", "[END]", "[END]", "Worry C", "[END]"])
    helper = client_factory(["Heard A.", "Heard C."])
    out = tmp_path / "nested" / "records.data.jsonl"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = api.run_batch(helper, user, policy, scenarios, round=2, seed=9, out_path=out, now=now)
    assert [r.scenario_id for r in result.records] == ["a", "c"]
    assert [f.scenario_id for f in result.failures] == ["b"]
    assert result.failures[0].reason.strip()
    assert result.summary() == {"round": 2, "conversations": 2, "failed": 1}
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [ConversationRecord.model_validate_json(line) for line in lines] == result.records
    assert all(r.created_at == now for r in result.records)
    failure_lines = out.with_suffix(".failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in failure_lines] == [
        {"scenario_id": "b", "reason": result.failures[0].reason},
    ]
    assert [call["seed"] for call in user.calls] == [
        api.scenario_seed(9, sid) for sid in ("a", "a", "b", "c", "c")
    ]


def test_batch_discards_partial_failed_conversation_and_carries_on(api, tmp_path, policy, scenario_factory, client_factory):
    from loop.errors import LLMError
    user = client_factory(["First worry", "More worry", "Another person", "[END]"])
    helper = client_factory(["I hear you.", LLMError("failed halfway"), "I hear you too."])
    out = tmp_path / "records.jsonl"
    result = api.run_batch(helper, user, policy, [scenario_factory("bad"), scenario_factory("good")],
                           round=1, out_path=out)
    assert [r.scenario_id for r in result.records] == ["good"]
    assert [f.scenario_id for f in result.failures] == ["bad"]
    assert [json.loads(line)["scenario_id"] for line in out.read_text().splitlines()] == ["good"]


def test_successful_batch_creates_no_failure_file(api, tmp_path, policy, scenario_factory, client_factory):
    out = tmp_path / "records.jsonl"
    out.with_suffix(".failures.jsonl").write_text('{"scenario_id":"old","reason":"stale"}\n', encoding="utf-8")
    result = api.run_batch(client_factory(["Okay."]), client_factory(["Worry", "[END]"]), policy,
                           [scenario_factory()], round=1, out_path=out)
    assert result.summary() == {"round": 1, "conversations": 1, "failed": 0}
    assert not out.with_suffix(".failures.jsonl").exists()


def test_all_failed_batch_writes_no_records(api, tmp_path, policy, scenario_factory, client_factory):
    out = tmp_path / "records.jsonl"
    previous = api.run_batch(client_factory(["Okay."]), client_factory(["Worry"]), policy,
                             [scenario_factory("old")], round=1, max_turns=1, out_path=out)
    assert len(previous.records) == 1
    assert len(out.read_text().splitlines()) == 1
    result = api.run_batch(client_factory([]), client_factory(["[END]", "[END]"]), policy,
                           [scenario_factory("a"), scenario_factory("b")], round=1, out_path=out)
    assert result.records == []
    assert result.summary() == {"round": 1, "conversations": 0, "failed": 2}
    assert not out.exists() or not out.read_text().strip()
    assert len(out.with_suffix(".failures.jsonl").read_text().splitlines()) == 2


def test_each_batch_replaces_records_and_failures_from_previous_run(api, tmp_path, policy, scenario_factory, client_factory):
    out = tmp_path / "records.jsonl"
    for number in (1, 2):
        result = api.run_batch(client_factory(["Okay."]), client_factory(["Worry", "[END]"]), policy,
                               [scenario_factory(f"good-{number}"), scenario_factory(f"bad-{number}")],
                               round=number, max_turns=1, out_path=out)
        assert len(out.read_text().splitlines()) == 1
        assert json.loads(out.read_text())["scenario_id"] == f"good-{number}"
        failures = [json.loads(line) for line in out.with_suffix(".failures.jsonl").read_text().splitlines()]
        assert failures == [{"scenario_id": f"bad-{number}", "reason": result.failures[0].reason}]


def test_batch_without_output_path_returns_records(api, policy, scenario_factory, client_factory):
    result = api.run_batch(client_factory(["Okay."]), client_factory(["Worry", "[END]"]), policy,
                           [scenario_factory()], round=1)
    assert [r.scenario_id for r in result.records] == ["acceptance-01"]
