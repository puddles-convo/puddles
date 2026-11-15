import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from loop.errors import ConversationFailed
from loop.llm import FakeClient
from loop.policy import Policy
from loop.runner import load_scenarios, run_batch, run_conversation, scenario_seed
from schemas import Scenario

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
POLICY = Policy(version="policy-x", text="# policy-x\n\n## Safety\nS1. Be safe.\n",
                safety_text="## Safety\nS1. Be safe.", rules={"S1": "Be safe."})


def _scenario(sid="fixture-0001"):
    return Scenario(scenario_id=sid, split="validation", emotion_type="anxiety",
                    problem_type="job crisis", situation="Worried.", initial_intensity=4)


def _user(lines):
    """Simulated user that says each line in turn, then [END]."""
    def reply(system, messages):
        spoken = sum(m["role"] == "assistant" for m in messages)
        return lines[spoken] if spoken < len(lines) else "[END]"
    return FakeClient(reply)


def _helper():
    return FakeClient(lambda system, messages: f"Reply {len(messages)}.")


def test_scenario_seed_is_stable_bounded_and_scenario_specific():
    assert scenario_seed(None, "a") is None
    seed_a = scenario_seed(7, "a")
    assert seed_a == scenario_seed(7, "a")
    assert 0 <= seed_a < 2**31
    assert seed_a != scenario_seed(7, "b")
    assert seed_a != scenario_seed(8, "a")


def test_scenario_seed_is_the_same_in_another_process():
    code = "from loop.runner import scenario_seed; print(scenario_seed(7, 'fixture-0001'))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, check=True, env={"PYTHONHASHSEED": "123"})
    assert int(out.stdout) == scenario_seed(7, "fixture-0001")


def test_load_scenarios_reads_fixture():
    scenarios = load_scenarios(ROOT / "fixtures" / "scenarios.jsonl")
    assert [s.scenario_id for s in scenarios] == [f"fixture-000{i}" for i in range(1, 6)]
    assert {s.split for s in scenarios} == {"validation"}
    assert {"guarded", "resistant"} <= {s.openness for s in scenarios}


def test_load_scenarios_skips_blank_lines_and_validates(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("\n" + _scenario().model_dump_json() + "\n\n", encoding="utf-8")
    assert load_scenarios(path) == [_scenario()]
    path.write_text('{"scenario_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_scenarios(path)


def test_run_conversation_alternates_until_user_ends():
    helper, user = _helper(), _user(["one", "two"])
    record = run_conversation(helper, user, POLICY, _scenario(), round=3, seed=7, now=NOW)
    assert [(t.speaker, t.text) for t in record.turns] == [
        ("user", "one"), ("helper", "Reply 1."), ("user", "two"), ("helper", "Reply 3."),
    ]
    assert record.id == "conv-r03-fixture-0001"
    assert record.written_by == "helper"
    assert record.created_at == NOW
    assert record.source == "simulated"
    assert record.scenario_id == "fixture-0001"
    assert record.split == "validation"
    assert record.policy_version == "policy-x"
    expected_seed = scenario_seed(7, "fixture-0001")
    assert {c["seed"] for c in helper.calls + user.calls} == {expected_seed}


def test_run_conversation_stops_after_max_turns_helper_replies():
    helper, user = _helper(), _user(["a", "b", "c", "d"])
    record = run_conversation(helper, user, POLICY, _scenario(), round=1, max_turns=2, now=NOW)
    assert [t.speaker for t in record.turns] == ["user", "helper", "user", "helper"]
    assert len(user.calls) == 2


def test_run_conversation_with_zero_max_turns_has_one_user_line():
    helper = _helper()
    record = run_conversation(helper, _user(["a", "b"]), POLICY, _scenario(), round=1,
                              max_turns=0, now=NOW)
    assert [t.speaker for t in record.turns] == ["user"]
    assert helper.calls == []


def test_run_conversation_uses_record_id_and_turn_count_for_helper(monkeypatch):
    import loop.runner as runner_module
    seen = []
    original = runner_module.helper_reply

    def spy(*args, **kwargs):
        seen.append((kwargs["session_id"], kwargs["turn"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "helper_reply", spy)
    run_conversation(_helper(), _user(["a", "b"]), POLICY, _scenario(), round=1, now=NOW)
    assert seen == [("conv-r01-fixture-0001", 1), ("conv-r01-fixture-0001", 2)]


def test_created_at_defaults_to_current_utc_time():
    record = run_conversation(_helper(), _user(["a"]), POLICY, _scenario(), round=1)
    assert record.created_at.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - record.created_at).total_seconds()) < 60


@pytest.mark.parametrize("helper, user", [
    (_helper(), _user([])),                       # ends before saying anything
    (FakeClient([]), _user(["a"])),               # helper LLMError
    (FakeClient(["", ""]), _user(["a"])),         # HelperError
    (_helper(), FakeClient(["", ""])),            # SimulatedUserError
    (_helper(), FakeClient([])),                  # simulated user LLMError
])
def test_run_conversation_failures_raise_conversation_failed(helper, user):
    with pytest.raises(ConversationFailed) as info:
        run_conversation(helper, user, POLICY, _scenario(), round=1, now=NOW)
    assert info.value.scenario_id == "fixture-0001"
    assert info.value.reason


def test_run_batch_continues_when_http_body_is_cut_short(monkeypatch):
    import http.client
    import loop.llm as llm_module
    from loop.llm import OpenAICompatClient

    class CutShort:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise http.client.IncompleteRead(b'{"choi', 40)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", lambda request, timeout: CutShort())
    helper = OpenAICompatClient("http://host/v1", "m")
    result = run_batch(helper, _user(["a"]), POLICY, [_scenario("one"), _scenario("two")],
                       round=1, now=NOW)
    assert result.records == []
    assert [f.scenario_id for f in result.failures] == ["one", "two"]
    assert all("LLMError" in f.reason for f in result.failures)


def _flaky_user(fail_id):
    def reply(system, messages):
        if fail_id in system:
            return ""
        spoken = sum(m["role"] == "assistant" for m in messages)
        return "Hi." if spoken == 0 else "[END]"
    return FakeClient(reply)


def _named(sid):
    s = _scenario(sid)
    return s.model_copy(update={"situation": f"Situation of {sid}."})


def test_run_batch_continues_after_failure_and_writes_files(tmp_path):
    scenarios = [_named("ok-1"), _named("bad-2"), _named("ok-3")]
    out = tmp_path / "runs" / "round-02" / "conversations.jsonl"
    result = run_batch(_helper(), _flaky_user("bad-2"), POLICY, scenarios, round=2,
                       out_path=out, now=NOW)
    assert [r.scenario_id for r in result.records] == ["ok-1", "ok-3"]
    assert [(f.scenario_id, bool(f.reason)) for f in result.failures] == [("bad-2", True)]
    assert result.summary() == {"round": 2, "conversations": 2, "failed": 1}
    lines = out.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["conv-r02-ok-1", "conv-r02-ok-3"]
    failures = (out.parent / "conversations.failures.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in failures] == [
        {"scenario_id": "bad-2", "reason": result.failures[0].reason}]


def test_run_batch_without_failures_writes_no_failures_file(tmp_path):
    out = tmp_path / "conversations.jsonl"
    result = run_batch(_helper(), _user(["a"]), POLICY, [_scenario()], round=1,
                       out_path=out, now=NOW)
    assert result.summary() == {"round": 1, "conversations": 1, "failed": 0}
    assert len(out.read_text().splitlines()) == 1
    assert not (tmp_path / "conversations.failures.jsonl").exists()


def test_run_batch_overwrites_records_and_deletes_stale_failures_file(tmp_path):
    out = tmp_path / "conversations.jsonl"
    stale = tmp_path / "conversations.failures.jsonl"
    out.write_text("old line 1\nold line 2\nold line 3\n", encoding="utf-8")
    stale.write_text('{"scenario_id": "old", "reason": "earlier run"}\n', encoding="utf-8")
    run_batch(_helper(), _user(["a"]), POLICY, [_scenario()], round=1, out_path=out, now=NOW)
    lines = out.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["conv-r01-fixture-0001"]
    assert not stale.exists()


def test_run_batch_with_failures_replaces_earlier_failures_file(tmp_path):
    out = tmp_path / "conversations.jsonl"
    stale = tmp_path / "conversations.failures.jsonl"
    stale.write_text('{"scenario_id": "old", "reason": "earlier run"}\n', encoding="utf-8")
    run_batch(_helper(), _flaky_user("bad-1"), POLICY, [_named("bad-1")], round=1,
              out_path=out, now=NOW)
    assert [json.loads(line)["scenario_id"] for line in stale.read_text().splitlines()] == ["bad-1"]


def test_run_batch_without_out_path_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_batch(_helper(), _user(["a"]), POLICY, [_scenario()], round=1, now=NOW)
    assert list(tmp_path.iterdir()) == []


def test_a_tag_goes_into_the_conversation_id():
    from loop.runner import conversation_id

    assert conversation_id(3, "s1") == "conv-r03-s1"
    assert conversation_id(3, "s1", "val") == "conv-r03-val-s1"
    record = run_conversation(_helper(), _user(["Hello."]), POLICY, _scenario(), round=3, now=NOW, tag="val")
    assert record.id == "conv-r03-val-fixture-0001" and record.scenario_id == "fixture-0001"
    untagged = run_conversation(_helper(), _user(["Hello."]), POLICY, _scenario(), round=3, now=NOW, tag=None)
    assert untagged.id == "conv-r03-fixture-0001"


def test_a_tagged_batch_shares_no_id_with_the_untagged_batch_of_the_same_round(tmp_path):
    scenarios = [_scenario("fixture-0001"), _scenario("fixture-0002")]
    train = run_batch(_helper(), _user(["Hello."]), POLICY, scenarios, round=1, now=NOW,
                      out_path=tmp_path / "round-01-train.jsonl")
    validation = run_batch(_helper(), _user(["Hello."]), POLICY, scenarios, round=1, now=NOW,
                           out_path=tmp_path / "round-01-validation.jsonl", tag="val")
    assert [r.id for r in validation.records] == ["conv-r01-val-fixture-0001", "conv-r01-val-fixture-0002"]
    assert not {r.id for r in train.records} & {r.id for r in validation.records}
    assert len((tmp_path / "round-01-train.jsonl").read_text().splitlines()) == 2  # not overwritten


@pytest.mark.parametrize("tag", ["", "Val", "val-1", "1val", "va l", "val\n"])
def test_a_bad_tag_is_refused_before_any_conversation_runs(tag):
    from loop.runner import conversation_id

    helper, user = _helper(), _user(["Hello."])
    with pytest.raises(ValueError, match="tag"):
        run_batch(helper, user, POLICY, [_scenario()], round=1, tag=tag)
    assert helper.calls == [] and user.calls == []
    with pytest.raises(ValueError, match="tag"):
        conversation_id(1, "s1", tag)
