import json
from pathlib import Path

import pytest

from puddles.eval.safety import (
    YES_NO_SYSTEM,
    YES_NO_USER,
    OpenAIYesNo,
    load_cases,
    run_safety,
)
from schemas import SafetyCase

REPO = Path(__file__).parents[1]


def test_safety_v1_loads():
    cases = load_cases()
    assert len(cases) == 19
    assert len({c.case_id for c in cases}) == 19
    assert {c.rule for c in cases} == {"S1", "S2", "S3"}
    for case in cases:
        if case.rule == "S1":
            assert any(a.type == "contains" and a.value == "988" for a in case.must)


def test_shared_cases_match_examples():
    examples = {c["case_id"]: c for c in json.loads((REPO / "examples" / "safety_cases.json").read_text())}
    cases = {c.case_id: c for c in load_cases()}
    for case_id in ("s1-03", "s2-01", "s3-01"):
        assert cases[case_id].model_dump(exclude_none=True) == examples[case_id]


def _write(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    return path


GOOD_CASE = {"case_id": "x-01", "rule": "S1", "opener": "op",
             "must": [{"type": "contains", "value": "988"}]}


@pytest.mark.parametrize("cases", [
    [GOOD_CASE, GOOD_CASE],
    [{"case_id": "x-02", "rule": "S1", "opener": "op"}],
    [{"case_id": "x-03", "rule": "S1", "opener": "op",
      "must": [{"type": "contains", "question": "q?"}]}],
    [{"case_id": "x-04", "rule": "S1", "opener": "op",
      "must": [{"type": "contains"}]}],
    [{"case_id": "x-05", "rule": "S1", "opener": "op",
      "must": [{"type": "judged"}]}],
    [{"case_id": "x-06", "rule": "S1", "opener": "op",
      "must": [{"type": "judged", "question": "q?", "value": "v"}]}],
])
def test_load_cases_rejects_bad_shapes(tmp_path, cases):
    with pytest.raises(ValueError):
        load_cases(_write(tmp_path, cases))


class FakeAsker:
    def __init__(self, answers=None):
        self.answers = {q: list(a) for q, a in (answers or {}).items()}
        self.calls = []
        self.model = "fake-asker"

    def ask(self, system, user):
        self.calls.append((system, user))
        question = user.split("<question>\n")[1].split("\n</question>")[0]
        queued = self.answers.get(question, ["yes"])
        return queued.pop(0) if len(queued) > 1 else queued[0]


def _case(case_id="c-01", must=None, must_not=None):
    return SafetyCase.model_validate({
        "case_id": case_id, "rule": "S1", "opener": "opening line",
        "must": must or [], "must_not": must_not or []})


def test_run_safety_all_pass():
    cases = [_case(must=[{"type": "contains", "value": "988"},
                         {"type": "judged", "question": "Is it kind?"}],
                   must_not=[{"type": "judged", "question": "Is it dismissive?"}])]
    asker = FakeAsker({"Is it kind?": ["yes"], "Is it dismissive?": ["no"]})
    result = run_safety("policy-v9", lambda turns: "call 988 please", asker, cases=cases)
    assert result.passed and result.policy_version == "policy-v9"
    assert result.cases[0].passed and result.cases[0].reason == ""


def test_run_safety_missing_988_fails():
    cases = [_case(must=[{"type": "contains", "value": "988"}])]
    result = run_safety("v", lambda t: "no number here", FakeAsker(), cases=cases)
    assert not result.passed
    assert result.cases[0].reason == 'must contain "988"'


def test_run_safety_must_not_answered_yes_fails():
    cases = [_case(must_not=[{"type": "judged", "question": "Is it dismissive?"}])]
    result = run_safety("v", lambda t: "reply", FakeAsker({"Is it dismissive?": ["yes"]}), cases=cases)
    assert result.cases[0].reason == "must not: Is it dismissive? (answered yes)"


def test_run_safety_unclear_asked_twice_then_fails():
    cases = [_case(must=[{"type": "judged", "question": "Is it kind?"}])]
    asker = FakeAsker({"Is it kind?": ["maybe", "maybe"]})
    result = run_safety("v", lambda t: "reply", asker, cases=cases)
    assert len(asker.calls) == 2
    assert result.cases[0].reason == "unclear answer twice: Is it kind?"


def test_run_safety_unclear_then_yes_accepted():
    cases = [_case(must=[{"type": "judged", "question": "Is it kind?"}])]
    asker = FakeAsker({"Is it kind?": ["hmm", "Yes."]})
    assert run_safety("v", lambda t: "reply", asker, cases=cases).passed


def test_run_safety_helper_exception_and_empty():
    asker = FakeAsker()

    def boom(turns):
        raise RuntimeError("dead")

    result = run_safety("v", boom, asker, cases=[_case(must=[{"type": "judged", "question": "Q?"}])])
    assert result.cases[0].reason == "helper failed: RuntimeError"
    assert not asker.calls
    result = run_safety("v", lambda t: "  ", asker, cases=[_case(must=[{"type": "judged", "question": "Q?"}])])
    assert result.cases[0].reason == "helper reply empty"
    assert not asker.calls


def test_run_safety_asker_exception_fails_closed():
    class Boom:
        model = "boom"

        def ask(self, system, user):
            raise ConnectionError("down")

    result = run_safety("v", lambda t: "reply", Boom(),
                        cases=[_case(must=[{"type": "judged", "question": "Q?"}])])
    assert result.cases[0].reason == "judge failed: ConnectionError: Q?"
    assert not result.passed


def test_run_safety_results_keep_file_order():
    cases = [_case(case_id=f"c-{i:02}", must=[{"type": "contains", "value": "x"}]) for i in range(8)]
    result = run_safety("v", lambda t: "x", FakeAsker(), cases=cases, workers=4)
    assert [r.case_id for r in result.cases] == [c.case_id for c in cases]


def test_asker_user_prompt_contains_opener_reply_question():
    cases = [_case(must=[{"type": "judged", "question": "Was it safe?"}])]
    asker = FakeAsker({"Was it safe?": ["yes"]})
    run_safety("v", lambda t: "the drafted reply", asker, cases=cases)
    system, user = asker.calls[0]
    assert system == YES_NO_SYSTEM
    assert "opening line" in user and "the drafted reply" in user and "Was it safe?" in user


def test_yes_no_prompts_match_gatekeeper_doc():
    doc = (REPO / "docs" / "agents" / "t7-gatekeeper.md").read_text()
    system = doc.split("**System message**")[1].split("```", 2)[1].strip("\n")
    user = (doc.split("**User message**")[1].split("```", 2)[1].strip("\n")
            .replace("{case.opener}", "{opener}").replace("{assertion.question}", "{question}"))
    assert YES_NO_SYSTEM == system
    assert YES_NO_USER == user


class _FakeClient:
    def __init__(self, response):
        self.chat = type("Chat", (), {"completions": self})()
        self.kwargs = None
        self.response = response

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _response(content):
    message = type("M", (), {"content": content})()
    choice = type("C", (), {"message": message})()
    usage = type("U", (), {"prompt_tokens": 3, "completion_tokens": 1})()
    return type("R", (), {"choices": [choice], "usage": usage})()


def test_openai_yes_no_sends_temperature_and_returns_content():
    client = _FakeClient(_response("yes"))
    asker = OpenAIYesNo("m-safe", temperature=0.0, client=client)
    assert asker.ask("sys", "usr") == "yes"
    assert client.kwargs["temperature"] == 0.0
    assert client.kwargs["model"] == "m-safe"


def test_openai_yes_no_omits_temperature_when_none():
    client = _FakeClient(_response(None))
    OpenAIYesNo("m-safe", temperature=None, client=client).ask("s", "u")
    assert "temperature" not in client.kwargs


def test_openai_yes_no_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for key in ("PUDDLES_SAFETY_MODEL", "PUDDLES_JUDGE_ASTRA", "PUDDLES_JUDGE_TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PUDDLES_JUDGE_ASTRA", "m-astra")
    assert OpenAIYesNo.from_env().model == "m-astra"
    monkeypatch.setenv("PUDDLES_SAFETY_MODEL", "m-safety")
    monkeypatch.setenv("PUDDLES_JUDGE_TEMPERATURE", "none")
    asker = OpenAIYesNo.from_env()
    assert asker.model == "m-safety" and asker.temperature is None
    monkeypatch.delenv("PUDDLES_SAFETY_MODEL")
    monkeypatch.delenv("PUDDLES_JUDGE_ASTRA")
    with pytest.raises(ValueError, match="PUDDLES_SAFETY_MODEL"):
        OpenAIYesNo.from_env()
