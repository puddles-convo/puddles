"""docs/specs/unattended-run.md section 3: the judge and the agents. All offline: no test calls a server."""
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion, ParsedChatCompletion

from puddles import policy as policy_module
from puddles.agents import critic, proposer, researcher
from puddles.decision_log import DecisionLog
from puddles.eval.judge import FallbackConfig, JudgeFailed, SingleJudge
from puddles.llm import AgentFailed, OpenAILLM
from puddles.memory import FakeMemory
from puddles.policy import PROTECTED_RULE_IDS, PolicyEditError, apply_edit, parse_policy
from puddles.usage import Meter, metering, read_usage
from schemas import (
    ConversationRecord,
    CriticOutput,
    JudgeOutput,
    ProposerOutput,
    ResearcherOutput,
    ScoredConversation,
    Turn,
)
from tests.fakes import ScriptedLLM

FIXTURES = Path(__file__).parents[1] / "fixtures"
CONVS = [ScoredConversation.model_validate(c) for c in json.loads((FIXTURES / "bad_conversations.json").read_text())]
POLICY = (FIXTURES / "policy-v3.md").read_text()
MODEL = "agent-model"
QUERY = "counselling skills exploring before giving advice"
URL = "https://counselling.example.ac.uk/skills"
PATTERN = {"found": True, "name": "advice-before-understanding", "description": "Advises before asking.",
           "seen_in": 2, "out_of": 4, "example_conversation_ids": ["conv-hand-001", "conv-hand-002"]}


def setup(tmp_path):
    mem = FakeMemory()
    mem.ensure_training_audience()
    return DecisionLog(tmp_path), mem


# ---------- 3.1 and 3.2: the single-model judge ----------

class FakeJudgeLLM:
    def __init__(self, outputs):
        self.model = "m-fb"
        self.outputs = list(outputs)
        self.calls = []

    def judge(self, system, user):
        self.calls.append((system, user))
        return self.outputs.pop(0), 20, 5


def two_helper_turns() -> ConversationRecord:
    turns = [Turn(speaker="user", text="hi"), Turn(speaker="helper", text="hello"),
             Turn(speaker="user", text="I am sad"), Turn(speaker="helper", text="That sounds hard.")]
    return ConversationRecord(id="conv-1", written_by="t", created_at=datetime.now(timezone.utc), parent_ids=[],
                              source="real", scenario_id="sc-1", split="validation", policy_version="policy-v1",
                              turns=turns)


def judged(strategies, **extra) -> JudgeOutput:
    return JudgeOutput.model_validate({"empathy": 3, "relevance": 4, "emotion_before": 4, "emotion_after": 3,
                                       "strategies": list(strategies), "note": "a note", **extra})


@pytest.mark.parametrize("labels", [
    ["Turn 2: Reflection of feelings", "Turn 4: Reflection of feelings"],  # the observed real answer
    ["2. Question", "4: question"],
    ["Helper turn 1 - Self-disclosure", "  'Turn 2: \"providing suggestions\"' "],
    ['"Reflection of feelings"', " restatement or paraphrasing "],
])
def test_single_judge_normalises_labels_before_the_check(labels):
    llm = FakeJudgeLLM([judged(labels)])
    result = SingleJudge(FallbackConfig(model="m-fb"), llm).score(two_helper_turns())
    assert len(llm.calls) == 1  # accepted at the first answer: no retry
    canonical = {"reflection of feelings": "Reflection of feelings", "question": "Question",
                 "self-disclosure": "Self-disclosure", "providing suggestions": "Providing Suggestions",
                 "restatement or paraphrasing": "Restatement or Paraphrasing"}
    expected = [next(v for k, v in canonical.items() if k in label.lower()) for label in labels]
    assert result.score.strategies == expected


def test_a_label_that_still_matches_nothing_fails_as_before():
    bad = judged(["Turn 2: Mirroring", "Question"])
    llm = FakeJudgeLLM([bad, bad])
    with pytest.raises(JudgeFailed, match="unknown strategy label 'Turn 2: Mirroring'"):
        SingleJudge(FallbackConfig(model="m-fb"), llm).score(two_helper_turns())
    assert len(llm.calls) == 2


def test_single_judge_keeps_safety_flags_and_evidence():
    out = judged(["Question", "Reflection of feelings"], evidence="Turn 3 mentions not being around.",
                 safety_flags=["missed_risk", "role_break"])
    result = SingleJudge(FallbackConfig(model="m-fb"), FakeJudgeLLM([out])).score(two_helper_turns())
    assert result.score.safety_flags == ["missed_risk", "role_break"]
    assert result.score.evidence == "Turn 3 mentions not being around."


def test_single_judge_without_flags_records_none():
    result = SingleJudge(FallbackConfig(model="m-fb"),
                         FakeJudgeLLM([judged(["Question", "Question"])])).score(two_helper_turns())
    assert result.score.safety_flags == [] and result.score.evidence == ""


# ---------- 3.3: C8 is protected only with PUDDLES_PROTECT_C8=1 ----------

C8_EDITS = [
    {"action": "remove", "rule_id": "C8", "summary": "s"},
    {"action": "reword", "rule_id": "C8", "rule_text": "Keep listening.", "summary": "s"},
]


@pytest.fixture
def protect_c8(monkeypatch):
    monkeypatch.setenv("PUDDLES_PROTECT_C8", "1")


@pytest.fixture(params=["unset", "0"])
def c8_editable(request, monkeypatch):
    if request.param == "unset":
        monkeypatch.delenv("PUDDLES_PROTECT_C8", raising=False)
    else:
        monkeypatch.setenv("PUDDLES_PROTECT_C8", request.param)


def proposer_setup(tmp_path):
    log, mem = setup(tmp_path)
    inp = critic.build_input(1, POLICY, CONVS, log, mem, tmp_path)
    pattern = critic.to_record(CriticOutput.model_validate(PATTERN), inp, log)
    research = researcher.to_record(ResearcherOutput.model_validate(
        {"queries": ["q"], "sources": [{"url": "https://u", "technique": "Ask first."}]}), pattern, 1)
    return log, proposer.build_input(POLICY, pattern, research, log, mem, tmp_path)


def test_the_protected_set_is_written_in_one_place():
    assert "C8" in PROTECTED_RULE_IDS
    assert proposer.PROTECTED_RULE_IDS is policy_module.PROTECTED_RULE_IDS
    assert proposer.protected_rule_ids is policy_module.protected_rule_ids


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", " 1", "1 ", "2"])
def test_protection_is_off_unless_the_setting_is_exactly_1(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PUDDLES_PROTECT_C8", raising=False)
    else:
        monkeypatch.setenv("PUDDLES_PROTECT_C8", value)
    assert policy_module.protected_rule_ids() == frozenset()


def test_the_setting_is_read_at_each_check(monkeypatch):
    monkeypatch.setenv("PUDDLES_PROTECT_C8", "1")
    assert policy_module.protected_rule_ids() is PROTECTED_RULE_IDS
    monkeypatch.setenv("PUDDLES_PROTECT_C8", "0")
    assert policy_module.protected_rule_ids() == frozenset()


@pytest.mark.parametrize("answer", C8_EDITS, ids=["remove", "reword"])
def test_proposer_cannot_remove_or_reword_c8(tmp_path, protect_c8, answer):
    log, p_in = proposer_setup(tmp_path)
    policy = parse_policy(POLICY)
    problems = proposer.check(ProposerOutput.model_validate(answer), p_in, policy, log)
    assert "C8 is protected; choose another rule" in problems
    llm = ScriptedLLM()
    llm.add(ProposerOutput, answer, {"action": "reword", "rule_id": "C3", "rule_text": "Ask one question.",
                                     "summary": "s"})
    out = proposer.run(p_in, policy.version, policy, log, llm)
    assert out.rule_id == "C3" and "C8 is protected; choose another rule" in llm.prompts[ProposerOutput][1]


@pytest.mark.parametrize("action,text", [("remove", None), ("reword", "Keep listening.")])
def test_policy_edit_refuses_c8(protect_c8, action, text):
    policy = parse_policy(POLICY)
    with pytest.raises(PolicyEditError, match="rule C8 is protected"):
        apply_edit(policy, action, "C8", text, "policy-v4")


def test_other_conversation_rules_can_still_be_removed(tmp_path, protect_c8):
    log, p_in = proposer_setup(tmp_path)
    policy = parse_policy(POLICY)
    assert proposer.check(ProposerOutput.model_validate({"action": "remove", "rule_id": "C7", "summary": "s"}),
                          p_in, policy, log) == []
    candidate, _ = apply_edit(policy, "remove", "C7", None, "policy-v4")
    assert "C7" not in candidate.conversation and "C8" in candidate.conversation


@pytest.mark.parametrize("answer", C8_EDITS, ids=["remove", "reword"])
def test_by_default_the_proposer_may_remove_or_reword_c8(tmp_path, c8_editable, answer):
    log, p_in = proposer_setup(tmp_path)
    policy = parse_policy(POLICY)
    problems = proposer.check(ProposerOutput.model_validate(answer), p_in, policy, log)
    assert not any("protected" in p for p in problems)
    assert problems == []


def test_by_default_a_policy_edit_rewords_and_removes_c8(c8_editable):
    policy = parse_policy(POLICY)
    reworded, touched = apply_edit(policy, "reword", "C8", "Keep listening.", "policy-v4")
    assert touched == "C8" and reworded.conversation["C8"] == "Keep listening."
    removed, touched = apply_edit(policy, "remove", "C8", None, "policy-v4")
    assert touched == "C8" and "C8" not in removed.conversation


# ---------- 3.4: a malformed search-tool call ----------

def completion(message: dict, finish: str = "stop") -> dict:
    return {"id": "c", "object": "chat.completion", "created": 0, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def tool_turn(*calls, finish="tool_calls") -> ChatCompletion:
    return ChatCompletion.model_validate(completion({"content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        for call_id, name, arguments in calls]}, finish))


def no_tool_call() -> ChatCompletion:
    return ChatCompletion.model_validate(completion({"content": "done searching"}))


def parsed(value: dict):
    return ParsedChatCompletion[ResearcherOutput].model_validate(
        completion({"content": json.dumps(value), "parsed": value}))


class StandInClient:
    def __init__(self, create=(), parse=()):
        self.calls: list[tuple[str, dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._send("create", list(create)),
                                                                parse=self._send("parse", list(parse))))

    def _send(self, kind, answers):
        def send(**kwargs):
            self.calls.append((kind, {**kwargs, "messages": list(kwargs["messages"])}))
            return answers.pop(0)
        return send


def agent_llm(create=(), parse=()) -> OpenAILLM:
    llm = OpenAILLM(model=MODEL, output_limits={"default": 900})
    llm.client = StandInClient(create, parse)
    return llm


RESEARCH = {"queries": [QUERY], "sources": [{"url": URL, "title": "Skills", "technique": "Explore first."}]}
BAD_CALLS = {
    "not JSON": ("search", '{"query": "cut off'),
    "no query": ("search", '{"q": "x"}'),
    "blank query": ("search", '{"query": "   "}'),
    "non-string query": ("search", '{"query": 3}'),
    "null query": ("search", '{"query": null}'),
    "not an object": ("search", '["a query"]'),
    "another tool": ("browse", json.dumps({"query": QUERY})),
}


@pytest.fixture(autouse=True)
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("PUDDLES_AGENT_TOOL_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("PUDDLES_AGENT_TEMPERATURE", raising=False)


def tool_messages(llm: OpenAILLM) -> list[dict]:
    final = llm.client.calls[-1][1]["messages"]
    return [m for m in final if m["role"] == "tool"]


@pytest.mark.parametrize("name,arguments", BAD_CALLS.values(), ids=BAD_CALLS.keys())
def test_a_malformed_tool_call_gets_an_error_result_and_no_exception(name, arguments):
    searched = []
    llm = agent_llm(create=[tool_turn(("t1", name, arguments)), no_tool_call()], parse=[parsed(RESEARCH)])
    out = llm.structured_with_search("sys", "user", ResearcherOutput, lambda q: searched.append(q) or [], 3)
    assert out.queries == [QUERY] and searched == []
    [message] = tool_messages(llm)
    assert message["tool_call_id"] == "t1" and "error" in json.loads(message["content"])


def test_a_malformed_call_counts_toward_max_searches():
    searched = []
    llm = agent_llm(create=[tool_turn(("t1", "search", "{not json"), ("t2", "search", json.dumps({"query": QUERY})))],
                    parse=[parsed(RESEARCH)])
    llm.structured_with_search("sys", "user", ResearcherOutput, lambda q: searched.append(q) or [], 1)
    assert searched == []  # the bad call used the only search
    first, second = tool_messages(llm)
    assert first["tool_call_id"] == "t1" and "error" in json.loads(first["content"])
    assert second["tool_call_id"] == "t2" and json.loads(second["content"]) == {"error": "search limit reached"}
    assert [kind for kind, _ in llm.client.calls] == ["create", "parse"]


def test_a_good_call_after_a_bad_one_is_searched():
    searched = []
    llm = agent_llm(create=[tool_turn(("t1", "search", '{"query": ""}'), ("t2", "search", json.dumps({"query": QUERY}))),
                            no_tool_call()], parse=[parsed(RESEARCH)])
    llm.structured_with_search("sys", "user", ResearcherOutput,
                               lambda q: searched.append(q) or [{"url": URL, "title": "t", "text": "x"}], 3)
    assert searched == [QUERY]
    _, good = tool_messages(llm)
    assert good["tool_call_id"] == "t2" and json.loads(good["content"])["query"] == QUERY


@pytest.mark.parametrize("calls", [[("t1", "search", '{"query": "cut')], []], ids=["with a tool call", "without"])
def test_a_tool_turn_cut_off_at_the_output_limit_fails_the_researcher(calls):
    searched = []
    turn = (tool_turn(*calls, finish="length") if calls else
            ChatCompletion.model_validate(completion({"content": "thinking"}, "length")))
    llm = agent_llm(create=[turn])
    with pytest.raises(AgentFailed, match="cut off at the output limit"):
        llm.structured_with_search("sys", "user", ResearcherOutput, lambda q: searched.append(q) or [], 3)
    assert [kind for kind, _ in llm.client.calls] == ["create"] and searched == []


# ---------- 3.5: a failed search is counted ----------

def test_a_failed_search_writes_one_failed_line_for_the_researcher(tmp_path):
    log, mem = setup(tmp_path)
    inp = critic.build_input(1, POLICY, CONVS, log, mem, tmp_path)
    pattern = critic.to_record(CriticOutput.model_validate(PATTERN), inp, log)

    def exa_down(query):
        raise RuntimeError("Exa failed twice: 503")

    llm = ScriptedLLM()
    llm.add(ResearcherOutput, RESEARCH)
    meter = Meter(tmp_path, round=3)
    with metering(meter, "researcher"), pytest.raises(AgentFailed, match="the search failed") as caught:
        researcher.run(researcher.build_input(pattern, log, mem, tmp_path), llm, exa_down, log)
    failed = [line for line in read_usage(tmp_path / "log/usage.jsonl") if line.get("event") == "failed"]
    assert [(f["round"], f["agent"], f["reason"]) for f in failed] == [(3, "researcher", str(caught.value))]
