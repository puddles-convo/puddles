"""Offline tests for the case files and scripts/try_agent.py (docs/specs/case-files.md, section 3)."""
import copy
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from puddles import llm as llm_module
from puddles.agents import critic, proposer, researcher
from puddles.policy import at_length_limit, parse_policy
from schemas import CriticInput, CriticOutput, ProposerInput, ProposerOutput, ResearcherInput, ResearcherOutput
from scripts import try_agent
from tests.fakes import FakeSearch, ScriptedLLM

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "fixtures" / "cases"
FILES = {"critic": CriticInput, "researcher": ResearcherInput, "proposer": ProposerInput}


def load(agent):
    return json.loads((CASES / f"{agent}.json").read_text())


# Hand-written good answers that meet every case's expectations, in file order.
CRITIC_ANSWERS = [
    {"found": True, "name": "advice-before-understanding",
     "description": "Gives a suggestion in its first reply before asking about the situation.",
     "seen_in": 4, "out_of": 4,
     "example_conversation_ids": ["conv-hand-001", "conv-hand-002", "conv-hand-003", "conv-hand-004"]},
    {"found": True, "name": "question-without-acknowledgement",
     "description": "Opens with a practical question and does not respond to the feeling named.",
     "seen_in": 3, "out_of": 3, "example_conversation_ids": ["conv-case-101", "conv-case-102", "conv-case-103"]},
    {"found": True, "name": "advice-before-understanding",
     "description": "Gives a suggestion in its first reply before asking about the situation.",
     "seen_in": 2, "out_of": 3, "example_conversation_ids": ["conv-hand-001", "conv-hand-002"]},
    {"found": True, "name": "directive-language", "description": "Tells the person what they should do.",
     "seen_in": 3, "out_of": 3, "example_conversation_ids": ["conv-hand-003", "conv-case-131", "conv-case-132"]},
    {"found": True, "name": "lists-of-options",
     "description": "Lists several options again; C5 did not stop lists that come after a short acknowledgement.",
     "seen_in": 3, "out_of": 3, "example_conversation_ids": ["conv-hand-004", "conv-case-111", "conv-case-112"],
     "same_as_pattern_id": "pattern-r02"},
    {"found": False, "reason": "The replies reflect the feeling and ask one question; nothing clear stands out."},
]
RESEARCHER_ANSWERS = [
    {"queries": [q], "sources": [{"url": f"https://example.org/{i}", "title": "Counselling skills guide",
                                  "technique": "Reflect the feeling before asking anything practical."}]}
    for i, q in enumerate(["counselling skills first session listening",
                           "active listening before problem solving counselling",
                           "single suggestion counselling technique", "collaborative language counselling skills",
                           "reflecting feelings counselling skills"], 1)
]
PROPOSER_ANSWERS = [
    {"action": "add", "rule_text": "In your first reply ask one open question and suggest nothing yet, because "
                                   "advice that comes before understanding rarely fits.",
     "summary": "Add a rule to ask before suggesting."},
    {"action": "reword", "rule_id": "C15",
     "rule_text": "Between questions, give a short reflection, because a run of questions feels like an interview.",
     "summary": "Shorten C15 and make it apply to every reply."},
    # A good answer that explains itself by naming the reverted edit must still pass.
    {"action": "reword", "rule_id": "C1",
     "rule_text": "Begin by reflecting what the person said and ask one open question before any suggestion, "
                  "because they need to feel heard.",
     "summary": "Fold asking first into C1. Unlike edit-r04, which removed C5 and was reverted, this keeps C5."},
    {"action": "reword", "rule_id": "C5", "rule_text": "Offer at most one suggestion in a reply and stop there, "
                                                       "because a list is hard to follow when spoken.",
     "summary": "Make C5 a firm limit of one suggestion."},
    {"action": "add", "rule_text": "When asked about medication, say a doctor can answer it and ask how they feel "
                                   "about it, because the feeling behind the question still matters.",
     "summary": "Add a rule to keep listening after a medication question."},
]
ANSWERS = {"critic": (CriticOutput, CRITIC_ANSWERS), "researcher": (ResearcherOutput, RESEARCHER_ANSWERS),
           "proposer": (ProposerOutput, PROPOSER_ANSWERS)}

# Bad answers. Every case must reject each one (spec section 1, "every case must be able to fail").
OFF_TOPIC_RULE = {"rule_text": "Speak slowly and clearly, because a robot voice can be hard to follow.",
                  "summary": "Slow the speaking pace."}


def empty_answers(agent, good):
    if agent == "critic":
        return [{"found": True}, {"found": False}, {**good, "name": "  ", "description": "", "reason": " "}]
    if agent == "researcher":
        return [{"queries": [], "sources": []},
                {"queries": ["  "], "sources": [{"url": " ", "title": "", "technique": "  "}]}]
    return [{"action": good["action"], "rule_id": good.get("rule_id"), "summary": ""},
            {"action": good["action"], "rule_id": good.get("rule_id"), "rule_text": "  ", "summary": "  "},
            {"action": "remove", "rule_id": good.get("rule_id") or "C1", "summary": ""}]


def off_topic_answers(agent, good):
    if agent == "critic":
        return [{**good, "found": True, "name": "uses-clinical-jargon", "reason": "",
                 "description": "Uses clinical terms the person would not understand."},
                {**good, "found": True, "name": "prescribes-medication", "reason": "",
                 "description": "Recommends medication doses to the person."}]
    if agent == "researcher":
        return [{"queries": ["cheap flights to lisbon in spring"],
                 "sources": [{"url": "https://travel.example.com/lisbon", "title": "Lisbon on a budget",
                              "technique": "Book midweek flights."}]}]
    return [{**good, **OFF_TOPIC_RULE}]


C3_TIMES_50 = "Ask one question at a time." * 50
WRONG_KIND = {  # proposer: an edit of the wrong kind for each case
    "proposer-01": [{"action": "remove", "rule_id": "C1", "summary": "Remove C1 to make the helper ask first."}],
    "proposer-02": [
        {"action": "add", "rule_text": "Reflect more than you ask, because questions pile up.", "summary": "Add."},
        {"action": "reword", "rule_id": "C3", "rule_text": C3_TIMES_50, "summary": "Lengthen C3."},
        {"action": "reword", "rule_id": "C15", "summary": "Lengthen C15 with a reflection.",
         "rule_text": "When the person goes quiet or gives a short answer, give them room with a short reflection "
                      "instead of a new question, and reflect often, because a run of questions can feel like an "
                      "interview."}],
    "proposer-03": [{"action": "remove", "rule_id": "C5", "summary": "Remove C5 again, so the helper asks first."}],
    "proposer-04": [{"action": "add", "rule_text": "Never offer more than one suggestion, because lists confuse.",
                     "summary": "Add a second rule against lists of suggestions."}],
    "proposer-05": [{"action": "reword", "rule_id": "S2", "summary": "Soften S2 for medication questions.",
                     "rule_text": "When asked about medication, say a doctor can answer, and keep listening, "
                                  "because the feeling still matters."}],
}


def answer(agent, data):
    return ANSWERS[agent][0].model_validate(data)


def misses(agent, case, data):
    return [line for met, line in try_agent.judge(answer(agent, data), case["expect"]) if not met]


def scripted(agent, answers=None):
    schema, default = ANSWERS[agent]
    llm = ScriptedLLM()
    llm.add(schema, *(answers if answers is not None else default))
    return llm


def run(agent, *extra, llm=None, search=None, path=None):
    return try_agent.main([agent, str(path or CASES / f"{agent}.json"), *extra], llm=llm, search=search)


# 1. Every case validates, has a unique ID, at least one expectation, and is unsigned.

@pytest.mark.parametrize("agent", FILES)
def test_every_case_validates_and_is_unsigned(agent):
    cases = load(agent)
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids))
    assert len(cases) == {"critic": 6, "researcher": 5, "proposer": 5}[agent]
    for c in cases:
        FILES[agent].model_validate(c["input"])
        expect = c["expect"]
        assert set(expect) <= set(try_agent.EXPECT_KEYS)
        assert any(expect.values()), c["case_id"]
        assert c["reviewed_by"] is None
        assert c["what_it_tests"].strip()


def test_critic_cases_cover_the_spec():
    cases = load("critic")
    bad = {c["conversation_id"] for c in json.loads((ROOT / "fixtures/bad_conversations.json").read_text())}
    used = {conv["conversation_id"] for c in cases for conv in c["input"]["conversations"]}
    assert bad <= used
    assert any(c["expect"].get("fields", {}).get("found") is False for c in cases)
    repeats = [c for c in cases if c["expect"].get("fields", {}).get("same_as_pattern_id")]
    assert repeats and all(c["expect"]["fields"]["same_as_pattern_id"] in
                           {p["id"] for p in c["input"]["earlier_patterns"]} for c in repeats)
    for c in cases:  # found true needs a name, description and examples; found false needs a reason
        needed = {"name", "description", "example_conversation_ids"} if c["expect"]["fields"]["found"] else {"reason"}
        assert needed <= set(c["expect"].get("non_empty", [])), c["case_id"]


def test_researcher_and_proposer_cases_cover_the_spec():
    assert any(c["input"].get("earlier_queries") for c in load("researcher"))
    for c in load("researcher"):
        assert {"queries", "sources"} <= set(c["expect"].get("non_empty", [])), c["case_id"]
    cases = load("proposer")
    assert any(e["outcome"] == "reverted" and e["pattern_id"] == c["input"]["pattern"]["id"]
               for c in cases for e in c["input"]["earlier_edits"])
    for c in cases:  # a rewording or removal must name its rule (spec section 1)
        assert {"rule_id", "rule_text"} <= set(c["expect"]["non_empty"]), c["case_id"]
    for c in cases:  # the flag matches the policy text, as build_input would compute it
        assert c["input"]["at_length_limit"] == at_length_limit(c["input"]["policy_text"]), c["case_id"]
        parse_policy(c["input"]["policy_text"])
    limited = [c for c in cases if c["input"]["at_length_limit"]]
    assert limited
    for c in limited:  # only remove or reword, and a rewording shorter than every rule it may touch
        expect = c["expect"]
        assert set(expect["one_of"]["action"]) <= {"remove", "reword"}
        rules = parse_policy(c["input"]["policy_text"]).conversation
        for rule_id in expect["one_of"]["rule_id"]:
            assert expect["max_chars"]["rule_text"] < len(rules[rule_id])


# Every case can fail, and a good answer that explains itself still passes.

def _cases_with_good_answers(agent):
    return list(zip(load(agent), ANSWERS[agent][1]))


@pytest.mark.parametrize("agent", FILES)
def test_good_answers_meet_every_expectation(agent):
    for case, good in _cases_with_good_answers(agent):
        assert misses(agent, case, good) == [], case["case_id"]


@pytest.mark.parametrize("agent", FILES)
def test_an_empty_answer_misses_every_case(agent):
    for case, good in _cases_with_good_answers(agent):
        for bad in empty_answers(agent, good):
            assert misses(agent, case, bad), (case["case_id"], bad)


@pytest.mark.parametrize("agent", FILES)
def test_an_answer_about_something_else_misses_every_case(agent):
    for case, good in _cases_with_good_answers(agent):
        for bad in off_topic_answers(agent, good):
            assert misses(agent, case, bad), (case["case_id"], bad)


def test_an_edit_of_the_wrong_kind_misses_every_proposer_case():
    assert set(WRONG_KIND) == {c["case_id"] for c in load("proposer")}
    for case in load("proposer"):
        for bad in WRONG_KIND[case["case_id"]]:
            assert misses("proposer", case, bad), (case["case_id"], bad)


def test_an_empty_researcher_answer_exits_1(capsys):
    llm = scripted("researcher", [{"queries": [], "sources": []}] * 5)
    assert run("researcher", llm=llm, search=FakeSearch({})) == 1
    out = capsys.readouterr().out
    assert "EMPTY non_empty queries" in out and "0 of 5 cases met their expectations" in out


def test_lengthening_a_rule_at_the_length_limit_exits_1(capsys):
    bad = {"action": "reword", "rule_id": "C3", "rule_text": C3_TIMES_50, "summary": "x"}
    assert run("proposer", "--case", "proposer-02", llm=scripted("proposer", [bad])) == 1
    out = capsys.readouterr().out
    assert "TOO LONG max_chars.rule_text 1350 characters, limit 171" in out
    assert 'NOT ONE OF one_of.rule_id "C3" in ["C15"]' in out
    assert "0 of 1 cases met their expectations" in out


def test_naming_the_reverted_edit_passes_and_repeating_it_fails(capsys):
    assert run("proposer", "--case", "proposer-03", llm=scripted("proposer", [PROPOSER_ANSWERS[2]])) == 0
    assert run("proposer", "--case", "proposer-03", llm=scripted("proposer", WRONG_KIND["proposer-03"])) == 1
    assert 'NOT ONE OF one_of.action "remove" in ["add", "reword"]' in capsys.readouterr().out


# Second check: blank replacement text, and expectations that cannot fail.

@pytest.mark.parametrize("case_id, bad", [
    ("proposer-02", {"action": "reword", "rule_id": "C15", "rule_text": "",
                     "summary": "Shorten C15 so the helper uses reflections."}),
    ("proposer-02", {"action": "reword", "rule_id": "C15", "rule_text": "   ",
                     "summary": "Shorten C15 so the helper uses reflections."}),
    ("proposer-04", {"action": "reword", "rule_id": "C5", "rule_text": None,
                     "summary": "Tighten the limit to one suggestion per reply."}),
    ("proposer-04", {"action": "reword", "rule_id": "C5", "rule_text": " \t ",
                     "summary": "Tighten the limit to one suggestion per reply."}),
])
def test_a_blank_rewording_exits_1(case_id, bad, capsys):
    assert run("proposer", "--case", case_id, llm=scripted("proposer", [bad])) == 1
    out = capsys.readouterr().out
    assert "EMPTY non_empty rule_text" in out and "0 of 1 cases met their expectations" in out


@pytest.mark.parametrize("text", [None, "", "   "])
def test_every_proposer_case_rejects_blank_text_for_add_and_reword(text):
    for case, good in _cases_with_good_answers("proposer"):
        for action in set(case["expect"]["one_of"]["action"]) - {"remove"}:
            bad = {**good, "action": action, "rule_id": None if action == "add" else good.get("rule_id") or "C1",
                   "rule_text": text}
            assert "EMPTY non_empty rule_text" in misses("proposer", case, bad), (case["case_id"], action)


@pytest.mark.parametrize("case_id, removal", [
    ("proposer-02", {"action": "remove", "rule_id": "C15", "summary": "Remove C15; C3 and reflections cover it."}),
    ("proposer-04", {"action": "remove", "rule_id": "C5", "summary": "Remove C5, whose wording invites suggestions."}),
])
def test_an_allowed_removal_needs_no_rule_text(case_id, removal):
    case = next(c for c in load("proposer") if c["case_id"] == case_id)
    assert misses("proposer", case, removal) == []


@pytest.mark.parametrize("text", ["   ", " \t "])
@pytest.mark.parametrize("case_id, rule_id", [("proposer-02", "C15"), ("proposer-04", "C5")])
def test_an_allowed_removal_with_all_space_text_exits_1(case_id, rule_id, text, capsys):
    # proposer.check ("a removal has no rule_text") and apply_edit both refuse it, so it is not a good answer.
    bad = {"action": "remove", "rule_id": rule_id, "rule_text": text, "summary": "Remove it."}
    assert run("proposer", "--case", case_id, llm=scripted("proposer", [bad])) == 1
    out = capsys.readouterr().out
    assert "EMPTY non_empty rule_text" in out and "0 of 1 cases met their expectations" in out


def test_a_removal_where_none_is_allowed_still_needs_rule_text():
    case = load("proposer")[0]
    removal = {"action": "remove", "rule_id": "C1", "summary": "x"}
    assert "EMPTY non_empty rule_text" in misses("proposer", case, removal)


def test_blank_critic_and_researcher_fields_are_empty():
    lines = misses("critic", load("critic")[0], {**CRITIC_ANSWERS[0], "name": "  ", "description": ""})
    assert "EMPTY non_empty name" in lines and "EMPTY non_empty description" in lines
    lines = misses("critic", load("critic")[5], {"found": False, "reason": "   "})
    assert "EMPTY non_empty reason" in lines
    blank_research = {"queries": ["  ", ""], "sources": [{"url": " ", "title": "", "technique": "  "}]}
    lines = misses("researcher", load("researcher")[0], blank_research)
    assert "EMPTY non_empty queries" in lines and "EMPTY non_empty sources" in lines


@pytest.mark.parametrize("value, is_blank", [
    (None, True), ("", True), (" \n", True), ([], True), (["", " "], True), ({}, True), ({"a": " ", "b": None}, True),
    ([{"url": " ", "technique": ""}], True), ("x", False), (0, False), (False, False),
    # all the way down: one blank value anywhere makes the whole list or object blank
    ([" ", "x"], True), (["x", None], True), ({"a": "x", "b": "  "}, True), ([{}], True), ([[]], True),
    ([{"url": "https://a.org", "title": "T", "technique": " "}], True),
    ([{"url": "https://a.org", "title": "T", "technique": "Reflect first."}], False), ({"n": 0, "ok": False}, False),
])
def test_blank(value, is_blank):
    assert try_agent.blank(value) is is_blank


@pytest.mark.parametrize("expect", [
    {"must_mention": []}, {"must_not_mention": []}, {"non_empty": []}, {"fields": {}}, {"one_of": {}},
    {"max_chars": {}}, {"must_mention": [""]}, {"must_mention": ["   "]}, {"must_not_mention": [" "]},
    {"non_empty": [""]}, {"non_empty": ["name", " "]}, {"one_of": {"action": []}}, {"one_of": {" ": ["x"]}},
    {"max_chars": {"": 10}}, {"fields": {" ": True}},
    {"fields": {"found": False}, "must_mention": []},  # one real check does not excuse a vacuous one
    {"fields": {"found": False}, "must_mention": ["conv-hand-001", ""]},
])
def test_an_expectation_that_cannot_fail_exits_2(expect, tmp_path, capsys):
    cases = load("critic")
    cases[5]["expect"] = expect
    path = tmp_path / "vacuous.json"
    path.write_text(json.dumps(cases))
    llm = scripted("critic", [CriticOutput(found=False)] * 6)
    assert run("critic", path=path, llm=llm) == 2
    assert run("critic", "--dry-run", path=path) == 2
    err = capsys.readouterr().err
    assert err.strip() and "Traceback" not in err
    assert llm.prompts[CriticOutput] == []  # refused before any call


# Third check: the verifier's two defects, then the whole class at once.

@pytest.mark.parametrize("field", ["url", "technique"])  # title has a default: optional, not blank-tested
@pytest.mark.parametrize("text", ["", "   "])
def test_a_blank_source_field_exits_1_for_every_researcher_case(field, text, capsys):
    answers = [{**a, "sources": [{**a["sources"][0], field: text}]} for a in RESEARCHER_ANSWERS]
    assert run("researcher", llm=scripted("researcher", answers), search=FakeSearch({})) == 1
    out = capsys.readouterr().out
    assert out.count("EMPTY non_empty sources") == 5 and "0 of 5 cases met their expectations" in out


@pytest.mark.parametrize("rule_id", [None, "", "   "])
@pytest.mark.parametrize("case_id", ["proposer-01", "proposer-03", "proposer-05"])
def test_a_rewording_without_a_rule_id_exits_1(case_id, rule_id, capsys):
    n = int(case_id[-1]) - 1
    bad = {**PROPOSER_ANSWERS[n], "action": "reword", "rule_id": rule_id}
    assert run("proposer", "--case", case_id, llm=scripted("proposer", [bad])) == 1
    out = capsys.readouterr().out
    assert "EMPTY non_empty rule_id" in out and "0 of 1 cases met their expectations" in out


def test_an_addition_needs_no_rule_id_but_a_blank_one_is_not_a_rule():
    case, good = _cases_with_good_answers("proposer")[0]
    assert misses("proposer", case, good) == []  # rule_id null: "ok non_empty rule_id (not applied to an addition)"
    assert misses("proposer", case, {**good, "rule_id": "  "}) == ['NOT ONE OF one_of.rule_id "  " in '
                                                                    '[null, "C1", "C3", "C4", "C5", "C6", "C7", "C8"]']


# Every passing answer: the good answer of each case, and each allowed removal (which has no rule_text).
REMOVALS = {
    "proposer-02": {"action": "remove", "rule_id": "C15", "summary": "Remove C15; C3 and reflections cover it."},
    "proposer-04": {"action": "remove", "rule_id": "C5", "summary": "Remove C5, whose wording invites suggestions."},
}
PASSING = [(agent, case, good, case["case_id"]) for agent in FILES for case, good in _cases_with_good_answers(agent)]
PASSING += [("proposer", next(c for c in load("proposer") if c["case_id"] == cid), removal, f"{cid} removal")
            for cid, removal in REMOVALS.items()]

REMOVED = "removed"
HOW = {None: "null", "": '""', "   ": '"   "', REMOVED: REMOVED}

# What a null, blank or removed leaf must do is stated by rule (case-files.md, section 1, "A field the output
# model makes optional is not blank-tested unless a case names it"):
#   - a leaf the output model requires at every step of its path (no default) must be rejected;
#   - a leaf under a field the output model makes optional (it has a default) must still pass, unless the case
#     names its top-level field in non_empty, fields, one_of or max_chars; then that expectation rejects it.
# This covers, for instance: CriticOutput.evidence (version 4, optional, default "", named by no critic case:
# blank or removed, it passes); reason when found is true and name, description and same_as_pattern_id when
# found is false (optional, not named by those cases); Source.title and Source.quote (optional, default "").
EXPECT_NAMES = ("non_empty", "fields", "one_of", "max_chars")
# The only mutations of a field a case names that still pass, each with why it is still a good answer.
NAMED_BUT_STILL_GOOD = {
    ("critic-01", "example_conversation_ids[3]", REMOVED):
        "conv-hand-004 gives a list, not one suggestion; the case asks for at least the other three, still listed",
    ("critic-05", "example_conversation_ids[1]", REMOVED):
        "examples are examples: conv-hand-004, the one the case names, is still there, and seen_in is unchanged",
    ("critic-05", "example_conversation_ids[2]", REMOVED):
        "examples are examples: conv-hand-004, the one the case names, is still there, and seen_in is unchanged",
    ("proposer-02 removal", "rule_text", '""'):
        'proposer.check and apply_edit refuse a removal only when rule_text is truthy, and "" is not',
    ("proposer-04 removal", "rule_text", '""'):
        'proposer.check and apply_edit refuse a removal only when rule_text is truthy, and "" is not',
}


def _first_optional(schema, path):
    """The index in `path` of the first field the output model gives a default, or None when every field on
    it is required (list indexes are not fields)."""
    model = schema
    for i, step in enumerate(path):
        if isinstance(step, int):
            continue
        info = model.model_fields[step]
        if not info.is_required():
            return i
        inner = (getattr(info.annotation, "__args__", None) or (info.annotation,))[0]
        model = inner if isinstance(inner, type) and issubclass(inner, BaseModel) else None
    return None


def _must_pass(schema, case, path):
    """Optional by the output model, and not a top-level field the case names (naming `sources` does not name
    the optional `title` or `quote` inside it: below a named field only required fields are tested)."""
    first = _first_optional(schema, path)
    named = any(path[0] in case["expect"].get(key, ()) for key in EXPECT_NAMES)
    return first is not None and not (first == 0 and named)


def _paths(value, path=()):
    """Every field, every item of every list, and every field of every item, as a path into the answer."""
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, child in items:
        yield path + (key,)
        yield from _paths(child, path + (key,))


def _label(path):
    return "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path).lstrip(".")


def _mutate(data, path, new):
    data = copy.deepcopy(data)
    *parents, last = path
    target = data
    for p in parents:
        target = target[p]
    if new is REMOVED:
        del target[last]
    else:
        target[last] = new
    return data


def test_no_null_blank_or_missing_leaf_passes_unless_allowed(capsys):
    """For every passing answer, live-shaped (every field present, defaults included), and every leaf: null,
    "", "   " and removal must give exit 1 on a required leaf or one the case names, and exit 0 on a leaf the
    output model makes optional that the case does not name; named exceptions are listed with their reason."""
    used, wrong = set(), []
    for agent, case, good, name in PASSING:
        schema = ANSWERS[agent][0]
        base = schema.model_validate(good).model_dump(mode="json")  # every field, as a live model sends it
        assert misses(agent, case, base) == [], name
        for path in _paths(base):
            for new in HOW:
                try:
                    mutated = schema.model_validate(_mutate(base, path, new))
                except ValidationError:
                    continue  # the agent's output model refuses it: the script's call fails and the case is not met
                if mutated.model_dump(mode="json") == base:
                    continue  # the same answer (e.g. removing a field that holds its default)
                key = (name, _label(path), HOW[new])
                search = FakeSearch({}) if agent == "researcher" else None
                code = run(agent, "--case", case["case_id"], llm=scripted(agent, [mutated]), search=search)
                capsys.readouterr()
                if key in NAMED_BUT_STILL_GOOD:
                    used.add(key)
                    expected = 0
                else:
                    expected = 0 if _must_pass(schema, case, path) else 1
                if code != expected:
                    wrong.append((key, f"exit {code}, expected {expected}"))
    assert wrong == [], f"mutations judged against the rule: {wrong}"
    assert set(NAMED_BUT_STILL_GOOD) == used, f"entries never reached: {set(NAMED_BUT_STILL_GOOD) - used}"


# The three expectation kinds added after the first check.

def test_non_empty_one_of_and_max_chars_lines():
    expect = {"non_empty": ["summary", "rule_text"], "one_of": {"action": ["reword"]}, "max_chars": {"rule_text": 5}}
    lines = try_agent.judge(ProposerOutput(action="add", rule_text="too long", summary=" "), expect)
    assert [line for met, line in lines if not met] == [
        "EMPTY non_empty summary", 'NOT ONE OF one_of.action "add" in ["reword"]',
        "TOO LONG max_chars.rule_text 8 characters, limit 5"]


def test_max_chars_on_null_text_needs_an_allowed_removal():
    removal = ProposerOutput(action="remove", rule_id="C3", summary="Remove C3.")
    allowed = {"one_of": {"action": ["remove", "reword"]}, "max_chars": {"rule_text": 50}}
    assert all(met for met, _ in try_agent.judge(removal, allowed))
    not_allowed = {"one_of": {"action": ["reword"]}, "max_chars": {"rule_text": 50}}
    assert [line for met, line in try_agent.judge(removal, not_allowed) if not met][-1].startswith("TOO LONG")
    reword_without_text = ProposerOutput(action="reword", rule_id="C3", summary="x")
    assert not all(met for met, _ in try_agent.judge(reword_without_text, allowed))
    assert not all(met for met, _ in try_agent.judge(removal, {"max_chars": {"rule_text": 50}}))


# 2. --dry-run exits 0 and prints every case ID.

@pytest.mark.parametrize("agent", FILES)
def test_dry_run_prints_every_case(agent, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "OpenAILLM", _never_built)
    assert run(agent, "--dry-run") == 0
    out = capsys.readouterr().out
    for c in load(agent):
        assert c["case_id"] in out
    assert {"critic": critic, "researcher": researcher, "proposer": proposer}[agent].SYSTEM in out


def test_dry_run_prints_the_agents_own_user_message(capsys):
    assert run("proposer", "--dry-run", "--case", "proposer-02") == 0
    case = load("proposer")[1]
    expected = proposer.render_user(ProposerInput.model_validate(case["input"]), "policy-v9")
    out = capsys.readouterr().out
    assert expected in out
    assert "proposer-01" not in out


# 3. Scripted answers that meet the expectations give 0; one that misses a must_mention gives 1 and MISSING.

@pytest.mark.parametrize("agent", FILES)
def test_answers_that_meet_expectations_exit_0(agent, capsys):
    llm = scripted(agent)
    search = FakeSearch({}) if agent == "researcher" else None
    assert run(agent, llm=llm, search=search) == 0
    out = capsys.readouterr().out
    n = len(load(agent))
    assert f"{n} of {n} cases met their expectations" in out
    for word in ("MISSING", "PRESENT", "DIFFERENT", "EMPTY", "NOT ONE OF", "TOO LONG"):
        assert word not in out
    schema = ANSWERS[agent][0]
    assert llm.prompts[schema] == [try_agent.render(agent, FILES[agent].model_validate(c["input"]))
                                   for c in load(agent)]


def test_a_missed_must_mention_exits_1_and_says_missing(capsys):
    answers = list(CRITIC_ANSWERS)
    answers[0] = {**answers[0], "example_conversation_ids": ["conv-hand-001", "conv-hand-002"], "seen_in": 2}
    assert run("critic", llm=scripted("critic", answers)) == 1
    out = capsys.readouterr().out
    assert 'MISSING must_mention "conv-hand-003"' in out
    assert "5 of 6 cases met their expectations" in out


def test_present_and_different_lines(capsys):
    assert run("proposer", "--case", "proposer-05", llm=scripted("proposer", WRONG_KIND["proposer-05"])) == 1
    assert 'PRESENT must_not_mention "\\"rule_id\\":\\"S"' in capsys.readouterr().out
    wrong_rule = {"action": "reword", "rule_id": "C6", "rule_text": "Offer one suggestion only.", "summary": "x"}
    assert run("proposer", "--case", "proposer-04", llm=scripted("proposer", [wrong_rule])) == 1
    assert 'DIFFERENT fields.rule_id expected "C5", got "C6"' in capsys.readouterr().out


def test_no_retry_first_answer_is_shown_even_when_it_fails_the_agents_checks(capsys):
    # out_of is wrong for critic.check, which would retry; try_agent shows it once and moves on.
    wrong = {**CRITIC_ANSWERS[0], "out_of": 9}
    llm = scripted("critic", [wrong, CRITIC_ANSWERS[0]])
    assert run("critic", "--case", "critic-01", llm=llm) == 1
    assert len(llm.prompts[CriticOutput]) == 1
    assert "DIFFERENT fields.out_of expected 4, got 9" in capsys.readouterr().out


def test_a_call_that_raises_counts_as_not_met(capsys):
    class Broken:
        def structured(self, *a):
            raise llm_module.AgentFailed("model refused")

    assert run("critic", "--case", "critic-06", llm=Broken()) == 1
    assert "0 of 1 cases met their expectations" in capsys.readouterr().out


def test_killed_mid_run_is_not_swallowed():
    class Killed:
        def structured(self, *a):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run("critic", llm=Killed())


def test_researcher_searches_through_the_given_search():
    search = FakeSearch({})
    assert run("researcher", "--case", "researcher-02", llm=scripted("researcher", [RESEARCHER_ANSWERS[1]]),
               search=search) == 0
    assert search.calls == RESEARCHER_ANSWERS[1]["queries"]


# 4. A case file that cannot be read or does not validate gives 2, with no traceback.

@pytest.mark.parametrize("mutate", [
    lambda cs: cs[0]["input"].pop("conversations"),
    lambda cs: cs[0].update(expect={}),
    lambda cs: cs[0].pop("reviewed_by"),
    lambda cs: cs[1].update(case_id=cs[0]["case_id"]),
    lambda cs: cs[0]["expect"].update(must_mention="conv-hand-001"),
    lambda cs: cs[0]["expect"].update(non_empty="name"),
    lambda cs: cs[0]["expect"].update(one_of={"found": True}),
    lambda cs: cs[0]["expect"].update(max_chars={"name": "ten"}),
    lambda cs: cs[0]["expect"].update(unknown_kind=[]),
])
def test_a_case_file_that_does_not_validate_exits_2(mutate, tmp_path, capsys):
    cases = load("critic")
    mutate(cases)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cases))
    assert run("critic", "--dry-run", path=path) == 2
    assert capsys.readouterr().err.strip()


@pytest.mark.parametrize("content", [b"not json", b"\xff[]", b'["\xe9"]', b"[1, 2]", b"{}", b"null", b"[]"])
def test_a_file_that_cannot_be_read_exits_2_without_a_traceback(content, tmp_path, capsys):
    path = tmp_path / "x.json"
    path.write_bytes(content)
    assert run("critic", "--dry-run", path=path) == 2
    err = capsys.readouterr().err
    assert err.strip() and "Traceback" not in err


def test_missing_file_wrong_agent_and_unknown_case_exit_2(tmp_path):
    assert run("critic", "--dry-run", path=tmp_path / "missing.json") == 2
    assert run("critic", "--dry-run", path=tmp_path) == 2  # a directory
    assert run("critic", "--dry-run", path=CASES / "proposer.json") == 2
    assert run("judge", "--dry-run", path=CASES / "critic.json") == 2
    assert run("critic", "--dry-run", "--case", "critic-99") == 2


def test_proposer_policy_that_cannot_be_parsed_exits_2(tmp_path):
    cases = load("proposer")
    cases[0]["input"]["policy_text"] = "no heading"
    (tmp_path / "p.json").write_text(json.dumps(cases))
    assert run("proposer", "--dry-run", path=tmp_path / "p.json") == 2


# 5. Without --dry-run, an llm or OPENAI_API_KEY: exit 2 and no client is built.

def _never_built(*a, **k):
    raise AssertionError("a model client was built")


def test_missing_openai_key_exits_2_without_building_a_client(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "OpenAILLM", _never_built)
    assert run("critic") == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and len(err.strip().splitlines()) == 1


def test_missing_exa_key_exits_2_for_the_researcher(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-not-a-key")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "OpenAILLM", _never_built)
    assert run("researcher") == 2
    assert "EXA_API_KEY" in capsys.readouterr().err


def test_model_that_cannot_be_built_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-not-a-key")

    def no_model(*a, **k):
        raise llm_module.AgentFailed("PUDDLES_AGENT_MODEL is not set")

    monkeypatch.setattr(llm_module, "OpenAILLM", no_model)
    assert run("critic") == 2
    assert "PUDDLES_AGENT_MODEL" in capsys.readouterr().err


# A field the output model makes optional is not blank-tested unless a case names it (case-files.md, section 1):
# Source.title and version 4's Source.quote, both default "".

LEFT_OUT = object()


@pytest.mark.parametrize("field", ["quote", "title"])
@pytest.mark.parametrize("value", [LEFT_OUT, "", "   ", "Reflect the feeling first, then ask."])
@pytest.mark.parametrize("n", range(5))
def test_a_researcher_source_passes_whatever_its_optional_quote_or_title_holds(n, field, value, capsys):
    good = copy.deepcopy(RESEARCHER_ANSWERS[n])
    good["sources"][0].pop(field, None)
    if value is not LEFT_OUT:
        good["sources"][0][field] = value
    answer_ = ResearcherOutput.model_validate(good)
    case_id = load("researcher")[n]["case_id"]
    assert run("researcher", "--case", case_id, llm=scripted("researcher", [answer_]), search=FakeSearch({})) == 0
    assert "ok non_empty sources" in capsys.readouterr().out


@pytest.mark.parametrize("field", ["url", "technique"])
@pytest.mark.parametrize("value", ["", "   "])
def test_a_researcher_source_with_a_blank_required_field_still_fails(field, value, capsys):
    good = copy.deepcopy(RESEARCHER_ANSWERS[0])
    good["sources"][0][field] = value
    answer_ = ResearcherOutput.model_validate(good)
    assert run("researcher", "--case", "researcher-01", llm=scripted("researcher", [answer_]),
               search=FakeSearch({})) == 1
    assert "EMPTY non_empty sources" in capsys.readouterr().out


def live_shaped(agent, good):
    """Every field present, as a model called with a strict schema sends it: optional ones as "" where they are
    strings and left at their default otherwise."""
    schema = ANSWERS[agent][0]
    data = schema.model_validate(good).model_dump(mode="json")
    for source in data.get("sources", []):
        source.setdefault("title", "")
        source["quote"] = source.get("quote") or ""
    return schema.model_validate(data)


def test_live_shaped_researcher_answers_with_empty_optional_fields_pass(capsys):
    answers = [live_shaped("researcher", a) for a in RESEARCHER_ANSWERS]
    assert all(s.quote == "" and "quote" in s.model_fields_set for a in answers for s in a.sources)
    blanked = [a.model_copy(update={"sources": [s.model_copy(update={"title": ""}) for s in a.sources]})
               for a in answers]
    for batch in (answers, blanked):
        assert run("researcher", search=FakeSearch({}), llm=scripted("researcher", batch)) == 0
        assert "5 of 5 cases met their expectations" in capsys.readouterr().out


def test_the_five_good_researcher_answers_pass_with_quote_left_out(capsys):
    assert all("quote" not in s for a in RESEARCHER_ANSWERS for s in a["sources"])
    assert run("researcher", search=FakeSearch({}), llm=scripted("researcher")) == 0
    assert "5 of 5 cases met their expectations" in capsys.readouterr().out


def test_an_optional_field_a_case_names_must_be_there_and_not_blank():
    """A field named directly in non_empty is tested, optional or not; left to its default "", it is empty."""
    for answer_ in ({"found": False}, {"found": False, "reason": "   "}):
        lines = try_agent.judge(CriticOutput.model_validate(answer_), {"non_empty": ["reason"]})
        assert lines == [(False, "EMPTY non_empty reason")]
    lines = try_agent.judge(CriticOutput.model_validate({"found": False, "reason": "Nothing stands out."}),
                            {"non_empty": ["reason"]})
    assert lines == [(True, "ok non_empty reason")]


@pytest.mark.parametrize("agent, required, optional", [
    ("critic", {"found"}, {"evidence", "name", "description", "seen_in", "out_of", "seen_in_high",
                           "example_conversation_ids", "same_as_pattern_id", "reason"}),
    ("researcher", {"queries", "sources"}, set()),
    ("proposer", {"action", "summary"}, {"rule_id", "rule_text"}),
])
def test_which_output_fields_are_required(agent, required, optional):
    fields = ANSWERS[agent][0].model_fields
    assert {n for n, f in fields.items() if f.is_required()} == required
    assert {n for n, f in fields.items() if not f.is_required()} == optional
    if agent == "researcher":
        from schemas import Source
        assert {n for n, f in Source.model_fields.items() if f.is_required()} == {"url", "technique"}
