import re as _re

import pytest

from loop.errors import HelperError, LLMError
from loop.helper import helper_reply, trim_to_sentence
from loop.llm import FakeClient
from loop.policy import Policy
from loop.prompts import helper_messages, render_helper_system
from schemas import HelperReply, Turn

POLICY = Policy(version="policy-x", text="# policy-x\n\n## Safety\nS1. Be safe.\n",
                safety_text="## Safety\nS1. Be safe.", rules={"S1": "Be safe."})
TURNS = [Turn(speaker="user", text="I can't sleep.")]


def _reply(replies, **kwargs):
    client = FakeClient(replies)
    result = helper_reply(client, POLICY, TURNS, session_id="conv-r01-x", turn=2, **kwargs)
    return client, result


def test_helper_reply_calls_model_once_and_wraps_reply():
    client, reply = _reply(["  That sounds hard. What keeps you up?  "], seed=5, user_notes=["n"])
    assert reply == HelperReply(session_id="conv-r01-x", turn=2,
                                text="That sounds hard. What keeps you up?",
                                policy_version="policy-x", fixed_handoff_response=False)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["system"] == render_helper_system(POLICY, ["n"])
    assert call["messages"] == helper_messages(TURNS)
    assert call["max_tokens"] == 80
    assert call["seed"] == 5


def test_empty_reply_is_retried_once():
    client, reply = _reply(["   ", "I'm here."])
    assert reply.text == "I'm here."
    assert len(client.calls) == 2


def test_two_empty_replies_raise_helper_error():
    with pytest.raises(HelperError):
        _reply(["", "  \n"])


def test_llm_error_propagates():
    with pytest.raises(LLMError):
        _reply([])


@pytest.mark.parametrize("raw, expected", [
    ("That sounds hard. What keeps you", "That sounds hard."),
    ("Are you okay? I wonder whether", "Are you okay?"),
    ('You said "I give up." And then you', 'You said "I give up."'),
    ("(I hear you.) It sounds like", "(I hear you.)"),
    ("That sounds hard!", "That sounds hard!"),
    ('She said "enough."', 'She said "enough."'),
    ("no full stop at all", "no full stop at all"),
])
def test_reply_is_cut_to_last_complete_sentence(raw, expected):
    _, reply = _reply([raw])
    assert reply.text == expected


# --- trim_to_sentence: abbreviations and numbers are not sentence ends -------------------

REAL_DRAFT_TAIL = (
    "If you might act on those thoughts or feel in immediate danger, call emergency services now. "
    "In the U.S. or Canada, call or text 988; elsewhere, contact your"
)


def test_real_draft_is_cut_before_the_fragment_not_at_us():
    assert trim_to_sentence(REAL_DRAFT_TAIL) == (
        "If you might act on those thoughts or feel in immediate danger, call emergency services now.")


INITIALISMS_AND_WORDS = ["U.S.", "U.K.", "e.g.", "i.e.", "a.m.", "p.m.", "A.M.", "E.g.",
                         "vs.", "etc.", "approx."]
TITLES = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St."]


@pytest.mark.parametrize("abbr", INITIALISMS_AND_WORDS + TITLES)
@pytest.mark.parametrize("after", ["then", "3 times", ", then", "(then"])
def test_abbreviation_before_lowercase_digit_or_punctuation_is_not_a_boundary(abbr, after):
    raw = f"I hear you. Think of {abbr} {after} we can talk and"
    assert trim_to_sentence(raw) == "I hear you."


@pytest.mark.parametrize("abbr", INITIALISMS_AND_WORDS)
@pytest.mark.parametrize("after", ["Then", '"Then', "(Then"])
def test_initialism_or_word_before_uppercase_is_a_boundary(abbr, after):
    raw = f"I hear you. Think of {abbr} {after} we can talk and"
    assert trim_to_sentence(raw) == f"I hear you. Think of {abbr}"


@pytest.mark.parametrize("abbr", TITLES)
def test_title_before_uppercase_is_not_a_boundary(abbr):
    assert trim_to_sentence(f"I hear you. Talk to {abbr} Lee about it and") == "I hear you."


def test_complete_sentence_ending_on_abbreviation_is_kept():
    assert (trim_to_sentence("I'm listening. You can call or text 988 in the U.S. If you")
            == "I'm listening. You can call or text 988 in the U.S.")
    assert (trim_to_sentence("I hear you. Bring water, snacks, etc. When you")
            == "I hear you. Bring water, snacks, etc.")


def test_trailing_whitespace_after_title_matches_stripped_text():
    raw = "I hear you. Talk to Dr. "
    assert trim_to_sentence(raw) == trim_to_sentence(raw.strip()) == "I hear you. Talk to Dr."


def test_no_as_a_whole_sentence_is_a_boundary():
    assert trim_to_sentence("No. I don't think that fits. And") == "No. I don't think that fits."


def test_abbreviation_in_brackets_is_not_a_boundary():
    assert trim_to_sentence("I hear you. Some things (e.g. walks) help and") == "I hear you."


def test_decimal_is_not_a_boundary():
    assert trim_to_sentence("I hear you. Sleep about 7.5 hours and") == "I hear you."
    assert trim_to_sentence("Call 988. It is free and") == "Call 988."


def test_abbreviation_as_last_token_ends_the_text():
    assert trim_to_sentence("I am in the U.S.") == "I am in the U.S."
    assert trim_to_sentence("Talk to your Dr. Or a friend, etc.") == "Talk to your Dr. Or a friend, etc."


@pytest.mark.parametrize("raw, expected", [
    ('You asked "why me?" I wonder', 'You asked "why me?"'),
    ('She shouted "stop!" and then', 'She shouted "stop!"'),
    ("(Really?) That is a lot to", "(Really?)"),
])
def test_bang_and_question_with_closer(raw, expected):
    assert trim_to_sentence(raw) == expected


def test_text_with_no_boundary_is_unchanged():
    assert trim_to_sentence("In the U.S. or Canada, call or text") == "In the U.S. or Canada, call or text"
    assert trim_to_sentence("no full stop at all") == "no full stop at all"


_OLD_CLOSERS = "\"'”’)]}"
_OLD_ENDS_COMPLETE = _re.compile(rf"[.!?][{_re.escape(_OLD_CLOSERS)}]?$")
_OLD_SENTENCE_END = _re.compile(rf"[.!?][{_re.escape(_OLD_CLOSERS)}]?(?=\s)")


def _old_trim_to_sentence(text: str) -> str:
    if _OLD_ENDS_COMPLETE.search(text):
        return text
    ends = list(_OLD_SENTENCE_END.finditer(text))
    return text[:ends[-1].end()] if ends else text


@pytest.mark.parametrize("raw", [
    "That sounds hard. What keeps you up at night? I wonder whether",
    "I'm sorry you're going through this. It makes sense to feel tired. Would it help to",
    "You've carried a lot today! Let's slow down. What feels most pressing right",
    'You said "nothing helps." That sounds exhausting. Can you tell me',
    "(I'm listening.) Take your time. When did this",
    "It sounds lonely.\nWhat would a good evening look like?\nMaybe we could",
    "That sounds hard. What keeps you up?",
    "Thank you for telling me. ",
    "It has been 3 weeks. That is a long time. How are",
])
def test_ordinary_replies_match_old_behaviour(raw):
    assert trim_to_sentence(raw) == _old_trim_to_sentence(raw)
