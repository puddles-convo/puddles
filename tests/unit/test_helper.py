import pytest

from loop.errors import HelperError, LLMError
from loop.helper import helper_reply
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
