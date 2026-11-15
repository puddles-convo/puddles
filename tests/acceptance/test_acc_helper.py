"""Helper call forwarding, empty-only retry, and sentence completion."""

import importlib

import pytest


@pytest.fixture
def api():
    return importlib.import_module("loop.helper")


def test_helper_plain_client_receives_prompt_history_and_options(api, policy, turns, client_factory):
    from loop.prompts import helper_messages, render_helper_system
    from schemas import HelperReply
    client = client_factory(["  I hear you.  "])
    reply = api.helper_reply(client, policy, turns, session_id="session-test", turn=3,
                             user_notes=("Works nights.",), max_tokens=47, seed=19)
    assert isinstance(reply, HelperReply)
    assert reply.model_dump() == dict(session_id="session-test", turn=3, text="I hear you.",
                                      policy_version=policy.version, fixed_handoff_response=False)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["system"] == render_helper_system(policy, ("Works nights.",))
    assert call["messages"] == helper_messages(turns)
    assert (call["max_tokens"], call["seed"]) == (47, 19)


def test_helper_default_token_cap_and_seed(api, policy, client_factory):
    client = client_factory(["Okay."])
    api.helper_reply(client, policy, (), session_id="s", turn=1)
    assert (client.calls[0]["max_tokens"], client.calls[0]["seed"]) == (80, None)


def test_helper_retries_one_empty_reply(api, policy, turns, client_factory):
    client = client_factory([" \n\t", "  Tell me more.  "])
    reply = api.helper_reply(client, policy, turns, session_id="s", turn=1, seed=11)
    assert reply.text == "Tell me more."
    assert len(client.calls) == 2


def test_helper_second_empty_fails_without_third_call(api, policy, client_factory):
    from loop.errors import HelperError
    client = client_factory(["", " \n", "Should never be used."])
    with pytest.raises(HelperError):
        api.helper_reply(client, policy, (), session_id="s", turn=1)
    assert len(client.calls) == 2


@pytest.mark.parametrize(("raw", "expected"), [
    ("  A whole sentence. A fragment  ", "A whole sentence."),
    ("First! Second? Unfinished", "First! Second?"),
    ("No punctuation at all", "No punctuation at all"),
    ("A complete thought.", "A complete thought."),
    ('She said, "I understand."', 'She said, "I understand."'),
    ('He said "ok." And', 'He said "ok."'),
    ("(That sounds hard.)", "(That sounds hard.)"),
    ("[That sounds hard!]", "[That sounds hard!]"),
])
def test_helper_sentence_cleanup(api, policy, client_factory, raw, expected):
    client = client_factory([raw])
    assert api.helper_reply(client, policy, (), session_id="s", turn=1).text == expected
    assert len(client.calls) == 1


def test_helper_transport_error_propagates_without_retry(api, policy, client_factory):
    from loop.errors import HelperError, LLMError
    client = client_factory([LLMError("transport broke"), "Unused."])
    with pytest.raises((LLMError, HelperError)):
        api.helper_reply(client, policy, (), session_id="s", turn=1)
    assert len(client.calls) == 1
