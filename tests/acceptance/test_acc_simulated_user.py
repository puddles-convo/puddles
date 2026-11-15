"""Only the exact stripped end marker ends a simulated conversation."""

import importlib

import pytest


@pytest.fixture
def api():
    return importlib.import_module("loop.simulated_user")


def test_simulated_user_request_and_default_options(api, scenario_factory, turns, client_factory):
    from loop.prompts import render_simulated_user_system, simulated_user_messages
    scenario = scenario_factory()
    client = client_factory(["  I am still worried.\n"])
    assert api.next_user_line(client, scenario, turns, seed=23) == "I am still worried."
    assert client.calls == [dict(
        system=render_simulated_user_system(scenario), messages=simulated_user_messages(turns),
        temperature=0.9, seed=23, max_tokens=120,
    )]


def test_simulated_user_custom_token_cap_and_default_seed(api, scenario_factory, client_factory):
    client = client_factory(["I am tired."])
    api.next_user_line(client, scenario_factory(), (), max_tokens=63)
    assert (client.calls[0]["max_tokens"], client.calls[0]["seed"]) == (63, None)


@pytest.mark.parametrize("raw", ["[END]", " \n[END]\t"])
def test_exact_end_marker_returns_none_without_retry(api, scenario_factory, client_factory, raw):
    client = client_factory([raw])
    assert api.END_MARKER == "[END]"
    assert api.next_user_line(client, scenario_factory(), ()) is None
    assert len(client.calls) == 1


@pytest.mark.parametrize("raw", ["[end]", "[END].", "I said [END] yesterday."])
def test_near_end_markers_are_ordinary_text(api, scenario_factory, client_factory, raw):
    client = client_factory([raw])
    assert api.next_user_line(client, scenario_factory(), ()) == raw


@pytest.mark.parametrize("invalid", [
    "", " \n\t", " ".join(["word"] * 81),
    "AS AN AI I cannot feel.", "I am a LANGUAGE MODEL.",
    "This is a SiMuLaTiOn.", "I was simulating a conversation.",
])
def test_invalid_line_is_retried_once(api, scenario_factory, client_factory, invalid):
    client = client_factory([invalid, "I am having a hard day."])
    assert api.next_user_line(client, scenario_factory(), (), seed=5) == "I am having a hard day."
    assert len(client.calls) == 2


def test_second_invalid_line_fails_and_does_not_try_a_third(api, scenario_factory, client_factory):
    from loop.errors import SimulatedUserError
    client = client_factory(["", "A language model", "Unused"])
    with pytest.raises(SimulatedUserError):
        api.next_user_line(client, scenario_factory(), ())
    assert len(client.calls) == 2


def test_valid_eighty_word_boundary(api, scenario_factory, client_factory):
    line = " ".join(["worried"] * 80)
    client = client_factory([line])
    assert api.next_user_line(client, scenario_factory(), ()) == line
    assert len(client.calls) == 1


def test_retry_may_end_conversation(api, scenario_factory, client_factory):
    client = client_factory(["", "[END]"])
    assert api.next_user_line(client, scenario_factory(), ()) is None
    assert len(client.calls) == 2


def test_simulated_user_transport_error_propagates_without_retry(api, scenario_factory, client_factory):
    from loop.errors import LLMError, SimulatedUserError
    client = client_factory([LLMError("offline"), "Unused"])
    with pytest.raises((LLMError, SimulatedUserError)):
        api.next_user_line(client, scenario_factory(), ())
    assert len(client.calls) == 1
