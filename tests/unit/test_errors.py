import pytest

from loop.errors import (ConfigError, ConversationFailed, HelperError, LLMError, PolicyError,
                         SimulatedUserError)


@pytest.mark.parametrize("cls", [LLMError, ConfigError, PolicyError, HelperError,
                                 SimulatedUserError, ConversationFailed])
def test_each_error_is_an_exception(cls):
    assert issubclass(cls, Exception)


def test_conversation_failed_keeps_scenario_and_reason():
    exc = ConversationFailed("fixture-0001", "helper went quiet")
    assert exc.scenario_id == "fixture-0001"
    assert exc.reason == "helper went quiet"
