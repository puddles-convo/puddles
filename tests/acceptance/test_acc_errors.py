"""The public exception contract does not depend on runner failures."""

import pytest


@pytest.mark.parametrize("name", [
    "LLMError", "ConfigError", "PolicyError", "HelperError",
    "SimulatedUserError", "ConversationFailed",
])
def test_public_errors_subclass_exception(name):
    from loop import errors
    assert issubclass(getattr(errors, name), Exception)


def test_conversation_failed_accepts_positional_scenario_and_reason():
    from loop.errors import ConversationFailed
    error = ConversationFailed("id", "why")
    assert error.scenario_id == "id"
    assert error.reason == "why"
