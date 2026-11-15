"""Exception classes for the conversation loop."""


class LLMError(Exception):
    """A chat model call failed: transport, server or unexpected body."""


class ConfigError(Exception):
    """A required setting, such as an environment variable, is missing."""


class PolicyError(Exception):
    """A policy file is malformed."""


class HelperError(Exception):
    """The helper gave no usable reply."""


class SimulatedUserError(Exception):
    """The simulated user gave no usable line."""


class ConversationFailed(Exception):
    """A conversation could not be completed."""

    def __init__(self, scenario_id: str, reason: str) -> None:
        super().__init__(f"{scenario_id}: {reason}")
        self.scenario_id = scenario_id
        self.reason = reason
