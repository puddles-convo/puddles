"""Exception classes for the conversation loop."""


class LLMError(Exception):
    """A chat model call failed: transport, server or unexpected body."""

    status: int | None = None  # the HTTP status, when the server answered with an error


class DeadlineExceeded(LLMError):
    """The time allowed for a call, retries included, ran out."""


class SchemaError(LLMError):
    """The reply did not match the schema, and neither did the retry."""


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
