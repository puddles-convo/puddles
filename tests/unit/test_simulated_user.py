import pytest

from loop.errors import SimulatedUserError
from loop.llm import FakeClient
from loop.prompts import render_simulated_user_system, simulated_user_messages
from loop.simulated_user import END_MARKER, next_user_line
from schemas import Scenario, Turn

SCENARIO = Scenario(scenario_id="s-1", split="validation", emotion_type="sadness",
                    problem_type="breakup", situation="My partner left.", initial_intensity=3)
TURNS = [Turn(speaker="user", text="Hi."), Turn(speaker="helper", text="Hello. What's up?")]


def test_end_marker_value():
    assert END_MARKER == "[END]"


def test_next_user_line_calls_model_with_expected_arguments():
    client = FakeClient(["  It's been a rough week.  "])
    assert next_user_line(client, SCENARIO, TURNS, seed=11) == "It's been a rough week."
    call, = client.calls
    assert call["system"] == render_simulated_user_system(SCENARIO)
    assert call["messages"] == simulated_user_messages(TURNS)
    assert call["temperature"] == 0.9
    assert call["seed"] == 11
    assert call["max_tokens"] == 120


def test_end_marker_returns_none():
    assert next_user_line(FakeClient([" [END] "]), SCENARIO, TURNS) is None


@pytest.mark.parametrize("bad", ["", "   ", " ".join(["word"] * 81),
                                 "As an AI I can't feel that.", "I'm just a LANGUAGE MODEL.",
                                 "This simulation is odd."])
def test_invalid_line_is_retried_once(bad):
    client = FakeClient([bad, "Okay."])
    assert next_user_line(client, SCENARIO, TURNS) == "Okay."
    assert len(client.calls) == 2


def test_eighty_words_is_allowed():
    line = " ".join(["word"] * 80)
    assert next_user_line(FakeClient([line]), SCENARIO, TURNS) == line


def test_two_invalid_lines_raise():
    with pytest.raises(SimulatedUserError):
        next_user_line(FakeClient(["", "I am simulated."]), SCENARIO, TURNS)
