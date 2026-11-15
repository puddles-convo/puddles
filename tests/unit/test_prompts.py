import re
from pathlib import Path

import pytest

from loop.policy import Policy
from loop.prompts import (helper_messages, render_helper_system, render_simulated_user_system,
                          simulated_user_messages)
from schemas import Scenario, Turn

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = re.compile(r"\{[a-z_][^{}]*\}")

POLICY = Policy(version="policy-x", text="\n# policy-x\n\n## Safety\nS1. Be safe.\n\n",
                safety_text="## Safety\nS1. Be safe.", rules={"S1": "Be safe."})
SCENARIO = Scenario(scenario_id="s-1", split="validation", emotion_type="anxiety",
                    problem_type="job crisis", situation="Worried about losing my job.",
                    initial_intensity=4, openness="guarded")
TURNS = [Turn(speaker="user", text="u1"), Turn(speaker="helper", text="h1"),
         Turn(speaker="user", text="u2")]


def _first_system_block(agent_file):
    text = (ROOT / "docs" / "agents" / agent_file).read_text(encoding="utf-8")
    after = text.split("### System message", 1)[1]
    return re.search(r"^```[^\n]*\n(.*?)\n```", after, re.S | re.M).group(1)


@pytest.mark.parametrize("txt, agent_file", [
    ("helper_system.txt", "t2-i3-helper.md"),
    ("simulated_user_system.txt", "t1-simulated-user.md"),
])
def test_prompt_files_match_agent_docs(txt, agent_file):
    prompt = (ROOT / "loop" / "prompts" / txt).read_text(encoding="utf-8")
    assert prompt == _first_system_block(agent_file)


def test_render_helper_system_without_notes():
    system = render_helper_system(POLICY)
    assert '<policy version="policy-x">\n# policy-x\n\n## Safety\nS1. Be safe.\n</policy>' in system
    assert ("<what_you_know_about_this_person>\nNothing yet. This is your first conversation.\n"
            "</what_you_know_about_this_person>") in system
    assert not PLACEHOLDER.search(system)


def test_render_helper_system_with_notes():
    system = render_helper_system(POLICY, ["Works nights.", "Has a dog."])
    assert ("<what_you_know_about_this_person>\n- Works nights.\n- Has a dog.\n"
            "</what_you_know_about_this_person>") in system
    assert "Nothing yet" not in system
    assert not PLACEHOLDER.search(system)


def test_helper_messages_map_roles_in_order():
    assert helper_messages(TURNS) == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "h1"},
        {"role": "user", "content": "u2"},
    ]


def test_render_simulated_user_system_fills_scenario_only():
    system = render_simulated_user_system(SCENARIO)
    assert "- Situation: Worried about losing my job." in system
    assert "- Main feeling: anxiety, intensity 4 out of 5" in system
    assert "- Kind of problem: job crisis" in system
    assert "- Openness: guarded   (open, guarded or resistant)" in system
    assert not PLACEHOLDER.search(system)
    template = (ROOT / "loop" / "prompts" / "simulated_user_system.txt").read_text()
    expected = (template.replace("{situation}", SCENARIO.situation)
                .replace("{emotion_type}", "anxiety").replace("{initial_intensity}", "4")
                .replace("{problem_type}", "job crisis").replace("{openness}", "guarded"))
    assert system == expected


def test_simulated_user_messages_start_with_begin_and_swap_roles():
    begin = {"role": "user", "content": "(The helper is listening. Begin.)"}
    assert simulated_user_messages([]) == [begin]
    assert simulated_user_messages(TURNS) == [
        begin,
        {"role": "assistant", "content": "u1"},
        {"role": "user", "content": "h1"},
        {"role": "assistant", "content": "u2"},
    ]


def test_helper_prompt_guides_state_aim_and_boundaries():
    """The live helper reads the person's state, picks one aim, and keeps to its limits."""
    text = render_helper_system(POLICY)
    for phrase in ("main feeling", "rising, steady or easing", "Heard:", "Steadier:", "Understood:",
                   "Next step:", "go back to an earlier aim", "Never diagnose",
                   "follow the Safety section of the policy", "the policy wins", "at most about 40 words"):
        assert phrase in text
    # The guidance comes before the policy, and the helper is never told a risk level.
    assert text.index("Heard:") < text.index("<policy ")
    assert "risk" not in text.lower() and "crisis" not in text.lower()
