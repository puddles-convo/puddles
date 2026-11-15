"""Builders for the helper and simulated user prompts."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from loop.policy import Policy
from schemas import Scenario, Turn

PROMPT_DIR = Path(__file__).parent / "prompts"
NO_NOTES = "Nothing yet. This is your first conversation."
NOTES_PLACEHOLDER = '{user_notes, or "Nothing yet. This is your first conversation."}'
BEGIN_MESSAGE = "(The helper is listening. Begin.)"


def _template(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _fill(template: str, values: dict[str, str]) -> str:
    """Replace each placeholder in one pass, so filled-in text is never rescanned."""
    pattern = re.compile("|".join(re.escape(key) for key in values))
    return pattern.sub(lambda m: values[m.group(0)], template)


def render_helper_system(policy: Policy, user_notes: Sequence[str] = ()) -> str:
    """The helper's system message for this policy and these notes."""
    notes = "\n".join(f"- {note}" for note in user_notes) if user_notes else NO_NOTES
    return _fill(_template("helper_system.txt"), {
        "{policy_version}": policy.version,
        "{policy_text}": policy.text.strip(),
        NOTES_PLACEHOLDER: notes,
    })


def helper_messages(turns: Sequence[Turn]) -> list[dict[str, str]]:
    """The conversation as the helper sees it: user turns as user, its own as assistant."""
    roles = {"user": "user", "helper": "assistant"}
    return [{"role": roles[t.speaker], "content": t.text} for t in turns]


def render_simulated_user_system(scenario: Scenario) -> str:
    """The simulated user's system message for this scenario."""
    return _fill(_template("simulated_user_system.txt"), {
        "{situation}": scenario.situation,
        "{emotion_type}": scenario.emotion_type,
        "{initial_intensity}": str(scenario.initial_intensity),
        "{problem_type}": scenario.problem_type,
        "{openness}": scenario.openness,
    })


def simulated_user_messages(turns: Sequence[Turn]) -> list[dict[str, str]]:
    """The conversation with roles swapped, opened by a fixed user message."""
    roles = {"helper": "user", "user": "assistant"}
    return [{"role": "user", "content": BEGIN_MESSAGE},
            *({"role": roles[t.speaker], "content": t.text} for t in turns)]
