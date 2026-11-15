"""Spec-only inputs. No imports from the implementation during collection."""

from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo_root():
    return ROOT


@pytest.fixture
def client_factory():
    class ScriptedClient:
        def __init__(self, replies, *, name=None, order=None):
            self.replies = iter(replies)
            self.calls = []
            self.name = name
            self.order = order

        def complete(self, system, messages, *, max_tokens, temperature=0.7, seed=None):
            self.calls.append(deepcopy(dict(
                system=system, messages=messages, max_tokens=max_tokens,
                temperature=temperature, seed=seed,
            )))
            if self.order is not None:
                self.order.append(self.name)
            try:
                reply = next(self.replies)
            except StopIteration:
                pytest.fail(f"Unexpected extra call to {self.name or 'scripted client'}")
            if isinstance(reply, Exception):
                raise reply
            return reply

    return ScriptedClient


@pytest.fixture
def policy_path(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(
        '# acceptance-v1\n\n'
        '## Safety (protected: no edit may change this section)\n'
        'S1. Keep the person safe.\n\n'
        '## Conversation\n'
        'C1. Listen before suggesting.\n', encoding="utf-8",
    )
    return path


@pytest.fixture
def policy(policy_path):
    from loop.policy import load_policy
    return load_policy(policy_path)


@pytest.fixture
def scenario_factory():
    from schemas import Scenario

    def make(scenario_id="acceptance-01", **changes):
        values = dict(
            scenario_id=scenario_id, split="validation", emotion_type="worry",
            problem_type="work", situation="My hours have been cut.",
            initial_intensity=4, openness="guarded",
        )
        values.update(changes)
        return Scenario(**values)

    return make


@pytest.fixture
def turns():
    from schemas import Turn
    return [
        Turn(speaker="user", text="I am worried."),
        Turn(speaker="helper", text="What happened?"),
        Turn(speaker="user", text="My hours were cut."),
    ]


@pytest.fixture
def clean_env(monkeypatch):
    for prefix in ("HELPER", "SIMUSER"):
        for suffix in ("URL", "MODEL", "API_KEY"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
