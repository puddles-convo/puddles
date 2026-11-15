"""Command line: run one batch of simulated conversations.

python -m loop.run_batch --policy policies/policy-v1.md --scenarios fixtures/scenarios.jsonl \
    --round 1 --out runs/round-01/conversations.jsonl [--max-turns 10] [--seed 7] [--fake]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import NoReturn, Sequence

from pydantic import ValidationError

from loop.errors import ConfigError, PolicyError
from loop.llm import ChatClient, FakeClient, client_from_env
from loop.policy import load_policy
from loop.runner import load_scenarios, run_batch

FAKE_USER_LINES = [
    "I don't really know where to start. Things have been a lot lately.",
    "It's mostly work. I can't switch off at night.",
    "Yeah. I guess I just needed to say it to someone.",
]
FAKE_HELPER_LINES = [
    "That sounds like a lot to carry. What has been weighing on you most?",
    "Not being able to switch off sounds exhausting. How long has it been like this?",
    "I'm glad you said it out loud. What would help a little tonight?",
]


def _fake_user(system: str, messages: list[dict[str, str]]) -> str:
    spoken = sum(1 for m in messages if m["role"] == "assistant")
    return FAKE_USER_LINES[spoken] if spoken < len(FAKE_USER_LINES) else "[END]"


def _fake_helper(system: str, messages: list[dict[str, str]]) -> str:
    replied = sum(1 for m in messages if m["role"] == "assistant")
    return FAKE_HELPER_LINES[replied % len(FAKE_HELPER_LINES)]


def _print_error(message: str) -> None:
    """Print message to standard error as exactly one line, whatever it contains."""
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    """Argument parser whose usage errors are a one-line message and exit code 2."""

    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _Parser(prog="python -m loop.run_batch", description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fake", action="store_true",
                        help="use built-in scripted clients: no network, no keys")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch and return the exit code."""
    args = _parse(argv)
    current_file = None  # the file being read or written, named in a file error
    try:
        current_file = args.policy
        policy = load_policy(args.policy)
        current_file = args.scenarios
        scenarios = load_scenarios(args.scenarios)
        current_file = None
        helper_client: ChatClient
        user_client: ChatClient
        if args.fake:
            helper_client, user_client = FakeClient(_fake_helper), FakeClient(_fake_user)
        else:
            helper_client, user_client = client_from_env("HELPER"), client_from_env("SIMUSER")
        current_file = args.out
        result = run_batch(helper_client, user_client, policy, scenarios, round=args.round,
                           max_turns=args.max_turns, seed=args.seed, out_path=args.out)
    except (ConfigError, PolicyError, OSError, UnicodeDecodeError, ValidationError) as exc:
        detail = str(exc)
        where = f"{current_file}: " if current_file and current_file not in detail else ""
        _print_error(f"{where}{type(exc).__name__}: {detail}")
        return 2
    print(json.dumps(result.summary()))
    return 0 if result.records else 1


if __name__ == "__main__":
    sys.exit(main())
