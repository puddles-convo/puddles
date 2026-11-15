"""Run one training round as a LangGraph graph, traced in MLflow.

    python -m scripts.run_graph_round --fake --root runs/dev
    python -m scripts.run_graph_round --root runs/dev --policy policies/policy-v1.md --max-turns 6

--fake uses scripted chat clients, a scripted model, a fake search and in-process memory: no keys,
no network. Without it the clients come from HELPER_*, SIMUSER_*, OPENAI_API_KEY,
PUDDLES_AGENT_MODEL, EXA_API_KEY and PAST_API_KEY.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Callable, NoReturn, Sequence, TypeVar

from openai import OpenAIError
from pydantic import BaseModel, ValidationError

from loop.errors import ConfigError, PolicyError
from loop.llm import ChatClient, FakeClient
from loop.runner import load_scenarios
from loop.run_batch import _fake_helper, _fake_user
from puddles import tracing
from puddles.decision_log import DecisionLog
from puddles.graph import RoundDeps, check_policy_file, run_round
from puddles.llm import LLM, AgentFailed
from puddles.memory import FakeMemory, Memory, PastError
from puddles.round import RoundResult, policy_in_force, stand_in_gate
from schemas import CriticOutput, ProposerOutput, ResearcherOutput

T = TypeVar("T", bound=BaseModel)

FAKE_URL = "https://counselling.example.org/skills/{slug}"


class FakeLLM:
    """A scripted model whose answers pass each agent's checks, worked out from the prompt."""

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        if schema is CriticOutput:
            ids = re.findall(r'<conversation id="([^"]+)"', user)
            return schema.model_validate({
                "found": True, "name": "advice-before-understanding",
                "description": "The helper moves on to suggestions before asking about the situation.",
                "seen_in": 1, "out_of": len(ids), "example_conversation_ids": ids[:1]})
        if schema is ProposerOutput:
            pattern_id = _pattern_id(user)
            return schema.model_validate({
                "action": "add", "summary": f"Ask first, for {pattern_id}.",
                "rule_text": f"Before any suggestion, ask one open question about the situation ({pattern_id})."})
        raise TypeError(f"FakeLLM has no answer for {schema.__name__}")

    def structured_with_search(self, system: str, user: str, schema: type[T],
                               search: Callable[[str], list[dict]], max_searches: int) -> T:
        if schema is not ResearcherOutput:
            raise TypeError(f"FakeLLM has no searched answer for {schema.__name__}")
        query = f"counselling skills: open questions before advice ({_pattern_id(user)})"
        results = search(query)
        return schema.model_validate({
            "queries": [query],
            "sources": [{"url": r["url"], "title": r["title"], "technique": "Ask one open question first."}
                        for r in results[:1]]})


def _pattern_id(user: str) -> str:
    match = re.search(r'<failure_pattern id="([^"]+)"', user)
    return match.group(1) if match else "pattern"


def fake_search(query: str) -> list[dict]:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return [{"url": FAKE_URL.format(slug=slug), "title": "Counselling skills", "text": "Ask before advising."}]


def _print_error(message: str) -> None:
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = _Parser(prog="python -m scripts.run_graph_round", description=__doc__.splitlines()[0])
    p.add_argument("--root", default="runs/dev", help="where log/, records/, conversations/ and policies/ go")
    p.add_argument("--policy", default="policies/policy-v1.md", help="the starting policy")
    p.add_argument("--scenarios", default="fixtures/scenarios.jsonl")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gate", choices=["kept", "reverted"], default="reverted",
                   help="what the stand-in gate decides")
    p.add_argument("--fake", action="store_true", help="scripted clients, model, search and memory")
    p.add_argument("--mlflow", default=None, help="MLflow tracking address")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    root = Path(args.root)
    current_file = None
    try:
        policy_path = policy_in_force(root, Path(args.policy))
        current_file = str(policy_path)
        check_policy_file(policy_path)
        current_file = args.scenarios
        scenarios = load_scenarios(args.scenarios)
        current_file = None
        helper_client: ChatClient
        user_client: ChatClient
        llm: LLM
        memory: Memory
        if args.fake:
            helper_client, user_client = FakeClient(_fake_helper), FakeClient(_fake_user)
            llm, search, memory = FakeLLM(), fake_search, FakeMemory()
        else:
            from loop.llm import client_from_env
            from puddles import exa
            from puddles.llm import OpenAILLM
            from puddles.memory import PastClient

            helper_client, user_client = client_from_env("HELPER"), client_from_env("SIMUSER")
            if not os.environ.get("EXA_API_KEY"):
                raise ConfigError("environment variable EXA_API_KEY is not set")
            llm, search, memory = OpenAILLM(), exa.search, PastClient()
    except (ConfigError, PolicyError, OSError, UnicodeDecodeError, ValidationError, ValueError,
            AgentFailed, PastError, OpenAIError) as exc:
        detail = str(exc)
        where = f"{current_file}: " if current_file and current_file not in detail else ""
        _print_error(f"{where}{type(exc).__name__}: {detail}")
        return 2

    decision = args.gate

    def gate(result: RoundResult):
        return stand_in_gate(result, decision)

    tracing.setup(tracking_uri=args.mlflow)
    deps = RoundDeps(helper_client=helper_client, user_client=user_client, llm=llm, search=search,
                     memory=memory, log=DecisionLog(root), root=root, gate=gate)
    try:
        state = run_round(deps, policy_path=policy_path, scenarios=scenarios, max_turns=args.max_turns,
                          seed=args.seed)
    except ConfigError as exc:  # a model or the Exa search has no price in the run's config file
        _print_error(f"{type(exc).__name__}: {exc}")
        return 2
    event = state["event"]
    print(f"round {event.round}: {event.outcome}. {event.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
