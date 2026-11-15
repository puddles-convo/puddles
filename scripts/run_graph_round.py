"""Run one training round as a LangGraph graph, traced in MLflow.

    python -m scripts.run_graph_round --fake --root runs/dev
    python -m scripts.run_graph_round --root runs/dev --policy policies/policy-v1.md --max-turns 6

--gate kept or --gate reverted is a stand-in that decides so. --gate real runs the gatekeeper in
puddles/gate.py; it promotes nothing unless --noise-measured says --noise is a measured figure.
Without --gate, the gate is reverted with --fake and real without it.

--fake uses scripted chat clients, a scripted model, a fake search and in-process memory: no keys,
no network. Without it the clients come from HELPER_*, SIMUSER_*, OPENAI_API_KEY,
PUDDLES_AGENT_MODEL, EXA_API_KEY and PAST_API_KEY, the judge from puddles.eval.judge.judge_from_env,
and for --gate real the safety check's model from PUDDLES_SAFETY_MODEL or PUDDLES_JUDGE_ASTRA
(puddles/gate_join.py).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NoReturn, Sequence, TypeVar

from openai import OpenAIError
from pydantic import BaseModel, ValidationError

from loop.errors import ConfigError, PolicyError
from loop.llm import ChatClient, FakeClient
from loop.runner import load_scenarios
from loop.run_batch import _fake_helper, _fake_user
from puddles import gate as real_gate
from puddles import tracing
from puddles.decision_log import DecisionLog
from puddles.graph import RoundDeps, check_policy_file, round_meter, run_round
from puddles.judging import Judge, MeteredJudge
from puddles.llm import LLM, AgentFailed
from puddles.memory import FakeMemory, Memory, PastError, now
from puddles.policy import TOKEN_CAP
from puddles.round import RoundResult, policy_in_force, stand_in_gate
from schemas import (
    ConversationRecord,
    CriticOutput,
    GateOutcome,
    ProposerOutput,
    ResearcherOutput,
    SafetyResult,
    ScoreRecord,
)

T = TypeVar("T", bound=BaseModel)

FAKE_URL = "https://counselling.example.org/skills/{slug}"


@dataclass
class _Judged:
    score: ScoreRecord
    usage: list = field(default_factory=list)  # a scripted judge makes no model call, so it reports none


class FakeJudge:
    """For --fake, the round's judge and the gate's: fixed scores. It reads no conversation."""

    version = "judge-fake"

    def score(self, record: ConversationRecord) -> _Judged:
        # Only the ID is read. No strategy labels: no turn was read, so none can be claimed.
        return _Judged(ScoreRecord(
            id=f"score-{record.id}", written_by="judge", created_at=now(), parent_ids=[record.id],
            conversation_id=record.id, judge_version=self.version, empathy=3, relevance=3, emotion_before=3,
            emotion_after=3, strategies=[],
            note="Scripted judge: fixed scores, the conversation was not read."))


def fake_safety(policy_version: str, reply: real_gate.Reply) -> SafetyResult:
    """For --fake --gate real: asks the helper nothing and passes. Not a safety check."""
    return SafetyResult(policy_version=policy_version, case_file="stand-in: no safety cases were run",
                        passed=True, cases=[])


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
        if schema.__name__ == "ToolboxOutput":  # PUDDLES_PROPOSER=toolbox: a card first, then rules
            pattern_id = _pattern_id(user)
            urls = re.findall(r"^- (https?://\S+)$", user.split("<source_urls>")[-1], re.M)
            if re.search(r"^K\d+\. ", user, re.M) or not urls:
                return schema.model_validate({
                    "considered": [{"tool": "add_skill_card", "rejected_because": "a card exists"},
                                   {"tool": "set_examples", "rejected_because": "scripted"}],
                    "tool": "edit_rule", "rule_op": "add",
                    "rule_text": f"Before any suggestion, ask one open question about the situation ({pattern_id}).",
                    "hypothesis": "Scripted.", "prediction": "Scripted."})
            return schema.model_validate({
                "considered": [{"tool": "edit_rule", "rejected_because": "scripted"},
                               {"tool": "set_examples", "rejected_because": "scripted"}],
                "tool": "add_skill_card", "card_op": "add", "card_name": f"Ask first ({pattern_id})",
                "when_to_use": "the person describes a problem", "when_not_to_use": "they ask a direct question",
                "steps": ["Reflect what they said", "Ask one open question"], "source_urls": urls[:1],
                "hypothesis": "Scripted.", "prediction": "Scripted."})
        raise TypeError(f"FakeLLM has no answer for {schema.__name__}")

    def structured_with_search(self, system: str, user: str, schema: type[T],
                               search: Callable[[str], list[dict]], max_searches: int) -> T:
        if schema is not ResearcherOutput:
            raise TypeError(f"FakeLLM has no searched answer for {schema.__name__}")
        query = f"counselling skills: open questions before advice ({_pattern_id(user)})"
        results = search(query)
        if results and "error" in results[0]:  # already run for this pattern in an earlier round
            query += f" ({len(re.findall(r'^- ', user, re.M))} earlier)"
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
    p.add_argument("--gate", choices=["kept", "reverted", "real"], default=None,
                   help="kept or reverted: what the stand-in gate decides. real: the gatekeeper. "
                        "Default: reverted with --fake, real without it")
    p.add_argument("--validation", default=None,
                   help="the scenarios the gate tests on, all of them; default: the --scenarios file")
    p.add_argument("--noise", type=float, default=0.0, help="the run-to-run noise in quality")
    p.add_argument("--noise-measured", action="store_true",
                   help="--noise is a measured figure; without it the real gate keeps nothing")
    p.add_argument("--token-cap", type=int, default=TOKEN_CAP, help="the most tokens a policy may have")
    p.add_argument("--fake", action="store_true", help="scripted clients, model, search and memory")
    p.add_argument("--mlflow", default=None, help="MLflow tracking address")
    args = p.parse_args(argv)
    if args.gate is None:
        args.gate = "reverted" if args.fake else "real"
    return args


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
        validation = scenarios
        if args.gate == "real":
            if args.validation is not None:
                current_file = args.validation
                validation = load_scenarios(args.validation)
            if args.noise < 0:
                raise ConfigError(f"--noise cannot be negative: {args.noise}")
        current_file = None
        helper_client: ChatClient
        user_client: ChatClient
        llm: LLM
        memory: Memory
        judge: Judge  # whatever judge_from_env returns; its kind is not looked at
        asker = None  # the real safety check's yes-or-no client; None with --fake
        if args.fake:
            helper_client, user_client = FakeClient(_fake_helper), FakeClient(_fake_user)
            llm, search, memory = FakeLLM(), fake_search, FakeMemory()
            judge = FakeJudge()
        else:
            from loop.llm import client_from_env
            from puddles import exa
            from puddles.llm import OpenAILLM
            from puddles.memory import PastClient

            helper_client, user_client = client_from_env("HELPER"), client_from_env("SIMUSER")
            if not os.environ.get("EXA_API_KEY"):
                raise ConfigError("environment variable EXA_API_KEY is not set")
            llm, search, memory = OpenAILLM(), exa.search, PastClient()
            from puddles.eval import judge as eval_judge

            # Looked up on the module now, so a test can replace it there. A missing variable is a ValueError.
            judge = eval_judge.judge_from_env()
            if args.gate == "real":
                from puddles.eval import safety as eval_safety

                # Built here, so a missing PUDDLES_SAFETY_MODEL (a ValueError) stops the run before any write.
                asker = eval_safety.OpenAIYesNo.from_env()
    except (ConfigError, PolicyError, OSError, UnicodeDecodeError, ValidationError, ValueError,
            AgentFailed, PastError, OpenAIError) as exc:
        detail = str(exc)
        where = f"{current_file}: " if current_file and current_file not in detail else ""
        _print_error(f"{where}{type(exc).__name__}: {detail}")
        return 2

    decision = args.gate

    def stand_in(result: RoundResult) -> GateOutcome:
        return stand_in_gate(result, decision)

    gate: Callable[[RoundResult], GateOutcome] = stand_in
    if decision == "real":
        traced_helper = tracing.TracedClient(helper_client, "helper")
        traced_user = tracing.TracedClient(user_client, "simulated_user")
        if asker is None:  # --fake
            gate_deps = real_gate.GateDeps(
                helper_client=traced_helper, user_client=traced_user, judge=MeteredJudge(judge),
                safety=fake_safety, root=root, scenarios=validation, noise_measured=args.noise_measured,
                max_turns=args.max_turns, seed=args.seed)
        else:
            from puddles.gate_join import real_gate_deps

            gate_deps = real_gate_deps(
                helper_client=traced_helper, user_client=traced_user, judge=judge, root=root,
                scenarios=validation, noise_measured=args.noise_measured, max_turns=args.max_turns,
                seed=args.seed, asker=asker)
        gate = real_gate.make_gate(gate_deps, validation_scenario_ids=[s.scenario_id for s in validation],
                                   noise=args.noise, token_cap=args.token_cap)

    deps = RoundDeps(helper_client=helper_client, user_client=user_client, llm=llm, search=search,
                     memory=memory, log=DecisionLog(root), root=root, gate=gate, start_policy=Path(args.policy),
                     judge_panel=judge)
    if not args.fake:
        from puddles import exa
        from puddles.gate_join import safety_models
        from puddles.judging import judge_models
        from puddles.usage import model_name

        # Every model the round will use, gathered as run_round gathers them, plus the safety model, which
        # run_round knows nothing of. Checked here, before tracing is set up (it prints and can write under
        # the root), before memory and before any model call. run_round's own check stays as a second guard.
        models = [model_name(c) for c in (deps.helper_client, deps.user_client, deps.llm)]
        models += judge_models(deps.judge_panel)
        if asker is not None:
            models += safety_models(asker)
        try:
            round_meter(deps).prices.check(models, search=deps.search is exa.search)
        except ConfigError as exc:
            _print_error(f"{type(exc).__name__}: {exc}")
            return 2
    tracing.setup(tracking_uri=args.mlflow)
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
