"""One training round as a LangGraph graph: converse, judge, critic, researcher, proposer, gate, finish.

The contract is docs/specs/graph.md. The judge and gate are stand-ins until stage 2 replaces them
through `RoundDeps`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypedDict

import mlflow
from langgraph.graph import END, START, StateGraph

from loop.errors import PolicyError
from loop.llm import ChatClient
from loop.policy import Policy, load_policy
from loop.runner import Failure, run_batch
from puddles import exa
from puddles.config import load_config
from puddles.decision_log import DecisionLog
from puddles.judging import (
    CRITIC_CONVERSATIONS,
    Judge,
    judge_models,
    save_scores,
    score_conversations,
    scored_conversations,
    select_for_critic,
    train_scores_path,
)
from puddles.llm import LLM, AgentFailed
from puddles.memory import Memory
from puddles.round import (
    RoundResult,
    finish_round,
    retry_pending,
    run_critic,
    run_proposer,
    run_researcher,
    stand_in_gate,
)
from puddles.round_view import gate_metrics, with_summary
from puddles.tracing import TracedClient, TracedLLM
from puddles.usage import Meter, metering, model_name
from schemas import ConversationRecord, GateOutcome, RoundEvent, Scenario, ScoredConversation

NO_CONVERSATION_REASON = "No conversation finished."
NOTHING_SCORED_REASON = "The judge scored no conversation."
STUB_NOTE = "Stub judge: fixed scores, the conversation was not read."


def stub_judge(records: list[ConversationRecord]) -> list[ScoredConversation]:
    """Fixed scores for every record, in order. Stage 2 replaces it."""
    return [ScoredConversation(conversation_id=r.id, empathy=3, relevance=3, note=STUB_NOTE,
                               turns=list(r.turns)) for r in records]


def stand_in_reverted(result: RoundResult) -> GateOutcome:
    return stand_in_gate(result, "reverted")


def check_policy_file(path: Path) -> Policy:
    """Refuse Windows line endings, then load and validate the policy file."""
    if b"\r" in Path(path).read_bytes():
        raise PolicyError(f"{path}: contains a carriage return; use Unix line endings")
    return load_policy(path)


@dataclass
class RoundDeps:
    helper_client: ChatClient
    user_client: ChatClient
    llm: LLM
    search: Callable[[str], list[dict]]
    memory: Memory
    log: DecisionLog
    root: Path
    judge: Callable[[list[ConversationRecord]], list[ScoredConversation]] = stub_judge
    gate: Callable[[RoundResult], GateOutcome] = stand_in_reverted
    usage: Optional[Meter] = None  # None: a meter on `root`, with the prices in the run's config file
    # Set: the starting policy gets its .meta.json, a promoted one gets its own, and the replay check runs.
    start_policy: Optional[Path] = None
    # Set: the judge node runs this judge through puddles.judging and `judge` is not called.
    judge_panel: Optional[Judge] = None
    critic_conversations: int = CRITIC_CONVERSATIONS


class RoundState(TypedDict, total=False):
    round: int
    policy_path: Path
    scenarios: list[Scenario]
    max_turns: int
    seed: Optional[int]
    records: list[ConversationRecord]
    failures: list[Failure]
    scored: list[ScoredConversation]
    result: RoundResult
    gate: Optional[GateOutcome]
    event: RoundEvent
    unscored: list[str]


def _failed(agent: str, exc: AgentFailed) -> str:
    return f"{agent} failed: {exc}"


def round_meter(deps: RoundDeps) -> Meter:
    """The round's meter. Made once and kept on `deps`, so the graph and `run_round` share it."""
    if deps.usage is None:
        deps.usage = Meter(deps.root, load_config().prices)
    return deps.usage


def build_round_graph(deps: RoundDeps):
    """The compiled graph, with no checkpointer.

    Each node runs inside `metering(meter, <agent>)`, so every model call and Exa search made in
    it is written to `<root>/log/usage.jsonl` and added to the round's tally.
    """
    root = Path(deps.root)
    meter = round_meter(deps)
    # A scripted model does not report its calls, so it is wrapped. OpenAILLM reports its own.
    llm: LLM = deps.llm if getattr(deps.llm, "reports_usage", False) else TracedLLM(deps.llm)

    def converse(state: RoundState) -> dict:
        policy = check_policy_file(state["policy_path"])
        round_ = state["round"]
        meter.start_round(round_)
        with metering(meter):
            batch = run_batch(TracedClient(deps.helper_client, "helper"),
                              TracedClient(deps.user_client, "simulated_user"),
                              policy, state["scenarios"], round=round_,
                              max_turns=state.get("max_turns", 10), seed=state.get("seed"),
                              out_path=root / "conversations" / f"round-{round_:02d}-train.jsonl")
        return {"records": batch.records, "failures": batch.failures}

    def judge(state: RoundState) -> dict:
        records = state.get("records", [])
        unscored: list[str] = []
        with metering(meter, "judge"):
            if deps.judge_panel is None:
                scored = deps.judge(records)
            else:
                run = score_conversations(deps.judge_panel, records)
                save_scores(run.scores, train_scores_path(root, state["round"]))
                scored = select_for_critic(scored_conversations(records, run.scores), deps.critic_conversations)
                unscored = run.unscored
        update: dict = {"scored": scored, "unscored": unscored}
        if not scored:
            update["result"] = RoundResult(
                round=state["round"], outcome="no_pattern", base_policy_path=state["policy_path"],
                reason=NO_CONVERSATION_REASON if not records else NOTHING_SCORED_REASON)
            update["result"].unscored = list(unscored)
        return update

    def critic(state: RoundState) -> dict:
        try:
            with metering(meter, "critic"):
                result = run_critic(state["round"], state["scored"], state["policy_path"], deps.log,
                                    deps.memory, llm, root)
        except AgentFailed as exc:
            result = RoundResult(round=state["round"], outcome="no_pattern",
                                 base_policy_path=state["policy_path"], reason=_failed("critic", exc))
        result.unscored = list(state.get("unscored", []))
        return {"result": result}

    def researcher(state: RoundState) -> dict:
        result = state["result"]
        try:
            with metering(meter, "researcher"):
                run_researcher(result, deps.log, deps.memory, llm, deps.search, root)
        except AgentFailed as exc:
            result.research = None
            result.refs.pop("research", None)
            result.outcome, result.reason = "no_evidence", _failed("researcher", exc)
        return {"result": result}

    def proposer(state: RoundState) -> dict:
        result = state["result"]
        try:
            with metering(meter, "proposer"):
                run_proposer(result, deps.log, deps.memory, llm, root)
        except AgentFailed as exc:
            result.edit, result.candidate_path = None, None
            result.refs.pop("edit", None)
            result.reason = _failed("proposer", exc)
        return {"result": result}

    def gate(state: RoundState) -> dict:
        with metering(meter, "gatekeeper"):
            return {"gate": deps.gate(state["result"])}

    def finish(state: RoundState) -> dict:
        event = finish_round(state["result"], state.get("gate"), deps.log, deps.memory, root,
                             start_policy=deps.start_policy)
        return {"event": event}

    def after_judge(state: RoundState) -> str:
        return "critic" if state.get("scored") else "finish"

    def after_critic(state: RoundState) -> str:
        return "researcher" if state["result"].pattern else "finish"

    def after_researcher(state: RoundState) -> str:
        research = state["result"].research
        return "proposer" if research is not None and research.sources else "finish"

    def after_proposer(state: RoundState) -> str:
        return "gate" if state["result"].edit is not None else "finish"

    b = StateGraph(RoundState)
    for fn in (converse, judge, critic, researcher, proposer, gate, finish):
        b.add_node(fn.__name__, with_summary(fn.__name__, fn, root))
    b.add_edge(START, "converse")
    b.add_edge("converse", "judge")
    b.add_conditional_edges("judge", after_judge, ["critic", "finish"])
    b.add_conditional_edges("critic", after_critic, ["researcher", "finish"])
    b.add_conditional_edges("researcher", after_researcher, ["proposer", "finish"])
    b.add_conditional_edges("proposer", after_proposer, ["gate", "finish"])
    b.add_edge("gate", "finish")
    b.add_edge("finish", END)
    return b.compile()


def run_round(deps: RoundDeps, *, policy_path: Path, scenarios: list[Scenario],
              max_turns: int = 10, seed: int | None = None) -> RoundState:
    """Run one round inside one MLflow run. The caller sets tracing up first."""
    policy = check_policy_file(policy_path)  # first: a bad file must leave memory and the log untouched
    meter = round_meter(deps)
    # Before any call: a model with no price, or an unpriced Exa search, stops the run here.
    meter.prices.check([model_name(c) for c in (deps.helper_client, deps.user_client, deps.llm)]
                       + judge_models(deps.judge_panel),
                       search=deps.search is exa.search)
    if deps.start_policy is not None:
        from puddles.gate import ensure_starting_meta  # puddles.gate imports this module

        ensure_starting_meta(Path(deps.root), Path(deps.start_policy))
    deps.memory.ensure_training_audience()
    retry_pending(deps.log, deps.memory, deps.root)
    round_ = deps.log.next_round()
    graph = build_round_graph(deps)
    with mlflow.start_run(run_name=f"round-{round_:02d}"):
        mlflow.log_params({"round": round_, "policy_version": policy.version})
        state: RoundState = graph.invoke({"round": round_, "policy_path": Path(policy_path),
                                          "scenarios": list(scenarios), "max_turns": max_turns,
                                          "seed": seed, "gate": None})
        mlflow.set_tag("outcome", state["event"].outcome)
        mlflow.set_tag("root", str(Path(deps.root).resolve()))
        mlflow.log_metrics({"conversations": len(state.get("records", [])),
                            "conversations_failed": len(state.get("failures", [])),
                            "conversations_unscored": len(state.get("unscored", [])),
                            **meter.metrics(), **gate_metrics(state, Path(deps.root))})
    return state
