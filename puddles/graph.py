"""One training round as a LangGraph graph: converse, judge, critic, researcher, proposer, gate, finish.

The contract is docs/specs/graph.md. With `RoundDeps.judge_llm` set the real judge runs; without
it the stub does. The gate is a stand-in until stage 2 replaces it through `RoundDeps`.
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
from puddles.agents import judge as judge_agent
from puddles.decision_log import DecisionLog
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
from puddles.tracing import TracedClient
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
    judge_llm: Optional[LLM] = None  # set: the real judge runs on this model, and `judge` is not called
    judge_version: str = ""
    critic_conversations: int = judge_agent.CRITIC_CONVERSATIONS  # the lowest-scoring n go to the critic


class RoundState(TypedDict, total=False):
    round: int
    policy_path: Path
    scenarios: list[Scenario]
    max_turns: int
    seed: Optional[int]
    records: list[ConversationRecord]
    failures: list[Failure]
    scored: list[ScoredConversation]
    unscored: list[str]
    result: RoundResult
    gate: Optional[GateOutcome]
    event: RoundEvent


def _failed(agent: str, exc: AgentFailed) -> str:
    return f"{agent} failed: {exc}"


def build_round_graph(deps: RoundDeps):
    """The compiled graph, with no checkpointer."""
    root = Path(deps.root)

    def converse(state: RoundState) -> dict:
        policy = check_policy_file(state["policy_path"])
        round_ = state["round"]
        batch = run_batch(TracedClient(deps.helper_client, "helper"),
                          TracedClient(deps.user_client, "simulated_user"),
                          policy, state["scenarios"], round=round_,
                          max_turns=state.get("max_turns", 10), seed=state.get("seed"),
                          out_path=root / "conversations" / f"round-{round_:02d}-train.jsonl")
        return {"records": batch.records, "failures": batch.failures}

    def judge(state: RoundState) -> dict:
        records = state.get("records", [])
        update: dict
        if deps.judge_llm is None:
            scored = deps.judge(records)
            update = {"scored": scored}
        else:
            judged = judge_agent.score_conversations(records, deps.judge_llm, deps.judge_version)
            judge_agent.save_scores(judged.scores, judge_agent.train_scores_path(root, state["round"]))
            scored = judge_agent.select_for_critic(
                judge_agent.scored_conversations(records, judged.scores), deps.critic_conversations)
            update = {"scored": scored, "unscored": judged.unscored}
        if not scored:
            update["result"] = RoundResult(
                round=state["round"], outcome="no_pattern", base_policy_path=state["policy_path"],
                reason=NO_CONVERSATION_REASON if not records else NOTHING_SCORED_REASON)
        return update

    def critic(state: RoundState) -> dict:
        try:
            result = run_critic(state["round"], state["scored"], state["policy_path"], deps.log,
                                deps.memory, deps.llm, root)
        except AgentFailed as exc:
            result = RoundResult(round=state["round"], outcome="no_pattern",
                                 base_policy_path=state["policy_path"], reason=_failed("critic", exc))
        return {"result": result}

    def researcher(state: RoundState) -> dict:
        result = state["result"]
        try:
            run_researcher(result, deps.log, deps.memory, deps.llm, deps.search, root)
        except AgentFailed as exc:
            result.research = None
            result.refs.pop("research", None)
            result.outcome, result.reason = "no_evidence", _failed("researcher", exc)
        return {"result": result}

    def proposer(state: RoundState) -> dict:
        result = state["result"]
        try:
            run_proposer(result, deps.log, deps.memory, deps.llm, root)
        except AgentFailed as exc:
            result.edit, result.candidate_path = None, None
            result.refs.pop("edit", None)
            result.reason = _failed("proposer", exc)
        return {"result": result}

    def gate(state: RoundState) -> dict:
        return {"gate": deps.gate(state["result"])}

    def finish(state: RoundState) -> dict:
        state["result"].unscored = list(state.get("unscored", []))
        event = finish_round(state["result"], state.get("gate"), deps.log, deps.memory, root)
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
        b.add_node(fn.__name__, fn)
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
        mlflow.log_metrics({"conversations": len(state.get("records", [])),
                            "conversations_failed": len(state.get("failures", [])),
                            "conversations_unscored": len(state.get("unscored", []))})
    return state
