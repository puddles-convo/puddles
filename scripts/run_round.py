"""Run steps 4 to 6 on scored conversations, then log the round and copy it into past.dev.

Needs PAST_API_KEY, OPENAI_API_KEY, PUDDLES_AGENT_MODEL and EXA_API_KEY. Until the gatekeeper
(J7) exists, --gate picks a stand-in outcome; it is labelled as a stand-in in the log and memory.

    python -m scripts.run_round --root runs/demo --gate reverted
    python -m scripts.run_round --root runs/demo --gate kept
"""
import argparse
import json
from pathlib import Path

from puddles import exa
from puddles.decision_log import DecisionLog
from puddles.llm import OpenAILLM
from puddles.memory import PastClient
from puddles.round import finish_round, memory_trace, prepare_round, run_agents, stand_in_gate
from schemas import ScoredConversation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/demo", help="where records/, log/ and policies/ go")
    ap.add_argument("--conversations", default="fixtures/bad_conversations.json")
    ap.add_argument("--policy", default="fixtures/policy-v3.md", help="starting policy")
    ap.add_argument("--gate", choices=["kept", "reverted"], default="reverted")
    args = ap.parse_args()

    root = Path(args.root)
    log, memory = DecisionLog(root), PastClient()
    memory.ensure_training_audience()
    start = Path(args.policy)
    policy_path = prepare_round(log, memory, root, start)
    conversations = [ScoredConversation.model_validate(c)
                     for c in json.loads(Path(args.conversations).read_text())]
    round_ = log.next_round()
    result = run_agents(round_, conversations, policy_path, log, memory, OpenAILLM(), exa.search, root)
    gate = stand_in_gate(result, args.gate) if result.outcome == "candidate" else None
    event = finish_round(result, gate, log, memory, root, start)
    print(f"round {round_}: {event.outcome}. {event.reason}")
    for row in memory_trace(log, root):
        print(f"  {row['record']:<14} written by {row['written_by']:<11} read by {', '.join(row['read_by']) or '-'}")


if __name__ == "__main__":
    main()
