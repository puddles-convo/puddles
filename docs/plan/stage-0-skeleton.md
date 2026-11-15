# Stage 0: the skeleton

**Target:** 18:00. **Owner:** unassigned. **Needs:** nothing. **Blocks:** every later stage.

Issue: [#30](https://github.com/puddles-convo/puddles/issues/30).

## Goal

One training round runs as a LangGraph graph, from conversations to the log entry, with an MLflow trace of every step. Agents that do not exist yet are dummies. Nothing calls a real model.

## What it wires together

| Node | Calls | Real or dummy now |
|---|---|---|
| `converse` | `loop.runner.run_batch(helper_client, user_client, policy, scenarios, round=..., max_turns=..., seed=..., out_path=...)`, which returns a `BatchResult` with `records`, `failures` and `round` | Real code, `loop.llm.FakeClient` for both models |
| `judge` | A stub that turns each `ConversationRecord` into a `ScoredConversation` with fixed scores | Dummy until stage 2 |
| `critic` | `puddles.agents.critic`: `build_input`, `run`, `to_record` | Real code, `ScriptedLLM` |
| `researcher` | `puddles.agents.researcher`: `build_input`, `run`, `to_record`, `save_search_log` | Real code, `ScriptedLLM` and `FakeSearch` |
| `proposer` | `puddles.agents.proposer`: `build_input`, `run`, `apply` | Real code, `ScriptedLLM` |
| `gate` | `puddles.round.stand_in_gate(result, decision)` | Dummy until stage 2 |
| `finish` | `puddles.round.finish_round(result, gate, log, memory, root)`, which returns the `RoundEvent` | Real code, `puddles.memory.FakeMemory` |

```
START -> converse -> judge -> critic --(pattern)--> researcher --(sources)--> proposer --(edit)--> gate -> finish -> END
                                 |                       |                        |
                                 +---(no pattern)--------+---(no sources)---------+---(no edit)-----------> finish
```

On the three short paths the gate has not run, so `finish` calls `finish_round(result, None, ...)`. The existing function already handles that; do not make up a gate result.

## What the state carries

| Key | Set by | What it is |
|---|---|---|
| `round`, `policy_path`, `scenarios`, `max_turns`, `seed` | The caller | The round's settings |
| `records`, `failures` | `converse` | From the `BatchResult`. Keep the failures; later stages count them. |
| `scored` | `judge` | `list[ScoredConversation]` |
| `result` | `critic`, `researcher`, `proposer` | `puddles.round.RoundResult`. Its `outcome` is only `candidate`, `no_pattern` or `no_evidence`. |
| `gate` | `gate` | `GateOutcome`, or `None` if the gate did not run |
| `event` | `finish` | The `RoundEvent`. **The round's final outcome is `event.outcome`**, not `result.outcome`. |

## Checklist

- [ ] Add `langgraph` and `mlflow` to `pyproject.toml` and `requirements.txt`.
- [ ] Split `puddles.round.run_agents` into three functions, one per agent, at its two early returns. Keep `run_agents` as a wrapper that calls the three, so `tests/test_memory_wiring.py` passes unchanged.
- [ ] `puddles/graph.py`: `build_round_graph(deps)` returns the compiled graph. `deps` carries the two chat clients, the structured model, the search function, the memory, the log and the root folder, so tests can pass fakes.
- [ ] `converse` loads the policy with `loop.policy.load_policy(path)` and passes `out_path=root / "conversations" / "round-NN-train.jsonl"`, so the conversations are saved. The gate's own conversations get a different file and different IDs in stage 2. `run_batch` writes the failures file beside it. The agents read the same policy file as text and parse it with `puddles.policy.parse_policy`.
- [ ] Each agent node catches `puddles.llm.AgentFailed` and follows the table in the [plan README](README.md#when-an-agents-answer-fails-twice). A failed researcher leaves `result.research` empty, and a failed proposer leaves `result.edit` empty, so the routing below tests for both. `stand_in_gate` asserts that an edit exists; never send it a round without one.
- [ ] Before the graph starts, call `memory.ensure_training_audience()` and then `puddles.round.retry_pending(log, memory, root)`, as `scripts/run_round.py` does.
- [ ] `puddles/tracing.py`: `setup()` sets the tracking address from `MLFLOW_TRACKING_URI` (default `sqlite:///runs/mlflow.db`) and calls `mlflow.langchain.autolog()` and `mlflow.openai.autolog()`.
- [ ] `TracedClient` in `puddles/tracing.py`: wraps any `ChatClient` and opens one span per `complete()` call. `loop.llm` uses plain HTTP, so MLflow's automatic OpenAI tracing does not see those calls.
- [ ] One MLflow run per round: the policy version as a parameter, `event.outcome` as a tag, conversations run and failed as metrics.
- [ ] `scripts/run_graph_round.py --fake --root runs/dev`: one round with every fake, no keys.
- [ ] The same script without `--fake` takes `--policy`, `--scenarios` and `--max-turns`, and builds the real clients from the environment. Stage 3 uses it.
- [ ] `tests/test_graph.py`, listed below.

## Tests that must pass

All offline, in `tests/test_graph.py`.

1. A scripted round that reaches the proposer ends with one `round` event in the log and `event.outcome` of `kept` or `reverted`, matching the stand-in gate.
2. A critic answer with `found: false` ends as `no_pattern`. The researcher and proposer are never called.
3. A researcher answer with no sources ends as `no_evidence`. The proposer is never called.
4. Each of the three runs leaves one MLflow trace that has a span for every node that ran, and the model-call spans sit under the `converse` span.
5. Two rounds in a row give round numbers 1 and 2 from `DecisionLog.next_round()`.
6. A batch in which one scenario fails still finishes the round. The failure is in the state, in the failures file and in the MLflow metric.
7. The conversations file for the round exists and holds one line per finished conversation.
8. Each agent failing twice gives the outcome in the README table, and the failure is in the event's reason. A failed researcher never reaches the proposer. A failed proposer never reaches the gate, and its round is logged as `no_pattern` with the pattern attached, which is what `finish_round` does today.
9. Both policy loaders agree: `policies/policy-v1.md` and `policies/baseline-plain.md` load in both, and the candidate file the proposer writes loads with `loop.policy.load_policy` with its Safety text byte-for-byte the Safety text of the file it was made from.
10. A starting policy with a rule wrapped onto a second line, or with Windows line endings, is refused before the round starts.

`python -m pytest` still passes in full.

## Checked already

A throwaway version of this graph ran on 14 November with LangGraph 1.2.22, MLflow 3.18.0 and Python 3.12: six nodes, the real `run_batch` with `FakeClient`, dummies for the rest. All three paths ran, and each left one trace with a span per node and 30 model-call spans under `converse`.

- **Call `mlflow.flush_trace_async_logging()` before reading traces in a test.** Traces are written in the background. Without the call, the first read returned 2 of 3 traces.
- **Install early.** The first install of `langgraph` and `mlflow` took 29 minutes on the venue network. Later runs took 9 seconds.
- **Spans opened with `mlflow.start_span` inside a node attach to that node's span** when the graph is run with `invoke`. Async runs were not tried.

The wiring, adapted from what ran:

```python
from langgraph.graph import END, START, StateGraph

b = StateGraph(RoundState)
for fn in (converse, judge, critic, researcher, proposer, gate, finish):
    b.add_node(fn.__name__, fn)
b.add_edge(START, "converse")
b.add_edge("converse", "judge")
b.add_edge("judge", "critic")
b.add_conditional_edges("critic", lambda s: "researcher" if s["result"].pattern else "finish")
b.add_conditional_edges("researcher", lambda s: "proposer" if s["result"].research and s["result"].research.sources else "finish")
b.add_conditional_edges("proposer", lambda s: "gate" if s["result"].edit else "finish")
b.add_edge("gate", "finish")
b.add_edge("finish", END)
graph = b.compile()
```

The spike ended its two short paths at `END` and had no `finish` node. Routing through `finish`, the third short path, the failure handling and the saved conversations file are new here and have not been run.

## Not in this stage

- No checkpointer. A stopped loop resumes from the decision log (stage 6).
- No real model calls.
- No merging of `loop/` and `puddles/`.
- No saved scores, and no `scores` or `safety` in the round event. Those arrive with the real judge and gate in stage 2.

## Done when

`python -m scripts.run_graph_round --fake --root runs/dev` prints the round's outcome, `runs/dev/log/decisions.jsonl` has the event, `runs/dev/conversations/` has the round's file, and `mlflow ui --backend-store-uri sqlite:///runs/mlflow.db` shows the trace.
