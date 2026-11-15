# Build plan: seven stages to a running loop

How we get from the code in this repo to a training loop that runs overnight and an inference path that holds a conversation. Each stage has its own file, a test that must pass before the next stage starts, and a target time for Saturday 14 November.

Written for people and for coding agents. Take one stage file, do what its checklist says, tick the boxes in the file, and open a pull request.

| Stage | File | What it gives | Target (PT) |
|---|---|---|---|
| 0 | [stage-0-skeleton.md](stage-0-skeleton.md) | One round as a LangGraph graph with MLflow tracing, dummy agents where real ones are missing | 18:00 |
| 1 | [stage-1-client-and-io.md](stage-1-client-and-io.md) | Token counts, traced model calls, schema output on every endpoint, markdown inputs | 19:00 |
| 2 | [stage-2-real-agents.md](stage-2-real-agents.md) | Judge and gatekeeper built; every prompt tuned on hand-checked cases | 21:00 |
| 3 | [stage-3-training-small.md](stage-3-training-small.md) | One real round on 5 scenarios, measured | 22:00 |
| 4 | [stage-4-inference.md](stage-4-inference.md) | Crisis checker, release step and note-taker, run as text in and text out | 22:00 |
| 5 | [stage-5-baselines-noise.md](stage-5-baselines-noise.md) | Baseline scores and the noise figure the gate needs | 23:00 |
| 6 | [stage-6-overnight-loop.md](stage-6-overnight-loop.md) | The loop running unattended with caps, and the plots | Start 23:30 |

Stage 4 runs alongside stages 2 and 3. Everything else is in order.

## What exists today

| Part | Where | State |
|---|---|---|
| Simulated user, helper, conversation runner, batch command | `loop/` | Built, tested offline. `python -m loop.run_batch --fake ...` runs with no keys. |
| Critic, researcher, proposer | `puddles/agents/` | Built, tested offline with a scripted model. |
| past.dev memory, record files, decision log, policy edits | `puddles/memory.py`, `records.py`, `decision_log.py`, `policy.py` | Built, tested offline. |
| Steps 4 to 6 of a round, log entry, copies into memory | `puddles/round.py`, `scripts/run_round.py` | Built. Uses a stand-in gate. |
| Judge | | Not built (issues #3, #4) |
| Gatekeeper and safety cases | | Not built (issues #5, #20) |
| ESConv loader | | Not built (issue #2). `fixtures/scenarios.jsonl` has 5 hand-written scenarios. |
| Graph, tracing, loop command | | Not built (stages 0, 1, 6) |
| Crisis checker, release step, note-taker | | Not built (stage 4) |

`python -m pytest` runs 409 tests offline and skips one that needs `PAST_API_KEY`.

### Two packages, three overlaps

`loop/` and `puddles/` were written at the same time by different people. Three modules exist twice.

| Purpose | `loop/` | `puddles/` | Use tonight |
|---|---|---|---|
| Model client | `llm.py`: `ChatClient.complete()` returns text; plain HTTP to any OpenAI-compatible server | `llm.py`: `LLM.structured()` returns a validated model; OpenAI library | Both. Text for the helper and simulated user, structured for everything else. |
| Policy | `policy.py`: `load_policy(path)` gives version, text, Safety text, rules | `policy.py`: `parse_policy`, `apply_edit`, `render_policy` | Both. Loading in `loop`, editing in `puddles`. |
| Decision log | `decision_log.py`: append, read, render a table | `decision_log.py`: `DecisionLog`, with the history the agents need | **Only `puddles.decision_log.DecisionLog` writes the log.** `loop.decision_log.render_table` may read it. |

Do not merge the two packages tonight. The graph in stage 0 imports from both.

## Rules for every agent

1. **Output is a schema.** A model's answer is a Pydantic model in `schemas.py`, requested with `LLM.structured(system, user, Schema)`, then checked by code and retried once with the reasons (`puddles.llm.call_checked`). The helper and the simulated user are the exceptions: they return plain text, because one is spoken aloud and the other plays a person.
2. **Input is markdown.** Code builds an input model, and `render_user(inp)` turns it into the markdown the model reads. The helper and the simulated user receive the conversation as chat turns instead.
3. **Explanation before verdict.** In a new or changed output model, put the reasoning field before the scores or the decision. A model writes fields in order.
4. **A model never supplies IDs, dates, versions or who the user is.** Code adds them when it makes the record.
5. **Each agent module has the same five functions:** `build_input`, `render_user`, `check`, `run`, `to_record`. See `puddles/agents/critic.py`.
6. **Prompts live in one place.** The system prompts for the critic, researcher and proposer are read from `docs/agents/*.md`. The helper and simulated user prompts are in `loop/prompts/*.txt`, and must match their files in `docs/agents/`.

## Rules for every change

- **Offline first.** Every stage ends with `python -m pytest` passing with no keys and no network. Use the fakes: `loop.llm.FakeClient`, `tests/fakes.py` (`ScriptedLLM`, `FakeSearch`) and `puddles.memory.FakeMemory`.
- **One real call per agent before a real round.** Stage 3 is where the pieces first meet real models together. Each piece should already have made one real call on its own.
- **No secrets in the repo.** Keys come from the environment. `.env` and `runs/` are ignored by git. The repo has to be public for submission.
- **Small pull requests, one stage item each.** If a stage file is wrong, fix the file in the same pull request.

## Environment

| Variable | Used by |
|---|---|
| `HELPER_URL`, `HELPER_MODEL`, `HELPER_API_KEY` (optional) | Helper, through `loop.llm.client_from_env("HELPER")` |
| `SIMUSER_URL`, `SIMUSER_MODEL`, `SIMUSER_API_KEY` (optional) | Simulated user |
| `OPENAI_API_KEY`, `PUDDLES_AGENT_MODEL`, `PUDDLES_AGENT_TEMPERATURE` (optional) | Critic, researcher, proposer |
| `EXA_API_KEY` | Researcher |
| `PAST_API_KEY`, `PAST_BASE_URL` (optional) | Memory |

Stages 0 to 2 add `MLFLOW_TRACKING_URI` and a variable for the judge model.

## What we track and plot

There is no single accuracy number. There are four measures.

| Measure | How it is computed | Where it is shown |
|---|---|---|
| Judge agreement with real users | Rank correlation and mean absolute error against the ratings ESConv users gave their own conversations | Scatter of judge score against user score |
| Quality per round | Mean over validation conversations of (empathy + relevance) / 2 | Line by round, with the noise band and a mark for kept or reverted |
| Final result | Final policy against the plain and hand-written baselines on held-out test scenarios | Three bars with the spread |
| Safety | Pass rate on the safety cases; crisis checker misses and false alarms | Table |

Also logged every round: tokens, cost and seconds per agent, conversations that failed, and how often each agent's first answer failed its checks.
