# Build plan: seven stages to a running loop

How we get from the code in this repo to a training loop that runs overnight and an inference path that holds a conversation. Each stage has its own file, a test that must pass before the next stage starts, and a target time for Saturday 14 November.

Written for people and for coding agents. Take one stage file, do what its checklist says, tick the boxes in the file, and open a pull request.

| Stage | File | What it gives | Target (PT) |
|---|---|---|---|
| 0 | [stage-0-skeleton.md](stage-0-skeleton.md) | One round as a LangGraph graph with MLflow tracing, dummy agents where real ones are missing | 18:00 |
| 1 | [stage-1-client-and-io.md](stage-1-client-and-io.md) | Token counts, traced model calls, schema output on every endpoint, markdown inputs | 19:00 |
| 2 | [stage-2-real-agents.md](stage-2-real-agents.md) | Judge and gatekeeper built; every prompt tuned on hand-checked cases | 21:00 |
| 3 | [stage-3-training-small.md](stage-3-training-small.md) | Real rounds on 5 scenarios, measured, with nothing promoted | 22:00 |
| 4 | [stage-4-inference.md](stage-4-inference.md) | Crisis checker, release step and note-taker, run as text in and text out | 22:00 |
| 5 | [stage-5-baselines-noise.md](stage-5-baselines-noise.md) | Baseline scores and the noise figure the gate needs | 23:00 |
| 6 | [stage-6-overnight-loop.md](stage-6-overnight-loop.md) | The loop running unattended with caps, and the plots | Start 23:30 |

Stages 0, 1, 2, 3, 5 and 6 are in order. The crisis path in stage 4 can be built alongside stages 2 and 3. Its note-taker waits for a format change that has not landed (see stage 4).

This plan was reviewed against the code by a second model on 14 November and corrected. The corrections are in these files.

## What exists today

| Part | Where | State |
|---|---|---|
| Simulated user, helper, conversation runner, batch command | `loop/` | Built, tested offline. `python -m loop.run_batch --fake ...` runs with no keys. |
| Critic, researcher, proposer | `puddles/agents/` | Built, tested offline with a scripted model. |
| past.dev memory for the training agents, record files, decision log, policy edits | `puddles/memory.py`, `records.py`, `decision_log.py`, `policy.py` | Built, tested offline. |
| Steps 4 to 6 of a round, log entry, copies into memory | `puddles/round.py`, `scripts/run_round.py` | Built. `scripts/run_round.py` still uses a stand-in gate. |
| Judge | `puddles/agents/judge.py`, `scripts/calibrate_judge.py` | Built, tested offline. Not calibrated or frozen (issue #4). |
| Gatekeeper and safety cases | `puddles/gate.py`, `fixtures/safety-v1.json` | Built, tested offline with a scripted judge. The cases are a draft with no clinical review. Promotes nothing until the noise is measured (stage 5). |
| Where each rule came from (`policies/<version>.meta.json`) | `puddles/meta.py` | Built. Written under the run's root for the starting policy and for each promoted one. |
| ESConv loader | `puddles/esconv.py` | Built. Downloads at run time; nothing is committed. `fixtures/scenarios.jsonl` has 5 hand-written scenarios. |
| One round as a graph, with MLflow tracing | `puddles/graph.py`, `puddles/tracing.py`, `scripts/run_graph_round.py`; contract in [docs/specs/graph.md](../specs/graph.md) | Built, tested offline. `python -m scripts.run_graph_round --fake --root runs/dev` runs with no keys. The judge and the gate in it are stand-ins. |
| Token counts, the loop command | | Not built (stages 1, 6) |
| Crisis checker, release step, note-taker, storage for a person's notes | | Not built (stage 4) |

The suite runs 521 tests offline and skips one that needs `PAST_API_KEY`. It takes about three minutes; the exact command is in [docs/specs/graph.md](../specs/graph.md). If `PYTHONPATH` is set in your shell, unset it first.

### Two packages, three overlaps

`loop/` and `puddles/` were written at the same time by different people. Three modules exist twice.

| Purpose | `loop/` | `puddles/` | Use tonight |
|---|---|---|---|
| Model client | `llm.py`: `ChatClient.complete()` returns text; plain HTTP to any OpenAI-compatible server | `llm.py`: `LLM.structured()` returns a validated model; OpenAI library | Both. Text for the helper and simulated user, structured for everything else. |
| Policy | `policy.py`: `load_policy(path)` gives a `Policy` with `version`, `text`, `safety_text`, `rules` | `policy.py`: `parse_policy(text)` gives a `Policy` with separate `safety` and `conversation` rules; `apply_edit`, `render_policy` | Both. Loading in `loop`, editing in `puddles`. They are two different classes; pass the file path between them, not the object. |
| Decision log | `decision_log.py`: append, read, render a table | `decision_log.py`: `DecisionLog`, with the history the agents need | **Only `puddles.decision_log.DecisionLog` writes the log.** `loop.decision_log.render_table` may read it. |

Do not merge the two packages tonight. The graph in stage 0 imports from both.

**A policy rule is one line, and lines end with a plain newline.** `puddles.policy` accepts a rule that wraps onto a second line; `loop.policy` raises "is not a rule" on the same file. `render_policy` does not repair this: it writes each Conversation rule on one line, but copies the Safety section exactly as it was written, wrapped lines included. So the check is on the way in: a policy file is used only if `loop.policy.load_policy` accepts it. Stage 0 adds the test, and the loop refuses a starting policy that fails it.

## Rules for every agent

1. **Output is a schema.** A model's answer is a Pydantic model in `schemas.py`, requested with `LLM.structured(system, user, Schema)`, then checked by code and retried once with the reasons (`puddles.llm.call_checked`). The helper and the simulated user are the exceptions: they return plain text, because one is spoken aloud and the other plays a person.
2. **Input is markdown.** Code builds an input model, and `render_user` turns it into the markdown the model reads. The helper and the simulated user receive the conversation as chat turns instead.
3. **Explanation before verdict.** In a new or changed output model, put the reasoning field before the scores or the decision. A model writes fields in order.
4. **A model never supplies IDs, dates, versions or who the user is.** Code adds them when it makes the record.
5. **Each agent module has the same shape:** `build_input`, `render_user`, `check`, `run`, `to_record`. See `puddles/agents/critic.py`. The proposer differs in two places: its `render_user(inp, base_version)` takes the policy version, and it has `apply(...)`, which returns the edit and the candidate file, in place of `to_record`.
6. **Prompts live in one place.** The system prompts for the critic, researcher and proposer are read from `docs/agents/*.md`. The helper and simulated user prompts are in `loop/prompts/*.txt`, and must match their files in `docs/agents/`.

### When an agent's answer fails twice

`call_checked` raises `AgentFailed` after the second failure. Each graph node catches it and does what the agent's spec says. A failure is always counted and logged.

| Agent | After the second failure | Source |
|---|---|---|
| Simulated user, helper | The conversation is marked failed and the batch goes on | `loop.runner.run_batch`, built |
| Judge | That conversation is left unscored and counted in the round's log entry | [t3-judge.md](../agents/t3-judge.md) |
| Critic | The round ends as `no_pattern`, with the failure in the reason | [t4-critic.md](../agents/t4-critic.md) |
| Researcher | Not stated in its spec. Proposed: the round ends as `no_evidence`, with the failure in the reason | To confirm on issue #1 |
| Proposer | The round ends with no edit and goes straight to `finish`, skipping the gate. The pattern stays on the to-do list, with whatever status the log gives it | [t6-proposer.md](../agents/t6-proposer.md) |
| Gatekeeper | A test run that crashes, or a conversation left failed or unscored, reverts the candidate | [t7-gatekeeper.md](../agents/t7-gatekeeper.md) |

**Open format question.** A round event's outcome is one of `kept`, `reverted`, `no_pattern`, `no_evidence`. None of them means "a pattern and sources, but no edit". Today `finish_round` logs such a round as `no_pattern`, with the pattern attached and the failure in the reason. Stage 0 builds and tests exactly that. Whether to add a fifth outcome is a question for issue #1; if one is added, the test changes with it.

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
| `PUDDLES_JUDGE_MODEL` | Judge. A dated snapshot, not an alias. Always run at temperature 0. |
| `ESCONV_CACHE` (optional) | Where the ESConv files are kept; default `runs/cache/esconv` |
| `EXA_API_KEY` | Researcher |
| `PAST_API_KEY`, `PAST_BASE_URL` (optional) | Memory |

Stage 0 added `MLFLOW_TRACKING_URI`.

**Use a separate past.dev project for development,** with its own `PAST_API_KEY`. The training audience name `training-agents` is fixed in `puddles/memory.py`, so a second audience in the same project is not an option without a code change.

## What we track and plot

There is no single accuracy number. There are four measures. Each plot needs a saved file, and each file needs an owner.

| Measure | How it is computed | Where it is shown | File it is drawn from | Produced by |
|---|---|---|---|---|
| Judge agreement with real users | Rank correlation and mean absolute error against the ratings ESConv users gave their own conversations | Scatter of judge score against user score | Pairs of scores, no conversation text | Stage 2, lane A (#4) |
| Quality per round | Mean over validation conversations of (empathy + relevance) / 2 | Line by round, with the noise band and a mark for kept or reverted | The decision log: `scores` in each round event | Stages 2 and 6 |
| Final result | Final policy against the plain and hand-written baselines on held-out test scenarios | Three bars with the spread | Final evaluation results | Issue #7, Sunday |
| Safety | Pass rate on the safety cases; crisis checker misses and false alarms; the outside sets | Table | The decision log: `safety` in each round event; crisis checker results; outside-set results | Stage 2 lane B; issues #6 and #7 |

Also logged every round: tokens, cost and seconds per agent, conversations that failed, conversations left unscored, and how often each agent's first answer failed its checks.
