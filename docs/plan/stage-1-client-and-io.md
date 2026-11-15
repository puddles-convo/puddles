# Stage 1: the model clients and the input and output rules

**Target:** 19:00. **Owner:** unassigned. **Needs:** stage 0. **Blocks:** stages 2 and 3, and the crisis checker in stage 4.

## Goal

Every model call is traced, counts its tokens and its cost, and follows the two rules: output as a schema, input as markdown. After this stage a new agent is a prompt, an input model, an output model and a few small functions.

## Where we start

| Client | Returns | Tokens | Schema output | Traced by MLflow |
|---|---|---|---|---|
| `loop.llm.OpenAICompatClient` (helper, simulated user) | Text | Not read from the response | No | No, until `TracedClient` from stage 0 |
| `puddles.llm.OpenAILLM` (critic, researcher, proposer) | A validated Pydantic model | Not recorded | Yes, through `chat.completions.parse` | Yes, through `mlflow.openai.autolog()` |

The critic, researcher and proposer already follow both rules: each has a `render_user` for the markdown and an output model in `schemas.py`.

## Checklist

**Tokens and cost**

- [x] `OpenAICompatClient.complete` reads `usage` from the response and reports prompt and completion tokens. Keep the return value as text; report usage through a callback or a counter on the client, so the existing tests for `loop/` do not change. *Built as `client.last_usage`, a `CallUsage`, set by every request.*
- [x] `OpenAILLM` reports `resp.usage` the same way, for `structured` and for every call inside `structured_with_search`.
- [x] **A response with no `usage` is estimated, not counted as zero:** characters divided by four, and the call is marked as estimated. A zero would hide spend from the cap in stage 6.
- [x] Every attempt counts: a retry after a failed check, and a schema fallback, each add their own tokens.
- [x] A `Usage` tally keyed by agent name: tokens in and out, calls, estimated calls, and for the researcher the number of Exa searches.
- [x] A prices table in the run's config file: dollars per million input and output tokens for each model, and dollars per Exa search. **Whoever holds each account fills it in from the provider's price page.** No price is assumed in code; a model with no price stops the run.
- [x] **Every model call has an output limit.** `loop.llm` already sends `max_tokens`. `puddles.llm.OpenAILLM` sends none, in `_parse` or in `structured_with_search`: add one, set per agent in the run's config. Without a limit the most a call can cost is unknown. *Sent as `max_completion_tokens`. An answer cut off at the limit raises `AgentFailed`. The values in `config/run.toml` are 4000 for every agent and have not been measured; stage 3 sets them.*
- [x] **Every model call writes to `<root>/log/usage.jsonl` twice.** Before the request goes out: a `sent` line with a call ID, the round, the agent, the model, and the most the call can cost (the prompt's estimated tokens plus its output limit, at the model's prices). When the answer arrives: a `done` line with the same call ID and the real tokens and cost. *A request that fails, with an HTTP error or no answer, gets no `done` line, so it is counted at its estimate.*
- [x] **Every Exa request writes the same two lines,** with the configured price per search as the cost on both. An Exa line has no model and no tokens. `puddles.exa.search` tries a failed request a second time; each try is its own request with its own two lines. *A try that fails still gets its `done` line at the full price, with an `error` field.*
- [x] Spend is the sum of the `done` lines, plus the `sent` figure for any call that has no `done` line, because a request cut off by a crash may still be billed.
- [x] When an agent fails twice, a `failed` line names the round and the agent. Stage 6 reads this file after a restart to know what has been spent and which rounds failed. *Written by `call_checked`, by `OpenAILLM` when the model refuses or is cut off, and by `TracedClient.complete_json`. A conversation that fails in `converse` writes no `failed` line; it is in the failures file beside the conversations.*
- [x] At the end of a round the graph logs the totals to MLflow (`tokens_in.<agent>`, `tokens_out.<agent>`, `cost_usd`). *Also `calls.<agent>`, `estimated_calls.<agent>` and `searches.<agent>`.*

**Schema output on an OpenAI-compatible endpoint**

The crisis checker runs on the helper's model, and a fallback judge may run on an open model. Both need schema output without the OpenAI library.

- [x] `OpenAICompatClient.complete_json(system, messages, schema)`: sends `response_format` with the model's JSON schema, validates the reply with Pydantic, and returns the model.
- [x] If the server refuses `response_format` (HTTP 400 or 422), fall back to asking for JSON in the prompt, then validate. The client remembers the refusal, so later calls skip the refused request. Either way a reply that fails validation is retried once, then raises.
- [x] `complete_json` takes a deadline for the whole call, retries included. The crisis checker has 5 seconds in total. *Each request gets the time left as its timeout, and an answer that arrives late is discarded; both raise `DeadlineExceeded`. The timeout is the socket's, so a server that sends its answer a few bytes at a time can run past the deadline before the call notices. The release step in stage 4 must not rely on this alone.*
- [x] `TracedClient` wraps `complete_json` as well as `complete`, with one span per attempt.
- [ ] Ask the robot pair which of the two their helper endpoint supports, and write the answer in this file.

**Markdown input**

- [x] `puddles/agents/render.py`: `transcript(turns)` renders a conversation as markdown, one line per turn, `**Person:**` and `**Helper:**`.
- [ ] Use it wherever a conversation is shown to a model that is not taking part in it. **Not done, and it needs a decision.** The only such place today is the critic, and [t4-critic.md](../agents/t4-critic.md) and [t3-judge.md](../agents/t3-judge.md) both fix the format as numbered lines, `1. user: ...`. Changing what the critic reads is a prompt change, which this stage leaves alone. Stage 2 picks one format and makes the two agent files and the code agree.
- [ ] No conversation IDs, policy versions or scenario details in what the judge sees. **Left for stage 2,** which builds the judge. `transcript` takes the turns and nothing else, so a judge prompt built from it cannot carry them.

**Counting retries**

- [x] `call_checked` reports when a first answer failed its checks. The graph logs `first_answer_failed.<agent>` per round.


## What was built: the interfaces later stages use

| Need | Use |
|---|---|
| A model call from a new graph node, counted and traced | Run the node's work inside `with puddles.usage.metering(meter, "<agent>")`. `puddles.graph` already does this for every node, the judge and the gate included, so a judge or gate passed in through `RoundDeps` is metered under `judge` or `gatekeeper` with no more code. |
| A plain-text model call | `TracedClient(client, "<agent>").complete(...)`. It uses the meter set by `metering`, or one passed as `TracedClient(client, name, meter=...)`. |
| Schema output from the helper's endpoint | `TracedClient(client, "crisis_checker").complete_json(system, messages, Schema, max_tokens=..., deadline_s=5)`. Returns the validated model. Raises `loop.errors.SchemaError` after two bad replies, `DeadlineExceeded` when time runs out, `LLMError` for anything else; the first two are subclasses of `LLMError`. |
| The same, in a test | `FakeClient([...json text...]).complete_json(...)`. The retry and the deadline behave as on a real client. |
| Schema output from the OpenAI API | `OpenAILLM().structured(...)`, as before. It reports its own requests, with the output limit of the agent named by `metering`. |
| A scripted model in a fake round | Nothing: the graph wraps any `LLM` that lacks `reports_usage` in `puddles.tracing.TracedLLM`, which estimates its tokens. |
| What has been spent | `puddles.usage.spend(root / "log/usage.jsonl")`. The lines themselves: `read_usage(path)`. |
| The round's tally | `meter.usage["<agent>"]` (`tokens_in`, `tokens_out`, `calls`, `estimated_calls`, `searches`, `cost_usd`, `first_answer_failed`, `failed`), and `meter.metrics()`. |
| Prices and output limits | `puddles.config.load_config()` reads `config/run.toml`, or the file named by `PUDDLES_CONFIG`. Stage 6 adds its own keys to the same file. |
| A switch the helper's server needs, such as thinking mode off | `HELPER_EXTRA_BODY`, a JSON object merged into every request. `SIMUSER_EXTRA_BODY` likewise. |

A client with no `model` attribute (`FakeClient`, `ScriptedLLM`) is logged as the model `fake` and costs nothing. That is the only price in the code. Every real model needs a row in the prices table, with `0` written out if it is free.

### The usage file

One JSON object per line in `<root>/log/usage.jsonl`. Every line has `event`, `round`, `agent` and `at`.

| `event` | `kind` | Other fields |
|---|---|---|
| `sent` | `model` | `call_id`, `model`, `prompt_tokens_estimate`, `max_tokens`, `max_cost_usd` |
| `done` | `model` | `call_id`, `model`, `tokens_in`, `tokens_out`, `estimated`, `cost_usd` |
| `sent` | `search` | `call_id`, `max_cost_usd` |
| `done` | `search` | `call_id`, `cost_usd`, and `error` if the request failed |
| `failed` | | `reason` |

## Tests that must pass

All offline, in `tests/test_stage1.py`, numbered as here.

1. A fake HTTP response with `usage` gives the right token counts. One without `usage` gives an estimate marked as estimated, and does not fail.
2. `complete_json` returns a validated model for a good reply, retries once on a bad one, and raises on two bad ones. The tally holds the tokens of both attempts.
3. `complete_json` falls back when the fake server answers the first request with HTTP 400.
4. `complete_json` raises when its deadline passes during the retry.
5. `transcript` output for the four conversations in `fixtures/bad_conversations.json` contains every turn in order and nothing else.
6. After one fake round, the MLflow run has token metrics for every agent that ran, and each model-call span carries its own counts, the researcher's search calls included.
7. Separately, with a fake HTTP server: a `complete_json` call through `TracedClient` leaves one span per attempt, each with its own counts. The fake round does not call `complete_json`, so it cannot test this.
8. A model with no entry in the prices table stops the run before the first call.
9. The `done` lines for a round in `<root>/log/usage.jsonl` add up to the spans.
10. A batch killed in the middle of `converse` leaves a `done` line for every call that had finished. A `sent` line with no `done` line is counted at its estimate.
11. A request made through `OpenAILLM` carries the configured output limit. An Exa search leaves a `sent` and a `done` line with the per-search price and no token fields.

## Unknowns to settle with one real call each

- **Number ranges in a strict schema.** `JudgeOutput` uses `Field(ge=1, le=5)`. Whether the OpenAI API accepts `minimum` and `maximum` in a strict schema has not been tried here. If it refuses, send the schema without the bounds and keep them in the agent's `check` function.
- **Thinking mode on the helper model.** The design has it off. Confirm how the endpoint is told, and pass it through `extra_body` on the client.

## Not in this stage

- No new agents.
- No change to the helper or simulated user prompts.
- past.dev credits are not in the cost figure. Watch them by hand on the past.dev dashboard.

## Done when

A fake round shows, in MLflow, one span per model call with its token counts, the usage file has the round's lines, and `python -m pytest` passes.
