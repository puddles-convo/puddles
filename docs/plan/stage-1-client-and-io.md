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

- [ ] `OpenAICompatClient.complete` reads `usage` from the response and reports prompt and completion tokens. Keep the return value as text; report usage through a callback or a counter on the client, so the existing tests for `loop/` do not change.
- [ ] `OpenAILLM` reports `resp.usage` the same way, for `structured` and for every call inside `structured_with_search`.
- [ ] **A response with no `usage` is estimated, not counted as zero:** characters divided by four, and the call is marked as estimated. A zero would hide spend from the cap in stage 6.
- [ ] Every attempt counts: a retry after a failed check, and a schema fallback, each add their own tokens.
- [ ] A `Usage` tally keyed by agent name: tokens in and out, calls, estimated calls, and for the researcher the number of Exa searches.
- [ ] A prices table in the run's config file: dollars per million input and output tokens for each model, and dollars per Exa search. **Whoever holds each account fills it in from the provider's price page.** No price is assumed in code; a model with no price stops the run.
- [ ] At the end of a round the graph logs the tally to MLflow (`tokens_in.<agent>`, `tokens_out.<agent>`, `cost_usd`) and appends one line to `<root>/log/usage.jsonl`. The line also names any agent that failed twice. Stage 6 reads that file to know what has been spent after a restart.

**Schema output on an OpenAI-compatible endpoint**

The crisis checker runs on the helper's model, and a fallback judge may run on an open model. Both need schema output without the OpenAI library.

- [ ] `OpenAICompatClient.complete_json(system, messages, schema)`: sends `response_format` with the model's JSON schema, validates the reply with Pydantic, and returns the model.
- [ ] If the server refuses `response_format`, fall back to asking for JSON in the prompt, then validate. Either way a reply that fails validation is retried once, then raises.
- [ ] `complete_json` takes a deadline for the whole call, retries included. The crisis checker has 3 seconds in total.
- [ ] `TracedClient` wraps `complete_json` as well as `complete`, with one span per attempt.
- [ ] Ask the robot pair which of the two their helper endpoint supports, and write the answer in this file.

**Markdown input**

- [ ] `puddles/agents/render.py`: `transcript(turns)` renders a conversation as markdown, one line per turn, `**Person:**` and `**Helper:**`. Use it wherever a conversation is shown to a model that is not taking part in it.
- [ ] No conversation IDs, policy versions or scenario details in what the judge sees.

**Counting retries**

- [ ] `call_checked` reports when a first answer failed its checks. The graph logs `first_answer_failed.<agent>` per round.

## Tests that must pass

All offline.

1. A fake HTTP response with `usage` gives the right token counts. One without `usage` gives an estimate marked as estimated, and does not fail.
2. `complete_json` returns a validated model for a good reply, retries once on a bad one, and raises on two bad ones. The tally holds the tokens of both attempts.
3. `complete_json` falls back when the fake server answers the first request with HTTP 400.
4. `complete_json` raises when its deadline passes during the retry.
5. `transcript` output for the four conversations in `fixtures/bad_conversations.json` contains every turn in order and nothing else.
6. After one fake round, the MLflow run has token metrics for every agent that ran, and each model-call span carries its own counts. That includes `complete_json` spans and the researcher's search calls.
7. A model with no entry in the prices table stops the run before the first call.
8. `<root>/log/usage.jsonl` has one line for the round, and its totals match the spans.

## Unknowns to settle with one real call each

- **Number ranges in a strict schema.** `JudgeOutput` uses `Field(ge=1, le=5)`. Whether the OpenAI API accepts `minimum` and `maximum` in a strict schema has not been tried here. If it refuses, send the schema without the bounds and keep them in the agent's `check` function.
- **Thinking mode on the helper model.** The design has it off. Confirm how the endpoint is told, and pass it through `extra_body` on the client.

## Not in this stage

- No new agents.
- No change to the helper or simulated user prompts.
- past.dev credits are not in the cost figure. Watch them by hand on the past.dev dashboard.

## Done when

A fake round shows, in MLflow, one span per model call with its token counts, the usage file has the round's line, and `python -m pytest` passes.
