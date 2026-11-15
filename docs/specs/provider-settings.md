# Provider settings and provider errors

This is the contract for four things the first real round showed (2026-11-15, `gpt-6-luna` on OpenAI's
chat completions address). A run on scripted clients cannot show any of them, because each is an
answer from a real server.

Nothing here changes a request when the new settings are unset.

## 1. The name of the output-limit field

`loop.llm.client_from_env(prefix)` reads one more optional variable, `{prefix}_MAX_TOKENS_PARAM`.

- Unset or empty: the request body carries the limit as `max_tokens`, as before.
- `max_tokens` or `max_completion_tokens`: the body carries the limit under that name and does not
  carry the other name.
- Any other value: `ConfigError`, naming the variable and the two allowed values. This is raised when
  the client is built, before any request.

`PUDDLES_AGENT_MAX_TOKENS_PARAM` makes the same choice for the note-taker's `PlainLLM` (`docs/specs/chat-notes.md`), with the default `max_completion_tokens`, as `OpenAILLM` sends. A variable that is set but empty is the same as one that is not set, as for every other optional variable in this file; any other value that is not one of the two names is a `ConfigError` before any request.

`OpenAICompatClient` takes the same choice as a constructor argument with the default `max_tokens`.
The Python argument of `complete`, `complete_json` and `request` is still called `max_tokens`; only the
field in the body sent to the server changes.

Why: current OpenAI models refuse `max_tokens` with a 400 and ask for `max_completion_tokens`. Other
servers (vLLM, Tinker) want `max_tokens`.

## 2. A value in `{prefix}_EXTRA_BODY` replaces the same field of the request

This is how the code already behaves; it is written down and tested because a real run depends on it.

- A key in `{prefix}_EXTRA_BODY` replaces the same key in the request body, whatever the caller passed.
  `{"temperature": 1}` therefore sends temperature 1 for a helper call made with 0.7.
- The client does not inspect the object's fields. It merges them and sends the result.

Why: `gpt-6-luna` accepts only the default temperature; the helper asks for 0.7 and the simulated
user for 0.9, and both were refused. The judge has `PUDDLES_JUDGE_TEMPERATURE=none` for the same case,
and the agents send no temperature unless `PUDDLES_AGENT_TEMPERATURE` is set.

## 3. Reasoning effort on the researcher's search requests

`puddles.llm.OpenAILLM` reads one more optional variable, `PUDDLES_AGENT_TOOL_REASONING_EFFORT`.

- Set: every request that carries the search tool has `reasoning_effort` set to that value.
- A request that carries no tool does not get the field from this variable. That covers `structured`
  (the critic and the proposer) and the researcher's last request, which asks for the final answer.
- Unset or empty: no request gets the field.

Why: `gpt-6-luna` refuses function tools on chat completions unless `reasoning_effort` is `none`.
The setting is limited to the tool requests on purpose. With reasoning off for every agent, the critic
missed a failure it had found with reasoning on (same policy, same five scenarios).

## 4. A provider error fails the agent, not the run

In `puddles.llm.OpenAILLM`, when a request raises an `openai.OpenAIError`:

- the method raises `AgentFailed`. Its reason names the kind of error and carries the provider's
  message, cut to 300 characters;
- with a meter active, one `failed` line is written for the agent named by `puddles.usage.metering`
  (through `puddles.usage.note_failed`, as the other failures in this class do). The request has no
  `done` line, as `puddles.tracing.model_call` already says for a block that raises;
- the original error is the `__cause__` of the `AgentFailed`.

`LengthFinishReasonError` keeps its own handling and its own reason. `KeyboardInterrupt` and
`SystemExit` are not caught.

`puddles.llm.call_checked` does not retry an `AgentFailed`: a refused request is not a failed check,
and the SDK has already retried what is worth retrying (rate limits, server errors).

The graph already turns an `AgentFailed` into an outcome (`puddles/graph.py`), so with this rule a
provider error in the critic gives `no_pattern`, in the researcher gives `no_evidence`, and in the
proposer gives a round with no edit. In each case the round is finished and written to the decision
log with the agent and the provider's message in its reason, and `scripts.run_graph_round` exits 0.
Before this rule the same error ended the process with a traceback and the round was not logged.

## 5. A search that fails

`puddles.exa.search` raises `RuntimeError` when Exa fails twice. When the researcher's search raises
it, the researcher fails with `AgentFailed` (reason: the search failed, with the error's message), and
the round ends `no_evidence` as in section 4. A missing `EXA_API_KEY` is still refused before the round
starts, by `scripts.run_graph_round`.

## Settings for `gpt-6-luna` on OpenAI

These are the settings the first complete real round ran with.

```
HELPER_URL=https://api.openai.com/v1
HELPER_MODEL=gpt-6-luna
HELPER_MAX_TOKENS_PARAM=max_completion_tokens
HELPER_EXTRA_BODY={"temperature": 1, "reasoning_effort": "none"}
SIMUSER_URL=https://api.openai.com/v1
SIMUSER_MODEL=gpt-6-luna
SIMUSER_MAX_TOKENS_PARAM=max_completion_tokens
SIMUSER_EXTRA_BODY={"temperature": 1, "reasoning_effort": "none"}
PUDDLES_AGENT_MODEL=gpt-6-luna
PUDDLES_AGENT_TOOL_REASONING_EFFORT=none
PUDDLES_JUDGE_TEMPERATURE=none
```

`reasoning_effort` is `none` for the helper and the simulated user because their replies are capped
at 80 and 120 tokens, and with reasoning on a reply spent about half of that on reasoning.

## Not covered

- A provider that fails every request (a wrong key) makes every round end `no_pattern` and the loop
  goes on paying for conversations. Stopping a loop after repeated provider failures belongs to the
  loop driver.
- Errors from the helper's and the simulated user's server during a conversation, and from the judge,
  keep their present handling.
