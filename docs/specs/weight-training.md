# Weight training in the loop

The loop can post-train the helper model between rounds, not only edit its rules. DPO runs first, and GRPO is an option. A new adapter goes through the same kind of gate as a rule edit, and it is kept only if quality rises by more than the noise and safety does not get worse. This is README section 8, tools 4 and 8 (`train_adapter` with DPO or GRPO). Training runs on Tinker, and by default Tinker also serves each adapter, so no model server of ours is needed.

Code: `puddles/weights/`. Command: `python -m scripts.run_loop --gate real --weights auto` (or `dpo`, `grpo`). Tests: `tests/test_weights.py`, all offline.

## When it trains

After each round, `WeightStep.after_round` checks two things:

- **Trigger:** the last `rounds_without_keep` rounds (3 by default) kept no rule edit. Rounds before the last training run don't count again.
- **Budget:** fewer than `max_training_runs` (3) training runs so far.

The method is set by `--weights`:
- `dpo` or `grpo`: always that method.
- `auto`: DPO, then GRPO for the next run after a DPO adapter is reverted.

The trigger is code, not the proposer's choice. The proposer's toolbox (`ProposerOutputV2`) is not wired into the round yet.

## One training run

1. **Data.** Conversations come from the last `source_rounds` rounds (`conversations/round-NN-train.jsonl`). The run picks `prompts` helper turns. At each turn, the weights being trained sample `samples` replies to the same context, under the policy in force. The loop's judge scores each reply as the last turn of the conversation so far, using quality = (empathy + relevance) / 2. Everything goes to `adapters/<id>/dataset.jsonl`.
2. **Training on Tinker** (`TinkerBackend`, LoRA rank `rank`). Training starts from the base model, or from the adapter in force.
   - **DPO:** each context's best and worst reply form a pair, if the judge separates them by at least `min_margin`. The loss is DPO with `dpo_beta`, computed with `forward_backward_custom`. The reference is the weights the run started from.
   - **GRPO:** each context's replies get group-normalised advantages, (q − mean) / std. Tied groups are dropped. The update uses Tinker's `importance_sampling` loss. This repeats `grpo_iterations` times, each time sampling from the updated weights.
   - **No data:** if no usable pair or group is found, the run is logged `no_data` and nothing is trained.
3. **Save.** `save_weights_for_sampler` and `save_state` keep the weights in Tinker storage.
4. **Serve** (`weights.serve`):
   - `tinker` (default): the candidate is a chat client on Tinker's OpenAI-compatible endpoint, with the sampler checkpoint's path (`tinker://…/sampler_weights/<id>`) as its `model`. Nothing is exported or hosted by us.
   - `vllm`: `tinker_cookbook.weights.build_lora_adapter` writes a PEFT adapter to `adapters/<id>/peft`. It is loaded with `POST {HELPER_URL}/load_lora_adapter` and chosen per request by its name.
5. **Gate** (`puddles/weights/gate.py`). The policy file stays fixed; only the weights change. The adapter is kept only if all of these hold:
   - every safety case passes with it;
   - no fewer safety cases pass than with the current weights;
   - every validation conversation completes and is scored;
   - quality beats the current scores by more than the noise;
   - neither empathy nor relevance falls by more than the noise;
   - the noise has been measured.
6. **Keep or revert.**
   - **Kept:** the rounds and the gate use the adapter from now on. `scores/current.json` is deleted, so the next rule edit is measured on the new weights. The next training run starts from this adapter.
   - **Reverted:** with `vllm`, the adapter is unloaded; on Tinker the checkpoint just stays unused.

## Records

Each training run appends one line to `log/adapters.jsonl`: method, trigger, parent adapter, policy version, data counts, dataset hash, adapter hash (`vllm` only), Tinker paths, the name it is served as, metrics, gate result, safety counts and reason. The adapter in force is the last `kept` line.

After a restart, `restore()` checks the kept adapters' hashes and loads the adapter in force again. A changed or missing file stops the run.

A training run that crashes is logged `failed` and costs only that run. An error outside a run is a `weight_error` line in `log/loop.jsonl`, and the loop goes on.

## Settings and keys

- **Settings:** the `[weights]` table in `config/run.toml`. `base_model` must be Tinker's name for the helper's model.
- **Keys:** `TINKER_API_KEY`. `PUDDLES_TINKER_EXTRA_BODY` (a JSON object) is merged into every request to a Tinker-served adapter. Put the switch that turns the model's thinking off there: for Qwen3.5-9B, `{"reasoning_effort": false}`.
- **Fully on Tinker:** point the helper at Tinker too:
  - `HELPER_URL` = `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`
  - `HELPER_MODEL` = the base model's Tinker ID
  - `HELPER_API_KEY` = the Tinker key
  - `HELPER_USER_AGENT` = any name, e.g. `puddles/0.1`, and `HELPER_EXTRA_BODY` = `{"reasoning_effort": false}`

  Then no model is hosted by us at all. The judge and the simulated user still need their own endpoints.
- **With `serve = "vllm"`:**
  - `PUDDLES_ADAPTER_VOLUME` (a Modal volume the helper mounts) and `PUDDLES_ADAPTER_SERVER_DIR` (its mount path, `/adapters` by default). Without them, the server must share this machine's disk.
  - vLLM started with `--enable-lora` and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. [PR #79](https://github.com/puddles-convo/puddles/pull/79) adds the LoRA flags, but not the volume mount or that variable. The vLLM docs say runtime loading is for trusted environments only.
- **Python packages** for a real run, not in `requirements.txt`: `tinker`, `torch`, `transformers`, plus `tinker_cookbook` with `serve = "vllm"`.

## Not built or not checked

- **Tinker smoke run only, not a loop run.** `python -m scripts.weights_smoke` (Qwen/Qwen3.5-9B, 15 Nov) sampled replies, took one DPO and one GRPO step, saved the weights and got a reply from the saved checkpoint through Tinker's chat endpoint, in about 70 seconds. It ranks replies by length, not with the judge, and no full loop with the gate has run on Tinker. Tinker's OpenAI-compatible endpoint is labelled beta.
- **Tinker gotchas found by that run:** Cloudflare in front of the chat endpoint refuses urllib's default User-Agent (error 1010), so `OpenAICompatClient` takes an optional `user_agent`: Tinker-served adapters send one, and a helper on Tinker needs `HELPER_USER_AGENT` set (any name, e.g. `puddles/0.1`). Unset, requests are byte-for-byte as before. Qwen3.5-9B reasons by default there and takes only `true` or `false` for `reasoning_effort`, so set `PUDDLES_TINKER_EXTRA_BODY='{"reasoning_effort": false}'` (and `HELPER_EXTRA_BODY` the same when the helper runs on Tinker). With `serve = "vllm"`, that run must also check that `build_lora_adapter` and vLLM's LoRA path handle Qwen3.5-9B's layers. Qwen3-8B is the README's fallback.
- **Safe LoRA:** `safe_lora.py` has the projection and its tests, but it is not applied to exported adapters. That needs the base and instruct weights of the helper's layers.
- **No safety data mixed into training.** Safety is enforced only by the gate's two safety checks.
- **Tinker cost isn't metered.** Judge calls made while building data are metered under the gatekeeper.
- **Judge-gaming risk.** The loop's own judge is the reward. A held-out judge check, as in the README's GRPO row, is not added.
