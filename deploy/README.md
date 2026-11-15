# The helper model on Modal

`deploy/modal_helper.py` serves **Qwen3.5-9B** with **vLLM 0.32.0** as an OpenAI-compatible API on Modal. It is a separate Modal app, `puddles-helper`, so deploying or stopping it never touches the robot's always-on `puddles-stack` API.

The repo's helper, the crisis checker, `scripts/chat.py` and training runs all reach it through the existing `HELPER_*` variables. No code in `loop/` or `puddles/` changes.

| | |
|---|---|
| Model | `Qwen/Qwen3.5-9B`, served as `qwen3.5-9b`, text only (`--language-model-only`) |
| GPU | L40S by default (`MODAL_HELPER_GPU`: L4, A10G, L40S, A100-40GB, A100-80GB, H100, H200) |
| Region | `us-east`, where the robot's API runs (`MODAL_HELPER_REGION`; empty means any region) |
| Scaling | Scales to zero after 15 min idle. `MODAL_HELPER_MIN_CONTAINERS=1` keeps one warm, **only while the demo runs** |
| Auth | Bearer key. vLLM reads `VLLM_API_KEY` from the Modal Secret `puddles-helper-key`; the app refuses to start without it |
| Weights | Cached on the Modal Volume `puddles-helper-hf-cache`, so only the first start downloads them (about 18 GB) |
| LoRA | Off. `MODAL_HELPER_LORA=1` adds `--enable-lora --max-lora-rank 32 --max-loras 2`, for a fine-tuned adapter later |

All settings are in `deploy/helper_config.py`, which is tested offline in `tests/test_modal_helper.py`.

## Set up once

No new environment is needed. `uv` runs Modal's CLI with the pinned version, the same `uv run --no-project --with ...` style as the team's test command (`docs/specs/graph.md`), and installs nothing into the repo's `.venv`:

```bash
MODAL="uv run --no-project --python 3.12 --with modal==1.6.3 modal"   # the version in deploy/requirements.txt

$MODAL token new                      # opens the browser; pick the workspace that holds your Modal credits

# the server's key: keep the value, the clients need it as HELPER_API_KEY
export HELPER_API_KEY="$(openssl rand -hex 32)"
$MODAL secret create puddles-helper-key VLLM_API_KEY="$HELPER_API_KEY"
```

Deploying needs only `modal`: `deploy/modal_helper.py` imports nothing from the repo except `deploy/helper_config.py`.

Put `HELPER_API_KEY` in your `.env` (never commit it). Anyone with the URL and the key can use your GPU.

## Deploy and stop

```bash
$MODAL deploy deploy/modal_helper.py
# prints the URL, e.g. https://<workspace>--puddles-helper-serve.modal.run

$MODAL app stop puddles-helper        # stops it, and the bill
```

For the demo only:

```bash
MODAL_HELPER_MIN_CONTAINERS=1 $MODAL deploy deploy/modal_helper.py
# afterwards, deploy again without it, or stop the app
```

## Point the repo at it

```bash
export HELPER_URL="https://<workspace>--puddles-helper-serve.modal.run/v1"
export HELPER_MODEL="qwen3.5-9b"
export HELPER_API_KEY="..."                                   # the key from the set-up step
export HELPER_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'

# the eval code (puddles/eval/helper.py, compose.yaml) uses its own names for the same endpoint:
export PUDDLES_HELPER_BASE_URL="$HELPER_URL" PUDDLES_HELPER_MODEL="$HELPER_MODEL" PUDDLES_HELPER_API_KEY="$HELPER_API_KEY"
```

**Thinking must be off.** The helper has 80 tokens per reply and the crisis checker has 40. With Qwen's thinking on, they spend those tokens reasoning and the reply breaks.
- Thinking is turned off per request with `HELPER_EXTRA_BODY`, which the client merges into every request.
- vLLM 0.32 may also support a server-side default for this, but that was not confirmed, so it is not relied on.
- The smoke test fails if any reply contains thinking text.

**Price row.** `config/run.toml` refuses to run a model without a row in `[prices.models]`. A self-hosted model is billed by Modal per GPU-hour, not per token. Add `"qwen3.5-9b" = { input_usd_per_million = 0, output_usd_per_million = 0 }` for runs that check prices, and count the GPU time separately.

## Test it

The smoke test needs the repo's own dependencies. Run it from the repo root with `uv`, or with any existing `.venv` that has `requirements.txt` installed:

```bash
PY="uv run --no-project --python 3.12 --with-requirements requirements.txt python"   # or: .venv/bin/python
$PY -m scripts.modal_smoke                                     # one helper reply, one harmless and one crisis check
$PY -m scripts.modal_smoke --warm 20 --concurrent 5 --check-auth
```

The smoke test uses the repo's own client, helper and crisis checker, so a pass means the inference code works against the endpoint. It fails on:
- an empty reply;
- thinking text in a reply;
- an unparseable checker line;
- the crisis message not rated `crisis`;
- a checker slower than its 3-second limit;
- with `--check-auth`, a request without the key being accepted.

Then stage 4's "done when" check (needs `PAST_API_KEY` and `OPENAI_API_KEY` for the note-taker):

```bash
$PY -m scripts.chat --policy policies/policy-v1.md --user-id user-01 --remember-me no --training no --root runs/chat
```

That means 10 turns, then one crisis message, such as "I don't want to be here anymore", which must get the fixed crisis response.

### Live results (fill in; "not run" until then)

| Check | Result |
|---|---|
| Cold start (first request after deploy / after scale-to-zero) | not run |
| Helper, warm: median / p95 (20 calls) | not run |
| Checker, warm: median / p95 (20 calls), must stay well under 3 s | not run |
| 5 concurrent helper calls: slowest | not run |
| Request without the key refused (401) | not run |
| `scripts/chat.py`: 10 turns + crisis message gets the fixed response | not run |
| Scaled to zero after the idle timeout (no running containers) | not run |

## Cost

You pay for GPU time while a container runs, including the 15 idle minutes before it scales to zero. Check current prices at modal.com/pricing. As a rough guide, an L40S costs about half as much per hour as an H100. Stop the app (`modal app stop puddles-helper`) when nobody needs it, and never leave `MODAL_HELPER_MIN_CONTAINERS=1` on overnight unless a run needs it.

## Not here

Speech to text and text to speech belong to the robot repo (`puddles-stack`). A web endpoint wrapping `puddles.inference.turn.run_turn` for the robot is not built yet; it waits for a decision.
