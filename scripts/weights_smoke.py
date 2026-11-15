"""A short real run of the weight-training path on Tinker (docs/specs/weight-training.md).

Starts a LoRA client, samples replies to two fixture contexts, takes one DPO step and one GRPO step,
saves the weights and asks the saved adapter for a reply through Tinker's OpenAI-compatible endpoint.
Replies are ranked by length instead of a judge, so no other key is needed. Needs TINKER_API_KEY.

    python -m scripts.weights_smoke --base-model Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loop.helper import helper_reply
from loop.policy import load_policy
from loop.prompts import helper_messages, render_helper_system
from puddles.weights.backend import TinkerBackend
from puddles.weights.data import Group, dpo_pairs, grpo_groups
from puddles.weights.serving import TinkerAdapterLoader
from schemas import Turn

CONTEXTS = [
    [Turn(speaker="user", text="i got laid off today. twelve years and they walked me out in ten minutes")],
    [Turn(speaker="user", text="i can't sleep. i keep replaying the argument with my sister"),
     Turn(speaker="helper", text="That sounds exhausting. What keeps coming back to you?"),
     Turn(speaker="user", text="that she said i always make everything about me")],
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--policy", default="policies/policy-v1.md")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--name", default=f"smoke-{int(time.time())}")
    args = p.parse_args(argv)

    t0 = time.time()

    def say(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    policy = load_policy(Path(args.policy))
    system = render_helper_system(policy)
    backend = TinkerBackend(args.base_model, rank=16)
    backend.start(None)
    say(f"LoRA training client on {args.base_model}")

    groups = []
    for i, turns in enumerate(CONTEXTS):
        messages = helper_messages(turns)
        tokens = backend.prompt_tokens(system, messages)
        cands = backend.sample(tokens, system, messages, args.samples, max_tokens=80, temperature=0.9)
        for c in cands:
            c.quality = len(c.tokens) / 10  # stand-in for the judge: longer scores higher, so groups differ
        groups.append(Group(f"smoke-{i}", len(turns), system, messages, tokens, cands))
        say(f"context {i}: {len(tokens)} prompt tokens, {len(cands)} replies, "
            f"qualities {[c.quality for c in cands]}")
        for c in cands:
            print(f"    - {c.text[:140]!r} ({len(c.tokens)} tokens, {len(c.logprobs)} logprobs)")

    pairs = dpo_pairs(groups, min_margin=0.5)
    if pairs:
        say(f"DPO step on {len(pairs)} pairs: {backend.train_dpo(pairs, beta=0.1, learning_rate=1e-5, epochs=1)}")
    else:
        say("DPO: no pair separated by 0.5 (all replies the same length band)")
    usable = grpo_groups(groups)
    if usable:
        say(f"GRPO step on {len(usable)} groups: {backend.train_grpo(usable, learning_rate=4e-5)}")
    else:
        say("GRPO: every group tied")

    saved = backend.save(args.name)
    say(f"saved: sampler {saved.sampler_path}, state {saved.state_path}")

    loader = TinkerAdapterLoader.from_env()
    if not loader.extra_body:
        loader.extra_body = {"reasoning_effort": False}  # the helper runs with thinking off
    client = loader.load(args.name, saved.sampler_path, None)
    reply = helper_reply(client, policy, CONTEXTS[0], session_id="smoke", turn=1)
    say(f"adapter reply via Tinker's chat endpoint: {reply.text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
