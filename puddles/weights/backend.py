"""Trainers. TinkerBackend runs LoRA training on Tinker; FakeBackend is the offline stand-in.

Both sample from the weights being trained, so DPO pairs and GRPO groups are on-policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from puddles.weights.data import Candidate, Group, Pair


@dataclass
class Saved:
    sampler_path: str
    state_path: str


class Backend(Protocol):
    base_model: str

    def start(self, from_state: Optional[str]) -> None: ...

    def prompt_tokens(self, system: str, messages: list[dict[str, str]]) -> list[int]: ...

    def sample(self, prompt_tokens: list[int], system: str, messages: list[dict[str, str]], n: int, *,
               max_tokens: int, temperature: float) -> list[Candidate]: ...

    def train_dpo(self, pairs: Sequence[Pair], *, beta: float, learning_rate: float,
                  epochs: int) -> dict[str, float]: ...

    def train_grpo(self, groups: Sequence[tuple[Group, list[Candidate], list[float]]], *,
                   learning_rate: float) -> dict[str, float]: ...

    def save(self, name: str) -> Saved: ...

    def export(self, saved: Saved, out_dir: Path) -> Path: ...


def _done(x: Any) -> Any:
    """Tinker returns futures from its sync calls; wait for one."""
    return x.result() if hasattr(x, "result") else x


class TinkerBackend:
    """LoRA training of the helper's model on Tinker (TINKER_API_KEY).

    DPO uses `forward_backward_custom` with the reference logprobs from the weights the run started from;
    GRPO uses the built-in `importance_sampling` loss. Export needs `tinker_cookbook` (build_lora_adapter).
    """

    def __init__(self, base_model: str, rank: int = 16) -> None:
        self.base_model, self.rank = base_model, rank
        self.tc: Any = None
        self.sampler: Any = None
        self.tok: Any = None

    def start(self, from_state: Optional[str]) -> None:
        import tinker

        service = tinker.ServiceClient()
        if from_state:
            self.tc = _done(service.create_training_client_from_state(path=from_state))
        else:
            self.tc = _done(service.create_lora_training_client(base_model=self.base_model, rank=self.rank))
        self.tok = self.tc.get_tokenizer()
        self.sampler = _done(self.tc.save_weights_and_get_sampling_client())

    def prompt_tokens(self, system: str, messages: list[dict[str, str]]) -> list[int]:
        chat = [{"role": "system", "content": system}, *messages]
        # enable_thinking=False matches the helper's serving (thinking off); other templates ignore it.
        return list(self.tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True,
                                                 enable_thinking=False))

    def sample(self, prompt_tokens: list[int], system: str, messages: list[dict[str, str]], n: int, *,
               max_tokens: int, temperature: float) -> list[Candidate]:
        from tinker import types

        res = _done(self.sampler.sample(prompt=types.ModelInput.from_ints(prompt_tokens), num_samples=n,
                                        sampling_params=types.SamplingParams(max_tokens=max_tokens,
                                                                             temperature=temperature)))
        return [Candidate(text=self.tok.decode(s.tokens, skip_special_tokens=True).strip(), tokens=list(s.tokens),
                          logprobs=list(s.logprobs or [])) for s in res.sequences]

    @staticmethod
    def _datum(prompt: list[int], completion: list[int], **inputs: list[float]) -> Any:
        from tinker import types

        full = prompt + completion
        return types.Datum(model_input=types.ModelInput.from_ints(full[:-1]),
                           loss_fn_inputs={"target_tokens": full[1:], **inputs})

    def _ref_sum(self, prompt: list[int], completion: list[int]) -> float:
        from tinker import types

        lps = _done(self.sampler.compute_logprobs(types.ModelInput.from_ints(prompt + completion)))
        return float(sum(lp for lp in lps[len(prompt):] if lp is not None))

    def train_dpo(self, pairs: Sequence[Pair], *, beta: float, learning_rate: float,
                  epochs: int) -> dict[str, float]:
        import torch
        from tinker import types

        data, masks, refs = [], [], []
        for p in pairs:
            for c in (p.chosen, p.rejected):
                n_prefix = len(p.group.prompt_tokens) - 1
                weights = [0.0] * n_prefix + [1.0] * len(c.tokens)
                data.append(self._datum(p.group.prompt_tokens, c.tokens, weights=weights))
                masks.append(torch.tensor(weights))
                refs.append(self._ref_sum(p.group.prompt_tokens, c.tokens))  # the starting weights: frozen

        def loss_fn(_data: list, logprobs: list) -> tuple:
            sums = torch.stack([(lp * m).sum() for lp, m in zip(logprobs, masks, strict=True)])
            ref = torch.tensor(refs)
            z = beta * ((sums[0::2] - ref[0::2]) - (sums[1::2] - ref[1::2]))
            loss = -torch.nn.functional.logsigmoid(z).mean()
            return loss, {"dpo_loss": loss.item(), "accuracy": (z > 0).float().mean().item(),
                          "margin": (z / beta).mean().item()}

        metrics: dict[str, float] = {}
        for _ in range(epochs):
            out = self.tc.forward_backward_custom(data, loss_fn)
            _done(self.tc.optim_step(types.AdamParams(learning_rate=learning_rate)))
            metrics = {k: float(v) for k, v in _done(out).metrics.items()}
        self.sampler = _done(self.tc.save_weights_and_get_sampling_client())
        return metrics

    def train_grpo(self, groups: Sequence[tuple[Group, list[Candidate], list[float]]], *,
                   learning_rate: float) -> dict[str, float]:
        from tinker import types

        data = []
        for g, cands, adv in groups:
            n_prefix = len(g.prompt_tokens) - 1
            for c, a in zip(cands, adv, strict=True):
                if not c.tokens or len(c.logprobs) != len(c.tokens):
                    continue
                data.append(self._datum(g.prompt_tokens, c.tokens, logprobs=[0.0] * n_prefix + c.logprobs,
                                        advantages=[0.0] * n_prefix + [a] * len(c.tokens)))
        if not data:
            return {}
        out = self.tc.forward_backward(data, "importance_sampling")
        _done(self.tc.optim_step(types.AdamParams(learning_rate=learning_rate)))
        metrics = {k: float(v) for k, v in _done(out).metrics.items()}
        self.sampler = _done(self.tc.save_weights_and_get_sampling_client())  # the next iteration samples these
        return metrics

    def save(self, name: str) -> Saved:
        sampler = _done(self.tc.save_weights_for_sampler(name=name))
        state = _done(self.tc.save_state(name=name))
        return Saved(sampler_path=sampler.path, state_path=state.path)

    def export(self, saved: Saved, out_dir: Path) -> Path:
        from tinker_cookbook import weights

        raw = weights.download(tinker_path=saved.sampler_path, output_dir=str(out_dir / "download"))
        peft = out_dir / "peft"
        weights.build_lora_adapter(base_model=self.base_model, adapter_path=raw, output_path=str(peft))
        return peft


class FakeBackend:
    """Scripted: fixed replies of different lengths, no weights. For --fake and tests."""

    REPLIES = (
        "That sounds really hard.",
        "It sounds like this has been weighing on you. What feels heaviest right now?",
        "Okay.",
        "I hear how tired you are. Would it help to talk through what happened?",
    )

    def __init__(self, base_model: str = "fake-base") -> None:
        self.base_model = base_model
        self.started_from: Optional[str] = "unset"
        self.trained: list[tuple[str, int]] = []

    def start(self, from_state: Optional[str]) -> None:
        self.started_from = from_state

    def prompt_tokens(self, system: str, messages: list[dict[str, str]]) -> list[int]:
        return list(range(1 + len(messages)))

    def sample(self, prompt_tokens: list[int], system: str, messages: list[dict[str, str]], n: int, *,
               max_tokens: int, temperature: float) -> list[Candidate]:
        texts = [self.REPLIES[i % len(self.REPLIES)] for i in range(n)]
        return [Candidate(text=t, tokens=list(range(len(t.split()))), logprobs=[-1.0] * len(t.split()))
                for t in texts]

    def train_dpo(self, pairs: Sequence[Pair], *, beta: float, learning_rate: float,
                  epochs: int) -> dict[str, float]:
        self.trained.append(("dpo", len(pairs)))
        return {"dpo_loss": 0.69, "accuracy": 0.5}

    def train_grpo(self, groups: Sequence[tuple[Group, list[Candidate], list[float]]], *,
                   learning_rate: float) -> dict[str, float]:
        self.trained.append(("grpo", len(groups)))
        return {"loss:sum": 0.0}

    def save(self, name: str) -> Saved:
        return Saved(sampler_path=f"fake://{name}/sampler", state_path=f"fake://{name}/state")

    def export(self, saved: Saved, out_dir: Path) -> Path:
        peft = out_dir / "peft"
        peft.mkdir(parents=True, exist_ok=True)
        (peft / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": self.base_model,
                                                              "r": 16, "fake": True}) + "\n")
        (peft / "adapter_model.safetensors").write_bytes(b"fake")
        return peft
