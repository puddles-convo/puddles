"""Settings and the vLLM command for the helper model on Modal.

Kept free of any `modal` import, so the offline tests can check the command without Modal installed.
`deploy/modal_helper.py` reads these settings when it is deployed and bakes them into the image's
environment, so the container serves exactly what was deployed.

The server's API key is never on the command line: vLLM reads it from the VLLM_API_KEY environment
variable, which comes from a Modal Secret.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

VLLM_VERSION = "0.32.0"  # Qwen3.5 needs vLLM >= 0.17; this release supports --language-model-only and LoRA
PORT = 8000
APP_NAME = "puddles-helper"
SECRET_NAME = "puddles-helper-key"  # holds VLLM_API_KEY
HF_CACHE_VOLUME = "puddles-helper-hf-cache"
VLLM_CACHE_VOLUME = "puddles-helper-vllm-cache"

GPUS = ("L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "H100", "H200")

# Environment variables read at deploy time. Every one is optional.
ENV = {
    "model": "MODAL_HELPER_MODEL",
    "served_name": "MODAL_HELPER_SERVED_NAME",
    "gpu": "MODAL_HELPER_GPU",
    "region": "MODAL_HELPER_REGION",
    "min_containers": "MODAL_HELPER_MIN_CONTAINERS",
    "scaledown_s": "MODAL_HELPER_SCALEDOWN_S",
    "max_model_len": "MODAL_HELPER_MAX_MODEL_LEN",
    "gpu_memory_utilization": "MODAL_HELPER_GPU_MEMORY_UTILIZATION",
    "concurrency": "MODAL_HELPER_CONCURRENCY",
    "text_only": "MODAL_HELPER_TEXT_ONLY",
    "lora": "MODAL_HELPER_LORA",
    "max_lora_rank": "MODAL_HELPER_MAX_LORA_RANK",
    "max_loras": "MODAL_HELPER_MAX_LORAS",
}


class SettingsError(ValueError):
    pass


def _bool(value: str, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise SettingsError(f"{name}={value!r} is not a yes/no value")


def _int(value: str, name: str, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise SettingsError(f"{name}={value!r} is not a whole number") from exc
    if not lo <= number <= hi:
        raise SettingsError(f"{name}={number} must be between {lo} and {hi}")
    return number


@dataclass(frozen=True)
class HelperSettings:
    model: str = "Qwen/Qwen3.5-9B"
    served_name: str = "qwen3.5-9b"
    gpu: str = "L40S"
    region: str = "us-east"  # the robot's Modal API runs there; empty means any region
    min_containers: int = 0  # scale to zero; set 1 only while the demo runs
    scaledown_s: int = 15 * 60
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    concurrency: int = 32
    text_only: bool = True  # Qwen3.5-9B is a vision-language model; the helper only needs text
    lora: bool = False
    max_lora_rank: int = 32
    max_loras: int = 2

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "HelperSettings":
        d = cls()
        get = lambda key: env.get(ENV[key], "").strip()  # noqa: E731
        gpu = get("gpu") or d.gpu
        if gpu not in GPUS:
            raise SettingsError(f"{ENV['gpu']}={gpu!r} is not one of {', '.join(GPUS)}")
        util = get("gpu_memory_utilization")
        try:
            memory = float(util) if util else d.gpu_memory_utilization
        except ValueError as exc:
            raise SettingsError(f"{ENV['gpu_memory_utilization']}={util!r} is not a number") from exc
        if not 0.3 <= memory <= 0.95:
            raise SettingsError(f"{ENV['gpu_memory_utilization']}={memory} must be between 0.3 and 0.95")
        return cls(
            model=get("model") or d.model,
            served_name=get("served_name") or d.served_name,
            gpu=gpu,
            region=env.get(ENV["region"], d.region).strip(),
            min_containers=_int(get("min_containers"), ENV["min_containers"], 0, 2) if get("min_containers") else d.min_containers,
            scaledown_s=_int(get("scaledown_s"), ENV["scaledown_s"], 60, 3600) if get("scaledown_s") else d.scaledown_s,
            max_model_len=_int(get("max_model_len"), ENV["max_model_len"], 1024, 65536) if get("max_model_len") else d.max_model_len,
            gpu_memory_utilization=memory,
            concurrency=_int(get("concurrency"), ENV["concurrency"], 1, 256) if get("concurrency") else d.concurrency,
            text_only=_bool(get("text_only"), ENV["text_only"]) if get("text_only") else d.text_only,
            lora=_bool(get("lora"), ENV["lora"]) if get("lora") else d.lora,
            max_lora_rank=_int(get("max_lora_rank"), ENV["max_lora_rank"], 1, 64) if get("max_lora_rank") else d.max_lora_rank,
            max_loras=_int(get("max_loras"), ENV["max_loras"], 1, 8) if get("max_loras") else d.max_loras,
        )

    def as_env(self) -> dict[str, str]:
        """The same settings as environment variables, for the container to read back."""
        values = {
            "model": self.model, "served_name": self.served_name, "gpu": self.gpu, "region": self.region,
            "min_containers": self.min_containers, "scaledown_s": self.scaledown_s,
            "max_model_len": self.max_model_len, "gpu_memory_utilization": self.gpu_memory_utilization,
            "concurrency": self.concurrency, "text_only": "1" if self.text_only else "0",
            "lora": "1" if self.lora else "0", "max_lora_rank": self.max_lora_rank, "max_loras": self.max_loras,
        }
        return {ENV[key]: str(value) for key, value in values.items()}


def vllm_command(s: HelperSettings) -> list[str]:
    """The `vllm serve` command. Thinking mode is turned off per request by the client
    (HELPER_EXTRA_BODY), not here; see deploy/README.md."""
    cmd = [
        "vllm", "serve", s.model,
        "--served-model-name", s.served_name,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(s.max_model_len),
        "--gpu-memory-utilization", str(s.gpu_memory_utilization),
        "--uvicorn-log-level", "info",
    ]
    if s.text_only:
        cmd.append("--language-model-only")
    if s.lora:
        cmd += ["--enable-lora", "--max-lora-rank", str(s.max_lora_rank), "--max-loras", str(s.max_loras)]
    return cmd
