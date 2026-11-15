"""Putting an adapter in front of the helper: on Tinker (no server of ours) or on our vLLM server.

The vLLM helper endpoint is vLLM (PR #79's Modal app) started with `--enable-lora` and
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True. An adapter is loaded with POST {HELPER_URL}/load_lora_adapter and
then chosen per request by its name in the `model` field, so the candidate and the model in force can be
compared on the same server.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Protocol

from loop.errors import ConfigError
from loop.llm import ChatClient, FakeClient, OpenAICompatClient
from loop.run_batch import _fake_helper


class AdapterLoader(Protocol):
    needs_export: bool  # True when the server needs the exported PEFT files

    def load(self, name: str, sampler_path: Optional[str], adapter_dir: Optional[Path]) -> ChatClient: ...

    def unload(self, name: str) -> None: ...


class TinkerAdapterLoader:
    """Serves the adapter from Tinker itself, so no helper server of ours is needed.

    Tinker's OpenAI-compatible endpoint takes a sampler checkpoint's path (`tinker://.../sampler_weights/...`)
    as the `model` of a chat request, so the candidate is just a client with that model name. Nothing is
    exported, uploaded or unloaded. PUDDLES_TINKER_EXTRA_BODY (a JSON object) is merged into each request:
    it is where the switch that turns the model's thinking off goes.
    """

    needs_export = False

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0,
                 extra_body: Optional[dict] = None, max_tokens_param: str = "max_tokens") -> None:
        self.base_url, self.api_key, self.timeout = base_url, api_key, timeout
        self.extra_body, self.max_tokens_param = dict(extra_body or {}), max_tokens_param

    @classmethod
    def from_env(cls) -> "TinkerAdapterLoader":
        from puddles.eval.judge import TINKER_BASE_URL

        key = os.environ.get("TINKER_API_KEY")
        if not key:
            raise ConfigError("environment variable TINKER_API_KEY is not set")
        raw = os.environ.get("PUDDLES_TINKER_EXTRA_BODY")
        try:
            extra = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ConfigError(f"PUDDLES_TINKER_EXTRA_BODY is not JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise ConfigError("PUDDLES_TINKER_EXTRA_BODY must be a JSON object")
        return cls(os.environ.get("PUDDLES_TINKER_BASE_URL") or TINKER_BASE_URL, key, extra_body=extra)

    def load(self, name: str, sampler_path: Optional[str], adapter_dir: Optional[Path]) -> ChatClient:
        if not sampler_path or not sampler_path.startswith("tinker://"):
            raise RuntimeError(f"{name} has no Tinker sampler checkpoint to serve from")
        return OpenAICompatClient(self.base_url, sampler_path, api_key=self.api_key, timeout=self.timeout,
                                  extra_body=self.extra_body, max_tokens_param=self.max_tokens_param)

    def unload(self, name: str) -> None:
        pass  # the checkpoint stays in Tinker storage; nothing is loaded anywhere


class VLLMAdapterLoader:
    """Loads adapters into a running vLLM server.

    The server reads `lora_path` from its own disk. With PUDDLES_ADAPTER_VOLUME set, the adapter is first
    copied to that Modal volume (`modal volume put`) under PUDDLES_ADAPTER_SERVER_DIR, which is where the
    helper app mounts it; otherwise the server is assumed to share this machine's disk.
    """

    needs_export = True

    def __init__(self, base: OpenAICompatClient, *, volume: Optional[str] = None,
                 server_dir: str = "/adapters") -> None:
        self.base, self.volume, self.server_dir = base, volume, server_dir.rstrip("/")

    @classmethod
    def from_env(cls, base: OpenAICompatClient) -> "VLLMAdapterLoader":
        return cls(base, volume=os.environ.get("PUDDLES_ADAPTER_VOLUME") or None,
                   server_dir=os.environ.get("PUDDLES_ADAPTER_SERVER_DIR", "/adapters"))

    def _post(self, path: str, body: dict) -> str:
        headers = {"Content-Type": "application/json"}
        if self.base.api_key:
            headers["Authorization"] = f"Bearer {self.base.api_key}"
        req = urllib.request.Request(f"{self.base.base_url}/{path}", data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=max(self.base.timeout, 300)) as resp:
            return resp.read().decode("utf-8", "replace")

    def _server_path(self, name: str, adapter_dir: Path) -> str:
        if not self.volume:
            return str(Path(adapter_dir).resolve())
        if shutil.which("modal") is None:
            raise ConfigError("PUDDLES_ADAPTER_VOLUME is set but the modal command is not installed")
        remote = f"{self.server_dir}/{name}"
        subprocess.run(["modal", "volume", "put", "--force", self.volume, str(adapter_dir), f"/{name}"],
                       check=True, capture_output=True, text=True)
        return remote

    def load(self, name: str, sampler_path: Optional[str], adapter_dir: Optional[Path]) -> ChatClient:
        if adapter_dir is None:
            raise RuntimeError(f"{name} was not exported, so vLLM has nothing to load")
        path = self._server_path(name, adapter_dir)
        try:
            self._post("load_lora_adapter", {"lora_name": name, "lora_path": path})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if "already" not in detail.lower():  # loaded before a restart: the server still has it
                raise RuntimeError(f"vLLM refused adapter {name}: HTTP {exc.code} {detail[:300]}") from exc
        return OpenAICompatClient(self.base.base_url, name, api_key=self.base.api_key, timeout=self.base.timeout,
                                  extra_body=self.base.extra_body, max_tokens_param=self.base.max_tokens_param)

    def unload(self, name: str) -> None:
        try:
            self._post("unload_lora_adapter", {"lora_name": name})
        except urllib.error.URLError:
            pass  # a reverted adapter left on the server is unused, not harmful


class FakeAdapterLoader:
    """For --fake: the 'adapter' is the scripted helper again."""

    def __init__(self, needs_export: bool = True) -> None:
        self.needs_export = needs_export
        self.loaded: list[str] = []

    def load(self, name: str, sampler_path: Optional[str], adapter_dir: Optional[Path]) -> ChatClient:
        if self.needs_export and not (adapter_dir and (Path(adapter_dir) / "adapter_config.json").exists()):
            raise RuntimeError(f"{adapter_dir} has no adapter_config.json")
        if not self.needs_export and not sampler_path:
            raise RuntimeError(f"{name} has no sampler checkpoint")
        self.loaded.append(name)
        return FakeClient(_fake_helper)

    def unload(self, name: str) -> None:
        if name in self.loaded:
            self.loaded.remove(name)
