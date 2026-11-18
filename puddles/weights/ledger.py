"""The adapter log: `<root>/log/adapters.jsonl`, append-only, one line per training run.

The decision log says which policy is in force; this log says which adapter is. The adapter in force
is the one named by the last line whose outcome is `kept`.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

PATH = Path("log") / "adapters.jsonl"
Outcome = Literal["kept", "reverted", "no_data", "failed"]


class AdapterEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: Literal["adapter"] = "adapter"
    after_round: int  # the round event this training run followed
    adapter_id: str
    method: Literal["dpo", "grpo"]
    trigger: str
    base_model: str
    parent_adapter: Optional[str] = None  # the adapter it started from; None for the base model
    policy_version: str  # the policy in force while it was trained and tested
    groups: int = 0
    pairs: int = 0
    dataset_sha256: Optional[str] = None
    sampler_path: Optional[str] = None  # where the trainer keeps the weights
    state_path: Optional[str] = None  # weights and optimizer state, to train on from here
    adapter_dir: Optional[str] = None  # the exported PEFT adapter, relative to the root
    adapter_sha256: Optional[str] = None
    served_as: Optional[str] = None  # the model name the helper endpoint serves it under
    metrics: dict[str, float] = {}
    outcome: Outcome
    failed_checks: list[str] = []
    quality_before: Optional[float] = None
    quality_after: Optional[float] = None
    safety_passed: Optional[int] = None
    safety_total: Optional[int] = None
    reason: str
    at: str = ""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_dir(path: Path) -> str:
    """One hash over every file's relative name and bytes, in name order."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def read(root: Path) -> list[AdapterEvent]:
    path = Path(root) / PATH
    if not path.exists():
        return []
    return [AdapterEvent.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append(root: Path, event: AdapterEvent) -> AdapterEvent:
    if not event.at:
        event = event.model_copy(update={"at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    path = Path(root) / PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(event.model_dump_json() + "\n")
    return event


def in_force(root: Path) -> Optional[AdapterEvent]:
    kept = [e for e in read(root) if e.outcome == "kept"]
    return kept[-1] if kept else None


def check(root: Path) -> list[str]:
    """Problems with the kept adapters' files: a missing or changed dataset or adapter. Empty when sound."""
    problems = []
    root = Path(root)
    for e in read(root):
        if e.outcome != "kept":
            continue
        data = root / "adapters" / e.adapter_id / "dataset.jsonl"
        if e.dataset_sha256 and (not data.exists() or sha256_file(data) != e.dataset_sha256):
            problems.append(f"{e.adapter_id}: dataset.jsonl is missing or changed")
        if e.adapter_dir and e.adapter_sha256:
            d = root / e.adapter_dir
            if not d.is_dir() or sha256_dir(d) != e.adapter_sha256:
                problems.append(f"{e.adapter_id}: the adapter files are missing or changed")
    return problems
