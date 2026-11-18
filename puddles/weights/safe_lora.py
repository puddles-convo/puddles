"""Safe LoRA (arXiv 2405.16833): project a LoRA update back toward the safety-aligned direction.

For each layer, V = W_aligned - W_base (the instruct model minus its base model) defines the projection
C = V Vᵀ / ‖V‖_F. A layer whose update ΔW = B A has cosine similarity below `tau` with C ΔW is replaced by
C ΔW, which is (C B) A: only B changes, so the adapter keeps its rank and format.

This module is the arithmetic. Applying it to an exported adapter needs the base and instruct weights of
the helper's layers, which the loop does not download (docs/specs/weight-training.md, "Not built").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerResult:
    name: str
    similarity: float
    projected: bool


def projection(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("the alignment matrix is zero")
    return (v @ v.T) / norm


def project_layer(a: np.ndarray, b: np.ndarray, v: np.ndarray, tau: float) -> tuple[np.ndarray, float, bool]:
    """B for one layer (shape d_out x r), its similarity, and whether it was projected."""
    c = projection(v)
    delta = b @ a
    proj = c @ delta
    denom = np.linalg.norm(delta) * np.linalg.norm(proj)
    similarity = float((delta * proj).sum() / denom) if denom else 1.0
    if similarity < tau:
        return c @ b, similarity, True
    return b, similarity, False


def project_adapter(layers: dict[str, tuple[np.ndarray, np.ndarray]], alignment: dict[str, np.ndarray],
                    tau: float = 0.5) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[LayerResult]]:
    """Every layer with an alignment matrix is checked; layers without one are left as they are."""
    out, report = {}, []
    for name, (a, b) in layers.items():
        if name not in alignment:
            out[name] = (a, b)
            continue
        new_b, sim, projected = project_layer(a, b, alignment[name], tau)
        out[name] = (a, new_b)
        report.append(LayerResult(name, sim, projected))
    return out, report
