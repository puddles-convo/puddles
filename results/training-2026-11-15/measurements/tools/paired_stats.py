"""Reproduce the headline comparisons from the saved scores. No model is called.

Run from the repository root:
    python results/training-2026-11-15/measurements/tools/paired_stats.py

Each policy was measured on sets of 20 conversations, one per validation scenario. Scores are averaged per
scenario over a policy's sets, policies are compared scenario by scenario, and the 95% interval comes from
resampling the 20 scenarios 5,000 times (seed 7).
"""
import glob
import json
import random
import statistics as st
from pathlib import Path

M = Path(__file__).resolve().parents[1]
A = M / "astra"


def scenario(conversation_id: str) -> str:
    return conversation_id.split("-")[-1]


def load(path) -> dict[str, tuple[int, int]]:
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[scenario(d["conversation_id"])] = (d["empathy"], d["relevance"])
    return out


def per_scenario(sets: list[dict]) -> dict[str, dict[str, float]]:
    ids = sorted(set.intersection(*[set(s) for s in sets]))
    return {i: {"quality": st.mean((s[i][0] + s[i][1]) / 2 for s in sets),
                "empathy": st.mean(s[i][0] for s in sets),
                "relevance": st.mean(s[i][1] for s in sets)} for i in ids}


def interval(diffs: list[float], n: int = 5000) -> tuple[float, float]:
    rng = random.Random(7)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def report(title: str, groups: dict[str, list[dict]], pairs: list[tuple[str, str]]) -> None:
    print("==", title)
    scores = {name: per_scenario(sets) for name, sets in groups.items()}
    for name, q in scores.items():
        row = {k: round(st.mean(v[k] for v in q.values()), 3) for k in ("quality", "empathy", "relevance")}
        print(f"  {name:10s} sets {len(groups[name])} scenarios {len(q)} {row}")
    for a, b in pairs:
        ids = sorted(set(scores[a]).intersection(scores[b]))
        d = [scores[a][i]["quality"] - scores[b][i]["quality"] for i in ids]
        lo, hi = interval(d)
        better, equal = sum(x > 0 for x in d), sum(x == 0 for x in d)
        print(f"  {a} minus {b}: {st.mean(d):+.3f} [{lo:+.2f}, {hi:+.2f}] | better {better}, equal {equal}, "
              f"worse {len(d) - better - equal} of {len(d)}")


LABELS = {"start": ["v1", "v1b"], "control": ["plain-s2", "plain-s2b"], "final": ["v4", "v4b"]}
PAIRS = [("final", "start"), ("final", "control"), ("control", "start")]
report("validation scenarios, loop judge (gpt-6.1-sol)",
       {g: [load(A / "sets" / f"{name}.sol.scores.jsonl") for name in names] for g, names in LABELS.items()}, PAIRS)
report("validation scenarios, final judge (gpt-6-astra)",
       {g: [load(A / f"{name}.scores.jsonl") for name in names] for g, names in LABELS.items()}, PAIRS)
HELD = {"start": "heldout-v1", "control": "heldout-control", "final": "heldout-v4"}
report("held-out scenarios, loop judge (gpt-6.1-sol)",
       {g: [load(glob.glob(str(M / "heldout" / folder / "scores.jsonl"))[0])] for g, folder in HELD.items()}, PAIRS)
HELD_LABELS = {"start": "heldout-v1", "control": "heldout-control", "final": "heldout-v4"}
report("held-out scenarios, final judge (gpt-6-astra)",
       {g: [load(A / f"{name}.scores.jsonl")] for g, name in HELD_LABELS.items()}, PAIRS)


def pooled(title: str, validation: dict[str, list[dict]], held: dict[str, list[dict]]) -> None:
    """All 40 scenarios together: the gatekeeper's 20 (two sets per policy) and the 20 held-out (one set)."""
    print("==", title)
    scores = {}
    for name in validation:
        a, b = per_scenario(validation[name]), per_scenario(held[name])
        scores[name] = {**{"v" + k: v for k, v in a.items()}, **{"h" + k: v for k, v in b.items()}}
    for a, b in PAIRS:
        ids = sorted(set(scores[a]).intersection(scores[b]))
        d = [scores[a][i]["quality"] - scores[b][i]["quality"] for i in ids]
        lo, hi = interval(d)
        better, equal = sum(x > 0 for x in d), sum(x == 0 for x in d)
        print(f"  {a} minus {b}: {st.mean(d):+.3f} [{lo:+.2f}, {hi:+.2f}] | better {better}, equal {equal}, "
              f"worse {len(d) - better - equal} of {len(d)}")


pooled("all 40 scenarios, loop judge (gpt-6.1-sol)",
       {g: [load(A / "sets" / f"{name}.sol.scores.jsonl") for name in names] for g, names in LABELS.items()},
       {g: [load(A / "sets" / f"{name}.sol.scores.jsonl")] for g, name in HELD_LABELS.items()})
pooled("all 40 scenarios, final judge (gpt-6-astra)",
       {g: [load(A / f"{name}.scores.jsonl") for name in names] for g, names in LABELS.items()},
       {g: [load(A / f"{name}.scores.jsonl")] for g, name in HELD_LABELS.items()})
