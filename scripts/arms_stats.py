"""The toolbox comparison from saved scores. No model is called.

    python scripts/arms_stats.py runs/measure

Same method as results/training-2026-11-15/measurements/tools/paired_stats.py: scores are averaged
per scenario over a policy's sets, policies are compared scenario by scenario, and the 95% interval
comes from resampling the scenarios 5,000 times (seed 7).

The rule, written before the run: the toolbox counts as better than rules only if the interval of
"toolbox minus policy" is above zero over all 40 scenarios under both judges.
"""
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/measure")
LABELS = ("start", "policy", "toolbox")
PAIRS = [("toolbox", "policy"), ("toolbox", "start"), ("policy", "start")]


def scenario(conversation_id: str) -> str:
    return conversation_id.split("-")[-1]


def load(path) -> dict[str, tuple[int, int]]:
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[scenario(d["conversation_id"])] = (d["empathy"], d["relevance"])
    return out


def per_scenario(sets: list[dict]) -> dict[str, float]:
    ids = sorted(set.intersection(*[set(s) for s in sets])) if sets else []
    return {i: st.mean((s[i][0] + s[i][1]) / 2 for s in sets) for i in ids}


def interval(diffs: list[float], n: int = 5000) -> tuple[float, float]:
    rng = random.Random(7)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def loop_sets(label: str, part: str) -> list[dict]:
    return [load(p) for p in sorted(glob.glob(str(OUT / f"{label}-{part}" / "*" / "scores.jsonl")))]


def astra_sets(label: str, part: str) -> list[dict]:
    return [load(p) for p in sorted(glob.glob(str(OUT / "astra" / f"{label}-{part}-*.scores.jsonl")))]


def compare(title: str, scores: dict[str, dict[str, float]]) -> dict:
    print("==", title)
    for name in LABELS:
        q = scores[name]
        print(f"  {name:8s} scenarios {len(q):2d} quality {st.mean(q.values()):.3f}" if q else f"  {name:8s} no scores")
    result = {}
    for a, b in PAIRS:
        ids = sorted(set(scores[a]).intersection(scores[b]))
        if len(ids) < 2:
            print(f"  {a} minus {b}: not enough scenarios")
            continue
        d = [scores[a][i] - scores[b][i] for i in ids]
        lo, hi = interval(d)
        better, equal = sum(x > 0 for x in d), sum(x == 0 for x in d)
        print(f"  {a} minus {b}: {st.mean(d):+.3f} [{lo:+.2f}, {hi:+.2f}] | better {better}, equal {equal}, "
              f"worse {len(d) - better - equal} of {len(d)}")
        result[f"{a}-{b}"] = (st.mean(d), lo, hi, len(d))
    return result


verdict = {}
for judge, getter in (("loop judge (gpt-6.1-sol)", loop_sets), ("final judge (gpt-6-astra)", astra_sets)):
    val = {n: per_scenario(getter(n, "val")) for n in LABELS}
    held = {n: per_scenario(getter(n, "held")) for n in LABELS}
    compare(f"validation scenarios, {judge}", val)
    compare(f"held-out scenarios, {judge}", held)
    pooled = {n: {**{"v" + k: v for k, v in val[n].items()}, **{"h" + k: v for k, v in held[n].items()}}
              for n in LABELS}
    verdict[judge] = compare(f"all scenarios, {judge}", pooled).get("toolbox-policy")

print("== rule written before the run")
ok = [v for v in verdict.values() if v]
if len(ok) == 2 and all(v[1] > 0 for v in ok):
    print("  toolbox better than rules only: the interval is above zero under both judges")
elif len(ok) == 2 and all(v[2] < 0 for v in ok):
    print("  toolbox worse than rules only: the interval is below zero under both judges")
else:
    print("  no measurable difference between toolbox and rules only (or a judge is missing)")
