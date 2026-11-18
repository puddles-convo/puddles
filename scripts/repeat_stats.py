"""Compare policies measured N times on the same scenarios. No model is called.

    python scripts/repeat_stats.py runs/repeat-measure

For each policy: the quality of each run (a set of 20), their mean and spread. For each pair: the paired
difference per scenario averaged over all runs, with a 95% interval from resampling the scenarios 5,000
times (seed 7), and how many of the N run-by-run comparisons favour each side.
"""
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/repeat-measure")
BASE = ("control", "v1", "v4")
EXTRA = tuple(sorted(p.name for p in OUT.iterdir() if p.is_dir() and p.name not in BASE)) if OUT.is_dir() else ()
LABELS = BASE + EXTRA  # extra folders: e.g. the policy in force of each arm (scripts/compare_inforce.sh)
PAIRS = [("v4", "v1"), ("v4", "control"), ("control", "v1")]
PAIRS += [(x, y) for x in EXTRA for y in ("v4", "v1", "control")]
PAIRS += [(x, y) for i, x in enumerate(EXTRA) for y in EXTRA[i + 1:]]
LAST_NIGHT = {"control": 3.55, "v1": 3.38, "v4": 3.90}  # held-out, loop judge, one set each


def runs(label: str) -> dict[int, dict[str, float]]:
    out = {}
    for path in sorted(glob.glob(str(OUT / label / "*-run-*" / "scores.jsonl"))):
        n = int(Path(path).parent.name.rsplit("-", 1)[1])
        rows = [json.loads(line) for line in open(path)]
        out[n] = {r["conversation_id"].split("-")[-1]: (r["empathy"] + r["relevance"]) / 2 for r in rows}
    return out


def interval(diffs, n=5000):
    rng = random.Random(7)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


data = {lab: runs(lab) for lab in LABELS}
print("== each policy: quality of every run (a set of 20 held-out conversations)")
for lab in LABELS:
    qs = [st.mean(s.values()) for _, s in sorted(data[lab].items()) if s]
    if not qs:
        print(f"  {lab:8s} no runs yet")
        continue
    sd = st.stdev(qs) if len(qs) > 1 else 0.0
    print(f"  {lab:8s} runs {len(qs):2d}  mean {st.mean(qs):.3f}  sd {sd:.3f}  min {min(qs):.3f}  max {max(qs):.3f}"
          + (f"  | last night {LAST_NIGHT[lab]:.2f}" if lab in LAST_NIGHT else ""))
    print(f"           {' '.join(f'{q:.2f}' for q in qs)}")

print("== pairs: per-scenario difference averaged over all runs (95% interval over scenarios)")
for a, b in PAIRS:
    if not data[a] or not data[b]:
        continue
    ids = sorted(set.intersection(*[set(s) for s in data[a].values()], *[set(s) for s in data[b].values()]))
    mean_a = {i: st.mean(s[i] for s in data[a].values()) for i in ids}
    mean_b = {i: st.mean(s[i] for s in data[b].values()) for i in ids}
    d = [mean_a[i] - mean_b[i] for i in ids]
    lo, hi = interval(d)
    common = sorted(set(data[a]) & set(data[b]))
    wins = sum(st.mean(data[a][n].values()) > st.mean(data[b][n].values()) for n in common)
    ties = sum(st.mean(data[a][n].values()) == st.mean(data[b][n].values()) for n in common)
    print(f"  {a} minus {b}: {st.mean(d):+.3f} [{lo:+.2f}, {hi:+.2f}] over {len(ids)} scenarios"
          f" | run by run: {a} higher in {wins} of {len(common)}, equal {ties}")
