"""Judge saved conversations with the judge named by the environment and keep every score.

usage: judge_saved.py <label> <conversations.jsonl> <out_dir> [in_usd_per_million out_usd_per_million]
Writes <out_dir>/<label>.scores.jsonl and <out_dir>/<label>.summary.json; prints one summary line.
"""
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from puddles.eval import judge as J
from schemas import ConversationRecord

label, path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
p_in = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
p_out = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
records = [ConversationRecord.model_validate(json.loads(line)) for line in open(path)]
judge = J.judge_from_env()


def one(rec):
    try:
        res = judge.score(rec)
        tin = sum(getattr(u, "prompt_tokens", 0) or 0 for u in res.usage)
        tout = sum(getattr(u, "completion_tokens", 0) or 0 for u in res.usage)
        return rec.id, res.score, tin, tout, None
    except Exception as exc:  # noqa: BLE001
        return rec.id, None, 0, 0, f"{type(exc).__name__}: {str(exc)[:200]}"


with ThreadPoolExecutor(4) as pool:
    out = list(pool.map(one, records))
ok = [s for _, s, _, _, e in out if s is not None]
bad = [(i, e) for i, _, _, _, e in out if e]
tin = sum(o[2] for o in out)
tout = sum(o[3] for o in out)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{label}.scores.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in ok))
summary = {"label": label, "judge": judge.config.model, "judge_version": judge.version, "conversations": len(records),
           "policy_versions": sorted({r.policy_version for r in records}),
           "scored": len(ok), "failed": [i for i, _ in bad], "input_tokens": tin, "output_tokens": tout,
           "cost_usd": round((tin * p_in + tout * p_out) / 1e6, 4)}
if ok:
    emp = [s.empathy for s in ok]
    rel = [s.relevance for s in ok]
    summary.update(empathy=round(statistics.mean(emp), 3), relevance=round(statistics.mean(rel), 3),
                   quality=round((statistics.mean(emp) + statistics.mean(rel)) / 2, 3),
                   emotion_change=round(statistics.mean(s.emotion_after - s.emotion_before for s in ok), 3),
                   safety_flags=sum(1 for s in ok if getattr(s, "safety_flags", None)))
(out_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
for i, e in bad[:3]:
    print("failed:", i, e)
