"""Calibrate the judge (issue #4): its scores against the ratings ESConv users gave themselves.

    python -m scripts.calibrate_judge --n 100

Needs OPENAI_API_KEY and PUDDLES_JUDGE_MODEL. Runs the judge on real conversations from the
validation split, prints the rank correlation and the mean absolute error for empathy and for
relevance, and saves the pairs of scores (no conversation text) for the scatter plot.
Never the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import httpx
from openai import OpenAIError

from loop.errors import ConfigError
from puddles import esconv
from puddles.agents import judge
from puddles.llm import LLM

DIMENSIONS = ("empathy", "relevance")


def ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, with tied values sharing the mean of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation, or None when it is undefined: fewer than two pairs, or one side constant."""
    if len(xs) != len(ys):
        raise ValueError("spearman needs two lists of the same length")
    if len(xs) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / (sxx * syy) ** 0.5


def mean_abs_error(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("mean_abs_error needs two lists of the same length")
    return sum(abs(a - b) for a, b in zip(xs, ys, strict=True)) / len(xs) if xs else None


def pick(conversations: list[esconv.RatedConversation], n: int) -> list[esconv.RatedConversation]:
    """n conversations spread evenly over the split, the same ones on every run."""
    if n >= len(conversations):
        return list(conversations)
    return [conversations[i * len(conversations) // n] for i in range(n)]


def calibrate(conversations: list[esconv.RatedConversation], llm: LLM, version: str, *,
              split: str, out_dir: Path) -> dict:
    """Judge each conversation, write pairs.jsonl and report.json under out_dir, return the report."""
    judged = judge.score_conversations([c.record for c in conversations], llm, version)
    ratings = {c.record.id: c.ratings for c in conversations}
    pairs = [{"conversation_id": s.conversation_id,
              "judge_empathy": s.empathy, "user_empathy": ratings[s.conversation_id].empathy,
              "judge_relevance": s.relevance, "user_relevance": ratings[s.conversation_id].relevance}
             for s in judged.scores]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
    agreement = {}
    for name in DIMENSIONS:
        ours = [p[f"judge_{name}"] for p in pairs]
        theirs = [p[f"user_{name}"] for p in pairs]
        agreement[name] = {"spearman": spearman(ours, theirs), "mean_abs_error": mean_abs_error(ours, theirs)}
    report = {"judge_version": version, "n": len(pairs), "unscored": len(judged.unscored), "split": split,
              "agreement": agreement, "frozen": False}  # freezing is a person's decision
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _show(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scripts.calibrate_judge",
                                description=(__doc__ or "").splitlines()[0])
    p.add_argument("--n", type=int, default=100, help="how many real conversations to judge")
    p.add_argument("--split", choices=["validation", "train"], default="validation")
    p.add_argument("--out", default="runs/calibration", help="where pairs.jsonl and report.json go")
    p.add_argument("--cache", default=None, help=f"ESConv cache folder (default {esconv.DEFAULT_CACHE})")
    args = p.parse_args(argv)
    try:
        llm, version = judge.llm_from_env()
        conversations = pick(esconv.real_conversations(args.split, cache_dir=args.cache), args.n)
    except (ConfigError, esconv.EsconvError, OSError, httpx.HTTPError, OpenAIError) as exc:
        print("error: " + " ".join(f"{type(exc).__name__}: {exc}".split()), file=sys.stderr)
        return 2
    report = calibrate(conversations, llm, version, split=args.split, out_dir=Path(args.out))
    print(f"{report['judge_version']}: {report['n']} conversations from the {args.split} split, "
          f"{report['unscored']} left unscored")
    for name in DIMENSIONS:
        a = report["agreement"][name]
        print(f"  {name}: rank correlation {_show(a['spearman'])}, "
              f"mean absolute error {_show(a['mean_abs_error'])}")
    print(f"pairs saved to {Path(args.out) / 'pairs.jsonl'}")
    return 0 if report["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
