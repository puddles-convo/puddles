"""Stage 6: the plots for docs/results/.

Every number comes from a file: the round events in <root>/log/decisions.jsonl,
the calibration pairs, the baseline results.json, and an optional final-eval JSON.
A missing input skips that plot with a printed note; nothing is invented.

    python -m scripts.plots --root runs/overnight --out docs/results
    python -m scripts.plots --baseline-root runs/baseline --final-eval runs/final/result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from puddles.decision_log import DecisionLog
from puddles.records import load_record
from schemas import ResearchNote, RoundEvent


def _quality(scores) -> float | None:
    return scores.quality if scores is not None else None


def quality_figure(rounds: list[RoundEvent], out: Path) -> bool:
    """Quality by round, noise band, kept/reverted marks."""
    scored = [r for r in rounds if r.scores is not None]
    if not scored:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [r.round for r in scored]
    after = [_quality(r.scores.after) for r in scored]
    before = [_quality(r.scores.before) for r in scored]
    noise = next((r.scores.noise for r in scored if r.scores.noise), 0.0)
    if before and noise:
        ref = before[0] if before[0] is not None else after[0]
        ax.axhspan(ref - noise, ref + noise, color="gray", alpha=0.15, label=f"noise ±{noise:.2f}")
    style = {"kept": ("^", "tab:green"), "reverted": ("v", "tab:red")}
    for outcome, (marker, color) in style.items():
        pts = [
            (r.round, _quality(r.scores.after))
            for r in scored
            if r.outcome == outcome and _quality(r.scores.after) is not None
        ]
        if pts:
            ax.scatter(
                [p[0] for p in pts], [p[1] for p in pts], marker=marker, color=color, s=80, zorder=3, label=outcome
            )
    plain = [
        (r.round, _quality(r.scores.after))
        for r in scored
        if r.outcome not in style and _quality(r.scores.after) is not None
    ]
    if plain:
        ax.scatter(
            [p[0] for p in plain], [p[1] for p in plain], marker="o", color="tab:gray", s=50, zorder=2, label="other"
        )
    line = [(x, y) for x, y in zip(xs, after, strict=True) if y is not None]
    if len(line) > 1:
        ax.plot([p[0] for p in line], [p[1] for p in line], color="tab:blue", alpha=0.5, zorder=1)
    ax.set_xlabel("round")
    ax.set_ylabel("quality")
    ax.set_title("Quality by round")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def _source_for(event: RoundEvent, root: Path) -> str:
    ref = event.records.get("research")
    if ref is None:
        return "—"
    note = load_record(ref, ResearchNote, root)
    if note is None or not note.sources:
        return "—"
    return note.sources[0].title or note.sources[0].url


def rounds_table(rounds: list[RoundEvent], root: Path, out: Path) -> bool:
    """One row per round: pattern, source, edit, outcome, reason."""
    if not rounds:
        return False
    header = ["round", "pattern", "source", "edit", "outcome", "reason"]
    rows = [header]
    for r in rounds:
        pattern = r.pattern.name if r.pattern else "—"
        edit = f"{r.edit.action} {r.edit.rule_id}" if r.edit else "—"
        rows.append([str(r.round), pattern, _source_for(r, root), edit, r.outcome, r.reason])
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(cellText=rows, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for j in range(len(header)):
        table[0, j].set_facecolor("#dddddd")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _pairs(line: dict) -> tuple[float, float, float, float] | None:
    """(judge_empathy, user_empathy, judge_relevance, user_relevance) or None."""
    try:
        if "judge_empathy" in line:
            return (
                float(line["judge_empathy"]),
                float(line["user_empathy"]),
                float(line["judge_relevance"]),
                float(line["user_relevance"]),
            )
        score, ratings = line["score"], line["ratings"]
        return (
            float(score["empathy"]),
            float(ratings["empathy"]),
            float(score["relevance"]),
            float(ratings["relevance"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def calibration_figure(pairs_path: Path, out: Path) -> bool:
    """Judge score against user score, one panel per dimension."""
    if not pairs_path.exists():
        return False
    empathy, relevance = [], []
    for line in pairs_path.read_text().splitlines():
        if not line.strip():
            continue
        pair = _pairs(json.loads(line))
        if pair is not None:
            empathy.append(pair[:2])
            relevance.append(pair[2:])
    if not empathy:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, pts, title in ((axes[0], empathy, "empathy"), (axes[1], relevance, "relevance")):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=25, alpha=0.7)
        ax.plot([1, 5], [1, 5], "k--", linewidth=1)
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
        ax.set_xlabel("judge")
        ax.set_title(title)
    axes[0].set_ylabel("user")
    fig.suptitle("Judge vs user scores")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def baselines_figure(results_path: Path, final_path: Path | None, out: Path) -> bool:
    """Final policy against both baselines; spread is the measured noise."""
    if not results_path.exists():
        return False
    data = json.loads(results_path.read_text())
    bars: list[tuple[str, float, float | None]] = []
    seen: set[str] = set()
    for group in data.get("groups", []):
        for version, entry in group.get("policies", {}).items():
            if version in seen:
                continue
            seen.add(version)
            mean = entry.get("mean_scores", {}).get("quality")
            if mean is not None:
                bars.append((version, float(mean), entry.get("noise")))
    if final_path is not None and final_path.exists():
        final = json.loads(final_path.read_text())
        q = final.get("quality")
        if q is None and isinstance(final.get("scores"), dict):
            q = final["scores"].get("quality")
        if q is not None:
            bars.append(("final", float(q), None))
    if not bars:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], yerr=[b[2] / 2 if b[2] else 0 for b in bars], capsize=4)
    ax.set_xticks(list(xs), [b[0] for b in bars], rotation=20, ha="right")
    ax.set_ylabel("quality")
    ax.set_title("Policy quality vs baselines")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("runs/overnight"))
    parser.add_argument("--baseline-root", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--calibration", type=Path, default=Path("runs/calibration/pairs.jsonl"))
    parser.add_argument("--final-eval", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    log = DecisionLog(args.root)
    rounds = log.rounds()

    plots = [
        ("quality.png", lambda: quality_figure(rounds, args.out / "quality.png")),
        ("rounds.png", lambda: rounds_table(rounds, args.root, args.out / "rounds.png")),
        ("calibration.png", lambda: calibration_figure(args.calibration, args.out / "calibration.png")),
        (
            "baselines.png",
            lambda: baselines_figure(args.baseline_root / "results.json", args.final_eval, args.out / "baselines.png"),
        ),
    ]
    wrote = 0
    for name, plot in plots:
        try:
            ok = plot()
        except Exception as e:  # a bad input file must not kill the other plots
            print(f"skip {name}: {e}", file=sys.stderr)
            continue
        if ok:
            print(f"wrote {args.out / name}")
            wrote += 1
        else:
            print(f"skip {name}: missing or empty input")
    print(f"{wrote}/{len(plots)} plots written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
