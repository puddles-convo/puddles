"""Calibrate the judge panel against the seeker ratings ESConv humans left."""
import argparse
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Protocol

from pydantic import BaseModel

from puddles.eval.data import ESCONV_REVISION, RealConversation, SeekerRatings, load_esconv
from puddles.eval.judge import FallbackConfig, JudgeResult, PanelConfig, judge_from_env
from puddles.llm import AgentFailed
from schemas import ConversationRecord, JudgeOutput

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
FROZEN_PATH = REPO / "calibration" / "frozen.json"


class JudgeLike(Protocol):
    kind: Literal["panel", "fallback"]
    config: "PanelConfig | FallbackConfig"

    @property
    def version(self) -> str: ...

    def score(self, record: ConversationRecord) -> JudgeResult: ...


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j + 2) / 2
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / (sxx * syy) ** 0.5


def spearman(pairs: list[tuple[int, int]]) -> Optional[float]:
    if len(pairs) < 2:
        return None
    return _pearson(_ranks([float(a) for a, _ in pairs]), _ranks([float(b) for _, b in pairs]))


def qwk(pairs: list[tuple[int, int]]) -> Optional[float]:
    n = len(pairs)
    if n == 0:
        return None
    count_j = [0] * 6
    count_h = [0] * 6
    for a, b in pairs:
        count_j[a] += 1
        count_h[b] += 1
    observed = sum((a - b) ** 2 for a, b in pairs) / n
    expected = sum(count_j[a] * count_h[b] * (a - b) ** 2 for a in range(6) for b in range(6)) / n**2
    if expected == 0:
        return None
    return 1 - observed / expected


class DimAgreement(BaseModel):
    n: int
    spearman: Optional[float] = None
    mean_abs_error: Optional[float] = None
    qwk: Optional[float] = None
    exact_match: Optional[float] = None


def agreement(pairs: list[tuple[int, int]]) -> DimAgreement:
    n = len(pairs)
    if n == 0:
        return DimAgreement(n=0)
    return DimAgreement(
        n=n,
        spearman=spearman(pairs),
        mean_abs_error=sum(abs(a - b) for a, b in pairs) / n,
        qwk=qwk(pairs),
        exact_match=sum(1 for a, b in pairs if a == b) / n,
    )


def _mae(pairs: list[tuple[int, int]]) -> Optional[float]:
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None


class SourceAgreement(BaseModel):
    model: str
    empathy: DimAgreement
    relevance: DimAgreement
    emotion_before_mae: Optional[float] = None
    emotion_after_mae: Optional[float] = None
    emotion_change_mae: Optional[float] = None
    strategy_exact_match: Optional[float] = None
    strategy_n: int = 0


Row = tuple[JudgeOutput, SeekerRatings, list[Optional[str]]]


def _source_agreement(model: str, rows: list[Row]) -> SourceAgreement:
    def dim(field: str) -> list[tuple[int, int]]:
        return [(getattr(out, field), getattr(ratings, field)) for out, ratings, _ in rows
                if getattr(ratings, field) is not None]

    before = [(out.emotion_before, ratings.emotion_before) for out, ratings, _ in rows
              if ratings.emotion_before is not None]
    after = [(out.emotion_after, ratings.emotion_after) for out, ratings, _ in rows
             if ratings.emotion_after is not None]
    change = [(out.emotion_before - out.emotion_after, ratings.emotion_before - ratings.emotion_after)
              for out, ratings, _ in rows
              if ratings.emotion_before is not None and ratings.emotion_after is not None]
    strategy_hits, strategy_n = 0, 0
    for out, _, esconv_strategies in rows:
        for predicted, gold in zip(out.strategies, esconv_strategies, strict=False):
            if gold is not None:
                strategy_n += 1
                strategy_hits += int(predicted == gold)
    return SourceAgreement(
        model=model,
        empathy=agreement(dim("empathy")),
        relevance=agreement(dim("relevance")),
        emotion_before_mae=_mae(before),
        emotion_after_mae=_mae(after),
        emotion_change_mae=_mae(change),
        strategy_exact_match=strategy_hits / strategy_n if strategy_n else None,
        strategy_n=strategy_n,
    )


class CalibrationReport(BaseModel):
    judge_version: str
    judge_kind: Literal["panel", "fallback"]
    judge_config: "PanelConfig | FallbackConfig"
    split: Literal["train", "validation"]
    esconv_revision: str
    seed: int
    n_requested: Optional[int]
    n: int
    failed: list[str]
    agreement: dict[str, DimAgreement]
    sources: dict[str, SourceAgreement]
    usage: dict[str, dict[str, int]]
    frozen: bool = False
    created_at: datetime


def calibrate(split: Literal["train", "validation"], judge: JudgeLike,
              n: Optional[int] = None, seed: int = 0, workers: int = 2,
              ) -> tuple[CalibrationReport, list[dict]]:
    if split not in ("train", "validation"):
        raise ValueError(f"calibration split must be 'train' or 'validation', got {split!r}")
    eligible = [conv for conv in load_esconv(split)
                if any(t.speaker == "helper" for t in conv.record.turns)
                and (conv.ratings.empathy is not None or conv.ratings.relevance is not None)]
    if n is not None and n < len(eligible):
        eligible = random.Random(seed).sample(eligible, n)
    eligible.sort(key=lambda conv: conv.scenario.scenario_id)

    scored: dict[str, tuple[RealConversation, JudgeResult]] = {}
    failed = []

    def run(conv: RealConversation) -> tuple[str, Optional[JudgeResult]]:
        try:
            return conv.record.id, judge.score(conv.record)
        except AgentFailed as e:
            logger.warning("scoring failed for %s: %s", conv.record.id, e)
            return conv.record.id, None

    by_id = {conv.record.id: conv for conv in eligible}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record_id, result in pool.map(run, eligible):
            if result is None:
                failed.append(record_id)
            else:
                scored[record_id] = (by_id[record_id], result)
    failed.sort()

    sources: dict[str, SourceAgreement] = {}
    if scored:
        if judge.kind == "fallback":
            rows: dict[str, list[Row]] = {"fallback": []}
            models: dict[str, str] = {}
        else:
            rows = {"astra": [], "A/initial": [], "A/final": [],
                    "B/initial": [], "B/final": [], "C/initial": [], "C/final": []}
            any_result = next(iter(scored.values()))[1]
            models = {f"{v.judge}/initial": v.model for v in any_result.verdicts}
            models.update({f"{v.judge}/final": v.model for v in any_result.verdicts})
        for conv, result in scored.values():
            if judge.kind == "fallback":
                rows["fallback"].append((_score_to_output(result), conv.ratings, conv.strategies))
            else:
                rows["astra"].append((_score_to_output(result), conv.ratings, conv.strategies))
                for verdict in result.verdicts:
                    rows[f"{verdict.judge}/initial"].append((verdict.initial, conv.ratings, conv.strategies))
                    rows[f"{verdict.judge}/final"].append((verdict.final, conv.ratings, conv.strategies))
        for key, source_rows in rows.items():
            if key == "astra" and isinstance(judge.config, PanelConfig):
                model = judge.config.astra
            elif key == "fallback" and isinstance(judge.config, FallbackConfig):
                model = judge.config.model
            else:
                model = models.get(key, "")
            sources[key] = _source_agreement(model, source_rows)
    primary = "fallback" if judge.kind == "fallback" else "astra"

    usage: dict[str, dict[str, int]] = {}
    for _, result in scored.values():
        for entry in result.usage:
            totals = usage.setdefault(entry.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            totals["calls"] += 1
            totals["prompt_tokens"] += entry.prompt_tokens
            totals["completion_tokens"] += entry.completion_tokens

    primary_source = sources.get(primary)
    report = CalibrationReport(
        judge_version=judge.version,
        judge_kind=judge.kind,
        judge_config=judge.config,
        split=split,
        esconv_revision=ESCONV_REVISION,
        seed=seed,
        n_requested=n,
        n=len(scored),
        failed=failed,
        agreement={
            "empathy": primary_source.empathy if primary_source else DimAgreement(n=0),
            "relevance": primary_source.relevance if primary_source else DimAgreement(n=0),
        },
        sources=sources,
        usage=usage,
        created_at=datetime.now(timezone.utc),
    )

    details = [
        {
            "conversation_id": conv.record.id,
            "ratings": conv.ratings.model_dump(),
            "esconv_strategies": conv.strategies,
            "score": {field: getattr(result.score, field) for field in
                      ("empathy", "relevance", "emotion_before", "emotion_after", "strategies", "note")},
            "verdicts": [verdict.model_dump() for verdict in result.verdicts],
        }
        for conv, result in (scored[conv.record.id] for conv in eligible if conv.record.id in scored)
    ]
    return report, details


def _score_to_output(result: JudgeResult) -> JudgeOutput:
    score = result.score
    return JudgeOutput(empathy=score.empathy, relevance=score.relevance, emotion_before=score.emotion_before,
                       emotion_after=score.emotion_after, strategies=score.strategies, note=score.note)


def frozen_path(kind: str) -> Path:
    return FROZEN_PATH.parent / ("frozen-fallback.json" if kind == "fallback" else "frozen.json")


def freeze(report: CalibrationReport, path: Optional[Path] = None) -> CalibrationReport:
    if report.split != "validation":
        raise ValueError(f"only a validation calibration can be frozen, got split {report.split!r}")
    frozen = report.model_copy(update={"frozen": True})
    path = Path(path) if path is not None else frozen_path(report.judge_kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frozen.model_dump_json(indent=2))
    return frozen


def check_frozen(judge, path: Optional[Path] = None) -> CalibrationReport:
    path = Path(path) if path is not None else frozen_path(judge.kind)
    if not path.is_file():
        raise ValueError("judge is not frozen; run calibration on validation with --freeze")
    report = CalibrationReport.model_validate(json.loads(path.read_text()))
    if report.judge_version != judge.version:
        raise ValueError(f"judge is frozen at {report.judge_version} but the judge is {judge.version}; recalibrate")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m puddles.eval.calibrate",
        description="Score ESConv conversations with the judge panel and report agreement with seeker ratings. "
                    "Write --output under runs/ (gitignored): the details file contains judge notes about "
                    "ESConv conversations (CC BY-NC).",
    )
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, default=None,
                        help="JSONL per scored conversation; defaults to --output with a .details.jsonl suffix")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    judge = judge_from_env()
    report, details = calibrate(args.split, judge, n=args.n, seed=args.seed, workers=args.workers)
    if args.freeze:
        report = freeze(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2))
    details_path = args.details or args.output.with_suffix(".details.jsonl")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details))
    source = "fallback" if judge.kind == "fallback" else "astra"
    print(f"scored {report.n} conversations, {len(report.failed)} failed")
    for dim in ("empathy", "relevance"):
        a = report.agreement[dim]
        spearman_text = f"{a.spearman:.3f}" if a.spearman is not None else "n/a"
        mae_text = f"{a.mean_abs_error:.3f}" if a.mean_abs_error is not None else "n/a"
        print(f"{source} {dim}: spearman={spearman_text} mae={mae_text} (n={a.n})")


if __name__ == "__main__":
    main()
