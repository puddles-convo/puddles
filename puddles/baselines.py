"""Stage 5: score a policy on fixed scenario IDs, and measure run-to-run noise.

Each scoring run writes `conversations.jsonl`, `scores.jsonl` and `result.json` into its
own folder under the runs root. Two runs may be compared only when their settings
fingerprints match: the same scenario IDs, seed, turn limit and judge version. A run's
scores are reported only when every conversation came back scored; an average over the
conversations that survived can look like a gain.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Protocol, Sequence

from pydantic import BaseModel

from loop.errors import ConfigError
from loop.llm import ChatClient
from loop.runner import run_batch
from puddles.graph import check_policy_file
from puddles.tracing import TracedClient
from puddles.usage import Meter, Prices
from schemas import ConversationRecord, Scenario, ScoreRecord, Scores


class BaselineError(ConfigError):
    """The scenarios, the ID file or the run settings cannot be used for a baseline."""


class Judge(Protocol):
    """What a baseline run needs from a judge: its frozen version and a score per conversation.

    `score` returns an object with `.score` (a `ScoreRecord`) and `.usage` (token counts
    per model). `puddles.eval.judge.JudgePanel` and `SingleJudge` both match.
    """

    version: str

    def score(self, record: ConversationRecord) -> Any: ...


class BaselineResult(BaseModel):
    """One scoring run's evidence, written to `result.json`."""

    policy_version: str
    run: int
    n: int  # the scenarios the run was given; scored ones are len(scores.jsonl)
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    conversations_failed: list[str] = []
    unscored: list[str] = []
    complete: bool = False  # true only when every conversation came back scored
    empathy: Optional[float] = None
    relevance: Optional[float] = None
    quality: Optional[float] = None
    emotion_change: Optional[float] = None
    seed: Optional[int] = None
    max_turns: int = 10
    judge_version: str = ""
    scenario_ids_sha256: str = ""


def load_id_file(path: str | Path) -> list[str]:
    """One scenario ID per line. A blank file or a repeated ID is a BaselineError."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    ids = [line for line in lines if line]
    if not ids:
        raise BaselineError(f"{path}: no scenario IDs")
    seen = set()
    for scenario_id in ids:
        if scenario_id in seen:
            raise BaselineError(f"{path}: repeated scenario ID {scenario_id!r}")
        seen.add(scenario_id)
    return ids


def resolve_ids(scenarios: Sequence[Scenario], ids: Sequence[str]) -> list[Scenario]:
    """The scenarios named by `ids`, in order. An unknown ID, a repeat or a test-split
    scenario is a BaselineError: the validation set never touches the test split."""
    by_id = {s.scenario_id: s for s in scenarios}
    chosen, seen = [], set()
    for scenario_id in ids:
        if scenario_id in seen:
            raise BaselineError(f"repeated scenario ID {scenario_id!r}")
        seen.add(scenario_id)
        scenario = by_id.get(scenario_id)
        if scenario is None:
            raise BaselineError(f"unknown scenario ID {scenario_id!r}")
        if scenario.split == "test":
            raise BaselineError(f"{scenario_id!r} is in the test split; baselines never use it")
        chosen.append(scenario)
    if not chosen:
        raise BaselineError("no scenario IDs given")
    return chosen


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def average(records: Sequence[ScoreRecord]) -> Scores:
    """The gate's formula: means over the records, quality the mean of (empathy + relevance) / 2."""
    if not records:
        raise BaselineError("no score records to average")
    n = len(records)
    return Scores(
        empathy=sum(r.empathy for r in records) / n,
        relevance=sum(r.relevance for r in records) / n,
        quality=sum((r.empathy + r.relevance) / 2 for r in records) / n,
        emotion_change=sum(r.emotion_before - r.emotion_after for r in records) / n,
    )


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as temp:
        temp.write(json.dumps(data, indent=2) + "\n")
    Path(temp.name).replace(path)


def score_once(policy_path: str | Path, scenarios: Sequence[Scenario], ids: Sequence[str], *,
               helper_client: ChatClient, user_client: ChatClient, judge: Judge,
               meter: Meter, prices: Optional[Prices] = None, run: int = 1,
               seed: Optional[int] = None, max_turns: int = 10,
               run_dir: str | Path) -> BaselineResult:
    """Converse on the named scenarios, judge each conversation, write the run's evidence.

    `meter` writes the helper's and simulated user's usage lines into
    `<meter root>/log/usage.jsonl` and holds their tallies; the judge's tokens come back
    in each `JudgeResult.usage` and are costed at `prices`.
    """
    ids = list(ids)
    policy = check_policy_file(Path(policy_path))
    chosen = resolve_ids(scenarios, ids)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prices = prices if prices is not None else meter.prices

    meter.start_round(0)
    started = time.monotonic()
    batch = run_batch(TracedClient(helper_client, "helper", meter=meter),
                      TracedClient(user_client, "simulated_user", meter=meter),
                      policy, chosen, round=0, max_turns=max_turns, seed=seed,
                      out_path=run_dir / "conversations.jsonl")

    scores: list[ScoreRecord] = []
    unscored: list[str] = []
    judge_usage: list[Any] = []
    for record in batch.records:
        try:
            result = judge.score(record)
            score = result.score
        except Exception:
            unscored.append(record.id)
            continue
        if not isinstance(score, ScoreRecord) or score.conversation_id != record.id:
            unscored.append(record.id)
            continue
        scores.append(score)
        judge_usage.extend(getattr(result, "usage", []) or [])
    seconds = time.monotonic() - started

    scores_path = run_dir / "scores.jsonl"
    scores_path.write_text("".join(s.model_dump_json() + "\n" for s in scores), encoding="utf-8")

    tally = meter.usage
    judge_tokens_in = sum(getattr(u, "prompt_tokens", 0) for u in judge_usage)
    judge_tokens_out = sum(getattr(u, "completion_tokens", 0) for u in judge_usage)
    judge_cost = sum(prices.cost(getattr(u, "model", ""), getattr(u, "prompt_tokens", 0),
                                 getattr(u, "completion_tokens", 0)) for u in judge_usage)

    failed_ids = [f.scenario_id for f in batch.failures]
    complete = not failed_ids and not unscored and len(scores) == len(ids)
    measured = average(scores) if complete else None

    result = BaselineResult(
        policy_version=policy.version, run=run, n=len(ids), seconds=seconds,
        input_tokens=sum(u.tokens_in for u in tally.values()) + judge_tokens_in,
        output_tokens=sum(u.tokens_out for u in tally.values()) + judge_tokens_out,
        cost_usd=sum(u.cost_usd for u in tally.values()) + judge_cost,
        conversations_failed=sorted(failed_ids), unscored=sorted(unscored), complete=complete,
        empathy=measured.empathy if measured else None,
        relevance=measured.relevance if measured else None,
        quality=measured.quality if measured else None,
        emotion_change=measured.emotion_change if measured else None,
        seed=seed, max_turns=max_turns, judge_version=judge.version,
        scenario_ids_sha256=ids_sha256(ids))
    _atomic_write_json(run_dir / "result.json", result.model_dump())
    return result


def fingerprint(result: BaselineResult) -> dict:
    """What must match for two runs to be compared."""
    return {"seed": result.seed, "max_turns": result.max_turns,
            "judge_version": result.judge_version, "scenario_ids_sha256": result.scenario_ids_sha256}


def noise(results: Sequence[BaselineResult]) -> float:
    """Highest quality minus lowest across runs with identical settings. Needs at least two."""
    complete = [r for r in results if r.complete]
    if len(complete) < 2:
        raise BaselineError("noise needs at least two complete runs")
    marks = {json.dumps(fingerprint(r), sort_keys=True) for r in complete}
    if len(marks) != 1:
        raise BaselineError("runs differ in seed, turn limit, judge or scenarios; they cannot be compared")
    qualities = [r.quality for r in complete]
    return max(qualities) - min(qualities)


def mean_scores(results: Sequence[BaselineResult]) -> Scores:
    """The mean of several runs' scores: the gate's starting `current_scores`."""
    complete = [r for r in results if r.complete]
    if not complete:
        raise BaselineError("no complete runs to average")
    return Scores(
        empathy=sum(r.empathy for r in complete) / len(complete),
        relevance=sum(r.relevance for r in complete) / len(complete),
        quality=sum(r.quality for r in complete) / len(complete),
        emotion_change=sum(r.emotion_change for r in complete) / len(complete))


def read_results(root: str | Path) -> list[BaselineResult]:
    """Every `result.json` under the root, in file order."""
    out = []
    for path in sorted(Path(root).glob("*/result.json")):
        out.append(BaselineResult.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    return out


def summarize(root: str | Path) -> dict:
    """Aggregate every run under the root into `results.json`.

    Runs are grouped by their settings fingerprint and then by policy. A policy with two
    or more complete runs gets its noise and its mean scores. Groups whose fingerprints
    differ are reported side by side and never compared.
    """
    root = Path(root)
    results = read_results(root)
    groups: dict[str, list[BaselineResult]] = {}
    for result in results:
        groups.setdefault(json.dumps(fingerprint(result), sort_keys=True), []).append(result)
    out_groups = []
    for mark, runs in sorted(groups.items()):
        by_policy: dict[str, dict] = {}
        for policy_version in {r.policy_version for r in runs}:
            same = [r for r in runs if r.policy_version == policy_version]
            entry: dict[str, Any] = {"runs": [r.run for r in same],
                                     "complete": sum(1 for r in same if r.complete)}
            complete = [r for r in same if r.complete]
            if len(complete) >= 2:
                entry["noise"] = noise(complete)
                entry["mean_scores"] = mean_scores(complete).model_dump()
            elif complete:
                entry["mean_scores"] = mean_scores(complete).model_dump()
            by_policy[policy_version] = entry
        out_groups.append({"fingerprint": json.loads(mark), "policies": by_policy})
    summary = {"root": str(root), "runs": [r.model_dump() for r in results], "groups": out_groups}
    _atomic_write_json(root / "results.json", summary)
    return summary
