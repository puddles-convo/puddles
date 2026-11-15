"""Gatekeeper (step 7): tests a candidate policy, then keeps it or leaves the policy in force alone.

The contract is docs/specs/gate.md; the rule is in docs/agents/t7-gatekeeper.md. It is code, not
a model. The judge and the safety check are passed in through `GateDeps`, so this module never
imports `puddles.eval`.

It never writes a policy file. `finish_round` writes the promoted policy only when the decision
is `kept`, so a reverted candidate leaves the policy in force byte-for-byte as it was. Anything
that goes wrong while testing reverts: nothing is kept on incomplete evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from loop.helper import helper_reply
from loop.llm import ChatClient
from loop.policy import Policy
from loop.runner import run_batch
from puddles import replay
from puddles.graph import check_policy_file
from puddles.memory import now
from puddles.policy import TOKEN_CAP, estimate_tokens, parse_policy
from puddles.round import RoundResult
from schemas import (
    ConversationRecord,
    GateInput,
    GateOutcome,
    LogSafety,
    LogScores,
    SafetyResult,
    Scenario,
    ScoreRecord,
    Scores,
    Turn,
)

VALIDATION_TAG = "val"  # the candidate's validation conversations: conv-rNN-val-<scenario>
CURRENT_TAG = "current"  # the policy in force: conv-r00-current-<scenario>
BASELINE_ROUND = 0
NOT_MEASURED = Scores(empathy=0, relevance=0)  # scores are 1 to 5, so 0 can only mean "not measured"

# Check name -> the words a reason uses for it, in deciding order. The numbers are the spec's.
CHECKS: dict[str, str] = {
    "candidate_file": "a policy file that does not load",
    "safety_section": "a changed Safety section (check 1)",
    "one_rule": "not exactly one rule changed (check 2)",
    "token_cap": "the token cap (check 3)",
    "safety_cases": "a failed safety check (check 4)",
    "complete_evidence": "incomplete evidence",
    "quality_gain": "no quality gain above the noise (check 5)",
    "dimension_drop": "a dimension falling by more than the noise (check 6)",
    "noise_not_measured": "unmeasured noise",
}
STATIC = ("safety_section", "one_rule", "token_cap")
NOTHING_ELSE_RAN = "Nothing else was run: no conversation, no judge call, no safety check."
NOISE_NOTE = "the noise has not been measured, so nothing is promoted"


class GateError(RuntimeError):
    pass


class Judge(Protocol):
    version: str

    def score(self, record: ConversationRecord) -> Any:  # the returned object has `.score`, a ScoreRecord
        ...


Reply = Callable[[list[Turn]], str]
SafetyCheck = Callable[[str, Reply], SafetyResult]  # (policy_version, reply) -> SafetyResult


@dataclass
class GateDeps:
    helper_client: ChatClient
    user_client: ChatClient
    judge: Judge
    safety: SafetyCheck
    root: Path
    scenarios: list[Scenario]  # where validation_scenario_ids are looked up
    noise_measured: bool = False  # stage 5 turns it on; until then nothing is promoted
    max_turns: int = 10
    seed: Optional[int] = None


@dataclass
class Evidence:
    """One scoring run of one policy on a list of scenarios."""

    expected: int
    scores: Optional[Scores] = None  # set only when every scenario came back with a score
    failed: list[str] = field(default_factory=list)  # scenario IDs whose conversation failed
    unscored: list[str] = field(default_factory=list)  # conversation IDs the judge did not score

    @property
    def complete(self) -> bool:
        return self.scores is not None and not self.failed and not self.unscored

    def describe(self) -> str:
        parts = []
        if self.failed:
            parts.append(f"{len(self.failed)} of {self.expected} conversations failed ({', '.join(self.failed)})")
        if self.unscored:
            parts.append(f"{len(self.unscored)} of {self.expected} were not scored by the judge "
                         f"({', '.join(self.unscored)})")
        return "; ".join(parts) or "the scores are missing"


@dataclass
class GateReport:
    outcome: GateOutcome
    failed: list[str]  # names from CHECKS, in CHECKS order; empty only when kept
    would_keep: bool  # what the six checks said, whatever the noise flag then did
    checks: dict[str, str]  # every key of CHECKS -> "passed", "failed" or "not_run"
    log_scores: Optional[LogScores] = None  # None when validation gave no complete scores
    log_safety: Optional[LogSafety] = None  # None when the safety check did not run
    safety: Optional[SafetyResult] = None


# ---------- scores ----------

def average(scores: list[ScoreRecord]) -> Scores:
    """Means over the conversations. Quality is the mean of (empathy + relevance) / 2."""
    if not scores:
        raise GateError("no scores to average")
    n = len(scores)
    return Scores(empathy=sum(s.empathy for s in scores) / n,
                  relevance=sum(s.relevance for s in scores) / n,
                  quality=sum((s.empathy + s.relevance) / 2 for s in scores) / n,
                  emotion_change=sum(s.emotion_before - s.emotion_after for s in scores) / n)


def with_quality(scores: Scores) -> Scores:
    if scores.quality is not None:
        return scores
    return scores.model_copy(update={"quality": (scores.empathy + scores.relevance) / 2})


def _batch_name(tag: str) -> str:
    return "validation" if tag == VALIDATION_TAG else tag


def _pick(pool: list[Scenario], scenario_ids: list[str]) -> list[Scenario]:
    if not scenario_ids:
        raise GateError("no scenarios to score on")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise GateError("a scenario ID is listed twice")
    by_id = {s.scenario_id: s for s in pool}
    missing = [i for i in scenario_ids if i not in by_id]
    if missing:
        raise GateError(f"scenarios not found: {', '.join(missing)}")
    return [by_id[i] for i in scenario_ids]


def score_policy(deps: GateDeps, policy_path: Path, scenario_ids: list[str], round_: int, tag: str) -> Evidence:
    """Converse on every scenario under the policy, then judge each conversation once.

    Both files are saved, the scores even when some are missing. Scores are averaged only when
    every conversation finished and was scored: averaging the survivors can look like a gain.
    """
    policy = check_policy_file(Path(policy_path))
    scenarios = _pick(deps.scenarios, scenario_ids)
    name = f"round-{round_:02d}-{_batch_name(tag)}.jsonl"
    root = Path(deps.root)
    batch = run_batch(deps.helper_client, deps.user_client, policy, scenarios, round=round_,
                      max_turns=deps.max_turns, seed=deps.seed, tag=tag,
                      out_path=root / "conversations" / name)
    scores: list[ScoreRecord] = []
    unscored: list[str] = []
    for record in batch.records:
        try:
            score = deps.judge.score(record).score
            if not isinstance(score, ScoreRecord) or score.conversation_id != record.id:
                raise GateError(f"the judge gave no score for {record.id}")
            scores.append(score)
        except Exception:  # noqa: BLE001  that conversation is unscored; the rest are still scored
            unscored.append(record.id)
    scores_path = root / "scores" / name
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text("".join(s.model_dump_json() + "\n" for s in scores), encoding="utf-8")
    evidence = Evidence(expected=len(scenarios), failed=[f.scenario_id for f in batch.failures],
                        unscored=unscored)
    if not evidence.failed and not evidence.unscored and len(scores) == len(scenarios):
        evidence.scores = average(scores)
    return evidence


def current_path(root: Path) -> Path:
    return Path(root) / "scores" / "current.json"


def _write_current(root: Path, policy_version: str, scenario_ids: list[str], judge_version: str,
                   scores: Scores) -> None:
    path = current_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"policy_version": policy_version, "scenario_ids": list(scenario_ids), "judge_version": judge_version,
            "scores": scores.model_dump()}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _saved_current(root: Path, policy_version: str, scenario_ids: list[str], judge_version: str) -> Optional[Scores]:
    """The kept scores, if they are for this policy, these scenarios and this judge. A file that
    cannot be read counts as missing, so the policy is scored again."""
    path = current_path(root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["policy_version"], data["scenario_ids"], data["judge_version"]) != \
                (policy_version, list(scenario_ids), judge_version):
            return None
        return Scores.model_validate(data["scores"])
    except Exception:  # noqa: BLE001
        return None


def current_scores(deps: GateDeps, policy_path: Path, scenario_ids: list[str]) -> Scores:
    """The scores of the policy in force on the validation scenarios, kept in scores/current.json.

    Scored as round 0 with the tag `current` when the file is missing or is for another policy,
    another list of scenarios or another judge. Incomplete evidence raises GateError.
    """
    version = check_policy_file(Path(policy_path)).version
    saved = _saved_current(deps.root, version, scenario_ids, deps.judge.version)
    if saved is not None:
        return saved
    evidence = score_policy(deps, Path(policy_path), scenario_ids, BASELINE_ROUND, CURRENT_TAG)
    if not evidence.complete or evidence.scores is None:
        raise GateError(f"the policy in force could not be scored: {evidence.describe()}")
    _write_current(deps.root, version, scenario_ids, deps.judge.version, evidence.scores)
    return evidence.scores


# ---------- the decision ----------

def _num(x: float) -> str:
    x = round(x, 4)
    return f"{x:.2f}" if round(x, 2) == x else f"{x}"


def _changed_rules(current: Policy, candidate: Policy) -> list[str]:
    """Conversation rules added, removed or reworded, by ID, text compared exactly."""
    def conversation(p: Policy) -> dict[str, str]:
        return {rid: text for rid, text in p.rules.items() if rid.startswith("C")}
    a, b = conversation(current), conversation(candidate)
    return [rid for rid in dict.fromkeys([*a, *b]) if a.get(rid) != b.get(rid)]


def _version(path: str) -> str:
    """A policy's version from its title line, or the file name if it cannot be read."""
    try:
        first = Path(path).read_text(encoding="utf-8").splitlines()[0]
    except Exception:  # noqa: BLE001
        first = ""
    return first[2:].strip() if first.startswith("# ") else Path(path).stem


def _reason(failed: list[str], details: dict[str, str], noise_measured: bool, would_keep: bool,
            verdict: str, extra: str = "") -> str:
    if not failed:
        return f"Kept: {verdict}."
    if would_keep and failed == ["noise_not_measured"]:
        return (f"Reverted on {CHECKS['noise_not_measured']}: {NOISE_NOTE}. "
                f"The six checks would have kept the candidate: {verdict}.")
    first, *rest = failed
    parts = [f"Reverted on {CHECKS[first]}: {details[first]}."]
    for name in rest:
        parts.append(f"Also {CHECKS[name]}: {details[name]}.")
    if extra:
        parts.append(extra)
    return " ".join(parts)


@dataclass
class _Run:
    """What the evaluation found so far, so a crash part-way still reports what ran."""

    inp: GateInput
    deps: GateDeps
    checks: dict[str, str] = field(default_factory=lambda: {k: "not_run" for k in CHECKS})
    details: dict[str, str] = field(default_factory=dict)
    running: Optional[str] = None
    current_version: Optional[str] = None
    candidate_version: Optional[str] = None
    safety: Optional[SafetyResult] = None
    after: Optional[Scores] = None
    extra: str = ""

    def fail(self, name: str, detail: str) -> None:
        self.checks[name], self.details[name] = "failed", detail

    def report(self, verdict: str = "") -> GateReport:
        inp, deps = self.inp, self.deps
        noise_measured = bool(deps.noise_measured)
        self.checks["noise_not_measured"] = "passed" if noise_measured else "failed"
        if not noise_measured:
            self.details["noise_not_measured"] = NOISE_NOTE
        eight = [k for k in CHECKS if k != "noise_not_measured"]
        would_keep = all(self.checks[k] == "passed" for k in eight)
        failed = [k for k in CHECKS if self.checks[k] == "failed"]
        kept = would_keep and noise_measured and not failed
        before = with_quality(inp.current_scores)
        after = self.after if self.after is not None else NOT_MEASURED
        safety = self.safety
        if kept:
            assert self.candidate_version is not None
            in_force = self.candidate_version.removesuffix("-candidate")
        else:
            in_force = self.current_version or _version(inp.current_policy)
        outcome = GateOutcome(
            id=f"gate-r{inp.round:02d}", written_by="gatekeeper", created_at=now(), parent_ids=[inp.edit_id],
            round=inp.round, edit_id=inp.edit_id, decision="kept" if kept else "reverted", before=before,
            after=after, safety_passed=bool(safety is not None and safety.passed), policy_in_force=in_force,
            reason=_reason(failed, self.details, noise_measured, would_keep, verdict, self.extra))
        log_safety = None
        if safety is not None:
            log_safety = LogSafety(case_file=safety.case_file, passed=safety.passed, cases=len(safety.cases),
                                   failed_case_ids=[c.case_id for c in safety.cases if not c.passed])
        log_scores = LogScores(before=before, after=self.after, noise=inp.noise) if self.after is not None else None
        return GateReport(outcome=outcome, failed=failed, would_keep=would_keep, checks=dict(self.checks),
                          log_scores=log_scores, log_safety=log_safety, safety=safety)


def _scores_dir(deps: GateDeps) -> Path:
    path = Path(deps.root) / "scores"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _evaluate(run: _Run) -> GateReport:
    inp, deps = run.inp, run.deps
    # 1. Static checks, no model call.
    run.running = "candidate_file"
    try:
        current = check_policy_file(Path(inp.current_policy))
        run.current_version = current.version
        candidate = check_policy_file(Path(inp.candidate_policy))
        run.candidate_version = candidate.version
    except Exception as exc:  # noqa: BLE001  a file that does not load is this check failing
        run.fail("candidate_file", f"{type(exc).__name__}: {exc}")
        run.extra = NOTHING_ELSE_RAN
        return run.report()
    run.checks["candidate_file"] = "passed"
    run.running = "safety_section"
    if candidate.safety_sha256 != current.safety_sha256:
        run.fail("safety_section", "the candidate's Safety section differs from the current policy's")
    else:
        run.checks["safety_section"] = "passed"
    run.running = "one_rule"
    changed = _changed_rules(current, candidate)
    if len(changed) != 1:
        which = f" ({', '.join(changed)})" if changed else ""
        run.fail("one_rule", f"{len(changed)} Conversation rules differ from the current policy{which}, "
                             "not exactly one")
    else:
        run.checks["one_rule"] = "passed"
    run.running = "token_cap"
    tokens = estimate_tokens(candidate.text)
    if tokens > inp.token_cap:
        run.fail("token_cap", f"the candidate is about {tokens} tokens, over the cap of {inp.token_cap}")
    else:
        run.checks["token_cap"] = "passed"
    if any(run.checks[k] == "failed" for k in STATIC):
        run.extra = NOTHING_ELSE_RAN
        return run.report()

    # 2. The safety check, against the helper alone under the candidate, with no crisis checker.
    run.running = "safety_cases"
    session = f"safety-r{inp.round:02d}"

    def reply(turns: list[Turn]) -> str:
        return helper_reply(deps.helper_client, candidate, turns, session_id=session, turn=1).text

    safety = deps.safety(candidate.version, reply)
    if not isinstance(safety, SafetyResult):
        raise GateError(f"the safety check returned {type(safety).__name__}, not a SafetyResult")
    (_scores_dir(deps) / f"round-{inp.round:02d}-safety.json").write_text(
        safety.model_dump_json(indent=2) + "\n", encoding="utf-8")
    run.safety = safety
    if safety.passed:
        run.checks["safety_cases"] = "passed"
    else:
        failed_ids = [c.case_id for c in safety.cases if not c.passed]
        which = f" ({', '.join(failed_ids)})" if failed_ids else ""
        run.fail("safety_cases", f"{len(failed_ids)} of {len(safety.cases)} safety cases failed{which}")

    # 3. Validation. It runs whatever the safety check said.
    run.running = "complete_evidence"
    evidence = score_policy(deps, Path(inp.candidate_policy), list(inp.validation_scenario_ids), inp.round,
                            VALIDATION_TAG)
    if not evidence.complete or evidence.scores is None:
        run.fail("complete_evidence", f"of the validation conversations, {evidence.describe()}")
        return run.report()
    run.checks["complete_evidence"] = "passed"

    # 4. The two score checks. Differences are rounded: the scores are means of whole numbers.
    run.running = "quality_gain"
    before, after = with_quality(inp.current_scores), evidence.scores
    assert before.quality is not None and after.quality is not None
    gain = round(after.quality - before.quality, 9)
    moved = f"quality {'rose' if gain >= 0 else 'fell'} by {_num(abs(gain))}"
    if gain > inp.noise:
        run.checks["quality_gain"] = "passed"
    else:
        run.fail("quality_gain", f"{moved}, which is not more than the noise of {_num(inp.noise)}")
    run.running = "dimension_drop"
    drops = []
    for name in ("empathy", "relevance"):
        drop = round(getattr(before, name) - getattr(after, name), 9)
        if drop > inp.noise:
            drops.append(f"{name} fell by {_num(drop)}, more than the noise of {_num(inp.noise)}")
    if drops:
        run.fail("dimension_drop", " and ".join(drops))
    else:
        run.checks["dimension_drop"] = "passed"
    run.after = after
    run.running = None
    verdict = (f"{moved}, above the noise of {_num(inp.noise)}; no dimension fell by more than the noise; "
               f"the safety check passed ({len(safety.cases)} cases)")
    return run.report(verdict)


def _save_report(deps: GateDeps, round_: int, report: GateReport) -> None:
    data = {"round": round_, "decision": report.outcome.decision, "would_keep": report.would_keep,
            "failed": report.failed, "checks": report.checks, "reason": report.outcome.reason}
    (_scores_dir(deps) / f"round-{round_:02d}-gate.json").write_text(json.dumps(data, indent=2) + "\n",
                                                                     encoding="utf-8")


def _crashed(run: _Run, exc: BaseException) -> GateReport:
    """The revert for a crash: the check that was running failed, and the evidence is incomplete."""
    before_evidence = ("candidate_file", *STATIC, "safety_cases")
    if run.running in before_evidence and run.checks[run.running] == "not_run":
        run.fail(run.running, f"it could not finish ({type(exc).__name__}: {exc})")
    if run.running == "safety_cases":
        run.safety = None
    run.after = None
    for name in ("quality_gain", "dimension_drop"):  # without complete evidence they did not run
        run.checks[name] = "not_run"
        run.details.pop(name, None)
    run.fail("complete_evidence", f"the test run crashed ({type(exc).__name__}: {exc})")
    run.extra = ""
    return run.report()


def run_gate(inp: GateInput, deps: GateDeps) -> GateReport:
    """Apply the checks in order. It never raises: anything that goes wrong reverts the candidate."""
    run = _Run(inp=inp, deps=deps)
    try:
        report = _evaluate(run)
        _save_report(deps, inp.round, report)
        return report
    except Exception as exc:  # noqa: BLE001  any crash at all must end as a revert, never as a keep
        report = _crashed(run, exc)
    try:
        _save_report(deps, inp.round, report)
    except Exception:  # noqa: BLE001  the decision is already a revert; a lost report file cannot change it
        pass
    return report


# ---------- the gate as the graph calls it ----------

class _Gate:
    """`RoundDeps.gate`. Never raises; keeps the last report on `last_report`."""

    def __init__(self, deps: GateDeps, validation_scenario_ids: list[str], noise: float, token_cap: int,
                 safety_case_set: str) -> None:
        self.deps, self.ids, self.noise = deps, list(validation_scenario_ids), noise
        self.token_cap, self.safety_case_set = token_cap, safety_case_set
        self.last_report: Optional[GateReport] = None

    def __call__(self, result: RoundResult) -> GateOutcome:
        result.scores, result.safety = None, None
        try:
            report = self._gate(result)
        except Exception as exc:  # noqa: BLE001  the candidate is reverted untested
            report = self._untested(result, f"the gate could not run ({type(exc).__name__}: {exc})")
        self.last_report = report
        return report.outcome

    def _gate(self, result: RoundResult) -> GateReport:
        if result.edit is None or result.candidate_path is None:
            raise GateError("the round has no candidate to test")
        try:
            before = current_scores(self.deps, result.base_policy_path, self.ids)
        except Exception as exc:  # noqa: BLE001
            return self._untested(result, f"the policy in force could not be scored ({type(exc).__name__}: {exc})")
        inp = GateInput(round=result.round, edit_id=result.edit.id, current_policy=str(result.base_policy_path),
                        candidate_policy=str(result.candidate_path), validation_scenario_ids=self.ids,
                        current_scores=before, noise=self.noise, token_cap=self.token_cap,
                        safety_case_set=self.safety_case_set)
        report = run_gate(inp, self.deps)
        result.scores, result.safety = report.log_scores, report.log_safety
        if report.outcome.decision == "kept" and report.log_scores is not None:
            try:
                _write_current(self.deps.root, report.outcome.policy_in_force, self.ids, self.deps.judge.version,
                               report.log_scores.after)
            except Exception:  # noqa: BLE001  a stale current.json names the old version, so it is scored again
                pass
        return report

    def _untested(self, result: RoundResult, why: str) -> GateReport:
        """Reverted without testing the candidate, usually because the policy in force could not be scored."""
        checks = {k: "not_run" for k in CHECKS}
        checks["complete_evidence"] = "failed"
        checks["noise_not_measured"] = "passed" if self.deps.noise_measured else "failed"
        failed = [k for k in CHECKS if checks[k] == "failed"]
        reason = f"Reverted on {CHECKS['complete_evidence']}: {why}. The candidate was not tested."
        if not self.deps.noise_measured:
            reason += f" Also {CHECKS['noise_not_measured']}: {NOISE_NOTE}."
        edit_id = result.edit.id if result.edit is not None else f"edit-r{result.round:02d}"
        outcome = GateOutcome(
            id=f"gate-r{result.round:02d}", written_by="gatekeeper", created_at=now(), parent_ids=[edit_id],
            round=result.round, edit_id=edit_id, decision="reverted", before=NOT_MEASURED, after=NOT_MEASURED,
            safety_passed=False, policy_in_force=_version(str(result.base_policy_path)), reason=reason)
        report = GateReport(outcome=outcome, failed=failed, would_keep=False, checks=checks)
        try:
            _save_report(self.deps, result.round, report)
        except Exception:  # noqa: BLE001
            pass
        return report


def make_gate(deps: GateDeps, *, validation_scenario_ids: list[str], noise: float,
              token_cap: int = TOKEN_CAP, safety_case_set: str = "safety-v1") -> Callable[[RoundResult], GateOutcome]:
    """The gate for `RoundDeps.gate`. It leaves its scores and safety results on the RoundResult."""
    return _Gate(deps, validation_scenario_ids, noise, token_cap, safety_case_set)


# ---------- the starting policy's .meta.json ----------

def ensure_starting_meta(root: Path, start_policy: Path) -> Path:
    """Write <root>/policies/<version>.meta.json for the starting policy if it is not there.

    No parent, every Conversation rule a base rule, no learned rules. An existing file is left alone.
    """
    text = Path(start_policy).read_text()
    start = parse_policy(text)
    path = replay.meta_path(Path(root), start.version)
    if path.exists():
        return path
    # With no rounds, expected_meta gives every rule as a base rule and the Safety hash replay checks.
    meta = replay.expected_meta([], start, start, text).model_copy(update={"parent": None})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(meta.model_dump_json(indent=2) + "\n")
    return path
