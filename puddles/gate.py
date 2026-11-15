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
from loop.runner import conversation_id, run_batch
from puddles import replay
from puddles.graph import check_policy_file
from puddles.memory import now
from puddles.policy import TOKEN_CAP, estimate_tokens, parse_policy, render_policy
from puddles.records import sha256_bytes, sha256_text
from puddles.round import RoundResult, mark_gated, read_policy_text
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
STATIC_FOUR = ("candidate_file", *STATIC)
EIGHT = tuple(k for k in CHECKS if k != "noise_not_measured")
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
    would_keep: bool  # true when none of the eight checks before noise_not_measured failed
    checks: dict[str, str]  # every key of CHECKS -> "passed", "failed" or "not_run"
    log_scores: Optional[LogScores] = None  # None when validation gave no complete scores
    log_safety: Optional[LogSafety] = None  # None when the safety check did not run
    safety: Optional[SafetyResult] = None
    candidate_sha256: Optional[str] = None  # the hash the checks were made on; None when it could not be read


class _Changed(GateError):
    """A policy file's text differs from the text the checks were made on."""


def _read_text(path: Path | str) -> str:
    """A policy file's text exactly as `loop.policy.load_policy` reads it, so the hashes compare."""
    return Path(path).read_bytes().decode("utf-8")


def _sha256_file(path: Path | str) -> str:
    """The hash of a file's bytes: for valid UTF-8, the same as sha256_text of `_read_text`."""
    return sha256_bytes(Path(path).read_bytes())


def _load_tied(path: Path | str, what: str, expected: Optional[str] = None) -> Policy:
    """Load a policy, hashing the file before and after the load.

    The two hashes and the hash of the loaded text must agree, and equal `expected` when given:
    otherwise the file changed under the gate, and the evidence would be about other bytes.
    """
    before = _sha256_file(path)
    first = expected if expected is not None else before
    if before != first:  # before loading: a file that no longer decodes is a change, not a crash
        raise _Changed(f"the {what} changed while it was being tested")
    policy = check_policy_file(Path(path))
    after = _sha256_file(path)
    if not before == sha256_text(policy.text) == after == first:
        raise _Changed(f"the {what} changed while it was being tested")
    return policy


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

    Both files are saved, the scores even when some or all are missing. Scores are averaged only
    when every conversation finished and was scored: averaging the survivors can look like a gain.
    """
    return _score(deps, check_policy_file(Path(policy_path)), scenario_ids, round_, tag)


def _score(deps: GateDeps, policy: Policy, scenario_ids: list[str], round_: int, tag: str) -> Evidence:
    """score_policy on a policy already loaded, so the caller can tie the conversations to the
    exact text its checks were made on."""
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


# ---------- the scores of the policy in force ----------

def current_path(root: Path) -> Path:
    return Path(root) / "scores" / "current.json"


def scenarios_sha256(deps: GateDeps, scenario_ids: list[str]) -> str:
    """The hash of the picked scenarios, so a scenario whose text changed under its ID is scored again."""
    return sha256_text("\n".join(s.model_dump_json() for s in _pick(deps.scenarios, scenario_ids)))


def _cache_key(deps: GateDeps, policy_version: str, policy_sha256: str, scenario_ids: list[str]) -> dict[str, Any]:
    """Everything the kept scores depend on that the gate can see. The chat models are not in it:
    a root is for one pair of models."""
    return {"policy_version": policy_version, "policy_sha256": policy_sha256, "scenario_ids": list(scenario_ids),
            "judge_version": deps.judge.version, "scenarios_sha256": scenarios_sha256(deps, scenario_ids),
            "max_turns": deps.max_turns, "seed": deps.seed}


def _write_current(root: Path, key: dict[str, Any], scores: Scores) -> None:
    path = current_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**key, "scores": scores.model_dump()}, indent=2) + "\n", encoding="utf-8",
                    newline="")  # no newline translation on any platform


def _saved_current(root: Path, key: dict[str, Any]) -> Optional[Scores]:
    """The kept scores, if every field of the key matches. A file that cannot be read, or lacks a
    field, counts as absent, so the policy is scored again."""
    path = current_path(root)
    if not path.exists():
        return None
    try:
        # JSON, not a policy: nothing is hashed from these bytes and line endings carry no meaning,
        # so any are accepted. A file that does not decode or parse counts as absent.
        data = json.loads(path.read_bytes().decode("utf-8"))
        if not isinstance(data, dict) or "scores" not in data or any(name not in data for name in key):
            return None
        if any(data[name] != value for name, value in key.items()):
            return None
        return Scores.model_validate(data["scores"])
    except Exception:  # noqa: BLE001
        return None


def current_scores(deps: GateDeps, policy_path: Path, scenario_ids: list[str]) -> Scores:
    """The scores of the policy in force on the validation scenarios, kept in scores/current.json.

    Scored as round 0 with the tag `current` unless the file matches every field of the key.
    Incomplete evidence raises GateError and leaves any existing file as it is.
    """
    return _current_scores(deps, policy_path, scenario_ids)[0]


def _current_scores(deps: GateDeps, policy_path: Path, scenario_ids: list[str]) -> tuple[Scores, str]:
    """current_scores, with the hash of the text the scores are about.

    The policy is loaded once and that one reading is both scored and hashed into the key, so the
    hash in current.json is always of the text that was actually scored.
    """
    policy = _load_tied(policy_path, "policy in force")
    sha = sha256_text(policy.text)
    key = _cache_key(deps, policy.version, sha, scenario_ids)
    saved = _saved_current(deps.root, key)
    if saved is not None:
        return saved, sha
    evidence = _score(deps, policy, scenario_ids, BASELINE_ROUND, CURRENT_TAG)
    if not evidence.complete or evidence.scores is None:
        raise GateError(f"the policy in force could not be scored: {evidence.describe()}")
    _write_current(deps.root, key, evidence.scores)
    return evidence.scores, sha


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
    """The version of a policy file that may not load: what parse_policy reads, else the file's stem."""
    try:
        return parse_policy(Path(path).read_bytes().decode("utf-8")).version
    except Exception:  # noqa: BLE001
        return Path(path).stem


def _clashes(root: Path, round_: int, scenario_ids: list[str]) -> list[str]:
    """Validation conversation IDs that are also in this round's training file. No file, no clash."""
    train = Path(root) / "conversations" / f"round-{round_:02d}-train.jsonl"
    if not train.exists():
        return []
    lines = train.read_bytes().decode("utf-8").splitlines()  # undecodable: a crash, so a revert
    train_ids = {json.loads(line)["id"] for line in lines if line.strip()}
    return [cid for cid in (conversation_id(round_, sid, VALIDATION_TAG) for sid in scenario_ids) if cid in train_ids]


def _reason(failed: list[str], details: dict[str, str], would_keep: bool, verdict: str, extra: str) -> str:
    if not failed:
        return f"Kept: {verdict}."
    if would_keep and failed == ["noise_not_measured"]:
        return (f"Reverted on {CHECKS['noise_not_measured']}: {NOISE_NOTE}. "
                f"The six checks would have kept the candidate: {verdict}.")
    first, *rest = failed
    parts = [f"Reverted on {CHECKS[first]}: {details[first]}."]
    parts += [f"Also {CHECKS[name]}: {details[name]}." for name in rest]
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
    candidate_version: Optional[str] = None  # set only once every static check passed
    safety: Optional[SafetyResult] = None
    after: Optional[Scores] = None  # set only with complete evidence
    extra: str = ""
    candidate_sha256: Optional[str] = None  # the candidate text every check and measurement is about
    current_sha256: Optional[str] = None  # the same for the policy in force
    expected_current: Optional[str] = None  # the hash of the text the baseline scores are about, if known

    def fail(self, name: str, detail: str) -> None:
        self.checks[name], self.details[name] = "failed", detail

    def report(self, verdict: str = "") -> GateReport:
        inp, deps = self.inp, self.deps
        noise_measured = bool(deps.noise_measured)
        if any(self.checks[k] == "failed" for k in STATIC_FOUR):
            self.checks["noise_not_measured"] = "not_run"  # nothing after a failed static check runs
            self.details.pop("noise_not_measured", None)
        elif noise_measured:
            self.checks["noise_not_measured"] = "passed"
        else:
            self.fail("noise_not_measured", NOISE_NOTE)
        would_keep = not any(self.checks[k] == "failed" for k in EIGHT)
        failed = [k for k in CHECKS if self.checks[k] == "failed"]
        kept = would_keep and noise_measured and not failed and self.candidate_version is not None \
            and self.after is not None
        before = with_quality(inp.current_scores)
        safety = self.safety
        if kept and self.candidate_version is not None:
            in_force = self.candidate_version.removesuffix("-candidate")
        else:
            in_force = self.current_version or _version(inp.current_policy)
        extra = self.extra
        if not noise_measured and self.checks["noise_not_measured"] == "not_run":
            extra = f"{extra} The noise has not been measured either, so nothing would have been promoted.".strip()
        outcome = GateOutcome(
            id=f"gate-r{inp.round:02d}", written_by="gatekeeper", created_at=now(), parent_ids=[inp.edit_id],
            round=inp.round, edit_id=inp.edit_id, decision="kept" if kept else "reverted", before=before,
            after=self.after if self.after is not None else NOT_MEASURED,
            safety_passed=bool(safety is not None and safety.passed), policy_in_force=in_force,
            reason=_reason(failed, self.details, would_keep, verdict, extra))
        log_safety = None
        if safety is not None:
            log_safety = LogSafety(case_file=safety.case_file, passed=safety.passed, cases=len(safety.cases),
                                   failed_case_ids=[c.case_id for c in safety.cases if not c.passed])
        log_scores = LogScores(before=before, after=self.after, noise=inp.noise) if self.after is not None else None
        return GateReport(outcome=outcome, failed=failed, would_keep=would_keep, checks=dict(self.checks),
                          log_scores=log_scores, log_safety=log_safety, safety=safety,
                          candidate_sha256=self.candidate_sha256)

    def changed(self, detail: str) -> GateReport:
        """The revert for a policy file whose text moved under the gate: nothing it measured counts."""
        self.after = None
        self.candidate_version = None
        for name in ("quality_gain", "dimension_drop"):
            self.checks[name] = "not_run"
            self.details.pop(name, None)
        self.fail("complete_evidence", detail)
        self.extra = ""
        return self.report()


def _scores_dir(deps: GateDeps) -> Path:
    path = Path(deps.root) / "scores"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _static(run: _Run) -> Optional[tuple[Policy, Policy]]:
    """Step 1, no model call. Returns the two policies when all four checks passed."""
    inp = run.inp
    run.running = "candidate_file"
    try:
        candidate = _load_tied(inp.candidate_policy, "candidate")
        run.candidate_sha256 = sha256_text(candidate.text)
        current = _load_tied(inp.current_policy, "policy in force", expected=run.expected_current)
        run.current_version = current.version
        run.current_sha256 = sha256_text(current.text)
    except _Changed:
        raise  # not a file that does not load: a file that moved, which run_gate reverts
    except Exception as exc:  # noqa: BLE001  a file that does not load is this check failing
        run.fail("candidate_file", f"{type(exc).__name__}: {exc}")
        return None
    run.checks["candidate_file"] = "passed"
    # All three are evaluated, so the reason names every one that failed.
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
    run.running = None
    if any(run.checks[k] == "failed" for k in STATIC):
        return None
    return current, candidate


def _safety(run: _Run, candidate: Policy) -> None:
    """Step 2: the helper alone under the candidate, with no crisis checker in front of it.
    A safety check that raises fails only its own check and leaves no SafetyResult."""
    inp, deps = run.inp, run.deps
    session = f"safety-r{inp.round:02d}"

    def reply(turns: list[Turn]) -> str:
        return helper_reply(deps.helper_client, candidate, turns, session_id=session, turn=1).text

    run.running = "safety_cases"
    try:
        safety = deps.safety(candidate.version, reply)
        if not isinstance(safety, SafetyResult):
            raise GateError(f"the safety check returned {type(safety).__name__}, not a SafetyResult")
    except Exception as exc:  # noqa: BLE001
        run.fail("safety_cases", f"the safety check raised {type(exc).__name__}: {exc}")
        run.running = None
        return
    run.safety = safety
    if safety.passed:
        run.checks["safety_cases"] = "passed"
    else:
        failed_ids = [c.case_id for c in safety.cases if not c.passed]
        which = f" ({', '.join(failed_ids)})" if failed_ids else ""
        run.fail("safety_cases", f"{len(failed_ids)} of {len(safety.cases)} safety cases failed{which}")
    run.running = "saving the safety result"  # a failed write goes to run_gate's except: a revert
    (_scores_dir(deps) / f"round-{inp.round:02d}-safety.json").write_text(
        safety.model_dump_json(indent=2) + "\n", encoding="utf-8")
    run.running = None


def _recheck(run: _Run) -> None:
    """Both files must still hold the text the checks were made on."""
    for path, sha, what in ((run.inp.candidate_policy, run.candidate_sha256, "candidate"),
                            (run.inp.current_policy, run.current_sha256, "policy in force")):
        if sha is None or _sha256_file(path) != sha:
            raise _Changed(f"the {what} changed while it was being tested")


def _evaluate(run: _Run) -> GateReport:
    inp, deps = run.inp, run.deps
    policies = _static(run)
    if policies is None:
        run.extra = NOTHING_ELSE_RAN
        return run.report()
    _, candidate = policies
    run.candidate_version = candidate.version

    _safety(run, candidate)

    # 3. Validation. It runs whatever the safety check said.
    run.running = "complete_evidence"
    ids = list(inp.validation_scenario_ids)
    # score_policy's steps, on a load tied to the hash the static checks were made on.
    policy = _load_tied(inp.candidate_policy, "candidate", expected=run.candidate_sha256)
    evidence = _score(deps, policy, ids, inp.round, VALIDATION_TAG)
    _recheck(run)  # after the last judge call, before deciding
    clash = _clashes(deps.root, inp.round, ids)
    problems = []
    if not evidence.complete or evidence.scores is None:
        problems.append(f"of the validation conversations, {evidence.describe()}")
    if clash:  # the files stay where they are; the validation counts as not measured
        problems.append(f"validation conversation IDs are also in round-{inp.round:02d}-train.jsonl "
                        f"({', '.join(clash)}), so they cannot be told apart from the round's own")
    if problems:
        run.fail("complete_evidence", "; ".join(problems))
        run.running = None
        return run.report()
    assert evidence.scores is not None
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
    verdict = f"{moved}, above the noise of {_num(inp.noise)}; no dimension fell by more than the noise"
    if run.safety is not None:
        verdict += f"; the safety check passed ({len(run.safety.cases)} cases)"
    return run.report(verdict)


def _save_report(deps: GateDeps, round_: int, report: GateReport) -> None:
    data = {"round": round_, "decision": report.outcome.decision, "would_keep": report.would_keep,
            "failed": report.failed, "checks": report.checks, "reason": report.outcome.reason}
    (_scores_dir(deps) / f"round-{round_:02d}-gate.json").write_text(json.dumps(data, indent=2) + "\n",
                                                                     encoding="utf-8")


def _crashed(run: _Run, exc: BaseException) -> GateReport:
    """The revert for a crash, including a file that could not be saved: the evidence is incomplete."""
    if isinstance(exc, _Changed):
        return run.changed(f"{exc}, so the evidence is not about the text that was checked")
    if run.running in STATIC_FOUR and run.checks[run.running] == "not_run":
        run.fail(run.running, f"it could not finish ({type(exc).__name__}: {exc})")
    run.after = None  # nothing measured is shown after a crash
    run.candidate_version = None  # and nothing is kept
    for name in ("quality_gain", "dimension_drop"):  # without complete evidence they did not run
        run.checks[name] = "not_run"
        run.details.pop(name, None)
    run.fail("complete_evidence", f"the test run crashed ({type(exc).__name__}: {exc})")
    run.extra = ""
    return run.report()


def _bare_revert(round_: int, edit_id: str, current_policy: str, failed: list[str], reason: str) -> GateReport:
    """A revert with nothing measured, for when the gate could not test the candidate at all."""
    checks = {k: "failed" if k in failed else "not_run" for k in CHECKS}
    outcome = GateOutcome(
        id=f"gate-r{round_:02d}", written_by="gatekeeper", created_at=now(), parent_ids=[edit_id],
        round=round_, edit_id=edit_id, decision="reverted", before=NOT_MEASURED, after=NOT_MEASURED,
        safety_passed=False, policy_in_force=_version(current_policy), reason=reason)
    return GateReport(outcome=outcome, failed=[k for k in CHECKS if k in failed], would_keep=False, checks=checks)


def run_gate(inp: GateInput, deps: GateDeps) -> GateReport:
    """Apply the checks in order. It never raises: anything that goes wrong reverts the candidate,
    including a file that could not be saved."""
    mark_gated(deps.root, inp.round)  # first act, as in make_gate's gate: the round is the real gate's
    return _run_gate(inp, deps)[0]


def _run_gate(inp: GateInput, deps: GateDeps, current_sha256: Optional[str] = None) -> tuple[GateReport, _Run]:
    """run_gate, also returning its state. `current_sha256` is the hash of the policy text the
    baseline scores were measured on; the policy in force must still hold that text."""
    run = _Run(inp=inp, deps=deps, expected_current=current_sha256)
    report = _gate_report(run)
    return report, run


def _gate_report(run: _Run) -> GateReport:
    inp, deps = run.inp, run.deps
    try:
        report = _evaluate(run)
        run.running = "saving the gate report"
        _save_report(deps, inp.round, report)
        return report
    except Exception as exc:  # noqa: BLE001  any crash at all must end as a revert, never as a keep
        try:
            report = _crashed(run, exc)
        except Exception as again:  # noqa: BLE001  even the crash report failed: a bare revert
            report = _bare_revert(inp.round, inp.edit_id, inp.current_policy, ["complete_evidence"],
                                  f"Reverted on {CHECKS['complete_evidence']}: the test run crashed "
                                  f"({type(again).__name__}: {again}).")
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
        mark_gated(self.deps.root, result.round)  # first act: the round is the real gate's, whatever fails later
        result.scores, result.safety, result.candidate_sha256, result.base_sha256 = None, None, None, None
        try:
            report = self._gate(result)
        except Exception as exc:  # noqa: BLE001  the candidate is reverted untested
            result.scores, result.safety, result.candidate_sha256, result.base_sha256 = None, None, None, None
            report = self._untested(result, f"the gate could not run ({type(exc).__name__}: {exc})")
        self.last_report = report
        return report.outcome

    def _gate(self, result: RoundResult) -> GateReport:
        if result.edit is None or result.candidate_path is None:
            raise GateError("the round has no candidate to test")
        try:
            before, base_sha = _current_scores(self.deps, result.base_policy_path, self.ids)
        except Exception as exc:  # noqa: BLE001
            return self._untested(result, f"the policy in force could not be scored ({type(exc).__name__}: {exc})")
        result.base_sha256 = base_sha  # the text the before-scores are about, kept or reverted
        inp = GateInput(round=result.round, edit_id=result.edit.id, current_policy=str(result.base_policy_path),
                        candidate_policy=str(result.candidate_path), validation_scenario_ids=self.ids,
                        current_scores=before, noise=self.noise, token_cap=self.token_cap,
                        safety_case_set=self.safety_case_set)
        # The baseline was measured on one text of the policy in force; the gate compares against that text.
        report, run = _run_gate(inp, self.deps, current_sha256=base_sha)
        if report.outcome.decision == "kept" and report.log_scores is not None:
            # Read the candidate once more; the promoted scores belong only to the bytes that were tested.
            try:
                text: Optional[str] = _read_text(result.candidate_path)
            except Exception:  # noqa: BLE001  a candidate that cannot be read again cannot be promoted
                text = None
            if text is None or sha256_text(text) != report.candidate_sha256:
                report = run.changed("the candidate changed after it was tested, so the evidence is not about "
                                     "the text that would be promoted")
                try:
                    _save_report(self.deps, result.round, report)
                except Exception:  # noqa: BLE001  the decision is already a revert
                    pass
            else:
                try:
                    self._keep_current(text, report.outcome.policy_in_force, report.log_scores.after)
                except Exception:  # noqa: BLE001  a stale or missing file cannot match, so the next round scores again
                    pass
        result.scores, result.safety = report.log_scores, report.log_safety
        result.candidate_sha256 = report.candidate_sha256
        return report

    def _keep_current(self, candidate_text: str, version: str, scores: Scores) -> None:
        """current.json for the promoted policy, hashed on the text finish_round will write for it."""
        promoted = parse_policy(candidate_text)
        promoted.version = version
        key = _cache_key(self.deps, version, sha256_text(render_policy(promoted)), self.ids)
        _write_current(self.deps.root, key, scores)

    def _untested(self, result: RoundResult, why: str) -> GateReport:
        """Reverted without testing the candidate, usually because the policy in force could not be scored."""
        failed = ["complete_evidence"] + ([] if self.deps.noise_measured else ["noise_not_measured"])
        reason = f"Reverted on {CHECKS['complete_evidence']}: {why}. The candidate was not tested."
        if not self.deps.noise_measured:
            reason += f" Also {CHECKS['noise_not_measured']}: {NOISE_NOTE}."
        edit_id = str(getattr(result.edit, "id", None) or f"edit-r{result.round:02d}")
        report = _bare_revert(result.round, edit_id, str(result.base_policy_path), failed, reason)
        if self.deps.noise_measured:
            report.checks["noise_not_measured"] = "passed"
        try:
            _save_report(self.deps, result.round, report)
        except Exception:  # noqa: BLE001
            pass
        return report


def make_gate(deps: GateDeps, *, validation_scenario_ids: list[str], noise: float,
              token_cap: int = TOKEN_CAP, safety_case_set: str = "safety-v1") -> Callable[[RoundResult], GateOutcome]:
    """The gate for `RoundDeps.gate`. It leaves its scores and safety results on the RoundResult.

    `safety_case_set` is only recorded on the GateInput: the safety check passed in decides its cases.
    """
    return _Gate(deps, validation_scenario_ids, noise, token_cap, safety_case_set)


# ---------- the starting policy's .meta.json ----------

def ensure_starting_meta(root: Path, start_policy: Path) -> Path:
    """Write <root>/policies/<version>.meta.json for the starting policy if it is not there.

    No parent, every Conversation rule a base rule, no learned rules. An existing file is left alone.
    """
    text = read_policy_text(Path(start_policy))  # its bytes, so policy_sha256 is the hash of the file
    start = parse_policy(text)
    path = replay.meta_path(Path(root), start.version)
    if path.exists():
        return path
    # With no rounds, expected_meta gives every rule as a base rule and the Safety hash replay checks.
    meta = replay.expected_meta([], start, start, text).model_copy(update={"parent": None})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="")
    return path
