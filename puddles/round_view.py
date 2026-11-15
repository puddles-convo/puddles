"""The round view: a summary on each node's span, numbers on the round's run, and a report per round.

The contract is docs/specs/round-view.md. Everything here only reads what a round left behind: it
never writes to the decision log, a record, a policy, a conversations or a scores file, and never
calls a model, Exa or past.dev. No turn of any conversation is copied into a summary or a report.

A killed attempt at the same round number can leave files behind, and a crash can cut a line in
half. Nothing read here may stop a summary or a report: a file or line that cannot be read, or that
belongs to an earlier attempt, is left out and the report says so.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import mlflow

from puddles import replay
from puddles.decision_log import DecisionLog
from puddles.judging import CRITIC_CONVERSATIONS, scored_conversations, select_for_critic
from puddles.memory import READS_PATH
from puddles.policy import parse_policy
from puddles.records import load_record
from puddles.round import memory_trace
from puddles.usage import USAGE_PATH
from schemas import (
    ConversationRecord,
    FailurePattern,
    GateOutcome,
    MemoryCopyEvent,
    PolicyEdit,
    ResearchNote,
    RoundEvent,
    ScoreRecord,
)

SUMMARY_KEY = "puddles.summary"
NODES = ("converse", "judge", "critic", "researcher", "proposer", "gate", "finish")
CHECK_WORDS = {"passed": "passed", "failed": "failed", "not_run": "not run"}
STRAY_SCORES = "{n} score(s) in the scores file belong to no conversation of this round."
UNREADABLE = "unreadable"  # shown in place of a value that is not of its kind; the value itself never is


class ReportError(RuntimeError):
    """The root has no log, the log cannot be read, or the round is not in it."""


# ---------- where a round's files are ----------

def _rel(path: Path, root: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def conversations_path(root: Path, round_: int) -> Path:
    return Path(root) / "conversations" / f"round-{round_:02d}-train.jsonl"


def failures_file(root: Path, round_: int) -> Path:
    return conversations_path(root, round_).with_suffix(".failures.jsonl")


def scores_path(root: Path, round_: int) -> Path:
    return Path(root) / "scores" / f"round-{round_:02d}-train.jsonl"


def gate_report_path(root: Path, round_: int) -> Path:
    return Path(root) / "scores" / f"round-{round_:02d}-gate.json"


def report_path(root: Path, round_: int) -> Path:
    return Path(root) / "reports" / f"round-{round_:02d}.md"


# ---------- reading what a round left, without ever raising ----------

def _why(exc: BaseException) -> str:
    """Only the kind of error: its message can quote the input it failed on, which may be a spoken line."""
    return type(exc).__name__


def _lines(path: Path, root: Path, problems: list[str]) -> list[dict]:
    """The whole JSON objects of a JSONL file. Unreadable lines are counted in `problems`, not raised."""
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"{_rel(path, root)} could not be read ({_why(exc)}).")
        return []
    out, bad = [], 0
    for line in raw:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except (ValueError, RecursionError):
            bad += 1
            continue
        if isinstance(data, dict):
            out.append(data)
        else:
            bad += 1
    if bad:
        problems.append(f"{bad} line(s) of {_rel(path, root)} could not be read and were left out.")
    return out


def is_int(x: Any) -> bool:
    """A whole number as JSON gives one; a bool is not a number here."""
    return isinstance(x, int) and not isinstance(x, bool)


def is_number(x: Any) -> bool:
    """A finite number as JSON gives one; not a bool, not NaN, not infinity."""
    return (is_int(x) or isinstance(x, float)) and math.isfinite(x)


def finite(x: Any) -> Optional[float]:
    """The value if it is a finite number, else None: a value that cannot be read is never shown."""
    return x if is_number(x) else None


def _score_line(data: dict) -> bool:
    """What the view takes from a score line, checked as JSON gave it, before any coercion."""
    return (isinstance(data.get("conversation_id"), str) and isinstance(data.get("judge_version"), str)
            and all(is_int(data.get(k)) and 1 <= data[k] <= 5 for k in ("empathy", "relevance")))


def _models(path: Path, root: Path, model: type, problems: list[str],
            check: Callable[[dict], bool] = lambda data: True) -> list:
    out, bad = [], 0
    for data in _lines(path, root, problems):
        try:
            if not check(data):
                raise ValueError("a field the view shows is not of its kind")
            out.append(model.model_validate(data))
        except Exception:  # noqa: BLE001  a line that is not a whole record is left out
            bad += 1
    if bad:
        problems.append(f"{bad} line(s) of {_rel(path, root)} are not valid {model.__name__} lines and were left out.")
    return out


STATUSES = ("passed", "failed", "not_run")
DECISIONS = ("kept", "reverted")


def _gate_checks() -> tuple[str, ...]:
    from puddles.gate import CHECKS  # puddles.gate imports puddles.graph, which imports this module

    return tuple(CHECKS)


def _clean_gate_report(data: Any, round_: int, gate: Optional[GateOutcome], name: str,
                       where: list[str]) -> Optional[dict]:
    """The gate report as the view uses it, every value checked for what it is.

    Returns {"checks": {check: status or "unreadable"}, "unknown": count, "readable": bool,
    "failed": list or None}, or None when the file is not this round's gate report. A check value
    that is not passed, failed or not_run, and a check whose name is not a gate check, are never shown.
    """
    if not isinstance(data, dict):
        where.append(f"{name} is not a gate report; its checks are not shown.")
        return None
    rnd, decision, checks = data.get("round", round_), data.get("decision"), data.get("checks", {})
    if not is_int(rnd) or (decision is not None and decision not in DECISIONS) or not isinstance(checks, dict):
        where.append(f"{name} is not a gate report (its round, decision or checks are unreadable); "
                     "its checks are not shown.")
        return None
    if rnd != round_ or (gate is not None and decision is not None and decision != gate.decision):
        where.append(f"{name} does not match this round's gate decision; it was left by an earlier attempt "
                     "and its checks are not shown.")
        return None
    known = _gate_checks()
    clean: dict[str, str] = {}
    unknown = bad = 0
    for check, status in checks.items():
        if check not in known:
            unknown += 1
        elif status in STATUSES:
            clean[check] = status
        else:
            clean[check], bad = "unreadable", bad + 1
    if bad:
        where.append(f"{bad} check value(s) in {name} are not passed, failed or not_run; they are shown as "
                     "unreadable and the round's failed checks are not worked out.")
    if unknown:
        where.append(f"{unknown} check(s) in {name} have a name that is not a gate check; they are unreadable "
                     "and not shown.")
    failed = data.get("failed")
    failed_names = list(failed) if isinstance(failed, list) and all(f in known for f in failed) else None
    return {"checks": clean, "unknown": unknown, "readable": not bad and not unknown, "failed": failed_names}


@dataclass
class RoundFiles:
    """What one round left on disk, as far as it could be read; `problems` says what could not."""

    records: list[ConversationRecord] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    scores: Optional[list[ScoreRecord]] = None  # this round's scores; None when there are none saved
    stray_scores: int = 0  # lines of the scores file that belong to no conversation of this round
    gate_report: Optional[dict] = None  # None: no usable gate report (none, unreadable or not this round's)
    gate_report_exists: bool = False
    problems: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def read_round_files(root: Path, round_: int, records: Optional[list[ConversationRecord]] = None,
                     gate: Optional[GateOutcome] = None) -> RoundFiles:
    """Read the round's conversations, failures, scores and gate report. Never raises.

    `records`, when given, are the round's conversations as the graph holds them; otherwise they
    are read from the conversations file. `gate`, when given, is the round's gate outcome, used to
    refuse a gate report that an earlier attempt left.
    """
    root = Path(root)
    files = RoundFiles()
    conv = files.problems["Conversations"]
    files.records = list(records) if records is not None else _models(
        conversations_path(root, round_), root, ConversationRecord, conv,
        lambda data: isinstance(data.get("id"), str))
    unreadable_failures = 0
    for data in _lines(failures_file(root, round_), root, conv):
        scenario, reason = data.get("scenario_id"), data.get("reason")
        if not isinstance(scenario, str) or not isinstance(reason, str):
            unreadable_failures += 1
        files.failures.append({"scenario_id": scenario if isinstance(scenario, str) else UNREADABLE,
                               "reason": reason if isinstance(reason, str) else UNREADABLE})
    if unreadable_failures:
        conv.append(f"{unreadable_failures} line(s) of {_rel(failures_file(root, round_), root)} have a scenario "
                    "or a reason that is not text; it is shown as unreadable.")

    path = scores_path(root, round_)
    if path.exists():
        ids = {r.id for r in files.records}
        current = []
        for s in _models(path, root, ScoreRecord, conv, _score_line):
            if s.conversation_id in ids:  # membership is by conversation ID in this round's file, nothing else
                current.append(s)
            else:
                files.stray_scores += 1
        # The real judge rewrites the file every round, so a file holding only stray lines was left by an
        # earlier attempt and this round's judge saved none (the stub judge).
        files.scores = None if not current and files.stray_scores else current

    gate_path = gate_report_path(root, round_)
    files.gate_report_exists = gate_path.exists()
    if files.gate_report_exists:
        where = files.problems["Gate"]
        name = _rel(gate_path, root)
        try:
            data = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            where.append(f"{name} could not be read ({_why(exc)}); its checks are not shown.")
        else:
            files.gate_report = _clean_gate_report(data, round_, gate, name, where)
    return files


def _failed_checks(report: Optional[dict]) -> Optional[list[str]]:
    """The names of the failed checks, or None when there is no gate report or part of it is unreadable."""
    if report is None or not report["readable"]:
        return None
    if report["checks"]:
        return [name for name, status in report["checks"].items() if status == "failed"]
    return report["failed"]


def noise_measured(report: Optional[dict]) -> bool:
    """The noise is a measurement only when a readable gate report marks noise_not_measured as passed."""
    return report is not None and report["readable"] and report["checks"].get("noise_not_measured") == "passed"


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


# ---------- 1. a summary on each node's span ----------

def node_summary(node: str, state: Mapping[str, Any], update: Mapping[str, Any], root: Path) -> dict:
    """What the node did, in a few numbers and names, without any conversation text. Never raises:
    a value that cannot be worked out is null."""
    try:
        return _node_summary(node, state, update, Path(root))
    except Exception as exc:  # noqa: BLE001  the view never breaks a round
        return {"error": _why(exc)}


def _node_summary(node: str, state: Mapping[str, Any], update: Mapping[str, Any], root: Path) -> dict:
    round_ = int(state["round"])
    if node == "converse":
        return {"finished": len(update.get("records", [])), "failed": len(update.get("failures", [])),
                "file": _rel(conversations_path(root, round_), root)}
    if node == "judge":
        records = state.get("records")
        files = read_round_files(root, round_, records=list(records) if records is not None else None)
        if files.scores:
            scored: list = files.scores
            version: Optional[str] = files.scores[0].judge_version
        else:  # no scores saved this round: what the node returned is all that was scored
            scored, version = list(update.get("scored", [])), None
        return {"scored": len(scored), "unscored": len(update.get("unscored", [])),
                "mean_empathy": _mean([s.empathy for s in scored]),
                "mean_relevance": _mean([s.relevance for s in scored]), "judge_version": version}
    result = update.get("result", state.get("result"))
    if node == "critic":
        if result is not None and result.pattern is not None:
            p = result.pattern
            return {"pattern": p.name, "seen_in": p.seen_in, "out_of": p.out_of}
        return {"pattern": None, "reason": result.reason if result is not None else None}
    if node == "researcher":
        research = result.research if result is not None else None
        if research is None:  # the researcher failed: nothing was counted
            return {"queries": None, "sources": None, "reason": result.reason if result is not None else None}
        return {"queries": len(_queries(research)), "sources": [s.url for s in research.sources]}
    if node == "proposer":
        edit = result.edit if result is not None else None
        if edit is None:
            return {"edit": None, "reason": result.reason if result is not None else None}
        return {"action": edit.action, "rule_id": edit.rule_id, "rule_text": edit.rule_text}
    if node == "gate":
        outcome = update.get("gate")
        scores = result.scores if result is not None else None
        safety = result.safety if result is not None else None
        report = read_round_files(root, round_, records=[], gate=outcome).gate_report
        decision = outcome.decision if outcome is not None else None
        return {"decision": decision if decision in DECISIONS else None,
                "quality_before": finite(scores.before.quality) if scores is not None else None,
                "quality_after": finite(scores.after.quality) if scores is not None else None,
                "noise": finite(scores.noise) if scores is not None and noise_measured(report) else None,
                "safety_passed": safety.passed if safety is not None and isinstance(safety.passed, bool) else None,
                "failed_checks": _failed_checks(report)}
    if node == "finish":
        event = update.get("event")
        try:
            copies: Optional[list] = [{"record_id": e.record_id, "status": e.status} for e in DecisionLog(root).events()
                                      if isinstance(e, MemoryCopyEvent) and e.round == round_]
        except Exception:  # noqa: BLE001  a log that cannot be read gives no copies, not an empty list
            copies = None
        return {"outcome": event.outcome if event is not None else None,
                "policy_in_force": event.policy.in_force if event is not None else None, "copies": copies}
    return {}


def _queries(research: ResearchNote) -> list[str]:
    return list(research.queries) + ([research.query] if research.query and research.query not in
                                     research.queries else [])


def _put_summary(node: str, state: Mapping[str, Any], update: Mapping[str, Any], root: Path) -> None:
    summary = node_summary(node, state, update, root)
    span = mlflow.get_current_active_span()
    if span is None:
        return
    if span.name == node:
        span.set_attribute(SUMMARY_KEY, summary)
        return
    # Not the node's own span: a child span holds the summary instead, under the same key.
    with mlflow.start_span(name=f"summary:{node}") as child:
        child.set_attribute(SUMMARY_KEY, summary)


def with_summary(node: str, fn: Callable[[Any], dict], root: Path) -> Callable[[Any], dict]:
    """`fn`, with its summary set on the node's span on the way out. The view never breaks a round."""

    def wrapped(state: Any) -> dict:
        update = fn(state)  # an exception from the node passes through, with no summary
        try:
            _put_summary(node, state, update or {}, root)
        except Exception:  # noqa: BLE001  a summary that cannot be made or set is left out
            pass
        return update

    wrapped.__name__ = getattr(fn, "__name__", node)
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


# ---------- 2. numbers on the round's MLflow run ----------

def gate_metrics(state: Mapping[str, Any], root: Path) -> dict[str, float]:
    """`kept` whenever the gate decided; the quality numbers only when the gate measured them, and the
    noise only when the gate report says it was measured. Never raises."""
    try:
        outcome = state.get("gate")
        if outcome is None:
            return {}
        if outcome.decision not in DECISIONS:
            return {}
        out: dict[str, float] = {"kept": 1 if outcome.decision == "kept" else 0}
        result = state.get("result")
        scores = result.scores if result is not None else None
        if scores is not None:
            if finite(scores.before.quality) is not None:
                out["quality_before"] = scores.before.quality
            if finite(scores.after.quality) is not None:
                out["quality_after"] = scores.after.quality
            report = read_round_files(Path(root), int(state["round"]), records=[], gate=outcome).gate_report
            if noise_measured(report) and finite(scores.noise) is not None:
                out["noise"] = scores.noise
        return out
    except Exception:  # noqa: BLE001  the view never breaks a round
        return {}


# ---------- 3. the report ----------

def read_rounds(root: Path) -> list[RoundEvent]:
    """The round events in the root's log. ReportError when there is no log or it cannot be read."""
    log = DecisionLog(Path(root))
    if not log.path.exists():
        raise ReportError(f"{log.path} does not exist")
    try:
        return log.rounds()
    except Exception as exc:  # noqa: BLE001  any unreadable log is the same refusal
        raise ReportError(f"{log.path} cannot be read ({_why(exc)})") from exc


def _cell(value: Any) -> str:
    text = "" if value is None else " ".join(str(value).split())
    return text.replace("|", "\\|")


def _row(*cells: Any) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _num(x: Optional[float]) -> str:
    return "not measured" if x is None else f"{x:.2f}"


def _nothing_scored(reason: str) -> bool:
    from puddles.graph import NO_CONVERSATION_REASON, NOTHING_SCORED_REASON  # puddles.graph imports this module

    return reason.startswith(NO_CONVERSATION_REASON) or reason.startswith(NOTHING_SCORED_REASON)


def nodes_that_ran(event: RoundEvent) -> set[str]:
    """Worked out from the log alone, as the contract gives it."""
    ran = {"converse", "judge", "finish"}
    if not _nothing_scored(event.reason):
        ran.add("critic")
    if event.pattern is not None:
        ran.add("researcher")
    if "research" in event.records and event.outcome != "no_evidence":
        ran.add("proposer")
    if "outcome" in event.records:
        ran.add("gate")
    return ran


def _record(event: RoundEvent, kind: str, model: type, root: Path, problems: list[str]) -> Any:
    ref = event.records.get(kind)
    if ref is None:
        return None
    try:
        record = load_record(ref, model, root)
    except Exception as exc:  # noqa: BLE001  a half-written record file is reported, not raised
        problems.append(f"The {kind} record {ref.path} could not be read ({_why(exc)}).")
        return None
    if record is None:
        problems.append(f"The {kind} record {ref.path} is missing or no longer matches the log.")
    return record


def build_report(root: Path, round_: int) -> str:
    """The round's report as markdown. It reads the root's files and writes nothing.

    Only a missing or unreadable log, or a round not in it, raises (ReportError). Anything else the
    round left that cannot be read is left out, and the report says so in the section it belongs to.
    """
    root = Path(root)
    rounds = read_rounds(root)
    event = next((r for r in rounds if r.round == round_), None)
    if event is None:
        raise ReportError(f"round {round_} is not in {DecisionLog(root).path}")
    problems: dict[str, list[str]] = defaultdict(list)
    pattern = _record(event, "pattern", FailurePattern, root, problems["Nodes"])
    research = _record(event, "research", ResearchNote, root, problems["Nodes"])
    edit = _record(event, "edit", PolicyEdit, root, problems["Nodes"])
    outcome = _record(event, "outcome", GateOutcome, root, problems["Nodes"])
    files = read_round_files(root, round_, gate=outcome)
    for name, lines in files.problems.items():
        problems[name] += lines
    copies = [e for e in DecisionLog(root).events() if isinstance(e, MemoryCopyEvent) and e.round == round_]

    sections = [
        ("Nodes", _nodes_section(event, root, files, pattern, research, edit, copies)),
        ("Conversations", _conversations_section(files)),
        ("Memory", _memory_section(event, root, copies, problems["Memory"])),
        ("Policy change", _policy_section(event, rounds, root)),
        ("Gate", _gate_section(event, files)),
        ("Cost", _cost_section(root, round_, problems["Cost"])),
    ]
    out = [f"# Round {round_}: {event.outcome}", "", event.reason, ""]
    for name, body in sections:
        out += [f"## {name}", "", *body]
        if problems.get(name):
            out += [*problems[name], ""]
    return "\n".join(out).rstrip("\n") + "\n"


def _nodes_section(event: RoundEvent, root: Path, files: RoundFiles, pattern: Optional[FailurePattern],
                   research: Optional[ResearchNote], edit: Optional[PolicyEdit], copies: list) -> list[str]:
    round_ = event.round
    ran = nodes_that_ran(event)
    records, scores = files.records, files.scores
    fail_file = failures_file(root, round_)

    def where(kind: str) -> str:
        ref = event.records.get(kind)
        return f"{ref.path}, the decision log" if ref is not None else "the decision log"

    rows: dict[str, tuple[str, str, str]] = {}
    went = _rel(conversations_path(root, round_), root) + (f", {_rel(fail_file, root)}" if fail_file.exists() else "")
    rows["converse"] = (f"policy {event.policy.base}",
                        f"{len(records)} conversations finished, {len(files.failures)} failed", went)
    if scores is not None:
        unscored = len([r for r in records if r.id not in {s.conversation_id for s in scores}])
        rows["judge"] = (f"{len(records)} conversations", f"{len(scores)} scored, {unscored} unscored",
                         _rel(scores_path(root, round_), root))
    else:
        rows["judge"] = (f"{len(records)} conversations", "scores that were not saved (no scores file)",
                         "the critic")
    if pattern is not None:
        critic_out = f"pattern {pattern.name} (seen in {pattern.seen_in} of {pattern.out_of})"
    elif event.pattern is not None:
        critic_out = f"pattern {event.pattern.name} (seen in {event.pattern.seen_in} of {event.pattern.out_of})"
    else:
        critic_out = "no pattern"
    given = pattern.out_of if pattern is not None else _critic_count(scores)
    rows["critic"] = (f"{given} scored conversations" if given is not None else "the scored conversations",
                      critic_out, where("pattern"))
    rows["researcher"] = (f"pattern {event.pattern.name}" if event.pattern else "",
                          f"{len(research.sources)} sources from {len(_queries(research))} queries"
                          if research is not None else "no research note that could be read", where("research"))
    if edit is not None:
        proposer_out = f"{edit.action or 'add'} {edit.rule_id or ''}".strip()
        proposer_went = f"{where('edit')}, {edit.policy_path}"
    elif event.edit is not None:
        proposer_out, proposer_went = f"{event.edit.action} {event.edit.rule_id}", where("edit")
    else:
        proposer_out, proposer_went = "no edit", "the decision log"
    rows["proposer"] = ("the pattern and the research note", proposer_out, proposer_went)
    gate_went = where("outcome")
    if files.gate_report is not None:
        gate_went += f", {_rel(gate_report_path(root, round_), root)}"
    rows["gate"] = (f"candidate {event.policy.new}" if event.policy.new else "the candidate",
                    f"{event.outcome}", gate_went)
    finish_went = "the decision log, memory"
    if event.outcome == "kept" and event.policy.new:
        finish_went += f", policies/{event.policy.new}.md"
    rows["finish"] = (f"the round's result ({event.outcome})",
                      f"the round event, {len(copies)} memory copy events", finish_went)

    out = [_row("Node", "Received", "Produced", "Went to"), _row("---", "---", "---", "---")]
    for node in NODES:
        if node in ran:
            out.append(_row(node, *rows[node]))
        else:
            out.append(_row(node, "did not run", "", ""))
    return out + [""]


def _critic_count(scores: Optional[list]) -> Optional[int]:
    if not scores:
        return None
    return min(len(scores), CRITIC_CONVERSATIONS)


def _conversations_section(files: RoundFiles) -> list[str]:
    records, scores = files.records, files.scores
    out: list[str] = []
    by_id = {s.conversation_id: s for s in scores} if scores is not None else {}
    given: set[str] = set()
    if scores:
        given = {s.conversation_id for s in select_for_critic(scored_conversations(records, scores))}
    if scores is None:
        out += ["No scores file: the judge's scores were not saved, so the scores and the critic column are empty.",
                ""]
    out += [_row("ID", "Empathy", "Relevance", "Critic"), _row("---", "---", "---", "---")]
    for record in records:
        score = by_id.get(record.id)
        if scores is None:
            out.append(_row(record.id, "", "", ""))
        elif score is None:
            out.append(_row(record.id, "", "", "unscored"))
        else:
            out.append(_row(record.id, score.empathy, score.relevance, "yes" if record.id in given else "no"))
    for failure in files.failures:
        out.append(_row(f"{failure['scenario_id']} (failed)", "", "", f"failed: {failure['reason']}"))
    if not records and not files.failures:
        out.append(_row("none", "", "", ""))
    out.append("")
    if files.stray_scores:
        out += [STRAY_SCORES.format(n=files.stray_scores), ""]
    return out


def _reads_file_readable(root: Path) -> bool:
    """memory_trace trusts every line of the reads file; a reader that is not text, or record IDs that are
    a string rather than a list (it would match by substring), would be shown or matched wrongly."""
    path = root / READS_PATH
    if not path.exists():
        return True
    problems: list[str] = []
    lines = _lines(path, root, problems)
    if problems:
        return False
    return all(isinstance(r.get("reader"), str) and isinstance(r.get("record_ids"), list)
               and all(isinstance(i, str) for i in r["record_ids"]) for r in lines)


def _memory_section(event: RoundEvent, root: Path, copies: list, problems: list[str]) -> list[str]:
    out: list[str] = []
    try:
        if not _reads_file_readable(root):
            raise ValueError("a line of the reads file is not a reader's name and a list of record IDs")
        rows = [r for r in memory_trace(DecisionLog(root), root) if r["round"] == event.round]
    except Exception as exc:  # noqa: BLE001  a half-written reads file or record: say so, list what the log holds
        problems.append(f"Who read the round's records could not be worked out ({_why(exc)}).")
        rows = [{"record": ref.id, "written_by": "?", "read_by": None} for ref in event.records.values()]
    for row in rows:
        if row["read_by"] is None:
            readers = "unknown"
        else:
            readers = ", ".join(row["read_by"]) if row["read_by"] else "no agent yet"
        out.append(f"- {row['record']}: written by {row['written_by']}, read later by {readers}")
    if not rows:
        out.append("- The round wrote no record.")
    out.append("")
    for copy in copies:
        out.append(f"- memory copy of {copy.record_id}: {copy.status} (attempt {copy.attempt})")
    if not copies:
        out.append("- No memory copy event.")
    return out + [""]


def _earlier_text(rule_id: str, event: RoundEvent, rounds: list[RoundEvent], root: Path) -> Optional[str]:
    """The rule's text before this round: from the last kept edit of it in the log, else the base policy file."""
    text: Optional[str] = None
    found = False
    for r in rounds:
        if r.round >= event.round:
            break
        if r.outcome == "kept" and r.edit is not None and r.edit.rule_id == rule_id:
            text, found = r.edit.rule_text, True
    if found:
        return text
    path = replay.policy_path(root, event.policy.base)
    try:
        return parse_policy(path.read_text()).conversation.get(rule_id) if path.exists() else None
    except Exception:  # noqa: BLE001  a base file that cannot be read gives no earlier text
        return None


def _policy_section(event: RoundEvent, rounds: list[RoundEvent], root: Path) -> list[str]:
    out: list[str] = []
    edit = event.edit
    if edit is None:
        out.append("No edit.")
    else:
        out.append("```diff")  # fenced, so the + and - lines are not read as a markdown list
        if edit.action in ("reword", "remove"):
            old = _earlier_text(edit.rule_id, event, rounds, root)
            out.append(f"- {edit.rule_id}. {old}" if old is not None
                       else f"- {edit.rule_id}. (the earlier text is not in the root's files)")
        if edit.action in ("add", "reword"):
            out.append(f"+ {edit.rule_id}. {edit.rule_text or ''}".rstrip())
        out += ["```", ""]
        out.append(f"The edit was {event.outcome}.")
    out += ["", f"Policy in force after the round: {event.policy.in_force}", ""]
    return out


def _quality(x: Any, measured: bool) -> str:
    """not measured when there are no scores, unreadable when the figure is not a finite number."""
    if not measured or x is None:
        return "not measured"
    return _num(x) if finite(x) is not None else UNREADABLE


def _gate_section(event: RoundEvent, files: RoundFiles) -> list[str]:
    if "outcome" not in event.records:
        return ["The gate did not run.", ""]
    report = files.gate_report
    if report is None and not files.gate_report_exists:
        return ["Stand-in gate: nothing was measured.", ""]
    out: list[str] = []
    for name, status in (report["checks"] if report is not None else {}).items():
        out.append(f"- {name}: {CHECK_WORDS.get(status, UNREADABLE)}")
    out.append("")
    s = event.scores
    if s is None:
        noise = "not measured"
    elif finite(s.noise) is None:
        noise = UNREADABLE
    elif noise_measured(report):
        noise = _num(s.noise)
    else:
        noise = f"not measured; the figure given was {s.noise:.2f}"
    out.append(f"Quality before: {_quality(s.before.quality if s else None, s is not None)}. "
               f"Quality after: {_quality(s.after.quality if s else None, s is not None)}. Noise: {noise}.")
    if event.safety is None:
        out.append("The safety check did not run.")
    elif event.safety.failed_case_ids:
        out.append("Safety cases that failed: " + ", ".join(event.safety.failed_case_ids) + ".")
    else:
        out.append(f"No safety case failed ({event.safety.cases} cases, {event.safety.case_file}).")
    return out + [""]


def _dollars(x: float) -> str:
    """Four decimal places; a sum of finite figures can still overflow, and then it is unreadable."""
    return f"{x:.4f}" if math.isfinite(x) else UNREADABLE


def _cost_section(root: Path, round_: int, problems: list[str]) -> list[str]:
    tally: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
    bad = 0
    for line in _lines(root / USAGE_PATH, root, problems):
        if line.get("event") != "done":
            continue
        rnd = line.get("round")
        if is_int(rnd) and rnd != round_:
            continue
        # A finished call of this round, or one whose round cannot be read: every value it gives is checked.
        kind, agent, usd = line.get("kind"), line.get("agent"), line.get("cost_usd")
        tokens = (line.get("tokens_in"), line.get("tokens_out"))
        ok = is_int(rnd) and isinstance(agent, str) and is_number(usd) and usd >= 0 and (
            kind == "search" or (kind == "model" and all(is_int(n) and n >= 0 for n in tokens)))
        if not ok:
            bad += 1  # a missing or non-finite number is never counted as 0
            continue
        calls, tokens_in, tokens_out = (1, tokens[0], tokens[1]) if kind == "model" else (0, 0, 0)
        t = tally[agent]
        t["calls"] += calls
        t["in"] += tokens_in
        t["out"] += tokens_out
        t["usd"] += usd
    if bad:
        problems.append(f"{bad} finished call(s) in {USAGE_PATH}, of this round or of a round that cannot be read, "
                        "have an agent, a kind or a number that is missing or could not be read (not a finite "
                        "number); they are left out of the totals.")
    out = [_row("Agent", "Calls", "Tokens in", "Tokens out", "Dollars"), _row("---", "---", "---", "---", "---")]
    for agent in sorted(tally):
        t = tally[agent]
        out.append(_row(agent, int(t["calls"]), int(t["in"]), int(t["out"]), _dollars(t["usd"])))
    total = sum(t["usd"] for t in tally.values())
    out.append(_row("total", int(sum(t["calls"] for t in tally.values())), int(sum(t["in"] for t in tally.values())),
                    int(sum(t["out"] for t in tally.values())), _dollars(total)))
    return out + [""]
