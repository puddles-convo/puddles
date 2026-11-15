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
import re
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

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


# ---------- what the view will show, by shape (the contract's last section) ----------
#
# Every value the view prints, traces, sums, averages, compares or uses as a key goes through exactly
# one of name() (rule 1), number() (rule 2), enum() (rule 3) or text() (rule 4). An error is described
# only by _why() (rule 5), and every MLflow write the view makes goes through guarded() (rule 6).
#
# A number is always taken from the JSON as the file holds it, never from a model: a model turns
# `true` into 1 and "4.5" into 4.5. A string or a Literal field is never coerced by a model (a model
# refuses a non-string for it), so name(), enum() and text() may take a model's value.

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+=#~-]*")
RANGES = {"score": (1, 5), "count": (0, 10 ** 9), "tokens": (0, 10 ** 12), "cost": (0, 10 ** 6), "noise": (0, 5)}
OUTCOMES = ("kept", "reverted", "no_pattern", "no_evidence")
DECISIONS = ("kept", "reverted")
STATUSES = ("passed", "failed", "not_run")
COPY_STATUSES = ("confirmed", "pending")
ACTIONS = ("add", "reword", "remove")
USAGE_KINDS = ("model", "search")

Number = Union[int, float]


# Rule 8: while a report section is built, every value its shape function shows as `unreadable` is
# counted here, and the section ends with one line giving the count (see _counted). Outside a section
# (a summary, or a check used only to decide), nothing is counted.
_UNREADABLE_SHOWN: ContextVar[Optional[list[int]]] = ContextVar("unreadable_shown", default=None)


def unreadable() -> str:
    """The word shown in place of a value that is not of its kind, counted for the section's line."""
    counter = _UNREADABLE_SHOWN.get()
    if counter is not None:
        counter[0] += 1
    return UNREADABLE


def is_name(x: Any) -> bool:
    """Rule 1's test alone, for a decision; nothing is shown, so nothing is counted."""
    return isinstance(x, str) and 1 <= len(x) <= 200 and NAME_PATTERN.fullmatch(x) is not None


def name(x: Any) -> str:
    """Rule 1. An ID, agent, model, judge or policy version, file name or path, reader, writer, check
    name, or MLflow ID: shown only when it is 1 to 200 characters of the pattern, else `unreadable`.
    A placeholder such as `?` that a file holds is a name like any other, and fails."""
    return x if is_name(x) else unreadable()


def number(x: Any, kind: str) -> Optional[Number]:
    """Rule 2. The value as read from the file, when it is an int that is not a bool or a finite float,
    within the range of its kind (score, count, tokens, cost, noise); otherwise None, never a guess.
    The type is looked at before anything else, so a huge int is compared, never turned into a float."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    low, high = RANGES[kind]
    return x if low <= x <= high else None


def enum(x: Any, allowed: tuple, otherwise: Any = UNREADABLE) -> Any:
    """Rule 3. One of the allowed values, of the same type (so 1 is not True), or `otherwise`."""
    if any(type(x) is type(a) and x == a for a in allowed):
        return x
    return unreadable() if otherwise == UNREADABLE else otherwise


def text(x: Any) -> str:
    """Rule 4, only for the free texts the contract allows: a reason, a pattern's name and description,
    a rule's text, an edit's summary, a source's URL, title and technique. Anything else is a name."""
    return x if isinstance(x, str) else unreadable()


def _why(exc: BaseException) -> str:
    """Rule 5. Only the kind of error: its message can quote the input it failed on, which may be a spoken line."""
    return type(exc).__name__


def guarded(write: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Rule 6. An MLflow write for the view: a failure is swallowed, so the view never breaks a round."""
    try:
        write(*args, **kwargs)
    except Exception:  # noqa: BLE001
        pass


def _at(data: Any, *keys: str) -> Any:
    """A raw value inside nested JSON objects, or None when a level is missing or not an object."""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _measured(raw: Any, kind: str, fmt: str = "{:.2f}") -> str:
    """A number for the report: `not measured` when absent, `unreadable` when not of its kind."""
    if raw is None:
        return "not measured"
    value = number(raw, kind)
    return unreadable() if value is None else fmt.format(value)


def _counted(build: Callable[..., list[str]], *args: Any) -> list[str]:
    """Rule 8: the lines `build` returns, ending with how many values in them could not be read."""
    counter = [0]
    token = _UNREADABLE_SHOWN.set(counter)
    try:
        lines = build(*args)
    finally:
        _UNREADABLE_SHOWN.reset(token)
    if counter[0]:
        lines = [*lines, f"{counter[0]} value(s) here could not be read and are shown as unreadable.", ""]
    return lines


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


def _score_line(data: dict) -> bool:
    """What the view takes from a score line, checked as JSON gave it, before any coercion."""
    return is_name(data.get("conversation_id")) and all(
        number(data.get(k), "score") is not None for k in ("empathy", "relevance"))


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


def _gate_checks() -> tuple[str, ...]:
    from puddles.gate import CHECKS  # puddles.gate imports puddles.graph, which imports this module

    return tuple(CHECKS)


def _clean_gate_report(data: Any, round_: int, gate: Optional[GateOutcome], name_: str,
                       where: list[str]) -> Optional[dict]:
    """The gate report as the view uses it, every value checked for what it is.

    Returns {"checks": {check: status or "unreadable"}, "unknown": count, "readable": bool,
    "failed": list or None}, or None when the file is not this round's gate report. A check value
    that is not passed, failed or not_run, and a check whose name is not a gate check, are never shown.
    """
    if not isinstance(data, dict):
        where.append(f"{name_} is not a gate report; its checks are not shown.")
        return None
    rnd, decision, checks = data.get("round"), data.get("decision"), data.get("checks")
    if (number(rnd, "count") is None or (decision is not None and enum(decision, DECISIONS, None) is None)
            or not isinstance(checks, dict)):
        where.append(f"{name_} is not a gate report (its round, decision or checks are unreadable); "
                     "its checks are not shown.")
        return None
    if rnd != round_ or (gate is not None and decision is not None and decision != gate.decision):
        where.append(f"{name_} does not match this round's gate decision; it was left by an earlier attempt "
                     "and its checks are not shown.")
        return None
    known = _gate_checks()
    clean: dict[str, str] = {}
    unknown = bad = 0
    for check, status in checks.items():
        if enum(check, known, None) is None:
            unknown += 1
        elif enum(status, STATUSES, None) is not None:
            clean[check] = status
        else:
            clean[check], bad = UNREADABLE, bad + 1
    if bad:
        where.append(f"{bad} check value(s) in {name_} are not passed, failed or not_run; they are shown as "
                     "unreadable and the round's failed checks are not worked out.")
    if unknown:
        where.append(f"{unknown} check(s) in {name_} have a name that is not a gate check; they are unreadable "
                     "and not shown.")
    failed = data.get("failed")
    failed_names = (list(failed) if isinstance(failed, list) and all(enum(f, known, None) is not None for f in failed)
                    else None)
    return {"checks": clean, "unknown": unknown, "readable": not bad and not unknown, "failed": failed_names}


@dataclass
class RoundFiles:
    """What one round left on disk, as far as it could be read; `problems` says what could not."""

    records: list[ConversationRecord] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)  # {"scenario_id", "reason"} as the file holds them
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
        conversations_path(root, round_), root, ConversationRecord, conv, lambda data: is_name(data.get("id")))
    unreadable_failures = 0
    for data in _lines(failures_file(root, round_), root, conv):
        scenario, reason = data.get("scenario_id"), data.get("reason")  # checked where they are shown
        unreadable_failures += not is_name(scenario) or not isinstance(reason, str)
        files.failures.append({"scenario_id": scenario, "reason": reason})
    if unreadable_failures:
        conv.append(f"{unreadable_failures} line(s) of {_rel(failures_file(root, round_), root)} have a scenario "
                    "that is not an ID or a reason that is not text; it is shown as unreadable.")

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
        file_name = _rel(gate_path, root)
        try:
            data = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            where.append(f"{file_name} could not be read ({_why(exc)}); its checks are not shown.")
        else:
            files.gate_report = _clean_gate_report(data, round_, gate, file_name, where)
    return files


def _failed_checks(report: Optional[dict]) -> Optional[list[str]]:
    """The names of the failed checks, or None when there is no gate report or part of it is unreadable."""
    if report is None or not report["readable"]:
        return None
    if report["checks"]:
        return [check for check, status in report["checks"].items() if status == "failed"]
    return report["failed"]


def noise_measured(report: Optional[dict]) -> bool:
    """The noise is a measurement only when a readable gate report marks noise_not_measured as passed."""
    return report is not None and report["readable"] and report["checks"].get("noise_not_measured") == "passed"


def _mean(values: list[Optional[Number]]) -> Optional[float]:
    readable = [v for v in values if v is not None]
    return sum(readable) / len(readable) if readable else None


def _log(root: Path) -> list[tuple[Any, dict]]:
    """Each event of the decision log with the JSON its line holds, so numbers are read as written.
    [] when there is no log; raises when it cannot be read."""
    log = DecisionLog(Path(root))
    if not log.path.exists():
        return []
    events = log.events()
    raw = [json.loads(line) for line in log.path.read_text().splitlines() if line.strip()]
    return list(zip(events, raw))


def _round_copies(log: list[tuple[Any, dict]], round_: int) -> list[tuple[MemoryCopyEvent, dict]]:
    return [(e, raw) for e, raw in log if isinstance(e, MemoryCopyEvent) and number(raw.get("round"), "count") == round_]


# ---------- 1. a summary on each node's span ----------

def node_summary(node: str, state: Mapping[str, Any], update: Mapping[str, Any], root: Path) -> dict:
    """What the node did, in a few numbers and names, without any conversation text. Never raises:
    a value that cannot be worked out is null."""
    try:
        return _node_summary(node, state, update, Path(root))
    except Exception as exc:  # noqa: BLE001  the view never breaks a round
        return {"error": _why(exc)}


def _count(items: Any) -> Optional[int]:
    """How many there are, or None when the node gave none: a missing list is not an empty one."""
    return len(items) if isinstance(items, (list, tuple)) else None


def _node_summary(node: str, state: Mapping[str, Any], update: Mapping[str, Any], root: Path) -> dict:
    round_ = int(state["round"])
    if node == "converse":
        return {"finished": _count(update.get("records")), "failed": _count(update.get("failures")),
                "file": name(_rel(conversations_path(root, round_), root))}
    if node == "judge":
        records = state.get("records")
        files = read_round_files(root, round_, records=list(records) if records is not None else None)
        if files.scores:
            scored: list = files.scores
            version: Optional[str] = name(files.scores[0].judge_version)
        else:  # no scores saved this round: what the node returned is all that was scored
            scored, version = list(update.get("scored", [])), None
        return {"scored": len(scored), "unscored": _count(update.get("unscored")),
                "mean_empathy": _mean([number(s.empathy, "score") for s in scored]),
                "mean_relevance": _mean([number(s.relevance, "score") for s in scored]), "judge_version": version}
    result = update.get("result", state.get("result"))
    if node == "critic":
        if result is not None and result.pattern is not None:
            p = result.pattern
            return {"pattern": text(p.name), "seen_in": number(p.seen_in, "count"),
                    "out_of": number(p.out_of, "count")}
        return {"pattern": None, "reason": text(result.reason) if result is not None else None}
    if node == "researcher":
        research = result.research if result is not None else None
        if research is None:  # the researcher failed: nothing was counted
            return {"queries": None, "sources": None, "reason": text(result.reason) if result is not None else None}
        return {"queries": len(_queries(research)), "sources": [text(s.url) for s in research.sources]}
    if node == "proposer":
        edit = result.edit if result is not None else None
        if edit is None:
            return {"edit": None, "reason": text(result.reason) if result is not None else None}
        return {"action": enum(edit.action, ACTIONS) if edit.action is not None else None,
                "rule_id": name(edit.rule_id) if edit.rule_id is not None else None,
                "rule_text": text(edit.rule_text) if edit.rule_text is not None else None}
    if node == "gate":
        outcome = update.get("gate")
        scores = result.scores if result is not None else None
        safety = result.safety if result is not None else None
        report = read_round_files(root, round_, records=[], gate=outcome).gate_report
        return {"decision": enum(outcome.decision, DECISIONS, None) if outcome is not None else None,
                "quality_before": number(scores.before.quality, "score") if scores is not None else None,
                "quality_after": number(scores.after.quality, "score") if scores is not None else None,
                "noise": number(scores.noise, "noise") if scores is not None and noise_measured(report) else None,
                "safety_passed": _safety_passed(safety),
                "failed_checks": _failed_checks(report)}
    if node == "finish":
        event = update.get("event")
        try:
            copies: Optional[list] = [{"record_id": name(e.record_id), "status": enum(e.status, COPY_STATUSES)}
                                      for e, _ in _round_copies(_log(root), round_)]
        except Exception:  # noqa: BLE001  a log that cannot be read gives no copies, not an empty list
            copies = None
        return {"outcome": enum(event.outcome, OUTCOMES) if event is not None else None,
                "policy_in_force": name(event.policy.in_force) if event is not None else None, "copies": copies}
    return {}


def _safety_passed(safety: Any) -> Optional[bool]:
    """Rule 7: a safety result with no case is not a pass. None when there is no result."""
    if safety is None:
        return None
    if number(safety.cases, "count") == 0:
        return False
    return enum(safety.passed, (True, False), None)


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
        guarded(_put_summary, node, state, update or {}, root)
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
        if outcome is None or enum(outcome.decision, DECISIONS, None) is None:
            return {}
        out: dict[str, float] = {"kept": 1 if outcome.decision == "kept" else 0}
        result = state.get("result")
        scores = result.scores if result is not None else None
        if scores is not None:
            for key, value in (("quality_before", number(scores.before.quality, "score")),
                               ("quality_after", number(scores.after.quality, "score"))):
                if value is not None:
                    out[key] = value
            report = read_round_files(Path(root), int(state["round"]), records=[], gate=outcome).gate_report
            noise = number(scores.noise, "noise")
            if noise_measured(report) and noise is not None:
                out["noise"] = noise
        return out
    except Exception:  # noqa: BLE001  the view never breaks a round
        return {}


def log_round_view(state: Mapping[str, Any], root: Path) -> None:
    """The view's tag and numbers on the round's run: the root's resolved path and the gate's metrics.
    Called after the round event is on disk, so nothing here may raise."""
    guarded(lambda: mlflow.set_tag("root", str(Path(root).resolve())))
    metrics = gate_metrics(state, root)
    if metrics:
        guarded(mlflow.log_metrics, metrics)


# ---------- 3. the report ----------

def _read_log(root: Path) -> list[tuple[Any, dict]]:
    log = DecisionLog(Path(root))
    if not log.path.exists():
        raise ReportError(f"{log.path} does not exist")
    try:
        return _log(root)
    except Exception as exc:  # noqa: BLE001  any unreadable log is the same refusal
        raise ReportError(f"{log.path} cannot be read ({_why(exc)})") from exc


def _rounds(log: list[tuple[Any, dict]]) -> list[tuple[RoundEvent, dict]]:
    """The round events whose round, as the line holds it, is a readable count."""
    return [(e, raw) for e, raw in log if isinstance(e, RoundEvent) and number(raw.get("round"), "count") is not None]


def read_rounds(root: Path) -> list[RoundEvent]:
    """The round events in the root's log. ReportError when there is no log or it cannot be read."""
    return [e for e, _ in _rounds(_read_log(root))]


def _cell(value: Any) -> str:
    shown = "" if value is None else " ".join(str(value).split())
    return shown.replace("|", "\\|")


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


def _record(event: RoundEvent, kind: str, model: type, root: Path, problems: list[str]) -> tuple[Any, dict]:
    """The record of this kind and the JSON its file holds (for its numbers), or (None, {})."""
    ref = event.records.get(kind)
    if ref is None:
        return None, {}
    try:
        record = load_record(ref, model, root)
        raw = json.loads((root / ref.path).read_bytes()) if record is not None else {}
    except Exception as exc:  # noqa: BLE001  a half-written record file is reported, not raised
        problems.append(f"The {kind} record {name(ref.path)} could not be read ({_why(exc)}).")
        return None, {}
    if record is None:
        problems.append(f"The {kind} record {name(ref.path)} is missing or no longer matches the log.")
    return record, raw if isinstance(raw, dict) else {}


def build_report(root: Path, round_: int) -> str:
    """The round's report as markdown. It reads the root's files and writes nothing.

    Only a missing or unreadable log, or a round not in it, raises (ReportError). Anything else the
    round left that cannot be read is left out, and the report says so in the section it belongs to.
    """
    root = Path(root)
    log = _read_log(root)
    rounds = _rounds(log)
    found = next(((e, raw) for e, raw in rounds if raw["round"] == round_), None)
    if found is None:
        raise ReportError(f"round {round_} is not in {DecisionLog(root).path}")
    event, raw_event = found
    problems: dict[str, list[str]] = defaultdict(list)
    pattern = _record(event, "pattern", FailurePattern, root, problems["Nodes"])
    research, _ = _record(event, "research", ResearchNote, root, problems["Nodes"])
    edit, _ = _record(event, "edit", PolicyEdit, root, problems["Nodes"])
    outcome, _ = _record(event, "outcome", GateOutcome, root, problems["Nodes"])
    files = read_round_files(root, round_, gate=outcome)
    for section, lines in files.problems.items():
        problems[section] += lines
    copies = _round_copies(log, round_)

    sections = [
        ("Nodes", _counted(_nodes_section, event, raw_event, root, files, pattern, research, edit, copies)),
        ("Conversations", _counted(_conversations_section, files)),
        ("Memory", _counted(_memory_section, event, root, copies, problems["Memory"])),
        ("Policy change", _counted(_policy_section, event, [e for e, _ in rounds], root)),
        ("Gate", _counted(_gate_section, event, raw_event, files)),
        ("Cost", _counted(_cost_section, root, round_, problems["Cost"])),
    ]
    out = _counted(lambda: [f"# Round {round_}: {enum(event.outcome, OUTCOMES)}", "", text(event.reason), ""])
    for section, body in sections:
        out += [f"## {section}", "", *body]
        if problems.get(section):
            out += [*problems[section], ""]
    return "\n".join(out).rstrip("\n") + "\n"


def _nodes_section(event: RoundEvent, raw_event: dict, root: Path, files: RoundFiles,
                   pattern_record: tuple[Optional[FailurePattern], dict], research: Optional[ResearchNote],
                   edit: Optional[PolicyEdit], copies: list) -> list[str]:
    round_ = event.round
    ran = nodes_that_ran(event)
    records, scores = files.records, files.scores
    fail_file = failures_file(root, round_)
    pattern, raw_pattern = pattern_record

    def where(kind: str) -> str:
        ref = event.records.get(kind)
        return f"{name(ref.path)}, the decision log" if ref is not None else "the decision log"

    def seen(raw: Any) -> str:
        return f"seen in {_measured(_at(raw, 'seen_in'), 'count', '{}')} of {_measured(_at(raw, 'out_of'), 'count', '{}')}"

    rows: dict[str, tuple[str, str, str]] = {}
    went = name(_rel(conversations_path(root, round_), root)) + (
        f", {name(_rel(fail_file, root))}" if fail_file.exists() else "")
    rows["converse"] = (f"policy {name(event.policy.base)}",
                        f"{len(records)} conversations finished, {len(files.failures)} failed", went)
    if scores is not None:
        unscored = len([r for r in records if r.id not in {s.conversation_id for s in scores}])
        rows["judge"] = (f"{len(records)} conversations", f"{len(scores)} scored, {unscored} unscored",
                         name(_rel(scores_path(root, round_), root)))
    else:
        rows["judge"] = (f"{len(records)} conversations", "scores that were not saved (no scores file)",
                         "the critic")
    if pattern is not None:
        critic_out = f"pattern {text(pattern.name)} ({seen(raw_pattern)})"
    elif event.pattern is not None:
        critic_out = f"pattern {text(event.pattern.name)} ({seen(_at(raw_event, 'pattern'))})"
    else:
        critic_out = "no pattern"
    given = _measured(raw_pattern.get("out_of"), "count", "{}") if pattern is not None else _critic_count(scores)
    rows["critic"] = (f"{given} scored conversations" if given is not None else "the scored conversations",
                      critic_out, where("pattern"))
    rows["researcher"] = (f"pattern {text(event.pattern.name)}" if event.pattern else "",
                          f"{len(research.sources)} sources from {len(_queries(research))} queries"
                          if research is not None else "no research note that could be read", where("research"))
    if edit is not None:
        logged = event.edit
        action = edit.action if edit.action is not None else (logged.action if logged is not None else None)
        rule = edit.rule_id if edit.rule_id is not None else (logged.rule_id if logged is not None else None)
        proposer_out = f"{enum(action, ACTIONS)} {name(rule)}"
        proposer_went = f"{where('edit')}, {name(edit.policy_path)}"
    elif event.edit is not None:
        proposer_out = f"{enum(event.edit.action, ACTIONS)} {name(event.edit.rule_id)}"
        proposer_went = where("edit")
    else:
        proposer_out, proposer_went = "no edit", "the decision log"
    rows["proposer"] = ("the pattern and the research note", proposer_out, proposer_went)
    gate_went = where("outcome")
    if files.gate_report is not None:
        gate_went += f", {name(_rel(gate_report_path(root, round_), root))}"
    rows["gate"] = (f"candidate {name(event.policy.new)}" if event.policy.new else "the candidate",
                    enum(event.outcome, OUTCOMES), gate_went)
    finish_went = "the decision log, memory"
    if event.outcome == "kept" and event.policy.new:
        finish_went += f", {name(f'policies/{event.policy.new}.md')}"
    rows["finish"] = (f"the round's result ({enum(event.outcome, OUTCOMES)})",
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
            out.append(_row(name(record.id), "", "", ""))
        elif score is None:
            out.append(_row(name(record.id), "", "", "unscored"))
        else:
            out.append(_row(name(record.id), _measured(score.empathy, "score", "{}"),
                            _measured(score.relevance, "score", "{}"), "yes" if record.id in given else "no"))
    for failure in files.failures:
        out.append(_row(f"{name(failure['scenario_id'])} (failed)", "", "", f"failed: {text(failure['reason'])}"))
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
    ids = {ref.id for ref in event.records.values()}
    try:
        if not _reads_file_readable(root):
            raise ValueError("a line of the reads file is not a reader's name and a list of record IDs")
        rows = [r for r in memory_trace(DecisionLog(root), root) if r["round"] == event.round and r["record"] in ids]
    except Exception as exc:  # noqa: BLE001  a half-written reads file or record: say so, list what the log holds
        problems.append(f"Who read the round's records could not be worked out ({_why(exc)}).")
        rows = [{"record": ref.id, "written_by": None, "read_by": None} for ref in event.records.values()]
    for row in rows:
        if row["read_by"] is None:
            readers = "unknown"
        else:
            readers = ", ".join(name(r) for r in row["read_by"]) if row["read_by"] else "no agent yet"
        out.append(f"- {name(row['record'])}: written by {name(row['written_by'])}, read later by {readers}")
    if not rows:
        out.append("- The round wrote no record.")
    out.append("")
    for copy, raw in copies:
        out.append(f"- memory copy of {name(copy.record_id)}: {enum(copy.status, COPY_STATUSES)} "
                   f"(attempt {_measured(raw.get('attempt'), 'count', '{}')})")
    if not copies:
        out.append("- No memory copy event.")
    return out + [""]


def _earlier_text(rule_id: str, event: RoundEvent, rounds: list[RoundEvent], root: Path) -> Optional[str]:
    """The rule's text before this round: from the last kept edit of it in the log, else the base policy file."""
    text_: Optional[str] = None
    found = False
    for r in rounds:
        if r.round >= event.round:
            break
        if r.outcome == "kept" and r.edit is not None and r.edit.rule_id == rule_id:
            text_, found = r.edit.rule_text, True
    if found:
        return text_
    if not is_name(event.policy.base):
        return None
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
        rule_id = name(edit.rule_id)
        out.append("```diff")  # fenced, so the + and - lines are not read as a markdown list
        if edit.action in ("reword", "remove"):
            old = _earlier_text(edit.rule_id, event, rounds, root) if is_name(edit.rule_id) else None
            out.append(f"- {rule_id}. {text(old)}" if old is not None
                       else f"- {rule_id}. (the earlier text is not in the root's files)")
        if edit.action in ("add", "reword"):
            out.append(f"+ {rule_id}. {text(edit.rule_text)}")
        out += ["```", ""]
        out.append(f"The edit was {enum(event.outcome, OUTCOMES)}.")
    out += ["", f"Policy in force after the round: {name(event.policy.in_force)}", ""]
    return out


def _gate_section(event: RoundEvent, raw_event: dict, files: RoundFiles) -> list[str]:
    if "outcome" not in event.records:
        return ["The gate did not run.", ""]
    report = files.gate_report
    if report is None and not files.gate_report_exists:
        return ["Stand-in gate: nothing was measured.", ""]
    out: list[str] = []
    for check, status in (report["checks"] if report is not None else {}).items():  # names checked by enum()
        out.append(f"- {check}: {CHECK_WORDS[status] if status in CHECK_WORDS else unreadable()}")
    out.append("")
    if event.scores is None:
        before = after = noise = "not measured"
    else:
        before = _measured(_at(raw_event, "scores", "before", "quality"), "score")
        after = _measured(_at(raw_event, "scores", "after", "quality"), "score")
        figure = number(_at(raw_event, "scores", "noise"), "noise")
        if figure is None:
            noise = unreadable()
        elif noise_measured(report):
            noise = _num(figure)
        else:
            noise = f"not measured; the figure given was {figure:.2f}"
    out.append(f"Quality before: {before}. Quality after: {after}. Noise: {noise}.")
    cases = _at(raw_event, "safety", "cases")
    if event.safety is None:
        out.append("The safety check did not run.")
    elif number(cases, "count") == 0:  # rule 7: whatever `passed` says, no case ran
        case_file = event.safety.case_file
        out.append("No safety case was run." + (f" Case file: {case_file}." if is_name(case_file) else ""))
    elif event.safety.failed_case_ids:
        out.append("Safety cases that failed: " + ", ".join(name(c) for c in event.safety.failed_case_ids) + ".")
    else:
        out.append(f"No safety case failed ({_measured(cases, 'count', '{}')} cases, "
                   f"{name(event.safety.case_file)}).")
    return out + [""]


def _cost_section(root: Path, round_: int, problems: list[str]) -> list[str]:
    tally: dict[str, dict[str, Number]] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
    bad = 0
    for line in _lines(root / USAGE_PATH, root, problems):
        if enum(line.get("event"), ("done",), None) is None:
            continue
        rnd = number(line.get("round"), "count")
        if rnd is not None and rnd != round_:
            continue
        # A finished call of this round, or one whose round cannot be read: every value it gives is checked.
        kind = enum(line.get("kind"), USAGE_KINDS, None)
        agent = line.get("agent") if is_name(line.get("agent")) else None
        usd = number(line.get("cost_usd"), "cost")
        tokens = (number(line.get("tokens_in"), "tokens"), number(line.get("tokens_out"), "tokens"))
        ok = rnd is not None and agent is not None and usd is not None and kind is not None and (
            kind == "search" or all(n is not None for n in tokens))
        if not ok:
            bad += 1  # a value that could not be read is never counted, and never as 0
            continue
        calls, tokens_in, tokens_out = (1, tokens[0], tokens[1]) if kind == "model" else (0, 0, 0)
        t = tally[agent]
        t["calls"] += calls
        t["in"] += tokens_in
        t["out"] += tokens_out
        t["usd"] += usd
    if bad:
        problems.append(f"{bad} finished call(s) in {USAGE_PATH}, of this round or of a round that cannot be read, "
                        "have an agent, a kind or a number that is missing or could not be read (not a number "
                        "of its kind and range); they are left out of the totals.")
    # Each readable value is bounded, so these sums are finite and short enough to print.
    out = [_row("Agent", "Calls", "Tokens in", "Tokens out", "Dollars"), _row("---", "---", "---", "---", "---")]
    for agent in sorted(tally):
        t = tally[agent]
        out.append(_row(agent, int(t["calls"]), int(t["in"]), int(t["out"]), f"{t['usd']:.4f}"))
    out.append(_row("total", int(sum(t["calls"] for t in tally.values())), int(sum(t["in"] for t in tally.values())),
                    int(sum(t["out"] for t in tally.values())), f"{sum(t['usd'] for t in tally.values()):.4f}"))
    return out + [""]
