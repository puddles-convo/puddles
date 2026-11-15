"""The round view: a summary on each node's span, numbers on the round's run, and a report per round.

The contract is docs/specs/round-view.md. Everything here only reads what a round left behind: it
never writes to the decision log, a record, a policy, a conversations or a scores file, and never
calls a model, Exa or past.dev. No turn of any conversation is copied into a summary or a report.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import mlflow

from puddles import replay
from puddles.decision_log import DecisionLog
from puddles.judging import CRITIC_CONVERSATIONS, scored_conversations, select_for_critic
from puddles.policy import parse_policy
from puddles.records import load_record
from puddles.round import memory_trace
from puddles.usage import USAGE_PATH, read_usage
from schemas import (
    ConversationRecord,
    FailurePattern,
    MemoryCopyEvent,
    PolicyEdit,
    ResearchNote,
    RoundEvent,
    ScoreRecord,
)

SUMMARY_KEY = "puddles.summary"
NODES = ("converse", "judge", "critic", "researcher", "proposer", "gate", "finish")
CHECK_WORDS = {"passed": "passed", "failed": "failed", "not_run": "not run"}


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


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scores(root: Path, round_: int) -> Optional[list[ScoreRecord]]:
    """The round's scores file, or None when there is none (the stub judge saves no scores)."""
    path = scores_path(root, round_)
    if not path.exists():
        return None
    return [ScoreRecord.model_validate(line) for line in _jsonl(path)]


def _gate_report(root: Path, round_: int) -> Optional[dict]:
    """The real gate's report, or None when the file does not exist (a stand-in gate)."""
    path = gate_report_path(root, round_)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _failed_checks(report: Optional[dict]) -> Optional[list[str]]:
    if report is None:
        return None
    checks = report.get("checks")
    if isinstance(checks, dict):
        return [name for name, status in checks.items() if status == "failed"]
    return list(report.get("failed", []))


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


# ---------- 1. a summary on each node's span ----------

def node_summary(node: str, state: Mapping[str, Any], update: Mapping[str, Any], root: Path) -> dict:
    """What the node did, in a few numbers and names, without any conversation text."""
    root = Path(root)
    round_ = int(state["round"])
    if node == "converse":
        return {"finished": len(update.get("records", [])), "failed": len(update.get("failures", [])),
                "file": _rel(conversations_path(root, round_), root)}
    if node == "judge":
        scores = _scores(root, round_)
        if scores is not None:
            empathy, relevance = [s.empathy for s in scores], [s.relevance for s in scores]
        else:  # no scores file: what the node returned is all that was scored
            scored = list(update.get("scored", []))
            empathy, relevance = [s.empathy for s in scored], [s.relevance for s in scored]
        return {"scored": len(empathy), "unscored": len(update.get("unscored", [])),
                "mean_empathy": _mean(empathy), "mean_relevance": _mean(relevance),
                "judge_version": scores[0].judge_version if scores else None}
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
        return {"decision": outcome.decision if outcome is not None else None,
                "quality_before": scores.before.quality if scores is not None else None,
                "quality_after": scores.after.quality if scores is not None else None,
                "noise": scores.noise if scores is not None else None,
                "safety_passed": safety.passed if safety is not None else None,
                "failed_checks": _failed_checks(_gate_report(root, round_))}
    if node == "finish":
        event = update.get("event")
        copies = [{"record_id": e.record_id, "status": e.status} for e in DecisionLog(root).events()
                  if isinstance(e, MemoryCopyEvent) and e.round == round_]
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

def gate_metrics(state: Mapping[str, Any]) -> dict[str, float]:
    """`kept` whenever the gate decided; the quality numbers only when the gate measured them."""
    outcome = state.get("gate")
    if outcome is None:
        return {}
    out: dict[str, float] = {"kept": 1 if outcome.decision == "kept" else 0}
    result = state.get("result")
    scores = result.scores if result is not None else None
    if scores is not None:
        if scores.before.quality is not None:
            out["quality_before"] = scores.before.quality
        if scores.after.quality is not None:
            out["quality_after"] = scores.after.quality
        out["noise"] = scores.noise
    return out


# ---------- 3. the report ----------

def read_rounds(root: Path) -> list[RoundEvent]:
    """The round events in the root's log. ReportError when there is no log or it cannot be read."""
    log = DecisionLog(Path(root))
    if not log.path.exists():
        raise ReportError(f"{log.path} does not exist")
    try:
        return log.rounds()
    except Exception as exc:  # noqa: BLE001  any unreadable log is the same refusal
        raise ReportError(f"{log.path} cannot be read ({type(exc).__name__}: {exc})") from exc


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


def _record(event: RoundEvent, kind: str, model: type, root: Path) -> Any:
    ref = event.records.get(kind)
    return load_record(ref, model, root) if ref is not None else None


def build_report(root: Path, round_: int) -> str:
    """The round's report as markdown. It reads the root's files and writes nothing."""
    root = Path(root)
    rounds = read_rounds(root)
    event = next((r for r in rounds if r.round == round_), None)
    if event is None:
        raise ReportError(f"round {round_} is not in {DecisionLog(root).path}")
    records = [ConversationRecord.model_validate(line) for line in _jsonl(conversations_path(root, round_))]
    failures = _jsonl(failures_file(root, round_))
    scores = _scores(root, round_)
    gate_report = _gate_report(root, round_)
    pattern = _record(event, "pattern", FailurePattern, root)
    research = _record(event, "research", ResearchNote, root)
    edit = _record(event, "edit", PolicyEdit, root)
    copies = [e for e in DecisionLog(root).events() if isinstance(e, MemoryCopyEvent) and e.round == round_]

    out = [f"# Round {round_}: {event.outcome}", "", event.reason, ""]
    out += _nodes_section(event, root, records, failures, scores, pattern, research, edit, gate_report, copies)
    out += _conversations_section(records, failures, scores)
    out += _memory_section(event, root, copies)
    out += _policy_section(event, rounds, root)
    out += _gate_section(event, gate_report)
    out += _cost_section(root, round_)
    return "\n".join(out).rstrip("\n") + "\n"


def _nodes_section(event: RoundEvent, root: Path, records: list, failures: list, scores: Optional[list],
                   pattern: Optional[FailurePattern], research: Optional[ResearchNote], edit: Optional[PolicyEdit],
                   gate_report: Optional[dict], copies: list) -> list[str]:
    round_ = event.round
    ran = nodes_that_ran(event)
    conv_file = conversations_path(root, round_)
    fail_file = failures_file(root, round_)

    def where(kind: str) -> str:
        ref = event.records.get(kind)
        return f"{ref.path}, the decision log" if ref is not None else "the decision log"

    rows: dict[str, tuple[str, str, str]] = {}
    went = _rel(conv_file, root) + (f", {_rel(fail_file, root)}" if fail_file.exists() else "")
    rows["converse"] = (f"policy {event.policy.base}",
                        f"{len(records)} conversations finished, {len(failures)} failed", went)
    if scores is not None:
        unscored = len([r for r in records if r.id not in {s.conversation_id for s in scores}])
        rows["judge"] = (f"{len(records)} conversations", f"{len(scores)} scored, {unscored} unscored",
                         _rel(scores_path(root, round_), root))
    else:
        rows["judge"] = (f"{len(records)} conversations", "scores that were not saved (no scores file)",
                         "the critic")
    if pattern is not None:
        critic_out = f"pattern {pattern.name} (seen in {pattern.seen_in} of {pattern.out_of})"
    else:
        critic_out = "no pattern"
    given = pattern.out_of if pattern is not None else _critic_count(scores)
    rows["critic"] = (f"{given} scored conversations" if given is not None else "the scored conversations",
                      critic_out, where("pattern"))
    rows["researcher"] = (f"pattern {event.pattern.name}" if event.pattern else "",
                          f"{len(research.sources)} sources from {len(_queries(research))} queries"
                          if research is not None else "no research note", where("research"))
    if edit is not None:
        proposer_out = f"{edit.action or 'add'} {edit.rule_id or ''}".strip()
        proposer_went = f"{where('edit')}, {edit.policy_path}"
    else:
        proposer_out, proposer_went = "no edit", "the decision log"
    rows["proposer"] = ("the pattern and the research note", proposer_out, proposer_went)
    gate_went = where("outcome")
    if gate_report is not None:
        gate_went += f", {_rel(gate_report_path(root, round_), root)}"
    rows["gate"] = (f"candidate {event.policy.new}" if event.policy.new else "the candidate",
                    f"{event.outcome}", gate_went)
    finish_went = "the decision log, memory"
    if event.outcome == "kept" and event.policy.new:
        finish_went += f", policies/{event.policy.new}.md"
    rows["finish"] = (f"the round's result ({event.outcome})",
                      f"the round event, {len(copies)} memory copy events", finish_went)

    out = ["## Nodes", "", _row("Node", "Received", "Produced", "Went to"), _row("---", "---", "---", "---")]
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


def _conversations_section(records: list, failures: list, scores: Optional[list]) -> list[str]:
    out = ["## Conversations", ""]
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
    for failure in failures:
        out.append(_row(f"{failure.get('scenario_id')} (failed)", "", "", f"failed: {failure.get('reason', '')}"))
    if not records and not failures:
        out.append(_row("none", "", "", ""))
    return out + [""]


def _memory_section(event: RoundEvent, root: Path, copies: list) -> list[str]:
    out = ["## Memory", ""]
    rows = [r for r in memory_trace(DecisionLog(root), root) if r["round"] == event.round]
    for row in rows:
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
    if path.exists():
        try:
            return parse_policy(path.read_text()).conversation.get(rule_id)
        except Exception:  # noqa: BLE001  a base file that does not parse gives no earlier text
            return None
    return None


def _policy_section(event: RoundEvent, rounds: list[RoundEvent], root: Path) -> list[str]:
    out = ["## Policy change", ""]
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


def _gate_section(event: RoundEvent, gate_report: Optional[dict]) -> list[str]:
    out = ["## Gate", ""]
    if "outcome" not in event.records:
        return out + ["The gate did not run.", ""]
    if gate_report is None:
        return out + ["Stand-in gate: nothing was measured.", ""]
    for name, status in (gate_report.get("checks") or {}).items():
        out.append(f"- {name}: {CHECK_WORDS.get(status, status)}")
    out.append("")
    s = event.scores
    out.append(f"Quality before: {_num(s.before.quality if s else None)}. "
               f"Quality after: {_num(s.after.quality if s else None)}. "
               f"Noise: {_num(s.noise if s else None)}.")
    if event.safety is None:
        out.append("The safety check did not run.")
    elif event.safety.failed_case_ids:
        out.append("Safety cases that failed: " + ", ".join(event.safety.failed_case_ids) + ".")
    else:
        out.append(f"No safety case failed ({event.safety.cases} cases, {event.safety.case_file}).")
    return out + [""]


def _cost_section(root: Path, round_: int) -> list[str]:
    tally: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
    for line in read_usage(root / USAGE_PATH):
        if line.get("event") != "done" or line.get("round") != round_:
            continue
        t = tally[str(line.get("agent"))]
        if line.get("kind") == "model":
            t["calls"] += 1
            t["in"] += int(line.get("tokens_in", 0))
            t["out"] += int(line.get("tokens_out", 0))
        t["usd"] += float(line.get("cost_usd", 0.0))
    out = ["## Cost", "", _row("Agent", "Calls", "Tokens in", "Tokens out", "Dollars"),
           _row("---", "---", "---", "---", "---")]
    for agent in sorted(tally):
        t = tally[agent]
        out.append(_row(agent, int(t["calls"]), int(t["in"]), int(t["out"]), f"{t['usd']:.4f}"))
    total = sum(t["usd"] for t in tally.values())
    out.append(_row("total", int(sum(t["calls"] for t in tally.values())), int(sum(t["in"] for t in tally.values())),
                    int(sum(t["out"] for t in tally.values())), f"{total:.4f}"))
    return out + [""]
