"""Steps 4 to 6 of a round, then the log entry and the copies into past.dev.

Within a round the critic, researcher and proposer hand records straight to each other.
Only after the round is logged are its records copied into memory, for later rounds to recall.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import httpx

from loop.errors import PolicyError
from puddles import replay
from puddles.agents import critic, proposer, researcher
from puddles.decision_log import DecisionLog, DecisionLogError
from puddles.llm import LLM
from puddles.memory import READS_PATH, Memory, PastError, entry_for, now
from puddles.policy import PolicyEditError, apply_edit, parse_policy, render_policy, safety_section
from puddles.records import load_record, record_path, save_record, sha256_bytes, sha256_text
from schemas import (
    FailurePattern,
    GateOutcome,
    LogEdit,
    LogPattern,
    LogPolicy,
    LogSafety,
    LogScores,
    MemoryCopyEvent,
    PolicyEdit,
    PolicyMeta,
    Record,
    RecordRef,
    ResearchNote,
    RoundEvent,
    ScoredConversation,
    Scores,
)

MODELS: dict[str, type[Record]] = {
    "pattern": FailurePattern, "research": ResearchNote, "edit": PolicyEdit, "outcome": GateOutcome,
}


@dataclass
class RoundResult:
    round: int
    outcome: Literal["candidate", "no_pattern", "no_evidence"]
    base_policy_path: Path
    reason: str = ""
    pattern: Optional[FailurePattern] = None
    status_before: Literal["open", "fixed", "recurring"] = "open"
    research: Optional[ResearchNote] = None
    edit: Optional[PolicyEdit] = None
    candidate_path: Optional[Path] = None
    refs: dict[str, RecordRef] = field(default_factory=dict)
    # Set by the gate, for the round event. None means not measured, never that it passed.
    scores: Optional[LogScores] = None
    safety: Optional[LogSafety] = None
    candidate_sha256: Optional[str] = None  # the candidate text the gate tested; finish_round promotes only that
    base_sha256: Optional[str] = None  # the base policy text the before-scores were measured on
    unscored: list[str] = field(default_factory=list)  # conversation IDs the round's judge left unscored


class PolicyChanged(RuntimeError):
    """A policy file is not the text its measurements were made on, so nothing may be filed under it."""


class CandidateChanged(PolicyChanged):
    """The candidate file is not the text the gate tested, so it must not be promoted."""


class RecordChanged(RuntimeError):
    """A record file of the round is missing or no longer has the hash it was saved with."""


def policy_in_force(root: Path, fallback: Path) -> Path:
    """The newest promoted policy file, for the stage-0 graph runner, which keeps no replay yet.
    scripts/run_round.py takes the policy in force from the log instead (`prepare_round`)."""
    promoted = sorted((root / "policies").glob("policy-v*.md"), key=lambda p: p.stat().st_mtime)
    promoted = [p for p in promoted if "candidate" not in p.name]
    return promoted[-1] if promoted else fallback


NO_EVIDENCE_REASON = "No trustworthy source was found for this pattern."


def unscored_sentence(ids: list[str]) -> str:
    """The sentence finish_round adds to the reason, so an unscored conversation is never silent."""
    noun = "conversation" if len(ids) == 1 else "conversations"
    return f"The judge left {len(ids)} {noun} unscored: {', '.join(ids)}."


def run_critic(round_: int, conversations: list[ScoredConversation], policy_path: Path,
               log: DecisionLog, memory: Memory, llm: LLM, root: Path) -> RoundResult:
    """Step 4. A new result, outcome no_pattern; with the pattern set if the critic found one."""
    policy_text = policy_path.read_text()
    result = RoundResult(round=round_, outcome="no_pattern", base_policy_path=policy_path)
    c_in = critic.build_input(round_, policy_text, conversations, log, memory, root)
    c_out = critic.run(c_in, llm, log)
    pattern = critic.to_record(c_out, c_in, log)
    if pattern is None:
        result.reason = c_out.reason
        return result
    result.pattern = pattern
    statuses = [p.status for p in log.earlier_patterns() if p.id == pattern.pattern_id and p.status]
    result.status_before = statuses[0] if statuses else "open"
    result.refs["pattern"] = save_record(pattern, round_, root)
    return result


def run_researcher(result: RoundResult, log: DecisionLog, memory: Memory, llm: LLM,
                   search: Callable[[str], list[dict]], root: Path) -> RoundResult:
    """Step 5. Sets the research note; with no sources the outcome becomes no_evidence."""
    assert result.pattern is not None
    pattern, round_ = result.pattern, result.round
    r_in = researcher.build_input(pattern, log, memory, root)
    r_out, searches = researcher.run(r_in, llm, search, log)
    research = researcher.to_record(r_out, pattern, round_)
    researcher.save_search_log(searches, round_, root)
    result.research = research
    result.refs["research"] = save_record(research, round_, root)
    if not research.sources:
        result.outcome, result.reason = "no_evidence", NO_EVIDENCE_REASON
    return result


def run_proposer(result: RoundResult, log: DecisionLog, memory: Memory, llm: LLM,
                 root: Path) -> RoundResult:
    """Step 6. Writes the candidate policy; the outcome becomes candidate."""
    assert result.pattern is not None and result.research is not None
    pattern, research, round_ = result.pattern, result.research, result.round
    policy_text = result.base_policy_path.read_text()
    p_in = proposer.build_input(policy_text, pattern, research, log, memory, root)
    policy = parse_policy(policy_text)
    p_out = proposer.run(p_in, policy.version, policy, log, llm)
    edit, candidate = proposer.apply(p_out, policy_text, log, pattern, research, round_, root)
    result.edit, result.candidate_path, result.outcome = edit, candidate, "candidate"
    result.refs["edit"] = save_record(edit, round_, root)
    return result


def run_agents(round_: int, conversations: list[ScoredConversation], policy_path: Path,
               log: DecisionLog, memory: Memory, llm: LLM, search: Callable[[str], list[dict]],
               root: Path) -> RoundResult:
    result = run_critic(round_, conversations, policy_path, log, memory, llm, root)
    if result.pattern is None:
        return result
    run_researcher(result, log, memory, llm, search, root)
    if result.outcome == "no_evidence":
        return result
    return run_proposer(result, log, memory, llm, root)


def stand_in_gate(result: RoundResult, decision: Literal["kept", "reverted"]) -> GateOutcome:
    """For development until the gatekeeper (J7) exists. It is labelled so in memory and the log."""
    assert result.edit is not None
    base = parse_policy(result.base_policy_path.read_text()).version
    kept_version = result.edit.new_policy_version.replace("-candidate", "")
    return GateOutcome(
        id=f"gate-r{result.round:02d}", written_by="gatekeeper", created_at=now(),
        parent_ids=[result.edit.id], round=result.round, edit_id=result.edit.id, decision=decision,
        before=Scores(empathy=0, relevance=0), after=Scores(empathy=0, relevance=0),
        safety_passed=decision == "kept", policy_in_force=kept_version if decision == "kept" else base,
        reason="Stand-in gate for development; no conversations were scored.")


def extras_for(record: Record, log: DecisionLog, root: Path) -> dict:
    """Values a memory sentence needs that live in other records."""
    if isinstance(record, ResearchNote):
        names = {r.pattern.pattern_id: r.pattern.name for r in log.rounds() if r.pattern}
        return {"pattern_name": names.get(record.pattern_id, "")}
    if isinstance(record, GateOutcome):
        ref = log.ref(record.edit_id)
        edit = load_record(ref, PolicyEdit, root) if ref else None
        event = next((r for r in log.rounds() if r.round == record.round), None)
        return {"rule_id": edit.rule_id if edit else None, "summary": edit.summary if edit else "",
                "noise": event.scores.noise if event and event.scores else None,
                "safety_cases": event.safety.cases if event and event.safety else None}
    return {}


def copy_to_memory(record: Record, ref: RecordRef, round_: int, memory: Memory, log: DecisionLog,
                   root: Path, attempt: int = 1, wait_s: float = 60) -> str:
    entry = entry_for(record, ref, round_, **extras_for(record, log, root))
    try:
        confirmed = memory.wait(memory.ingest(entry), timeout_s=wait_s)
    except (PastError, httpx.HTTPError):
        confirmed = False  # logged as pending, so retry_pending resends it
    status: Literal["confirmed", "pending"] = "confirmed" if confirmed else "pending"
    log.append(MemoryCopyEvent(round=round_, record_id=record.id, status=status, attempt=attempt))
    return status


MAX_COPY_ATTEMPTS = 3

# The log's size when retry_pending last finished, by log file. A call that finds the log unchanged
# is the same round start (prepare_round, then run_round), so it sends nothing again.
_RETRIED_AT: dict[Path, int] = {}


def _log_size(log: DecisionLog) -> int:
    return log.path.stat().st_size if log.path.exists() else 0


def _stop_if_stuck(log: DecisionLog) -> None:
    """Warn about copies pending after many attempts. It never raises: memory is a copy of the log,
    so a pending copy stays pending and is retried at the next round start."""
    import sys

    stuck = [f"{e.record_id} (round {e.round}, attempt {e.attempt})" for e in log.pending_copies()
             if e.attempt >= MAX_COPY_ATTEMPTS]
    if stuck:
        print(f"warning: memory copies still pending: {', '.join(stuck)}", file=sys.stderr)


def retry_pending(log: DecisionLog, memory: Memory, root: Path) -> None:
    """Run at the start of a round: send copies a crash cut short and resend pending ones, once per
    round start. A copy that fails stays pending; it never stops the loop or blocks a start."""
    if _RETRIED_AT.pop(log.path, None) == _log_size(log):
        return  # already tried at this round start
    if log.missing_copies() or log.pending_copies():
        from puddles.memory import prepare_training_audience

        prepare_training_audience(memory, root)  # the one try of this start needs the audience to exist
    pending = log.pending_copies()  # taken first: a missing copy that fails now waits for the next start
    for round_, missing in log.missing_copies():
        record = load_record(missing, replay_model(missing.id), root)
        if record is None:
            raise PastError(f"record file for {missing.id} (round {round_}) is missing or changed")
        copy_to_memory(record, missing, round_, memory, log, root)
    for event in pending:
        for kind, model in MODELS.items():
            round_event = next((r for r in log.rounds() if r.round == event.round), None)
            ref = round_event.records.get(kind) if round_event else None
            if ref and ref.id == event.record_id:
                record = load_record(ref, model, root)
                if record:
                    copy_to_memory(record, ref, event.round, memory, log, root, attempt=event.attempt + 1)
    _stop_if_stuck(log)
    _RETRIED_AT[log.path] = _log_size(log)


def replay_model(record_id: str) -> type[Record]:
    return MODELS["outcome" if record_id.startswith("gate-") else record_id.split("-")[0]]


def prepare_round(log: DecisionLog, memory: Memory, root: Path, start_policy: Path) -> Path:
    """Start of a round: undo an interrupted promotion, settle memory copies, run the replay check.
    Returns the policy file in force, taken from the log."""
    replay.undo_interrupted_promotion(log, root, start_policy)
    retry_pending(log, memory, root)
    return replay.check(log, root, start_policy)


def read_policy_text(path: Path, expected_sha256: Optional[str] = None,
                     changed: Optional[Exception] = None) -> str:
    """A policy file's text exactly as its bytes are: read once, decoded as UTF-8, no newline
    translation, so the hash of the text is the hash of the bytes on disk.

    With `expected_sha256`, the bytes must have that hash, or `changed` is raised; it is checked on
    the bytes, so a file that no longer decodes is a change, not a crash. Then the strict loader's
    rules (`puddles.graph.check_policy_file`): valid UTF-8, no byte-order mark, no carriage return.
    Anything else raises `PolicyError` before the caller writes anything.
    """
    changed = changed or PolicyChanged(f"{path} is not the text that was measured; nothing was written")
    data = Path(path).read_bytes()
    if expected_sha256 is not None and sha256_bytes(data) != expected_sha256:
        raise changed
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyError(f"{path}: not valid UTF-8 ({exc})") from exc
    if text.startswith("﻿"):
        raise PolicyError(f"{path}: starts with a byte-order mark")
    if "\r" in text:
        raise PolicyError(f"{path}: contains a carriage return; use Unix line endings")
    # A second look, as the gate hashes around a loader: the file as text mode reads it must be
    # exactly these bytes, so nothing translated and nothing moved between the two reads.
    if Path(path).read_text(encoding="utf-8") != text:
        raise changed
    return text


def gate_report_path(root: Path, round_: int) -> Path:
    """The report `puddles.gate.make_gate` writes for every round it is called on."""
    return Path(root) / "scores" / f"round-{round_:02d}-gate.json"


def gate_files(root: Path, round_: int) -> list[Path]:
    """Every file the real gate writes for a round (gate.md section 6)."""
    root = Path(root)
    return [gate_report_path(root, round_), root / "scores" / f"round-{round_:02d}-safety.json",
            root / "scores" / f"round-{round_:02d}-validation.jsonl",
            root / "conversations" / f"round-{round_:02d}-validation.jsonl"]


# (resolved root, round) for every round make_gate's gate was called on in this process. Only added to.
_GATED: set[tuple[str, int]] = set()


def mark_gated(root: Path, round_: int) -> None:
    """Called by the real gate as its first act: from here on the round is the gate's, whatever fails later."""
    _GATED.add((str(Path(root).resolve()), int(round_)))


def is_gated(root: Path, round_: int) -> bool:
    return (str(Path(root).resolve()), int(round_)) in _GATED


CANNOT_TELL = "cannot be looked at"


def _named(exc: OSError) -> str:
    return f"{type(exc).__name__}: {exc}"


def _absent(path: Path) -> Optional[str]:
    """None when `path` is confirmed not there (`os.lstat` raises FileNotFoundError); "" when it is
    there; otherwise why it cannot be told. Not being able to look is not the same as not being there."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"{path} {CANNOT_TELL} ({_named(exc)}), so it cannot be told that the gate left nothing there"
    return ""


def _folder_sign(path: Path) -> Optional[str]:
    """None when `path` is confirmed absent, or a folder whose entries can all be listed; else the sign."""
    why = _absent(path)
    if not why:
        if why is None:
            return None
        try:
            with os.scandir(path) as entries:
                for _ in entries:
                    pass
            return None
        except (NotADirectoryError, FileNotFoundError):  # a file, or a link to nothing
            return f"{path} is there and is not a folder, so nothing can be known about what the gate left"
        except OSError as exc:
            return (f"{path} is there but its entries {CANNOT_TELL} ({_named(exc)}), so nothing can be known "
                    "about what the gate left")
    return why


def gated_signs(result: RoundResult, root: Path) -> list[str]:
    """Every sign that the round is the real gate's, or that it cannot be told it is not (gate.md
    section 9); empty only when the absence of every sign is confirmed."""
    signs = []
    if any(v is not None for v in (result.scores, result.safety, result.base_sha256, result.candidate_sha256)):
        signs.append("the result carries the gate's measurements or hashes, so it went through the gate")
    if is_gated(root, result.round):
        signs.append("make_gate's gate was called for this root and round in this process")
    root = Path(root)
    signs += [s for s in (_folder_sign(p) for p in (root, root / "scores", root / "conversations")) if s]
    for path in gate_files(root, result.round):
        why = _absent(path)
        if why is not None:
            signs.append(why or f"{path} is there, written by the real gate")
    return signs


def finish_round(result: RoundResult, gate: Optional[GateOutcome], log: DecisionLog, memory: Memory,
                 root: Path, start_policy: Optional[Path] = None, scores: Optional[LogScores] = None,
                 safety: Optional[LogSafety] = None) -> RoundEvent:
    """Log the round, copy its records into memory, then run the replay check.

    It works in two halves. First it reads every file it needs once (the base policy, the
    candidate, the starting policy, the round's record files) and makes every check it can:
    the gate's hashes (whatever the decision, on the files' bytes), the candidate being the base
    with the logged edit applied (when scores or a safety result are filed), the records being
    the round's own, the round not being logged already, and the replay check run on what the
    log and the policy in force would be after this round. Only then does it write: the gate's
    record, the promoted policy and its .meta.json, the round event, the memory copies. So a
    check that fails leaves nothing behind.

    A promoted policy is written just before the round event; prepare_round removes it if a
    crash comes between the two. Without `start_policy` (the stage-0 graph runner) no .meta.json
    is written and no replay check runs. With a gate outcome, the gate's `result.scores` and
    `result.safety` go into the event; without one both stay empty.

    `scores` and `safety` are for a caller with a gate of its own (scripts/dataset_rounds.py). A
    round is the real gate's as soon as `make_gate`'s gate, or `run_gate`, is called for it, and it
    refuses them with ValueError, before anything is read or written, on any sign of that
    (`gated_signs`): one of `result.scores`, `result.safety`, `result.base_sha256`,
    `result.candidate_sha256` set; the gate called for this root and round in this process
    (`is_gated`); any file the gate writes for the round there, a dangling link included; or
    `<root>/scores` there and not a folder. Not being able to look is not the same as not being
    there: they are taken only when the absence of every sign is confirmed (`os.lstat` raising
    FileNotFoundError; `<root>`, `scores` and `conversations` each absent or a folder that can be
    listed; the decision log absent or there), and refused when it cannot be told. A gate outcome
    belongs to one round and one edit: one whose round or edit ID is not the result's is refused with ValueError, whatever else is passed.
    Left outside: another process's gate killed before it wrote any file for the round; a caller
    that deletes the gate's files; a real gate's outcome carried to another root in the same
    process (an outcome does not say which root it was measured on); and, each only when no gate
    file reached disk, two spellings of one root on a file system that ignores case, a reloaded
    `puddles.round`, or a gate run in a forked child. Otherwise they go into the event exactly as
    given; finish_round does not tie them to any file's contents, and the caller answers for them.
    """
    if gate is not None and (gate.round != result.round or result.edit is None
                             or gate.edit_id != result.edit.id):
        raise ValueError(f"the gate outcome {gate.id} is for round {gate.round} and {gate.edit_id}, not round "
                         f"{result.round} and {result.edit.id if result.edit else 'no edit'}; nothing was written")
    gated = any(v is not None for v in (result.scores, result.safety, result.base_sha256, result.candidate_sha256))
    if scores is not None or safety is not None:
        signs = gated_signs(result, root)
        # Whether the round is logged already is told from the log; a log that cannot be looked at tells nothing.
        why = _absent(log.path)
        if why:
            signs.append(f"the decision log {why}")
        if signs:
            known = [s for s in signs if CANNOT_TELL not in s]
            head = (f"round {result.round} is the real gate's" if known else
                    f"it cannot be told that round {result.round} is not the real gate's")
            raise ValueError(f"{head} ({'; '.join(signs)}); its measurements cannot be replaced by the "
                             "caller's scores or safety; nothing was written")
    # ---- read each file once, as its bytes, and check what was measured against those bytes ----
    base_text = read_policy_text(
        result.base_policy_path, result.base_sha256 if gate is not None else None,
        PolicyChanged(f"{result.base_policy_path} is not the text the before-scores were measured on; "
                      "nothing was written"))
    candidate_text: Optional[str] = None
    # Whatever the decision: a reverted event carries this hash beside the candidate's scores and safety.
    tested = gate is not None and result.candidate_sha256 is not None
    if gate is not None:
        assert result.candidate_path is not None and result.edit is not None
        candidate_text = read_policy_text(
            result.candidate_path, result.candidate_sha256,
            CandidateChanged(f"{result.candidate_path} is not the text the gate tested; nothing was written"))
    start_text = read_policy_text(Path(start_policy)) if start_policy is not None else None
    records: dict[str, Record] = {}
    for kind, model in MODELS.items():
        if kind in result.refs:
            loaded = load_record(result.refs[kind], model, root)
            if loaded is None:
                raise RecordChanged(f"the {kind} record {result.refs[kind].path} is missing or changed on disk")
            records[kind] = loaded

    # The event copies the pattern and the edit from the result: they must be the records just read.
    for kind in ("pattern", "edit"):
        mine = getattr(result, kind)
        if kind in records and (mine is None or records[kind].model_dump(mode="json") != mine.model_dump(mode="json")):
            raise RecordChanged(f"the {kind} record {result.refs[kind].path} is not the round's {kind}; "
                                "nothing was written")

    if any(r.round == result.round for r in log.rounds()):
        raise DecisionLogError(f"round {result.round} is already in {log.path}; nothing was written")
    # Only when a measurement is filed: a candidate the static checks stopped is by its nature not
    # the base with the edit, nothing was measured on it, and its revert must still be logged.
    if tested and (result.scores is not None or result.safety is not None):
        assert candidate_text is not None and result.edit is not None
        _check_logged_edit(base_text, candidate_text, result.edit, log, result.candidate_path)

    # ---- build the round in memory ----
    base_version = parse_policy(base_text).version
    policy = LogPolicy(base=base_version, base_sha256=sha256_text(base_text), in_force=base_version)
    outcome: Literal["kept", "reverted", "no_pattern", "no_evidence"]
    outcome = "no_evidence" if result.outcome == "no_evidence" else "no_pattern"
    reason = result.reason
    promoted: Optional[tuple[str, str]] = None
    refs = dict(result.refs)
    gate_bytes: Optional[bytes] = None
    if gate is not None and candidate_text is not None:
        # The same bytes save_record would write, so the ref is known before anything is written.
        gate_bytes = (gate.model_dump_json(indent=2) + "\n").encode("utf-8")
        refs["outcome"] = RecordRef(id=gate.id, path=record_path(result.round, gate.id),
                                    sha256=sha256_bytes(gate_bytes))
        records["outcome"] = GateOutcome.model_validate_json(gate_bytes)
        outcome, reason = gate.decision, gate.reason
        candidate = parse_policy(candidate_text)
        if gate.decision == "kept":
            candidate.version = gate.policy_in_force
            text = render_policy(candidate)  # rendered from the one reading of the candidate
            promoted = (candidate.version, text)
            policy.new, policy.new_sha256, policy.in_force = candidate.version, sha256_text(text), candidate.version
        else:
            policy.new, policy.new_sha256 = candidate.version, sha256_text(candidate_text)
    if result.unscored:
        reason = f"{reason or outcome} {unscored_sentence(result.unscored)}"
    p = result.pattern
    event = RoundEvent(
        round=result.round, outcome=outcome, records=refs, policy=policy, reason=reason or outcome,
        pattern=LogPattern(id=p.id, same_as_pattern_id=p.same_as_pattern_id, name=p.name, seen_in=p.seen_in,
                           out_of=p.out_of, status_before=result.status_before) if p else None,
        edit=LogEdit(id=result.edit.id, action=result.edit.action or "add", rule_id=result.edit.rule_id or "",
                     rule_text=result.edit.rule_text) if result.edit and gate else None,
        # Gated: the gate's measurements, with a gate outcome only. Ungated: the caller's, as given.
        scores=(result.scores if gate is not None else None) if gated else scores,
        safety=(result.safety if gate is not None else None) if gated else safety)
    meta = None
    if start_policy is not None and start_text is not None:
        meta = _replay_before_writing(log, root, start_policy, start_text, event, promoted)

    # ---- write ----
    if gate_bytes is not None:
        path = root / refs["outcome"].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gate_bytes)
        result.refs["outcome"] = refs["outcome"]
    if promoted:
        version, text = promoted
        # UTF-8 with no newline translation: exactly the bytes new_sha256 is the hash of.
        replay.policy_path(root, version).write_text(text, encoding="utf-8", newline="")
        if meta is not None:
            replay.meta_path(root, version).write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8",
                                                       newline="")
    log.append(event)
    for kind in MODELS:
        if kind in records:
            copy_to_memory(records[kind], refs[kind], result.round, memory, log, root)
    if start_policy is not None:
        replay.check(log, root, start_policy)  # the files as written; checked once already before writing
    return event


def _check_logged_edit(base_text: str, candidate_text: str, edit: PolicyEdit, log: DecisionLog,
                       candidate_path: Optional[Path]) -> None:
    """The candidate the gate tested must be the base with the logged edit applied, by the
    proposer's own function: the same Safety section, the same Conversation rules by ID and text.
    Otherwise its scores would be filed under an edit the conversations did not run on."""
    action = edit.action or "add"
    try:
        base, candidate = parse_policy(base_text), parse_policy(candidate_text)
        expected, rule_id = apply_edit(base, action, None if action == "add" else edit.rule_id, edit.rule_text,
                                       candidate.version, used_ids=log.used_rule_ids())
    except PolicyEditError as exc:
        raise PolicyChanged(f"{candidate_path} is not the base policy with {edit.id} applied ({exc}); "
                            "nothing was written") from exc
    if rule_id != edit.rule_id or safety_section(expected) != safety_section(candidate) \
            or expected.conversation != candidate.conversation:
        raise PolicyChanged(f"{candidate_path} is not the base policy with {edit.id} applied; nothing was written")


def _replay_before_writing(log: DecisionLog, root: Path, start_path: Path, start_text: str, event: RoundEvent,
                           promoted: Optional[tuple[str, str]]) -> Optional[PolicyMeta]:
    """The replay check on the log and the policy in force as they would be after this round.

    It mirrors `puddles.replay.check`, which reads only files, so that a round that would fail it
    is refused before anything is written. For a kept round it returns the promoted policy's
    .meta.json, built from the same texts.
    """
    rounds = log.rounds() + [event]
    start = parse_policy(start_text)
    previous, previous_sha = start.version, sha256_text(start_text)
    for r in rounds:
        if (r.policy.base, r.policy.base_sha256) != (previous, previous_sha):
            raise replay.ReplayError(f"round {r.round} starts from {r.policy.base}, but {previous} was in force")
        if r.outcome == "kept":
            if not (r.policy.new and r.policy.new_sha256 and r.policy.in_force == r.policy.new):
                raise replay.ReplayError(f"round {r.round} was kept but names no new policy in force")
            previous, previous_sha = r.policy.new, r.policy.new_sha256
        elif r.policy.in_force != r.policy.base:
            raise replay.ReplayError(f"round {r.round} was {r.outcome} but changed the policy in force")
    if promoted is None:
        # Nothing new comes into force: the files the check reads afterwards are the ones it reads now.
        replay.check(log, root, start_path)
        return None
    version, text = promoted
    policy = parse_policy(text)
    expected = replay.expected_rules(rounds, start)
    if set(policy.conversation) != set(expected):
        raise replay.ReplayError(f"{version} would have rules {sorted(policy.conversation)}; "
                                 f"the log gives {sorted(expected)}")
    changed = [rid for rid, t in expected.items() if sha256_text(policy.conversation[rid]) != sha256_text(t)]
    if changed:
        raise replay.ReplayError(f"{version}: the text of {changed} would differ from the log")
    if sha256_text(safety_section(policy)) != sha256_text(safety_section(start)):
        raise replay.ReplayError(f"{version}: the Safety section would differ from the starting policy's")
    return replay.expected_meta(rounds, start, policy, text)


def memory_trace(log: DecisionLog, root: Path) -> list[dict]:
    """For each record copied into memory: who wrote it and which agents later read it."""
    import json

    reads_path = root / READS_PATH
    reads = [json.loads(line) for line in reads_path.read_text().splitlines()] if reads_path.exists() else []
    rows = []
    for event in log.rounds():
        for kind, ref in event.records.items():
            record = load_record(ref, MODELS[kind], root)
            readers = sorted({r["reader"] for r in reads if ref.id in r["record_ids"]})
            rows.append({"record": ref.id, "round": event.round,
                         "written_by": record.written_by if record else "?", "read_by": readers})
    return rows
