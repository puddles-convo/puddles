"""Running simulated conversations, one at a time or as a batch."""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Sequence

from loop.errors import ConversationFailed, HelperError, LLMError, SimulatedUserError
from loop.helper import helper_reply
from loop.llm import ChatClient
from loop.policy import Policy
from loop.simulated_user import next_user_line
from schemas import ConversationRecord, Scenario, Turn


def scenario_seed(seed: int | None, scenario_id: str) -> int | None:
    """A stable per-scenario seed below 2**31, or None when seed is None."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"{seed}:{scenario_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def load_scenarios(path: str | PathLike[str]) -> list[Scenario]:
    """Read one Scenario per non-blank line of a JSON Lines file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Scenario.model_validate_json(line) for line in lines if line.strip()]


TAG_RE = re.compile(r"[a-z][a-z0-9]*")


def conversation_id(round: int, scenario_id: str, tag: str | None = None) -> str:
    """conv-rNN-<scenario>, or conv-rNN-<tag>-<scenario> for a batch that is not the round's training batch.

    The tag keeps a second batch in the same round, such as the gate's validation run, from
    reusing the IDs of the first.
    """
    if tag is None:
        return f"conv-r{round:02d}-{scenario_id}"
    if not isinstance(tag, str) or not TAG_RE.fullmatch(tag):
        raise ValueError(f"tag must be lower-case letters and digits, starting with a letter: {tag!r}")
    return f"conv-r{round:02d}-{tag}-{scenario_id}"


def run_conversation(helper_client: ChatClient, user_client: ChatClient, policy: Policy,
                     scenario: Scenario, *, round: int, max_turns: int = 10,
                     seed: int | None = None, now: datetime | None = None,
                     tag: str | None = None) -> ConversationRecord:
    """Let the simulated user and the helper talk until one stops; return the record."""
    record_id = conversation_id(round, scenario.scenario_id, tag)
    conv_seed = scenario_seed(seed, scenario.scenario_id)
    turns: list[Turn] = []
    replies = 0
    try:
        while True:
            line = next_user_line(user_client, scenario, turns, seed=conv_seed)
            if line is None:
                if not turns:
                    raise ConversationFailed(scenario.scenario_id,
                                             "simulated user ended before saying anything")
                break
            turns.append(Turn(speaker="user", text=line))
            if replies >= max_turns:
                break
            reply = helper_reply(helper_client, policy, turns, session_id=record_id,
                                 turn=replies + 1, seed=conv_seed)
            turns.append(Turn(speaker="helper", text=reply.text))
            replies += 1
            if replies >= max_turns:
                break
    except (LLMError, HelperError, SimulatedUserError) as exc:
        raise ConversationFailed(scenario.scenario_id, f"{type(exc).__name__}: {exc}") from exc
    return ConversationRecord(
        id=record_id,
        written_by="helper",
        created_at=now or datetime.now(timezone.utc),
        source="simulated",
        scenario_id=scenario.scenario_id,
        split=scenario.split,
        policy_version=policy.version,
        turns=turns,
    )


@dataclass(frozen=True)
class Failure:
    """A conversation that could not be completed."""

    scenario_id: str
    reason: str


@dataclass
class BatchResult:
    """The records and failures of one batch."""

    records: list[ConversationRecord]
    failures: list[Failure]
    round: int

    def summary(self) -> dict[str, int]:
        return {"round": self.round, "conversations": len(self.records),
                "failed": len(self.failures)}


def failures_path(out_path: str | PathLike[str]) -> Path:
    """The failures file next to out_path: its last suffix becomes .failures.jsonl."""
    return Path(out_path).with_suffix(".failures.jsonl")


def conversation_workers() -> int:
    """PUDDLES_CONVERSATION_WORKERS: conversations run at once in a batch (default 1, one after another)."""
    raw = os.environ.get("PUDDLES_CONVERSATION_WORKERS", "").strip()
    if not raw:
        return 1
    if not raw.isdigit() or not 1 <= int(raw) <= 32:
        raise ValueError(f"PUDDLES_CONVERSATION_WORKERS={raw!r}: must be a whole number from 1 to 32")
    return int(raw)


def run_batch(helper_client: ChatClient, user_client: ChatClient, policy: Policy,
              scenarios: Sequence[Scenario], *, round: int, max_turns: int = 10,
              seed: int | None = None, out_path: str | PathLike[str] | None = None,
              now: datetime | None = None, tag: str | None = None) -> BatchResult:
    """Run every scenario in order; a failure is recorded and the batch goes on.

    `tag` goes into every conversation ID (see `conversation_id`). The file at `out_path` is
    replaced, so a tagged batch needs its own path as well.
    """
    conversation_id(round, "", tag)  # a bad tag is the caller's mistake, not a failed conversation
    result = BatchResult(records=[], failures=[], round=round)

    def one(scenario: Scenario):
        try:
            return run_conversation(helper_client, user_client, policy, scenario,
                                    round=round, max_turns=max_turns, seed=seed, now=now, tag=tag)
        except ConversationFailed as exc:
            return Failure(exc.scenario_id, exc.reason)

    workers = conversation_workers()
    if workers > 1 and len(scenarios) > 1:
        # Each conversation runs in a copy of this context, so the usage meter and the trace
        # still see it; results are kept in scenario order, so files are as a sequential run's.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # the copies are made here, in the calling thread: a copy made inside a worker is empty
            contexts = [contextvars.copy_context() for _ in scenarios]
            outcomes = list(pool.map(lambda pair: pair[0].run(one, pair[1]), zip(contexts, scenarios)))
    else:
        outcomes = [one(sc) for sc in scenarios]
    for outcome in outcomes:
        (result.failures if isinstance(outcome, Failure) else result.records).append(outcome)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(r.model_dump_json() + "\n" for r in result.records),
                       encoding="utf-8")
        if result.failures:
            failures_path(out).write_text(
                "".join(json.dumps({"scenario_id": f.scenario_id, "reason": f.reason}) + "\n"
                        for f in result.failures),
                encoding="utf-8")
        else:
            failures_path(out).unlink(missing_ok=True)
    return result
