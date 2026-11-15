"""Token and cost accounting: the prices table, the usage log and the tally per agent.

Every model call and every Exa request writes to `<root>/log/usage.jsonl` twice: a `sent` line
before the request goes out, holding the most it can cost, and a `done` line when the answer
arrives, holding what it did cost. The file is append-only. Spend is the sum of the `done`
lines, plus the `sent` figure of every call that has no `done` line, because a request cut off
by a crash or an error may still be billed.

A `Meter` writes the lines and keeps the round's tally. Code that makes a call finds the meter
and the agent's name with `current()`; the graph sets them with `metering(meter, agent)`.
"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from loop.errors import ConfigError
from loop.llm import CallUsage

USAGE_PATH = "log/usage.jsonl"
# A client with no `model` attribute is a scripted stand-in, not a model. Its calls are logged
# under this name and cost nothing. Every real model needs a row in the prices table.
FAKE_MODEL = "fake"
SEARCH_KIND, MODEL_KIND = "search", "model"


class MissingPrice(ConfigError):
    """A model or the Exa search has no price in the run's config file."""


@dataclass(frozen=True)
class ModelPrice:
    input_usd_per_million: float
    output_usd_per_million: float


@dataclass
class Prices:
    """Dollars per million tokens for each model, and dollars per Exa search. Nothing is assumed."""

    models: dict[str, ModelPrice] = field(default_factory=dict)
    exa_search_usd: Optional[float] = None

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        if model == FAKE_MODEL:
            return 0.0
        price = self.models.get(model)
        if price is None:
            raise MissingPrice(f"no price for model {model!r} in the prices table")
        return (tokens_in * price.input_usd_per_million + tokens_out * price.output_usd_per_million) / 1e6

    def search(self) -> float:
        if self.exa_search_usd is None:
            raise MissingPrice("no price for an Exa search (prices.exa_search_usd) in the prices table")
        return self.exa_search_usd

    def check(self, models: Iterable[str], search: bool = False) -> None:
        """Raise MissingPrice, naming everything that has no price."""
        missing = sorted({f"model {m!r}" for m in models if m != FAKE_MODEL and m not in self.models})
        if search and self.exa_search_usd is None:
            missing.append("an Exa search (prices.exa_search_usd)")
        if missing:
            raise MissingPrice(f"no price for {', '.join(missing)} in the prices table")


def model_name(client: Any) -> str:
    """The client's model, or FAKE_MODEL for a scripted stand-in."""
    name = getattr(client, "model", None)
    return name if isinstance(name, str) and name else FAKE_MODEL


@dataclass
class AgentUsage:
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    estimated_calls: int = 0
    searches: int = 0
    cost_usd: float = 0.0
    first_answer_failed: int = 0
    failed: int = 0


@dataclass(frozen=True)
class Call:
    """A request that has been logged as sent."""

    id: str
    agent: str
    kind: str
    model: Optional[str]
    max_cost_usd: float


class Meter:
    """Writes the usage log for one root and keeps the tally of the current round."""

    def __init__(self, root: Path | str, prices: Optional[Prices] = None, round: int = 0) -> None:
        self.path = Path(root) / USAGE_PATH
        self.prices = prices if prices is not None else Prices()
        self.round = round
        self.usage: dict[str, AgentUsage] = {}

    def start_round(self, round_: int) -> None:
        self.round = round_
        self.usage = {}

    def _agent(self, agent: str) -> AgentUsage:
        return self.usage.setdefault(agent, AgentUsage())

    def _write(self, line: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamped = {**line, "at": datetime.now(timezone.utc).isoformat()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(stamped) + "\n")

    def sent(self, agent: str, model: str, prompt_tokens: int, max_tokens: int) -> Call:
        """Log a model call before it goes out. Raises MissingPrice if the model has no price."""
        most = self.prices.cost(model, prompt_tokens, max_tokens)
        call = Call(uuid.uuid4().hex, agent, MODEL_KIND, model, most)
        self._write({"event": "sent", "call_id": call.id, "kind": MODEL_KIND, "round": self.round,
                     "agent": agent, "model": model, "prompt_tokens_estimate": prompt_tokens,
                     "max_tokens": max_tokens, "max_cost_usd": most})
        return call

    def done(self, call: Call, usage: CallUsage) -> float:
        """Log the answer to a model call and add it to the tally. Returns its cost in dollars."""
        assert call.model is not None
        cost = self.prices.cost(call.model, usage.tokens_in, usage.tokens_out)
        self._write({"event": "done", "call_id": call.id, "kind": MODEL_KIND, "round": self.round,
                     "agent": call.agent, "model": call.model, "tokens_in": usage.tokens_in,
                     "tokens_out": usage.tokens_out, "estimated": usage.estimated, "cost_usd": cost})
        tally = self._agent(call.agent)
        tally.tokens_in += usage.tokens_in
        tally.tokens_out += usage.tokens_out
        tally.calls += 1
        tally.estimated_calls += int(usage.estimated)
        tally.cost_usd += cost
        return cost

    def note_done(self, agent: str, model: str, tokens_in: int, tokens_out: int,
                  estimated: bool = False) -> float:
        """Log a finished call reported after the fact, like the judge panel's usage list.

        There is no `sent` line to match, so the call_id is fresh; `spend` counts `done` lines
        either way. Returns the cost in dollars.
        """
        cost = self.prices.cost(model, tokens_in, tokens_out)
        self._write({"event": "done", "call_id": uuid.uuid4().hex, "kind": MODEL_KIND,
                     "round": self.round, "agent": agent, "model": model,
                     "tokens_in": tokens_in, "tokens_out": tokens_out,
                     "estimated": estimated, "cost_usd": cost})
        tally = self._agent(agent)
        tally.tokens_in += tokens_in
        tally.tokens_out += tokens_out
        tally.calls += 1
        tally.estimated_calls += int(estimated)
        tally.cost_usd += cost
        return cost

    def search_sent(self, agent: str) -> Call:
        """Log one Exa request before it goes out. Raises MissingPrice if a search has no price."""
        price = self.prices.search()
        call = Call(uuid.uuid4().hex, agent, SEARCH_KIND, None, price)
        self._write({"event": "sent", "call_id": call.id, "kind": SEARCH_KIND, "round": self.round,
                     "agent": agent, "max_cost_usd": price})
        return call

    def search_done(self, call: Call, error: Optional[str] = None) -> float:
        """Log the end of one Exa request. A request that failed is still counted at the full price."""
        line = {"event": "done", "call_id": call.id, "kind": SEARCH_KIND, "round": self.round,
                "agent": call.agent, "cost_usd": call.max_cost_usd}
        if error:
            line["error"] = error
        self._write(line)
        tally = self._agent(call.agent)
        tally.searches += 1
        tally.cost_usd += call.max_cost_usd
        return call.max_cost_usd

    def first_answer_failed(self, agent: str) -> None:
        self._agent(agent).first_answer_failed += 1

    def failed(self, agent: str, reason: str = "") -> None:
        """Log that an agent gave no usable answer after its retry."""
        self._agent(agent).failed += 1
        self._write({"event": "failed", "round": self.round, "agent": agent, "reason": reason})

    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage.values())

    def metrics(self) -> dict[str, float]:
        """The round's totals, named as they are logged to MLflow."""
        out: dict[str, float] = {"cost_usd": self.cost_usd()}
        for agent, u in sorted(self.usage.items()):
            out[f"first_answer_failed.{agent}"] = u.first_answer_failed
            if u.calls:
                out[f"tokens_in.{agent}"] = u.tokens_in
                out[f"tokens_out.{agent}"] = u.tokens_out
                out[f"calls.{agent}"] = u.calls
                out[f"estimated_calls.{agent}"] = u.estimated_calls
            if u.searches:
                out[f"searches.{agent}"] = u.searches
        return out


def read_usage(path: Path | str) -> list[dict]:
    """Every whole line of a usage file. A last line cut short by a crash is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            line = json.loads(raw)
        except ValueError:
            continue
        if isinstance(line, dict):
            lines.append(line)
    return lines


def spend_by_round(lines: Iterable[dict]) -> dict[int, float]:
    """Dollars per round from usage lines: every `done` line, plus each `sent` line with no
    `done` line, at its estimate. Lines with no round count as round 0."""
    lines = list(lines)
    finished = {line.get("call_id") for line in lines if line.get("event") == "done"}
    per: dict[int, float] = {}
    for line in lines:
        if line.get("event") == "done":
            amount = line.get("cost_usd", 0.0)
        elif line.get("event") == "sent" and line.get("call_id") not in finished:
            amount = line.get("max_cost_usd", 0.0)
        else:
            continue
        round_ = line.get("round") or 0
        per[round_] = per.get(round_, 0.0) + amount
    return per


def spend(path: Path | str) -> float:
    """Dollars spent: every `done` line, plus the `sent` estimate of each call with no `done` line."""
    return sum(spend_by_round(read_usage(path)).values())


_CURRENT: ContextVar[tuple[Optional[Meter], Optional[str]]] = ContextVar("puddles_usage", default=(None, None))


@contextmanager
def metering(meter: Optional[Meter], agent: Optional[str] = None) -> Iterator[Optional[Meter]]:
    """Inside the block, model calls and searches are logged to `meter` under `agent`."""
    token = _CURRENT.set((meter, agent))
    try:
        yield meter
    finally:
        _CURRENT.reset(token)


def current() -> tuple[Optional[Meter], Optional[str]]:
    """The meter and agent name set by `metering`, or (None, None)."""
    return _CURRENT.get()


def note_first_answer_failed(agent: str) -> None:
    meter, _ = current()
    if meter is not None:
        meter.first_answer_failed(agent)


def note_failed(agent: str, reason: str = "") -> None:
    meter, _ = current()
    if meter is not None:
        meter.failed(agent, reason)
