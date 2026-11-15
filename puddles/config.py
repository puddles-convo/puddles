"""The run's config file: the prices table, the output limits and the loop's settings.

The file is `config/run.toml`, or the path in the environment variable PUDDLES_CONFIG.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loop.errors import ConfigError
from puddles.usage import ModelPrice, Prices
from schemas import Scores

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "run.toml"
DEFAULT_LIMIT = "default"


@dataclass
class RunConfig:
    prices: Prices
    output_limits: dict[str, int]  # agent name -> most tokens one answer may have; "default" is required


def _number(value: Any, what: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError(f"{path}: {what} must be a number, zero or more")
    return float(value)


def load_config(path: Optional[Path | str] = None) -> RunConfig:
    path = Path(path or os.environ.get("PUDDLES_CONFIG") or DEFAULT_PATH)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the config file: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from exc

    table = data.get("prices", {})
    models = {}
    for name, row in table.get("models", {}).items():
        if not isinstance(row, dict) or set(row) != {"input_usd_per_million", "output_usd_per_million"}:
            raise ConfigError(
                f"{path}: prices.models.{name!r} needs input_usd_per_million and "
                "output_usd_per_million, and nothing else"
            )
        models[name] = ModelPrice(
            _number(row["input_usd_per_million"], f"the input price of {name!r}", path),
            _number(row["output_usd_per_million"], f"the output price of {name!r}", path),
        )
    exa = table.get("exa_search_usd")
    prices = Prices(models=models, exa_search_usd=None if exa is None else _number(exa, "prices.exa_search_usd", path))

    limits = data.get("output_limits", {})
    if DEFAULT_LIMIT not in limits:
        raise ConfigError(f"{path}: output_limits.{DEFAULT_LIMIT} is missing; every model call needs a limit")
    for agent, limit in limits.items():
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ConfigError(f"{path}: output_limits.{agent} must be a whole number, one or more")
    return RunConfig(prices=prices, output_limits=dict(limits))


def output_limit(limits: dict[str, int], agent: Optional[str]) -> int:
    """The agent's limit, or the default when the agent has none of its own."""
    return limits.get(agent or DEFAULT_LIMIT, limits[DEFAULT_LIMIT])


# ---------------------------------------------------------------------------------------------
# The loop's settings: the [loop] table of the same file, read by scripts/run_loop.py.
# ---------------------------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """What the overnight loop needs. `None` means "not set": the caps that are None never
    fire, and the noise fields stay empty until stage 5's real runs fill them in."""

    max_rounds: Optional[int] = None
    conversations_per_round: int = 5
    conversations_for_critic: Optional[int] = None
    max_turns: int = 10
    spend_limit_usd: Optional[float] = None
    round_cost_estimate_usd: Optional[float] = None  # stage 3's measurement, scaled to these settings
    seed: int = 7
    validation_ids: str = "fixtures/validation_ids.txt"
    train_ids: str = "fixtures/train_ids.txt"
    noise: Optional[float] = None
    noise_measured: bool = False
    current_scores: Optional[Scores] = None
    judge_version: Optional[str] = None


def _whole(value: Any, what: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{path}: loop.{what} must be a whole number, one or more")
    return value


def load_loop_config(path: Optional[Path | str] = None) -> LoopConfig:
    """The [loop] table. Missing keys take the defaults above; wrong types are refused."""
    path = Path(path or os.environ.get("PUDDLES_CONFIG") or DEFAULT_PATH)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the config file: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from exc

    table = data.get("loop", {})
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: [loop] must be a table")

    def opt_number(key: str) -> Optional[float]:
        value = table.get(key)
        return None if value is None else _number(value, f"loop.{key}", path)

    current = table.get("current_scores")
    try:
        scores = None if current is None else Scores.model_validate(current)
    except Exception as exc:
        raise ConfigError(f"{path}: loop.current_scores is not a Scores table: {exc}") from exc

    noise = opt_number("noise")
    if noise is not None and noise < 0:
        raise ConfigError(f"{path}: loop.noise must be zero or more")
    max_rounds = table.get("max_rounds")
    seed = table.get("seed", 7)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"{path}: loop.seed must be a whole number")
    return LoopConfig(
        max_rounds=None if max_rounds is None else _whole(max_rounds, "max_rounds", path),
        conversations_per_round=_whole(table.get("conversations_per_round", 5), "conversations_per_round", path),
        conversations_for_critic=(
            None
            if table.get("conversations_for_critic") is None
            else _whole(table["conversations_for_critic"], "conversations_for_critic", path)
        ),
        max_turns=_whole(table.get("max_turns", 10), "max_turns", path),
        spend_limit_usd=opt_number("spend_limit_usd"),
        round_cost_estimate_usd=opt_number("round_cost_estimate_usd"),
        seed=seed,
        validation_ids=str(table.get("validation_ids", "fixtures/validation_ids.txt")),
        train_ids=str(table.get("train_ids", "fixtures/train_ids.txt")),
        noise=noise,
        noise_measured=bool(table.get("noise_measured", False)),
        current_scores=scores,
        judge_version=table.get("judge_version"),
    )
