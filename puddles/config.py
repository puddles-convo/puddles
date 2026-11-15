"""The run's config file: the prices table and the output limit of each agent.

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
            raise ConfigError(f"{path}: prices.models.{name!r} needs input_usd_per_million and "
                              "output_usd_per_million, and nothing else")
        models[name] = ModelPrice(_number(row["input_usd_per_million"], f"the input price of {name!r}", path),
                                  _number(row["output_usd_per_million"], f"the output price of {name!r}", path))
    exa = table.get("exa_search_usd")
    prices = Prices(models=models,
                    exa_search_usd=None if exa is None else _number(exa, "prices.exa_search_usd", path))

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
