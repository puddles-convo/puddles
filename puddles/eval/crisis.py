"""The crisis path data: keywords, the checker's answer format and the fixed lines."""
import json
import re
from pathlib import Path
from typing import Literal, Optional

REPO = Path(__file__).resolve().parents[2]

_LEVELS = ("none", "concern", "crisis")
_CATEGORIES = ("self", "danger", "unsure")


def normalise(text: str) -> str:
    text = text.lower().replace("'", "").replace("\u2019", "").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_keywords(path: Path = REPO / "safety" / "keywords-v1.txt") -> list[str]:
    keywords = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        keyword = normalise(line)
        if not keyword or line.lstrip().startswith("#"):
            continue
        if keyword in keywords:
            raise ValueError(f"duplicate keyword after normalisation: {keyword!r}")
        keywords.append(keyword)
    return keywords


def keyword_match(text: str, keywords: list[str]) -> Optional[str]:
    haystack = normalise(text)
    for keyword in keywords:
        if keyword in haystack:
            return keyword
    return None


def parse_checker_line(
    line: str,
) -> tuple[Literal["none", "concern", "crisis"], Optional[Literal["self", "danger", "unsure"]], str]:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != 3:
        raise ValueError(f"checker line must have exactly 3 parts, got {len(parts)}: {line!r}")
    level, category, reason = parts
    if level not in _LEVELS:
        raise ValueError(f"unknown level {level!r}: {line!r}")
    if level == "crisis":
        if category not in _CATEGORIES:
            raise ValueError(f"crisis needs a category in {_CATEGORIES}, got {category!r}: {line!r}")
        return "crisis", category, reason  # type: ignore[return-value]
    if category != "-":
        raise ValueError(f"{level} must have a dash category, got {category!r}: {line!r}")
    return level, None, reason  # type: ignore[return-value]


def load_crisis_path(path: Path = REPO / "safety" / "crisis-path-v1.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
