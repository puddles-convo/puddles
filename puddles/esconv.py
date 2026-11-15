"""ESConv (issue #2): scenarios by split, and the real conversations with the users' own ratings.

Downloaded from Hugging Face at run time and kept in a local cache that git ignores. The
licence is CC BY-NC 4.0, so no row is ever committed. An ID is the split plus the row's
position in the published file, so it is the same on every machine.
"""
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

import httpx

from schemas import ConversationRecord, Scenario, Turn

EsconvSplit = Literal["train", "validation", "test"]
BASE_URL = "https://huggingface.co/datasets/thu-coai/esconv/resolve/main"
FILES: dict[str, str] = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}
SIZES: dict[str, int] = {"train": 910, "validation": 195, "test": 195}  # the published splits
DEFAULT_CACHE = "runs/cache/esconv"  # runs/ is ignored by git
CACHE_VARIABLE = "ESCONV_CACHE"
SPEAKERS = {"usr": "user", "sys": "helper"}
PUBLISHED = datetime(2021, 6, 1, tzinfo=timezone.utc)  # a fixed date, so a loaded record never changes
Fetch = Callable[[str], str]


class EsconvError(RuntimeError):
    pass


@dataclass(frozen=True)
class UserRatings:
    """What the person rated their own conversation, 1 to 5. Emotion is intensity: 5 is worst."""

    empathy: int
    relevance: int
    emotion_before: int
    emotion_after: int


@dataclass(frozen=True)
class RatedConversation:
    record: ConversationRecord
    ratings: UserRatings


def _download(url: str) -> str:
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.text


def _check_split(split: str, allow_test: bool) -> None:
    if split not in FILES:
        raise EsconvError(f"unknown split {split!r}; use train, validation or test")
    if split == "test" and not allow_test:
        raise EsconvError("the test split is loaded only with allow_test=True; "
                          "it is for the final evaluation and nothing else")


def load_rows(split: EsconvSplit, *, allow_test: bool = False, cache_dir: Optional[Path] = None,
              fetch: Optional[Fetch] = None) -> list[dict]:
    """The split's rows in published order, from the cache or, the first time, from Hugging Face."""
    _check_split(split, allow_test)
    cache = Path(cache_dir or os.environ.get(CACHE_VARIABLE) or DEFAULT_CACHE)
    path = cache / FILES[split]
    if not path.exists():
        text = (fetch or _download)(f"{BASE_URL}/{FILES[split]}")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != SIZES[split]:
        raise EsconvError(f"{path} has {len(rows)} rows; the published {split} split has {SIZES[split]}. "
                          "Delete the file to download it again.")
    return rows


def row_id(split: str, index: int) -> str:
    return f"esconv-{split}-{index:04d}"


def _score(value: object) -> Optional[int]:
    """A rating as an integer from 1 to 5, or None when the row has none."""
    try:
        number = int(str(value).strip())
    except ValueError:
        return None
    return number if 1 <= number <= 5 else None


def _seeker(row: dict) -> dict:
    return (row.get("survey_score") or {}).get("seeker") or {}


def scenarios(split: EsconvSplit, *, allow_test: bool = False, cache_dir: Optional[Path] = None,
              fetch: Optional[Fetch] = None) -> list[Scenario]:
    """One scenario per row that says how intense the person's distress was at the start."""
    out = []
    for index, row in enumerate(load_rows(split, allow_test=allow_test, cache_dir=cache_dir, fetch=fetch)):
        intensity = _score(_seeker(row).get("initial_emotion_intensity"))
        if intensity is None or not str(row.get("situation", "")).strip():
            continue  # never guessed
        out.append(Scenario(scenario_id=row_id(split, index), split=split,
                            emotion_type=row["emotion_type"], problem_type=row["problem_type"],
                            situation=row["situation"].strip(), initial_intensity=intensity))
    return out


def real_conversations(split: EsconvSplit, *, allow_test: bool = False,
                       cache_dir: Optional[Path] = None,
                       fetch: Optional[Fetch] = None) -> list[RatedConversation]:
    """The real conversations whose users gave all four ratings. About one row in ten gave none."""
    out = []
    for index, row in enumerate(load_rows(split, allow_test=allow_test, cache_dir=cache_dir, fetch=fetch)):
        seeker = _seeker(row)
        values = [_score(seeker.get(key)) for key in
                  ("empathy", "relevance", "initial_emotion_intensity", "final_emotion_intensity")]
        if any(v is None for v in values):
            continue
        empathy, relevance, before, after = (v for v in values if v is not None)
        turns = [Turn(speaker=SPEAKERS[d["speaker"]], text=d["text"].strip())  # type: ignore[arg-type]
                 for d in row["dialog"] if d.get("text", "").strip()]
        record = ConversationRecord(
            id=row_id(split, index), written_by="esconv", created_at=PUBLISHED, source="real",
            scenario_id=row_id(split, index), split=split, policy_version="esconv", turns=turns)
        out.append(RatedConversation(record, UserRatings(empathy, relevance, before, after)))
    return out
