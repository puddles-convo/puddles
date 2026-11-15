"""Load the ESConv dataset into puddles scenarios and conversation records."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Optional
from urllib.request import urlopen

from pydantic import Field

from schemas import ConversationRecord, Scenario, Strict, Turn

ESCONV = "thu-coai/esconv"
ESCONV_REVISION = "32c54c15e7651cd68dfab18eee39f63ff0cc1e82"
SPLITS = {"train": ("train.txt", 910), "validation": ("valid.txt", 195), "test": ("test.txt", 195)}
REAL_POLICY_VERSION = "esconv-human"

_SPEAKERS: dict[str, Literal["user", "helper"]] = {"usr": "user", "sys": "helper"}
_RATING_KEYS = {
    "empathy": "empathy",
    "relevance": "relevance",
    "emotion_before": "initial_emotion_intensity",
    "emotion_after": "final_emotion_intensity",
}
_VALID_SCORES = {"1", "2", "3", "4", "5"}


def download(repo: str, revision: str, filename: str) -> str:
    root = Path(os.getenv("PUDDLES_CACHE", Path.home() / ".cache" / "puddles"))
    path = root / repo / revision / filename
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp.write(data)
        Path(temp.name).replace(path)
    return path.read_text(encoding="utf-8-sig")


class SeekerRatings(Strict):
    """Post-conversation survey scores the seeker gave, None when unreported."""

    empathy: Optional[int] = Field(default=None, ge=1, le=5)
    relevance: Optional[int] = Field(default=None, ge=1, le=5)
    emotion_before: Optional[int] = Field(default=None, ge=1, le=5)
    emotion_after: Optional[int] = Field(default=None, ge=1, le=5)


class RealConversation(Strict):
    """An ESConv conversation: scenario, transcript and the seeker's ratings."""

    scenario: Scenario
    record: ConversationRecord
    ratings: SeekerRatings
    strategies: list[Optional[str]]  # one per helper turn; kept out of Turn so the judge never sees labels


def _score(value: object) -> Optional[int]:
    text = str(value).strip()
    return int(text) if text in _VALID_SCORES else None


def load_esconv(split: Literal["train", "validation", "test"], *, allow_test: bool = False) -> list[RealConversation]:
    if split == "test" and not allow_test:
        raise ValueError(
            "The test split is reserved for the final evaluation (#7); pass allow_test=True to load it."
        )
    if split not in SPLITS:
        raise ValueError(f"Unknown ESConv split {split!r}; expected one of {sorted(SPLITS)}")
    filename, expected = SPLITS[split]
    rows = [json.loads(line) for line in download(ESCONV, ESCONV_REVISION, filename).split("\n") if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"ESConv {split}: expected {expected} rows, got {len(rows)}")
    conversations = []
    for i, row in enumerate(rows):
        scenario_id = f"esconv-{split}-{i:04d}"
        seeker = row["survey_score"]["seeker"]
        scenario = Scenario(
            scenario_id=scenario_id,
            split=split,
            emotion_type=row["emotion_type"],
            problem_type=row["problem_type"],
            situation=row["situation"],
            initial_intensity=int(seeker["initial_emotion_intensity"]),
        )
        turns, strategies = [], []
        for turn in row["dialog"]:
            speaker = _SPEAKERS.get(turn["speaker"])
            if speaker is None:
                raise ValueError(f"{scenario_id}: unknown speaker {turn['speaker']!r}")
            turns.append(Turn(speaker=speaker, text=turn["text"]))
            if speaker == "helper":
                strategies.append(turn.get("strategy"))
        record = ConversationRecord(
            id=f"conv-{scenario_id}",
            written_by="esconv-loader",
            created_at=datetime.now(timezone.utc),
            parent_ids=[],
            source="real",
            scenario_id=scenario_id,
            split=split,
            policy_version=REAL_POLICY_VERSION,
            turns=turns,
        )
        ratings = SeekerRatings(**{field: _score(seeker.get(key)) for field, key in _RATING_KEYS.items()})
        conversations.append(RealConversation(scenario=scenario, record=record, ratings=ratings, strategies=strategies))
    return conversations


def load_scenarios(split: Literal["train", "validation", "test"], *, allow_test: bool = False) -> list[Scenario]:
    return [conversation.scenario for conversation in load_esconv(split, allow_test=allow_test)]
