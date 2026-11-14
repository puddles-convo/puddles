"""Handoff formats for puddles: one Pydantic model per handoff.

Starting point for issue #1. Field names are a proposal; change them here
and every agent that imports these models changes with them.
"""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

Score = int  # 1 to 5, the scale ESConv users rated on


class Record(BaseModel):
    """Fields every handoff record carries, so any record can be traced."""

    id: str
    schema_version: int = SCHEMA_VERSION
    written_by: str  # agent name
    created_at: datetime
    parent_ids: list[str] = []  # the records this one was made from


# ---------- training ----------

class Turn(BaseModel):
    speaker: Literal["user", "helper"]
    text: str


class ConversationRecord(Record):
    """H2 in training (helper -> judge); H4 in inference (speaker -> note-taker)."""

    source: Literal["simulated", "live"]
    scenario_id: Optional[str] = None  # None for live sessions
    split: Optional[Literal["train", "validation", "test"]] = None
    policy_version: str
    turns: list[Turn]


class ScoreRecord(Record):
    """H3: judge -> critic."""

    conversation_id: str
    judge_version: str
    empathy: Score = Field(ge=1, le=5)
    relevance: Score = Field(ge=1, le=5)
    emotion_before: Score = Field(ge=1, le=5)  # intensity, 5 = worst
    emotion_after: Score = Field(ge=1, le=5)
    strategies: list[str]  # one per helper turn, ESConv strategy names
    note: str


class FailurePattern(Record):
    """H4: critic -> researcher."""

    round: int
    name: str
    description: str
    seen_in: int
    out_of: int
    example_conversation_ids: list[str]


class Source(BaseModel):
    url: str
    title: str
    technique: str  # one line


class ResearchNote(Record):
    """H5: researcher -> proposer."""

    round: int
    pattern_id: str
    query: str
    sources: list[Source]


class PolicyEdit(Record):
    """H6: proposer -> gatekeeper. The policy text itself is a file."""

    round: int
    pattern_id: str
    research_id: str
    base_policy_version: str
    new_policy_version: str
    policy_path: str
    summary: str


class Scores(BaseModel):
    empathy: float
    relevance: float
    emotion_change: float  # before minus after, higher is better


class GateOutcome(Record):
    """H7: gatekeeper -> next round. Also one line of the decision log."""

    round: int
    edit_id: str
    decision: Literal["kept", "reverted"]
    before: Scores
    after: Scores
    safety_passed: bool
    policy_in_force: str


# ---------- inference ----------

class UserMessage(BaseModel):
    """H1: listener -> crisis checker."""

    session_id: str
    turn: int
    text: str


class CheckedMessage(UserMessage):
    """H2: crisis checker -> helper."""

    risk: Literal["none", "concern", "crisis"]
    reason: str = ""


class HelperReply(BaseModel):
    """H3: helper -> speaker."""

    session_id: str
    turn: int
    text: str
    policy_version: str
    fixed_handoff_response: bool = False  # True when risk was "crisis"


class UserNote(Record):
    """Note-taker -> memory: a fact about this user for next time."""

    user_id: str
    session_id: str
    fact: str
