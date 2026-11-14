"""Record formats for puddles: one Pydantic model per handoff, log event and file.

Version 2. A proposal for the team to settle under issue #1: change a model here
and every agent that imports it changes with it.

Sections: shared pieces, training records, what each model returns, the decision
log, policy and safety files, inference records, user memory, and the shape of a
past.dev memory entry. docs/agents.md explains how they fit together.
"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 2

Score = int  # 1 to 5, the scale ESConv users rated on
Split = Literal["train", "validation", "test", "live"]


class Strict(BaseModel):
    """Base for formats where an unknown field is a mistake, not an extension."""

    model_config = ConfigDict(extra="forbid")


class Record(BaseModel):
    """Fields every stored record carries, so any record can be traced."""

    id: str
    schema_version: int = SCHEMA_VERSION
    written_by: str  # agent name
    created_at: datetime
    parent_ids: list[str] = []  # the records this one was made from


class Turn(BaseModel):
    speaker: Literal["user", "helper"]
    text: str


# ---------- training records ----------

class Scenario(Strict):
    """What the simulated user plays. Built from an ESConv row or a live session."""

    scenario_id: str
    split: Split
    emotion_type: str
    problem_type: str
    situation: str
    initial_intensity: Score = Field(ge=1, le=5)
    openness: Literal["open", "guarded", "resistant"] = "open"
    from_session: Optional[str] = None  # set when made from a live session


class ConversationRecord(Record):
    """A whole conversation: simulated (training), live (robot) or real (ESConv)."""

    source: Literal["simulated", "live", "real"]
    scenario_id: Optional[str] = None  # None for live sessions
    split: Optional[Split] = None
    policy_version: str
    turns: list[Turn]
    # live sessions only
    remember_me: bool = False  # may the robot keep notes about this person
    consent_for_training: bool = False  # may this session be used to improve it
    helped: Optional[Literal["yes", "no", "no answer"]] = None
    timings_ms: list[dict[str, int]] = []  # per turn: stt, first_token, tts, ...


class ScoreRecord(Record):
    """Judge -> critic."""

    conversation_id: str
    judge_version: str
    empathy: Score = Field(ge=1, le=5)
    relevance: Score = Field(ge=1, le=5)
    emotion_before: Score = Field(ge=1, le=5)  # intensity, 5 = worst
    emotion_after: Score = Field(ge=1, le=5)
    strategies: list[str]  # one per helper turn, ESConv strategy names
    note: str


class FailurePattern(Record):
    """Critic -> researcher."""

    round: int
    name: str
    description: str
    seen_in: int
    out_of: int  # the conversations the critic was given
    example_conversation_ids: list[str]
    same_as_pattern_id: Optional[str] = None  # set on a recurrence


class Source(BaseModel):
    url: str
    title: str = ""
    technique: str  # one line


class ResearchNote(Record):
    """Researcher -> proposer."""

    round: int
    pattern_id: str
    queries: list[str] = []
    query: Optional[str] = None  # version 1 field; use queries
    sources: list[Source]


EditAction = Literal["add", "reword", "remove"]


class PolicyEdit(Record):
    """Proposer -> gatekeeper. Exactly one rule added, reworded or removed."""

    round: int
    pattern_id: str
    research_id: str
    base_policy_version: str
    new_policy_version: str
    policy_path: str
    summary: str
    action: Optional[EditAction] = None
    rule_id: Optional[str] = None  # permanent ID such as C2
    rule_text: Optional[str] = None  # None for a removal


class Scores(BaseModel):
    empathy: float
    relevance: float
    quality: Optional[float] = None  # mean of (empathy + relevance) / 2
    emotion_change: Optional[float] = None  # reported, not gated


class GateOutcome(Record):
    """Gatekeeper -> next round."""

    round: int
    edit_id: str
    decision: Literal["kept", "reverted"]
    before: Scores
    after: Scores
    safety_passed: bool
    policy_in_force: str
    reason: str = ""


class GateInput(Strict):
    """What the gatekeeper is given."""

    round: int
    edit_id: str
    current_policy: str
    candidate_policy: str
    validation_scenario_ids: list[str]
    current_scores: Scores
    noise: float = Field(ge=0)
    token_cap: int
    safety_case_set: str


# ---------- what each agent is given (code builds these, then renders the prompt) ----------

class SimulatedUserInput(Strict):
    scenario: Scenario
    turns: list[Turn]


class HelperInput(Strict):
    policy_version: str
    user_notes: list[str] = []  # empty in training
    turns: list[Turn]


class ScoredConversation(Strict):
    conversation_id: str
    empathy: Score
    relevance: Score
    note: str
    turns: list[Turn]


class PatternBrief(Strict):
    id: str
    name: str
    description: Optional[str] = None
    status: Optional[Literal["open", "fixed", "recurring"]] = None
    rule_id: Optional[str] = None  # the rule that fixed it, if any


class CriticInput(Strict):
    round: int
    policy_text: str
    earlier_patterns: list[PatternBrief] = []  # complete list, built from the log
    from_memory: list[str] = []  # detail found by recall, loaded from record files
    conversations: list[ScoredConversation]  # the lowest-scoring ones, in full


class ResearcherInput(Strict):
    pattern: PatternBrief
    earlier_queries: list[str] = []  # complete list, built from the log


class EditSummary(Strict):
    id: str
    action: EditAction
    rule_id: str
    outcome: Literal["kept", "reverted"]
    summary: str


class ProposerInput(Strict):
    policy_text: str
    at_length_limit: bool
    pattern: PatternBrief
    sources: list[Source]
    earlier_edits: list[EditSummary] = []  # complete list, built from the log


class CrisisCheckerInput(Strict):
    session_id: str
    turn: int
    recent_turns: list[Turn] = []
    text: str


class ExistingNote(Strict):
    note_id: str
    fact: str


class NoteTakerInput(Strict):
    today: str  # ISO date, supplied by code
    may_add_notes: bool  # false when the person declined to be remembered
    existing_notes: list[ExistingNote] = []
    turns: list[Turn]


# ---------- what each model returns (code adds IDs, dates and versions) ----------

class JudgeInput(Strict):
    """Turns only, so the judge cannot tell which policy produced them."""

    turns: list[Turn]


class JudgeOutput(Strict):
    empathy: Score = Field(ge=1, le=5)
    relevance: Score = Field(ge=1, le=5)
    emotion_before: Score = Field(ge=1, le=5)
    emotion_after: Score = Field(ge=1, le=5)
    strategies: list[str]
    note: str


class CriticOutput(Strict):
    found: bool
    name: Optional[str] = None
    description: Optional[str] = None
    seen_in: int = 0
    out_of: int = 0
    example_conversation_ids: list[str] = []
    same_as_pattern_id: Optional[str] = None
    reason: str = ""  # why nothing was found, when found is false


class ResearcherOutput(Strict):
    queries: list[str]
    sources: list[Source]  # empty means the round ends as no_evidence


class ProposerOutput(Strict):
    action: EditAction
    rule_id: Optional[str] = None  # existing ID for reword or remove; None for add
    rule_text: Optional[str] = None  # None for remove
    summary: str


# ---------- the decision log (append-only) ----------

class RecordRef(Strict):
    """Pointer from the log to a record file under records/."""

    id: str
    path: str
    sha256: str


class LogPattern(Strict):
    id: str
    name: str
    seen_in: int
    out_of: int
    status_before: Literal["open", "fixed", "recurring"]


class LogEdit(Strict):
    id: str
    action: EditAction
    rule_id: str
    rule_text: Optional[str] = None


class LogPolicy(Strict):
    base: str
    base_sha256: str
    new: Optional[str] = None
    new_sha256: Optional[str] = None
    in_force: str


class LogScores(Strict):
    before: Scores
    after: Scores
    noise: float = Field(ge=0)


class LogSafety(Strict):
    case_file: str
    passed: bool
    cases: int
    failed_case_ids: list[str] = []


class RoundEvent(Strict):
    """One per round. Rounds that end early leave the later parts empty."""

    event: Literal["round"] = "round"
    round: int
    outcome: Literal["kept", "reverted", "no_pattern", "no_evidence"]
    records: dict[str, RecordRef] = {}
    pattern: Optional[LogPattern] = None
    edit: Optional[LogEdit] = None
    policy: LogPolicy
    scores: Optional[LogScores] = None
    safety: Optional[LogSafety] = None
    reason: str


class MemoryCopyEvent(Strict):
    """Appended each time a copy to past.dev is confirmed or gives up waiting."""

    event: Literal["memory_copy"] = "memory_copy"
    round: int
    record_id: str
    status: Literal["confirmed", "pending"]
    attempt: int = Field(ge=1)


# ---------- policy and safety files ----------

class LearnedRule(Strict):
    edit: str
    action: EditAction
    pattern: str
    round: int
    text_sha256: str


class PolicyMeta(Strict):
    """policies/<version>.meta.json: where each rule came from."""

    policy_version: str
    parent: Optional[str] = None
    policy_sha256: str
    safety_sha256: str  # the same in every version
    base_rules: list[str]
    learned_rules: dict[str, LearnedRule]


class Assertion(Strict):
    type: Literal["contains", "judged"]
    value: Optional[str] = None  # for contains: checked by code
    question: Optional[str] = None  # for judged: yes or no, asked of the judge model


class SafetyCase(Strict):
    case_id: str
    rule: Literal["S1", "S2", "S3"]
    opener: str
    must: list[Assertion] = []  # each must be true of the reply
    must_not: list[Assertion] = []  # each must be false


class SafetyCaseResult(Strict):
    case_id: str
    passed: bool
    reason: str = ""


class SafetyResult(Strict):
    policy_version: str
    case_file: str = "safety-v1.json"
    passed: bool  # false if any case failed
    cases: list[SafetyCaseResult]


# ---------- inference ----------

class UserMessage(BaseModel):
    """Listener -> crisis checker and helper."""

    session_id: str
    turn: int
    text: str


class CheckedMessage(UserMessage):
    """Crisis checker -> release step."""

    risk: Literal["none", "concern", "crisis"]
    category: Optional[Literal["self", "danger", "unsure"]] = None  # crisis only
    reason: str = ""


class HelperReply(BaseModel):
    """Release step -> speaker: the helper's draft, or a fixed response."""

    session_id: str
    turn: int
    text: str
    policy_version: str
    fixed_handoff_response: bool = False


# ---------- user memory ----------

class UserNote(Record):
    """Note-taker -> user memory: one fact about one person."""

    user_id: str
    session_id: str
    fact: str


class NoteAction(Strict):
    """What the note-taker returns for each change."""

    action: Literal["add", "replace", "delete"]
    note_id: Optional[str] = None  # for replace and delete
    fact: Optional[str] = None  # for add and replace


class LiveNoteIndex(Strict):
    """Kept by the brain. Only notes listed here reach the helper's prompt."""

    user_id: str
    note_ids: list[str]


# ---------- past.dev ----------

class MemoryEntry(Strict):
    """The shape of one entry sent to past.dev."""

    id: str
    content: str  # the sentence a search matches; carries the IDs
    label: str
    timestamp: datetime
    identity: str  # the agent that produced it
    audience: str  # training-agents, or one person's private audience
    metadata: dict[str, Any] = {}
