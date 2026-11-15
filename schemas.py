"""Record formats for puddles: one Pydantic model per handoff, log event and file.

Version 3. A proposal for the team to settle under issue #1: change a model here
and every agent that imports it changes with it.

Version 3 matches the version 2 prompts (docs/agent-prompts.md): evidence fields
written before scores, the judge's safety flags, the critic's comparison with
high-scoring conversations, a supporting quote per source, the proposer's toolbox
(Intervention), the gate's assertion check, calibration example files and the
held-out jury.

Sections: shared pieces, training records, what each model returns, the decision
log, policy and safety files, inference records, user memory, the shape of a
past.dev memory entry, example files and the held-out jury. docs/agents.md
explains how they fit together.
"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 3

Score = int  # 1 to 5, the scale ESConv users rated on
Split = Literal["train", "validation", "test", "live"]

# What the judge flags as unsafe in a conversation. Reported to the gate on its own,
# never folded into the quality scores.
SafetyFlag = Literal["missed_risk", "diagnosis_or_medical", "role_break", "harmful_validation"]

# The proposer's toolbox, version 1, plus the option to change nothing.
ToolAction = Literal[
    "edit_rule", "add_skill_card", "set_examples", "train_adapter", "merge_adapters", "no_change"
]


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
    # version 3: what makes the simulated user open up gradually
    opener: Optional[str] = None  # the real seeker's first message in ESConv
    reveal_facts: list[str] = []  # 2 to 4 facts, in the order the real seeker revealed them
    min_turns: int = Field(default=4, ge=1)  # fewest turns before the simulated user may end


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
    evidence: str = ""  # version 3: the judge's reasoning, written before its scores
    safety_flags: list[SafetyFlag] = []  # version 3: goes to the gate separately


class FailurePattern(Record):
    """Critic -> researcher."""

    round: int
    name: str
    description: str
    seen_in: int
    out_of: int  # the conversations the critic was given
    example_conversation_ids: list[str]
    same_as_pattern_id: Optional[str] = None  # set on a recurrence
    seen_in_high: Optional[int] = None  # version 3: how many high-scoring ones show it too


class Source(BaseModel):
    url: str
    title: str = ""
    quote: str = ""  # version 3: the passage on the page that supports the technique
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
    """Version 2 record for an edit_rule action. Version 3 records every action as an
    Intervention (below); this model stays so existing records still load."""

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
    edit_id: str  # the PolicyEdit or Intervention ID
    decision: Literal["kept", "reverted"]
    before: Scores
    after: Scores
    safety_passed: bool
    policy_in_force: str
    reason: str = ""
    prediction_met: Optional[bool] = None  # version 3: did the proposer's prediction come true


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
    # version 3: rendered text, one item each. Empty on the robot for now (open issue:
    # how skills and examples are shipped to the cloud brain); the prompt then says "none".
    skills: list[str] = Field(default=[], max_length=3)
    examples: list[str] = []


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
    conversations: list[ScoredConversation]  # the lowest-scoring ones, in full (<low_scoring>)
    high_scoring: list[ScoredConversation] = []  # version 3: a few of the best, for comparison


class ResearcherInput(Strict):
    pattern: PatternBrief
    earlier_queries: list[str] = []  # complete list, built from the log


class HarnessBrief(Strict):
    """What the proposer is shown of the harness version in force."""

    version: str
    policy_text: str
    at_length_limit: bool
    skill_cards: list[str] = []  # "name: one-line summary"
    example_situations: list[str] = []
    adapter: Optional[str] = None  # adapter ID in force, or None


class LeverStats(Strict):
    """One row of the scoreboard, built by code from earlier gate outcomes."""

    action: ToolAction
    tried: int = Field(ge=0)
    kept: int = Field(ge=0)
    average_gain: float = 0.0  # change in quality when kept
    average_cost: str = ""  # e.g. "4 min, $0.30"


class ActionSummary(Strict):
    """One earlier action, built by code from the decision log."""

    id: str
    action: ToolAction
    target: str = ""  # e.g. rule C2, card reflect-before-advice, adapter-01
    outcome: Literal["kept", "reverted", "no_change"]
    prediction: str = ""
    observed: str = ""
    summary: str


class Budget(Strict):
    rounds_left: int = Field(ge=0)
    training_runs_left: int = Field(ge=0)
    exploration_round: bool = False


class ProposerInput(Strict):
    harness: HarnessBrief
    pattern: PatternBrief
    sources: list[Source]
    scoreboard: list[LeverStats] = []
    earlier_actions: list[ActionSummary] = []  # complete list, built from the log
    budget: Budget
    tools: list[ToolAction]  # shuffled by code each round


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
    evidence: str  # first, so the judge reasons before it scores
    empathy: Score = Field(ge=1, le=5)
    relevance: Score = Field(ge=1, le=5)
    emotion_before: Score = Field(ge=1, le=5)
    emotion_after: Score = Field(ge=1, le=5)
    strategies: list[str]
    safety_flags: list[SafetyFlag] = []
    note: str


class CriticOutput(Strict):
    evidence: str  # first
    found: bool
    name: Optional[str] = None
    description: Optional[str] = None
    seen_in: int = 0
    out_of: int = 0
    seen_in_high: int = 0  # if about as common as in the low-scoring set, not the cause
    example_conversation_ids: list[str] = []
    same_as_pattern_id: Optional[str] = None
    reason: str = ""  # why nothing was found, when found is false


class ResearcherOutput(Strict):
    queries: list[str]
    sources: list[Source]  # empty means the round ends as no_evidence


class Considered(Strict):
    action: ToolAction
    rejected_because: str


class EditRuleArgs(Strict):
    op: EditAction
    rule_id: Optional[str] = None  # existing ID for reword or remove; None for add
    rule_text: Optional[str] = None  # None for remove


class SkillCardArgs(Strict):
    op: Literal["add", "rewrite", "retire"]
    card_id: Optional[str] = None  # existing ID for rewrite or retire
    name: str = ""
    when_to_use: str = ""
    when_not_to_use: str = ""
    steps: list[str] = []
    source_urls: list[str] = []  # must come from the researcher's sources


class ExamplesArgs(Strict):
    situation: str
    k: int = Field(ge=1, le=3)


class TrainAdapterArgs(Strict):
    method: Literal["dpo", "sft", "grpo"] = "dpo"  # version 1 toolbox: dpo
    pairs_from: str  # which conversations, e.g. "pattern-r07 scenarios, current harness"
    rank: Literal[16, 32] = 16
    epochs: Literal[1, 2] = 1


class MergeArgs(Strict):
    adapter_id: str  # an adapter that passed the gate
    search: Literal["base_blend", "adapter_sum"] = "base_blend"


class NoChangeArgs(Strict):
    pass


ARGS_FOR = {
    "edit_rule": EditRuleArgs,
    "add_skill_card": SkillCardArgs,
    "set_examples": ExamplesArgs,
    "train_adapter": TrainAdapterArgs,
    "merge_adapters": MergeArgs,
    "no_change": NoChangeArgs,
}


class ProposerOutput(Strict):
    """Version 3: one action from the toolbox, with what it expects to happen.
    Code checks the arguments match the action and its limits before applying it."""

    considered: list[Considered] = Field(min_length=2)
    action: ToolAction
    args: dict[str, Any] = {}
    hypothesis: str
    prediction: str
    addresses: str  # pattern ID

    @model_validator(mode="after")
    def _args_match_action(self):
        ARGS_FOR[self.action].model_validate(self.args)
        return self


class Intervention(Record):
    """Proposer -> gatekeeper, as stored: the output plus what code adds."""

    round: int
    pattern_id: str
    research_id: Optional[str] = None  # None when the round had no sources
    base_harness_version: str
    new_harness_version: Optional[str] = None  # None for no_change
    action: ToolAction
    args: dict[str, Any] = {}
    considered: list[Considered] = []
    hypothesis: str
    prediction: str


class AssertionCheckOutput(Strict):
    """The gatekeeper's yes-or-no check of one judged safety assertion."""

    quote: str = ""  # the words in the reply that decide it
    answer: Literal["yes", "no"]


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


# ---------- example files (calibration examples for the prompts) ----------

class JudgeExample(Strict):
    """One entry of examples/judge_examples.json."""

    id: str
    source: str
    turns: list[Turn]
    expected: JudgeOutput


class CrisisLabel(Strict):
    level: Literal["crisis", "concern", "none"]
    category: Literal["self", "danger", "unsure", "-"]


class CrisisExample(Strict):
    """One entry of examples/crisis_checker_examples.json."""

    id: str
    recent_turns: list[Turn] = []
    text: str
    expected: CrisisLabel
    why: str
    in_prompt: bool = False  # the 4 in the prompt are never used to test it


# ---------- held-out jury (final evaluation only) ----------

class JurorPersonOutput(Strict):
    """Juror A: the person seeking support."""

    evidence: str
    score: Score = Field(ge=1, le=5)  # felt heard
    worst_turn: Optional[int] = None


class JurorCounsellorOutput(Strict):
    """Juror B: peer-support skill."""

    evidence: str
    score: Score = Field(ge=1, le=5)
    exploration: Score = Field(ge=1, le=5)
    stayed_in_role: bool


SafetyFailure = Literal[
    "missed_risk", "unsafe_advice", "method_detail", "diagnosis_or_medical",
    "harmful_validation", "role_break",
]


class JurorSafetyOutput(Strict):
    """Juror C: safety. A failure is a veto."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    evidence: str
    score: Score = Field(ge=1, le=5)
    passed: bool = Field(alias="pass")
    failures: list[SafetyFailure] = []


ScepticFlag = Literal[
    "empty_validation", "flattery", "stock_phrases", "repetition", "over_promising",
    "agreeing_to_please",
]


class JurorScepticOutput(Strict):
    """Juror D: substance behind the warmth."""

    evidence: str
    score: Score = Field(ge=1, le=5)  # substance
    flags: list[ScepticFlag] = []
