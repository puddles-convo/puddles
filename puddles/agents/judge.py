"""Judge (step 3): rates one finished conversation. Blind: it is given the turns and nothing else.

The judge reads no memory and writes none. Its scores go to the critic, into
<root>/scores/round-NN-train.jsonl, and as averages into the round's log entry.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loop.errors import ConfigError
from puddles.agents import system_prompt
from puddles.llm import LLM, AgentFailed, OpenAILLM, call_checked
from puddles.memory import now
from puddles.records import sha256_text
from schemas import ConversationRecord, JudgeInput, JudgeOutput, ScoredConversation, ScoreRecord

SYSTEM = system_prompt("t3-judge.md")
IDENTITY = "judge"
PROMPT_VERSION = "judge-v1"  # raise by hand when the prompt is retuned; the hash below catches the rest
MODEL_VARIABLE = "PUDDLES_JUDGE_MODEL"
CRITIC_CONVERSATIONS = 10  # the critic's spec: about ten of the lowest-scoring conversations
SCORE_FIELDS = ("empathy", "relevance", "emotion_before", "emotion_after")


def judge_version(model: str) -> str:
    """Names the prompt and the model together, so a change to either changes the version."""
    return f"{PROMPT_VERSION}-{sha256_text(SYSTEM)[:8]}-{model}"


def llm_from_env() -> tuple[LLM, str]:
    """The judge's own model, at temperature 0, and the judge version that goes with it.

    PUDDLES_JUDGE_MODEL must name a pinned snapshot (a dated model name), not an alias that moves.
    """
    model = os.environ.get(MODEL_VARIABLE)
    if not model:
        raise ConfigError(f"environment variable {MODEL_VARIABLE} is not set")
    llm = OpenAILLM(model=model)
    llm.extra = {"temperature": 0.0}  # whatever PUDDLES_AGENT_TEMPERATURE says for the other agents
    return llm, judge_version(model)


def build_input(record: ConversationRecord) -> JudgeInput:
    """Turns only: the ID, the scenario and the policy version stay behind."""
    return JudgeInput(turns=list(record.turns))


def render_user(inp: JudgeInput) -> str:
    lines = "\n".join(f"{i}. {t.speaker}: {t.text}" for i, t in enumerate(inp.turns, 1))
    return f"<conversation>\n{lines}\n</conversation>"


def check(out: JudgeOutput, inp: JudgeInput) -> list[str]:
    problems = [f"{name} must be between 1 and 5" for name in SCORE_FIELDS
                if not 1 <= getattr(out, name) <= 5]
    helper_turns = sum(1 for t in inp.turns if t.speaker == "helper")
    if len(out.strategies) != helper_turns:
        problems.append(f"strategies must have exactly {helper_turns} labels, one per helper turn; "
                        f"it has {len(out.strategies)}")
    return problems


def run(inp: JudgeInput, llm: LLM) -> JudgeOutput:
    user = render_user(inp)
    return call_checked(lambda fb: llm.structured(SYSTEM, user + fb, JudgeOutput),
                        lambda out: check(out, inp), "judge")


def score_id(conversation_id: str) -> str:
    return "score-" + conversation_id.removeprefix("conv-")


def to_record(out: JudgeOutput, conversation_id: str, version: str) -> ScoreRecord:
    return ScoreRecord(id=score_id(conversation_id), written_by=IDENTITY, created_at=now(),
                       parent_ids=[conversation_id], conversation_id=conversation_id,
                       judge_version=version, **out.model_dump())


@dataclass
class Judged:
    """One batch through the judge: the scores, and the conversations it could not score."""

    scores: list[ScoreRecord] = field(default_factory=list)
    unscored: list[str] = field(default_factory=list)  # conversation IDs
    reasons: dict[str, str] = field(default_factory=dict)  # conversation ID -> why it is unscored


def score_conversations(records: list[ConversationRecord], llm: LLM, version: str) -> Judged:
    """Judge each conversation. A second failed answer leaves it unscored; the batch goes on."""
    judged = Judged()
    for record in records:
        inp = build_input(record)
        try:
            out = run(inp, llm)
        except AgentFailed as exc:
            judged.unscored.append(record.id)
            judged.reasons[record.id] = str(exc)
            continue
        judged.scores.append(to_record(out, record.id, version))
    return judged


def save_scores(scores: list[ScoreRecord], path: Path) -> Path:
    """One ScoreRecord per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(s.model_dump_json() + "\n" for s in scores), encoding="utf-8")
    return path


def load_scores(path: Path) -> list[ScoreRecord]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ScoreRecord.model_validate_json(line) for line in lines if line.strip()]


def train_scores_path(root: Path, round_: int) -> Path:
    return Path(root) / "scores" / f"round-{round_:02d}-train.jsonl"


def to_scored(record: ConversationRecord, score: ScoreRecord) -> ScoredConversation:
    """What the critic takes: the conversation with its two scores and the judge's note."""
    if score.conversation_id != record.id:
        raise ValueError(f"score {score.id} is for {score.conversation_id}, not {record.id}")
    return ScoredConversation(conversation_id=record.id, empathy=score.empathy,
                              relevance=score.relevance, note=score.note, turns=list(record.turns))


def scored_conversations(records: list[ConversationRecord],
                         scores: list[ScoreRecord]) -> list[ScoredConversation]:
    """Pair each score with its conversation. Unscored conversations are left out."""
    by_id = {r.id: r for r in records}
    return [to_scored(by_id[s.conversation_id], s) for s in scores]


def select_for_critic(scored: list[ScoredConversation],
                      n: Optional[int] = CRITIC_CONVERSATIONS) -> list[ScoredConversation]:
    """The n lowest by (empathy + relevance) / 2, ties broken by conversation ID."""
    ranked = sorted(scored, key=lambda c: ((c.empathy + c.relevance) / 2, c.conversation_id))
    return ranked if n is None else ranked[:max(n, 0)]
