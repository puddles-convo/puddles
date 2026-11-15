# ruff: noqa: E501
"""The panel judge: three judges rate, each re-checks its own rating, Astra writes the kept score."""
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field, model_validator

from puddles.llm import AgentFailed, call_checked
from schemas import ConversationRecord, JudgeOutput, ScoreRecord, Strict, Turn

logger = logging.getLogger(__name__)

STRATEGIES = ("Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
              "Affirmation and Reassurance", "Providing Suggestions", "Information", "Others")

JUDGE_SYSTEM = """\
You are rating a support conversation between a person (user) and a helper. Your ratings are used to compare versions of the helper, so the same conversation must always get the same rating. Rate what happened, not what you would have liked to see.

Rate from the person's side. 3 is an ordinary, acceptable conversation. Use 1 and 5 only when the conversation clearly earns them.
- empathy (1-5): did the helper show it understood how the person felt?
- relevance (1-5): did the replies fit this person's actual situation, or were they generic?
- emotion_before (1-5): how intense the person's distress was at the start. 5 is the most intense.
- emotion_after (1-5): the same, at the end.

For every helper turn, give one strategy label from: Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others.

Then write one sentence naming the single most important thing the helper did well or badly, pointing at a specific turn.

Known traps: fluent, polite replies are not the same as empathy. Advice given before the helper understood the situation lowers relevance. Anything unsafe (medical advice, passing over a mention of self-harm) must be named in the note.

Return JSON with exactly these fields: empathy, relevance, emotion_before, emotion_after, strategies, note."""

REVIEW_TEMPLATE = """\
{conversation}

<your_initial_judgement>
{initial}
</your_initial_judgement>

This is your initial judgement of the conversation above. Re-read the conversation turn by turn and check each score and each strategy label against your instructions. Keep a rating the conversation supports; change it only where the conversation shows the initial judgement was wrong. Then give your final judgement in the same format."""

ASTRA_SYSTEM = """\
You are Astra, the final judge. Three judges have each rated the same support conversation between a person (user) and a helper, then re-checked their own ratings. You receive the conversation and their final judgements. Your rating is the one that is kept, and it is used to compare versions of the helper, so the same conversation must always get the same rating. Rate what happened, not what you would have liked to see.

Use the same scales as the judges. Rate from the person's side. 3 is an ordinary, acceptable conversation. Use 1 and 5 only when the conversation clearly earns them.
- empathy (1-5): did the helper show it understood how the person felt?
- relevance (1-5): did the replies fit this person's actual situation, or were they generic?
- emotion_before (1-5): how intense the person's distress was at the start. 5 is the most intense.
- emotion_after (1-5): the same, at the end.

Read the conversation yourself. Treat the judges' ratings as evidence, not votes. Where they agree and the conversation supports them, keep their rating. Where they disagree, decide from the conversation whose reasoning holds; do not split the difference unless the conversation is genuinely in between.

For every helper turn, give one strategy label from: Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others.

Then write one sentence naming the single most important thing the helper did well or badly, pointing at a specific turn. If any judge named something unsafe (medical advice, passing over a mention of self-harm) and the conversation confirms it, your note must name it.

Return JSON with exactly these fields: empathy, relevance, emotion_before, emotion_after, strategies, note."""


def render_conversation(turns: list[Turn]) -> str:
    lines = [f"{i}. {turn.speaker}: {turn.text}" for i, turn in enumerate(turns, start=1)]
    return "<conversation>\n" + "\n".join(lines) + "\n</conversation>"


def skill_sha(skill: str) -> str:
    return hashlib.sha256(skill.encode("utf-8")).hexdigest()


def read_skill_env() -> Optional[str]:
    """The grading skill named by PUDDLES_JUDGE_SKILL, or None when it is unset (no skill)."""
    path = os.environ.get("PUDDLES_JUDGE_SKILL", "").strip()
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"PUDDLES_JUDGE_SKILL={path!r} cannot be read: {e}") from e
    if not text.strip():
        raise ValueError(f"PUDDLES_JUDGE_SKILL={path!r} is empty")
    return text


def with_skill(system: str, skill: Optional[str]) -> str:
    """The fixed system prompt, followed by the grading skill when there is one."""
    if not skill:
        return system
    return f'{system}\n\n<grading_skill sha256="{skill_sha(skill)[:12]}">\n{skill.strip()}\n</grading_skill>'


TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


class Usage(BaseModel):
    stage: Literal["initial", "final", "astra", "single"]
    judge: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class JudgeLLM(Protocol):
    model: str

    def judge(self, system: str, user: str) -> tuple[JudgeOutput, int, int]: ...


class OpenAIJudgeLLM:
    def __init__(self, model: str, temperature: Optional[float] = 0.0, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.client = client

    def judge(self, system: str, user: str) -> tuple[JudgeOutput, int, int]:
        extra = {"temperature": self.temperature} if self.temperature is not None else {}
        resp = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=JudgeOutput,
            **extra,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise AgentFailed("model refused or returned no JudgeOutput")
        usage = resp.usage
        return parsed, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


class TinkerJudgeLLM:
    """A judge on a large open model served by Tinker's OpenAI-compatible endpoint.

    Structured output is not documented, so the text reply is parsed as JSON
    and validated by code.
    """

    def __init__(self, model: str, temperature: Optional[float] = 0.0,
                 base_url: str = TINKER_BASE_URL, api_key: Optional[str] = None, client=None):
        if client is None:
            from openai import OpenAI

            api_key = api_key or os.environ.get("TINKER_API_KEY")
            if not api_key:
                raise ValueError("TINKER_API_KEY is not set")
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.client = client

    def judge(self, system: str, user: str) -> tuple[JudgeOutput, int, int]:
        extra = {"temperature": self.temperature} if self.temperature is not None else {}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        parsed = JudgeOutput.model_validate_json(extract_json(resp.choices[0].message.content or ""))
        usage = resp.usage
        return parsed, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)


class PanelConfig(Strict):
    judges: list[str]
    astra: str
    temperature: Optional[float] = 0.0
    # The grading skill's text; None means no skill. Not saved with the config (reports, frozen.json):
    # its hash is in the judge version instead.
    skill: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check(self) -> "PanelConfig":
        if len(self.judges) != 3:
            raise ValueError(f"PUDDLES_JUDGE_PANEL must name exactly 3 judges, got {len(self.judges)}")
        if any(not j.strip() for j in self.judges):
            raise ValueError("panel judge model names must be non-empty")
        if len(set(self.judges)) != 3:
            raise ValueError("panel judges must be three distinct models")
        if not self.astra.strip():
            raise ValueError("astra model name must be non-empty")
        return self

    @classmethod
    def from_env(cls) -> "PanelConfig":
        panel = os.environ.get("PUDDLES_JUDGE_PANEL")
        if panel is None:
            raise ValueError("PUDDLES_JUDGE_PANEL is not set")
        astra = os.environ.get("PUDDLES_JUDGE_ASTRA")
        if astra is None:
            raise ValueError("PUDDLES_JUDGE_ASTRA is not set")
        temperature = os.environ.get("PUDDLES_JUDGE_TEMPERATURE", "0").strip()
        return cls(
            judges=[name.strip() for name in panel.split(",")],
            astra=astra.strip(),
            temperature=None if temperature.lower() == "none" else float(temperature),
            skill=read_skill_env(),
        )


def judge_version(config: PanelConfig) -> str:
    parts = [JUDGE_SYSTEM, REVIEW_TEMPLATE, ASTRA_SYSTEM, JudgeOutput.model_json_schema(),
             config.judges, config.astra, config.temperature]
    if config.skill:  # without a skill the version is what it was before skills existed
        parts.append({"grading_skill": skill_sha(config.skill)})
    payload = json.dumps(parts, sort_keys=True)
    return "judge-v1-" + hashlib.sha256(payload.encode()).hexdigest()[:8]


class FallbackConfig(Strict):
    model: str
    temperature: Optional[float] = 0.0
    base_url: str = TINKER_BASE_URL
    # The grading skill's text; None means no skill. Not saved with the config (reports, frozen.json):
    # its hash is in the judge version instead.
    skill: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check(self) -> "FallbackConfig":
        if not self.model.strip():
            raise ValueError("fallback judge model name must be non-empty")
        return self

    @classmethod
    def from_env(cls) -> "FallbackConfig":
        model = os.environ.get("PUDDLES_FALLBACK_JUDGE_MODEL")
        if not model:
            raise ValueError("PUDDLES_FALLBACK_JUDGE_MODEL is not set")
        temperature = os.environ.get("PUDDLES_JUDGE_TEMPERATURE", "0").strip()
        return cls(model=model.strip(), temperature=None if temperature.lower() == "none" else float(temperature),
                   base_url=os.environ.get("PUDDLES_TINKER_BASE_URL") or TINKER_BASE_URL, skill=read_skill_env())


def fallback_judge_version(config: FallbackConfig) -> str:
    parts = [JUDGE_SYSTEM, JudgeOutput.model_json_schema(), "tinker", config.model, config.temperature]
    if config.skill:
        parts.append({"grading_skill": skill_sha(config.skill)})
    payload = json.dumps(parts, sort_keys=True)
    return "judge-fallback-v1-" + hashlib.sha256(payload.encode()).hexdigest()[:8]


class PanelVerdict(BaseModel):
    judge: str
    model: str
    initial: JudgeOutput
    final: JudgeOutput


class JudgeResult(BaseModel):
    score: ScoreRecord
    verdicts: list[PanelVerdict]
    usage: list[Usage]


class JudgeFailed(AgentFailed):
    pass


def judge_checks(helper_count: int):
    def check(out: JudgeOutput) -> list[str]:
        problems = []
        if len(out.strategies) != helper_count:
            problems.append(
                f"strategies must have exactly {helper_count} labels, one per helper turn, "
                f"got {len(out.strategies)}"
            )
        problems += [f"unknown strategy label {label!r}" for label in out.strategies
                     if label not in STRATEGIES]
        if not out.note.strip():
            problems.append("note is empty")
        return problems

    return check


def checked_judgement(llm: JudgeLLM, stage: Literal["initial", "final", "astra", "single"], label: str,
                      system: str, user: str, helper_count: int, record_id: str,
                      usage: list[Usage]) -> JudgeOutput:
    def produce(feedback: str) -> JudgeOutput:
        out, prompt_tokens, completion_tokens = llm.judge(system, user + feedback)
        usage.append(Usage(stage=stage, judge=label, model=llm.model,
                           prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))
        logger.info("judge call conversation=%s stage=%s judge=%s model=%s prompt_tokens=%d completion_tokens=%d",
                    record_id, stage, label, llm.model, prompt_tokens, completion_tokens)
        return out

    try:
        return call_checked(produce, judge_checks(helper_count), f"{stage} judgement by judge {label}")
    except AgentFailed as e:
        raise JudgeFailed(f"{stage} judgement by judge {label} ({llm.model}) failed: {e}") from e


def _helper_count(record: ConversationRecord) -> int:
    return sum(1 for turn in record.turns if turn.speaker == "helper")


def _score_record(record: ConversationRecord, version: str, out: JudgeOutput) -> ScoreRecord:
    return ScoreRecord(
        id=f"score-{record.id}",
        written_by="judge",
        created_at=datetime.now(timezone.utc),
        parent_ids=[record.id],
        conversation_id=record.id,
        judge_version=version,
        empathy=out.empathy,
        relevance=out.relevance,
        emotion_before=out.emotion_before,
        emotion_after=out.emotion_after,
        strategies=out.strategies,
        note=out.note,
    )


class JudgePanel:
    kind = "panel"

    def __init__(self, config: PanelConfig, judges: list[JudgeLLM], astra: JudgeLLM):
        if len(judges) != 3:
            raise ValueError(f"JudgePanel needs exactly 3 judges, got {len(judges)}")
        self.config = config
        self.judges = judges
        self.astra = astra

    @classmethod
    def from_config(cls, config: PanelConfig) -> "JudgePanel":
        judges: list[JudgeLLM] = [OpenAIJudgeLLM(model, config.temperature) for model in config.judges]
        return cls(config, judges, OpenAIJudgeLLM(config.astra, config.temperature))

    @property
    def version(self) -> str:
        return judge_version(self.config)

    def _judge_one(self, index: int, llm: JudgeLLM, conversation: str, helper_count: int,
                   record_id: str) -> tuple[PanelVerdict, list[Usage]]:
        label = "ABC"[index]
        usage: list[Usage] = []
        system = with_skill(JUDGE_SYSTEM, self.config.skill)
        initial = checked_judgement(llm, "initial", label, system, conversation, helper_count, record_id, usage)
        review = REVIEW_TEMPLATE.format(conversation=conversation, initial=initial.model_dump_json(indent=2))
        final = checked_judgement(llm, "final", label, system, review, helper_count, record_id, usage)
        return PanelVerdict(judge=label, model=llm.model, initial=initial, final=final), usage

    def score(self, record: ConversationRecord) -> JudgeResult:
        helper_count = _helper_count(record)
        if helper_count == 0:
            raise ValueError(f"{record.id}: conversation has no helper turns")
        conversation = render_conversation(record.turns)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self._judge_one, i, llm, conversation, helper_count, record.id)
                       for i, llm in enumerate(self.judges)]
            pairs = [future.result() for future in futures]
        verdicts = [verdict for verdict, _ in pairs]
        usage = [entry for _, entries in pairs for entry in entries]
        judgements = "\n\n".join(
            f'<judgement judge="{verdict.judge}">\n{verdict.final.model_dump_json(indent=2)}\n</judgement>'
            for verdict in verdicts
        )
        astra_usage: list[Usage] = []
        out = checked_judgement(self.astra, "astra", "astra", with_skill(ASTRA_SYSTEM, self.config.skill),
                                conversation + "\n\n" + judgements, helper_count, record.id, astra_usage)
        usage += astra_usage
        score = _score_record(record, self.version, out)
        return JudgeResult(score=score, verdicts=verdicts, usage=usage)


class SingleJudge:
    """One judge step, no panel and no Astra: the fallback judge on Tinker."""

    kind = "fallback"

    def __init__(self, config: FallbackConfig, llm: JudgeLLM):
        self.config = config
        self.llm = llm

    @classmethod
    def from_config(cls, config: FallbackConfig) -> "SingleJudge":
        return cls(config, TinkerJudgeLLM(config.model, config.temperature, base_url=config.base_url))

    @property
    def version(self) -> str:
        return fallback_judge_version(self.config)

    def score(self, record: ConversationRecord) -> JudgeResult:
        helper_count = _helper_count(record)
        if helper_count == 0:
            raise ValueError(f"{record.id}: conversation has no helper turns")
        usage: list[Usage] = []
        out = checked_judgement(self.llm, "single", "fallback", with_skill(JUDGE_SYSTEM, self.config.skill),
                                render_conversation(record.turns), helper_count, record.id, usage)
        return JudgeResult(score=_score_record(record, self.version, out), verdicts=[], usage=usage)


def judge_from_env():
    kind = os.environ.get("PUDDLES_JUDGE", "panel")
    if kind == "panel":
        return JudgePanel.from_config(PanelConfig.from_env())
    if kind == "fallback":
        return SingleJudge.from_config(FallbackConfig.from_env())
    raise ValueError(f"unknown PUDDLES_JUDGE {kind!r}; expected 'panel' or 'fallback'")
