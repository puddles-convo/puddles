"""Run the persona pilot: three synthetic people, three sessions each, with notes and without.

    python -m scripts.persona_pilot --personas fixtures/personas.json --policy policies/policy-v2.md \
        --out results/persona-pilot [--turns 6] [--fake] [--only PERSONA_ID] [--start YYYY-MM-DD]

--fake is fully offline: scripted helper, checker, simulated user, note-taker and judge, and an
in-process past.dev. Without it the helper and checker come from HELPER_*, the simulated user from
SIMUSER_*, the notes from past.dev (PAST_API_KEY), the note-taker from OPENAI_API_KEY and
PUDDLES_AGENT_MODEL, and the judge from puddles.eval.judge.judge_from_env (PUDDLES_JUDGE_*).
Contract: docs/specs/persona-pilot.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, Sequence

from loop.errors import ConfigError, PolicyError
from loop.llm import FakeClient
from loop.prompts import NO_NOTES
from puddles.graph import check_policy_file
from puddles.inference import crisis
from puddles.inference.crisis import CrisisPath
from puddles.inference.note_store import FakeUserMemory, NoteStore
from puddles.inference.pilot import (DEFAULT_TURNS, Clients, Persona, arm_user_id, load_personas,
                                     render_report, run_pilot)
from puddles.llm import AgentFailed
from puddles.memory import PastError
from schemas import ConversationRecord, ScoreRecord
from scripts.chat import FAKE_HELPER_LINE, FakeNoteTaker, fake_checker

FAKE_REMEMBER = "I remember what you told me last time."
FAKE_USER_FIRST = "Yes, that is part of it."
FAKE_USER_LATER = "I suppose so."
NOTES_KEYS = ("PAST_API_KEY", "OPENAI_API_KEY")


def _print_error(message: str) -> None:
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = _Parser(prog="python -m scripts.persona_pilot", description=__doc__.splitlines()[0])
    p.add_argument("--personas", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--only", default=None)
    p.add_argument("--start", default=None, help="the day of session 1, YYYY-MM-DD; default today")
    return p.parse_args(argv)


# ---------- fakes ----------

def _fake_helper() -> FakeClient:
    def answer(system: str, messages: list[dict[str, str]]) -> str:
        first = sum(1 for m in messages if m["role"] == "user") == 1
        if first and NO_NOTES not in system:
            return f"{FAKE_REMEMBER} {FAKE_HELPER_LINE}"
        return FAKE_HELPER_LINE
    return FakeClient(answer)


def _fake_user(first_line: str) -> FakeClient:
    calls = {"n": 0}

    def answer(system: str, messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        return first_line if calls["n"] == 1 else FAKE_USER_LATER
    return FakeClient(answer)


def fake_clients(persona: Persona, session: int, arm: str, path: CrisisPath) -> Clients:
    plan = next(s for s in persona.sessions if s.session == session)
    first = FAKE_USER_FIRST
    if arm == "notes_on" and plan.fake_markers:
        user_id = arm_user_id(persona.persona_id, "notes_on")
        first += " " + " ".join(m.replace("{user_id}", user_id) for m in plan.fake_markers)
    return Clients(helper=_fake_helper(), checker=fake_checker(path), user=_fake_user(first), path=path)


class FakeJudge:
    """Scores 4 and 4 when the first helper reply starts with FAKE_REMEMBER, 3 and 3 otherwise."""

    version = "fake-judge"

    def score(self, record: ConversationRecord) -> Any:
        helper = [t for t in record.turns if t.speaker == "helper"]
        remembered = bool(helper) and helper[0].text.startswith(FAKE_REMEMBER)
        value = 4 if remembered else 3
        score = ScoreRecord(id=f"score-{record.id}", written_by="judge", created_at=record.created_at,
                            parent_ids=[record.id], conversation_id=record.id, judge_version=self.version,
                            empathy=value, relevance=value, emotion_before=4, emotion_after=3,
                            strategies=["Question"] * len(helper), note="scripted")
        return SimpleNamespace(score=score, usage=[])


# ---------- the command ----------

def _out_is_free(out: Path) -> bool:
    if not out.exists():
        return True
    return out.is_dir() and not any(out.iterdir())


def _session_line(s: dict) -> str:
    c = s["changes"]
    return (f"{s['persona_id']} {s['arm']} s{s['session']}: {s['score']['status']}, "
            f"notes +{len(c['added'])} ~{len(c['replaced'])} -{len(c['deleted'])}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    out = Path(args.out)
    try:
        if args.turns < 1:
            raise ValueError(f"--turns must be at least 1, got {args.turns}")
        start = date.fromisoformat(args.start) if args.start else date.today()
        personas = load_personas(Path(args.personas))
        if args.only is not None:
            personas = [p for p in personas if p.persona_id == args.only]
            if not personas:
                raise ValueError(f"--only {args.only}: no such persona in {args.personas}")
        policy = check_policy_file(Path(args.policy))
        if not _out_is_free(out):
            raise ValueError(f"{out} already holds files; give an --out that does not exist or is empty")
        path = crisis.load_path()
        if args.fake:
            memory: Any = FakeUserMemory()
            note_llm: Any = FakeNoteTaker()
            judge: Any = FakeJudge()

            def make_clients(persona: Persona, session: int, arm: str) -> Clients:
                return fake_clients(persona, session, arm, path)
        else:
            from loop.llm import client_from_env
            from puddles.eval.judge import judge_from_env
            from puddles.inference.note_store import PastUserMemory
            from puddles.llm import OpenAILLM

            def make_clients(persona: Persona, session: int, arm: str) -> Clients:
                return Clients(helper=client_from_env("HELPER"), checker=client_from_env("HELPER"),
                               user=client_from_env("SIMUSER"), path=path)

            make_clients(personas[0], 1, "notes_on")  # every client variable is checked before anything runs
            missing = [name for name in NOTES_KEYS if not os.environ.get(name)]
            if missing:
                raise ConfigError(f"{' and '.join(missing)} not set; the notes need them")
            if not os.environ.get("PUDDLES_AGENT_MODEL"):
                raise ConfigError("PUDDLES_AGENT_MODEL not set; the note-taker needs it")
            memory = PastUserMemory()
            note_llm = OpenAILLM()
            judge = judge_from_env()
    except (ConfigError, PolicyError, OSError, UnicodeDecodeError, ValueError, PastError, AgentFailed) as exc:
        _print_error(f"{type(exc).__name__}: {exc}")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    root = (out / "store").resolve()
    store = NoteStore(memory, root)
    from puddles import tracing

    tracing.setup(tracking_uri=f"sqlite:///{(out / 'mlflow.db').resolve()}")

    def on_session(s: dict) -> None:
        target = out / "sessions" / s["persona_id"] / s["arm"] / f"s{s['session']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        print(_session_line(s), flush=True)

    pilot = run_pilot(personas, make_clients=make_clients, policy=policy, store=store, note_llm=note_llm,
                      judge=judge, root=root, turns=args.turns, start=start, fake=args.fake,
                      policy_path=args.policy, on_session=on_session)
    (out / "pilot.json").write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(pilot), encoding="utf-8")
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
