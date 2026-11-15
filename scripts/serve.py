"""Serve the turn service over HTTP: the checked text path for the robot's brain. Contract: docs/specs/turn-service.md.

    PUDDLES_SERVICE_TOKEN=... python -m scripts.serve --policy policies/policy-v1.md --root runs/serve
    PUDDLES_SERVICE_TOKEN=... python -m scripts.serve --policy policies/policy-v1.md --root runs/serve --fake

The token comes from PUDDLES_SERVICE_TOKEN only, never from an argument, and is never printed or logged.
--fake uses the scripted helper, checker and note-taker of scripts/chat.py and an in-process past.dev: no
keys, no network. Without it the clients come from HELPER_*, the notes live on past.dev (PAST_API_KEY) and
the note-taker is the OpenAI model (OPENAI_API_KEY, PUDDLES_AGENT_MODEL); one of each is built at startup so
a missing variable stops the script before anything listens. Idle sessions are swept on a timer.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Mapping, NoReturn, Sequence

from loop.errors import ConfigError, PolicyError
from loop.llm import ChatClient
from puddles.graph import check_policy_file
from puddles.inference import crisis
from puddles.inference.live import PlainLLM
from puddles.inference.note_store import FakeUserMemory, NoteStore, PastUserMemory, UserMemory
from puddles.inference.service import ServiceConfig, create_app
from puddles.llm import LLM, AgentFailed, OpenAILLM
from puddles.memory import PastError
from scripts.chat import FakeNoteTaker, fake_checker, fake_helper

TOKEN_VAR = "PUDDLES_SERVICE_TOKEN"
NOTES_KEYS = ("PAST_API_KEY", "OPENAI_API_KEY")


def _print_error(message: str) -> None:
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _positive(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a number above zero")
    return value


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = _Parser(prog="python -m scripts.serve", description=(__doc__ or "").splitlines()[0])
    p.add_argument("--policy", required=True, help="the policy file, read again at the start of every session")
    p.add_argument("--root", required=True, help="where sessions/ goes")
    p.add_argument("--notes-root", default=None, help="where notes/<user_id>/ goes; default: the --root value")
    p.add_argument("--host", default="127.0.0.1", help="the address to listen on")
    p.add_argument("--port", type=int, default=8765, help="the port to listen on")
    p.add_argument("--mlflow", default=None, help="MLflow tracking address; used only for sessions with consent")
    p.add_argument("--sweep-every", type=_positive, default=30.0, help="seconds between sweeps of idle sessions")
    p.add_argument("--fake", action="store_true", help="scripted helper, checker and note-taker; no keys, no network")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace, env: Mapping[str, str]) -> ServiceConfig:
    """Everything the service needs, checked before anything listens. Raises on a missing piece."""
    token = env.get(TOKEN_VAR)
    if not token:
        raise ConfigError(f"{TOKEN_VAR} is not set")
    policy_path = Path(args.policy)
    check_policy_file(policy_path)  # refused at startup, not at the first person's session
    path = crisis.load_path()
    root = Path(args.root)
    notes_root = Path(args.notes_root) if args.notes_root is not None else root
    memory: UserMemory
    client_factory: Callable[[], ChatClient]
    note_llm_for: Callable[[bool], LLM]
    if args.fake:
        calls = itertools.count()
        lock = threading.Lock()

        def client_factory() -> ChatClient:
            with lock:  # the service asks twice per session, helper then checker
                first = next(calls) % 2 == 0
            return fake_helper() if first else fake_checker(path)

        def note_llm_for(consent: bool) -> LLM:
            if not consent:
                return FakeNoteTaker()
            from puddles import tracing  # imports MLflow, so only for a session with consent

            return tracing.TracedLLM(FakeNoteTaker())

        memory = FakeUserMemory()
    else:
        from loop.llm import client_from_env

        missing = [name for name in NOTES_KEYS if not env.get(name)]
        if missing:
            raise ConfigError(f"{' and '.join(missing)} not set; the notes need them")
        client_from_env("HELPER")  # one of each, once, so a missing variable is found now
        OpenAILLM()
        PlainLLM.from_env()
        memory = PastUserMemory()

        def client_factory() -> ChatClient:
            return client_from_env("HELPER")

        def note_llm_for(consent: bool) -> LLM:
            return OpenAILLM() if consent else PlainLLM.from_env()

    return ServiceConfig(token=token, policy_path=policy_path, root=root, store=NoteStore(memory, notes_root),
                         client_factory=client_factory, note_llm_for=note_llm_for, path=path,
                         mlflow_uri=args.mlflow)


def _sweeper(app: object, every: float, stop: threading.Event) -> threading.Thread:
    def loop() -> None:
        while not stop.wait(every):
            app.state.service.sweep()  # type: ignore[attr-defined]  # sweep raises nothing

    thread = threading.Thread(target=loop, name="puddles-sweep", daemon=True)
    thread.start()
    return thread


def main(argv: Sequence[str] | None = None, *, run: Callable[..., None] | None = None) -> int:
    args = _parse(argv)
    try:
        config = build_config(args, os.environ)
        app = create_app(config)
    except (ConfigError, PolicyError, OSError, UnicodeDecodeError, ValueError, PastError, AgentFailed) as exc:
        detail = str(exc)
        about_file = isinstance(exc, (OSError, UnicodeDecodeError)) and args.policy not in detail
        where = f"{args.policy}: " if about_file else ""
        _print_error(f"{where}{type(exc).__name__}: {detail}")
        return 2
    if run is None:
        import uvicorn

        run = uvicorn.run
    stop = threading.Event()
    _sweeper(app, args.sweep_every, stop)
    try:
        run(app, host=args.host, port=args.port)
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
