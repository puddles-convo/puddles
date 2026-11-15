"""Talk to the robot in text: one message per line, through the crisis path, the helper and the release step.

    python -m scripts.chat --policy policies/policy-v1.md --user-id user-01 --remember-me no \
        --training no --root runs/chat
    python -m scripts.chat --fake < messages.txt

--fake uses a scripted helper, checker and note-taker, and an in-process past.dev: no keys, no network.
Without it both clients come from HELPER_* (the checker runs on the helper's endpoint), the notes live
on past.dev (PAST_API_KEY) and the note-taker is the OpenAI model (OPENAI_API_KEY, PUDDLES_AGENT_MODEL),
reached through the OpenAI SDK with --training yes and over plain HTTP with --training no. With --training
no nothing typed is kept, except the notes under notes/<user_id>/ when the person agreed to be remembered.

The fake checker answers `none` unless the message holds one of these markers, so a session can be
scripted from a file: [checker:concern], [checker:crisis-self], [checker:crisis-danger],
[checker:crisis-unsure], [checker:fail], [checker:garbage].

The fake note-taker adds and deletes nothing unless a message holds one of these: [note:<kind>:<fact>]
adds a note, [forget:<note_id>] deletes one, [forget-everything] forgets everything.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, Sequence, TextIO, TypeVar

from loop.errors import ConfigError, LLMError, PolicyError
from loop.llm import ChatClient, FakeClient
from puddles import tracing
from puddles.graph import check_policy_file
from puddles.inference import crisis
from puddles.inference.crisis import CrisisPath
from puddles.inference.live import PlainLLM, finish_session, start_notes, tracing_off
from puddles.inference.note_store import FakeUserMemory, NoteStore, PastUserMemory, UserMemory
from puddles.inference.notes import NotesResult
from puddles.inference.session_log import session_path
from puddles.inference.turn import NOT_MEASURED_MS, Session, TurnResult, end_session, run_turn
from puddles.llm import LLM, AgentFailed, OpenAILLM
from puddles.memory import PastError

QUIT = "/quit"
FAKE_HELPER_LINE = "That sounds like a lot to carry. What has been on your mind most?"
_MARKER = re.compile(r"\[checker:([a-z-]+)\]")
_NOTE = re.compile(r"\[note:([a-z_]+):([^\]]+)\]")
_FORGET = re.compile(r"\[forget:([a-z0-9-]+)\]")
_FORGET_ALL = "[forget-everything]"
NOTES_KEYS = ("PAST_API_KEY", "OPENAI_API_KEY")
T = TypeVar("T")


def _print_error(message: str) -> None:
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = _Parser(prog="python -m scripts.chat", description=__doc__.splitlines()[0])
    p.add_argument("--policy", default="policies/policy-v1.md", help="the policy file")
    p.add_argument("--user-id", default="user-01", help="who is talking")
    p.add_argument("--remember-me", choices=["yes", "no"], default="no", help="may the robot remember them")
    p.add_argument("--training", choices=["yes", "no"], default="no", help="may this session be used for training")
    p.add_argument("--root", default="runs/chat", help="where sessions/ goes")
    p.add_argument("--notes-root", default=None, help="where notes/<user_id>/ goes; default: the --root value")
    p.add_argument("--session-id", default=None, help="default: made from the clock")
    p.add_argument("--fake", action="store_true", help="scripted helper, checker and note-taker; no keys, no network")
    p.add_argument("--mlflow", default=None, help="MLflow tracking address; used only with --training yes")
    return p.parse_args(argv)


def _latest_message(content: str, path: CrisisPath) -> str:
    """The person's message out of a rendered checker prompt, so markers in earlier turns are ignored."""
    between = path.user_template.split("{recent_turns}", 1)[1].split("{text}", 1)[0]
    after = path.user_template.split("{text}", 1)[1]
    text = content.rsplit(between, 1)[-1]
    return text[:len(text) - len(after)] if after and text.endswith(after) else text


def fake_checker(path: CrisisPath) -> FakeClient:
    """A checker that answers in Ansh's format from a marker in the message; its lines go through his parser."""
    def answer(system: str, messages: list[dict[str, str]]) -> str:
        found = _MARKER.search(_latest_message(messages[-1]["content"], path))
        rule = found.group(1) if found else "none"
        if rule == "fail":
            raise LLMError("scripted checker failure")
        if rule == "garbage":
            return "scripted unreadable answer"
        if rule.startswith("crisis-"):
            return f"crisis | {rule[len('crisis-'):]} | scripted crisis"
        if rule == "concern":
            return "concern | - | scripted concern"
        return "none | - | scripted none"
    return FakeClient(answer)


def fake_helper() -> FakeClient:
    return FakeClient(lambda system, messages: FAKE_HELPER_LINE)


class FakeNoteTaker:
    """A note-taker that acts only on markers in the conversation; it is never shown the markers' meaning.

    It makes no request and keeps no record, so `finish_session` lets it through without training consent.
    With --training yes it is wrapped in `TracedLLM`, which does not carry the mark.
    """

    untraced = True

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        conversation = user.rsplit("<conversation>\n", 1)[-1].rsplit("\n</conversation>", 1)[0]
        actions: list[dict[str, Any]] = [{"action": "add", "kind": kind, "fact": fact.strip()}
                                         for kind, fact in _NOTE.findall(conversation)]
        actions += [{"action": "delete", "note_id": note_id} for note_id in _FORGET.findall(conversation)]
        return schema.model_validate({"forget_everything": _FORGET_ALL in conversation,  # type: ignore[attr-defined]
                                      "actions": actions})

    def structured_with_search(self, system: str, user: str, schema: type[T], search: Any, max_searches: int) -> T:
        raise NotImplementedError("the note-taker does not search")


def _notes_line(result: NotesResult) -> str:
    """Counts and note IDs only: never a note's text or an error's text, which could repeat the person's words."""
    pending = ", ".join(result.undeleted) or "none"
    if result.failed:
        return f"notes: the notes step failed; not yet confirmed deleted: {pending}"
    return (f"notes: {len(result.added)} added, {len(result.replaced)} replaced, {len(result.confirmed)} confirmed, "
            f"{len(result.deleted)} deleted; not yet confirmed deleted: {pending}")


def _status_line(result: TurnResult) -> str:
    risk = result.checked.risk if result.checked is not None else result.check_status
    t = result.timings_ms

    def ms(key: str) -> str:
        value = t.get(key, NOT_MEASURED_MS)
        return "-" if value == NOT_MEASURED_MS else f"{value}ms"

    keyword = "yes" if result.keyword is not None else "no"
    return (f"  [risk={risk} keyword={keyword} checker={ms('checker')} helper={ms('helper')} "
            f"total={ms('total')}]")


def main(argv: Sequence[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    args = _parse(argv)
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    root = Path(args.root)
    session_id = args.session_id or datetime.now().strftime("chat-%Y%m%d-%H%M%S")
    training = args.training == "yes"
    try:
        session_path(root, session_id)  # refuses an ID that is not a plain name
        policy = check_policy_file(Path(args.policy))
        path = crisis.load_path()
        helper_client: ChatClient
        checker_client: ChatClient
        if args.fake:
            helper_client, checker_client = fake_helper(), fake_checker(path)
        else:
            from loop.llm import client_from_env

            # Two clients on the same endpoint, so the two threads never share one client's state.
            helper_client, checker_client = client_from_env("HELPER"), client_from_env("HELPER")
        memory: UserMemory
        note_llm: LLM
        if args.fake:
            memory, note_llm = FakeUserMemory(), FakeNoteTaker()
            if training:  # traced with the rest of the session, by consent
                note_llm = tracing.TracedLLM(note_llm)
        else:
            missing = [name for name in NOTES_KEYS if not os.environ.get(name)]
            if missing:
                raise ConfigError(f"{' and '.join(missing)} not set; the notes need them")
            memory = PastUserMemory()
            # With consent, OpenAILLM, never wrapped: it reports its own requests and MLflow traces them after
            # tracing.setup. Without, never the OpenAI SDK: another session can turn autologging on at any time.
            note_llm = OpenAILLM() if training else PlainLLM.from_env()
        store = NoteStore(memory, Path(args.notes_root) if args.notes_root is not None else root)
    except (ConfigError, PolicyError, OSError, UnicodeDecodeError, ValueError, PastError, AgentFailed) as exc:
        detail = str(exc)
        about_file = isinstance(exc, (OSError, UnicodeDecodeError)) and args.policy not in detail
        where = f"{args.policy}: " if about_file else ""
        _print_error(f"{where}{type(exc).__name__}: {detail}")
        return 2

    if training:
        tracing.setup(tracking_uri=args.mlflow)
    else:
        tracing_off()  # whatever an earlier session in this process switched on
    session = Session(session_id=session_id, policy=policy, helper_client=helper_client,
                      checker_client=checker_client, path=path, user_id=args.user_id,
                      remember_me=args.remember_me == "yes", consent_for_training=training, root=root)
    try:
        session.user_notes = start_notes(store, args.user_id)
        print(f"notes: {len(session.user_notes)} loaded", file=sys.stderr)
    except Exception:  # no text: an error could repeat a note
        print("notes: the notes step failed; none loaded", file=sys.stderr)
    for raw in stdin:
        message = raw.strip()
        if message == QUIT:
            break
        if not message:
            continue
        result = run_turn(session, message)
        print("robot: " + result.reply.text, file=stdout)
        print(_status_line(result), file=stdout)
    try:
        record, notes = finish_session(session, store, note_llm)
    except Exception:  # the session is never lost: end it as it would have ended
        record, notes = end_session(session), NotesResult(failed=True)
    print(_notes_line(notes), file=sys.stderr)
    if record is not None:
        print(f"kept: {session_path(root, session_id)}", file=stdout)
    else:
        print("kept: nothing", file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
