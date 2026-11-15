"""Talk to the robot in text: one message per line, through the crisis path, the helper and the release step.

    python -m scripts.chat --policy policies/policy-v1.md --user-id user-01 --remember-me no \
        --training no --root runs/chat
    python -m scripts.chat --fake < messages.txt

--fake uses a scripted helper and checker: no keys, no network. Without it both clients come from
HELPER_* (the checker runs on the helper's endpoint). With --training no nothing typed is kept.

The fake checker answers `none` unless the message holds one of these markers, so a session can be
scripted from a file: [checker:concern], [checker:crisis-self], [checker:crisis-danger],
[checker:crisis-unsure], [checker:fail], [checker:garbage].
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Sequence, TextIO

from loop.errors import ConfigError, LLMError, PolicyError
from loop.llm import ChatClient, FakeClient
from puddles import tracing
from puddles.graph import check_policy_file
from puddles.inference import crisis
from puddles.inference.crisis import CrisisPath
from puddles.inference.session_log import session_path
from puddles.inference.turn import NOT_MEASURED_MS, Session, TurnResult, end_session, run_turn

QUIT = "/quit"
FAKE_HELPER_LINE = "That sounds like a lot to carry. What has been on your mind most?"
_MARKER = re.compile(r"\[checker:([a-z-]+)\]")


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
    p.add_argument("--session-id", default=None, help="default: made from the clock")
    p.add_argument("--fake", action="store_true", help="scripted helper and checker; no keys, no network")
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
    except (ConfigError, PolicyError, OSError, UnicodeDecodeError, ValueError) as exc:
        detail = str(exc)
        about_file = isinstance(exc, (OSError, UnicodeDecodeError)) and args.policy not in detail
        where = f"{args.policy}: " if about_file else ""
        _print_error(f"{where}{type(exc).__name__}: {detail}")
        return 2

    if training:
        tracing.setup(tracking_uri=args.mlflow)
    session = Session(session_id=session_id, policy=policy, helper_client=helper_client,
                      checker_client=checker_client, path=path, user_id=args.user_id,
                      remember_me=args.remember_me == "yes", consent_for_training=training, root=root)
    for raw in stdin:
        message = raw.strip()
        if message == QUIT:
            break
        if not message:
            continue
        result = run_turn(session, message)
        print("robot: " + result.reply.text, file=stdout)
        print(_status_line(result), file=stdout)
    record = end_session(session)
    if record is not None:
        print(f"kept: {session_path(root, session_id)}", file=stdout)
    else:
        print("kept: nothing", file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
