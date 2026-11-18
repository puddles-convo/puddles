"""Run a scripted conversation through the live turn code and save what was said, turn by turn.

    HELPER_URL=... HELPER_MODEL=... python -m scripts.inference_run --policy policies/policy-v2.md \
        --messages fixtures/inference_messages.txt --out results/inference-run/transcript.jsonl
    python -m scripts.inference_run --fake --messages fixtures/inference_messages.txt --out /tmp/t.jsonl

Each message goes through run_turn exactly as in scripts/chat.py: keyword check, crisis checker and helper
in parallel, then the release step. Unlike chat.py there are no notes, no past.dev and no note-taker, so only
HELPER_* is needed. Each row also keeps the helper's draft, which shows how the prompt handled a turn even
when the release step spoke a fixed line instead. Use it for scripted test messages only, never for what a
real person said: the messages and replies are written to --out.

--deadline sets the crisis checker's deadline in seconds. Keep the path's 3 seconds on a real endpoint;
raise it only for a slow local model, where every turn would otherwise get the holding line.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Sequence

from loop.errors import ConfigError, PolicyError
from loop.llm import ChatClient
from loop.prompts import render_helper_system
from puddles.graph import check_policy_file
from puddles.inference import crisis
from puddles.inference.turn import Session, TurnResult, run_turn
from scripts.chat import fake_checker, fake_helper


class _Recording:
    """Keeps the helper's last draft, so a turn that released a fixed line still shows what the helper wrote."""

    def __init__(self, client: ChatClient) -> None:
        self.client = client
        self.last: str | None = None
        self.done = threading.Event()

    def next_turn(self) -> None:
        """A draft from an earlier turn that finishes late then lands in that turn's slots, not this one's."""
        self.last = None
        self.done = threading.Event()

    def complete(self, system: str, messages: list[dict[str, str]], *,
                 max_tokens: int, temperature: float = 0.7, seed: int | None = None) -> str:
        done = self.done
        try:
            text = self.client.complete(system, messages, max_tokens=max_tokens, temperature=temperature, seed=seed)
            if done is self.done:
                self.last = text
            return text
        finally:
            done.set()


def _from_helper(result: TurnResult) -> bool:
    """The release step's own rule: the draft is spoken only after a clean check."""
    checked = result.checked
    return (result.keyword is None and result.check_status == "answered" and checked is not None
            and checked.risk in ("none", "concern") and not result.helper_failed)


def _row(n: int, message: str, result: TurnResult, draft: str | None) -> dict[str, object]:
    checked = result.checked
    return {
        "turn": n,
        "person": message,
        "robot": result.reply.text,
        "released": "helper" if _from_helper(result) else "fixed line",
        "helper_draft": draft,
        "risk": checked.risk if checked is not None else None,
        "category": checked.category if checked is not None else None,
        "check_status": result.check_status,
        "keyword": result.keyword is not None,
        "timings_ms": result.timings_ms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", default="policies/policy-v2.md")
    p.add_argument("--messages", required=True, help="one message per line; blank lines are skipped")
    p.add_argument("--out", required=True, help="where the JSONL transcript goes")
    p.add_argument("--notes", default=None, help="optional file of notes about the person, one per line")
    p.add_argument("--deadline", type=float, default=None, help="crisis checker deadline, seconds")
    p.add_argument("--fake", action="store_true", help="scripted helper and checker, no network")
    args = p.parse_args(argv)
    try:
        policy = check_policy_file(Path(args.policy))
        path = crisis.load_path()
        helper_client: ChatClient
        checker_client: ChatClient
        if args.fake:
            helper_client, checker_client = fake_helper(), fake_checker(path)
        else:
            from loop.llm import client_from_env

            clients = client_from_env("HELPER"), client_from_env("HELPER")
            for client in clients:  # a slow model's requests must outlast the raised deadline
                client.timeout = max(client.timeout, args.deadline or 0.0)
            helper_client, checker_client = clients
        messages = [m.strip() for m in Path(args.messages).read_text(encoding="utf-8").splitlines() if m.strip()]
        notes = ([n.strip() for n in Path(args.notes).read_text(encoding="utf-8").splitlines() if n.strip()]
                 if args.notes else [])
    except (ConfigError, PolicyError, OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    recording = _Recording(helper_client)
    session = Session(session_id="inference-run", policy=policy, helper_client=recording,
                      checker_client=checker_client, path=path, user_notes=notes,
                      deadline_s=args.deadline, helper_wait_s=max(30.0, args.deadline or 0.0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    words = len(render_helper_system(policy, notes).split())
    print(f"policy {policy.version}, helper system message {words} words, {len(messages)} messages", file=sys.stderr)
    with out.open("w", encoding="utf-8") as fh:
        for n, message in enumerate(messages, 1):
            recording.next_turn()
            result = run_turn(session, message)
            recording.done.wait(session.helper_wait_s)  # a fixed line is released without waiting for the draft
            row = _row(n, message, result, recording.last)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"person: {message}\nrobot:  {result.reply.text}\n"
                  f"        [released={row['released']} risk={row['risk']} keyword={row['keyword']}]")
            if row["released"] != "helper":
                print(f"        helper draft, not spoken: {recording.last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
