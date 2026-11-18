"""Check the helper endpoint: one helper reply, one harmless and one crisis checker call, with latency.

    python -m scripts.modal_smoke                       # uses HELPER_URL, HELPER_MODEL, HELPER_API_KEY, HELPER_EXTRA_BODY
    python -m scripts.modal_smoke --warm 20 --concurrent 5 --check-auth

It uses the repo's own client, helper and crisis checker, so a pass means the inference code works
against this endpoint, not just that the server answers. Exits 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from loop.helper import helper_reply
from loop.policy import Policy, load_policy
from puddles.inference import crisis
from schemas import Turn

HELPER_MESSAGE = "I've been really stressed about work lately and I can't sleep."
HARMLESS_MESSAGE = "I had a long day at work, but I'm okay."
CRISIS_MESSAGE = "I don't want to be here anymore."
THINKING_MARKERS = ("<think>", "</think>")


def reply_problems(text: Optional[str]) -> list[str]:
    """Why a helper reply is unusable, or [] when it is fine."""
    if text is None or not text.strip():
        return ["the reply is empty"]
    lowered = text.lower()
    if any(marker in lowered for marker in THINKING_MARKERS):
        return ["the reply contains thinking text: turn thinking off (HELPER_EXTRA_BODY)"]
    return []


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _timed(fn: Callable[[], Any], clock: Callable[[], float]) -> tuple[Any, Optional[BaseException], float]:
    start = clock()
    try:
        return fn(), None, clock() - start
    except Exception as exc:  # reported, never hidden
        return None, exc, clock() - start


class Smoke:
    def __init__(self, helper_client, checker_client, policy: Policy, path: crisis.CrisisPath,
                 clock: Callable[[], float] = time.perf_counter):
        self.helper_client, self.checker_client = helper_client, checker_client
        self.policy, self.path, self.clock = policy, path, clock
        self.calls: list[dict] = []
        self.failures: list[str] = []

    def helper(self, name: str = "helper") -> float:
        turns = [Turn(speaker="user", text=HELPER_MESSAGE)]
        out, exc, seconds = _timed(lambda: helper_reply(self.helper_client, self.policy, turns,
                                                        session_id="smoke", turn=1), self.clock)
        text = getattr(out, "text", None)
        problems = [f"helper request failed: {type(exc).__name__}: {exc}"] if exc else reply_problems(text)
        self._record(name, seconds, problems, text)
        return seconds

    def checker(self, name: str, message: str, expect_crisis: bool) -> float:
        inp = crisis.build_input("smoke", 1, [], message)
        out, exc, seconds = _timed(lambda: crisis.run(self.checker_client, inp, self.path), self.clock)
        problems = []
        if exc is not None:
            problems.append(f"checker gave no usable answer ({getattr(exc, 'status', type(exc).__name__)}): {exc}")
        elif expect_crisis and out.risk != "crisis":
            problems.append(f"the crisis message was rated {out.risk!r}, not 'crisis'")
        if seconds > self.path.timeout_s:
            problems.append(f"checker took {seconds:.2f}s, over its {self.path.timeout_s:g}s limit")
        detail = None if out is None else f"{out.risk} | {out.category or '-'} | {out.reason}"
        self._record(name, seconds, problems, detail)
        return seconds

    def _record(self, name: str, seconds: float, problems: list[str], detail: Optional[str]) -> None:
        self.calls.append({"call": name, "seconds": round(seconds, 3), "ok": not problems, "detail": detail})
        self.failures += [f"{name}: {p}" for p in problems]


def _summary(values: list[float]) -> dict:
    return {"n": len(values), "median_s": round(statistics.median(values), 3),
            "p95_s": round(percentile(values, 95), 3), "max_s": round(max(values), 3)}


def run_smoke(helper_client, checker_client, policy: Policy, path: crisis.CrisisPath, *, warm: int = 0,
              concurrent: int = 0, clock: Callable[[], float] = time.perf_counter) -> dict:
    smoke = Smoke(helper_client, checker_client, policy, path, clock)
    report: dict = {"latency": {}}
    report["latency"]["first_helper_s"] = round(smoke.helper("helper (first call)"), 3)
    smoke.checker("checker, harmless message", HARMLESS_MESSAGE, expect_crisis=False)
    smoke.checker("checker, crisis message", CRISIS_MESSAGE, expect_crisis=True)
    if warm:
        helpers = [smoke.helper(f"helper warm {i + 1}") for i in range(warm)]
        checkers = [smoke.checker(f"checker warm {i + 1}", HARMLESS_MESSAGE, expect_crisis=False) for i in range(warm)]
        report["latency"]["helper_warm"] = _summary(helpers)
        report["latency"]["checker_warm"] = _summary(checkers)
    if concurrent:
        start = clock()
        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            list(pool.map(lambda i: smoke.helper(f"helper concurrent {i + 1}"), range(concurrent)))
        report["latency"]["concurrent"] = {"n": concurrent, "wall_s": round(clock() - start, 3)}
    report["calls"] = smoke.calls
    report["failures"] = smoke.failures
    report["passed"] = not smoke.failures
    return report


def check_auth(base_url: str, model: str, timeout: float = 120.0) -> tuple[bool, str]:
    """A request without the bearer key must be refused."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return False, f"a request without a key was accepted (HTTP {response.status})"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return True, f"refused without a key (HTTP {exc.code})"
        return False, f"unexpected HTTP {exc.code} without a key"


def main(argv: Optional[list[str]] = None) -> int:
    import os

    from loop.llm import client_from_env

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", default="policies/policy-v1.md")
    parser.add_argument("--warm", type=int, default=0, help="extra helper and checker calls for median and p95")
    parser.add_argument("--concurrent", type=int, default=0, help="helper calls sent at once")
    parser.add_argument("--check-auth", action="store_true", help="also check a request without the key is refused")
    args = parser.parse_args(argv)

    report = run_smoke(client_from_env("HELPER"), client_from_env("HELPER"), load_policy(args.policy),
                       crisis.load_path(), warm=args.warm, concurrent=args.concurrent)
    if args.check_auth:
        ok, message = check_auth(os.environ["HELPER_URL"], os.environ["HELPER_MODEL"])
        report["auth"] = message
        if not ok:
            report["failures"].append(f"auth: {message}")
            report["passed"] = False
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
