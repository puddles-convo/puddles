"""Run the critic, researcher or proposer on hand-written cases and print each answer beside its expectations.

    python -m scripts.try_agent critic fixtures/cases/critic.json --dry-run

For tuning a prompt: it sends the agent's own system prompt and rendered user message once per case and
shows the first answer as it came. It does not retry, and writes nothing: no decision log, no record file,
no memory. The case format is in docs/specs/case-files.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, ValidationError

from puddles import llm as llm_module
from puddles import usage
from puddles.agents import critic, proposer, researcher
from puddles.llm import LLM
from puddles.policy import PolicyEditError, parse_policy
from schemas import CriticInput, CriticOutput, ProposerInput, ProposerOutput, ResearcherInput, ResearcherOutput

Search = Callable[[str], list[dict]]

INPUTS: dict[str, type[BaseModel]] = {"critic": CriticInput, "researcher": ResearcherInput, "proposer": ProposerInput}
OUTPUTS: dict[str, type[BaseModel]] = {"critic": CriticOutput, "researcher": ResearcherOutput,
                                       "proposer": ProposerOutput}
SYSTEMS = {"critic": critic.SYSTEM, "researcher": researcher.SYSTEM, "proposer": proposer.SYSTEM}
CASE_KEYS = {"case_id", "what_it_tests", "input", "expect", "reviewed_by"}
EXPECT_KEYS = ("must_mention", "must_not_mention", "fields", "non_empty", "one_of", "max_chars")


class CaseFileError(ValueError):
    pass


def blank(value: Any) -> bool:
    """Null, empty or spaces, all the way down: a list or object is blank when it is empty or anything in it,
    at any depth, is blank. So one source with a blank url makes the whole `sources` list blank.
    Numbers and booleans are never blank."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        items = value.values() if isinstance(value, dict) else value
        return not value or any(blank(v) for v in items)
    return False


def _not_applied(name: str, values: dict[str, Any], allowed_removal: bool) -> Optional[str]:
    """Why a non_empty line does not apply to this answer, or None when it does."""
    # A removal has no replacement text. Only null or "" count as none: the proposer's check and apply_edit
    # both refuse a removal whose rule_text is truthy, and "   " is truthy, so it gets the line and is EMPTY.
    if name == "rule_text" and allowed_removal and values.get("rule_text") in (None, ""):
        return "not applied to a removal"
    if name == "rule_id" and values.get("action") == "add":  # ProposerOutput: an addition has no rule ID yet
        return "not applied to an addition"
    return None


def _vacuous(expect: dict[str, Any]) -> list[str]:
    """Entries that could never reject an answer: an empty list or object under any key, a blank phrase or field."""
    found = []
    for key, value in expect.items():
        if isinstance(value, (list, dict)) and not value:
            found.append(f"expect.{key} is empty")
        elif isinstance(value, list):
            found += [f"expect.{key} holds a blank entry" for v in value if isinstance(v, str) and not v.strip()][:1]
        elif isinstance(value, dict):
            found += [f"expect.{key} names a blank field" for k in value if not k.strip()][:1]
    return found


def _check_expect(expect: Any, where: str) -> None:
    """One check of the whole expect: every entry can reject an answer, so the case can fail."""
    if not isinstance(expect, dict) or not expect or set(expect) - set(EXPECT_KEYS):
        raise CaseFileError(f"{where}: expect needs at least one of {list(EXPECT_KEYS)} and nothing else")
    if vacuous := _vacuous(expect):
        raise CaseFileError(f"{where}: an expectation that cannot fail: {'; '.join(vacuous)}")
    for key in ("must_mention", "must_not_mention", "non_empty"):
        if key in expect and not (isinstance(expect[key], list) and all(isinstance(s, str) for s in expect[key])):
            raise CaseFileError(f"{where}: expect.{key} must be a list of strings")
    if "fields" in expect and not isinstance(expect["fields"], dict):
        raise CaseFileError(f"{where}: expect.fields must be an object")
    one_of = expect.get("one_of", {})
    if not isinstance(one_of, dict) or not all(isinstance(v, list) and v for v in one_of.values()):
        raise CaseFileError(f"{where}: expect.one_of must map each field to a non-empty list of values")
    max_chars = expect.get("max_chars", {})
    if not isinstance(max_chars, dict) or not all(type(n) is int and n >= 0 for n in max_chars.values()):
        raise CaseFileError(f"{where}: expect.max_chars must map each field to a whole number of characters")


def load_cases(agent: str, path: Path) -> list[dict[str, Any]]:
    """Read and validate a whole case file, so a bad case stops the run before any call is made.

    Raises on anything wrong with the file; `main` turns every such failure into exit code 2.
    """
    raw = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(raw, list) or not raw:
        raise CaseFileError(f"{path} must hold a non-empty JSON list of cases")
    seen: set[str] = set()
    cases = []
    for n, case in enumerate(raw, 1):
        where = f"{path} case {n}"
        if not isinstance(case, dict) or set(case) != CASE_KEYS:
            raise CaseFileError(f"{where}: a case has exactly the keys {sorted(CASE_KEYS)}")
        cid = case["case_id"]
        if not isinstance(cid, str) or not cid:
            raise CaseFileError(f"{where}: case_id must be a non-empty string")
        if cid in seen:
            raise CaseFileError(f"{where}: case_id {cid} is used twice")
        seen.add(cid)
        _check_expect(case["expect"], f"{where} ({cid})")
        try:
            inp = INPUTS[agent].model_validate(case["input"])
        except ValidationError as e:
            raise CaseFileError(f"{where} ({cid}): input is not a valid {INPUTS[agent].__name__}: {e}") from e
        try:
            user = render(agent, inp)
        except PolicyEditError as e:
            raise CaseFileError(f"{where} ({cid}): policy_text cannot be parsed: {e}") from e
        cases.append({**case, "parsed": inp, "user": user})
    return cases


def render(agent: str, inp: Any) -> str:
    """The agent's own render_user; the proposer's takes the version from the policy's first line."""
    if agent == "critic":
        return critic.render_user(inp)
    if agent == "researcher":
        return researcher.render_user(inp)
    return proposer.render_user(inp, parse_policy(inp.policy_text).version)


def ask(agent: str, user: str, model: LLM, search: Search) -> BaseModel:
    """One call, no retry. The agent's name selects its output limit; no meter, so nothing is logged."""
    with usage.metering(None, agent):
        if agent == "researcher":
            return model.structured_with_search(SYSTEMS[agent], user, OUTPUTS[agent], search,
                                                researcher.MAX_SEARCHES)
        return model.structured(SYSTEMS[agent], user, OUTPUTS[agent])


def judge(answer: BaseModel, expect: dict[str, Any]) -> list[tuple[bool, str]]:
    """One (met, line) per expectation."""
    text = answer.model_dump_json().lower()
    values = answer.model_dump(mode="json")
    # The blank test is on the answer as the model gave it, at every depth: an optional field it left out
    # and the model filled with its default is not there to be empty (case-files.md, section 1).
    given = answer.model_dump(mode="json", exclude_unset=True)
    lines = []
    for phrase in expect.get("must_mention", []):
        met = phrase.lower() in text
        lines.append((met, f"{'ok' if met else 'MISSING'} must_mention {json.dumps(phrase)}"))
    for phrase in expect.get("must_not_mention", []):
        met = phrase.lower() not in text
        lines.append((met, f"{'ok' if met else 'PRESENT'} must_not_mention {json.dumps(phrase)}"))
    for name, wanted in expect.get("fields", {}).items():
        got = values.get(name, "(no such field)")
        met = name in values and got == wanted
        detail = f"== {json.dumps(wanted)}" if met else f"expected {json.dumps(wanted)}, got {json.dumps(got)}"
        lines.append((met, f"{'ok' if met else 'DIFFERENT'} fields.{name} {detail}"))
    one_of = expect.get("one_of", {})
    allowed_removal = "remove" in one_of.get("action", []) and values.get("action") == "remove"
    for name in expect.get("non_empty", []):
        if why := _not_applied(name, values, allowed_removal):
            lines.append((True, f"ok non_empty {name} ({why})"))
            continue
        met = name in given and not blank(given[name])  # a field named here must still be given
        lines.append((met, f"{'ok' if met else 'EMPTY'} non_empty {name}"))
    for name, allowed in one_of.items():
        got = values.get(name, "(no such field)")
        met = name in values and got in allowed
        lines.append((met, f"{'ok' if met else 'NOT ONE OF'} one_of.{name} {json.dumps(got)} in {json.dumps(allowed)}"))
    for name, limit in expect.get("max_chars", {}).items():
        got = values.get(name)
        if isinstance(got, str):
            met, detail = len(got) <= limit, f"{len(got)} characters, limit {limit}"
        else:  # null or missing meets it only for an allowed removal, which has no text
            met = allowed_removal
            detail = f"{json.dumps(got)} for {json.dumps(values.get('action'))}, limit {limit}"
        lines.append((met, f"{'ok' if met else 'TOO LONG'} max_chars.{name} {detail}"))
    return lines


def missing_config(agent: str, have_llm: bool, have_search: bool) -> Optional[str]:
    if not have_llm and not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set; pass --dry-run to only print the messages"
    if agent == "researcher" and not have_search and not os.environ.get("EXA_API_KEY"):
        return "EXA_API_KEY is not set; the researcher needs it to search"
    return None


def main(argv: Sequence[str] | None = None, llm: LLM | None = None,
         search: Callable[[str], list[dict]] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.try_agent", description=__doc__.splitlines()[0])
    parser.add_argument("agent", help="critic, researcher or proposer")
    parser.add_argument("case_file", type=Path)
    parser.add_argument("--case", help="run only the case with this ID")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the messages; call nothing")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    if args.agent not in INPUTS:
        print(f"unknown agent {args.agent!r}: choose critic, researcher or proposer", file=sys.stderr)
        return 2
    try:  # one boundary: whatever stops the file being read or validated is exit 2, never a traceback
        cases = load_cases(args.agent, args.case_file)
    except Exception as e:
        print(f"cannot use {args.case_file}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if args.case:
        cases = [c for c in cases if c["case_id"] == args.case]
        if not cases:
            print(f"no case {args.case!r} in {args.case_file}", file=sys.stderr)
            return 2

    if args.dry_run:
        for c in cases:
            print(f"=== {c['case_id']}: {c['what_it_tests']}")
            print(f"--- system ---\n{SYSTEMS[args.agent]}\n--- user ---\n{c['user']}\n")
        print(f"{len(cases)} cases validated; nothing was sent")
        return 0

    problem = missing_config(args.agent, llm is not None, search is not None)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if llm is None:
        try:
            llm = llm_module.OpenAILLM()
        except Exception as e:  # e.g. PUDDLES_AGENT_MODEL not set
            print(f"cannot build the model: {e}", file=sys.stderr)
            return 2
    if search is None:
        from puddles import exa
        search = exa.search

    met_count = 0
    for c in cases:
        print(f"=== {c['case_id']}: {c['what_it_tests']}")
        try:
            answer = ask(args.agent, c["user"], llm, search)
        except Exception as e:  # a failed call is a case that did not meet its expectations
            print(f"answer: (none) {type(e).__name__}: {e}")
            print("FAILED the call did not return an answer\n")
            continue
        print(f"answer: {answer.model_dump_json(indent=2)}")
        results = judge(answer, c["expect"])
        for _, line in results:
            print(f"  {line}")
        print()
        met_count += all(met for met, _ in results)
    print(f"{met_count} of {len(cases)} cases met their expectations")
    return 0 if met_count == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
