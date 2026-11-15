"""Researcher (step 5): searches Exa for how counsellors handle the pattern."""
import json
from pathlib import Path
from typing import Callable

from puddles.agents import bullet_lines, system_prompt
from puddles.decision_log import DecisionLog
from puddles.llm import LLM, AgentFailed, call_checked
from puddles.memory import Memory, now, recall_records
from schemas import FailurePattern, PatternBrief, ResearcherInput, ResearcherOutput, ResearchNote

SYSTEM = system_prompt("t5-researcher.md")
IDENTITY = "researcher"
MAX_SEARCHES = 3
Search = Callable[[str], list[dict]]


def normalise(query: str) -> str:
    return " ".join(query.lower().split())


def build_input(pattern: FailurePattern, log: DecisionLog, memory: Memory, root: Path) -> ResearcherInput:
    queries, summary = log.earlier_queries(pattern.pattern_id)
    query = f"Research already done for: {pattern.name}. {pattern.description}"
    for note in recall_records(memory, log, query, IDENTITY, root):
        if isinstance(note, ResearchNote) and log.pattern_identity(note.pattern_id) == pattern.pattern_id:
            queries += [q for q in note.queries if q not in queries]
    return ResearcherInput(pattern=PatternBrief(id=pattern.pattern_id, name=pattern.name,
                                                description=pattern.description),
                           earlier_queries=queries, older_queries_summary=summary)


def render_user(inp: ResearcherInput) -> str:
    summary = f"\n{inp.older_queries_summary}" if inp.older_queries_summary else ""
    return (f'<failure_pattern id="{inp.pattern.id}">\n{inp.pattern.name}: {inp.pattern.description}\n'
            f"</failure_pattern>\n\n<earlier_queries>\n{bullet_lines(inp.earlier_queries)}{summary}\n</earlier_queries>")


def check(out: ResearcherOutput, earlier: set[str], returned_urls: set[str], searched: set[str]) -> list[str]:
    """`earlier` holds every query already run, normalised: the whole log, not only those shown."""
    problems = []
    unrun = [q for q in out.queries if normalise(q) not in searched]
    if unrun:
        problems.append(f"these queries were not searched: {unrun}")
    if not 1 <= len(out.queries) <= MAX_SEARCHES:
        problems.append(f"run between 1 and {MAX_SEARCHES} searches")
    repeats = [q for q in out.queries if normalise(q) in earlier]
    if repeats:
        problems.append(f"these queries were run in earlier rounds: {repeats}")
    invented = [s.url for s in out.sources if s.url not in returned_urls]
    if invented:
        problems.append(f"these URLs were not returned by any search: {invented}")
    if any("\n" in s.technique or not s.technique.strip() for s in out.sources):
        problems.append("each technique must be one line")
    return problems


def run(inp: ResearcherInput, llm: LLM, search: Search,
        decisions: DecisionLog) -> tuple[ResearcherOutput, list[dict]]:
    """Returns the output and every search made, so the results can be saved beside the note.

    A query already run in an earlier round is not sent to Exa; the model is told and can write another.
    """
    user = render_user(inp)
    earlier = {normalise(q) for q in decisions.all_queries() + inp.earlier_queries}
    log: list[dict] = []
    returned: set[str] = set()
    searched: set[str] = set()

    def logged_search(query: str) -> list[dict]:
        if normalise(query) in earlier:
            return [{"error": "this query was run in an earlier round and was not searched again; "
                              "write a different one"}]
        try:
            results = search(query)
        except RuntimeError as exc:  # puddles.exa.search, after Exa failed twice
            raise AgentFailed(f"the search failed: {exc}") from exc
        log.append({"query": query, "results": results})
        returned.update(r["url"] for r in results)
        searched.add(normalise(query))
        return results

    out = call_checked(
        lambda fb: llm.structured_with_search(SYSTEM, user + fb, ResearcherOutput, logged_search,
                                              MAX_SEARCHES),
        lambda o: check(o, earlier, returned, searched), "researcher")
    return out, log


def to_record(out: ResearcherOutput, pattern: FailurePattern, round_: int) -> ResearchNote:
    return ResearchNote(id=f"research-r{round_:02d}", written_by=IDENTITY, created_at=now(),
                        parent_ids=[pattern.id], round=round_, pattern_id=pattern.pattern_id,
                        queries=out.queries, sources=out.sources)


def save_search_log(searches: list[dict], round_: int, root: Path) -> Path:
    path = root / f"records/round-{round_:02d}/research-r{round_:02d}.searches.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(searches, indent=2) + "\n")
    return path
