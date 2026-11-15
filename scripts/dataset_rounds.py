"""Training rounds driven by the ESConv dataset, for when live training runs fail.

Each round goes through the same code path as a live round (records, decision log, policy files,
.meta.json, memory copies, replay check), but every model step is replaced by a step that reads
ESConv instead:

- critic: draws a seeded batch of ESConv training conversations, takes the lowest-rated ones, and
  names the catalogue pattern seen most often among them that also lowers seeker ratings across
  the whole training split
- researcher: the pattern's counselling sources, plus the training-split figures as evidence
- proposer: the one rule edit written for that pattern
- gate: compares seeker ratings on the held-out validation split for conversations that follow the
  new rule against those that break it. It keeps the edit only if quality rises by more than the
  noise and neither empathy nor relevance falls by more than the noise. It runs no conversations
  and no safety cases; the Safety section is checked unchanged.

After the last round, held-out.json reports the ESConv test split for each kept rule and maps the outside
sets in the README (CounselBench-Adv, VERA-MH) to the policy rules meant to handle them. None of it feeds a round.

Usage (no keys needed; past.dev is used only with --past and PAST_API_KEY):

    python -m scripts.dataset_rounds --root runs/esconv
"""
import argparse
import csv
import io
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Literal, Optional
from urllib.request import urlopen

from puddles import replay
from puddles.decision_log import DecisionLog
from puddles.memory import FakeMemory, Memory, PastClient, now
from puddles.policy import (
    apply_edit,
    candidate_version,
    estimate_tokens,
    parse_policy,
    render_policy,
    safety_section,
)
from puddles.records import save_record
from puddles.round import RoundResult, finish_round, memory_trace, prepare_round
from schemas import (
    FailurePattern,
    GateOutcome,
    LogSafety,
    LogScores,
    PolicyEdit,
    ResearchNote,
    Scores,
    Source,
)

ESCONV = "thu-coai/esconv"
ESCONV_REVISION = "32c54c15e7651cd68dfab18eee39f63ff0cc1e82"
SPLITS = {"train": ("train.txt", 910), "validation": ("valid.txt", 195), "test": ("test.txt", 195)}
CBADV = "izi-ano/CounselBench-Adv"
CBADV_REVISION = "57a2e2d5b709208197bbf8ff55124d4844067da3"
VERA = "SpringCare/VERA-MH"
VERA_REVISION = "2ea4823cade663da3143e15dd7f63641c175502d"
# Policy rules meant to handle each held-out category: a base rule ID, or a catalogue pattern whose kept rule
# counts (its ID depends on the run). A rule being present is not a test of the helper.
CBADV_RULES = {"apathetic": ["C1", "no-reassurance"], "assumptions": ["C1", "C3"], "symptoms": ["S2", "C8"],
               "judgmental": ["C6", "no-reassurance"], "medication": ["S2"], "therapy": ["S3", "C8"]}
VERA_RULES = {"None": ["C8"], "Low": ["S1", "C8"], "High": ["S1"], "Immediate": ["S1"]}
ESCONV_URL = f"https://huggingface.co/datasets/{ESCONV}"
ESCONV_PAPER = "https://arxiv.org/abs/2106.01144"

BATCH = 60  # training conversations drawn per round
FOR_CRITIC = 10  # lowest-rated conversations the critic reads
MIN_SEEN = 2  # a pattern must appear in at least this many of them
MIN_TRAIN_DROP = 0.05  # and lower mean quality on the training split by at least this much
NOISE_PARTS = 3  # validation split cut into this many seeded parts to measure noise
UNDERSTANDING = {"Question", "Restatement or Paraphrasing", "Reflection of feelings"}


@dataclass
class Conversation:
    id: str
    empathy: int
    relevance: int
    helper: list[tuple[str, Optional[str]]]  # (text, strategy) for each helper turn

    @property
    def quality(self) -> float:
        return (self.empathy + self.relevance) / 2

    def share(self, strategy: str) -> float:
        return sum(s == strategy for _, s in self.helper) / len(self.helper) if self.helper else 0.0

    def has(self, strategy: str) -> bool:
        return any(s == strategy for _, s in self.helper)


def _fetch(url: str, cache: str) -> str:
    path = Path(os.getenv("PUDDLES_CACHE", Path.home() / ".cache" / "puddles")) / cache
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as response:
            data = response.read()
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp.write(data)
        Path(temp.name).replace(path)
    return path.read_text(encoding="utf-8-sig")


def _download(filename: str) -> str:
    return _fetch(f"{ESCONV_URL}/resolve/{ESCONV_REVISION}/{filename}", f"{ESCONV}/{ESCONV_REVISION}/{filename}")


def parse_esconv(text: str, split: str) -> list[Conversation]:
    """Conversations with both seeker ratings; the others cannot be scored and are left out."""
    out = []
    for i, line in enumerate(row for row in text.split("\n") if row.strip()):
        row = json.loads(line)
        seeker = row["survey_score"]["seeker"]
        try:
            empathy, relevance = int(seeker["empathy"]), int(seeker["relevance"])
        except (KeyError, TypeError, ValueError):
            continue
        helper = [(t["text"], t.get("strategy")) for t in row["dialog"] if t["speaker"] == "sys"]
        out.append(Conversation(f"conv-esconv-{split}-{i:04d}", empathy, relevance, helper))
    return out


def load_split(split: Literal["train", "validation", "test"]) -> list[Conversation]:
    filename, expected = SPLITS[split]
    text = _download(filename)
    rows = sum(1 for row in text.split("\n") if row.strip())
    if rows != expected:
        raise ValueError(f"ESConv {split}: expected {expected} rows, got {rows}")
    return parse_esconv(text, split)


def _advice_before_understanding(c: Conversation) -> bool:
    strategies = [s for _, s in c.helper]
    first_advice = next((i for i, s in enumerate(strategies) if s == "Providing Suggestions"), None)
    first_ask = next((i for i, s in enumerate(strategies) if s in UNDERSTANDING), None)
    return first_advice is not None and (first_ask is None or first_advice < first_ask)


@dataclass
class Fix:
    """One catalogue pattern: how to spot it in ESConv, and the research and edit that address it."""

    name: str
    description: str
    present: Callable[[Conversation], bool]
    queries: list[str]
    sources: list[Source]
    action: Literal["add", "reword"]
    rule_text: str
    summary: str
    reword: Optional[str] = None  # the rule ID a rewording targets


CATALOGUE = [
    Fix("no-reassurance",
        "The helper never affirms the person's strengths or reassures them during the conversation.",
        lambda c: not c.has("Affirmation and Reassurance"),
        ["counselling affirmation and reassurance emotional support", "OARS affirmations motivational interviewing"],
        [Source(url=ESCONV_PAPER, title="Towards Emotional Support Dialog Systems (Liu et al., 2021)",
                technique="Affirmation and Reassurance is one of the eight support strategies: affirm the "
                          "person's strengths, motivation and capabilities and offer honest reassurance."),
         Source(url="https://en.wikipedia.org/wiki/Motivational_interviewing", title="Motivational interviewing",
                technique="OARS: open questions, affirmations, reflective listening and summaries; "
                          "affirmations recognise the person's strengths and efforts.")],
        "add",
        "At least once, honestly affirm something the person is doing or has done, such as reaching out or "
        "coping so far, because recognising effort helps people feel supported.",
        "Add a rule to affirm the person's effort or strengths at least once."),
    Fix("information-heavy",
        "More than a quarter of the helper's turns hand out facts or information rather than listening.",
        lambda c: c.share("Information") > 0.25,
        ["emotional support giving too much information", "active listening instead of lecturing"],
        [Source(url=ESCONV_PAPER, title="Towards Emotional Support Dialog Systems (Liu et al., 2021)",
                technique="Support moves through exploration and comforting before action; information "
                          "belongs to the action stage, after the problem is understood."),
         Source(url="https://en.wikipedia.org/wiki/Active_listening", title="Active listening",
                technique="Keep attention on the speaker's experience and reflect it back rather than "
                          "steering the conversation with your own content.")],
        "add",
        "Share facts or information only when the person asks or it answers their situation directly, one point "
        "at a time, because replies full of information feel like a lecture.",
        "Add a rule limiting unrequested information to one relevant point."),
    Fix("no-reflection-of-feelings",
        "The helper never reflects the person's feelings back to them.",
        lambda c: not c.has("Reflection of feelings"),
        ["reflection of feelings counselling skill", "reflective listening naming emotions"],
        [Source(url="https://en.wikipedia.org/wiki/Reflective_listening", title="Reflective listening",
                technique="Reflect both the content and the feeling of what the speaker said, so they can "
                          "confirm or correct it."),
         Source(url=ESCONV_PAPER, title="Towards Emotional Support Dialog Systems (Liu et al., 2021)",
                technique="Reflection of feelings is a core comforting strategy: articulate and describe the "
                          "seeker's feelings.")],
        "reword",
        "Begin by reflecting what the person said in your own words and name the feeling you hear, so they know "
        "they were heard before anything else happens.",
        "Reword C1 to name the person's feeling as well as their words.",
        reword="C1"),
    Fix("suggestion-heavy",
        "More than a third of the helper's turns are suggestions.",
        lambda c: c.share("Providing Suggestions") > 0.35,
        ["too much advice emotional support conversation", "motivational interviewing righting reflex"],
        [Source(url="https://en.wikipedia.org/wiki/Motivational_interviewing", title="Motivational interviewing",
                technique="Draw out the person's own reasons and ideas instead of supplying solutions."),
         Source(url=ESCONV_PAPER, title="Towards Emotional Support Dialog Systems (Liu et al., 2021)",
                technique="Providing Suggestions belongs to the action stage and should be offered carefully "
                          "after exploration and comforting.")],
        "add",
        "When you do suggest something, offer one small thing and ask whether it fits, rather than listing options.",
        "Add a rule to offer one suggestion at a time and check it fits."),
    Fix("advice-before-understanding",
        "The helper gives a suggestion before asking, restating or reflecting anything about the situation.",
        _advice_before_understanding,
        ["explore before advising counselling", "helping skills exploration stage before action"],
        [Source(url=ESCONV_PAPER, title="Towards Emotional Support Dialog Systems (Liu et al., 2021)",
                technique="Emotional support follows Hill's Helping Skills: exploration, then comforting, then "
                          "action; suggestions come after the problem is explored."),
         Source(url="https://en.wikipedia.org/wiki/Clara_Hill", title="Clara Hill",
                technique="The Helping Skills model has three stages, exploration, insight and action, in "
                          "that order.")],
        "add",
        "In your first two replies, ask one open question about their situation and do not give suggestions yet, "
        "because advice that comes before understanding rarely fits.",
        "Add a rule to explore the situation before any suggestion."),
]
BY_NAME = {f.name: f for f in CATALOGUE}


def mean_scores(convs: list[Conversation]) -> Scores:
    e = statistics.mean(c.empathy for c in convs)
    r = statistics.mean(c.relevance for c in convs)
    return Scores(empathy=round(e, 3), relevance=round(r, 3), quality=round((e + r) / 2, 3))


def mean_quality(convs: list[Conversation]) -> float:
    return statistics.mean(c.quality for c in convs)


def train_drop(fix: Fix, train: list[Conversation]) -> float:
    """How much lower mean quality is with the pattern than without it, on the training split."""
    with_, without = [c for c in train if fix.present(c)], [c for c in train if not fix.present(c)]
    if not with_ or not without:
        return 0.0
    return mean_quality(without) - mean_quality(with_)


def measure_noise(validation: list[Conversation], seed: int) -> float:
    """Highest minus lowest mean quality over seeded equal parts of the validation split: how much
    the score moves when nothing has changed, the dataset analogue of three runs of one policy."""
    order = validation[:]
    random.Random(seed).shuffle(order)
    size = len(order) // NOISE_PARTS
    qualities = [mean_quality(order[i * size:(i + 1) * size]) for i in range(NOISE_PARTS)]
    return round(max(qualities) - min(qualities), 3)


def closed_patterns(log: DecisionLog) -> set[str]:
    """Names the critic no longer raises: fixed by a kept rule, or already tried and reverted."""
    names = {h.brief.name for h in log.pattern_histories() if h.brief.status == "fixed"}
    names |= {r.pattern.name for r in log.reverted_edits() if r.pattern}
    return names


def dataset_critic(round_: int, train: list[Conversation], log: DecisionLog, seed: int,
                   policy_path: Path, root: Path) -> RoundResult:
    result = RoundResult(round=round_, outcome="no_pattern", base_policy_path=policy_path)
    batch = random.Random(seed * 1000 + round_).sample(train, min(BATCH, len(train)))
    lowest = sorted(batch, key=lambda c: (c.quality, c.id))[:FOR_CRITIC]
    closed = closed_patterns(log)
    candidates = []
    for fix in CATALOGUE:
        if fix.name in closed:
            continue
        seen = [c.id for c in lowest if fix.present(c)]
        drop = train_drop(fix, train)
        if len(seen) >= MIN_SEEN and drop >= MIN_TRAIN_DROP:
            candidates.append((len(seen), drop, fix, seen))
    if not candidates:
        result.reason = (f"No open catalogue pattern appeared in {MIN_SEEN} or more of the {len(lowest)} "
                         f"lowest-rated conversations while also lowering training-split quality.")
        return result
    _, drop, fix, seen = max(candidates, key=lambda t: (t[0], t[1]))
    earlier = log.identity_by_name(fix.name)
    result.pattern = FailurePattern(
        id=f"pattern-r{round_:02d}", written_by="critic", created_at=now(), parent_ids=[], round=round_,
        name=fix.name, description=fix.description, seen_in=len(seen), out_of=len(lowest),
        example_conversation_ids=seen[:5], same_as_pattern_id=earlier)
    statuses = [p.status for p in log.earlier_patterns() if p.id == result.pattern.pattern_id and p.status]
    result.status_before = statuses[0] if statuses else "open"
    result.refs["pattern"] = save_record(result.pattern, round_, root)
    return result


def dataset_researcher(result: RoundResult, train: list[Conversation], root: Path) -> RoundResult:
    assert result.pattern is not None
    fix = BY_NAME[result.pattern.name]
    with_ = [c for c in train if fix.present(c)]
    without = [c for c in train if not fix.present(c)]
    evidence = Source(
        url=ESCONV_URL, title=f"ESConv training split, revision {ESCONV_REVISION[:7]}",
        technique=(f"Seeker-rated quality is {mean_quality(with_):.2f} in the {len(with_)} conversations "
                   f"with this pattern and {mean_quality(without):.2f} in the {len(without)} without it."))
    result.research = ResearchNote(
        id=f"research-r{result.round:02d}", written_by="researcher", created_at=now(),
        parent_ids=[result.pattern.id], round=result.round, pattern_id=result.pattern.pattern_id,
        queries=fix.queries, sources=fix.sources + [evidence])
    result.refs["research"] = save_record(result.research, result.round, root)
    return result


def dataset_proposer(result: RoundResult, log: DecisionLog, root: Path) -> RoundResult:
    assert result.pattern is not None and result.research is not None
    fix = BY_NAME[result.pattern.name]
    policy = parse_policy(result.base_policy_path.read_text())
    new_version = candidate_version(policy.version)
    candidate, rule_id = apply_edit(policy, fix.action, fix.reword, fix.rule_text, new_version,
                                    used_ids=log.used_rule_ids())
    rel = f"policies/{new_version}-r{result.round:02d}.md"
    (root / "policies").mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(render_policy(candidate))
    result.edit = PolicyEdit(
        id=f"edit-r{result.round:02d}", written_by="proposer", created_at=now(),
        parent_ids=[result.pattern.id, result.research.id], round=result.round,
        pattern_id=result.pattern.pattern_id, research_id=result.research.id,
        base_policy_version=policy.version, new_policy_version=new_version, policy_path=rel,
        summary=fix.summary, action=fix.action, rule_id=rule_id, rule_text=fix.rule_text)
    result.candidate_path, result.outcome = root / rel, "candidate"
    result.refs["edit"] = save_record(result.edit, result.round, root)
    return result


def dataset_gate(result: RoundResult, validation: list[Conversation], noise: float,
                 token_cap: int = 800) -> tuple[GateOutcome, LogScores, LogSafety]:
    assert result.edit is not None and result.pattern is not None and result.candidate_path is not None
    fix = BY_NAME[result.pattern.name]
    base = parse_policy(result.base_policy_path.read_text())
    candidate_text = result.candidate_path.read_text()
    candidate = parse_policy(candidate_text)
    breaks = [c for c in validation if fix.present(c)]
    follows = [c for c in validation if not fix.present(c)]
    safety_same = safety_section(candidate) == safety_section(base)
    problems = []
    if not safety_same:
        problems.append("the Safety section changed")
    if estimate_tokens(candidate_text) > token_cap:
        problems.append(f"the policy is over {token_cap} tokens")
    if len(breaks) < 5 or len(follows) < 5:
        problems.append(f"too few validation conversations to compare ({len(breaks)} break the rule, "
                        f"{len(follows)} follow it)")
        before = after = mean_scores(validation)
    else:
        before, after = mean_scores(breaks), mean_scores(follows)
        gain = (after.quality or 0) - (before.quality or 0)
        if not gain > noise:
            problems.append(f"quality changed by {gain:+.2f}, not more than the noise of {noise:.2f}")
        for dim in ("empathy", "relevance"):
            fall = getattr(before, dim) - getattr(after, dim)
            if fall > noise:
                problems.append(f"{dim} fell by {fall:.2f}, more than the noise")
    decision: Literal["kept", "reverted"] = "reverted" if problems else "kept"
    kept_version = result.edit.new_policy_version.replace("-candidate", "")
    comparison = (f"On the ESConv validation split, the {len(follows)} conversations that follow the new rule "
                  f"rate {after.quality:.2f} against {before.quality:.2f} for the {len(breaks)} that break it "
                  f"(noise {noise:.2f}).")
    reason = (f"Dataset gate: {comparison} " + ("; ".join(problems).capitalize() + "." if problems else
              "The gain is above the noise and no dimension fell by more than the noise.")
              + " No conversations or safety cases were run; the Safety section was checked unchanged.")
    gate = GateOutcome(
        id=f"gate-r{result.round:02d}", written_by="gatekeeper", created_at=now(), parent_ids=[result.edit.id],
        round=result.round, edit_id=result.edit.id, decision=decision, before=before, after=after,
        safety_passed=safety_same, policy_in_force=kept_version if decision == "kept" else base.version,
        reason=reason)
    safety = LogSafety(case_file="none: dataset gate checks the Safety section only", passed=safety_same, cases=0)
    return gate, LogScores(before=before, after=after, noise=noise), safety


def run(root: Path, start: Path, rounds: int, seed: int, memory: Memory,
        train: list[Conversation], validation: list[Conversation]) -> list[str]:
    log = DecisionLog(root)
    memory.ensure_training_audience()
    noise = measure_noise(validation, seed)
    lines = []
    for _ in range(rounds):
        policy_path = prepare_round(log, memory, root, start)
        round_ = log.next_round()
        result = dataset_critic(round_, train, log, seed, policy_path, root)
        gate = scores = safety = None
        if result.pattern is not None:
            dataset_researcher(result, train, root)
            dataset_proposer(result, log, root)
            gate, scores, safety = dataset_gate(result, validation, noise)
        event = finish_round(result, gate, log, memory, root, start, scores=scores, safety=safety)
        name = event.pattern.name if event.pattern else "-"
        lines.append(f"round {round_}: {event.outcome:<10} {name:<28} policy in force {event.policy.in_force}")
        if closed_patterns(log) >= set(BY_NAME):
            break
    return lines


def load_counselbench_adv() -> dict[str, int]:
    """Prompts per category (6 x 20); the prompts themselves are CC BY-NC-ND and are not copied out."""
    url = f"https://huggingface.co/datasets/{CBADV}/resolve/{CBADV_REVISION}/counselbench_adv.csv"
    rows = list(csv.DictReader(io.StringIO(_fetch(url, f"{CBADV}/{CBADV_REVISION}/counselbench_adv.csv"))))
    if len(rows) != 20 or set(rows[0]) != set(CBADV_RULES):
        raise ValueError("CounselBench-Adv: expected 20 prompts in each of 6 categories")
    return {cat: sum(bool((r[cat] or "").strip()) for r in rows) for cat in CBADV_RULES}


def load_vera_personas() -> dict[str, int]:
    """VERA-MH suicide-risk personas per short risk level (100 in all)."""
    url = f"https://raw.githubusercontent.com/{VERA}/{VERA_REVISION}/data/SI/personas.tsv"
    rows = list(csv.DictReader(io.StringIO(_fetch(url, f"{VERA}/{VERA_REVISION}/personas.tsv")), delimiter="\t"))
    if len(rows) != 100:
        raise ValueError(f"VERA-MH: expected 100 personas, got {len(rows)}")
    return {level: sum(r["Short Current Suicide Risk Level"] == level for r in rows) for level in VERA_RULES}


def held_out(log: DecisionLog, root: Path, start: Path, test: list[Conversation],
             cbadv: dict[str, int], vera: dict[str, int]) -> dict:
    """Run once, after every round: the loop's decisions never see these numbers."""
    policy = replay.policy_in_force(log, root, start)
    rule_ids = parse_policy(policy.read_text()).rule_ids()
    kept = [r for r in log.rounds() if r.outcome == "kept" and r.pattern and r.edit]
    esconv = {}
    for r in kept:
        fix = BY_NAME[r.pattern.name]  # type: ignore[union-attr]
        breaks, follows = [c for c in test if fix.present(c)], [c for c in test if not fix.present(c)]
        esconv[r.edit.rule_id] = {"pattern": fix.name,  # type: ignore[union-attr]
                                  "breaks": [len(breaks), mean_scores(breaks).quality],
                                  "follows": [len(follows), mean_scores(follows).quality]}
    learned = {r.pattern.name: r.edit.rule_id for r in kept}  # type: ignore[union-attr]

    def cover(rules: list[str]) -> list[str]:
        return [i for i in (learned.get(x, x) for x in rules) if i in rule_ids]

    return {
        "policy": policy.stem,
        "esconv_test": {"conversations": len(test), "rules": esconv},
        "counselbench_adv": {cat: [n, cover(CBADV_RULES[cat])] for cat, n in cbadv.items()},
        "vera_mh_si": {level: [n, cover(VERA_RULES[level])] for level, n in vera.items()},
        "note": "esconv_test: [n, quality] for test conversations that break / follow each kept rule. "
                "counselbench_adv, vera_mh_si: [cases, policy rules meant to handle them]; "
                "scoring them needs the live helper (puddles.eval.external).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="runs/esconv")
    ap.add_argument("--policy", default="policies/policy-v1.md", help="the starting policy")
    ap.add_argument("--rounds", type=int, default=8,
                    help="most rounds; stops early once every catalogue pattern is fixed or reverted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--past", action="store_true", help="copy records into past.dev (needs PAST_API_KEY)")
    args = ap.parse_args()
    root = Path(args.root)
    memory: Memory = PastClient() if args.past else FakeMemory()
    for line in run(root, Path(args.policy), args.rounds, args.seed, memory,
                    load_split("train"), load_split("validation")):
        print(line)
    final = replay.policy_in_force(DecisionLog(root), root, Path(args.policy))
    print(f"policy in force: {final}")
    report = held_out(DecisionLog(root), root, Path(args.policy), load_split("test"),
                      load_counselbench_adv(), load_vera_personas())
    (root / "held-out.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(f"held-out check: {root / 'held-out.json'}")
    for row in memory_trace(DecisionLog(root), root):
        print(f"  {row['record']:<14} written by {row['written_by']:<11} read by {', '.join(row['read_by']) or '-'}")


if __name__ == "__main__":
    main()
