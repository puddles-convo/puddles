"""Build demo/index.html: one self-contained page that replays the committed records. No model, no network.

    python -m demo.build            # rewrite demo/index.html
    python -m demo.build --check    # exit 1 if demo/index.html is out of date

Every number and every line of text on the page is read from a file under results/ or policies/; the page
adds labels and layout only. The chat tab replays recorded real-model sessions; nothing in it is live.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.util
import io
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN = RESULTS / "training-2026-11-15" / "run"
POLICIES = ROOT / "policies" / "training-2026-11-15"  # the run's kept policies, v1 to v4
PAGE = ROOT / "demo" / "index.html"
TEMPLATE = ROOT / "demo" / "template.html"
MARK = "/*DATA*/"
RULE = re.compile(r"^([A-Z]\d+)\. (.*)$")
WEAKEST_SHOWN = 3  # conversations shown in full for each round, lowest quality first

# What the committed reports say about each recorded session; the records themselves do not hold these.
SMOKE = {"model": "gpt-6-luna", "date": "2026-11-15", "policy_file": "policies/policy-v2.md",
         "source": "results/inference-smoke/transcript-consented.jsonl"}
PILOT_MODEL = "gpt-6-luna"  # results/persona-pilot/README.md, "Models"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _offline_line() -> str:
    return _json(ROOT / "safety" / "crisis-path-v1.json").get("offline_line", "")


def route(*, fixed: bool, keyword: bool, status: str, risk: str | None, text: str) -> str:
    """Which rule of the release step spoke: the helper's draft, the fixed crisis response, or a waiting line."""
    if not fixed:
        return "draft"
    if keyword or (status == "answered" and risk == "crisis"):
        return "crisis"
    return "offline" if text == _offline_line() else "holding"


def _turn(line: dict) -> dict:
    checked = line.get("checked") or {}
    reply = line["reply"]
    status = line["check_status"]
    return {
        "user": line["user_text"],
        "reply": reply["text"],
        "route": route(fixed=reply["fixed_handoff_response"], keyword=line["keyword_matched"], status=status,
                       risk=checked.get("risk"), text=reply["text"]),
        "keyword_matched": line["keyword_matched"],
        "check_status": status,
        "risk": checked.get("risk"),
        "category": checked.get("category"),
        "reason": checked.get("reason"),
        "ms": line["timings_ms"],
    }


def _note(note: dict) -> dict:
    return {"id": note["note_id"], "kind": note.get("kind"), "fact": note["fact"]}


def smoke() -> dict:
    lines = _lines(ROOT / SMOKE["source"])
    return {"id": "work", "kind": "session", "source": SMOKE["source"], "model": SMOKE["model"],
            "policy": lines[0]["reply"]["policy_version"], "policy_file": SMOKE["policy_file"], "date": SMOKE["date"],
            "sessions": [{"label": "Session 1", "notes_given": None, "turns": [_turn(x) for x in lines],
                          "changes": None}]}


def personas() -> list[dict]:
    pilot = _json(RESULTS / "persona-pilot" / "pilot.json")
    out = []
    for persona in pilot["personas"]:
        pid = persona["persona_id"]
        sessions = []
        for n in (1, 2, 3):
            s = _json(RESULTS / "persona-pilot" / "sessions" / pid / "notes_on" / f"s{n}.json")
            lines = _lines(RESULTS / "persona-pilot" / "store" / "sessions" / f"{s['session_id']}.jsonl")
            changes = s["changes"]
            sessions.append({
                "label": f"Session {n}", "date": s["record"]["created_at"][:10],
                "notes_given": [_note(x) for x in s["notes_before"]],
                "turns": [_turn(x) for x in lines],
                "changes": {k: [_note(x) for x in changes[k]] for k in ("added", "replaced", "confirmed", "deleted")},
                "notes_after": [_note(x) for x in s["notes_after"]],
            })
        out.append({"id": pid, "kind": "session", "description": persona["description"],
                    "source": f"results/persona-pilot/sessions/{pid}/notes_on", "model": PILOT_MODEL,
                    "policy": pilot["policy_version"], "policy_file": pilot["policy_path"], "date": pilot["start"],
                    "sessions": sessions})
    return out


def crisis_pilot() -> dict:
    run = _json(RESULTS / "crisis-pilot" / "run.json")
    labelled = {x["id"]: x for x in _lines(ROOT / run["set"])}
    messages: dict[str, dict] = {}
    for r in _lines(RESULTS / "crisis-pilot" / "results.jsonl"):
        label = labelled[r["id"]]
        context = label.get("context") or []
        if label["text"] != r["text"] or len(context) != r["context_turns"]:
            raise ValueError(f"{run['set']} no longer matches the recorded run at {r['id']}")
        m = messages.setdefault(r["id"], {"id": r["id"], "kind": r["kind"], "text": r["text"],
                                          "expect": r["expect"], "expect_category": r["expect_category"],
                                          "context": context, "why": label["why"], "runs": []})
        m["runs"].append({
            "run": r["run"], "reply": r["spoken_text"], "spoken": r["spoken"],
            "route": route(fixed=r["fixed_handoff_response"], keyword=r["keyword_matched"],
                           status=r["check_status"], risk=r["risk"], text=r["spoken_text"]),
            "keyword_matched": r["keyword_matched"], "keyword": r["keyword"], "check_status": r["check_status"],
            "risk": r["risk"], "category": r["category"], "draft": r["helper_draft"], "ms": r["timings_ms"],
        })
    for m in messages.values():
        m["runs"].sort(key=lambda x: x["run"])
    return {"id": "crisis", "kind": "pilot", "source": "results/crisis-pilot/results.jsonl", "model": run["model"],
            "policy": Path(run["policy"]).stem, "policy_file": run["policy"], "messages": list(messages.values())}


def rules(text: str) -> list[dict]:
    """A policy file's sections and their rules, in file order."""
    out: list[dict] = []
    for line in text.splitlines():
        if line.startswith("## "):
            out.append({"section": line[3:].split(" (")[0], "rules": []})
        m = RULE.match(line)
        if m and out:
            out[-1]["rules"].append({"id": m.group(1), "text": m.group(2)})
    return out


def _rule_text(policy: str, rule_id: str | None) -> str | None:
    for group in rules((POLICIES / f"{policy}.md").read_text(encoding="utf-8")):
        for rule in group["rules"]:
            if rule["id"] == rule_id:
                return rule["text"]
    return None


def _round(event: dict) -> dict:
    n = event["round"]
    folder = RUN / "records" / f"round-{n:02d}"
    pattern = _json(folder / f"pattern-r{n:02d}.json")
    research = _json(folder / f"research-r{n:02d}.json")
    scores = {s["conversation_id"]: s for s in _lines(RUN / "scores" / f"round-{n:02d}-train.jsonl")}
    conversations = {c["id"]: c for c in _lines(RUN / "conversations" / f"round-{n:02d}-train.jsonl")}
    quality = {cid: (s["empathy"] + s["relevance"]) / 2 for cid, s in scores.items()}
    examples = [cid for cid in pattern["example_conversation_ids"] if cid in conversations and cid in scores]
    shown = sorted(examples, key=lambda cid: (quality[cid], cid))[:WEAKEST_SHOWN]
    out: dict[str, Any] = {
        "round": n, "outcome": event["outcome"], "reason": event["reason"],
        "policy_base": event["policy"]["base"], "policy_in_force": event["policy"]["in_force"],
        "train": [{"scenario": conversations[cid]["scenario_id"], "empathy": s["empathy"],
                   "relevance": s["relevance"]} for cid, s in sorted(scores.items())],
        "weakest": [{"scenario": conversations[cid]["scenario_id"], "empathy": scores[cid]["empathy"],
                     "relevance": scores[cid]["relevance"], "note": scores[cid]["note"],
                     "turns": [[t["speaker"], t["text"]] for t in conversations[cid]["turns"]]} for cid in shown],
        "pattern": {"name": pattern["name"], "description": pattern["description"], "seen_in": pattern["seen_in"],
                    "out_of": pattern["out_of"], "same_as": pattern["same_as_pattern_id"]},
        "research": {"queries": research["queries"],
                     "sources": [{k: s.get(k) for k in ("url", "title", "quote", "technique")}
                                 for s in research["sources"]]},
        "edit": None, "gate": None,
    }
    if event["edit"]:
        edit = _json(folder / f"edit-r{n:02d}.json")
        gate = _json(RUN / "scores" / f"round-{n:02d}-gate.json")
        safety = _json(RUN / "scores" / f"round-{n:02d}-safety.json")
        out["edit"] = {"action": edit["action"], "rule_id": edit["rule_id"], "rule_text": edit["rule_text"],
                       "summary": edit["summary"],
                       "before_text": _rule_text(edit["base_policy_version"], edit["rule_id"])}
        out["gate"] = {"decision": gate["decision"], "checks": [[k, v] for k, v in gate["checks"].items()],
                       "reason": gate["reason"],
                       "before": event["scores"]["before"], "after": event["scores"]["after"],
                       "noise": event["scores"]["noise"],
                       "safety": {"passed": safety["passed"], "cases": len(safety["cases"]),
                                  "failed": [c["case_id"] for c in safety["cases"] if not c["passed"]]}}
    return out


def _paired_stats() -> Any:
    """The run's own statistics script, loaded for its functions; what it prints on import is dropped."""
    path = RESULTS / "training-2026-11-15" / "measurements" / "tools" / "paired_stats.py"
    spec = importlib.util.spec_from_file_location("paired_stats", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def measurements() -> list[dict]:
    ps = _paired_stats()
    sets = ps.A / "sets"
    groups = {
        ("validation", "loop"): {g: [ps.load(sets / f"{x}.sol.scores.jsonl") for x in names]
                                 for g, names in ps.LABELS.items()},
        ("validation", "final"): {g: [ps.load(ps.A / f"{x}.scores.jsonl") for x in names]
                                  for g, names in ps.LABELS.items()},
        ("held-out", "loop"): {g: [ps.load(glob.glob(str(ps.M / "heldout" / folder / "scores.jsonl"))[0])]
                               for g, folder in ps.HELD.items()},
        ("held-out", "final"): {g: [ps.load(ps.A / f"{x}.scores.jsonl")] for g, x in ps.HELD_LABELS.items()},
    }
    out = []
    for (scenarios, judge), group in groups.items():
        per = {name: ps.per_scenario(s) for name, s in group.items()}
        block: dict[str, Any] = {"scenarios": scenarios, "judge": judge, "sets": len(group["final"]),
                                 "n": len(per["final"]), "policies": [], "pairs": []}
        for name, q in per.items():
            block["policies"].append({"name": name, **{k: round(st.mean(v[k] for v in q.values()), 3)
                                                       for k in ("quality", "empathy", "relevance")}})
        for a, b in ps.PAIRS:
            ids = sorted(set(per[a]).intersection(per[b]))
            d = [per[a][i]["quality"] - per[b][i]["quality"] for i in ids]
            lo, hi = ps.interval(d)
            better, equal = sum(x > 0 for x in d), sum(x == 0 for x in d)
            block["pairs"].append({"a": a, "b": b, "mean": round(st.mean(d), 3), "lo": round(lo, 3),
                                   "hi": round(hi, 3), "better": better, "equal": equal,
                                   "worse": len(d) - better - equal, "n": len(d)})
        out.append(block)
    return out


def _conversation(conversation: dict, loop: dict, final: dict) -> dict:
    """One measured conversation with both judges' scores; the strategy labels and the note are the final judge's."""
    replies = [t["text"] for t in conversation["turns"] if t["speaker"] == "helper"]
    if len(final["strategies"]) != len(replies):
        raise ValueError(f"{conversation['id']}: {len(final['strategies'])} labels for {len(replies)} replies")
    return {"turns": [[t["speaker"], t["text"]] for t in conversation["turns"]], "strategies": final["strategies"],
            "note": final["note"],
            "scores": {"loop": {"empathy": loop["empathy"], "relevance": loop["relevance"]},
                       "final": {"empathy": final["empathy"], "relevance": final["relevance"]}}}


def compare() -> dict:
    """The starting and the final policy on the same scenarios: the post-run measurement sets, conversation by
    conversation. The person is a model and is not scripted, so only the scenario is shared between the two."""
    ps = _paired_stats()
    groups = {"validation": {name: ps.LABELS[name] for name in ("start", "final")},
              "held-out": {name: [ps.HELD_LABELS[name]] for name in ("start", "final")}}
    scenarios: dict[str, dict] = {}
    policies: dict[str, str] = {}
    for group, labels in groups.items():
        for name, sets in labels.items():
            for label in sets:
                loop = {x["conversation_id"]: x for x in _lines(ps.A / "sets" / f"{label}.sol.scores.jsonl")}
                final = {x["conversation_id"]: x for x in _lines(ps.A / f"{label}.scores.jsonl")}
                for c in _lines(ps.A / "sets" / f"{label}.conversations.jsonl"):
                    if policies.setdefault(name, c["policy_version"]) != c["policy_version"]:
                        raise ValueError(f"{label}: {c['id']} is not from {policies[name]}")
                    s = scenarios.setdefault(c["scenario_id"], {"id": c["scenario_id"], "group": group,
                                                                "start": [], "final": []})
                    s[name].append(_conversation(c, loop[c["id"]], final[c["id"]]))
    out = sorted(scenarios.values(), key=lambda s: s["id"])
    for s in out:
        if len(s["start"]) != len(s["final"]):
            raise ValueError(f"{s['id']}: the two policies were not measured on the same number of sets")
        s["quality"] = {name: round(st.mean((c["scores"]["final"]["empathy"] + c["scores"]["final"]["relevance"]) / 2
                                            for c in s[name]), 3) for name in ("start", "final")}
    stats = []
    for name in ("start", "final"):
        conversations = [c for s in out for c in s[name]]
        replies = [text for c in conversations for speaker, text in c["turns"] if speaker == "helper"]
        counts: dict[str, int] = {}
        for c in conversations:
            for strategy in c["strategies"]:
                counts[strategy] = counts.get(strategy, 0) + 1
        stats.append({"name": name, "policy": policies[name], "conversations": len(conversations),
                      "replies": len(replies), "with_question": sum("?" in r for r in replies),
                      "words": round(st.mean(len(r.split()) for r in replies), 1),
                      "strategies": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))})
    return {"source": _rel(ps.A), "stats": stats, "scenarios": out}


def training() -> dict:
    events = [e for e in _lines(RUN / "log" / "decisions.jsonl") if e["event"] == "round"]
    usage = _lines(RUN / "log" / "usage.jsonl")
    stopped = _lines(RUN / "log" / "loop.jsonl")[-1]
    calls: dict[str, int] = {}
    for u in usage:
        if u["event"] == "sent":
            calls[u["agent"]] = calls.get(u["agent"], 0) + 1
    outcomes = [e["outcome"] for e in events]
    policies = POLICIES
    start, final = events[0]["policy"]["base"], events[-1]["policy"]["in_force"]
    meta = {v: _json(policies / f"{v}.meta.json") for v in (start, final)}
    return {
        "source": _rel(RUN),
        "stats": {"rounds": len(events), "kept": outcomes.count("kept"), "reverted": outcomes.count("reverted"),
                  "no_candidate": len(events) - outcomes.count("kept") - outcomes.count("reverted"),
                  "calls": sum(calls.values()), "calls_by_agent": dict(sorted(calls.items())),
                  "spend_usd": stopped["spend_usd"], "started": usage[0]["at"], "stopped": stopped["at"],
                  "stop_reason": stopped["reason"]},
        "rounds": [_round(e) for e in events],
        "policy": {"start": start, "final": final,
                   "start_rules": rules((policies / f"{start}.md").read_text(encoding="utf-8")),
                   "final_rules": rules((policies / f"{final}.md").read_text(encoding="utf-8")),
                   "safety_unchanged": meta[start]["safety_sha256"] == meta[final]["safety_sha256"],
                   "learned": meta[final]["learned_rules"]},
        "measurements": measurements(),
    }


def data() -> dict:
    return {"chat": [smoke(), *personas(), crisis_pilot()], "compare": compare(), "training": training()}


def render() -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(MARK) != 1:
        raise ValueError(f"{_rel(TEMPLATE)} must hold {MARK} exactly once")
    payload = json.dumps(data(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    payload = payload.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace(MARK, payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="exit 1 if demo/index.html is out of date")
    args = p.parse_args(argv)
    page = render()
    if args.check:
        if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != page:
            print(f"{_rel(PAGE)} is out of date: run python -m demo.build")
            return 1
        return 0
    PAGE.write_text(page, encoding="utf-8")
    print(f"wrote {_rel(PAGE)} ({len(page.encode('utf-8')) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
