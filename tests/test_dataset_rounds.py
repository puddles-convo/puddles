"""Dataset-driven rounds: same records, log, policy files and replay check as a live round."""
import json
from pathlib import Path

from puddles import replay
from puddles.decision_log import DecisionLog
from puddles.memory import FakeMemory
from scripts.dataset_rounds import CATALOGUE, Conversation, measure_noise, parse_esconv, run

START = Path(__file__).parents[1] / "policies" / "policy-v1.md"


def conv(i: int, split: str, empathy: int, relevance: int, strategies: list[str]) -> Conversation:
    return Conversation(f"conv-esconv-{split}-{i:04d}", empathy, relevance, [("hi", s) for s in strategies])


def dataset(split: str, n: int, low_without_reassurance: bool) -> list[Conversation]:
    out = []
    for i in range(n):
        reassures = i % 3 != 0
        steps = ["Question", "Reflection of feelings"] + (["Affirmation and Reassurance"] if reassures else [])
        low = low_without_reassurance and not reassures
        out.append(conv(i, split, 2 if low else 5, 3 if low else 5, steps))
    return out


def test_parse_skips_unrated_rows_and_keeps_helper_strategies():
    rated = {"survey_score": {"seeker": {"empathy": "4", "relevance": "5"}},
             "dialog": [{"speaker": "usr", "text": "a"}, {"speaker": "sys", "text": "b", "strategy": "Question"}]}
    unrated = {"survey_score": {"seeker": {"empathy": "", "relevance": "5"}}, "dialog": []}
    convs = parse_esconv("\n".join(json.dumps(r) for r in (rated, unrated)), "train")
    assert [(c.id, c.quality, c.helper) for c in convs] == [("conv-esconv-train-0000", 4.5, [("b", "Question")])]


def test_kept_edit_is_replay_valid_and_reverted_one_leaves_policy(tmp_path):
    train = dataset("train", 120, low_without_reassurance=True)
    lines = run(tmp_path, START, 3, 0, FakeMemory(), train, dataset("validation", 60, True))
    log = DecisionLog(tmp_path)
    rounds = log.rounds()
    assert rounds[0].outcome == "kept" and rounds[0].pattern.name == "no-reassurance"
    assert rounds[0].scores is not None and rounds[0].safety.cases == 0
    assert replay.policy_in_force(log, tmp_path, START).name == "policy-v2.md"
    replay.check(log, tmp_path, START)
    assert "C9." in (tmp_path / "policies" / "policy-v2.md").read_text()
    assert all(r.pattern is None or r.pattern.name != "no-reassurance" for r in rounds[1:])
    assert len(lines) == len(rounds)

    flat = tmp_path / "flat"
    run(flat, START, 1, 0, FakeMemory(), train, dataset("validation", 60, False))
    assert DecisionLog(flat).rounds()[0].outcome == "reverted"
    assert replay.policy_in_force(DecisionLog(flat), flat, START) == START


def test_noise_is_zero_when_every_rating_is_equal():
    assert measure_noise(dataset("validation", 30, False), 0) == 0.0
    assert {f.name for f in CATALOGUE} >= {"no-reassurance", "information-heavy"}
