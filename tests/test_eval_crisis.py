import re
from pathlib import Path

import pytest

from puddles.eval.crisis import (
    keyword_match,
    load_crisis_path,
    load_keywords,
    normalise,
    parse_checker_line,
)

REPO = Path(__file__).parents[1]
AGENTS_MD = (REPO / "docs" / "agents.md").read_text()
SECTION5 = AGENTS_MD.split("## 5. The crisis path")[1].split("## 6.")[0]


@pytest.mark.parametrize("raw,expected", [
    ("Don’t-want", "dont want"),
    ("  Kill   Myself! ", "kill myself!"),
    ("I'm NOT fine", "im not fine"),
])
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_keyword_match():
    keywords = load_keywords()
    assert keyword_match("I want to END it all", keywords) == "end it all"
    assert keyword_match("sometimes I self-harm", keywords) == "self harm"
    assert keyword_match("Sometimes I wonder if things would be better if I wasn't around any longer.",
                         keywords) is None


def test_load_keywords_skips_comments_and_covers_starting_set():
    keywords = load_keywords()
    assert len(keywords) == len(set(keywords))
    starting = ["kill myself", "end my life", "suicide", "want to die", "hurt myself",
                "harm myself", "not be around", "better off without me", "end it all"]
    for phrase in starting:
        assert phrase in keywords


def test_load_keywords_rejects_duplicates(tmp_path):
    path = tmp_path / "kw.txt"
    path.write_text("want to die\nwant-to-die\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_keywords(path)


def test_parse_checker_line_accepts_doc_example_and_levels():
    level, category, reason = parse_checker_line(
        "crisis | self | says things would be better if they weren't around")
    assert (level, category, reason) == ("crisis", "self", "says things would be better if they weren't around")
    for cat in ("self", "danger", "unsure"):
        assert parse_checker_line(f"crisis | {cat} | r")[:2] == ("crisis", cat)
    assert parse_checker_line("none | - | fine") == ("none", None, "fine")
    assert parse_checker_line("concern | - | heavy") == ("concern", None, "heavy")


@pytest.mark.parametrize("line", [
    "crisis | self",
    "a | b | c | d",
    "alarmed | - | x",
    "crisis | - | x",
    "none | self | x",
])
def test_parse_checker_line_rejects(line):
    with pytest.raises(ValueError):
        parse_checker_line(line)


def _doc_fixed_responses():
    return {k: v.strip() for k, v in
            re.findall(r"^\| `(self|danger|unsure)` \| [^|]+ \| (.+?) \|$", SECTION5, re.M)}


def test_crisis_path_file_matches_docs():
    data = load_crisis_path()
    doc = (REPO / "docs" / "agents" / "i2-crisis-checker.md").read_text()
    system = doc.split("### System message")[1].split("```", 2)[1].strip("\n")
    assert data["checker"]["system"] == system
    assert data["fixed_responses"] == _doc_fixed_responses()
    assert data["holding_line"] == "Sorry, I lost my thread for a moment. Could you say that again?"
    offline = re.search(r"^> (.+)$", SECTION5, re.M).group(1).strip()
    assert data["offline_line"] == offline
    fixed = data["fixed_responses"]
    assert "988" in fixed["self"] and "988" in fixed["unsure"]
    assert "911" in fixed["danger"] and "911" in fixed["unsure"]
    assert "911" in data["offline_line"] and "988" in data["offline_line"]
    for text in fixed.values():
        assert "robot" in text
    assert (REPO / data["keywords_file"]).is_file()
