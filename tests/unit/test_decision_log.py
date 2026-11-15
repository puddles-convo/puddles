"""Unit tests for loop/decision_log.py."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from loop.decision_log import (
    DecisionLogError,
    append_event,
    copy_status,
    pending_copies,
    read_events,
    render_table,
)
from schemas import MemoryCopyEvent, RoundEvent

REPO = Path(__file__).resolve().parents[2]
HEADER = "| Round | Outcome | Pattern | Edit | Quality | Safety | Policy in force | Reason |"


def example_round(**changes) -> RoundEvent:
    data = json.loads((REPO / "examples/log_round_event.json").read_text())
    data.update(changes)
    return RoundEvent.model_validate(data)


def copy(record_id: str, status: str, attempt: int = 1, round: int = 3) -> MemoryCopyEvent:
    return MemoryCopyEvent(round=round, record_id=record_id, status=status, attempt=attempt)


def early_round(round: int = 4) -> RoundEvent:
    return RoundEvent(
        round=round,
        outcome="no_pattern",
        policy={"base": "policy-v4", "base_sha256": "h", "in_force": "policy-v4"},
        reason="The critic found no clear pattern.",
    )


# ---------- append_event ----------

def test_append_creates_file_and_parents(tmp_path):
    path = tmp_path / "a" / "b" / "log.jsonl"
    append_event(path, example_round())
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "round"


def test_append_adds_one_line_and_keeps_earlier_bytes(tmp_path):
    path = tmp_path / "log.jsonl"
    append_event(path, example_round())
    before = path.read_bytes()
    append_event(path, copy("gate-r03", "confirmed"))
    after = path.read_bytes()
    assert after.startswith(before)
    assert len(after.decode().splitlines()) == 2


def test_append_duplicate_round_raises_and_writes_nothing(tmp_path):
    path = tmp_path / "log.jsonl"
    append_event(path, example_round())
    append_event(path, copy("gate-r03", "pending"))
    before = path.read_bytes()
    with pytest.raises(DecisionLogError):
        append_event(path, example_round(reason="again"))
    assert path.read_bytes() == before


def test_append_allows_memory_copy_for_logged_round(tmp_path):
    path = tmp_path / "log.jsonl"
    append_event(path, example_round())
    append_event(path, copy("gate-r03", "pending"))
    append_event(path, copy("gate-r03", "confirmed", attempt=2))
    assert len(read_events(path)) == 3


def test_append_after_file_missing_final_newline(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(copy("x", "pending").model_dump_json())
    append_event(path, copy("y", "pending"))
    assert [e.record_id for e in read_events(path)] == ["x", "y"]


# ---------- read_events ----------

def test_read_missing_file_is_empty(tmp_path):
    assert read_events(tmp_path / "nope.jsonl") == []


def test_read_round_trips_examples_in_order(tmp_path):
    path = tmp_path / "log.jsonl"
    events = [example_round(), copy("gate-r03", "confirmed"), early_round()]
    for event in events:
        append_event(path, event)
    got = read_events(path)
    assert got == events
    assert [type(e) for e in got] == [RoundEvent, MemoryCopyEvent, RoundEvent]


def test_read_example_files_directly(tmp_path):
    path = tmp_path / "log.jsonl"
    lines = [json.dumps(json.loads((REPO / f"examples/{name}").read_text()))
             for name in ("log_round_event.json", "log_memory_copy_event.json")]
    path.write_text("\n".join(lines) + "\n")
    got = read_events(path)
    assert isinstance(got[0], RoundEvent) and isinstance(got[1], MemoryCopyEvent)


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text("\n" + copy("a", "pending").model_dump_json() + "\n   \n\n"
                    + copy("b", "confirmed").model_dump_json() + "\n")
    assert [e.record_id for e in read_events(path)] == ["a", "b"]


def test_read_invalid_json_names_line(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(copy("a", "pending").model_dump_json() + "\n\n{not json\n")
    with pytest.raises(DecisionLogError, match="line 3"):
        read_events(path)


def test_read_invalid_event_names_line(tmp_path):
    path = tmp_path / "log.jsonl"
    bad = {"event": "memory_copy", "round": 1, "record_id": "a", "status": "lost", "attempt": 1}
    path.write_text(copy("a", "pending").model_dump_json() + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(DecisionLogError, match="line 2"):
        read_events(path)


@pytest.mark.parametrize("line", ['{"event": "other", "round": 1}', '{"round": 1}', "[1, 2]", "3"])
def test_read_unknown_or_missing_event_kind_raises(tmp_path, line):
    path = tmp_path / "log.jsonl"
    path.write_text(line + "\n")
    with pytest.raises(DecisionLogError, match="line 1"):
        read_events(path)


def test_append_to_malformed_log_raises_for_round(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text("{oops\n")
    with pytest.raises(DecisionLogError):
        append_event(path, early_round())
    assert path.read_text() == "{oops\n"


# ---------- copy_status and pending_copies ----------

def test_copy_status_replays_in_order():
    events = [example_round(), copy("a", "pending"), copy("b", "pending"),
              copy("a", "confirmed", attempt=2)]
    assert copy_status(events) == {"a": "confirmed", "b": "pending"}


def test_confirmed_stays_confirmed_after_pending():
    events = [copy("a", "confirmed"), copy("a", "pending", attempt=2)]
    assert copy_status(events) == {"a": "confirmed"}
    assert pending_copies(events) == []


def test_pending_copies_in_first_appearance_order():
    events = [copy("c", "pending"), copy("a", "pending"), copy("b", "confirmed"),
              copy("c", "pending", attempt=2), copy("d", "pending")]
    assert pending_copies(events) == ["c", "a", "d"]


def test_copy_status_empty():
    assert copy_status([]) == {}
    assert pending_copies([example_round()]) == []


# ---------- render_table ----------

def test_render_table_no_rounds_is_header_only():
    lines = render_table([copy("a", "pending")]).splitlines()
    assert lines[0] == HEADER
    assert len(lines) == 2
    assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}


def test_render_table_full_round():
    row = render_table([example_round()]).splitlines()[2]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert len(cells) == 8
    assert cells[0] == "3"
    assert cells[1] == "kept"
    assert "advice-before-understanding" in cells[2]
    assert "add" in cells[3] and "C2" in cells[3]
    assert cells[4] == "3.25 -> 3.75"
    assert cells[5].startswith("passed")
    assert cells[6] == "policy-v4"
    assert cells[7].startswith("Quality rose by 0.50")


def test_render_table_early_ending_round_uses_dashes():
    row = render_table([early_round()]).splitlines()[2]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells == ["4", "no_pattern", "-", "-", "-", "-", "policy-v4",
                     "The critic found no clear pattern."]


def test_render_table_rows_in_round_order():
    table = render_table([early_round(5), copy("a", "pending"), example_round(), early_round(4)])
    rounds = [line.strip("|").split("|")[0].strip() for line in table.splitlines()[2:]]
    assert rounds == ["3", "4", "5"]


def test_render_table_failed_safety_and_missing_quality():
    scores = {"before": {"empathy": 3.0, "relevance": 3.5},
              "after": {"empathy": 3.0, "relevance": 3.0}, "noise": 0.1}
    safety = {"case_file": "safety-v1.json", "passed": False, "cases": 18,
              "failed_case_ids": ["s1-03"]}
    event = example_round(round=5, outcome="reverted", scores=scores, safety=safety,
                          reason="Safety case s1-03 failed | reverted.")
    row = render_table([event]).splitlines()[2]
    assert "3.25 -> 3.00" in row
    assert "failed (s1-03)" in row
    assert "\\|" in row


# ---------- command line ----------

def test_cli_prints_table(tmp_path):
    path = tmp_path / "log.jsonl"
    append_event(path, example_round())
    result = subprocess.run([sys.executable, "-m", "loop.decision_log", str(path)],
                            cwd=REPO, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == render_table(read_events(path))
