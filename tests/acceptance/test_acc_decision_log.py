"""Append-only log, model validation, replay, and human-readable round table."""

import importlib
import json
import os
import re
import subprocess
import sys

import pytest


@pytest.fixture
def api():
    return importlib.import_module("loop.decision_log")


@pytest.fixture
def round_event():
    from schemas import LogPolicy, RoundEvent

    def make(number=1, **changes):
        values = dict(round=number, outcome="no_pattern", reason="No clear pattern.",
                      policy=LogPolicy(base="policy-v1", base_sha256="hash", in_force="policy-v1"))
        values.update(changes)
        return RoundEvent(**values)

    return make


@pytest.fixture
def copy_event():
    from schemas import MemoryCopyEvent

    def make(record_id="record-a", status="pending", attempt=1, round=1):
        return MemoryCopyEvent(round=round, record_id=record_id, status=status, attempt=attempt)

    return make


def test_append_creates_parents_one_line_and_preserves_existing_bytes(api, tmp_path, round_event, copy_event):
    from schemas import MemoryCopyEvent, RoundEvent
    path = tmp_path / "nested" / "decisions.jsonl"
    first = round_event()
    api.append_event(path, first)
    first_bytes = path.read_bytes()
    assert len(first_bytes.splitlines()) == 1
    assert first_bytes.endswith(b"\n")
    assert RoundEvent.model_validate_json(first_bytes) == first
    second = copy_event()
    api.append_event(path, second)
    second_bytes = path.read_bytes()
    assert second_bytes.startswith(first_bytes)
    added = second_bytes[len(first_bytes):]
    assert len(added.splitlines()) == 1
    assert added.endswith(b"\n")
    assert MemoryCopyEvent.model_validate_json(added) == second


def test_append_does_not_reserialize_existing_lines(api, tmp_path, round_event, copy_event):
    path = tmp_path / "log.jsonl"
    # Existing spacing and blank lines must survive an append byte for byte.
    prefix = (json.dumps(round_event().model_dump(mode="json"), separators=(", ", ":  ")) + "\n\n").encode()
    path.write_bytes(prefix)
    api.append_event(path, copy_event())
    assert path.read_bytes().startswith(prefix)
    assert api.read_events(path) == [round_event(), copy_event()]


def test_duplicate_round_rejected_without_any_write(api, tmp_path, round_event, copy_event):
    path = tmp_path / "log.jsonl"
    for event in (round_event(3), round_event(1), copy_event(round=3)):
        api.append_event(path, event)
    original = path.read_bytes()
    with pytest.raises(api.DecisionLogError):
        api.append_event(path, round_event(3, reason="Different content is still a duplicate."))
    assert path.read_bytes() == original
    # Memory-copy events for an existing round are allowed, including repeated copies.
    api.append_event(path, copy_event(round=3))
    assert len(api.read_events(path)) == 4


def test_read_models_in_file_order_skips_blanks(api, tmp_path, round_event, copy_event):
    from schemas import MemoryCopyEvent, RoundEvent
    events = [copy_event("a"), round_event(9), copy_event("a", "confirmed"), round_event(2)]
    path = tmp_path / "log.jsonl"
    path.write_text("\n \n" + "\n\n".join(event.model_dump_json() for event in events) + "\n", encoding="utf-8")
    loaded = api.read_events(path)
    assert loaded == events
    assert [type(event) for event in loaded] == [MemoryCopyEvent, RoundEvent, MemoryCopyEvent, RoundEvent]


def test_missing_log_is_empty_without_creating_file(api, tmp_path):
    path = tmp_path / "absent" / "log.jsonl"
    assert api.read_events(path) == []
    assert not path.exists()


@pytest.mark.parametrize("bad_line", [
    "{", "null", "[]", '{"event":"unknown"}', '{"event":"round","round":1}',
    '{"event":"memory_copy","round":1,"record_id":"a","status":"lost","attempt":1}',
    '{"event":"memory_copy","round":1,"record_id":"a","status":"pending","attempt":0}',
    '{"event":"memory_copy","round":1,"record_id":"a","status":"pending","attempt":1,"extra":true}',
])
def test_bad_log_line_reports_physical_one_based_line_number(api, tmp_path, round_event, bad_line):
    path = tmp_path / "log.jsonl"
    path.write_text(round_event().model_dump_json() + "\n\n  \n" + bad_line + "\n", encoding="utf-8")
    with pytest.raises(api.DecisionLogError) as caught:
        api.read_events(path)
    assert re.search(r"line\s*4\b", str(caught.value), re.IGNORECASE), str(caught.value)


def test_memory_copy_does_not_reserve_round_number(api, tmp_path, round_event, copy_event):
    path = tmp_path / "log.jsonl"
    events = [copy_event(round=5), round_event(5)]
    api.append_event(path, events[0])
    prefix = path.read_bytes()
    api.append_event(path, events[1])
    assert path.read_bytes().startswith(prefix)
    assert api.read_events(path) == events


def test_copy_replay_confirmed_is_absorbing_despite_later_pending_or_attempts(api, round_event, copy_event):
    events = [
        copy_event("a", "pending", 2), round_event(),
        copy_event("b", "confirmed", 3), copy_event("a", "confirmed", 3),
        copy_event("a", "pending", 4, round=2), copy_event("b", "pending", 1),
        copy_event("c", "pending", 1), copy_event("c", "pending", 2),
    ]
    assert api.copy_status(events) == {"a": "confirmed", "b": "confirmed", "c": "pending"}
    assert api.pending_copies(events) == ["c"]


def test_pending_copies_are_unique_in_first_appearance_order(api, copy_event):
    events = [copy_event("z"), copy_event("a"), copy_event("z", attempt=2),
              copy_event("m", "confirmed"), copy_event("b"), copy_event("m", "pending", 2)]
    assert api.pending_copies(events) == ["z", "a", "b"]


def test_no_copy_events_gives_empty_replay(api, round_event):
    assert api.copy_status([round_event()]) == {}
    assert api.pending_copies([]) == []


def table_cells(text):
    return [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in text.strip().splitlines()]


def assert_header_and_separator(rows):
    assert rows[0] == ["Round", "Outcome", "Pattern", "Edit", "Quality", "Safety", "Policy in force", "Reason"]
    assert len(rows[1]) == 8
    assert all(re.fullmatch(r":?-+:?", cell) for cell in rows[1])


@pytest.mark.parametrize("only_memory", [False, True])
def test_table_without_rounds_has_only_header_and_separator(api, copy_event, only_memory):
    rows = table_cells(api.render_table([copy_event()] if only_memory else []))
    assert len(rows) == 2
    assert_header_and_separator(rows)


def test_table_rows_sort_rounds_and_show_missing_parts_as_dash(api, round_event, copy_event):
    rows = table_cells(api.render_table([round_event(12), copy_event(round=12), round_event(2)]))
    assert_header_and_separator(rows)
    assert rows[2:] == [
        ["2", "no_pattern", "-", "-", "-", "-", "policy-v1", "No clear pattern."],
        ["12", "no_pattern", "-", "-", "-", "-", "policy-v1", "No clear pattern."],
    ]


def test_table_quality_and_populated_columns_from_documented_example(api, repo_root):
    from schemas import RoundEvent
    event = RoundEvent.model_validate_json((repo_root / "examples" / "log_round_event.json").read_text())
    rows = table_cells(api.render_table([event]))
    assert len(rows) == 3
    row = rows[2]
    assert row[0:2] == ["3", "kept"]
    # The exact representation of Pattern/Edit/Safety is not specified.
    assert all(cell and cell != "-" for cell in (row[2], row[3], row[5]))
    assert row[4] == "3.25 -> 3.75"
    assert row[6:] == ["policy-v4", event.reason]


@pytest.mark.parametrize("before, after, expected", [
    (None, None, "3.25 -> 3.75"),
    (None, 4.0, "3.25 -> 4.00"),
    (4.0, None, "4.00 -> 3.75"),
])
def test_table_quality_when_scores_have_no_explicit_quality(api, repo_root, before, after, expected):
    from schemas import RoundEvent
    data = json.loads((repo_root / "examples/log_round_event.json").read_text())
    data["scores"]["before"]["quality"] = before
    data["scores"]["after"]["quality"] = after
    # schemas.Scores and docs/agents.md define quality as the mean of empathy
    # and relevance. The scores part is present, so that mean is available.
    event = RoundEvent.model_validate(data)
    assert table_cells(api.render_table([event]))[2][4] == expected


def test_append_after_unterminated_line_preserves_bytes_and_separates_events(api, tmp_path, round_event, copy_event):
    path = tmp_path / "log.jsonl"
    first, second = round_event(), copy_event()
    prefix = first.model_dump_json().encode("utf-8")
    path.write_bytes(prefix)
    api.append_event(path, second)
    raw = path.read_bytes()
    assert raw.startswith(prefix + b"\n")
    assert len(raw.splitlines()) == 2
    assert api.read_events(path) == [first, second]


def test_decision_log_command_prints_table(api, repo_root, tmp_path, round_event):
    path = tmp_path / "log.jsonl"
    path.write_text(round_event().model_dump_json() + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(repo_root), PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-B", "-m", "loop.decision_log", str(path)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip("\n") == api.render_table(api.read_events(path)).rstrip("\n")
