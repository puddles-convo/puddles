"""Independent acceptance of unattended-run.md 5.8; all data stays in this clone."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types

import pytest

from puddles import parking
from puddles.decision_log import DecisionLog
from schemas import LogEdit, LogPattern, LogPolicy, LogScores, RoundEvent, Scores

REPO = Path(__file__).resolve().parents[2]
BAD = ("0", "-1", "1.5", "abc", "", " 1")


@pytest.fixture
def work(monkeypatch):
    monkeypatch.chdir(REPO)
    for key in ("PUDDLES_PARK_AFTER", "PUDDLES_PARK_REVERTED", "PUDDLES_GATE_REMEASURE_KEPT"):
        monkeypatch.delenv(key, raising=False)
    with TemporaryDirectory(prefix=".acc-park-after-", dir=REPO) as directory:
        yield Path(directory)


def event(n, *, qualities=(4, 4.5), noise=0.25, action="reword", base="policy-v1", kept=None):
    scores = None if qualities is None else LogScores(
        before=Scores(empathy=4, relevance=4, quality=qualities[0]),
        after=Scores(empathy=4, relevance=4, quality=qualities[1]), noise=noise)
    return RoundEvent(
        round=n, outcome="kept" if kept else "reverted", reason="independent acceptance",
        policy=LogPolicy(base=base, base_sha256="x", in_force=kept or base), scores=scores,
        pattern=LogPattern(id=f"pattern-r{n:02}",
                           same_as_pattern_id="pattern-r01" if n > 1 else None,
                           name="Listening", seen_in=1, out_of=1, status_before="open"),
        edit=LogEdit(id=f"edit-r{n:02}", action=action, rule_id="C1",
                     rule_text=None if action == "remove" else "Try listening first."))


def history(path, *events):
    log = DecisionLog(path)
    for item in events:
        log.append(item)
    return log


def test_acc_unset_and_two_preserve_parent_behavior(work, monkeypatch):
    source = subprocess.check_output(
        ["git", "show", "653cb4e:puddles/parking.py"], cwd=REPO, text=True)
    parent = types.ModuleType("acceptance_parent_parking")
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    exec(compile(source, "<parent parking>", "exec"), parent.__dict__)
    for action in ("reword", "remove", "add"):
        for index, qualities in enumerate(((4, 4.5), (4, 4), (4, 3.9), (4, 3), None, (None, 4), (4, None))):
            for count in range(4):
                log = history(work / f"{action}-{index}-{count}",
                              *(event(n, action=action, qualities=qualities) for n in range(1, count + 1)),
                              event(10, kept="policy-v2"))
                for value in (None, "2"):
                    if value is None:
                        monkeypatch.delenv(parking.PARK_AFTER, raising=False)
                    else:
                        monkeypatch.setenv(parking.PARK_AFTER, value)
                    assert parking.park_after() == 2
                    for version in ("policy-v1", "policy-v2", "unrelated"):
                        assert asdict(parking.parked(log, version)) == asdict(parent.parked(log, version))
                        assert [asdict(t) for t in parking.to_compare(log, version)] == [
                            asdict(t) for t in parent.to_compare(log, version)]


def test_acc_one_parks_each_failed_rule_and_pattern_whatever_scores(work, monkeypatch):
    monkeypatch.setenv(parking.PARK_AFTER, "1")
    for action in ("reword", "remove", "add"):
        for i, qualities in enumerate(((4, 4.5), (4, 4), (4, 3.9), (4, 3), None, (None, 4), (4, None))):
            log = history(work / f"{action}-{i}", event(1, action=action, qualities=qualities))
            current = parking.parked(log, "policy-v1")
            assert set(current.rules) == ({"C1"} if action != "add" else set())
            assert set(current.patterns) == {"pattern-r01"}
            assert [t.edit_id for t in current.patterns["pattern-r01"]] == ["edit-r01"]
            assert parking.parked(log, "unrelated").rules == {}
            assert parking.parked(log, "unrelated").patterns == {}
    # A keep and an editless revert are not failed tries, even with negative scores.
    editless = event(2, qualities=(4, 3))
    editless.edit = None
    log = history(work / "not-failures", event(1, kept="policy-v2"), editless)
    assert parking.parked(log, "policy-v1").rules == {}
    assert parking.parked(log, "policy-v1").patterns == {}


def test_acc_threshold_negative_shortcut_and_56_survive_keeps(work, monkeypatch):
    for threshold in (1, 2, 10):
        monkeypatch.setenv(parking.PARK_AFTER, str(threshold))
        log = history(work / f"count-{threshold}")
        for n in range(1, threshold + 1):
            log.append(event(n))
            parked = parking.parked(log, "policy-v1")
            assert set(parked.rules) == ({"C1"} if n == threshold else set())
            assert set(parked.patterns) == ({"pattern-r01"} if n == threshold else set())
        # Only a fall strictly beyond noise survives keeps, independent of the threshold.
        for i, (qualities, noise, carries) in enumerate((
            ((4, 3), .25, True), ((4, 3.9), .25, False), ((4, 3.75), .25, False),
            ((4.15, 3.875), .275, False),
            ((4, 4.5), .25, False), (None, .25, False))):
            log = history(work / f"carry-{threshold}-{i}", event(1, qualities=qualities, noise=noise),
                          event(2, kept="policy-v2"), event(3, base="policy-v2", kept="policy-v3"))
            if qualities and qualities[1] < qualities[0]:
                assert set(parking.parked(log, "policy-v1").rules) == {"C1"}
                assert set(parking.parked(log, "policy-v1").patterns) == {"pattern-r01"}
            for version in ("policy-v2", "policy-v3", "unrelated"):
                parked = parking.parked(log, version)
                expected = carries and version != "unrelated"
                assert set(parked.rules) == ({"C1"} if expected else set())
                assert set(parked.patterns) == ({"pattern-r01"} if expected else set())


@pytest.mark.parametrize("value", [*BAD, "0" * 5000], ids=[*map(repr, BAD), "long-zero"])
def test_acc_bad_value_outside_driver_defaults_without_raising(work, monkeypatch, value):
    monkeypatch.setenv(parking.PARK_AFTER, value)
    assert parking.park_after() == 2


@pytest.mark.parametrize("value", [*BAD, "1", "2", "10", None])
def test_acc_driver_rejects_before_clients_and_accepts_valid_settings(work, monkeypatch, capsys, value):
    import scripts.run_loop as driver
    from tests.test_unattended_driver import patch_real, write_config

    write_config(work, monkeypatch)
    clients = patch_real(monkeypatch)
    if value is not None:
        monkeypatch.setenv(parking.PARK_AFTER, value)
    root = work / "run"
    result = driver.main(["--root", str(root), "--gate", "real", "--max-rounds", "1", "--check"])
    captured = capsys.readouterr()
    if value in BAD:
        assert result == 2
        assert "PUDDLES_PARK_AFTER" in captured.err
        assert clients == []
        assert not root.exists()
    else:
        assert result == 0, captured.err
        assert "check ok:" in captured.out
        assert "OpenAILLM" in clients
