"""Acceptance from docs/specs/persona-pilot.md, authored blind to the pilot.

No pilot imports at collection time. Every output and tracking store is temporary.
Spec defects and the conservative readings used here are reported with delivery.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import builtins
import importlib
import json
import os
from pathlib import Path
import re
import socket
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PERSONAS = ROOT / "fixtures/personas.json"
START = date(2026, 1, 5)
ARMS = ("notes_on", "notes_off")
POLICY_SENTENCE = (
    "The policy is the same for every person: there are no per-person policy rules. "
    "The personal layer is the notes (situation, people, what_helps, goals, follow_up, "
    "preferences), which the note-taker keeps and the helper is given at the start of a session."
)
LIMITS_SENTENCE = (
    "Limits: three personas, one run each, simulated users and a model judge; "
    "differences this small are not evidence of an effect."
)
REMEMBER = "I remember what you told me last time."


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A stray network call fails, and MLflow cannot use a repository store."""
    import mlflow
    from mlflow.tracking import fluent

    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if (key.endswith("API_KEY") or key.startswith(("HELPER_", "SIMUSER_", "MLFLOW_"))
                or key.startswith("PUDDLES_JUDGE") or key in {
                    "OPENAI_BASE_URL", "PUDDLES_AGENT_MODEL", "PUDDLES_AGENT_TEMPERATURE",
                    "PUDDLES_FALLBACK_JUDGE_MODEL",
                }):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    old_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(fluent, "_active_experiment_id", fluent._active_experiment_id)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/tracking.db")

    def forbidden(*args, **kwargs):
        raise AssertionError("persona acceptance tests forbid network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    yield
    mlflow.flush_trace_async_logging()
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(old_uri)


@pytest.fixture
def pilot():
    return importlib.import_module("puddles.inference.pilot")


@pytest.fixture
def command(pilot):
    return importlib.import_module("scripts.persona_pilot")


@pytest.fixture
def personas(pilot):
    return pilot.load_personas(PERSONAS)


@pytest.fixture
def store(tmp_path):
    from puddles import tracing
    from puddles.inference.note_store import FakeUserMemory, NoteStore

    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/tracking.db")
    return NoteStore(FakeUserMemory(), tmp_path / "store")


def clients(pilot, answers=None, helper=None):
    from loop.llm import FakeClient
    from puddles.inference.crisis import load_path
    from scripts.chat import fake_checker

    path = load_path()
    return pilot.Clients(
        helper=helper or FakeClient(lambda s, m: "What would help you feel heard?"),
        checker=fake_checker(path),
        user=FakeClient(answers if answers is not None else lambda s, m: "I want to talk more."),
        path=path,
    )


def run_one(pilot, persona, n, arm, *, store, policy, tmp_path, bundle=None, **kwargs):
    from scripts.chat import FakeNoteTaker

    return pilot.run_session(
        persona, n, arm, clients=bundle or clients(pilot), policy=policy, store=store,
        note_llm=FakeNoteTaker(), root=tmp_path / "store", start=START, **kwargs,
    )


def users(record):
    return [t.text for t in record.turns if t.speaker == "user"]


def snapshot(note_id, fact, kind="situation", day="2026-01-05", due=None):
    return dict(note_id=note_id, kind=kind, fact=fact, last_confirmed=day, due_date=due)


def tree(path):
    """File bytes and empty directories both matter to the no-write promise."""
    if not path.exists():
        return None
    if path.is_file():
        return {".": path.read_bytes()}
    return {str(p.relative_to(path)): p.read_bytes() if p.is_file() else None
            for p in sorted(path.rglob("*"))}


def invoke(command, args):
    # argparse may raise SystemExit for its own argument validation.
    try:
        return command.main(args)
    except SystemExit as exc:
        return exc.code


def arguments(policy_path, out):
    return ["--personas", str(PERSONAS), "--policy", str(policy_path),
            "--out", str(out), "--start", START.isoformat()]


class Judge:
    version = "acceptance-judge"

    def __init__(self, modes=()):
        self.modes = iter(modes)
        self.calls = []

    def score(self, record):
        from schemas import ScoreRecord

        self.calls.append(record)
        mode = next(self.modes, "ok")
        if mode == "raise":
            raise RuntimeError("judge unavailable")
        score = ScoreRecord(
            id=f"score-{record.id}", conversation_id="another-conversation" if mode == "wrong" else record.id,
            judge_version=self.version, written_by="judge", created_at=record.created_at,
            empathy=5, relevance=2, emotion_before=4, emotion_after=3,
            strategies=["Question" for t in record.turns if t.speaker == "helper"], note="acceptance",
        )
        return SimpleNamespace(score=score, usage=[])


# 1. Arms differ only in memory.
def test_arms_share_policy_opener_and_scripted_turns(
        pilot, personas, store, policy, tmp_path, monkeypatch):
    from loop.prompts import NO_NOTES, render_helper_system
    from puddles.inference.live import finish_session
    from puddles.inference.turn import run_turn

    seed_fact = "Prefers listening before advice"
    seed = run_one(pilot, personas[0], 1, "notes_on", store=store, policy=policy, tmp_path=tmp_path,
                   bundle=clients(pilot, [f"Please listen. [note:preferences:{seed_fact}]"]), turns=2)
    assert [n["fact"] for n in seed.notes_after] == [seed_fact]
    seen_turns, finished = [], []

    def turn_spy(session, text):
        seen_turns.append((session.session_id, text))
        assert session.policy is policy
        assert session.consent_for_training is True
        return run_turn(session, text)

    def finish_spy(session, *args, **kwargs):
        finished.append((session.session_id, session.remember_me, list(session.user_notes), kwargs))
        return finish_session(session, *args, **kwargs)

    monkeypatch.setattr(pilot, "run_turn", turn_spy)
    monkeypatch.setattr(pilot, "finish_session", finish_spy)
    bundles = {arm: clients(pilot, ["A scripted second message.", "A scripted third message."])
               for arm in ARMS}
    results = {arm: run_one(pilot, personas[0], 2, arm, store=store, policy=policy,
                           tmp_path=tmp_path, bundle=bundles[arm], turns=3)
               for arm in ARMS}
    expected = [personas[0].sessions[1].opener, "A scripted second message.", "A scripted third message."]
    for arm in ARMS:
        result = results[arm]
        assert not result.failed and users(result.record) == expected
        assert [text for sid, text in seen_turns if sid == result.session_id] == expected
        assert len(bundles[arm].helper.calls) == 3
        assert len(bundles[arm].user.calls) == 2
        assert bundles[arm].helper is not bundles[arm].checker
        expected_notes = [seed_fact] if arm == "notes_on" else []
        assert result.notes_given == expected_notes
        assert all(c["system"] == render_helper_system(policy, expected_notes) for c in bundles[arm].helper.calls)
        assert all((NO_NOTES in c["system"]) is (arm == "notes_off") for c in bundles[arm].helper.calls)
    assert [{k: v for k, v in c.items() if k != "system"} for c in bundles["notes_on"].helper.calls] == [
        {k: v for k, v in c.items() if k != "system"} for c in bundles["notes_off"].helper.calls]
    assert bundles["notes_on"].user.calls == bundles["notes_off"].user.calls
    assert [(sid, remember, notes) for sid, remember, notes, _ in finished] == [
        ("laidoff-on-s2", True, [seed_fact]), ("laidoff-off-s2", False, [])]
    for _, _, _, kwargs in finished:
        assert kwargs["today"] == START + timedelta(days=7)
        assert kwargs["now"] == datetime(2026, 1, 12, 12, tzinfo=timezone.utc)


def test_notes_off_never_loads_or_stores_notes_even_when_on_has_them(
        pilot, command, personas, store, policy, tmp_path, monkeypatch):
    from loop.prompts import NO_NOTES
    from puddles.inference.live import finish_session

    person = personas[0]
    first = run_one(pilot, person, 1, "notes_on", store=store, policy=policy, tmp_path=tmp_path,
                    bundle=command.fake_clients(person, 1, "notes_on", clients(pilot).path), turns=2, fake=True)
    assert first.notes_after
    on_before = tree(store.root / "notes" / "laidoff-on")
    seen = []

    def finish_spy(session, *a, **kw):
        seen.append((session.user_id, session.remember_me, list(session.user_notes)))
        return finish_session(session, *a, **kw)

    def forbidden_start(*a, **kw):
        pytest.fail("notes_off must not call start_notes")

    monkeypatch.setattr(pilot, "finish_session", finish_spy)
    monkeypatch.setattr(pilot, "start_notes", forbidden_start)
    # An add marker tests consent, rather than relying on notes_off fake markers being empty.
    bundle = clients(pilot, ["Please listen. [note:preferences:Likes quiet conversation]"])
    result = run_one(pilot, person, 2, "notes_off", store=store, policy=policy,
                     tmp_path=tmp_path, bundle=bundle, turns=2)
    assert not result.failed and result.record.remember_me is False
    assert result.notes_given == result.notes_before == result.notes_after == []
    assert result.changes.added == [] and not result.changes.failed
    assert seen == [("laidoff-off", False, [])]
    assert all(NO_NOTES in call["system"] for call in bundle.helper.calls)
    for note in first.notes_after:
        assert all(note["fact"] not in c["system"] for c in bundle.helper.calls)
    assert store.index("laidoff-off").notes == [] and store.stored("laidoff-off") == {}
    assert tree(store.root / "notes" / "laidoff-on") == on_before


@pytest.mark.parametrize("turns,answers,expected_calls", [(1, [], 0), (6, ["[END]"], 1)])
def test_opener_only_and_user_end_are_successes(
        pilot, personas, store, policy, tmp_path, turns, answers, expected_calls):
    bundle = clients(pilot, answers)
    result = run_one(pilot, personas[0], 1, "notes_on", store=store, policy=policy,
                     tmp_path=tmp_path, bundle=bundle, turns=turns)
    assert not result.failed and result.reason == ""
    assert users(result.record) == [personas[0].sessions[0].opener]
    assert len(bundle.user.calls) == expected_calls


def test_failed_helper_still_counts_as_a_turn(pilot, personas, store, policy, tmp_path):
    from loop.errors import LLMError
    from loop.llm import FakeClient

    def fail(*a, **kw):
        raise LLMError("offline helper")

    bundle = clients(pilot, ["A second user message."], helper=FakeClient(fail))
    result = run_one(pilot, personas[0], 1, "notes_on", store=store, policy=policy,
                     tmp_path=tmp_path, bundle=bundle, turns=2)
    assert not result.failed and len(users(result.record)) == 2
    assert len(bundle.helper.calls) == 2 and len(bundle.user.calls) == 1
    assert len([t for t in result.record.turns if t.speaker == "helper"]) == 2


@pytest.mark.parametrize("turns", [0, -1])
def test_bad_turn_count_is_refused_before_any_work(pilot, personas, policy, tmp_path, turns):
    def forbidden(*a, **k):
        pytest.fail("turns below one must fail before touching a dependency")

    bomb = SimpleNamespace(index=forbidden, stored=forbidden)
    before = tree(tmp_path)
    with pytest.raises(ValueError):
        pilot.run_session(personas[0], 1, "notes_on", clients=None, policy=policy, store=bomb,
                          note_llm=None, root=tmp_path / "run", start=START, turns=turns)
    assert tree(tmp_path) == before


# 2. Memory carries across sessions and deletion stays deleted.
def test_memory_reaches_helper_and_grief_forgetting_persists(
        pilot, command, personas, store, policy, tmp_path, monkeypatch):
    from loop.prompts import render_helper_system
    from puddles.inference.live import finish_session

    person = personas[2]
    finishes = []

    def finish_spy(session, *a, **kw):
        finishes.append((session.session_id, kw["today"], kw["now"]))
        return finish_session(session, *a, **kw)

    monkeypatch.setattr(pilot, "finish_session", finish_spy)
    results, bundles = [], []
    for n in (1, 2, 3):
        bundle = command.fake_clients(person, n, "notes_on", clients(pilot).path)
        result = run_one(pilot, person, n, "notes_on", store=store, policy=policy,
                         tmp_path=tmp_path, bundle=bundle, turns=2, fake=True)
        assert not result.failed and not result.changes.failed
        assert all(c["system"] == render_helper_system(policy, result.notes_given) for c in bundle.helper.calls)
        assert any(p.read_text(encoding="utf-8").find(result.record.id) >= 0
                   for p in (tmp_path / "store" / "sessions").rglob("*.json"))
        results.append(result)
        bundles.append(bundle)
    one, two, three = results
    assert one.notes_before == [] and len(one.notes_after) == 3
    facts = [note["fact"] for note in one.notes_after]
    assert two.notes_given == facts
    assert all(fact in bundles[1].helper.calls[0]["system"] for fact in facts)
    forgotten = next(note for note in one.notes_after if note["note_id"] == "note-grief-on-002")
    assert "brother" in forgotten["fact"]
    assert two.changes.deleted == [{"note_id": forgotten["note_id"], "fact": forgotten["fact"]}]
    assert two.notes_before == one.notes_after
    for result in (two, three):
        assert forgotten["note_id"] not in [n["note_id"] for n in result.notes_after]
        assert forgotten["fact"] not in [n["fact"] for n in result.notes_after]
    assert three.notes_before == two.notes_after
    assert forgotten["fact"] not in three.notes_given
    assert all(forgotten["fact"] not in c["system"] for c in bundles[2].helper.calls)
    assert forgotten["note_id"] not in store.stored("grief-on")
    for n, (sid, today, now) in enumerate(finishes, 1):
        day = START + timedelta(days=7 * (n - 1))
        assert sid == f"grief-on-s{n}" and today == day
        assert now == datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=12)


def test_snapshot_is_ordered_file_only_read(pilot, tmp_path):
    from puddles.inference.note_store import NoteStore

    def forbidden(*a, **k):
        pytest.fail("snapshot_notes must never call memory")

    store = NoteStore(SimpleNamespace(recall=forbidden, ensure_audience=forbidden,
                                     ingest=forbidden, delete=forbidden, wait=forbidden), tmp_path / "store")
    before = tree(tmp_path)
    assert pilot.snapshot_notes(store, "reader-on") == []
    assert tree(tmp_path) == before
    folder = store.root / "notes" / "reader-on"
    folder.mkdir(parents=True)
    index = [dict(note_id=f"note-reader-on-{n:03d}", kind="follow_up" if n == 2 else "people",
                  last_confirmed="2026-01-05", due_date="2026-01-09" if n == 2 else None)
             for n in (2, 3, 1)]
    (folder / "index.json").write_text(json.dumps(dict(user_id="reader-on", notes=index)), encoding="utf-8")
    stored = {f"note-reader-on-{n:03d}": dict(
        id=f"note-reader-on-{n:03d}", user_id="reader-on", session_id="reader-on-s1",
        kind="people", fact=f"Stored fact {n}", last_confirmed="2025-12-01",
        created_at="2026-01-05T12:00:00Z", written_by="note-taker") for n in (1, 2, 4)}
    (folder / "notes.json").write_text(json.dumps(stored), encoding="utf-8")
    before = tree(tmp_path)
    assert pilot.snapshot_notes(store, "reader-on") == [
        snapshot("note-reader-on-002", "Stored fact 2", "follow_up", due="2026-01-09"),
        snapshot("note-reader-on-001", "Stored fact 1", "people"),
    ]
    assert tree(tmp_path) == before


def test_note_changes_uses_correct_snapshot_and_preserves_order(pilot):
    from puddles.inference.notes import NotesResult

    before = [snapshot("old", "old fact"), snapshot("both", "before fact")]
    after = [snapshot("new", "new fact"), snapshot("both", "after fact")]
    result = NotesResult(added=["new", "both", "missing"], replaced=["both", "new"],
                         confirmed=["new", "old", "missing"], deleted=["old", "both", "missing"],
                         failed=True, reason="scripted reason")
    originals = deepcopy((before, after, result))
    change = pilot.note_changes(before, after, result)
    assert asdict(change) == dict(
        added=[dict(note_id="new", fact="new fact"), dict(note_id="both", fact="after fact"),
               dict(note_id="missing", fact="")],
        replaced=[dict(note_id="both", fact="before fact"), dict(note_id="new", fact="new fact")],
        confirmed=[dict(note_id="new", fact="new fact"), dict(note_id="old", fact="old fact"),
                   dict(note_id="missing", fact="")],
        deleted=[dict(note_id="old", fact="old fact"), dict(note_id="both", fact="before fact"),
                 dict(note_id="missing", fact="")], failed=True, reason="scripted reason")
    assert (before, after, result) == originals
    assert asdict(pilot.note_changes([], [], NotesResult())) == dict(
        added=[], replaced=[], confirmed=[], deleted=[], failed=False, reason="")


# 3. Honest numbers, supplied entirely by a hand-built dictionary.
def hand_session(person, arm, n, status="scored", empathy=3, relevance=3):
    sid = f"{person}-{'on' if arm == 'notes_on' else 'off'}-s{n}"
    failed = status == "failed"
    return dict(
        persona_id=person, session=n, arm=arm, user_id=sid.rsplit("-s", 1)[0], session_id=sid,
        failed=failed, reason="session unavailable" if failed else "", notes_given=[],
        record=None if failed else dict(
            id=sid, written_by="live", created_at="2026-01-05T12:00:00Z", source="live",
            policy_version="hand-policy-17", remember_me=arm == "notes_on", consent_for_training=True,
            turns=[dict(speaker="user", text="User text"), dict(speaker="helper", text=f"{arm} first"),
                   dict(speaker="user", text="More text"), dict(speaker="helper", text=f"{arm} second")]),
        notes_before=[], notes_after=[],
        changes=dict(added=[], replaced=[], confirmed=[], deleted=[], failed=False, reason=""),
        score=dict(status=status, reason="judge unavailable" if status == "unscored" else "",
                   empathy=empathy if status == "scored" else None,
                   relevance=relevance if status == "scored" else None,
                   emotion_before=4 if status == "scored" else None,
                   emotion_after=3 if status == "scored" else None,
                   judge_version="hand-judge-19" if status == "scored" else None,
                   note="hand score" if status == "scored" else None))


def hand_pilot():
    modes = {
        "alpha": {"notes_on": [("scored", 5, 1), ("failed", 0, 0), ("scored", 4, 2)],
                  "notes_off": [("unscored", 0, 0), ("scored", 2, 5), ("scored", 1, 4)]},
        "beta": {"notes_on": [("scored", 2, 5), ("unscored", 0, 0), ("scored", 3, 1)],
                 "notes_off": [("scored", 5, 1), ("failed", 0, 0), ("unscored", 0, 0)]},
        "gamma": {"notes_on": [("failed", 0, 0)] * 3,
                  "notes_off": [("scored", 4, 2), ("scored", 3, 3), ("scored", 2, 5)]},
    }
    people = [dict(persona_id=p, description=f"Hand-built {p}", sessions={
        arm: [hand_session(p, arm, n, *values) for n, values in enumerate(arms[arm], 1)]
        for arm in ARMS}) for p, arms in modes.items()]
    alpha, beta, gamma = [p["sessions"]["notes_on"] for p in people]
    alpha[0]["notes_after"] = [snapshot("a", "A fact | with\na newline")]
    alpha[0]["changes"]["added"] = [dict(note_id="a", fact="A fact | with\na newline")]
    alpha[2]["notes_after"] = [snapshot("b", "Second fact"), snapshot("c", "Third fact")]
    alpha[2]["changes"]["deleted"] = [dict(note_id="a", fact="A fact | with\na newline")]
    alpha[2]["changes"]["replaced"] = [dict(note_id="old", fact="Replaced fact")]
    beta[2]["notes_after"] = [snapshot("d", "Fourth fact")]
    beta[2]["changes"]["deleted"] = [dict(note_id="e", fact="Deleted E"), dict(note_id="f", fact="Deleted F")]
    # A failed third session can retain snapshots, but these must not be counted.
    gamma[2]["notes_after"] = [snapshot("g", "Stale G"), snapshot("h", "Stale H")]
    people[0]["sessions"]["notes_off"][2]["record"]["turns"] = [dict(speaker="helper", text="Only one reply")]
    return dict(policy_version="hand-policy-17", policy_path="missing-policy.md", turns=7, fake=False,
                start=START.isoformat(), judge_version="hand-judge-19", policy_sentence=POLICY_SENTENCE,
                limits_sentence=LIMITS_SENTENCE, personas=people)


def render_without_files(pilot, data, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("render_report must derive everything from its dictionary without reading files")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        return pilot.render_report(data)


def table_row(report, name):
    rows = [line for line in report.splitlines() if line.startswith(f"| {name} |")]
    assert len(rows) == 1, (name, rows)
    return [c.strip() for c in rows[0].strip("|").split("|")][1:]


@pytest.mark.parametrize("values,expected", [([], "not measured"), ([1, 1, 2], 1.33),
                                             ([5, 2], 3.5), ([1, 2, 2], 1.67)])
def test_mean_empty_and_rounding(pilot, values, expected):
    assert pilot.mean_or_not_measured(values) == expected


def test_hand_built_report_arithmetic_statuses_counts_and_structure(pilot, monkeypatch):
    data = hand_pilot()
    before = deepcopy(data)
    report = render_without_files(pilot, data, monkeypatch)
    assert data == before
    assert report.startswith("# Persona pilot\n")
    assert report.count(POLICY_SENTENCE) == report.count(LIMITS_SENTENCE) == 1
    assert "Policy: hand-policy-17 (missing-policy.md). Turns per session: 7. Mode: real. Judge: hand-judge-19." in report
    assert report.count("### Notes after each session (notes_on)") == 3
    assert report.count("### Judge scores") == 3
    assert report.count("### Session 3, first two helper replies") == 3
    assert "## Overall" in report
    for label, on, off, diff in [
        ("Mean empathy, session 3", 3.5, 1.5, "+2.00"),
        ("Mean relevance, session 3", 1.5, 4.5, "-3.00"),
        ("Mean empathy, all sessions", 3.5, 2.83, "+0.67"),
        ("Mean relevance, all sessions", 2.25, 3.33, "-1.08"),
    ]:
        row = table_row(report, label)
        assert [float(row[0]), float(row[1])] == [on, off]
        assert row[2] == diff
    assert table_row(report, "Sessions scored") == ["4 of 9", "6 of 9", ""]
    assert table_row(report, "Sessions failed") == ["4", "1", ""]
    assert table_row(report, "Notes kept after session 3") == ["3", "0", ""]
    assert table_row(report, "Notes deleted on request") == ["3", "0", ""]
    alpha = report.split("## alpha:", 1)[1].split("## beta:", 1)[0]
    assert "| 1 | 5 | unscored | not measured | 1 | unscored | not measured |" in alpha
    assert "| 2 | failed | 2 | not measured | failed | 5 | not measured |" in alpha
    assert "| 3 | 4 | 1 | +3 | 2 | 4 | -2 |" in alpha
    assert "| 1 | notes_on first | Only one reply |" in alpha
    assert "| 2 | notes_on second | - |" in alpha
    assert "| 1 | situation | A fact \\| with a newline | added |" in alpha
    assert "| 3 |" in alpha and "A fact \\| with a newline | deleted |" in alpha
    assert "Replaced fact | replaced |" in alpha
    assert "| 2 | - | (session failed) | |" in alpha
    beta = report.split("## beta:", 1)[1].split("## gamma:", 1)[0]
    assert "| 1 | - | (no notes) | |" in beta
    assert "| 2 | - | (no notes) | |" in beta
    assert "| 1 | notes_on first | notes_off first |" in beta
    gamma = report.split("## gamma:", 1)[1].split("## Overall", 1)[0]
    assert "| 1 | - | notes_off first |" in gamma
    assert "Stale G" not in gamma and "Stale H" not in gamma


@pytest.mark.parametrize("empty_arm,status", [("notes_on", "failed"), ("notes_off", "unscored")])
def test_empty_arm_is_not_measured_in_every_mean(pilot, monkeypatch, empty_arm, status):
    data = hand_pilot()
    data["personas"] = data["personas"][:1]
    data["personas"][0]["sessions"][empty_arm] = [hand_session("alpha", empty_arm, n, status) for n in (1, 2, 3)]
    report = render_without_files(pilot, data, monkeypatch)
    col = ARMS.index(empty_arm)
    for measure in ("empathy", "relevance"):
        for scope in ("session 3", "all sessions"):
            row = table_row(report, f"Mean {measure}, {scope}")
            assert row[col] == row[2] == "not measured"
    assert table_row(report, "Sessions scored")[col] == "0 of 3"


def test_integer_score_differences_include_zero(pilot, monkeypatch):
    data = hand_pilot()
    data["personas"] = data["personas"][:1]
    data["personas"][0]["sessions"] = {
        arm: [hand_session("alpha", arm, n, empathy=2, relevance=5) for n in (1, 2, 3)] for arm in ARMS}
    report = render_without_files(pilot, data, monkeypatch)
    assert "| 1 | 2 | 2 | 0 | 5 | 5 | 0 |" in report


# 4. Failure handling and continuing to later sessions.
@pytest.mark.parametrize("exception_name", ["SimulatedUserError", "LLMError", "RuntimeError"])
def test_user_exception_still_finishes_and_pilot_continues(
        pilot, personas, store, policy, tmp_path, monkeypatch, exception_name):
    from loop import errors
    from puddles.inference.live import finish_session
    from scripts.chat import FakeNoteTaker

    exception_type = RuntimeError if exception_name == "RuntimeError" else getattr(errors, exception_name)
    person = personas[0].model_copy(deep=True)
    person.sessions[1].opener += " [note:what_helps:Failure session fact]"
    order, finished = [], []

    def make(persona, n, arm):
        order.append((persona.persona_id, arm, n))
        def reply(s, m):
            if arm == "notes_on" and n == 2:
                raise exception_type("scripted user failure")
            return "I want to keep talking."
        return clients(pilot, reply)

    def finish_spy(session, *a, **kw):
        finished.append(session.session_id)
        return finish_session(session, *a, **kw)

    monkeypatch.setattr(pilot, "finish_session", finish_spy)
    judge = Judge()
    result = pilot.run_pilot([person], make_clients=make, policy=policy, store=store,
                             note_llm=FakeNoteTaker(), judge=judge, root=tmp_path / "store",
                             turns=2, start=START)
    assert order == [(person.persona_id, arm, n) for arm in ARMS for n in (1, 2, 3)]
    assert len(finished) == 6 and "laidoff-on-s2" in finished
    on = result["personas"][0]["sessions"]["notes_on"]
    assert on[1]["failed"] and on[1]["record"] is None
    assert on[1]["reason"] == f"{exception_name}: scripted user failure"
    assert on[1]["score"]["status"] == "failed"
    assert on[1]["score"]["empathy"] is on[1]["score"]["relevance"] is None
    assert any(n["fact"] == "Failure session fact" for n in on[1]["notes_after"])
    assert "Failure session fact" in on[2]["notes_given"]
    assert on[2]["score"]["status"] == "scored" and not on[2]["failed"]
    assert len(judge.calls) == 5
    report = pilot.render_report(result)
    assert "| 2 | failed | 5 | not measured | failed | 2 | not measured |" in report
    assert table_row(report, "Sessions scored") == ["2 of 3", "3 of 3", ""]
    assert not (tmp_path / "store" / "pilot.json").exists()
    assert not (tmp_path / "store" / "report.md").exists()


def test_finish_exception_is_recorded_and_later_sessions_run(
        pilot, personas, store, policy, tmp_path, monkeypatch):
    from puddles.inference.live import finish_session
    from scripts.chat import FakeNoteTaker

    calls = []
    def finish(session, *a, **kw):
        calls.append(session.session_id)
        if session.session_id == "laidoff-on-s2":
            raise OSError("cannot finish")
        return finish_session(session, *a, **kw)

    monkeypatch.setattr(pilot, "finish_session", finish)
    judge = Judge()
    result = pilot.run_pilot(personas[:1], make_clients=lambda *a: clients(pilot), policy=policy,
                             store=store, note_llm=FakeNoteTaker(), judge=judge,
                             root=tmp_path / "store", turns=1, start=START)
    on = result["personas"][0]["sessions"]["notes_on"]
    assert len(calls) == 6 and len(judge.calls) == 5
    assert on[1]["record"] is None and on[1]["failed"]
    assert on[1]["reason"] == "OSError: cannot finish"
    assert on[1]["changes"] == dict(added=[], replaced=[], confirmed=[], deleted=[], failed=True, reason="")
    assert on[2]["score"]["status"] == "scored"


def test_judge_failure_and_wrong_conversation_are_unscored_and_continue(
        pilot, personas, store, policy, tmp_path):
    from scripts.chat import FakeNoteTaker

    judge = Judge(["raise", "wrong"])
    result = pilot.run_pilot(personas[:1], make_clients=lambda *a: clients(pilot), policy=policy,
                             store=store, note_llm=FakeNoteTaker(), judge=judge,
                             root=tmp_path / "store", turns=1, start=START)
    on = result["personas"][0]["sessions"]["notes_on"]
    assert len(judge.calls) == 6
    assert [s["score"]["status"] for s in on] == ["unscored", "unscored", "scored"]
    assert "RuntimeError: judge unavailable" == on[0]["score"]["reason"]
    assert "another-conversation" in on[1]["score"]["reason"]
    for session in on[:2]:
        assert session["failed"] is False and session["record"] is not None
        assert all(session["score"][k] is None for k in ("empathy", "relevance", "emotion_before", "emotion_after"))
    report = pilot.render_report(result)
    assert table_row(report, "Sessions scored") == ["1 of 3", "3 of 3", ""]
    assert float(table_row(report, "Mean empathy, all sessions")[0]) == 5


def test_score_session_never_judges_failed_result(pilot):
    result = pilot.SessionResult(
        persona_id="alpha", session=2, arm="notes_on", user_id="alpha-on", session_id="alpha-on-s2",
        notes_given=[], record=None, notes_before=[], notes_after=[],
        changes=pilot.NoteChanges([], [], [], [], False, ""), failed=True, reason="lost session")
    judge = Judge()
    score = pilot.score_session(judge, result)
    assert (score.status, score.score, score.reason) == ("failed", None, "lost session")
    assert judge.calls == []


def test_missing_record_is_failed(pilot, personas, store, policy, tmp_path, monkeypatch):
    from puddles.inference.notes import NotesResult

    monkeypatch.setattr(pilot, "finish_session", lambda *a, **kw: (None, NotesResult()))
    result = run_one(pilot, personas[0], 1, "notes_on", store=store, policy=policy, tmp_path=tmp_path, turns=1)
    assert result.failed and result.record is None and result.reason == "no record"


@pytest.mark.parametrize("where", ["user", "finish"])
def test_keyboard_interrupt_is_not_swallowed(pilot, personas, store, policy, tmp_path, monkeypatch, where):
    def interrupt(*a, **kw):
        raise KeyboardInterrupt()

    bundle = clients(pilot, interrupt)
    if where == "finish":
        monkeypatch.setattr(pilot, "finish_session", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_one(pilot, personas[0], 1, "notes_on", store=store, policy=policy,
                tmp_path=tmp_path, bundle=bundle, turns=1 if where == "finish" else 2)


# 5. The offline command and its output contract.
def assert_session_shape(value):
    from schemas import ConversationRecord

    assert set(value) == {"persona_id", "session", "arm", "user_id", "session_id", "failed", "reason",
                          "notes_given", "record", "notes_before", "notes_after", "changes", "score"}
    assert value["arm"] in ARMS and value["session"] in (1, 2, 3)
    suffix = "on" if value["arm"] == "notes_on" else "off"
    assert value["user_id"] == f"{value['persona_id']}-{suffix}"
    assert value["session_id"] == f"{value['user_id']}-s{value['session']}"
    assert type(value["failed"]) is bool and isinstance(value["reason"], str)
    assert isinstance(value["notes_given"], list) and all(isinstance(f, str) for f in value["notes_given"])
    for name in ("notes_before", "notes_after"):
        assert isinstance(value[name], list)
        for note in value[name]:
            assert set(note) == {"note_id", "kind", "fact", "last_confirmed", "due_date"}
            assert all(isinstance(note[k], str) for k in ("note_id", "kind", "fact"))
            date.fromisoformat(note["last_confirmed"])
            if note["due_date"] is not None:
                date.fromisoformat(note["due_date"])
    change = value["changes"]
    assert set(change) == {"added", "replaced", "confirmed", "deleted", "failed", "reason"}
    assert type(change["failed"]) is bool and isinstance(change["reason"], str)
    for name in ("added", "replaced", "confirmed", "deleted"):
        assert isinstance(change[name], list)
        assert all(set(n) == {"note_id", "fact"} and all(isinstance(v, str) for v in n.values()) for n in change[name])
    score = value["score"]
    assert set(score) == {"status", "reason", "empathy", "relevance", "emotion_before", "emotion_after", "judge_version", "note"}
    assert score["status"] in ("scored", "unscored", "failed") and isinstance(score["reason"], str)
    assert all(score[k] is None or isinstance(score[k], str) for k in ("judge_version", "note"))
    for key in ("empathy", "relevance", "emotion_before", "emotion_after"):
        assert type(score[key]) is int if score["status"] == "scored" else score[key] is None
    if value["record"] is not None:
        ConversationRecord.model_validate(value["record"])


@pytest.mark.parametrize("only", [None, "grief"])
def test_fake_command_files_shapes_report_and_private_stdout(
        pilot, command, policy_path, policy, tmp_path, capsys, only):
    out = tmp_path / "run"
    out.mkdir()  # An existing empty directory is explicitly allowed.
    args = arguments(policy_path, out) + ["--fake", "--turns", "2"]
    if only:
        args += ["--only", only]
    assert invoke(command, args) == 0
    stdout = capsys.readouterr().out
    data = json.loads((out / "pilot.json").read_text(encoding="utf-8"))
    assert set(data) == {"policy_version", "policy_path", "turns", "fake", "start", "judge_version",
                         "policy_sentence", "limits_sentence", "personas"}
    assert data["policy_version"] == policy.version and data["policy_path"] == str(policy_path)
    assert data["turns"] == 2 and data["fake"] is True and data["start"] == START.isoformat()
    assert data["judge_version"] == "fake-judge"
    assert data["policy_sentence"] == POLICY_SENTENCE and data["limits_sentence"] == LIMITS_SENTENCE
    expected_people = ["grief"] if only else ["laidoff", "examprep", "grief"]
    assert [p["persona_id"] for p in data["personas"]] == expected_people
    paths = list((out / "sessions").rglob("s*.json"))
    assert len(paths) == 6 * len(expected_people)
    raw = json.loads(PERSONAS.read_text(encoding="utf-8"))
    source = {p["persona_id"]: p for p in raw["personas"]}
    expected_stdout = []
    for person in data["personas"]:
        assert set(person) == {"persona_id", "description", "sessions"}
        assert set(person["sessions"]) == set(ARMS)
        assert person["description"] == source[person["persona_id"]]["description"]
        for arm in ARMS:
            assert len(person["sessions"][arm]) == 3
            for n, session in enumerate(person["sessions"][arm], 1):
                assert_session_shape(session)
                path = out / "sessions" / person["persona_id"] / arm / f"s{n}.json"
                text = path.read_text(encoding="utf-8")
                assert text.rstrip("\n") == json.dumps(json.loads(text), indent=2)
                assert json.loads(text) == session
                assert not session["failed"] and session["score"]["status"] == "scored"
                assert session["record"]["remember_me"] is (arm == "notes_on")
                assert session["record"]["consent_for_training"] is True
                turns = session["record"]["turns"]
                user_text = [t["text"] for t in turns if t["speaker"] == "user"]
                assert len(user_text) == 2
                spec = source[person["persona_id"]]["sessions"][n - 1]
                assert user_text[0] == spec["opener"]
                markers = [m.replace("{user_id}", session["user_id"]) for m in spec["fake_markers"]]
                expected_line = "Yes, that is part of it."
                if arm == "notes_on" and markers:
                    expected_line += " " + " ".join(markers)
                assert user_text[1] == expected_line
                first_reply = next(t["text"] for t in turns if t["speaker"] == "helper")
                assert first_reply.startswith(REMEMBER) is bool(session["notes_given"])
                expected_score = 4 if session["notes_given"] else 3
                assert session["score"]["empathy"] == session["score"]["relevance"] == expected_score
                assert session["score"]["emotion_before"] == 4 and session["score"]["emotion_after"] == 3
                assert session["score"]["judge_version"] == "fake-judge" and session["score"]["note"] == "scripted"
                if arm == "notes_off":
                    assert session["notes_given"] == session["notes_before"] == session["notes_after"] == []
                assert all(t["text"] not in stdout for t in turns)
                assert all(note["fact"] not in stdout for note in session["notes_after"])
                c = session["changes"]
                expected_stdout.append(f"{person['persona_id']} {arm} s{n}: scored, notes +{len(c['added'])} ~{len(c['replaced'])} -{len(c['deleted'])}")
    assert stdout.splitlines() == expected_stdout + [f"report: {out}/report.md"]
    assert (out / "store" / "sessions").is_dir()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert report == pilot.render_report(data)
    assert report.count(POLICY_SENTENCE) == report.count(LIMITS_SENTENCE) == 1
    assert report.count("### Notes after each session (notes_on)") == len(expected_people)
    assert report.count("### Judge scores") == report.count("### Session 3, first two helper replies") == len(expected_people)
    assert report.startswith("# Persona pilot\n") and "## Overall" in report
    for name in expected_people:
        section = report.split(f"## {name}:", 1)[1].split("\n## ", 1)[0]
        assert "| added |" in section
    grief = next(p for p in data["personas"] if p["persona_id"] == "grief")["sessions"]["notes_on"]
    brother = next(n for n in grief[0]["notes_after"] if n["note_id"] == "note-grief-on-002")
    assert grief[1]["changes"]["deleted"] == [dict(note_id=brother["note_id"], fact=brother["fact"])]
    assert f"{brother['fact']} | deleted |" in report
    for session in grief[1:]:
        assert brother["note_id"] not in [n["note_id"] for n in session["notes_after"]]
    assert brother["fact"] not in grief[2]["notes_given"]


@pytest.mark.parametrize("contents", ["pilot", "unrelated", "empty-subdirectory", "file"])
def test_nonempty_output_is_refused_without_changing_it(command, policy_path, tmp_path, contents):
    out = tmp_path / "occupied"
    if contents == "file":
        out.write_bytes(b"keep this file")
    else:
        out.mkdir()
        if contents == "empty-subdirectory":
            (out / "store").mkdir()
        else:
            (out / ("pilot.json" if contents == "pilot" else "unrelated.bin")).write_bytes(b"unchanged\x00")
    before = tree(tmp_path)
    assert invoke(command, arguments(policy_path, out) + ["--fake"]) == 2
    assert tree(tmp_path) == before


@pytest.mark.parametrize("case", ["unknown-only", "zero-turns", "bad-turns", "bad-date", "missing-personas",
                                  "invalid-json", "invalid-personas", "missing-policy", "invalid-policy", "crlf-policy"])
def test_bad_command_inputs_write_nothing(command, policy_path, tmp_path, case):
    out = tmp_path / "uncreated"
    args = arguments(policy_path, out) + ["--fake"]
    if case == "unknown-only":
        args += ["--only", "nobody"]
    elif case == "zero-turns":
        args += ["--turns", "0"]
    elif case == "bad-turns":
        args += ["--turns", "not-an-int"]
    elif case == "bad-date":
        args[args.index("--start") + 1] = "2026-02-30"
    elif case in ("missing-personas", "invalid-json", "invalid-personas"):
        path = tmp_path / "bad-personas.json"
        if case != "missing-personas":
            path.write_text("{" if case == "invalid-json" else '{"personas": []}', encoding="utf-8")
        args[args.index("--personas") + 1] = str(path)
    else:
        path = tmp_path / "bad-policy.md"
        if case == "invalid-policy":
            path.write_text("invalid policy", encoding="utf-8")
        elif case == "crlf-policy":
            path.write_bytes(policy_path.read_bytes().replace(b"\n", b"\r\n"))
        args[args.index("--policy") + 1] = str(path)
    before = tree(tmp_path)
    assert invoke(command, args) == 2
    assert not out.exists() and tree(tmp_path) == before


def test_missing_real_configuration_is_one_error_and_no_output(command, policy_path, tmp_path, capsys):
    out = tmp_path / "uncreated"
    before = tree(tmp_path)
    assert invoke(command, arguments(policy_path, out)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert re.fullmatch(r"error: (ConfigError|ValueError): .*HELPER_URL.*\n", captured.err)
    assert not out.exists() and tree(tmp_path) == before


def test_command_writes_results_and_exits_zero_despite_session_and_judge_failures(
        pilot, command, policy_path, tmp_path, monkeypatch, capsys):
    from puddles.inference.live import finish_session

    def finish(session, *a, **kw):
        if session.session_id == "grief-on-s2":
            raise OSError("failure-conversation-sentinel-37")
        return finish_session(session, *a, **kw)

    original_score = command.FakeJudge.score
    calls = []

    def score(self, record):
        calls.append(record.id)
        if len(calls) == 1:
            raise RuntimeError("failure-judge-sentinel-73")
        return original_score(self, record)

    monkeypatch.setattr(pilot, "finish_session", finish)
    monkeypatch.setattr(command.FakeJudge, "score", score)
    out = tmp_path / "run"
    assert invoke(command, arguments(policy_path, out) + ["--fake", "--only", "grief", "--turns", "1"]) == 0
    stdout = capsys.readouterr().out
    data = json.loads((out / "pilot.json").read_text(encoding="utf-8"))
    on = data["personas"][0]["sessions"]["notes_on"]
    assert [s["score"]["status"] for s in on] == ["unscored", "failed", "scored"]
    assert len(calls) == 5
    assert len(list((out / "sessions").rglob("s*.json"))) == 6
    for person in data["personas"]:
        for arm in ARMS:
            for session in person["sessions"][arm]:
                assert_session_shape(session)
    assert "grief notes_on s1: unscored, notes +0 ~0 -0" in stdout
    assert "grief notes_on s2: failed, notes +0 ~0 -0" in stdout
    assert "failure-conversation-sentinel-37" not in stdout
    assert "failure-judge-sentinel-73" not in stdout
    assert (out / "report.md").read_text(encoding="utf-8") == pilot.render_report(data)


def test_fake_clients_later_lines_and_fake_judge_contract(pilot, command, personas):
    from schemas import ConversationRecord, Turn

    person = personas[2]
    bundle = command.fake_clients(person, 2, "notes_on", clients(pilot).path)
    for expected in ("Yes, that is part of it. " + " ".join(
            m.replace("{user_id}", "grief-on") for m in person.sessions[1].fake_markers), "I suppose so."):
        assert bundle.user.complete("system", [], max_tokens=120) == expected
    judge = command.FakeJudge()
    assert judge.version == "fake-judge"
    for first, expected in ((REMEMBER + " Hello", 4), ("Hello", 3)):
        record = ConversationRecord(
            id="acceptance-record", written_by="live", source="live", created_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            policy_version="acceptance", turns=[Turn(speaker="user", text="Hi"), Turn(speaker="helper", text=first),
                                                Turn(speaker="helper", text="Another reply")])
        result = judge.score(record)
        assert result.usage == []
        score = result.score
        assert score.id == f"score-{record.id}" and score.conversation_id == record.id
        assert score.judge_version == "fake-judge" and score.written_by == "judge"
        assert score.created_at == record.created_at and score.strategies == ["Question", "Question"]
        assert score.empathy == score.relevance == expected
        assert (score.emotion_before, score.emotion_after, score.note) == (4, 3, "scripted")


# 6. Persona validation and the supplied fixture.
def test_persona_fixture_has_contract_people_sessions_and_scenarios(pilot, personas):
    from schemas import Scenario
    from puddles.inference.note_store import check_user_id

    assert [p.persona_id for p in personas] == ["laidoff", "examprep", "grief"]
    ids = []
    for person in personas:
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,11}", person.persona_id)
        assert check_user_id(person.persona_id + "-off") == person.persona_id + "-off"
        assert [s.session for s in person.sessions] == [1, 2, 3]
        for s in person.sessions:
            assert s.opener.strip() and isinstance(s.scenario, Scenario) and s.scenario.opener is None
            ids.append(s.scenario.scenario_id)
    assert len(set(ids)) == 9
    grief = personas[2]
    assert "forget" in grief.sessions[1].opener and "brother" in grief.sessions[1].opener
    assert "[forget:note-{user_id}-002]" in grief.sessions[1].fake_markers
    assert pilot.ARMS == ARMS and pilot.SESSIONS_PER_PERSONA == 3
    assert pilot.DEFAULT_TURNS == 6 and pilot.SESSION_GAP_DAYS == 7
    assert pilot.POLICY_SENTENCE == POLICY_SENTENCE and pilot.LIMITS_SENTENCE == LIMITS_SENTENCE
    assert pilot.NOT_MEASURED == "not measured" and pilot.FAILED == "failed"


@pytest.mark.parametrize("case", ["duplicate-persona", "long-id", "capital-id", "leading-hyphen", "slash-id", "empty-id",
                                  "two-sessions", "four-sessions", "out-of-order", "duplicate-session", "empty-opener",
                                  "whitespace-opener", "invalid-scenario", "duplicate-scenario", "extra-persona-field",
                                  "extra-session-field", "empty-personas", "not-object", "missing-personas", "not-list"])
def test_load_personas_refuses_invalid_contract_input(pilot, tmp_path, case):
    data = json.loads(PERSONAS.read_text(encoding="utf-8"))
    person = data["personas"][0]
    diagnostic = None
    if case == "duplicate-persona":
        data["personas"][1]["persona_id"] = "laidoff"
        diagnostic = "laidoff"
    elif case in ("long-id", "capital-id", "leading-hyphen", "slash-id", "empty-id"):
        person["persona_id"] = {"long-id": "a" * 13, "capital-id": "Laidoff", "leading-hyphen": "-bad",
                                "slash-id": "a/b", "empty-id": ""}[case]
    elif case in ("two-sessions", "four-sessions"):
        person["sessions"] = person["sessions"][:2] if case == "two-sessions" else person["sessions"] + [deepcopy(person["sessions"][2])]
        diagnostic = "laidoff"
    elif case == "out-of-order":
        person["sessions"][0], person["sessions"][1] = person["sessions"][1], person["sessions"][0]
        diagnostic = "laidoff"
    elif case == "duplicate-session":
        person["sessions"][1]["session"] = 1
        diagnostic = "laidoff"
    elif case in ("empty-opener", "whitespace-opener"):
        person["sessions"][0]["opener"] = "" if case == "empty-opener" else " \n\t "
        diagnostic = "laidoff"
    elif case == "invalid-scenario":
        person["sessions"][0]["scenario"]["initial_intensity"] = 6
        diagnostic = "laidoff"
    elif case == "duplicate-scenario":
        data["personas"][1]["sessions"][0]["scenario"]["scenario_id"] = "laidoff-s1"
    elif case == "extra-persona-field":
        person["unexpected"] = True
    elif case == "extra-session-field":
        person["sessions"][0]["unexpected"] = True
    elif case == "empty-personas":
        data = {"personas": []}
    elif case == "not-object":
        data = []
    elif case == "missing-personas":
        data = {}
    elif case == "not-list":
        data = {"personas": {}}
    path = tmp_path / "personas.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        pilot.load_personas(path)
    # Top-level errors have no persona to name. For identified personas, require context.
    if diagnostic:
        message = str(caught.value)
        assert diagnostic in message or (case != "duplicate-persona" and "session" in message.lower())


def test_loader_preserves_file_order_and_utf8(pilot, tmp_path):
    data = json.loads(PERSONAS.read_text(encoding="utf-8"))
    data["personas"].reverse()
    data["personas"][0]["description"] = "Grieving, with café memories"
    path = tmp_path / "personas.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    people = pilot.load_personas(path)
    assert [p.persona_id for p in people] == ["grief", "examprep", "laidoff"]
    assert people[0].description == "Grieving, with café memories"


def test_arm_ids_and_session_dates(pilot):
    assert pilot.arm_user_id("x", "notes_on") == "x-on"
    assert pilot.arm_user_id("x", "notes_off") == "x-off"
    with pytest.raises(ValueError):
        pilot.arm_user_id("x", "other")
    assert [pilot.session_day(START, n) for n in (1, 2, 3)] == [START, START + timedelta(days=7), START + timedelta(days=14)]
