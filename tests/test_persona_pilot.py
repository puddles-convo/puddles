"""The persona pilot (docs/specs/persona-pilot.md): offline, with the fakes of scripts.persona_pilot."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.errors import SimulatedUserError
from loop.llm import FakeClient
from loop.prompts import NO_NOTES
from loop.simulated_user import END_MARKER
from puddles import tracing
from puddles.graph import check_policy_file
from puddles.inference import crisis
from puddles.inference import pilot as P
from puddles.inference.note_store import FakeUserMemory, NoteStore
from puddles.inference.notes import NotesResult
from puddles.inference.session_log import record_path
from schemas import ScoreRecord
from scripts import persona_pilot as S
from scripts.chat import FakeNoteTaker

REPO = Path(__file__).resolve().parents[1]
PERSONAS = REPO / "fixtures" / "personas.json"
POLICY = REPO / "policies" / "policy-v2.md"
START = date(2026, 3, 2)


@pytest.fixture(autouse=True)
def _tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    tracing.setup(tracking_uri=uri)


@pytest.fixture
def env(tmp_path):
    path = crisis.load_path()
    return SimpleNamespace(path=path, policy=check_policy_file(POLICY), root=tmp_path / "store",
                           store=NoteStore(FakeUserMemory(), tmp_path / "store"), llm=FakeNoteTaker(),
                           personas={p.persona_id: p for p in P.load_personas(PERSONAS)})


def _run(env, pid, n, arm, *, clients=None, turns=3):
    clients = clients or S.fake_clients(env.personas[pid], n, arm, env.path)
    result = P.run_session(env.personas[pid], n, arm, clients=clients, policy=env.policy, store=env.store,
                           note_llm=env.llm, root=env.root, turns=turns, start=START, fake=True)
    return result, clients


def _write(tmp_path, data) -> Path:
    p = tmp_path / "personas.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _raw():
    return json.loads(PERSONAS.read_text(encoding="utf-8"))


# 1
def test_load_personas_file_and_refusals(tmp_path):
    personas = P.load_personas(PERSONAS)
    assert [p.persona_id for p in personas] == ["laidoff", "examprep", "grief"]
    assert all([s.session for s in p.sessions] == [1, 2, 3] for p in personas)

    def bad(change):
        data = _raw()
        change(data)
        with pytest.raises(ValueError):
            P.load_personas(_write(tmp_path, data))

    bad(lambda d: d["personas"].append(json.loads(json.dumps(d["personas"][0]))))
    bad(lambda d: d["personas"][0].update(persona_id="a" * 13))
    bad(lambda d: d["personas"][0].update(persona_id="Laidoff"))
    bad(lambda d: d["personas"][0].update(sessions=d["personas"][0]["sessions"][:2]))
    bad(lambda d: d["personas"][0].update(sessions=list(reversed(d["personas"][0]["sessions"]))))
    bad(lambda d: d["personas"][0]["sessions"][0].update(opener="   "))
    bad(lambda d: d["personas"][0]["sessions"][0]["scenario"].update(openness="chatty"))
    bad(lambda d: d.update(personas=[]))


# 2
def test_arm_user_id_and_session_day():
    assert P.arm_user_id("x", "notes_on") == "x-on"
    assert P.arm_user_id("x", "notes_off") == "x-off"
    with pytest.raises(ValueError):
        P.arm_user_id("x", "both")
    assert (P.session_day(START, 3) - P.session_day(START, 1)).days == 14


# 3
def test_snapshot_notes(env):
    assert P.snapshot_notes(env.store, "examprep-on") == []
    _run(env, "examprep", 1, "notes_on")
    snap = P.snapshot_notes(env.store, "examprep-on")
    assert [(n["kind"], n["fact"]) for n in snap] == [
        ("situation", "Has a statistics exam in three weeks after failing the mock"),
        ("preferences", "Wants concrete steps rather than reassurance")]
    assert snap[0]["note_id"] == "note-examprep-on-001"
    assert snap[0]["last_confirmed"] == START.isoformat() and snap[0]["due_date"] is None


# 4
def test_note_changes():
    empty = P.note_changes([], [], NotesResult())
    assert (empty.added, empty.replaced, empty.confirmed, empty.deleted) == ([], [], [], [])
    before = [{"note_id": "n-1", "kind": "people", "fact": "old fact", "last_confirmed": "2026-01-01",
               "due_date": None}]
    ch = P.note_changes(before, [], NotesResult(deleted=["n-1"]))
    assert ch.deleted == [{"note_id": "n-1", "fact": "old fact"}]


# 5
def test_notes_off_helper_never_given_notes(env):
    _run(env, "laidoff", 1, "notes_on")
    assert P.snapshot_notes(env.store, "laidoff-on")
    result, clients = _run(env, "laidoff", 2, "notes_off")
    assert result.user_id == "laidoff-off" and result.notes_given == []
    assert clients.helper.calls and all(NO_NOTES in c["system"] for c in clients.helper.calls)
    assert result.record is not None and result.record.remember_me is False
    assert P.snapshot_notes(env.store, "laidoff-off") == []


# 6
def test_notes_on_session_two_sees_session_one(env):
    _run(env, "grief", 1, "notes_on")
    result, clients = _run(env, "grief", 2, "notes_on")
    system = clients.helper.calls[0]["system"]
    assert "Father died two months ago" in system and NO_NOTES not in system
    assert record_path(env.root, "grief-on-s2").exists()
    assert result.record.remember_me is True


# 7
def test_arms_differ_only_in_memory(env, monkeypatch):
    seen = []
    real_turn, real_finish = P.run_turn, P.finish_session

    def spy_turn(session, text):
        seen.append((session.user_id, text, session.remember_me, session.consent_for_training))
        return real_turn(session, text)

    finished = []

    def spy_finish(session, *a, **k):
        finished.append(session.user_id)
        return real_finish(session, *a, **k)

    monkeypatch.setattr(P, "run_turn", spy_turn)
    monkeypatch.setattr(P, "finish_session", spy_finish)
    for arm in P.ARMS:
        clients = S.fake_clients(env.personas["examprep"], 1, "notes_off", env.path)  # same scripted user
        _run(env, "examprep", 1, arm, clients=clients, turns=4)
    on = [s for s in seen if s[0] == "examprep-on"]
    off = [s for s in seen if s[0] == "examprep-off"]
    assert len(on) == len(off) == 4
    assert [s[1] for s in on] == [s[1] for s in off]
    assert on[0][1] == env.personas["examprep"].sessions[0].opener
    assert all(s[2] for s in on) and not any(s[2] for s in off) and all(s[3] for s in seen)
    assert finished == ["examprep-on", "examprep-off"]


# 8
def test_turns_one_and_early_end(env):
    result, clients = _run(env, "laidoff", 1, "notes_off", turns=1)
    assert [t.speaker for t in result.record.turns] == ["user", "helper"] and not clients.user.calls
    ending = S.fake_clients(env.personas["laidoff"], 2, "notes_off", env.path)
    ending.user = FakeClient(lambda system, messages: END_MARKER)
    result, _ = _run(env, "laidoff", 2, "notes_off", clients=ending, turns=6)
    assert not result.failed and len(result.record.turns) == 2
    with pytest.raises(ValueError):
        _run(env, "laidoff", 3, "notes_off", turns=0)


class _CountingJudge(S.FakeJudge):
    def __init__(self):
        self.ids = []

    def score(self, record):
        self.ids.append(record.id)
        return super().score(record)


# 9
def test_failed_session_is_recorded_and_pilot_goes_on(env):
    def make(persona, n, arm):
        c = S.fake_clients(persona, n, arm, env.path)
        if n == 2:
            def boom(system, messages):
                raise SimulatedUserError("scripted failure")
            c.user = FakeClient(boom)
        return c

    judge = _CountingJudge()
    pilot = P.run_pilot([env.personas["grief"]], make_clients=make, policy=env.policy, store=env.store,
                        note_llm=env.llm, judge=judge, root=env.root, turns=3, start=START, fake=True)
    on = pilot["personas"][0]["sessions"]["notes_on"]
    assert on[1]["failed"] and on[1]["record"] is None and "SimulatedUserError" in on[1]["reason"]
    assert on[1]["score"]["status"] == "failed" and on[1]["score"]["empathy"] is None
    assert on[2]["score"]["status"] == "scored"
    assert "grief-on-s2" not in judge.ids and "grief-on-s3" in judge.ids
    report = P.render_report(pilot)
    assert "| 2 | failed | failed | not measured |" in report
    assert "| 2 | 0 |" not in report


# 10
def test_score_session_unscored(env):
    result, _ = _run(env, "laidoff", 1, "notes_off", turns=1)

    class Raising:
        version = "x"

        def score(self, record):
            raise RuntimeError("judge down")

    class Wrong(S.FakeJudge):
        def score(self, record):
            out = super().score(record)
            out.score = out.score.model_copy(update={"conversation_id": "someone-else"})
            return out

    s = P.score_session(Raising(), result)
    assert s.status == "unscored" and s.score is None and "judge down" in s.reason
    assert P.score_session(Wrong(), result).status == "unscored"
    assert P.score_session(S.FakeJudge(), result).status == "scored"


def _session(arm, n, status="scored", value=3, notes_after=(), deleted=(), failed=False, replies=("a", "b")):
    scored = status == "scored"
    turns = []
    for r in replies:
        turns += [{"speaker": "user", "text": "u"}, {"speaker": "helper", "text": r}]
    return {"persona_id": "p", "session": n, "arm": arm, "user_id": f"p-{arm}", "session_id": f"s{n}",
            "failed": failed, "reason": "", "notes_given": [],
            "record": None if failed else {"turns": turns},
            "notes_before": [], "notes_after": list(notes_after),
            "changes": {"added": [], "replaced": [], "confirmed": [], "deleted": list(deleted),
                        "failed": False, "reason": ""},
            "score": {"status": status, "reason": "", "empathy": value if scored else None,
                      "relevance": value if scored else None, "emotion_before": None, "emotion_after": None,
                      "judge_version": None, "note": None}}


def _pilot(on, off):
    return {"policy_version": "v", "policy_path": "p.md", "turns": 2, "fake": True, "start": "2026-03-02",
            "judge_version": "j", "policy_sentence": P.POLICY_SENTENCE, "limits_sentence": P.LIMITS_SENTENCE,
            "personas": [{"persona_id": "p", "description": "d", "sessions": {"notes_on": on, "notes_off": off}}]}


# 11
def test_not_measured_never_zero():
    assert P.mean_or_not_measured([]) == "not measured"
    on = [_session("notes_on", n, value=4) for n in (1, 2, 3)]
    off = [_session("notes_off", n, status="failed", failed=True) for n in (1, 2, 3)]
    report = P.render_report(_pilot(on, off))
    assert "| Mean empathy, session 3 | 4.00 | not measured | not measured |" in report
    assert "| Sessions scored | 3 of 3 | 0 of 3 | |" in report
    assert "| Sessions failed | 0 | 3 | |" in report


# 12
def test_fake_command_end_to_end(tmp_path):
    out = tmp_path / "out"
    assert S.main(["--personas", str(PERSONAS), "--policy", str(POLICY), "--out", str(out), "--fake",
                   "--turns", "3"]) == 0
    assert (out / "pilot.json").exists()
    assert len(list((out / "sessions").rglob("s*.json"))) == 18
    report = (out / "report.md").read_text(encoding="utf-8")
    assert report.count(P.POLICY_SENTENCE) == 1 and report.count(P.LIMITS_SENTENCE) == 1
    assert report.count("| added |") >= 3
    brother = "Argued with their brother at the funeral about selling the house"
    assert f"| 2 | people | {brother} | deleted |" in report
    pilot = json.loads((out / "pilot.json").read_text(encoding="utf-8"))
    grief = next(p for p in pilot["personas"] if p["persona_id"] == "grief")["sessions"]["notes_on"]
    for s in grief[1:]:
        assert brother not in [n["fact"] for n in s["notes_after"]]
    assert brother in [n["fact"] for n in grief[0]["notes_after"]]


# 13
def test_out_refusal_and_only(tmp_path):
    base = ["--personas", str(PERSONAS), "--policy", str(POLICY), "--fake", "--turns", "2"]
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("x", encoding="utf-8")
    assert S.main(base + ["--out", str(busy)]) == 2
    assert sorted(p.name for p in busy.iterdir()) == ["keep.txt"]
    assert S.main(base + ["--out", str(tmp_path / "a"), "--only", "nobody"]) == 2
    assert not (tmp_path / "a").exists()
    out = tmp_path / "b"
    assert S.main(base + ["--out", str(out), "--only", "grief"]) == 0
    assert len(list((out / "sessions").rglob("s*.json"))) == 6


# 14
def test_real_mode_missing_config(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HELPER_URL", raising=False)
    out = tmp_path / "real"
    assert S.main(["--personas", str(PERSONAS), "--policy", str(POLICY), "--out", str(out)]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("error:")
    assert not out.exists()


# 15
def test_report_numbers_from_dict_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # nothing on disk to read
    note = {"note_id": "n-1", "kind": "goals", "fact": "a | b", "last_confirmed": "2026-03-02", "due_date": None}
    on = [_session("notes_on", 1, value=4), _session("notes_on", 2, value=5),
          _session("notes_on", 3, value=4, notes_after=[note], replies=("on one",))]
    off = [_session("notes_off", 1, value=3), _session("notes_off", 2, status="unscored"),
           _session("notes_off", 3, value=2)]
    report = P.render_report(_pilot(on, off))
    assert "| 2 | 5 | unscored | not measured | 5 | unscored | not measured |" in report
    assert "| 3 | 4 | 2 | +2 |" in report
    assert "| Mean empathy, session 3 | 4.00 | 2.00 | +2.00 |" in report
    assert "| Mean empathy, all sessions | 4.33 | 2.50 | +1.83 |" in report
    assert "| Sessions scored | 3 of 3 | 2 of 3 | |" in report
    assert "| Notes kept after session 3 | 1 | 0 | |" in report
    assert "| 3 | goals | a \\| b |  |" in report
    assert "| 1 | on one | a |" in report and "| 2 | - | b |" in report


# ---------- repair round 1: --workers (amendment 8) and rounding (amendment 6) ----------

def _fake_main(out, *extra):
    return S.main(["--personas", str(PERSONAS), "--policy", str(POLICY), "--out", str(out), "--fake",
                   "--turns", "3", "--start", START.isoformat(), *extra])


def test_workers_three_gives_the_same_files_as_one(tmp_path, monkeypatch):
    # The records carry `timings_ms`, measured by the live turn code with time.monotonic, which differ
    # between any two runs, even two with --workers 1. The turn module's clock is frozen so that the
    # files can be compared whole; nothing else is set aside.
    from puddles.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "time", SimpleNamespace(monotonic=lambda: 1000.0))
    one, three = tmp_path / "one", tmp_path / "three"
    assert _fake_main(one, "--workers", "1") == 0
    assert _fake_main(three, "--workers", "3") == 0
    assert json.loads((one / "pilot.json").read_text()) == json.loads((three / "pilot.json").read_text())
    assert (one / "pilot.json").read_text() == (three / "pilot.json").read_text()
    assert (one / "report.md").read_text() == (three / "report.md").read_text()
    names = sorted(p.relative_to(one).as_posix() for p in (one / "sessions").rglob("*.json"))
    assert names == sorted(p.relative_to(three).as_posix() for p in (three / "sessions").rglob("*.json"))
    for name in names:
        assert (one / name).read_text() == (three / name).read_text()


def test_a_failing_chain_does_not_stop_the_others(env):
    def make(persona, n, arm):
        if persona.persona_id == "examprep" and arm == "notes_off":
            raise RuntimeError("scripted client failure")
        return S.fake_clients(persona, n, arm, env.path)

    store = NoteStore(S.LockedMemory(FakeUserMemory()), env.root)
    seen = []
    pilot = P.run_pilot(list(env.personas.values()), make_clients=make, policy=env.policy, store=store,
                        note_llm=env.llm, judge=S.FakeJudge(), root=env.root, turns=2, start=START,
                        fake=True, workers=3, on_session=seen.append)
    assert len(seen) == 18
    for p in pilot["personas"]:
        for arm, sessions in p["sessions"].items():
            assert [s["session"] for s in sessions] == [1, 2, 3]
            broken = p["persona_id"] == "examprep" and arm == "notes_off"
            assert all(s["failed"] == broken for s in sessions)
            if broken:
                assert all("scripted client failure" in s["reason"] for s in sessions)
            else:
                assert all(s["score"]["status"] == "scored" for s in sessions)
    with pytest.raises(ValueError):
        P.run_pilot([], make_clients=make, policy=env.policy, store=store, note_llm=env.llm,
                    judge=S.FakeJudge(), root=env.root, start=START, workers=0)


def test_workers_zero_exits_2(tmp_path):
    out = tmp_path / "zero"
    assert _fake_main(out, "--workers", "0") == 2
    assert not out.exists()


def test_difference_of_means_uses_unrounded_means():
    # notes_on 1, 1, 2 (mean 1.333...) and notes_off 1, 2, 2 (mean 1.666...): rounded first, the
    # difference would be 1.33 - 1.67 = -0.34; from the unrounded means it is -0.33.
    on = [_session("notes_on", n, value=v) for n, v in zip((1, 2, 3), (1, 1, 2))]
    off = [_session("notes_off", n, value=v) for n, v in zip((1, 2, 3), (1, 2, 2))]
    report = P.render_report(_pilot(on, off))
    assert "| Mean empathy, all sessions | 1.33 | 1.67 | -0.33 |" in report
    assert P.mean_or_not_measured([1, 1, 2]) == 1.33
    same = P.render_report(_pilot([_session("notes_on", 1, value=3)], [_session("notes_off", 1, value=3)]))
    assert "| 1 | 3 | 3 | 0 |" in same and "| Mean empathy, all sessions | 3.00 | 3.00 | 0.00 |" in same
