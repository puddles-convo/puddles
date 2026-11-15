"""Stage 6: the overnight loop — draws, caps, restart safety, the judge adapter. All offline."""

import json
from pathlib import Path

import mlflow
import pytest

from loop.errors import ConfigError
from loop.llm import FakeClient
from loop.run_batch import _fake_helper, _fake_user
from loop.runner import load_scenarios
from puddles import tracing
from puddles.agents.critic import select_for_critic
from puddles.config import LoopConfig, load_loop_config
from puddles.decision_log import DecisionLog, DecisionLogError
from puddles.eval.judge import JudgeResult
from puddles.graph import RoundDeps, run_round
from puddles.loop import (
    LoopError,
    agent_failure_streaks,
    check_id_files,
    draw_scenarios,
    judge_models,
    repeated_failure,
    run_loop,
    scoring_judge,
    worst_round_cost,
)
from puddles.memory import FakeMemory
from puddles.records import sha256_text
from puddles.round import stand_in_gate
from puddles.usage import Meter, Prices, read_usage, spend_by_round
from schemas import (
    ConversationRecord,
    CriticOutput,
    LogPolicy,
    ProposerOutput,
    ResearcherOutput,
    RoundEvent,
    Scenario,
    ScoredConversation,
    Turn,
)
from scripts.run_loop import _FakeJudge
from scripts.run_loop import main as run_loop_main
from tests.fakes import FakeSearch, ScriptedLLM

REPO = Path(__file__).parents[1]
POLICY = REPO / "policies" / "policy-v1.md"
SCENARIOS = load_scenarios(REPO / "fixtures" / "scenarios.jsonl")
URL = "https://counselling.example.ac.uk/skills"
QUERY = "counselling skills exploring before giving advice"


@pytest.fixture
def traced(tmp_path):
    tracing.setup(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.flush_trace_async_logging()


# ---------- fixtures ----------


def write_ids(path: Path, ids: list[str]) -> Path:
    path.write_text("".join(f"{i}\n" for i in ids))
    return path


def usage_line(event: str, round_: int, **fields) -> dict:
    return {"event": event, "round": round_, **fields}


def fake_round_event(round_: int, policy_text: str, outcome: str = "no_pattern") -> RoundEvent:
    """A round event replay.check accepts: the starting policy stays in force."""
    return RoundEvent(
        round=round_,
        outcome=outcome,
        reason="test",
        policy=LogPolicy(base="policy-v1", base_sha256=sha256_text(policy_text), in_force="policy-v1"),
    )


def append_event(root: Path, round_: int, policy_text: str) -> RoundEvent:
    event = fake_round_event(round_, policy_text)
    DecisionLog(root).append(event)
    return event


def make_run_one(root: Path, policy_text: str, usage: list[dict] | None = None):
    """A run_one that appends a consistent no_pattern event and optional usage lines."""

    def run_one(policy_path: Path, drawn: list[Scenario], round_: int) -> RoundEvent:
        event = append_event(root, round_, policy_text)
        if usage:
            path = root / "log" / "usage.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                for line in usage:
                    f.write(json.dumps(line) + "\n")
        return event

    return run_one


def loop_kwargs(root: Path, **overrides):
    kwargs = dict(
        log=DecisionLog(root),
        memory=FakeMemory(),
        root=root,
        start_policy=POLICY,
        scenarios=SCENARIOS[:4],
        out=lambda s: None,
    )
    kwargs.update(overrides)
    return kwargs


# ---------- draw_scenarios ----------


def test_same_seed_and_round_draw_the_same_scenarios():
    a = draw_scenarios(SCENARIOS, 3, seed=7, round_=4)
    b = draw_scenarios(SCENARIOS, 3, seed=7, round_=4)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]


def test_a_different_round_draws_a_different_set():
    a = draw_scenarios(SCENARIOS, 3, seed=7, round_=1)
    b = draw_scenarios(SCENARIOS, 3, seed=7, round_=2)
    assert {s.scenario_id for s in a} != {s.scenario_id for s in b}


def test_draw_takes_everything_when_fewer_than_n():
    drawn = draw_scenarios(SCENARIOS[:2], 5, seed=1, round_=1)
    assert len(drawn) == 2


def test_draw_refuses_n_below_one_and_an_empty_pool():
    with pytest.raises(LoopError):
        draw_scenarios(SCENARIOS, 0, seed=1, round_=1)
    with pytest.raises(LoopError):
        draw_scenarios([], 1, seed=1, round_=1)


# ---------- check_id_files ----------


def test_id_files_must_not_overlap(tmp_path):
    validation = write_ids(tmp_path / "v.txt", ["a", "b"])
    train = write_ids(tmp_path / "t.txt", ["c", "d"])
    v, t = check_id_files(validation, train)
    assert v == ["a", "b"] and t == ["c", "d"]
    write_ids(train, ["b", "c"])
    with pytest.raises(LoopError, match="share 1 IDs"):
        check_id_files(validation, train)


# ---------- spend and failure accounting ----------


def test_spend_by_round_groups_done_and_orphaned_sent():
    lines = [
        usage_line("done", 1, call_id="a", cost_usd=1.0),
        usage_line("sent", 1, call_id="b", max_cost_usd=4.0),  # no done: billed at estimate
        usage_line("sent", 2, call_id="c", max_cost_usd=2.0),
        usage_line("done", 2, call_id="c", cost_usd=1.5),  # done wins over its sent
        usage_line("failed", 3, agent="critic"),
    ]
    per_round = spend_by_round(lines)
    assert per_round == {1: 5.0, 2: 1.5}
    assert worst_round_cost(lines) == 5.0
    assert sum(per_round.values()) == 6.5


def test_failure_streaks_count_rounds_per_agent():
    lines = [
        usage_line("failed", 1, agent="critic"),
        usage_line("failed", 1, agent="critic"),  # two lines, one round
        usage_line("failed", 2, agent="critic"),
        usage_line("failed", 3, agent="critic"),
        usage_line("failed", 4, agent="researcher"),
        usage_line("failed", 5, agent="researcher"),
    ]
    assert agent_failure_streaks(lines) == {"critic": 3, "researcher": 2}
    assert repeated_failure(lines) == "critic"


def test_two_failures_a_clean_round_and_a_third_do_not_stop():
    lines = [usage_line("failed", r, agent="critic") for r in (1, 2)]
    lines.append(usage_line("done", 3, call_id="x", cost_usd=0.1))  # round 3: clean
    lines.append(usage_line("failed", 4, agent="critic"))
    assert repeated_failure(lines) is None


# ---------- run_loop driver ----------


def test_fake_loop_stops_at_max_rounds(tmp_path):
    cfg = LoopConfig(max_rounds=3, conversations_per_round=2)
    text = POLICY.read_text()
    summary = run_loop(cfg, run_one=make_run_one(tmp_path, text), **loop_kwargs(tmp_path))
    assert [r for r, _ in summary.rounds] == [1, 2, 3]
    assert summary.stop_reason == "max_rounds (3)"
    assert len(DecisionLog(tmp_path).rounds()) == 3


def test_max_rounds_counts_rounds_already_in_the_log(tmp_path):
    """A restarted loop keeps the same budget: 2 logged rounds + max 3 runs exactly 1."""
    text = POLICY.read_text()
    log = DecisionLog(tmp_path)
    for n in (1, 2):
        append_event(tmp_path, n, text)
    cfg = LoopConfig(max_rounds=3, conversations_per_round=1)
    summary = run_loop(cfg, run_one=make_run_one(tmp_path, text), **loop_kwargs(tmp_path, log=log))
    assert [r for r, _ in summary.rounds] == [3]


def test_spend_cap_stops_before_the_round_that_would_pass_it(tmp_path):
    usage = tmp_path / "log" / "usage.jsonl"
    usage.parent.mkdir(parents=True)
    usage.write_text(json.dumps(usage_line("done", 0, call_id="a", cost_usd=8.0)) + "\n")
    cfg = LoopConfig(max_rounds=10, spend_limit_usd=10.0, conversations_per_round=1)
    ran = []

    def run_one(policy_path, drawn, round_):
        ran.append(round_)
        return append_event(tmp_path, round_, POLICY.read_text())

    summary = run_loop(cfg, run_one=run_one, **loop_kwargs(tmp_path))
    # 8 spent + 1.5 * 8 worst round = 20 > 10: no round starts
    assert ran == [] and "spend limit" in summary.stop_reason


def test_orphaned_sent_spend_counts_toward_the_cap(tmp_path):
    """A round interrupted mid-call leaves a sent line with no done; it still counts."""
    usage = tmp_path / "log" / "usage.jsonl"
    usage.parent.mkdir(parents=True)
    usage.write_text(json.dumps(usage_line("sent", 1, call_id="lost", max_cost_usd=9.0)) + "\n")
    cfg = LoopConfig(max_rounds=10, spend_limit_usd=10.0, conversations_per_round=1)
    summary = run_loop(cfg, run_one=lambda *a: None, **loop_kwargs(tmp_path))
    assert "spend limit" in summary.stop_reason


def test_three_rounds_failing_on_the_same_agent_stop_the_loop(tmp_path):
    """failed lines carry the round they happened in; the streak needs three different rounds."""
    cfg = LoopConfig(max_rounds=10, conversations_per_round=1)
    text = POLICY.read_text()
    calls = []

    def run_one(policy_path, drawn, round_):
        calls.append(round_)
        path = tmp_path / "log" / "usage.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(usage_line("failed", round_, agent="critic")) + "\n")
        return append_event(tmp_path, round_, text)

    summary = run_loop(cfg, run_one=run_one, **loop_kwargs(tmp_path))
    assert calls == [1, 2, 3]
    assert summary.stop_reason == "critic failed in 3 rounds in a row"


def test_a_truncated_log_line_is_reported_not_read(tmp_path):
    log_path = tmp_path / "log" / "decisions.jsonl"
    log_path.parent.mkdir(parents=True)
    good = fake_round_event(1, POLICY.read_text()).model_dump_json()
    log_path.write_text(good + "\n" + '{"event": "round", "round": 2, "pol')
    cfg = LoopConfig(max_rounds=5, conversations_per_round=1)
    with pytest.raises(DecisionLogError, match="cut short"):
        run_loop(cfg, run_one=lambda *a: None, **loop_kwargs(tmp_path))


def test_fresh_root_starts_from_the_starting_policy(tmp_path):
    seen = []
    cfg = LoopConfig(max_rounds=1, conversations_per_round=1)

    def run_one(policy_path, drawn, round_):
        seen.append(policy_path)
        return append_event(tmp_path, round_, POLICY.read_text())

    run_loop(cfg, run_one=run_one, **loop_kwargs(tmp_path))
    assert seen == [POLICY]


# ---------- select_for_critic ----------


def scored(cid, empathy, relevance):
    return ScoredConversation(conversation_id=cid, empathy=empathy, relevance=relevance, note="n", turns=[])


def test_select_for_critic_takes_the_lowest_n_with_id_tiebreak():
    convs = [scored("c", 4, 4), scored("a", 2, 2), scored("b", 2, 3), scored("d", 1, 5)]
    picked = select_for_critic(convs, 2)
    # averages: a=2, b=2.5, d=3, c=4
    assert [c.conversation_id for c in picked] == ["a", "b"]
    assert [c.conversation_id for c in select_for_critic(convs, 10)] == ["a", "b", "d", "c"]
    with pytest.raises(ValueError):
        select_for_critic(convs, 0)


# ---------- scoring_judge ----------


class _Judge:
    version = "j-v1"

    def __init__(self, fail=False):
        self.fail = fail

    def score(self, record) -> JudgeResult:
        if self.fail:
            raise ValueError("no helper turns")
        return _FakeJudge().score(record)


def _record():
    return ConversationRecord(
        id="conv-x",
        written_by="test",
        created_at="2026-01-01T00:00:00Z",
        source="simulated",
        scenario_id="s1",
        split="train",
        policy_version="policy-v1",
        turns=[Turn(speaker="user", text="hi"), Turn(speaker="helper", text="hello")],
    )


def test_scoring_judge_scores_writes_and_logs_usage(tmp_path):
    meter = Meter(tmp_path, Prices())
    meter.start_round(3)
    judge = scoring_judge(_Judge(), meter, tmp_path)
    out = judge([_record()], 3)
    assert len(out) == 1 and out[0].conversation_id == "conv-x"
    scores = (tmp_path / "scores" / "round-03-train.jsonl").read_text().splitlines()
    assert len(scores) == 1 and json.loads(scores[0])["judge_version"] == "judge-fake-v0"
    done = [line for line in read_usage(tmp_path / "log" / "usage.jsonl") if line["agent"] == "judge"]
    assert len(done) == 1 and done[0]["model"] == "fake"


def test_scoring_judge_skips_failures_and_flags_a_dead_judge(tmp_path):
    meter = Meter(tmp_path, Prices())
    meter.start_round(1)
    judge = scoring_judge(_Judge(fail=True), meter, tmp_path)
    assert judge([_record()], 1) == []
    failed = [line for line in read_usage(tmp_path / "log" / "usage.jsonl") if line["event"] == "failed"]
    assert failed and failed[0]["agent"] == "judge"


def test_judge_models_names_every_priced_model():
    class Panel:
        config = type("C", (), {"judges": ["j1", "j2", "j3"], "astra": "astra-1"})()

    class Fallback:
        config = type("C", (), {"model": "fb-1"})()

    class Bare:
        pass

    assert judge_models(Panel()) == ["j1", "j2", "j3", "astra-1"]
    assert judge_models(Fallback()) == ["fb-1"]
    assert judge_models(Bare()) == []


# ---------- load_loop_config ----------


def test_loop_config_reads_the_loop_table(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("""
[loop]
max_rounds = 4
conversations_per_round = 2
spend_limit_usd = 12.5
noise = 0.2
noise_measured = true
current_scores = { empathy = 3.5, relevance = 3.2 }
judge_version = "panel-x"
""")
    cfg = load_loop_config(path)
    assert cfg.max_rounds == 4 and cfg.conversations_per_round == 2
    assert cfg.spend_limit_usd == 12.5 and cfg.noise == 0.2 and cfg.noise_measured
    assert cfg.current_scores.empathy == 3.5 and cfg.judge_version == "panel-x"


def test_loop_config_refuses_a_bad_value(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("[loop]\nmax_rounds = 'lots'\n")
    with pytest.raises(ConfigError):
        load_loop_config(path)
    cfg = LoopConfig()
    assert cfg.max_rounds is None and cfg.conversations_per_round == 5 and cfg.seed == 7


# ---------- graph plumbing ----------


def test_a_kept_round_writes_the_promoted_policy_and_its_meta(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(
        CriticOutput,
        {
            "found": True,
            "name": "p",
            "description": "d",
            "seen_in": 1,
            "out_of": len(SCENARIOS),
            "example_conversation_ids": ["conv-r01-fixture-0001"],
        },
    )
    llm.add(ResearcherOutput, {"queries": [QUERY], "sources": [{"url": URL, "title": "t", "technique": "x"}]})
    llm.add(ProposerOutput, {"action": "add", "rule_text": "Ask first.", "summary": "s"})
    deps = RoundDeps(
        helper_client=FakeClient(_fake_helper),
        user_client=FakeClient(_fake_user),
        llm=llm,
        search=FakeSearch({QUERY: [{"url": URL, "title": "t", "text": "x"}]}),
        memory=FakeMemory(),
        log=DecisionLog(tmp_path),
        root=tmp_path,
        gate=lambda r: stand_in_gate(r, "kept"),
        start_policy=POLICY,
    )
    run_round(deps, policy_path=POLICY, scenarios=SCENARIOS, max_turns=4, seed=1)
    promoted = [p for p in (tmp_path / "policies").glob("policy-v*.md") if "candidate" not in p.name]
    assert len(promoted) == 1
    meta = promoted[0].with_suffix(".meta.json")
    assert meta.exists() and json.loads(meta.read_text())["policy_version"] == promoted[0].stem


def test_critic_conversations_limits_what_the_critic_reads(tmp_path, traced):
    llm = ScriptedLLM()
    llm.add(CriticOutput, {"found": False, "reason": "fine"})
    deps = RoundDeps(
        helper_client=FakeClient(_fake_helper),
        user_client=FakeClient(_fake_user),
        llm=llm,
        search=FakeSearch({}),
        memory=FakeMemory(),
        log=DecisionLog(tmp_path),
        root=tmp_path,
        critic_conversations=2,
    )
    run_round(deps, policy_path=POLICY, scenarios=SCENARIOS[:4], max_turns=4, seed=1)
    prompt = llm.prompts[CriticOutput][0]
    assert prompt.count("<conversation id=") == 2


# ---------- the script ----------


def test_gate_real_is_refused_on_unmeasured_noise(tmp_path, capsys):
    # docs/specs/unattended-run.md 1.4 replaces "not built yet": the repo's config has noise_measured = false
    assert run_loop_main(["--root", str(tmp_path), "--gate", "real"]) == 2
    assert "noise_measured" in capsys.readouterr().err


def test_check_passes_on_a_fresh_fake_root(tmp_path, capsys):
    assert run_loop_main(["--root", str(tmp_path), "--fake", "--check"]) == 0
    assert "check ok" in capsys.readouterr().out


def test_fake_loop_runs_two_rounds_end_to_end(tmp_path, capsys):
    code = run_loop_main(
        ["--root", str(tmp_path), "--fake", "--max-rounds", "2", "--mlflow", f"sqlite:///{tmp_path}/mlflow.db"]
    )
    assert code == 0
    rounds = [e for e in DecisionLog(tmp_path).events() if e.event == "round"]
    assert [r.round for r in rounds] == [1, 2]
    assert "stopped: max_rounds (2)" in capsys.readouterr().out
