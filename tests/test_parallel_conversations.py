"""PUDDLES_CONVERSATION_WORKERS: a parallel batch gives the same records, in the same order, and the
usage meter still counts every call (it is found through a context variable)."""
import threading
import time

from loop.llm import FakeClient
from loop.policy import load_policy
from loop.runner import load_scenarios, run_batch
from puddles import usage
from puddles.usage import Meter


def _clients():
    seen = set()

    def helper(*a, **k):
        seen.add(threading.get_ident())
        time.sleep(0.01)
        return "That sounds hard. What happened?"

    return FakeClient(helper), FakeClient(lambda *a, **k: "It has been a rough week."), seen


def _batch(monkeypatch, workers):
    monkeypatch.setenv("PUDDLES_CONVERSATION_WORKERS", str(workers))
    helper, user, seen = _clients()
    policy = load_policy("policies/policy-v1.md")
    scenarios = load_scenarios("fixtures/scenarios.jsonl")[:6]
    out = run_batch(helper, user, policy, scenarios, round=1, max_turns=3, seed=7)
    return out, seen


def test_parallel_batch_equals_sequential(monkeypatch):
    seq, seen1 = _batch(monkeypatch, 1)
    par, seen4 = _batch(monkeypatch, 4)
    assert len(seen1) == 1 and len(seen4) > 1
    strip = lambda rs: [(r.id, [(t.speaker, t.text) for t in r.turns]) for r in rs]
    assert strip(seq.records) == strip(par.records)
    assert seq.failures == par.failures == []


def test_meter_counts_every_call_from_worker_threads(tmp_path):
    meter = Meter(tmp_path)

    def call(_):
        with usage.metering(meter, "helper"):
            m, agent = usage.current()
            m.note_done(agent, usage.FAKE_MODEL, 10, 5)

    from concurrent.futures import ThreadPoolExecutor
    import contextvars
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: contextvars.copy_context().run(call, i), range(200)))
    assert meter.usage["helper"].calls == 200
    assert len(usage.read_usage(tmp_path / usage.USAGE_PATH)) == 200


def test_a_meter_set_by_the_caller_sees_calls_made_in_worker_threads(monkeypatch, tmp_path):
    """Regression: the context was copied inside each worker, which is empty, so the usage log lost
    every helper and simulated-person call of a parallel batch."""
    from loop.llm import FakeClient

    monkeypatch.setenv("PUDDLES_CONVERSATION_WORKERS", "4")
    meter = Meter(tmp_path)
    seen = []

    def helper(*a, **k):
        seen.append(usage.current()[0] is meter)
        return "That sounds hard. What happened?"

    policy = load_policy("policies/policy-v1.md")
    scenarios = load_scenarios("fixtures/scenarios.jsonl")[:4]
    with usage.metering(meter, "helper"):
        run_batch(FakeClient(helper), FakeClient(lambda *a, **k: "Rough week."), policy, scenarios,
                  round=1, max_turns=2, seed=7)
    assert seen and all(seen)
