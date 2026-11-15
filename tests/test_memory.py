import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from puddles.memory import TRAINING_AUDIENCE, FakeMemory, PastClient, entry_for, ids_in_results
from schemas import FailurePattern, MemoryEntry, RecordRef

EXAMPLES = Path(__file__).parents[1] / "examples"


def pattern():
    return FailurePattern.model_validate_json((EXAMPLES / "training_H4_failure_pattern.json").read_text())


def test_entry_matches_memory_example():
    example = json.loads((EXAMPLES / "memory_entry.json").read_text())
    p = pattern()
    entry = entry_for(p, RecordRef(id=p.id, path="records/round-03/pattern-r03.json", sha256="h"), p.round)
    assert entry.id == example["id"] and entry.identity == "critic" and entry.audience == TRAINING_AUDIENCE
    assert entry.content.startswith(f"Failure pattern {p.id}, {p.name}, round {p.round}:")
    assert entry.metadata["record_id"] == p.id


def test_recurring_pattern_gets_new_entry_id():
    p = pattern()
    entry = entry_for(p, RecordRef(id=p.id, path="x", sha256="h"), round_=6)
    assert entry.id == "pattern-r03-seen-r06"


def test_fake_memory_keeps_training_memory_from_other_identities():
    mem = FakeMemory()
    mem.ensure_training_audience()
    p = pattern()
    mem.ingest(entry_for(p, RecordRef(id=p.id, path="x", sha256="h"), p.round))
    assert mem.recall("failure pattern advice", identity="critic")
    assert mem.recall("failure pattern advice", identity="simulated-user") == []


def test_ids_in_results_reads_source_ids_and_text():
    results = [{"content": "Edit edit-r05 was reverted", "sources": [{"sourceId": "gate-r05"}]},
               {"content": "Research research-r03 for pattern-r03", "sources": []}]
    assert ids_in_results(results) == ["gate-r05", "edit-r05", "research-r03", "pattern-r03"]


def test_past_client_ingests_polls_and_recalls():
    calls = []
    statuses = iter(["queued", "processing", "completed"])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.content))
        assert request.headers["authorization"] == "Bearer past_sk_test"
        if request.url.path.endswith("/ingest"):
            return httpx.Response(202, json={"ingestionId": "ing_1", "status": "queued"})
        if request.url.path.endswith("/ingest/ing_1"):
            return httpx.Response(200, json={"status": next(statuses)})
        if request.url.path.endswith("/recall"):
            return httpx.Response(200, json={"results": [{"content": "hello", "sources": []}]})
        return httpx.Response(404)

    client = PastClient(api_key="past_sk_test", transport=httpx.MockTransport(handler))
    entry = MemoryEntry(id="t-1", content="test", label="l", timestamp=datetime.now(timezone.utc),
                        identity="critic", audience=TRAINING_AUDIENCE)
    assert client.ingest(entry) == "ing_1"
    assert json.loads(calls[0][2])["id"] == "t-1"
    assert client.wait("ing_1", timeout_s=5, poll_s=0)
    assert client.recall("q", identity="critic")[0]["content"] == "hello"
    assert json.loads(calls[-1][2]) == {"query": "q", "identity": "critic", "level": "low", "limit": 20}


@pytest.mark.skipif(not os.environ.get("PAST_API_KEY"), reason="needs PAST_API_KEY")
def test_live_write_then_read_back():
    from scripts.pastdev_smoke import smoke

    assert smoke()
