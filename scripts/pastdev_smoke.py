"""Task B1 check: write one test record to past.dev and read it back.

    PAST_API_KEY=past_sk_... python -m scripts.pastdev_smoke
"""
import sys
import time
from datetime import datetime, timezone

from puddles.memory import TRAINING_AUDIENCE, PastClient
from schemas import MemoryEntry


def smoke() -> bool:
    client = PastClient()
    client.ensure_training_audience()
    marker = f"smoke-{int(time.time())}"
    entry = MemoryEntry(
        id=marker, label="critic: smoke test", identity="critic", audience=TRAINING_AUDIENCE,
        timestamp=datetime.now(timezone.utc),
        content=f"Smoke test record {marker}: the puddles memory wrapper can write and read back.",
        metadata={"type": "smoke_test"})
    ingestion = client.ingest(entry)
    print(f"sent {marker}, ingestion {ingestion}; waiting for it to complete")
    if not client.wait(ingestion, timeout_s=120):
        print("still processing after 120 s")
        return False
    results = client.recall(f"smoke test record {marker}", identity="critic")
    found = any(marker in r.get("content", "") or
                any(s.get("sourceId") == marker for s in r.get("sources", [])) for r in results)
    print("read back" if found else f"not found in {len(results)} results")
    return found


if __name__ == "__main__":
    sys.exit(0 if smoke() else 1)
