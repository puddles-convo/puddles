"""Record files under records/: every record is a file first, and the log points to it by hash."""
import hashlib
from pathlib import Path
from typing import Optional, TypeVar

from schemas import Record, RecordRef

T = TypeVar("T", bound=Record)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def record_path(round_: int, record_id: str) -> str:
    return f"records/round-{round_:02d}/{record_id}.json"


def save_record(record: Record, round_: int, root: Path) -> RecordRef:
    rel = record_path(round_, record.id)
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (record.model_dump_json(indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return RecordRef(id=record.id, path=rel, sha256=sha256_bytes(data))


def load_record(ref: RecordRef, model: type[T], root: Path) -> Optional[T]:
    """Load a record file, or None if it is missing or its hash does not match the log."""
    path = root / ref.path
    if not path.exists():
        return None
    data = path.read_bytes()
    if sha256_bytes(data) != ref.sha256:
        return None
    return model.model_validate_json(data)
