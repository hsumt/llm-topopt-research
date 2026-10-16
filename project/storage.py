"""Small shared helpers for the JSON records under artifacts/: timestamps, atomic writes, digests."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(data) -> str:
    """One JSON text per value: sorted keys, no whitespace, no NaN."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data) -> str:
    """SHA-256 of the canonical JSON text; used for plan and approval hashes."""
    return hashlib.sha256(canonical(data).encode()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    """Write through a temporary file and rename, so readers never see a partial record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
