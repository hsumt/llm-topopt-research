"""Persistent audit log for interactive formulation-review sessions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from project.paths import FORMULATION_SESSIONS_ROOT


def _slug(text: str, max_len: int = 36) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return (text or "problem")[:max_len]


def create_session(original_prompt: str) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{timestamp}_{_slug(original_prompt)}_{uuid4().hex[:8]}"
    path = FORMULATION_SESSIONS_ROOT / session_id
    path.mkdir(parents=True, exist_ok=False)
    return session_id, path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_event(session_dir: Path, event: dict) -> None:
    record = dict(event)
    record.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    with (session_dir / "dialogue.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def initialize_session(
    *,
    session_id: str,
    session_dir: Path,
    original_prompt: str,
    initial_spec: dict,
    initial_field_provenance: list[dict],
) -> None:
    write_json(session_dir / "initial_spec.json", initial_spec)
    write_json(
        session_dir / "initial_field_provenance.json",
        initial_field_provenance,
    )
    write_json(
        session_dir / "session.json",
        {
            "schema_version": 1,
            "session_id": session_id,
            "status": "in_progress",
            "original_prompt": original_prompt,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dialogue_file": "dialogue.jsonl",
            "initial_spec_file": "initial_spec.json",
        },
    )


def finalize_session(
    *,
    session_id: str,
    session_dir: Path,
    status: str,
    final_spec: dict,
    final_field_provenance: list[dict],
    summary: dict,
    preview_path: Path | None = None,
) -> None:
    write_json(session_dir / "final_spec.json", final_spec)
    write_json(
        session_dir / "final_field_provenance.json",
        final_field_provenance,
    )
    existing = {}
    session_path = session_dir / "session.json"
    if session_path.exists():
        try:
            existing = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    payload = {
        **existing,
        "schema_version": 1,
        "session_id": session_id,
        "status": status,
        **summary,
        "final_spec_file": "final_spec.json",
        "final_field_provenance_file": "final_field_provenance.json",
        "dialogue_file": "dialogue.jsonl",
    }
    if preview_path is not None:
        try:
            payload["intent_preview_file"] = str(preview_path.relative_to(session_dir))
        except ValueError:
            payload["intent_preview_file"] = str(preview_path)
    write_json(session_dir / "session.json", payload)
