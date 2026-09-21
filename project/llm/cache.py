"""Small local cache for deterministic LLM request replay.

The cache is keyed by model + system prompt + JSON payload + attachment hashes,
so changing prompts/code inputs naturally invalidates stale entries.  It stores
only local request fingerprints and raw model responses under artifacts/cache;
no network or hidden state is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from project.formulation.context import ContextAttachment
from project.paths import ARTIFACT_ROOT


def _enabled() -> bool:
    return os.getenv("LLM_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _attachment_fingerprints(attachments: Iterable[ContextAttachment] | None) -> list[dict[str, Any]]:
    output = []
    for item in attachments or []:
        output.append(
            {
                "name": item.name,
                "media_type": item.media_type,
                "sha256": hashlib.sha256(item.data).hexdigest(),
                "size": len(item.data),
            }
        )
    return output


def request_key(
    *,
    component: str,
    model: str,
    system_prompt: str,
    payload: Any,
    attachments: Iterable[ContextAttachment] | None = None,
) -> str:
    canonical = json.dumps(
        {
            "component": component,
            "model": model,
            "system_prompt": system_prompt,
            "payload": payload,
            "attachments": _attachment_fingerprints(attachments),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _path(component: str, key: str) -> Path:
    root = ARTIFACT_ROOT / "cache" / component
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.json"


def load_cached_response(
    *,
    component: str,
    model: str,
    system_prompt: str,
    payload: Any,
    attachments: Iterable[ContextAttachment] | None = None,
) -> dict[str, Any] | None:
    if not _enabled():
        return None
    key = request_key(
        component=component,
        model=model,
        system_prompt=system_prompt,
        payload=payload,
        attachments=attachments,
    )
    path = _path(component, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cache_key"] = key
        return data
    except Exception:
        return None


def save_cached_response(
    *,
    component: str,
    model: str,
    system_prompt: str,
    payload: Any,
    response_text: str,
    input_tokens: int,
    output_tokens: int,
    stop_reason: str | None,
    attachments: Iterable[ContextAttachment] | None = None,
) -> str | None:
    if not _enabled():
        return None
    try:
        key = request_key(
            component=component,
            model=model,
            system_prompt=system_prompt,
            payload=payload,
            attachments=attachments,
        )
        path = _path(component, key)
        path.write_text(
            json.dumps(
                {
                    "component": component,
                    "model": model,
                    "response_text": response_text,
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "stop_reason": stop_reason,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return key
    except Exception:
        return None
