"""Export formulation-session traces into model-agnostic JSONL training data.

This does NOT fine-tune a model. It creates auditable supervised examples from
human-confirmed dialogue traces so a later open-weight or supported fine-tuning
backend can be trained without changing the runtime architecture.

Run:
    /dolfinx-env/bin/python -m project.tools.export_formulation_training
"""

from __future__ import annotations

import json
from pathlib import Path

from project.paths import FORMULATION_SESSIONS_ROOT, FORMULATION_TRAINING_ROOT


def _iter_events(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def export_training_data() -> dict:
    critic_examples: list[dict] = []
    resolver_examples: list[dict] = []
    session_count = 0

    if not FORMULATION_SESSIONS_ROOT.exists():
        FORMULATION_SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

    for session_dir in sorted(FORMULATION_SESSIONS_ROOT.iterdir()):
        if not session_dir.is_dir():
            continue
        session_path = session_dir / "session.json"
        dialogue_path = session_dir / "dialogue.jsonl"
        if not session_path.exists() or not dialogue_path.exists():
            continue
        session = json.loads(session_path.read_text(encoding="utf-8"))
        if session.get("status") != "confirmed_for_execution":
            # Only human-confirmed sessions become positive training material.
            continue
        session_count += 1
        original_prompt = session.get("original_prompt")

        previous_spec = None
        for event in _iter_events(dialogue_path):
            if event.get("type") == "critic_result":
                result = dict(event.get("result", {}))
                target = {
                    "status": result.get("status"),
                    "review_recommended": result.get("review_recommended"),
                    "summary": result.get("summary"),
                    "issues": result.get("issues", []),
                }
                critic_examples.append(
                    {
                        "session_id": session.get("session_id"),
                        "task": "formulation_critique",
                        "input": {
                            "original_prompt": original_prompt,
                            "spec": event.get("spec"),
                        },
                        "target": target,
                    }
                )
                previous_spec = event.get("spec")

            elif event.get("type") == "resolver_result":
                resolver_examples.append(
                    {
                        "session_id": session.get("session_id"),
                        "task": "formulation_resolution",
                        "input": {
                            "spec": previous_spec,
                            "issue": event.get("issue"),
                            "user_answer": event.get("user_answer"),
                        },
                        "target": event.get("resolution"),
                    }
                )

    critic_path = FORMULATION_TRAINING_ROOT / "critic_examples.jsonl"
    resolver_path = FORMULATION_TRAINING_ROOT / "resolver_examples.jsonl"
    _write_jsonl(critic_path, critic_examples)
    _write_jsonl(resolver_path, resolver_examples)

    summary = {
        "confirmed_sessions": session_count,
        "critic_examples": len(critic_examples),
        "resolver_examples": len(resolver_examples),
        "critic_output": str(critic_path),
        "resolver_output": str(resolver_path),
        "note": (
            "These are training-data exports, not a model-training run. "
            "Weight training should be added only for a backend that supports it."
        ),
    }
    (FORMULATION_TRAINING_ROOT / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = export_training_data()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
