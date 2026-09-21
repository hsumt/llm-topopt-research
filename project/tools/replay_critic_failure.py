"""Replay a saved formulation-critic failure without making any API call.

Usage:
    python -m project.tools.replay_critic_failure artifacts/debug/critic_last_failure.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from project.formulation.models import CriticResult
from project.llm.formulation_critic import _extract_json, _normalize_critic_data


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m project.tools.replay_critic_failure <critic_last_failure.json>"
        )

    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))

    raw_text = str(data.get("raw_text", ""))
    extracted = data.get("extracted_data")
    if not isinstance(extracted, dict):
        extracted = _extract_json(raw_text)

    normalized, report = _normalize_critic_data(extracted)
    result = CriticResult.model_validate(normalized)

    print("PASS: saved critic response now validates offline")
    print()
    print("Critic normalization:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print()
    print(f"Status: {result.status}")
    print(f"Concerns: {len(result.concerns)}")
    print(f"Clarification questions: {len(result.clarification_packet.questions)}")
    print()
    print("No API call was made.")


if __name__ == "__main__":
    main()
