"""Replay a saved parser failure entirely offline.

Usage from repository root:

    python -m project.tools.replay_parser_failure \
        artifacts/debug/parser_last_failure.json

This performs schema normalization, Pydantic validation, provenance expansion,
and strict provenance validation without calling any LLM API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.parser.normalize import normalize_parser_payload
from project.parser.provenance import normalize_field_provenance, validate_field_provenance
from project.parser.schema import ParserResult, ProblemRoute


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
    route = ProblemRoute.model_validate(payload["route"])
    raw = payload.get("extracted_data")
    if not isinstance(raw, dict):
        raise SystemExit("Snapshot does not contain extracted_data JSON.")

    normalized, normalize_diag = normalize_parser_payload(raw, route)
    result = ParserResult.model_validate(normalized)
    provenance, provenance_diag = normalize_field_provenance(
        result.spec,
        result.field_provenance,
    )
    result = result.model_copy(update={"field_provenance": provenance})
    validate_field_provenance(result.spec, result.field_provenance)

    print("PASS: saved parser response now validates offline")
    print("\nSchema normalization:")
    print(json.dumps(normalize_diag, indent=2, default=str))
    print("\nProvenance normalization:")
    print(json.dumps(provenance_diag, indent=2, default=str))
    print("\nNo API call was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
