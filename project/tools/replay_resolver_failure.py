"""Replay a saved resolver failure without making an API call."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from project.formulation.patching import apply_operations_to_spec
from project.llm.formulation_resolver import decode_resolution_response
from project.parser.schema import ProblemSpec


def _restore_dumped_none_engineering_values(value):
    """Repair the lossy ``exclude_none=True`` snapshot representation only.

    An EngineeringValue may legitimately have ``value=None`` plus a unit.
    ``model_dump(exclude_none=True)`` removes that value key, so a saved resolver
    payload is not round-trippable without restoring it.  This is diagnostic
    replay logic only; it does not alter live session semantics.
    """
    if isinstance(value, list):
        return [_restore_dumped_none_engineering_values(item) for item in value]
    if isinstance(value, dict):
        out = {k: _restore_dumped_none_engineering_values(v) for k, v in value.items()}
        if set(out) == {"unit"}:
            out["value"] = None
        return out
    return value


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/debug/resolver_last_failure.json")
    if not path.exists():
        raise SystemExit(f"Resolver failure snapshot not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    raw = str(data.get("raw_text", ""))
    proposal, meta = decode_resolution_response(raw)

    print("PASS: saved resolver response decodes offline")
    print()
    print("Decode metadata:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    payload = data.get("payload") or {}
    current_spec = payload.get("current_spec")
    if current_spec is not None:
        spec = ProblemSpec.model_validate(_restore_dumped_none_engineering_values(current_spec))
        patched, normalized, patch_meta = apply_operations_to_spec(spec, proposal)
        print()
        print("PASS: normalized resolver patch applies to ProblemSpec offline")
        print()
        print("Patch normalization:")
        print(json.dumps(patch_meta, indent=2, ensure_ascii=False))
        print()
        print("Normalized operations:")
        print(normalized.model_dump_json(indent=2))
        print()
        print("Resulting ProblemSpec validates.")
        print("Boundary condition:")
        if patched.boundary_conditions:
            print(patched.boundary_conditions[0].model_dump_json(indent=2))
        print("Constraints:")
        if patched.optimization:
            for item in patched.optimization.constraints:
                print(f"- {item.id}: {item.quantity} {item.relation} {item.limit}")
    else:
        print()
        print("No current_spec in snapshot; proposal decode only.")

    print()
    print("No API call was made.")


if __name__ == "__main__":
    main()
