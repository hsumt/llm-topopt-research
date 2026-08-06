"""
Written 7/23/26 
Validates the audit trail for the fields. Checks that no field is missing, no duplicate fields exist, and that everything is correct. It's an extra reviewer.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from .schema import DefaultedField, FieldProvenance, ProblemSpec

_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _tokenize_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    for name, index in _PATH_TOKEN.findall(path):
        tokens.append(int(index) if index else name)
    if not tokens:
        raise ValueError(f"Invalid field path: {path}")
    return tokens


def get_path(data: dict, path: str):
    node: Any = data
    for token in _tokenize_path(path):
        node = node[token]
    return node


def canonical_field_paths(spec: ProblemSpec) -> list[str]:
    paths = [
        "name",
        "analysis.formulation",
        "analysis.unit_system",
        "analysis.thickness",
        "analysis.edge_traction_definition",
        "mesh.nx",
        "mesh.ny",
        "mesh.Lx",
        "mesh.Ly",
        "material.E",
        "material.nu",
    ]
    for i, _ in enumerate(spec.loads):
        paths.extend(
            [
                f"loads[{i}].location",
                f"loads[{i}].dof",
                f"loads[{i}].value",
                f"loads[{i}].kind",
            ]
        )
    for i, _ in enumerate(spec.bcs):
        paths.extend(
            [
                f"bcs[{i}].location",
                f"bcs[{i}].dof",
                f"bcs[{i}].value",
            ]
        )
    paths.extend(
        [
            "simp.penal",
            "simp.vol_frac",
            "simp.r_min",
            "simp.max_iter",
            "simp.tol_change",
        ]
    )
    return paths


def _values_match(actual: Any, recorded: Any) -> bool:
    if isinstance(actual, bool) or isinstance(recorded, bool):
        return actual is recorded
    if isinstance(actual, (int, float)) and isinstance(recorded, (int, float)):
        return math.isclose(float(actual), float(recorded), rel_tol=1.0e-12, abs_tol=1.0e-12)
    return actual == recorded


def validate_field_provenance(
    spec: ProblemSpec,
    defaulted_fields: List[DefaultedField],
    field_provenance: List[FieldProvenance],
) -> None:
    """Fail closed if semantic provenance is missing, contradictory, or stale."""
    canonical = canonical_field_paths(spec)
    expected = set(canonical)
    by_path: Dict[str, FieldProvenance] = {}
    duplicates: list[str] = []
    for item in field_provenance:
        if item.field_path in by_path:
            duplicates.append(item.field_path)
        by_path[item.field_path] = item
    if duplicates:
        raise ValueError(f"Duplicate field_provenance entries: {sorted(set(duplicates))}")

    missing = sorted(expected - set(by_path))
    extra = sorted(set(by_path) - expected)
    if missing or extra:
        raise ValueError(
            "Field provenance does not exactly cover the runnable specification. "
            f"missing={missing}, extra={extra}"
        )

    payload = spec.model_dump()
    stale = []
    for path in canonical:
        actual = get_path(payload, path)
        if not _values_match(actual, by_path[path].value):
            stale.append(
                f"{path}: spec={actual!r}, provenance={by_path[path].value!r}"
            )
    if stale:
        raise ValueError("Stale field provenance values: " + "; ".join(stale))

    default_paths = {field.field_path for field in defaulted_fields}
    missing_defaults = sorted(default_paths - expected)
    if missing_defaults:
        raise ValueError(
            f"defaulted_fields contains non-canonical paths: {missing_defaults}"
        )
    source_errors = []
    for path in canonical:
        source = by_path[path].source
        if path in default_paths and source != "defaulted":
            source_errors.append(
                f"{path} is listed as defaulted but provenance source is {source}"
            )
        if path not in default_paths and source == "defaulted":
            source_errors.append(
                f"{path} has defaulted provenance but is absent from defaulted_fields"
            )
        if source == "contradictory":
            source_errors.append(
                f"{path} is contradictory: {by_path[path].evidence}"
            )
    if source_errors:
        raise ValueError("Invalid semantic provenance: " + "; ".join(source_errors))


def summarize_semantic_assurance(
    field_provenance: List[FieldProvenance],
    *,
    final_preview_confirmed: bool,
) -> dict:
    grouped: dict[str, list[str]] = {}
    for item in field_provenance:
        grouped.setdefault(item.source, []).append(item.field_path)
    inferred = sorted(
        grouped.get("inferred_from_benchmark_name", [])
        + grouped.get("inferred_from_language", [])
    )
    defaults = sorted(grouped.get("defaulted", []))
    contradictions = sorted(grouped.get("contradictory", []))
    confirmation_required = bool(inferred or defaults)
    if contradictions:
        status = "contradictory"
    elif not confirmation_required:
        status = "fully_explicit"
    elif final_preview_confirmed:
        status = "user_confirmed"
    else:
        status = "unconfirmed_inference_or_defaults"
    return {
        "status": status,
        "confirmation_required": confirmation_required,
        "final_preview_confirmed": bool(final_preview_confirmed),
        "inferred_fields": inferred,
        "defaulted_fields": defaults,
        "contradictory_fields": contradictions,
        "source_counts": {key: len(value) for key, value in sorted(grouped.items())},
    }
