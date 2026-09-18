"""Deterministic semantic-provenance checks for the generic ProblemSpec.

The LLM emits *compact provenance scopes* (for example ``/geometry`` or
``/sources/0``), not one duplicated provenance record for every leaf in the
ProblemSpec. Python expands those scopes deterministically to exact leaf-level
records before strict validation.

This preserves a field-level audit trail while avoiding hundreds of redundant
LLM tokens and brittle failures caused by asking the model to enumerate every
leaf twice.
"""

from __future__ import annotations

import math
from typing import Any

from .schema import FieldProvenance, ProblemSpec


_ADMIN_KEYS = {"schema_version", "id"}
_NON_FORMULATION_ROOT_KEYS = {"requested_outputs"}


def _escape_pointer_token(token: str) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def _unescape_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _join_pointer(prefix: str, token: str | int) -> str:
    encoded = _escape_pointer_token(str(token))
    return f"{prefix}/{encoded}" if prefix else f"/{encoded}"


def _normalize_pointer(pointer: str) -> str:
    pointer = pointer.strip()
    if not pointer:
        return ""
    if not pointer.startswith("/"):
        raise ValueError(f"Not a JSON Pointer: {pointer!r}")
    if len(pointer) > 1:
        pointer = pointer.rstrip("/")
    return pointer


def get_pointer(data: Any, pointer: str) -> Any:
    if pointer == "":
        return data
    if not pointer.startswith("/"):
        raise ValueError(f"Not a JSON Pointer: {pointer!r}")

    node = data
    for raw in pointer.split("/")[1:]:
        token = _unescape_pointer_token(raw)
        if isinstance(node, list):
            node = node[int(token)]
        elif isinstance(node, dict):
            node = node[token]
        else:
            raise KeyError(pointer)
    return node


def _leaf_paths(value: Any, prefix: str = "") -> list[str]:
    """Return populated semantic leaves using JSON Pointer syntax.

    Administrative ``id`` fields and ``schema_version`` are intentionally not
    engineering assertions. Empty containers and ``None`` values likewise do
    not require provenance.
    """

    paths: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            if key in _ADMIN_KEYS:
                continue
            if not prefix and key in _NON_FORMULATION_ROOT_KEYS:
                continue
            child_prefix = _join_pointer(prefix, key)
            paths.extend(_leaf_paths(child, child_prefix))
        return paths

    if isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = _join_pointer(prefix, index)
            paths.extend(_leaf_paths(child, child_prefix))
        return paths

    if value is None:
        return paths

    paths.append(prefix)
    return paths


def required_provenance_paths(spec: ProblemSpec) -> list[str]:
    payload = spec.model_dump(exclude_none=True)
    return sorted(_leaf_paths(payload))


def _values_match(actual: Any, recorded: Any) -> bool:
    if isinstance(actual, bool) or isinstance(recorded, bool):
        return actual is recorded
    if isinstance(actual, (int, float)) and isinstance(recorded, (int, float)):
        return math.isclose(
            float(actual),
            float(recorded),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    return actual == recorded


def _scope_matches(scope: str, leaf: str) -> bool:
    """Return True when ``scope`` covers ``leaf``.

    ``/geometry`` covers ``/geometry/description`` and all deeper leaves.
    ``/sources/0`` covers only the first source object. Exact leaf paths also
    work. The root scope is deliberately disallowed because it would collapse
    all semantic provenance into one uninformative record.
    """

    if not scope or scope == "/":
        return False
    return leaf == scope or leaf.startswith(scope + "/")


def _scope_depth(scope: str) -> int:
    return len([part for part in scope.split("/") if part])


def normalize_field_provenance(
    spec: ProblemSpec,
    field_provenance: list[FieldProvenance],
) -> tuple[list[FieldProvenance], dict[str, Any]]:
    """Expand compact LLM provenance scopes to strict leaf-level provenance.

    The parser may emit a compact record for any semantic subtree, such as
    ``/geometry``, ``/materials/0`` or ``/optimization/objectives/0``. For each
    populated semantic leaf, Python chooses the most specific covering scope
    and binds the exact value from ``ProblemSpec``.

    Important behavior:
    - values are never trusted from the LLM; Python binds the authoritative
      value from ``ProblemSpec``;
    - a more-specific scope overrides a broader scope;
    - administrative/empty/nonexistent paths are dropped;
    - identical duplicate scopes are collapsed;
    - conflicting duplicate scopes fail closed;
    - an uncovered semantic leaf still fails closed.
    """

    expected = required_provenance_paths(spec)
    payload = spec.model_dump(exclude_none=True)

    # First normalize and deduplicate the compact rules themselves.
    scope_rules: dict[str, FieldProvenance] = {}
    dropped_extra: list[str] = []
    collapsed_duplicates: list[str] = []
    conflicting_duplicates: list[str] = []

    for raw in field_provenance:
        try:
            scope = _normalize_pointer(raw.field_path)
        except ValueError:
            dropped_extra.append(raw.field_path)
            continue

        covered = [leaf for leaf in expected if _scope_matches(scope, leaf)]
        if not covered:
            # Common examples: /.../id, /schema_version, an empty properties
            # container, or another non-semantic/admin path.
            dropped_extra.append(scope or raw.field_path)
            continue

        normalized_rule = raw.model_copy(update={"field_path": scope})
        previous = scope_rules.get(scope)
        if previous is None:
            scope_rules[scope] = normalized_rule
            continue

        same_metadata = (
            previous.source == normalized_rule.source
            and previous.evidence == normalized_rule.evidence
            and math.isclose(
                float(previous.confidence),
                float(normalized_rule.confidence),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        )
        if same_metadata:
            collapsed_duplicates.append(scope)
        else:
            conflicting_duplicates.append(scope)

    if conflicting_duplicates:
        raise ValueError(
            "Conflicting provenance scopes: "
            f"{sorted(set(conflicting_duplicates))}"
        )

    # ``unit_system`` is a normalized summary of the unit-bearing engineering
    # values already present in the specification. It is useful metadata, but
    # requiring the LLM to emit a separate provenance record for it caused
    # needless failures. When populated and not explicitly covered, bind a
    # transparent derived provenance record in Python.
    if "/unit_system" in expected and not any(
        _scope_matches(scope, "/unit_system") for scope in scope_rules
    ):
        scope_rules["/unit_system"] = FieldProvenance(
            field_path="/unit_system",
            source="derived_from_spec",
            evidence="derived from unit-bearing specification values",
            confidence=1.0,
        )

    expanded: list[FieldProvenance] = []
    uncovered: list[str] = []
    scopes_used: set[str] = set()

    for leaf in expected:
        candidates = [
            (scope, rule)
            for scope, rule in scope_rules.items()
            if _scope_matches(scope, leaf)
        ]
        if not candidates:
            uncovered.append(leaf)
            continue

        # Longest/most-specific JSON Pointer wins. This lets a general
        # /optimization rule be overridden by /optimization/constraints/0.
        scope, rule = max(
            candidates,
            key=lambda item: (_scope_depth(item[0]), len(item[0])),
        )
        scopes_used.add(scope)
        exact_value = get_pointer(payload, leaf)
        expanded.append(
            rule.model_copy(
                update={
                    "field_path": leaf,
                    "value": exact_value,
                }
            )
        )

    diagnostics: dict[str, Any] = {
        "input_scope_records": len(field_provenance),
        "accepted_scope_records": len(scope_rules),
        "expanded_leaf_records": len(expanded),
        "scopes_used": sorted(scopes_used),
        "dropped_extra": sorted(set(dropped_extra)),
        "collapsed_duplicates": sorted(set(collapsed_duplicates)),
        "conflicting_duplicates": sorted(set(conflicting_duplicates)),
        "uncovered": sorted(uncovered),
    }
    return expanded, diagnostics


def validate_field_provenance(
    spec: ProblemSpec,
    field_provenance: list[FieldProvenance],
) -> None:
    """Strictly validate already-expanded leaf-level provenance."""

    expected = set(required_provenance_paths(spec))
    by_path: dict[str, FieldProvenance] = {}
    duplicates: list[str] = []

    for item in field_provenance:
        if item.field_path in by_path:
            duplicates.append(item.field_path)
        by_path[item.field_path] = item

    if duplicates:
        raise ValueError(
            f"Duplicate field_provenance entries: {sorted(set(duplicates))}"
        )

    missing = sorted(expected - set(by_path))
    extra = sorted(set(by_path) - expected)
    if missing or extra:
        raise ValueError(
            "Field provenance must exactly cover populated semantic leaves. "
            f"missing={missing}, extra={extra}"
        )

    payload = spec.model_dump(exclude_none=True)
    stale: list[str] = []
    for path in sorted(expected):
        actual = get_pointer(payload, path)
        recorded = by_path[path].value
        if not _values_match(actual, recorded):
            stale.append(f"{path}: spec={actual!r}, provenance={recorded!r}")

    if stale:
        raise ValueError("Stale field provenance values: " + "; ".join(stale))


def reconcile_after_user_edit(
    old_provenance: list[FieldProvenance],
    new_spec: ProblemSpec,
    *,
    evidence: str,
) -> list[FieldProvenance]:
    """Rebuild exact coverage after a deterministic user-authorized edit."""

    payload = new_spec.model_dump(exclude_none=True)
    expected = required_provenance_paths(new_spec)
    old_by_path = {item.field_path: item for item in old_provenance}

    reconciled: list[FieldProvenance] = []
    for path in expected:
        actual = get_pointer(payload, path)
        previous = old_by_path.get(path)
        if previous is not None and _values_match(actual, previous.value):
            reconciled.append(previous)
        else:
            reconciled.append(
                FieldProvenance(
                    field_path=path,
                    source="user_clarification",
                    value=actual,
                    evidence=evidence.strip() or "user clarification",
                    confidence=1.0,
                )
            )

    validate_field_provenance(new_spec, reconciled)
    return reconciled
