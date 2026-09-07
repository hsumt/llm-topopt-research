"""Deterministic patch gate for formulation-clarification updates.

The LLM may interpret a user's answer into a small structured patch, but this
module is the authority that decides whether that patch is allowed. It never
changes FEM/SIMP code and it never permits edits outside the fields attached to
the currently reviewed formulation issue.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

from project.parser.provenance import canonical_field_paths
from project.parser.schema import FieldProvenance, ProblemSpec

_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")

# Numerical/algorithmic controls are deliberately outside the formulation
# dialogue. They may be changed through the existing explicit parser/default
# workflow, but a formulation question may not smuggle in optimizer tuning.
_NUMERICAL_CONTROL_FIELDS = {
    "mesh.nx",
    "mesh.ny",
    "simp.penal",
    "simp.r_min",
    "simp.max_iter",
    "simp.tol_change",
}


def tokenize_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    for name, index in _PATH_TOKEN.findall(path):
        tokens.append(int(index) if index else name)
    if not tokens:
        raise ValueError(f"Invalid field path: {path}")
    return tokens


def get_path(data: dict, path: str):
    node: Any = data
    for token in tokenize_path(path):
        node = node[token]
    return node


def set_path(data: dict, path: str, value) -> None:
    tokens = tokenize_path(path)
    node: Any = data
    for token in tokens[:-1]:
        node = node[token]
    node[tokens[-1]] = value


def values_match(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1.0e-12, abs_tol=1.0e-12)
    return a == b


def provenance_index(records: list[dict]) -> dict[str, dict]:
    return {str(record["field_path"]): record for record in records}


def synchronize_final_values(spec: ProblemSpec, records: list[dict]) -> list[dict]:
    """Return a copy of provenance with ``value``/``final_value`` bound to spec."""
    payload = spec.model_dump()
    out = deepcopy(records)
    for record in out:
        value = get_path(payload, record["field_path"])
        record["value"] = value
        record["final_value"] = value
    return out


def _record_original_parser_provenance(record: dict) -> None:
    """Preserve parser origin before a human clarification overwrites source."""
    record.setdefault("parser_source", record.get("source"))
    record.setdefault("parser_evidence", record.get("evidence"))
    record.setdefault("parser_value", record.get("value"))


def _validate_authorized_paths(
    *,
    spec: ProblemSpec,
    issue: dict,
    updates: list[dict],
    confirmed_fields: list[str],
) -> None:
    canonical = set(canonical_field_paths(spec))
    allowed = set(issue.get("affected_fields", []))

    touched = [str(item.get("field_path")) for item in updates] + list(confirmed_fields)
    duplicates = sorted({path for path in touched if touched.count(path) > 1})
    if duplicates:
        raise ValueError(f"Duplicate formulation patch field(s): {duplicates}")

    if not allowed and touched:
        raise ValueError(
            "This formulation concern has no editable canonical fields in the "
            "current solver schema. The user must reformulate or accept it as-is."
        )

    unknown = sorted(set(touched) - canonical)
    if unknown:
        raise ValueError(f"Formulation patch referenced non-canonical fields: {unknown}")

    unauthorized = sorted(set(touched) - allowed)
    if unauthorized:
        raise ValueError(
            "Formulation patch attempted to modify field(s) outside the current "
            f"issue authority: {unauthorized}; allowed={sorted(allowed)}"
        )

    numerical = sorted(set(touched) & _NUMERICAL_CONTROL_FIELDS)
    if numerical:
        raise ValueError(
            "Formulation dialogue may not tune numerical/optimizer controls: "
            + ", ".join(numerical)
        )


def apply_resolution(
    *,
    spec: ProblemSpec,
    final_field_provenance: list[dict],
    issue: dict,
    resolution: dict,
    user_answer: str,
) -> tuple[ProblemSpec, list[dict], dict]:
    """Apply one resolver result under a deterministic field-authority gate.

    Accepted resolver statuses:
    - ``update``: one or more allowed fields receive new values;
    - ``confirm_current``: allowed fields are explicitly confirmed unchanged.

    ``ask_again`` is intentionally not accepted here because it makes no state
    change and is handled by the dialogue controller.
    """

    status = resolution.get("status")
    if status not in {"update", "confirm_current"}:
        raise ValueError(f"Resolution status '{status}' cannot be applied")

    updates = list(resolution.get("updates", []))
    confirmed_fields = list(resolution.get("confirmed_fields", []))

    if status == "update" and not updates:
        raise ValueError("update resolution contains no field updates")
    if status == "confirm_current" and not confirmed_fields:
        # Default to all affected fields only when the resolver explicitly says
        # the current interpretation was confirmed.
        confirmed_fields = list(issue.get("affected_fields", []))
        if not confirmed_fields:
            raise ValueError(
                "Cannot confirm a cross-cutting issue with no canonical fields"
            )

    _validate_authorized_paths(
        spec=spec,
        issue=issue,
        updates=updates,
        confirmed_fields=confirmed_fields,
    )

    old_payload = spec.model_dump()
    new_payload = deepcopy(old_payload)

    applied_updates: list[dict] = []
    for update in updates:
        path = str(update["field_path"])
        old_value = get_path(new_payload, path)
        new_value = update.get("new_value")
        set_path(new_payload, path, new_value)
        applied_updates.append(
            {
                "field_path": path,
                "previous_value": old_value,
                "new_value": new_value,
            }
        )

    # Pydantic is the second deterministic gate: enum values, homogeneous BCs,
    # material bounds, mesh positivity, load structure, etc. must remain valid.
    new_spec = ProblemSpec.model_validate(new_payload)
    validated_payload = new_spec.model_dump()

    # Ensure the model validation did not coerce an update into an unexpected
    # value without us recording it.
    for update in applied_updates:
        actual = get_path(validated_payload, update["field_path"])
        update["new_value"] = actual

    records = synchronize_final_values(spec, final_field_provenance)
    by_path = provenance_index(records)

    for update in applied_updates:
        path = update["field_path"]
        record = by_path[path]
        _record_original_parser_provenance(record)
        record.update(
            {
                "source": "user_overridden",
                "value": update["new_value"],
                "final_value": update["new_value"],
                "evidence": user_answer,
                "confidence": 1.0,
                "interaction_status": "formulation_clarification_update",
            }
        )

    for path in confirmed_fields:
        record = by_path[path]
        _record_original_parser_provenance(record)
        current = get_path(validated_payload, path)
        record.update(
            {
                "source": "user_confirmed",
                "value": current,
                "final_value": current,
                "evidence": user_answer,
                "confidence": 1.0,
                "interaction_status": "formulation_clarification_confirmed",
            }
        )

    records = synchronize_final_values(new_spec, records)

    audit = {
        "status": status,
        "issue_category": issue.get("category"),
        "affected_fields": list(issue.get("affected_fields", [])),
        "user_answer": user_answer,
        "updates": applied_updates,
        "confirmed_fields": confirmed_fields,
    }
    return new_spec, records, audit


def provenance_models(records: list[dict]) -> list[FieldProvenance]:
    """Convert final audit dictionaries back to the canonical provenance model."""
    models: list[FieldProvenance] = []
    for record in records:
        models.append(
            FieldProvenance.model_validate(
                {
                    "field_path": record["field_path"],
                    "source": record["source"],
                    "value": record["value"],
                    "evidence": record.get("evidence"),
                    "confidence": record.get("confidence", 1.0),
                }
            )
        )
    return models


def build_review_field_provenance(
    *,
    spec: ProblemSpec,
    parser_field_provenance,
    confirmed_defaults: list[str],
    accepted_remaining_defaults: list[str],
    accepted_after_invalid_input: list[str],
    user_overrides: list[dict],
) -> list[dict]:
    """Bind parser provenance to the exact spec entering formulation review."""
    final_payload = spec.model_dump()
    override_paths = {item["field_path"] for item in user_overrides}
    records: list[dict] = []

    for item in parser_field_provenance:
        record = item.model_dump()
        record["final_value"] = get_path(final_payload, item.field_path)

        if item.field_path in override_paths:
            _record_original_parser_provenance(record)
            record["source"] = "user_overridden"
            record["value"] = record["final_value"]
            match = next(
                update for update in user_overrides
                if update["field_path"] == item.field_path
            )
            record["evidence"] = match.get(
                "response",
                f"User override before formulation review: {match['new_value']!r}",
            )
            record["confidence"] = 1.0
            record["interaction_status"] = "user_overridden"
        elif item.field_path in confirmed_defaults:
            _record_original_parser_provenance(record)
            record["source"] = "user_confirmed"
            record["value"] = record["final_value"]
            record["evidence"] = "User individually confirmed parser default"
            record["confidence"] = 1.0
            record["interaction_status"] = "individually_confirmed_default"
        elif item.field_path in accepted_remaining_defaults:
            record["interaction_status"] = "accepted_after_opt_out"
        elif item.field_path in accepted_after_invalid_input:
            record["interaction_status"] = "default_retained_after_invalid_input"
        elif item.source in {
            "inferred_from_benchmark_name",
            "inferred_from_language",
        }:
            record["interaction_status"] = "pending_formulation_review"
        else:
            record["interaction_status"] = "unchanged"

        records.append(record)

    return synchronize_final_values(spec, records)
