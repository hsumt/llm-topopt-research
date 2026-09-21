"""Deterministic application of LLM-proposed formulation patches."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from project.formulation.models import PatchOperation, ResolutionProposal
from project.parser.provenance import reconcile_after_user_edit, validate_field_provenance
from project.parser.schema import ParserResult, ProblemSpec


def _decode_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _encode_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"Patch path must be a JSON Pointer: {pointer!r}")
    return [_decode_token(token) for token in pointer.split("/")[1:]]


def _pointer(tokens: list[str]) -> str:
    return "/" + "/".join(_encode_token(token) for token in tokens)


def _parent_and_token(document: Any, pointer: str) -> tuple[Any, str]:
    tokens = _tokens(pointer)
    if not tokens:
        raise ValueError("Patching the entire ProblemSpec root is not allowed")

    node = document
    for token in tokens[:-1]:
        if isinstance(node, list):
            if token == "-":
                raise ValueError("'-' is only valid as the final token of an add")
            node = node[int(token)]
        elif isinstance(node, dict):
            if token not in node:
                raise KeyError(f"Patch parent path does not exist: {pointer}")
            node = node[token]
        else:
            raise KeyError(f"Patch parent path does not resolve to a container: {pointer}")
    return node, tokens[-1]


def _apply_one(document: Any, *, op: str, path: str, value: Any = None) -> None:
    if path == "/schema_version" or path.startswith("/schema_version/"):
        raise ValueError("schema_version is not user-editable")

    parent, token = _parent_and_token(document, path)

    if op == "add":
        if isinstance(parent, list):
            if token == "-":
                parent.append(deepcopy(value))
            else:
                index = int(token)
                if index < 0 or index > len(parent):
                    raise IndexError(path)
                parent.insert(index, deepcopy(value))
        elif isinstance(parent, dict):
            parent[token] = deepcopy(value)
        else:
            raise TypeError(f"Cannot add at {path}: parent is not a container")
        return

    if op == "replace":
        if isinstance(parent, list):
            index = int(token)
            parent[index] = deepcopy(value)
        elif isinstance(parent, dict):
            if token not in parent:
                raise KeyError(f"Replace target does not exist: {path}")
            parent[token] = deepcopy(value)
        else:
            raise TypeError(f"Cannot replace at {path}: parent is not a container")
        return

    if op == "remove":
        if isinstance(parent, list):
            del parent[int(token)]
        elif isinstance(parent, dict):
            if token not in parent:
                raise KeyError(f"Remove target does not exist: {path}")
            del parent[token]
        else:
            raise TypeError(f"Cannot remove at {path}: parent is not a container")
        return

    raise ValueError(f"Unsupported patch operation: {op!r}")


def _sanitize_solid_displacement_language(value: Any, spec: ProblemSpec) -> tuple[Any, bool]:
    """Drop model-invented rotational-DOF claims for a solid displacement model."""
    solid_displacement = any(
        physics.family == "solid_mechanics" and "displacement" in physics.fields
        for physics in spec.physics
    )
    if not solid_displacement:
        return value, False

    changed = False

    def clean(text: str) -> str:
        nonlocal changed
        original = text
        text = re.sub(
            r"\s*Rotational DOF are not constrained\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"[;,]?\s*rotational DOF are free\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = text.replace(" .", ".")
        if text != original:
            changed = True
        return text

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return clean(item)
        if isinstance(item, list):
            return [walk(x) for x in item]
        if isinstance(item, dict):
            return {k: walk(v) for k, v in item.items()}
        return item

    return walk(deepcopy(value)), changed


def normalize_resolution_proposal(
    spec: ProblemSpec,
    proposal: ResolutionProposal,
) -> tuple[ResolutionProposal, list[dict[str, Any]]]:
    """Normalize narrow mechanical patch mistakes without changing user intent.

    Rules:
    - ``replace`` of an absent optional dict member becomes ``add``.
    - numeric list-object replacements are retargeted by stable ``id`` when a
      stale index points at a different object.
    - unsupported rotational-DOF language is removed for a solid displacement
      formulation because the user did not supply that claim.
    """
    if proposal.action != "apply":
        return proposal, []

    document = spec.model_dump(exclude_none=True)
    normalized: list[PatchOperation] = []
    notes: list[dict[str, Any]] = []

    for operation in proposal.operations:
        op = operation.op
        path = operation.path
        value, language_changed = _sanitize_solid_displacement_language(operation.value, spec)
        if language_changed:
            notes.append(
                {
                    "kind": "drop_inapplicable_rotational_dof_language",
                    "path": path,
                }
            )

        parent, token = _parent_and_token(document, path)

        # Prevent stale numeric indices from replacing the wrong named object.
        if (
            op == "replace"
            and isinstance(parent, list)
            and token != "-"
            and isinstance(value, dict)
            and isinstance(value.get("id"), str)
        ):
            index = int(token)
            if index < 0 or index >= len(parent):
                raise IndexError(path)
            current = parent[index]
            proposed_id = value["id"]
            current_id = current.get("id") if isinstance(current, dict) else None
            if current_id != proposed_id:
                matches = [
                    i
                    for i, item in enumerate(parent)
                    if isinstance(item, dict) and item.get("id") == proposed_id
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Unsafe list-object replacement at {path}: target id "
                        f"{current_id!r} does not match proposed id {proposed_id!r}, "
                        "and no unique stable-id retarget is available."
                    )
                tokens = _tokens(path)
                old_path = path
                tokens[-1] = str(matches[0])
                path = _pointer(tokens)
                parent, token = _parent_and_token(document, path)
                notes.append(
                    {
                        "kind": "retarget_replace_by_stable_id",
                        "from": old_path,
                        "to": path,
                        "id": proposed_id,
                    }
                )

        # Optional fields disappear from exclude_none serialization, so the
        # resolver's semantic "replace" must mechanically become "add".
        if op == "replace" and isinstance(parent, dict) and token not in parent:
            op = "add"
            notes.append(
                {
                    "kind": "replace_missing_member_to_add",
                    "path": path,
                }
            )

        normalized_op = PatchOperation(op=op, path=path, value=value)
        normalized.append(normalized_op)

        # Normalize later operations against the state produced by earlier ones.
        _apply_one(document, op=op, path=path, value=value)

    return proposal.model_copy(update={"operations": normalized}), notes


def apply_operations_to_spec(
    spec: ProblemSpec,
    proposal: ResolutionProposal,
) -> tuple[ProblemSpec, ResolutionProposal, list[dict[str, Any]]]:
    """Apply/validate a proposal against only ProblemSpec, useful for offline replay."""
    normalized, notes = normalize_resolution_proposal(spec, proposal)
    if normalized.action != "apply":
        return spec.model_copy(deep=True), normalized, notes

    # Preserve explicit ``EngineeringValue(value=None, unit=...)`` entries.
    # ``exclude_none=True`` is lossy for those nested models and makes a valid
    # in-memory ProblemSpec fail its own round-trip validation after any edit.
    payload = spec.model_dump(exclude_none=False)
    for operation in normalized.operations:
        _apply_one(payload, op=operation.op, path=operation.path, value=operation.value)
    return ProblemSpec.model_validate(payload), normalized, notes


def apply_resolution(
    parser_result: ParserResult,
    proposal: ResolutionProposal,
    *,
    user_message: str,
) -> ParserResult:
    """Apply a resolution proposal and revalidate the complete parser contract."""

    if proposal.action != "apply":
        return parser_result.model_copy(deep=True)

    new_spec, proposal, _ = apply_operations_to_spec(parser_result.spec, proposal)

    provenance = reconcile_after_user_edit(
        parser_result.field_provenance,
        new_spec,
        evidence=user_message,
    )

    unresolved_ids = set(proposal.resolved_unresolved_ids)
    contradiction_ids = set(proposal.resolved_contradiction_ids)
    incorporated_ids = set(proposal.incorporated_context_ids)

    new_context = []
    for item in parser_result.context_candidates:
        copy = item.model_copy(deep=True)
        if copy.id in incorporated_ids:
            copy.incorporated_into_spec = True
        new_context.append(copy)

    result = ParserResult(
        spec=new_spec,
        field_provenance=provenance,
        unresolved_items=[
            item
            for item in parser_result.unresolved_items
            if item.id not in unresolved_ids
        ],
        contradictions=[
            item
            for item in parser_result.contradictions
            if item.id not in contradiction_ids
        ],
        context_candidates=new_context,
    )

    validate_field_provenance(result.spec, result.field_provenance)
    return result
