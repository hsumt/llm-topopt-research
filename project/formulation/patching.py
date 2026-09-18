"""Deterministic application of LLM-proposed formulation patches."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from project.formulation.models import ResolutionProposal
from project.parser.provenance import reconcile_after_user_edit, validate_field_provenance
from project.parser.schema import ParserResult, ProblemSpec


def _decode_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"Patch path must be a JSON Pointer: {pointer!r}")
    return [_decode_token(token) for token in pointer.split("/")[1:]]


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


def apply_resolution(
    parser_result: ParserResult,
    proposal: ResolutionProposal,
    *,
    user_message: str,
) -> ParserResult:
    """Apply a resolution proposal and revalidate the complete parser contract."""

    if proposal.action != "apply":
        return parser_result.model_copy(deep=True)

    payload = parser_result.spec.model_dump(exclude_none=True)
    for operation in proposal.operations:
        _apply_one(
            payload,
            op=operation.op,
            path=operation.path,
            value=operation.value,
        )

    # The schema is the hard boundary: an LLM cannot patch arbitrary keys or
    # create solver-specific configuration because ProblemSpec forbids extras.
    new_spec = ProblemSpec.model_validate(payload)

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