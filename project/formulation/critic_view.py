"""Compact, auditable view of formulation state for the LLM critic.

The full ParserResult contains leaf-expanded provenance for deterministic audit.
That representation is intentionally *not* sent back to the LLM: doing so is
expensive and adds no formulation information.  This module builds the much
smaller semantic view needed for critique while preserving the complete state
inside Python/session JSON.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from project.parser.schema import ParserResult, ProblemRoute


def _section(path: str) -> str:
    parts = [part for part in (path or "").split("/") if part]
    if not parts:
        return "/"
    head = parts[0]
    # Object-indexed sections retain the object index so different loads,
    # materials, physics blocks, etc. do not get conflated.
    if head in {
        "physics",
        "materials",
        "boundary_conditions",
        "initial_conditions",
        "sources",
        "load_cases",
        "couplings",
    } and len(parts) >= 2 and parts[1].isdigit():
        return f"/{head}/{parts[1]}"
    return f"/{head}"


def compact_provenance(parser_result: ParserResult) -> list[dict[str, Any]]:
    """Collapse leaf-expanded provenance into semantic-section summaries."""

    groups: dict[tuple[str, str, str, float], list[str]] = defaultdict(list)
    for record in parser_result.field_provenance:
        key = (
            _section(record.field_path),
            record.source,
            record.evidence,
            round(float(record.confidence), 3),
        )
        groups[key].append(record.field_path)

    output: list[dict[str, Any]] = []
    for (section, source, evidence, confidence), paths in sorted(groups.items()):
        output.append(
            {
                "scope": section,
                "source": source,
                "evidence": evidence[:420],
                "confidence": confidence,
                "leaf_count": len(paths),
                "sample_paths": paths[:3],
            }
        )
    return output


def _trim_text(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def compact_critic_payload(
    *,
    problem: str,
    route: ProblemRoute,
    parser_result: ParserResult,
    retrieved_context: list[dict] | None = None,
    recent_user_clarification: str | None = None,
) -> dict[str, Any]:
    """Return only information the critic needs to make formulation decisions."""

    retrieved = []
    for item in (retrieved_context or [])[:4]:
        copied = dict(item)
        copied["text"] = _trim_text(str(copied.get("text", "")), 1400)
        retrieved.append(copied)

    context_candidates = []
    for item in parser_result.context_candidates:
        # Background-only candidates are visible in the UI/session but usually
        # do not need to consume critic context.
        if item.relevance == "background" and not item.incorporated_into_spec:
            continue
        context_candidates.append(item.model_dump())

    return {
        "original_problem": problem.strip(),
        "route": route.model_dump(),
        "current_spec": parser_result.spec.model_dump(exclude_none=True),
        "unresolved_items": [item.model_dump() for item in parser_result.unresolved_items],
        "contradictions": [item.model_dump() for item in parser_result.contradictions],
        "context_candidates": context_candidates,
        "retrieved_context": retrieved,
        "provenance_summary": compact_provenance(parser_result),
        "recent_user_clarification": _trim_text(recent_user_clarification, 1800),
    }
