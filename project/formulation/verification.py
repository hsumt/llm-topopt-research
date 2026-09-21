"""Deterministic pre-solve readiness gate.

This is NOT physics verification.  It checks only whether the engineering
problem definition still contains unresolved human-intent decisions.
"""

from __future__ import annotations

from project.formulation.models import FormulationSession, ReadinessReport


def _paths_overlap(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    a = a.rstrip("/") or "/"
    b = b.rstrip("/") or "/"
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _concern_is_duplicate(session: FormulationSession, concern) -> bool:
    """True when a critic concern is already represented by parser state."""

    if not concern.related_fields:
        return False
    parser_paths = [
        item.field_path
        for item in session.parser_result.unresolved_items
        if item.field_path
    ]
    parser_paths.extend(
        path
        for contradiction in session.parser_result.contradictions
        for path in contradiction.field_paths
    )
    return any(
        _paths_overlap(field, parser_path)
        for field in concern.related_fields
        for parser_path in parser_paths
    )


def check_readiness(session: FormulationSession) -> ReadinessReport:
    blockers: list[str] = []
    warnings: list[str] = []

    result = session.parser_result

    for contradiction in result.contradictions:
        blockers.append(
            f"Contradiction {contradiction.id}: {contradiction.description}"
        )

    for item in result.unresolved_items:
        text = f"Unresolved {item.id}: {item.issue}"
        if item.required_for_execution:
            blockers.append(text)
        else:
            warnings.append(text)

    # The critic is an interpretation layer, not a second issue registry.  Only
    # novel critic concerns are counted; overlapping parser issues are displayed
    # through their canonical unresolved/contradiction record.
    for concern in session.critic_result.concerns:
        if _concern_is_duplicate(session, concern):
            continue
        text = f"Critic concern {concern.id}: {concern.description}"
        if concern.blocking:
            blockers.append(text)
        else:
            warnings.append(text)

    if not result.spec.physics:
        blockers.append("No physics family has been defined in ProblemSpec.")

    if result.spec.problem_kind == "unknown":
        blockers.append("The task type is still unknown.")

    if result.spec.problem_kind == "optimization":
        optimization = result.spec.optimization
        if optimization is None:
            blockers.append("Optimization was requested but OptimizationSpec is absent.")
        elif not optimization.objectives:
            blockers.append("Optimization was requested but no objective is defined.")

    return ReadinessReport(
        ready=not blockers,
        blockers=blockers,
        warnings=warnings,
    )
