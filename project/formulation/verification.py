"""Small deterministic gate for whether a formulation can be approved.

This is NOT physics verification. It only checks the state of the pre-solve
problem-definition workflow.
"""

from __future__ import annotations

from project.formulation.models import FormulationSession, ReadinessReport


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

    for concern in session.critic_result.concerns:
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