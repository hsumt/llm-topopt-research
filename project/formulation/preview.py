"""Human-readable mirrors of the evolving engineering formulation."""

from __future__ import annotations

import json
from typing import Any

from project.formulation.models import FormulationSession
from project.formulation.verification import check_readiness


def spec_json(session: FormulationSession) -> str:
    return json.dumps(
        session.parser_result.spec.model_dump(exclude_none=True),
        indent=2,
    )


def route_summary(session: FormulationSession) -> str:
    route = session.route
    physics = ", ".join(route.physics_families) or "unknown"
    dimension = f"{route.spatial_dimension}-D" if route.spatial_dimension else "unspecified"
    method = ", ".join(route.requested_methods) or "none explicitly requested"
    return (
        f"Task: {route.task_type}\n"
        f"Physics: {physics}\n"
        f"Dimension: {dimension}\n"
        f"Multiphysics: {route.multiphysics}\n"
        f"Requested methods: {method}"
    )


def issue_summary(session: FormulationSession) -> str:
    readiness = check_readiness(session)
    lines = []
    if readiness.blockers:
        lines.append("Blocking:")
        lines.extend(f"- {item}" for item in readiness.blockers)
    if readiness.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in readiness.warnings)
    if not lines:
        return "No blocking pre-solve formulation issues."
    return "\n".join(lines)


def _fmt_value(value: Any) -> str:
    if value is None:
        return "not stated"
    if hasattr(value, "value"):
        raw = getattr(value, "value")
        unit = getattr(value, "unit", None)
        return f"{raw}{(' ' + unit) if unit else ''}"
    return str(value)


def problem_bullets(session: FormulationSession) -> list[str]:
    """Bullets describing what the system currently believes the problem is."""

    spec = session.parser_result.spec
    bullets: list[str] = []
    bullets.append(f"**Task:** {spec.problem_kind}")
    if spec.geometry:
        if spec.geometry.spatial_dimension:
            bullets.append(f"**Dimension:** {spec.geometry.spatial_dimension}-D")
        if spec.geometry.description:
            bullets.append(f"**Geometry:** {spec.geometry.description}")
        for key, value in spec.geometry.parameters.items():
            bullets.append(f"**Geometry — {key}:** {_fmt_value(value)}")
        for region in spec.geometry.regions:
            role = f" ({region.role})" if region.role else ""
            bullets.append(f"**Region {region.id}{role}:** {region.description}")

    for physics in spec.physics:
        text = physics.family
        if physics.model:
            text += f" / {physics.model}"
        if physics.regime:
            text += f" / {physics.regime}"
        bullets.append(f"**Physics:** {text}")

    for material in spec.materials:
        text = material.model or material.id
        if material.region:
            text += f" in {material.region}"
        bullets.append(f"**Material:** {text}")
        for key, value in material.properties.items():
            bullets.append(f"**Material — {key}:** {_fmt_value(value)}")

    for bc in spec.boundary_conditions:
        kind = bc.kind or "restraint unresolved"
        field = bc.field or "field unresolved"
        bullets.append(
            f"**Boundary condition:** {bc.location} → {kind} on {field}"
        )

    for source in spec.sources:
        where = source.location or "location not stated"
        value = f", {_fmt_value(source.value)}" if source.value else ""
        bullets.append(f"**Load/source:** {source.kind} at {where}{value}")

    if spec.optimization:
        for objective in spec.optimization.objectives:
            bullets.append(
                f"**Objective:** {objective.sense} {objective.quantity}"
            )
        for constraint in spec.optimization.constraints:
            limit = _fmt_value(constraint.limit) if constraint.limit else "not stated"
            bullets.append(
                f"**Constraint:** {constraint.quantity} {constraint.relation} {limit}"
            )
        for variable in spec.optimization.design_variables:
            bullets.append(
                f"**Design variable:** {variable.kind} → {variable.target}"
            )

    if spec.manufacturing:
        if spec.manufacturing.process:
            bullets.append(f"**Manufacturing:** {spec.manufacturing.process}")
        for requirement in spec.manufacturing.requirements:
            bullets.append(f"**Manufacturing requirement:** {requirement}")

    # Assumptions are deliberately shown as assumptions rather than facts. A
    # user should be able to spot and challenge them before approval.
    for assumption in spec.assumptions:
        bullets.append(f"**Assumption (verify before approval):** {assumption}")
    return bullets


def context_bullets(session: FormulationSession) -> list[str]:
    bullets: list[str] = []
    for item in session.parser_result.context_candidates:
        state = "incorporated" if item.incorporated_into_spec else "not incorporated"
        bullets.append(
            f"**{item.relevance.upper()} — {item.id}:** {item.text} "
            f"_({state}; {item.reason})_"
        )
    if not bullets and session.supplied_context:
        bullets.append("Context was supplied, but no context candidates were extracted.")
    return bullets


def open_issue_bullets(session: FormulationSession) -> list[str]:
    """Canonical issue list with parser/critic duplicates removed."""

    readiness = check_readiness(session)
    bullets: list[str] = []
    for item in readiness.blockers:
        bullets.append(f"**BLOCKING:** {item}")
    for item in readiness.warnings:
        bullets.append(f"**WARNING:** {item}")
    return bullets


def problem_graph_dot(session: FormulationSession) -> str:
    """Graphviz problem map: a visual mirror of the current formulation structure."""

    spec = session.parser_result.spec

    def esc(text: str) -> str:
        return str(text).replace('"', "'").replace("\n", " ")

    lines = [
        "digraph formulation {",
        "rankdir=LR;",
        'node [shape=box, style="rounded"];',
        f'problem [label="{esc(spec.name)}\\n{esc(spec.problem_kind)}"];',
    ]

    if spec.geometry:
        geom = spec.geometry.description or "geometry"
        if spec.geometry.spatial_dimension:
            geom = f"{spec.geometry.spatial_dimension}-D | {geom}"
        lines.append(f'geometry [label="Geometry\\n{esc(geom)}"];')
        lines.append("problem -> geometry;")

    for i, physics in enumerate(spec.physics):
        label = physics.family
        if physics.model:
            label += f"\\n{physics.model}"
        lines.append(f'physics_{i} [label="Physics\\n{esc(label)}"];')
        lines.append(f"problem -> physics_{i};")

    if spec.boundary_conditions:
        labels = [f"{bc.location}: {bc.kind or 'restraint unresolved'}" for bc in spec.boundary_conditions[:4]]
        lines.append(f'bcs [label="Supports / BCs\\n{esc(" | ".join(labels))}"];')
        lines.append("problem -> bcs;")

    if spec.sources:
        labels = [f"{src.location or 'unspecified'}: {src.kind}" for src in spec.sources[:4]]
        lines.append(f'loads [label="Loads / Sources\\n{esc(" | ".join(labels))}"];')
        lines.append("problem -> loads;")

    if spec.optimization:
        if spec.optimization.objectives:
            labels = [f"{obj.sense} {obj.quantity}" for obj in spec.optimization.objectives[:3]]
            lines.append(f'objectives [label="Objective\\n{esc(" | ".join(labels))}"];')
            lines.append("problem -> objectives;")
        if spec.optimization.constraints:
            labels = [con.quantity for con in spec.optimization.constraints[:4]]
            lines.append(f'constraints [label="Constraints\\n{esc(" | ".join(labels))}"];')
            lines.append("problem -> constraints;")

    if spec.manufacturing:
        label = spec.manufacturing.process or "manufacturing requirements"
        lines.append(f'mfg [label="Manufacturing\\n{esc(label)}"];')
        lines.append("problem -> mfg;")

    lines.append("}")
    return "\n".join(lines)
