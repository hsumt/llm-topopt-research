"""Solver-independent engineering problem representation.

The pre-solve IR is intentionally *partial-friendly*: missing engineering
information is represented as missing/None and is handled by the formulation
critic and clarification workflow, not by making Pydantic reject the entire
parse. Solver/backend configuration remains downstream.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineeringValue(StrictModel):
    """A value attached to the engineering formulation.

    Scalars are accepted as shorthand and deterministically wrapped as
    ``{"value": scalar}``. This makes the IR tolerant of compact LLM output
    without weakening the canonical in-memory representation.
    """

    value: Any
    unit: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _wrap_scalar(cls, data):
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return data
        return {"value": data}


class RegionSpec(StrictModel):
    id: str
    description: str
    role: str | None = None
    tags: list[str] = Field(default_factory=list)


class GeometrySpec(StrictModel):
    spatial_dimension: Literal[1, 2, 3] | None = None
    description: str | None = None
    # Keep this flexible enough for either a short label ("Cartesian") or an
    # explicit axis map such as {"x": "normal to plate", "y": "up"}.
    coordinate_system: str | dict[str, str] | None = None
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)
    regions: list[RegionSpec] = Field(default_factory=list)


PhysicsFamily = Literal[
    "solid_mechanics",
    "thermal",
    "fluid",
    "electromagnetics",
    "acoustics",
    "mass_transport",
    "other",
    "unknown",
]


class PhysicsSpec(StrictModel):
    id: str
    family: PhysicsFamily
    model: str | None = None
    regime: Literal[
        "steady",
        "transient",
        "quasi_static",
        "frequency_domain",
        "eigenvalue",
        "unknown",
    ] | None = None
    fields: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class CouplingSpec(StrictModel):
    id: str
    physics_ids: list[str]
    kind: str
    description: str | None = None
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class MaterialSpec(StrictModel):
    id: str
    region: str | None = None
    model: str | None = None
    properties: dict[str, EngineeringValue] = Field(default_factory=dict)


class BoundaryConditionSpec(StrictModel):
    id: str
    physics_id: str | None = None
    location: str | None = None
    field: str | None = None
    kind: str | None = None
    components: list[str] = Field(default_factory=list)
    value: EngineeringValue | None = None
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class InitialConditionSpec(StrictModel):
    id: str
    physics_id: str | None = None
    region: str | None = None
    field: str | None = None
    value: EngineeringValue | None = None


class SourceSpec(StrictModel):
    id: str
    physics_id: str | None = None
    location: str | None = None
    field: str | None = None
    kind: str | None = None
    value: EngineeringValue | None = None
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class LoadCaseSpec(StrictModel):
    id: str
    name: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    boundary_condition_ids: list[str] = Field(default_factory=list)
    description: str | None = None


class DesignVariableSpec(StrictModel):
    id: str
    kind: str | None = None
    target: str | None = None
    region: str | None = None
    lower_bound: EngineeringValue | None = None
    upper_bound: EngineeringValue | None = None
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class ObjectiveSpec(StrictModel):
    id: str
    sense: Literal["minimize", "maximize", "target"] | None = None
    quantity: str | None = None
    region: str | None = None
    target: EngineeringValue | None = None
    weight: float | None = None
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class ConstraintSpec(StrictModel):
    id: str
    quantity: str | None = None
    relation: Literal["<=", ">=", "=", "range"] | None = None
    limit: EngineeringValue | None = None
    region: str | None = None
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class OptimizationSpec(StrictModel):
    design_variables: list[DesignVariableSpec] = Field(default_factory=list)
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)


class ManufacturingSpec(StrictModel):
    """Manufacturing requirements that constrain the engineering problem."""

    process: str | None = None
    material_form: str | None = None
    machine: str | None = None
    stock_material: str | None = None
    notes: str | None = None
    requirements: list[str] = Field(default_factory=list)
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class DiscretizationSpec(StrictModel):
    """Only user-stated discretization requirements."""

    method: str | None = None
    description: str | None = None
    parameters: dict[str, EngineeringValue] = Field(default_factory=dict)


class ProblemSpec(StrictModel):
    """Solver-independent statement of the engineering problem."""

    schema_version: str = "1.0"
    name: str
    problem_kind: Literal[
        "simulation",
        "optimization",
        "inverse_problem",
        "unknown",
    ] = "unknown"
    unit_system: str | None = None

    geometry: GeometrySpec | None = None
    physics: list[PhysicsSpec] = Field(default_factory=list)
    couplings: list[CouplingSpec] = Field(default_factory=list)
    materials: list[MaterialSpec] = Field(default_factory=list)
    boundary_conditions: list[BoundaryConditionSpec] = Field(default_factory=list)
    initial_conditions: list[InitialConditionSpec] = Field(default_factory=list)
    sources: list[SourceSpec] = Field(default_factory=list)
    load_cases: list[LoadCaseSpec] = Field(default_factory=list)
    optimization: OptimizationSpec | None = None
    manufacturing: ManufacturingSpec | None = None
    discretization: DiscretizationSpec | None = None

    requested_outputs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


ProvenanceSource = Literal[
    "explicit",
    "inferred_from_language",
    "inferred_from_standard_name",
    "user_confirmed",
    "user_overridden",
    "user_clarification",
    "derived_from_spec",
]


class FieldProvenance(StrictModel):
    """Compact provenance rule emitted by the parser."""

    field_path: str
    source: ProvenanceSource
    value: Any | None = None
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class UnresolvedItem(StrictModel):
    id: str
    field_path: str | None = None
    issue: str
    evidence: str | None = None
    question: str
    required_for_execution: bool = True


class Contradiction(StrictModel):
    id: str
    field_paths: list[str] = Field(default_factory=list)
    description: str
    evidence: list[str] = Field(default_factory=list)


class ContextCandidate(StrictModel):
    id: str
    text: str
    relevance: Literal["direct", "potential", "background"]
    related_fields: list[str] = Field(default_factory=list)
    reason: str
    incorporated_into_spec: bool = False


class ParserResult(StrictModel):
    spec: ProblemSpec
    field_provenance: list[FieldProvenance] = Field(default_factory=list)
    unresolved_items: list[UnresolvedItem] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    context_candidates: list[ContextCandidate] = Field(default_factory=list)


class ProblemRoute(StrictModel):
    """Semantic routing result. This is not a solver selection."""

    task_type: Literal[
        "simulation",
        "optimization",
        "inverse_problem",
        "unknown",
    ]
    physics_families: list[PhysicsFamily] = Field(default_factory=list)
    spatial_dimension: Literal[1, 2, 3] | None = None
    optimization_type: Literal[
        "topology",
        "shape",
        "sizing",
        "parameter",
        "other",
        "unknown",
    ] | None = None
    multiphysics: bool = False
    requested_methods: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
