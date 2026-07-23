"""Pydantic schemas for the natural-language topology-optimization parser.

The current deterministic solver is intentionally restricted to 2-D, small-strain,
linear-elastic problems. The schema rejects unsupported 3-D inputs instead of allowing
them to reach a 2-D DOLFINx implementation and fail or, worse, run incorrectly.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


Location = Literal[
    "left_edge", "right_edge", "top_edge", "bottom_edge",
    "top_left", "top_right", "bottom_left", "bottom_right",
    "right_tip", "right_center", "top_center", "bottom_center",
    "left_center",
]
Dof2D = Literal["x", "y"]
LoadKind = Literal["point_force", "edge_resultant", "edge_traction"]


class BoundaryCondition(BaseModel):
    location: Location
    dof: Dof2D
    value: float = 0.0

    @model_validator(mode="after")
    def validate_homogeneous_bc(self):
        if abs(self.value) > 1.0e-14:
            raise ValueError(
                "The current compliance solver supports homogeneous "
                "Dirichlet boundary conditions only (value=0)."
            )
        return self


class Load(BaseModel):
    location: Location
    dof: Dof2D
    value: float
    kind: LoadKind = Field(
        default="point_force",
        description=(
            "point_force is a discrete nodal resultant; edge_resultant is a "
            "total edge force; edge_traction is force per in-plane edge length."
        ),
    )


class Material(BaseModel):
    E: float
    nu: float

    @model_validator(mode="after")
    def validate_material(self):
        if self.E <= 0.0:
            raise ValueError("material.E must be positive")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError("material.nu must satisfy -1 < nu < 0.5")
        return self


class MeshConfig(BaseModel):
    nx: int
    ny: int
    nz: Optional[int] = None
    Lx: Optional[float] = None
    Ly: Optional[float] = None

    @model_validator(mode="after")
    def validate_mesh(self):
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("mesh.nx and mesh.ny must be positive")
        if self.nz is not None:
            raise ValueError(
                "The current DOLFINx solver is 2-D only; mesh.nz is unsupported."
            )
        if self.Lx is not None and self.Lx <= 0.0:
            raise ValueError("mesh.Lx must be positive")
        if self.Ly is not None and self.Ly <= 0.0:
            raise ValueError("mesh.Ly must be positive")
        return self


class SIMPConfig(BaseModel):
    penal: float
    vol_frac: float
    r_min: float = Field(
        description=(
            "Cone-equivalent physical filter radius. The Helmholtz PDE uses "
            "r_pde = r_min/(2*sqrt(3))."
        )
    )
    max_iter: int
    tol_change: float

    @model_validator(mode="after")
    def validate_simp(self):
        if self.penal < 1.0:
            raise ValueError("simp.penal must be >= 1")
        if not (0.0 < self.vol_frac < 1.0):
            raise ValueError("simp.vol_frac must lie strictly between 0 and 1")
        if self.r_min <= 0.0:
            raise ValueError("simp.r_min must be positive")
        if self.max_iter <= 0:
            raise ValueError("simp.max_iter must be positive")
        if self.tol_change <= 0.0:
            raise ValueError("simp.tol_change must be positive")
        return self


class ProblemSpec(BaseModel):
    name: str
    mesh: MeshConfig
    material: Material
    loads: List[Load]
    bcs: List[BoundaryCondition]
    simp: SIMPConfig

    @model_validator(mode="after")
    def validate_problem(self):
        if not self.loads:
            raise ValueError("At least one load is required")
        if not self.bcs:
            raise ValueError("At least one boundary condition is required")
        return self


class DefaultedField(BaseModel):
    """One field filled by the parser rather than stated by the user."""

    field_path: str
    default_used: Union[str, float, int]
    question: str


class ParserResult(BaseModel):
    """Complete runnable specification plus an audit trail of defaults."""

    spec: ProblemSpec
    defaulted_fields: List[DefaultedField] = Field(default_factory=list)
