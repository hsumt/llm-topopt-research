# schema.py
from pydantic import BaseModel
from typing import List, Optional


class BoundaryCondition(BaseModel):
    location: str
    dof: str
    value: float


class Load(BaseModel):
    location: str
    dof: str
    value: float


class Material(BaseModel):
    E: float
    nu: float


class MeshConfig(BaseModel):
    nx: int
    ny: int
    nz: Optional[int] = None
    # ADDED: physical dimensions. If omitted, main_from_spec defaults to
    # Lx = nx/ny (unit-height scaling), Ly = 1.0.
    # Required for non-unit-aspect problems (MBB 3:1, Michell 6:1, etc.)
    Lx: Optional[float] = None
    Ly: Optional[float] = None


class SIMPConfig(BaseModel):
    penal: float
    vol_frac: float
    r_min: float


class ProblemSpec(BaseModel):
    name: str
    mesh: MeshConfig
    material: Material
    loads: List[Load]
    bcs: List[BoundaryCondition]
    simp: SIMPConfig