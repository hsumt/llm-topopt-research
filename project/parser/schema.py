from pydantic import BaseModel
from typing import List, Optional


class BoundaryCondition(BaseModel):
    node_ids: List[int]
    dof: str
    value: float

class Load(BaseModel):
    node_ids: List[int] # nodes
    dof: str # x y z
    value: float # force


class Material(BaseModel):
    E: float # YM
    nu: float # Poisson's

class MeshConfig(BaseModel):
    nx: int # number of elements x
    ny: int # number of elements y
    nz: Optional[int] = None # number of elements in z for 3D


class SIMPConfig(BaseModel):
    penal: float # Penalization factor
    vol_frac: float # VOLUME FRACTION
    r_min: float # Minimum filter radius


class ProblemSpec(BaseModel):
    name: str
    mesh: MeshConfig
    material: Material
    loads: List[Load]
    bcs: List[BoundaryCondition]
    simp: SIMPConfig