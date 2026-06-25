"""
schema.py
"""
from pydantic import BaseModel
from typing import List, Optional, Union


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
    Lx: Optional[float] = None
    Ly: Optional[float] = None


class SIMPConfig(BaseModel):
    penal: float
    vol_frac: float
    r_min: float
    max_iter: int
    tol_change: float


class ProblemSpec(BaseModel):
    name: str
    mesh: MeshConfig
    material: Material
    loads: List[Load]
    bcs: List[BoundaryCondition]
    simp: SIMPConfig


class DefaultedField(BaseModel):
    """
    One field the parser filled in rather than reading from the prompt.

    field_path   : dotted/indexed path into ProblemSpec, e.g. "simp.r_min"
                   or "loads[0].location"
    default_used : the value the parser actually used. NOT always
                   numeric — e.g. "loads[0].location" defaults to a
                   string like "right_center", not a number.
    question     : a single, short question to get a real value instead,
                   stated as "<question>? Default: <value>"
    """
    field_path:   str
    default_used: Union[str, float, int]
    question:     str


class ParserResult(BaseModel):
    """
    Wrapper returned by parse_problem(). Always contains a complete,
    runnable spec (with defaults filled in) AND the list of which fields
    were defaulted, so the caller can decide whether to ask before running.
    """
    spec:            ProblemSpec
    defaulted_fields: List[DefaultedField] = []