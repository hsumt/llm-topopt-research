""" Translates the ProblemSpec from parser to the SIMP loop"""


import numpy as np
from dolfinx import fem
from petsc4py import PETSc


# ------------------------------------------------------------------
# Location string → geometric predicate
# ------------------------------------------------------------------

def _make_predicate(location: str, Lx: float, Ly: float):
    """
    Maps the parser's location string to a DOLFINx geometric predicate.
    Add new location strings here as new problem types are added.
    """
    predicates = {
        "left_edge":    lambda x: np.isclose(x[0], 0.0),
        "right_edge":   lambda x: np.isclose(x[0], Lx),
        "bottom_edge":  lambda x: np.isclose(x[1], 0.0),
        "top_edge":     lambda x: np.isclose(x[1], Ly),
        "right_tip":    lambda x: np.logical_and(
                            np.isclose(x[0], Lx),
                            np.isclose(x[1], 0.0)),
        "bottom_left":  lambda x: np.logical_and(
                            np.isclose(x[0], 0.0),
                            np.isclose(x[1], 0.0)),
        "bottom_right": lambda x: np.logical_and(
                            np.isclose(x[0], Lx),
                            np.isclose(x[1], 0.0)),
        "top_left":     lambda x: np.logical_and(
                            np.isclose(x[0], 0.0),
                            np.isclose(x[1], Ly)),
        "top_right":    lambda x: np.logical_and(
                            np.isclose(x[0], Lx),
                            np.isclose(x[1], Ly)),
    }
    if location not in predicates:
        raise ValueError(
            f"Unknown location '{location}'. "
            f"Valid options: {list(predicates.keys())}"
        )
    return predicates[location]


# ------------------------------------------------------------------
# DOF string → subspace index
# ------------------------------------------------------------------

_DOF_INDEX = {"x": 0, "y": 1, "z": 2}


# ------------------------------------------------------------------
# Main translation functions
# ------------------------------------------------------------------

def build_bcs_from_spec(spec, V, Lx: float, Ly: float) -> list:
    from dolfinx.fem import Constant
    from petsc4py import PETSc

    bcs = []
    for bc_spec in spec.bcs:
        predicate = _make_predicate(bc_spec.location, Lx, Ly)
        dof_idx   = _DOF_INDEX[bc_spec.dof]

        V_sub, _ = V.sub(dof_idx).collapse()

        dofs_raw = fem.locate_dofs_geometrical(
            (V.sub(dof_idx), V_sub),
            predicate
        )
        # dofs_raw[0] = parent space DOF indices
        # dofs_raw[1] = collapsed subspace DOF indices

        bc = fem.dirichletbc(
            Constant(V.mesh, PETSc.ScalarType(bc_spec.value)),  # overload 2
            dofs_raw[1],        # 1D int32 array, subspace indices
            V.sub(dof_idx)      # subspace
        )
        bcs.append(bc)

    return bcs


def build_load_from_spec(spec, V, Lx: float, Ly: float) -> fem.Function:
    F = fem.Function(V)
    F.x.array[:] = 0.0

    for load_spec in spec.loads:
        predicate = _make_predicate(load_spec.location, Lx, Ly)
        dof_idx   = _DOF_INDEX[load_spec.dof]

        V_sub, _ = V.sub(dof_idx).collapse()

        dofs_raw = fem.locate_dofs_geometrical(
            (V.sub(dof_idx), V_sub),
            predicate
        )
        # dofs_raw[0] = parent space DOF indices
        # dofs_raw[1] = subspace DOF indices
        parent_dofs = dofs_raw[0]   # 1D int32 array

        for parent_dof in parent_dofs:
            F.x.array[parent_dof] += load_spec.value

    F.x.scatter_forward()
    return F


def extract_simp_params(spec) -> dict:
    """
    Pull SIMP parameters directly from spec.
    Returns a plain dict so SIMP_MASTER stays decoupled from Pydantic.

    is r_min in element units (Sigmund MATLAB) or meters (Filter)
    """
    nelx = spec.mesh.nx
    nely = spec.mesh.ny
    Lx   = float(nelx) / float(nely)   # matches SIMP_MASTER geometry assumption
    
    # r_min from parser is in element units (Sigmund convention)
    # Helmholtz filter expects meters
    # element_size = Lx / nelx
    r_min_meters = spec.simp.r_min * (Lx / nelx)

    return {
        "penal":   spec.simp.penal,
        "volfrac": spec.simp.vol_frac,
        "r_min":   r_min_meters,        # ← converted to meters
        "E":       spec.material.E,
        "nu":      spec.material.nu,
        "nelx":    nelx,
        "nely":    nely,
    }