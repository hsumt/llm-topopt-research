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
    """
    Build Dirichlet BCs from ProblemSpec.bcs list.

    Each BC entry has: location (str), dof (str), value (float)
    """
    bcs = []
    for bc_spec in spec.bcs:
        predicate = _make_predicate(bc_spec.location, Lx, Ly)
        dof_idx   = _DOF_INDEX[bc_spec.dof]

        V_sub, _  = V.sub(dof_idx).collapse()
        dofs      = fem.locate_dofs_geometrical(
                        (V.sub(dof_idx), V_sub), predicate
                    )
        bc = fem.dirichletbc(
            PETSc.ScalarType(bc_spec.value),
            dofs,
            V.sub(dof_idx)
        )
        bcs.append(bc)

    return bcs


def build_load_from_spec(spec, V, Lx: float, Ly: float) -> fem.Function:
    """
    Build load vector from ProblemSpec.loads list.
    Supports multiple load entries (superposition).

    Each load entry has: location (str), dof (str), value (float)
    """
    F = fem.Function(V)
    F.x.array[:] = 0.0

    for load_spec in spec.loads:
        predicate = _make_predicate(load_spec.location, Lx, Ly)
        dof_idx   = _DOF_INDEX[load_spec.dof]

        V_sub, _  = V.sub(dof_idx).collapse()
        dofs      = fem.locate_dofs_geometrical(
                        (V.sub(dof_idx), V_sub), predicate
                    )

        # dofs is a 2-column array: col 0 = parent DOF, col 1 = subspace DOF
        # We write into the parent DOF indices
        for parent_dof, _ in dofs:
            F.x.array[parent_dof] += load_spec.value

    F.x.scatter_forward()
    return F


def extract_simp_params(spec) -> dict:
    """
    Pull SIMP parameters directly from spec.
    Returns a plain dict so SIMP_MASTER stays decoupled from Pydantic.

    is r_min in element units (Sigmund MATLAB) or meters (Filter)
    """
    return {
        "penal":    spec.simp.penal,
        "volfrac":  spec.simp.vol_frac,
        "r_min":    spec.simp.r_min,
        "E":        spec.material.E,
        "nu":       spec.material.nu,
        "nelx":     spec.mesh.nx,
        "nely":     spec.mesh.ny,
    }