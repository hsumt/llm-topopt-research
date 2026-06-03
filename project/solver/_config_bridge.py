"""
_config_bridge.py
Translates ProblemSpec from parser to SIMP loop inputs.

CHANGED:
  - _make_predicate: added center-edge locations (right_center, top_center,
    bottom_center, left_center). These require nelx/nely for tolerance. 
    Signature now includes nelx, nely.
  - build_bcs_from_spec, build_load_from_spec: pass nelx, nely to predicate.
  - extract_simp_params: reads Lx/Ly from schema if provided; returns them.
"""

import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.mesh import locate_entities_boundary, meshtags

# ------------------------------------------------------------------
# Location string → geometric predicate
# ------------------------------------------------------------------

def _make_predicate(location: str, Lx: float, Ly: float,
                    nelx: int = 80, nely: int = 50):
    """
    Maps the parser's location string to a DOLFINx geometric predicate.

    Center predicates (right_center, top_center, etc.) use a tolerance of
    60% of one element dimension. This finds exactly one node when the mesh
    count is even, and the two nearest nodes when odd (load splits equally).

    Add new location strings here as new problem types are added.
    """
    atol_x = (Lx / nelx) * 0.6   # 60% of element width
    atol_y = (Ly / nely) * 0.6   # 60% of element height

    predicates = {
        # Full edges
        "left_edge":   lambda x: np.isclose(x[0], 0.0),
        "right_edge":  lambda x: np.isclose(x[0], Lx),
        "bottom_edge": lambda x: np.isclose(x[1], 0.0),
        "top_edge":    lambda x: np.isclose(x[1], Ly),

        # Corners
        "right_tip":    lambda x: np.logical_and(
                            np.isclose(x[0], Lx), np.isclose(x[1], 0.0)),
        # "right_center": lambda x: np.logical_and(
        #                     np.isclose(x[0], Lx), np.isclose(x[1], Ly / 2.0, atol=atol_y)),
        "bottom_left":  lambda x: np.logical_and(
                            np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
        "bottom_right": lambda x: np.logical_and(
                            np.isclose(x[0], Lx),  np.isclose(x[1], 0.0)),
        "top_left":     lambda x: np.logical_and(
                            np.isclose(x[0], 0.0), np.isclose(x[1], Ly)),
        "top_right":    lambda x: np.logical_and(
                            np.isclose(x[0], Lx),  np.isclose(x[1], Ly)),

        # ADDED: Center-edge locations (task 1.1, MBB, Michell)
        # atol_y / atol_x: tolerance = 0.6 × element size in the perpendicular direction.
        # Ensures single-node selection for even mesh counts.
        "right_center": lambda x: np.logical_and(
                            np.isclose(x[0], Lx),
                            np.isclose(x[1], Ly / 2.0, atol=atol_y)),
        "left_center":  lambda x: np.logical_and(
                            np.isclose(x[0], 0.0),
                            np.isclose(x[1], Ly / 2.0, atol=atol_y)),
        "top_center":   lambda x: np.logical_and(
                            np.isclose(x[1], Ly),
                            np.isclose(x[0], Lx / 2.0, atol=atol_x)),
        "bottom_center": lambda x: np.logical_and(
                            np.isclose(x[1], 0.0),
                            np.isclose(x[0], Lx / 2.0, atol=atol_x)),
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

def build_bcs_from_spec(spec, V, Lx, Ly, nelx=80, nely=50):
    bcs = []
    domain = V.mesh
    fdim = domain.topology.dim - 1

    for bc_spec in spec.bcs:
        predicate = _make_predicate(bc_spec.location, Lx, Ly, nelx, nely)
        dof_idx = _DOF_INDEX[bc_spec.dof]

        # Corner locations must use vertex topology (dim=0)
        CORNER_LOCATIONS = {
            "right_tip", "bottom_left", "bottom_right", "top_left", "top_right"
        }
        if bc_spec.location in CORNER_LOCATIONS:
            domain.topology.create_connectivity(0, domain.topology.dim)
            entities = locate_entities_boundary(domain, 0, predicate)
        else:
            entities = locate_entities_boundary(domain, fdim, predicate)

        dofs = fem.locate_dofs_topological(V.sub(dof_idx), 
                   0 if bc_spec.location in CORNER_LOCATIONS else fdim, 
                   entities)

        bc = fem.dirichletbc(PETSc.ScalarType(bc_spec.value), dofs, V.sub(dof_idx))
        print(f"BC '{bc_spec.location}' dof='{bc_spec.dof}': entities={len(entities)}, dofs={len(dofs)}")
        bcs.append(bc)

    return bcs

def build_load_from_spec(spec, V, Lx, Ly, nelx=80, nely=50):
    F = fem.Function(V)
    F.x.array[:] = 0.0
    domain = V.mesh

    for load_spec in spec.loads:
        predicate = _make_predicate(load_spec.location, Lx, Ly, nelx, nely)
        dof_idx = _DOF_INDEX[load_spec.dof]

        # Point loads: use vertex topology (dim=0)
        vdim = 0
        domain.topology.create_connectivity(vdim, domain.topology.dim)
        vertices = locate_entities_boundary(domain, vdim, predicate)
        
        print(f"Load '{load_spec.location}' dof='{load_spec.dof}': vertices={len(vertices)}")

        if len(vertices) == 0:
            raise RuntimeError(
                f"build_load_from_spec: no vertex found for "
                f"location='{load_spec.location}', dof='{load_spec.dof}'."
            )

        dofs = fem.locate_dofs_topological(V.sub(dof_idx), vdim, vertices)

        print(f"  dofs (parent indices): {dofs}")

        load_per_node = load_spec.value / len(dofs)
        for d in dofs:
            F.x.array[d] += load_per_node  # dofs are already parent indices

    F.x.scatter_forward()
    return F


def extract_simp_params(spec) -> dict:
    """
    Pull SIMP + geometry parameters from spec.
    Returns a plain dict so SIMP_MASTER stays decoupled from Pydantic.

    CHANGED: Reads Lx/Ly from schema (new fields). Falls back to nelx/nely
             aspect-ratio default if not provided. Returns Lx/Ly so
             main_from_spec does not recompute them independently.

    r_min in spec is in element units (Sigmund MATLAB convention).
    Helmholtz filter expects meters → convert here.
    """
    nelx = spec.mesh.nx
    nely = spec.mesh.ny

    # CHANGED: use schema Lx/Ly when provided; default to unit-height scaling
    Lx = float(spec.mesh.Lx) if spec.mesh.Lx is not None else float(nelx) / float(nely)
    Ly = float(spec.mesh.Ly) if spec.mesh.Ly is not None else 1.0

    # r_min element units → meters
    r_min_meters = spec.simp.r_min * (Lx / nelx)

    return {
        "penal":   spec.simp.penal,
        "volfrac": spec.simp.vol_frac,
        "r_min":   r_min_meters,
        "E":       spec.material.E,
        "nu":      spec.material.nu,
        "nelx":    nelx,
        "nely":    nely,
        "Lx":      Lx,   # ADDED: returned so main_from_spec doesn't recompute
        "Ly":      Ly,   # ADDED
    }