"""Translate a validated ``ProblemSpec`` into deterministic DOLFINx inputs."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from dolfinx.mesh import locate_entities_boundary
from petsc4py import PETSc

from project.solver._03_OPTMZER._filters import (
    R_MIN_CONVENTION,
    r_pde_from_r_min,
)

EDGE_LOCATIONS = {"left_edge", "right_edge", "bottom_edge", "top_edge"}
POINT_LOCATIONS = {
    "right_tip", "right_center", "left_center", "top_center", "bottom_center",
    "bottom_left", "bottom_right", "top_left", "top_right",
}


def _make_predicate(location: str, Lx: float, Ly: float, nelx: int, nely: int):
    """Map a schema location name to a geometric predicate."""
    atol_x = (Lx / nelx) * 0.6
    atol_y = (Ly / nely) * 0.6

    predicates = {
        "left_edge": lambda x: np.isclose(x[0], 0.0),
        "right_edge": lambda x: np.isclose(x[0], Lx),
        "bottom_edge": lambda x: np.isclose(x[1], 0.0),
        "top_edge": lambda x: np.isclose(x[1], Ly),
        "bottom_left": lambda x: np.logical_and(
            np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)
        ),
        "bottom_right": lambda x: np.logical_and(
            np.isclose(x[0], Lx), np.isclose(x[1], 0.0)
        ),
        "top_left": lambda x: np.logical_and(
            np.isclose(x[0], 0.0), np.isclose(x[1], Ly)
        ),
        "top_right": lambda x: np.logical_and(
            np.isclose(x[0], Lx), np.isclose(x[1], Ly)
        ),
        # In this project, "tip" denotes the free-end load point of the
        # standard rectangular cantilever: the midpoint of the right edge.
        "right_tip": lambda x: np.logical_and(
            np.isclose(x[0], Lx),
            np.isclose(x[1], Ly / 2.0, atol=atol_y),
        ),
        "right_center": lambda x: np.logical_and(
            np.isclose(x[0], Lx),
            np.isclose(x[1], Ly / 2.0, atol=atol_y),
        ),
        "left_center": lambda x: np.logical_and(
            np.isclose(x[0], 0.0),
            np.isclose(x[1], Ly / 2.0, atol=atol_y),
        ),
        "top_center": lambda x: np.logical_and(
            np.isclose(x[1], Ly),
            np.isclose(x[0], Lx / 2.0, atol=atol_x),
        ),
        "bottom_center": lambda x: np.logical_and(
            np.isclose(x[1], 0.0),
            np.isclose(x[0], Lx / 2.0, atol=atol_x),
        ),
    }
    try:
        return predicates[location]
    except KeyError as exc:
        raise ValueError(f"Unknown location '{location}'") from exc


_DOF_INDEX = {"x": 0, "y": 1}


def _locate_boundary_vertices(domain, predicate):
    domain.topology.create_connectivity(0, domain.topology.dim)
    return locate_entities_boundary(domain, 0, predicate)


def build_bcs_from_spec(spec, V, Lx, Ly, nelx=80, nely=50):
    """Build component-wise Dirichlet conditions from the structured spec."""
    if V.mesh.comm.size != 1:
        raise NotImplementedError("The verified configuration bridge is serial-only.")

    bcs = []
    domain = V.mesh
    fdim = domain.topology.dim - 1

    for bc_spec in spec.bcs:
        predicate = _make_predicate(bc_spec.location, Lx, Ly, nelx, nely)
        dof_idx = _DOF_INDEX[bc_spec.dof]

        if bc_spec.location in EDGE_LOCATIONS:
            entities = locate_entities_boundary(domain, fdim, predicate)
            entity_dim = fdim
        else:
            entities = _locate_boundary_vertices(domain, predicate)
            entity_dim = 0

        dofs = fem.locate_dofs_topological(V.sub(dof_idx), entity_dim, entities)
        if len(dofs) == 0:
            raise RuntimeError(
                f"BC '{bc_spec.location}' ({bc_spec.dof}) selected zero DOFs."
            )
        bc = fem.dirichletbc(
            PETSc.ScalarType(bc_spec.value), dofs, V.sub(dof_idx)
        )
        print(
            f"BC '{bc_spec.location}' dof='{bc_spec.dof}': "
            f"entities={len(entities)}, dofs={len(dofs)}"
        )
        bcs.append(bc)
    return bcs


def _edge_nodal_weights(coords: np.ndarray, location: str) -> np.ndarray:
    """Trapezoidal tributary weights for ordered nodes on a straight edge."""
    if len(coords) < 2:
        raise RuntimeError(f"Edge load '{location}' selected fewer than two nodes")
    axis = 1 if location in {"left_edge", "right_edge"} else 0
    order = np.argsort(coords[:, axis])
    s = coords[order, axis]
    ds = np.diff(s)
    if np.any(ds <= 0.0):
        raise RuntimeError("Duplicate or unordered edge-load coordinates")
    ordered = np.empty(len(s), dtype=float)
    ordered[0] = ds[0] / 2.0
    ordered[-1] = ds[-1] / 2.0
    if len(s) > 2:
        ordered[1:-1] = (ds[:-1] + ds[1:]) / 2.0
    weights = np.empty_like(ordered)
    weights[order] = ordered
    return weights


def build_load_from_spec(spec, V, Lx, Ly, nelx=80, nely=50):
    """Build an algebraic nodal force vector with explicit load semantics.

    ``point_force``: ``value`` is a discrete nodal point-force resultant. A center
    predicate may select two neighboring nodes on odd meshes; the force is
    divided equally.

    ``edge_resultant``: ``value`` is the total resultant on the selected edge,
    distributed using normalized trapezoidal tributary lengths.

    ``edge_traction``: ``value`` is force per unit length. Multiplication by
    tributary edge length gives each nodal force.
    """
    if V.mesh.comm.size != 1:
        raise NotImplementedError("The verified load builder is serial-only.")

    F = fem.Function(V)
    F.x.array[:] = 0.0

    for load_spec in spec.loads:
        predicate = _make_predicate(load_spec.location, Lx, Ly, nelx, nely)
        dof_idx = _DOF_INDEX[load_spec.dof]

        # Use the collapsed-subspace form so parent DOFs and geometric
        # coordinates remain explicitly paired. This avoids relying on raw
        # interleaving or on vertex/DOF arrays having the same ordering.
        V_sub, _ = V.sub(dof_idx).collapse()
        located = fem.locate_dofs_geometrical((V.sub(dof_idx), V_sub), predicate)
        parent_dofs = np.asarray(located[0], dtype=np.int32)
        sub_dofs = np.asarray(located[1], dtype=np.int32)
        if len(parent_dofs) == 0:
            raise RuntimeError(
                f"Load '{load_spec.location}' ({load_spec.dof}) selected zero DOFs."
            )

        if load_spec.kind == "point_force":
            if load_spec.location in EDGE_LOCATIONS:
                raise ValueError(
                    f"A point_force requires a point/center/corner location, not "
                    f"'{load_spec.location}'. Use edge_resultant or edge_traction."
                )
            nodal_values = np.full(
                len(parent_dofs), load_spec.value / len(parent_dofs), dtype=float
            )
        else:
            if load_spec.location not in EDGE_LOCATIONS:
                raise ValueError(
                    f"{load_spec.kind} requires a full edge location, not "
                    f"'{load_spec.location}'."
                )
            coords = V_sub.tabulate_dof_coordinates()[sub_dofs]
            tributary = _edge_nodal_weights(coords, load_spec.location)
            if load_spec.kind == "edge_resultant":
                nodal_values = load_spec.value * tributary / tributary.sum()
            elif load_spec.kind == "edge_traction":
                nodal_values = load_spec.value * tributary
            else:
                raise ValueError(f"Unsupported load kind '{load_spec.kind}'")

        for dof, value in zip(parent_dofs, nodal_values):
            F.x.array[dof] += value

        print(
            f"Load '{load_spec.location}' kind='{load_spec.kind}' "
            f"dof='{load_spec.dof}': dofs={len(parent_dofs)}, "
            f"resultant={float(np.sum(nodal_values)):.6g}"
        )

    F.x.scatter_forward()
    if not np.all(np.isfinite(F.x.array)) or np.count_nonzero(F.x.array) == 0:
        raise RuntimeError("The assembled nodal load vector is empty or non-finite")
    return F


def extract_simp_params(spec) -> dict:
    """Extract solver parameters without silently converting units."""
    nelx = spec.mesh.nx
    nely = spec.mesh.ny
    Lx = float(spec.mesh.Lx) if spec.mesh.Lx is not None else nelx / nely
    Ly = float(spec.mesh.Ly) if spec.mesh.Ly is not None else 1.0

    element_size = min(Lx / nelx, Ly / nely)
    r_min_elements = float(spec.simp.r_min) / element_size
    if r_min_elements < 1.0:
        raise ValueError(
            f"r_min={spec.simp.r_min:g} spans only {r_min_elements:.3f} elements; "
            "use at least one element width to regularize the density field."
        )

    return {
        "penal": float(spec.simp.penal),
        "volfrac": float(spec.simp.vol_frac),
        "r_min_convention": R_MIN_CONVENTION,
        "r_min": float(spec.simp.r_min),
        "r_pde": r_pde_from_r_min(float(spec.simp.r_min)),
        "r_min_elements": r_min_elements,
        "element_size": element_size,
        "E": float(spec.material.E),
        "nu": float(spec.material.nu),
        "formulation": spec.analysis.formulation,
        "unit_system": spec.analysis.unit_system,
        "thickness": float(spec.analysis.thickness),
        "edge_traction_definition": spec.analysis.edge_traction_definition,
        "nelx": nelx,
        "nely": nely,
        "Lx": Lx,
        "Ly": Ly,
        "max_iter": int(spec.simp.max_iter),
        "tol_change": float(spec.simp.tol_change),
    }
