"""
_boundaries.py
BCs and loads for the hardcoded main() benchmark path.
For the spec-driven path, see _config_bridge.py.

CHANGED (Bug #3, Task 1.1):
  build_load: switched from fragile index arithmetic to V.sub(1) subspace
  approach. Changed load location from bottom-right tip to center-right.

CHANGED (Bug #9, Task 1.2/1.3):
  build_bcs_mbb, build_bcs_michell: replaced locate_entities_boundary +
  locate_dofs_topological with the collapsed-subspace form of
  locate_dofs_geometrical.

  Two constraints apply simultaneously in this DOLFINx version:

  (a) locate_dofs_topological requires facet->DOF connectivity to exist
      before the call. domain.topology.create_entities() is called AFTER
      BCs are constructed in _run_main(), so topological lookup silently
      returned empty DOF arrays (BC[0] n_dofs=2 instead of 41), leaving
      the structure under-constrained → singular K → compliance ~2.7e6.

  (b) locate_dofs_geometrical(V.sub(i), predicate) raises
      "Cannot tabulate coordinates for a FunctionSpace" — subspaces do
      not support coordinate tabulation directly.

  The correct pattern for subspace BCs is the two-argument collapsed form:
      V_sub, _ = V.sub(i).collapse()
      dofs_raw = fem.locate_dofs_geometrical((V.sub(i), V_sub), predicate)
      parent_dofs = dofs_raw[0]
      bc = fem.dirichletbc(value, parent_dofs, V.sub(i))

  This is identical to the pattern already used in all build_load_*
  functions and is robust to call order (no topology dependency).

CHANGED (Task 1.2/1.3):
  build_load_mbb, build_load_michell: replaced 2*i+1 raw DOF index
  arithmetic with V.sub(1).collapse() + locate_dofs_geometrical. Also
  fixed point-load tolerances from 1e-10 to (L/n)*0.6.
"""

import numpy as np
from petsc4py import PETSc
from dolfinx import fem


# ---------------------------------------------------------------------------
# CANTILEVER  (Sigmund 2001, Section 2)
# ---------------------------------------------------------------------------
# Geometry: rectangle [0, Lx] x [0, Ly], 2:1 aspect ratio.
# BC: left edge (x=0) fully clamped — u_x = u_y = 0 at all left-edge nodes.
# Load: F_y = -1 at center of right edge (x=Lx, y=Ly/2).
# ---------------------------------------------------------------------------

def build_bcs(V, domain):
    """
    Cantilever: fully clamp the left edge (x = 0).

    Passes a [0, 0] vector value to locate_dofs_geometrical on the full
    space V — the only case where the single-argument form is valid.
    Fixing only u_x would leave vertical rigid-body translation → singular K.

    Reference: Sigmund (2001), Struct Multidisc Optim 21:120-127, Section 2.
    """
    def left_edge(x):
        return np.isclose(x[0], 0.0)

    dofs = fem.locate_dofs_geometrical(V, left_edge)
    bc   = fem.dirichletbc(
        np.array([0.0, 0.0], dtype=PETSc.ScalarType), dofs, V
    )
    return [bc]


def build_load(V, domain, Lx: float, Ly: float, nely: int = 50):
    """
    Point load F_y = -1 at center of right edge (x=Lx, y=Ly/2).

    Tolerance 60% of element height: finds one node (even nely) or two
    nodes with load split equally (odd nely). Both are physically correct.

    Reference: Sigmund (2001), center-load cantilever variant.
    """
    atol = (Ly / nely) * 0.6

    def right_center(x):
        return np.logical_and(
            np.isclose(x[0], Lx),
            np.isclose(x[1], Ly / 2.0, atol=atol)
        )

    F = fem.Function(V)
    F.x.array[:] = 0.0

    V_y, _      = V.sub(1).collapse()
    dofs_raw    = fem.locate_dofs_geometrical((V.sub(1), V_y), right_center)
    parent_dofs = dofs_raw[0]

    if len(parent_dofs) == 0:
        raise RuntimeError(
            f"build_load: no node found at right center "
            f"(x={Lx}, y={Ly/2:.4f} ± {atol:.4f}). "
            f"Use even nely for a guaranteed center node."
        )

    load_per_node = -1.0 / len(parent_dofs)
    for d in parent_dofs:
        F.x.array[d] += load_per_node

    F.x.scatter_forward()
    return F


# ---------------------------------------------------------------------------
# MBB BEAM  (Sigmund 2001, Section 3 — half-symmetry model)
# ---------------------------------------------------------------------------
# Geometry: rectangle [0, Lx] x [0, Ly], 3:1 aspect ratio (Lx=3, Ly=1).
# This model represents the RIGHT half of the full symmetric beam.
#
# BCs:
#   Left edge (x=0):            symmetry plane → u_x = 0 only.
#                                u_y is FREE — symmetry, not a wall.
#   Bottom-right corner (Lx,0): roller → u_y = 0 only.
#
# Load: F_y = -1 at top-left corner (x=0, y=Ly) — the midspan point.
#
# Expected DOF counts (nely=40):
#   BC[0] u_x=0 on left edge: 41 DOFs  (nely+1 nodes)
#   BC[1] u_y=0 at BR corner:  1 DOF
# ---------------------------------------------------------------------------

def build_bcs_mbb(V, domain, Lx: float, Ly: float):
    """
    MBB beam half-symmetry BCs.

    Uses collapsed-subspace locate_dofs_geometrical throughout.
    Single-argument form locate_dofs_geometrical(V.sub(i), pred) raises
    "Cannot tabulate coordinates for a FunctionSpace" in this DOLFINx
    version. The two-argument collapsed form is required for subspaces.

    Reference: Sigmund (2001), Struct Multidisc Optim 21:120-127, Section 3.
    """
    tol = 1e-10

    # --- BC 0: left edge u_x = 0 (symmetry plane) ---
    def left_edge(x):
        return np.isclose(x[0], 0.0, atol=tol)

    V_x, _      = V.sub(0).collapse()
    dofs_raw    = fem.locate_dofs_geometrical((V.sub(0), V_x), left_edge)


    print(f"DEBUG dofs_raw type:      {type(dofs_raw)}")
    print(f"DEBUG dofs_raw length:    {len(dofs_raw)}")
    print(f"DEBUG dofs_raw[0] length: {len(dofs_raw[0])}")
    print(f"DEBUG dofs_raw[0][:10]:   {dofs_raw[0][:10]}")
    print(f"DEBUG dofs_raw[1][:10]:   {dofs_raw[1][:10]}")

    
    parent_dofs = dofs_raw[0]
    if len(parent_dofs) == 0:
        raise RuntimeError("build_bcs_mbb: left edge u_x BC found 0 DOFs.")
    bc_left = fem.dirichletbc(PETSc.ScalarType(0.0), parent_dofs, V.sub(0))

    # --- BC 1: bottom-right corner u_y = 0 (roller) ---
    def bottom_right(x):
        return np.logical_and(
            np.isclose(x[0], Lx, atol=tol),
            np.isclose(x[1], 0.0, atol=tol)
        )

    V_y, _      = V.sub(1).collapse()
    dofs_raw    = fem.locate_dofs_geometrical((V.sub(1), V_y), bottom_right)
    parent_dofs = dofs_raw[0]
    if len(parent_dofs) == 0:
        raise RuntimeError("build_bcs_mbb: bottom-right u_y BC found 0 DOFs.")
    bc_br = fem.dirichletbc(PETSc.ScalarType(0.0), parent_dofs, V.sub(1))

    return [bc_left, bc_br]


def build_load_mbb(V, domain, Lx: float, Ly: float, nely: int):
    """
    MBB beam: point load F_y = -1 at top-left corner (x=0, y=Ly).

    Tolerance (Ly/nely)*0.6 — same convention as build_load (cantilever).

    Reference: Sigmund (2001), Section 3.
    """
    atol_y = (Ly / nely) * 0.6
    atol_x = 1e-10

    def top_left(x):
        return np.logical_and(
            np.isclose(x[0], 0.0, atol=atol_x),
            np.isclose(x[1], Ly,  atol=atol_y)
        )

    F = fem.Function(V)
    F.x.array[:] = 0.0

    V_y, _      = V.sub(1).collapse()
    dofs_raw    = fem.locate_dofs_geometrical((V.sub(1), V_y), top_left)
    parent_dofs = dofs_raw[0]

    if len(parent_dofs) == 0:
        raise RuntimeError(
            f"build_load_mbb: no node found at top-left corner "
            f"(x=0.0, y={Ly:.4f} ± {atol_y:.4f}). "
            f"Check nely={nely} and Ly={Ly}."
        )

    load_per_node = -1.0 / len(parent_dofs)
    for d in parent_dofs:
        F.x.array[d] += load_per_node

    F.x.scatter_forward()
    return F


# ---------------------------------------------------------------------------
# MICHELL ARCH  (Michell 1904 / Bendsoe & Sigmund 2003)
# ---------------------------------------------------------------------------
# Geometry: square domain [0, Lx] x [0, Ly], typically Lx = Ly = 2.0.
#
# BCs:
#   Bottom-left corner  (0,  0): full pin → u_x = 0, u_y = 0
#   Bottom-right corner (Lx, 0): full pin → u_x = 0, u_y = 0
#
# Load: F_y = -1 at top center (x=Lx/2, y=Ly).
#
# Expected result: symmetric fan truss. Left-right asymmetry → BC/load error.
#
# Expected DOF counts: 1 per component per corner → 4 BCs of 1 DOF each.
# ---------------------------------------------------------------------------

def build_bcs_michell(V, domain, Lx: float, Ly: float):
    tol = 1e-10
    tol_x = (Lx / 200) * 0.6  # ideally pass nelx instead of hardcoding 200
    cx = Lx / 2.0

    V_x, _ = V.sub(0).collapse()
    V_y, _ = V.sub(1).collapse()

    def bottom_left(x):
        return np.logical_and(
            np.isclose(x[0], 0.0, atol=tol),
            np.isclose(x[1], 0.0, atol=tol)
        )

    def bottom_right(x):
        return np.logical_and(
            np.isclose(x[0], Lx, atol=tol),
            np.isclose(x[1], 0.0, atol=tol)
        )

    def bottom_center(x):
        return np.logical_and(
            np.isclose(x[0], cx, atol=tol_x),
            np.isclose(x[1], 0.0, atol=tol)
        )

    dofs_bl_x = fem.locate_dofs_geometrical((V.sub(0), V_x), bottom_left)[0]
    dofs_bl_y = fem.locate_dofs_geometrical((V.sub(1), V_y), bottom_left)[0]
    dofs_br_x = fem.locate_dofs_geometrical((V.sub(0), V_x), bottom_right)[0]  # was missing
    dofs_br_y = fem.locate_dofs_geometrical((V.sub(1), V_y), bottom_right)[0]
    # dofs_bc_y — remove entirely

    return [
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_bl_x, V.sub(0)),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_bl_y, V.sub(1)),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_br_x, V.sub(0)),  # was missing
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_br_y, V.sub(1)),
    ]


def build_load_michell(V, domain, Lx: float, Ly: float, nelx: int, nely: int):
    """
    Michell arch: point load F_y = -1 at top center (x=Lx/2, y=Ly).

    Tolerances (L/n)*0.6 in both directions — same convention as build_load.

    Reference: Michell (1904); Bendsoe & Sigmund (2003), Chapter 1.
    """
    atol_x = (Lx / nelx) * 0.6
    atol_y = (Ly / nely) * 0.6
    cx     = Lx / 2.0

    def top_center(x):
        return np.logical_and(
            np.isclose(x[0], cx, atol=atol_x),
            np.isclose(x[1], Ly, atol=atol_y)
        )

    F = fem.Function(V)
    F.x.array[:] = 0.0

    V_y, _      = V.sub(1).collapse()
    dofs_raw    = fem.locate_dofs_geometrical((V.sub(1), V_y), top_center)
    parent_dofs = dofs_raw[0]

    if len(parent_dofs) == 0:
        raise RuntimeError(
            f"build_load_michell: no node found at top center "
            f"(x={cx:.4f} ± {atol_x:.4f}, y={Ly:.4f} ± {atol_y:.4f}). "
            f"Check nelx={nelx}, nely={nely}, Lx={Lx}, Ly={Ly}."
        )

    load_per_node = -1.0 / len(parent_dofs)
    for d in parent_dofs:
        F.x.array[d] += load_per_node

    F.x.scatter_forward()
    return F