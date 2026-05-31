"""
_boundaries.py
BCs and loads for the hardcoded main() benchmark path.
For the spec-driven path, see _config_bridge.py.

CHANGED (Bug #3, Task 1.1):
  build_load: switched from fragile index arithmetic to V.sub(1) subspace approach.
              Changed load location from bottom_right to right_center (task 1.1).
"""

import numpy as np
from petsc4py import PETSc
from dolfinx import fem


def build_bcs(V, domain):
    """
    Cantilever: fully clamp the left edge (x = 0, y = 0).

    Both DOF components must be fixed. Fixing only x leaves
    vertical rigid-body translation → singular K.

    Reference: Sigmund (2001), Struct Multidisc Optim 21:120-127, Section 2.
    """
    def left_edge(x):
        return np.isclose(x[0], 0.0)

    dofs = fem.locate_dofs_geometrical(V, left_edge)
    bc = fem.dirichletbc(
        np.array([0.0, 0.0], dtype=PETSc.ScalarType), dofs, V
    )
    return [bc]


def build_load(V, domain, Lx: float, Ly: float, nely: int = 50):
    """
    Point load: -y direction at the center of the right edge (Lx, Ly/2).
    Task 1.1: changed from bottom-right tip to center-right.

    Tolerance: half an element height ensures exactly one node is found
    when nely is even (standard for benchmarks). If nely is odd, the two
    nearest nodes are both captured and the load is split equally — this is
    physically correct (consistent nodal loading).

    Uses V.sub(1) subspace approach to get y-DOF parent indices unambiguously.
    No index arithmetic — avoids the DOF layout assumption.

    Reference: Sigmund (2001), center-load cantilever variant.
    """
    atol = (Ly / nely) * 0.6   # 60% of element height: finds center node(s) only

    def right_center(x):
        return np.logical_and(
            np.isclose(x[0], Lx),
            np.isclose(x[1], Ly / 2.0, atol=atol)
        )

    F = fem.Function(V)
    F.x.array[:] = 0.0

    # Collapse y-component subspace → get (subspace, sub→parent DOF map)
    V_y, _ = V.sub(1).collapse()

    # locate_dofs_geometrical with a (parent_sub, collapsed_sub) tuple
    # returns (parent_dofs, sub_dofs); parent_dofs are indices into V.x.array
    dofs_raw = fem.locate_dofs_geometrical((V.sub(1), V_y), right_center)
    parent_dofs = dofs_raw[0]

    if len(parent_dofs) == 0:
        raise RuntimeError(
            f"build_load: no node found at right center "
            f"(x={Lx}, y={Ly/2:.4f} ± {atol:.4f}). "
            f"Use even nely for guaranteed center node."
        )

    # Distribute unit load equally across all found nodes
    # (for even nely: exactly one node; for odd nely: two nodes split evenly)
    load_per_node = -1.0 / len(parent_dofs)
    for d in parent_dofs:
        F.x.array[d] += load_per_node

    F.x.scatter_forward()
    return F