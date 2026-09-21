"""L-shaped design domain, meshed as a structured Q4 grid.

Geometry follows Holmberg et al. (2013) Fig. 4: an L of overall extent L x L
whose two arms are each 2L/5 wide.  The vertical arm is clamped along its top
edge; a downward point load acts at the top-right corner of the horizontal arm.
The re-entrant corner carries an initial geometric stress singularity, and the
paper deliberately places no radius there ("the design domain should be easy to
create and simple to mesh").

The void quadrant is not part of the design domain at all -- those cells are
never meshed, rather than being meshed and held at a low density.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LBracketGeometry:
    """Holmberg Fig. 4 L-beam. Lengths in mm."""

    L: float = 200.0
    arm_fraction: float = 0.4          # 2L/5, both arms
    thickness: float = 1.0
    n_cells_per_side: int = 40         # bounding-grid resolution (low fidelity)
    exclude_nx: int = 3                # design-space exclusion under the load,
    exclude_ny: int = 2                # 3 x 2 elements (Sec. 8.1, Fig. 3)

    @property
    def arm(self) -> float:
        return self.arm_fraction * self.L

    @property
    def h(self) -> float:
        return self.L / self.n_cells_per_side


@dataclass
class Mesh:
    """Structured Q4 mesh restricted to the L-shaped domain."""

    geom: LBracketGeometry
    coords: np.ndarray                 # (n_nodes, 2)
    edof: np.ndarray                   # (n_elem, 8) global dof indices
    centroids: np.ndarray              # (n_elem, 2)
    fixed_dofs: np.ndarray
    load_dofs: np.ndarray
    load_values: np.ndarray
    designable: np.ndarray             # (n_elem,) bool -- False = excluded, held solid
    n_dof: int = field(init=False)

    def __post_init__(self):
        self.n_dof = 2 * self.coords.shape[0]

    @property
    def n_elem(self) -> int:
        return self.edof.shape[0]

    @property
    def element_area(self) -> float:
        return self.geom.h ** 2

    @property
    def element_volume(self) -> float:
        return self.element_area * self.geom.thickness

    def force_vector(self) -> np.ndarray:
        F = np.zeros(self.n_dof, dtype=float)
        np.add.at(F, self.load_dofs, self.load_values)
        return F

    def reentrant_corner(self) -> np.ndarray:
        return np.array([self.geom.arm, self.geom.arm], dtype=float)

    def load_point(self) -> np.ndarray:
        return np.array([self.geom.L, self.geom.arm], dtype=float)


def build_lbracket(
    geom: LBracketGeometry | None = None,
    *,
    corner_fillet_radius: float = 0.0,
    exclude_load_elements: bool = True,
) -> Mesh:
    """Mesh the L-shaped domain.

    ``corner_fillet_radius`` removes cells whose centroid lies inside the
    re-entrant corner but outside a quarter-circle of that radius -- i.e. it
    declares a fillet in the *design domain*, which is a specification edit
    rather than a solver setting.  Zero reproduces the paper's sharp corner.
    """
    geom = geom or LBracketGeometry()
    n = geom.n_cells_per_side
    h = geom.h
    arm = geom.arm

    # The arm boundary and the load point must fall on node lines, otherwise the
    # re-entrant corner is not a mesh vertex and the load has no node to act on.
    cells_per_arm = geom.arm_fraction * n
    if abs(cells_per_arm - round(cells_per_arm)) > 1.0e-9:
        raise ValueError(
            f"n_cells_per_side={n} is incompatible with arm_fraction="
            f"{geom.arm_fraction}: the arm spans {cells_per_arm:.4f} elements. "
            f"Choose n so that arm_fraction*n is an integer "
            f"(for arm_fraction=0.4, any multiple of 5)."
        )

    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    cx = (ix + 0.5) * h
    cy = (iy + 0.5) * h

    inside_L = ~((cx > arm) & (cy > arm))

    if corner_fillet_radius > 0.0:
        # Material is removed from the *inner* side of the corner: cells in the
        # quadrant beyond the corner and within the fillet radius are cut away.
        corner = np.array([arm, arm])
        d = np.hypot(cx - corner[0], cy - corner[1])
        cut = (cx > arm - corner_fillet_radius) & (cy > arm - corner_fillet_radius)
        cut &= d < corner_fillet_radius
        cut &= ~((cx > arm) & (cy > arm))
        inside_L = inside_L & ~cut

    active = inside_L
    cell_ids = np.argwhere(active)          # rows of (iy, ix)
    cell_iy = cell_ids[:, 0]
    cell_ix = cell_ids[:, 1]
    n_elem = cell_ids.shape[0]

    # Global node numbering on the bounding grid, then compress to used nodes.
    nnx = n + 1
    def gnode(i, j):
        return j * nnx + i

    corners = np.stack(
        [
            gnode(cell_ix, cell_iy),          # BL
            gnode(cell_ix + 1, cell_iy),      # BR
            gnode(cell_ix + 1, cell_iy + 1),  # TR
            gnode(cell_ix, cell_iy + 1),      # TL
        ],
        axis=1,
    )
    used = np.unique(corners)
    remap = -np.ones((n + 1) * (n + 1), dtype=np.int64)
    remap[used] = np.arange(used.size)
    conn = remap[corners]

    gj, gi = np.divmod(used, nnx)
    coords = np.stack([gi * h, gj * h], axis=1).astype(float)

    edof = np.empty((n_elem, 8), dtype=np.int64)
    edof[:, 0::2] = 2 * conn
    edof[:, 1::2] = 2 * conn + 1

    centroids = np.stack([cell_ix * h + h / 2.0, cell_iy * h + h / 2.0], axis=1)

    # Clamp: top edge of the vertical arm (y = L, x <= arm).
    tol = 1.0e-9
    top = np.isclose(coords[:, 1], geom.L, atol=tol) & (coords[:, 0] <= arm + tol)
    fixed_nodes = np.flatnonzero(top)
    if fixed_nodes.size == 0:
        raise RuntimeError("clamped edge selected no nodes")
    fixed_dofs = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))

    # Load: downward point force at the top-right corner of the horizontal arm.
    load_node = np.flatnonzero(
        np.isclose(coords[:, 0], geom.L, atol=tol) & np.isclose(coords[:, 1], arm, atol=tol)
    )
    if load_node.size != 1:
        raise RuntimeError(f"load point selected {load_node.size} nodes; expected 1")
    load_dofs = np.array([2 * load_node[0] + 1], dtype=np.int64)
    load_values = np.array([1.0], dtype=float)   # scaled by the spec load magnitude

    designable = np.ones(n_elem, dtype=bool)
    if exclude_load_elements and geom.exclude_nx > 0 and geom.exclude_ny > 0:
        x_lo = geom.L - geom.exclude_nx * h - 1.0e-9
        y_lo = arm - geom.exclude_ny * h - 1.0e-9
        patch = (centroids[:, 0] >= x_lo) & (centroids[:, 1] >= y_lo) & (centroids[:, 1] <= arm)
        designable[patch] = False

    return Mesh(
        geom=geom,
        coords=coords,
        edof=edof,
        centroids=centroids,
        fixed_dofs=fixed_dofs,
        load_dofs=load_dofs,
        load_values=load_values,
        designable=designable,
    )
