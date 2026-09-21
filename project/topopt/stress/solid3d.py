"""2.5D extruded solid model: the L-bracket with an out-of-plane degree of freedom.

Why this module exists
----------------------
The plane-stress model in ``fem_q4.py`` follows Holmberg et al. (2013) and is the
right model for the paper's problem.  It is also structurally incapable of being
asked about out-of-plane loading: each node carries only ``u, v``, so a load
along z has no degree of freedom to act on.  That is not a weak answer, it is no
answer, and no predicate reading the in-plane stress field can detect it.

This module adds the missing dimension by **extrusion**: the same in-plane L-shaped
cell grid, swept through the thickness into ``n_layers`` of trilinear (H8) hex
elements, with the **density held constant through the thickness**.  Consequences,
all deliberate:

* the design variable count is unchanged -- one variable per in-plane cell, shared
  by every layer above it -- so the optimiser sees the same problem size and the
  2-D and 2.5D results are comparable cell by cell;
* every node carries ``u, v, w``, so a transverse load case is expressible;
* stress is recovered from the full 6-component 3-D state at each element
  centroid, so a von Mises measure that includes sigma_z, tau_yz and tau_zx exists.

With ``n_layers = 1`` and an in-plane load this reduces to a single-layer solid
that is *not* the same as plane stress (it is closer to plane strain in z through
the Poisson coupling), so ``verify_extrusion`` quantifies the difference rather
than assuming it away.

Known discretisation limit
--------------------------
Trilinear hexes with full 2x2x2 integration exhibit shear locking in transverse
bending when few elements span the thickness: the model is then **too stiff**,
which *underestimates* transverse displacement and stress.  That error is in the
non-conservative direction for detecting overstress, so a transverse finding from
a coarse layer count is a lower bound on the real severity.  Always report the
layer count alongside any transverse result, and refine layers to show the trend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- 3-D isotropic elasticity ------------------------------------------------

def elasticity_matrix_3d(E: float, nu: float) -> np.ndarray:
    """Isotropic D for strain order [ex, ey, ez, gxy, gyz, gzx] (engineering shear)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6), dtype=float)
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


# von Mises as a quadratic form on [sx, sy, sz, txy, tyz, tzx]:
#   vm^2 = 1/2[(sx-sy)^2 + (sy-sz)^2 + (sz-sx)^2] + 3(txy^2 + tyz^2 + tzx^2)
VM_M_3D = np.array(
    [
        [1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
        [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0],
        [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    ],
    dtype=float,
)

# H8 node ordering: bottom face counter-clockwise, then top face.
_H8_NAT = np.array(
    [
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)
_G = 1.0 / np.sqrt(3.0)


def _dN_natural(xi: float, eta: float, zeta: float) -> np.ndarray:
    """dN/d(xi,eta,zeta), shape (8, 3)."""
    s = _H8_NAT
    p = np.array([xi, eta, zeta], dtype=float)
    out = np.empty((8, 3), dtype=float)
    for a in range(8):
        sa = s[a]
        f = 1.0 + sa * p                      # (1+sa_i p_i) per axis
        out[a, 0] = 0.125 * sa[0] * f[1] * f[2]
        out[a, 1] = 0.125 * sa[1] * f[0] * f[2]
        out[a, 2] = 0.125 * sa[2] * f[0] * f[1]
    return out


def _B_at(coords: np.ndarray, xi: float, eta: float, zeta: float):
    """Strain-displacement matrix (6, 24) and det(J) at one natural point."""
    dN = _dN_natural(xi, eta, zeta)
    J = dN.T @ coords                          # (3,3)
    detJ = float(np.linalg.det(J))
    dN_xyz = dN @ np.linalg.inv(J)             # (8,3) derivatives wrt x,y,z
    B = np.zeros((6, 24), dtype=float)
    for a in range(8):
        dx, dy, dz = dN_xyz[a]
        c = 3 * a
        B[0, c + 0] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c + 0] = dy; B[3, c + 1] = dx
        B[4, c + 1] = dz; B[4, c + 2] = dy
        B[5, c + 0] = dz; B[5, c + 2] = dx
    return B, detJ


def h8_element_stiffness(coords: np.ndarray, *, E: float, nu: float) -> np.ndarray:
    """H8 stiffness by full 2x2x2 Gauss integration."""
    D = elasticity_matrix_3d(E, nu)
    K = np.zeros((24, 24), dtype=float)
    for xi in (-_G, _G):
        for eta in (-_G, _G):
            for zeta in (-_G, _G):
                B, detJ = _B_at(coords, xi, eta, zeta)
                if detJ <= 0.0:
                    raise ValueError(f"non-positive Jacobian determinant {detJ}")
                K += B.T @ D @ B * detJ
    return K


def unit_hex(hx: float, hy: float, hz: float) -> np.ndarray:
    """Corner coordinates of one brick, in the H8 node order."""
    return np.array(
        [
            [0, 0, 0], [hx, 0, 0], [hx, hy, 0], [0, hy, 0],
            [0, 0, hz], [hx, 0, hz], [hx, hy, hz], [0, hy, hz],
        ],
        dtype=float,
    )


# --- mesh --------------------------------------------------------------------

@dataclass
class Mesh3D:
    """Extruded L-bracket. ``design_of_elem`` maps each element to its in-plane cell."""

    coords: np.ndarray                 # (n_nodes, 3)
    edof: np.ndarray                   # (n_elem, 24)
    centroids: np.ndarray              # (n_elem, 3)
    inplane_centroids: np.ndarray      # (n_inplane, 2) -- filter operates on these
    design_of_elem: np.ndarray         # (n_elem,) -> in-plane cell index
    fixed_dofs: np.ndarray
    designable: np.ndarray             # (n_inplane,) bool
    L: float
    arm: float
    h: float
    thickness: float
    n_layers: int
    n_dof: int = field(init=False)

    def __post_init__(self):
        self.n_dof = 3 * self.coords.shape[0]

    @property
    def n_elem(self) -> int:
        return self.edof.shape[0]

    @property
    def n_inplane(self) -> int:
        return self.inplane_centroids.shape[0]

    @property
    def element_volume(self) -> float:
        return self.h * self.h * (self.thickness / self.n_layers)

    def reentrant_corner(self) -> np.ndarray:
        return np.array([self.arm, self.arm], dtype=float)

    def load_point(self) -> np.ndarray:
        return np.array([self.L, self.arm], dtype=float)


def build_lbracket_3d(
    *,
    L: float = 200.0,
    arm_fraction: float = 0.4,
    thickness: float = 1.0,
    n_cells_per_side: int = 40,
    n_layers: int = 2,
    corner_fillet_radius: float = 0.0,
    exclude_load_elements: bool = True,
    exclude_nx: int = 3,
    exclude_ny: int = 2,
) -> Mesh3D:
    n = int(n_cells_per_side)
    h = L / n
    arm = arm_fraction * L
    if abs(arm_fraction * n - round(arm_fraction * n)) > 1e-9:
        raise ValueError(
            f"n_cells_per_side={n} incompatible with arm_fraction={arm_fraction}: "
            f"the arm spans {arm_fraction*n:.4f} elements (needs an integer)"
        )
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    hz = thickness / n_layers

    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    cx = (ix + 0.5) * h
    cy = (iy + 0.5) * h
    inside = ~((cx > arm) & (cy > arm))
    if corner_fillet_radius > 0.0:
        d = np.hypot(cx - arm, cy - arm)
        cut = (cx > arm - corner_fillet_radius) & (cy > arm - corner_fillet_radius)
        cut &= d < corner_fillet_radius
        cut &= ~((cx > arm) & (cy > arm))
        inside = inside & ~cut

    cells = np.argwhere(inside)                # rows of (iy, ix)
    ciy, cix = cells[:, 0], cells[:, 1]
    n_inplane = cells.shape[0]
    inplane_centroids = np.stack([cix * h + h / 2.0, ciy * h + h / 2.0], axis=1)

    nnx = n + 1
    def gnode(i, j, k):
        return k * nnx * nnx + j * nnx + i

    corner_offsets = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                      (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    conn_blocks, design_blocks, centroid_blocks = [], [], []
    for k in range(n_layers):
        cols = [gnode(cix + dx, ciy + dy, k + dz) for dx, dy, dz in corner_offsets]
        conn_blocks.append(np.stack(cols, axis=1))
        design_blocks.append(np.arange(n_inplane))
        centroid_blocks.append(
            np.stack([inplane_centroids[:, 0], inplane_centroids[:, 1],
                      np.full(n_inplane, (k + 0.5) * hz)], axis=1)
        )
    corners = np.concatenate(conn_blocks, axis=0)
    design_of_elem = np.concatenate(design_blocks)
    centroids = np.concatenate(centroid_blocks, axis=0)

    used = np.unique(corners)
    remap = -np.ones(nnx * nnx * (n_layers + 1), dtype=np.int64)
    remap[used] = np.arange(used.size)
    conn = remap[corners]

    kk, rem = np.divmod(used, nnx * nnx)
    jj, ii = np.divmod(rem, nnx)
    coords = np.stack([ii * h, jj * h, kk * hz], axis=1).astype(float)

    edof = np.empty((conn.shape[0], 24), dtype=np.int64)
    for a in range(8):
        edof[:, 3 * a + 0] = 3 * conn[:, a]
        edof[:, 3 * a + 1] = 3 * conn[:, a] + 1
        edof[:, 3 * a + 2] = 3 * conn[:, a] + 2

    tol = 1e-9
    top = np.isclose(coords[:, 1], L, atol=tol) & (coords[:, 0] <= arm + tol)
    fixed_nodes = np.flatnonzero(top)
    if fixed_nodes.size == 0:
        raise RuntimeError("clamped edge selected no nodes")
    fixed_dofs = np.sort(np.concatenate(
        [3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2]))

    designable = np.ones(n_inplane, dtype=bool)
    if exclude_load_elements and exclude_nx > 0 and exclude_ny > 0:
        x_lo = L - exclude_nx * h - 1e-9
        y_lo = arm - exclude_ny * h - 1e-9
        patch = ((inplane_centroids[:, 0] >= x_lo)
                 & (inplane_centroids[:, 1] >= y_lo)
                 & (inplane_centroids[:, 1] <= arm))
        designable[patch] = False

    return Mesh3D(coords=coords, edof=edof, centroids=centroids,
                  inplane_centroids=inplane_centroids, design_of_elem=design_of_elem,
                  fixed_dofs=fixed_dofs, designable=designable,
                  L=L, arm=arm, h=h, thickness=thickness, n_layers=n_layers)


# --- system ------------------------------------------------------------------

class H8System:
    """Assembles and solves K(rho) u = F for the extruded mesh."""

    def __init__(self, mesh: Mesh3D, *, E: float, nu: float):
        self.mesh = mesh
        hz = mesh.thickness / mesh.n_layers
        self.ke = h8_element_stiffness(unit_hex(mesh.h, mesh.h, hz), E=E, nu=nu)
        self.B, _ = _B_at(unit_hex(mesh.h, mesh.h, hz), 0.0, 0.0, 0.0)
        self.D = elasticity_matrix_3d(E, nu)
        self.DB = self.D @ self.B                      # (6, 24)
        self._rows = np.repeat(mesh.edof, 24, axis=1).ravel()
        self._cols = np.tile(mesh.edof, (1, 24)).ravel()
        self._ke_flat = self.ke.ravel()
        self.free = np.setdiff1d(np.arange(mesh.n_dof), mesh.fixed_dofs)
        self._lu = None

    def factorize(self, scale: np.ndarray):
        data = (scale[:, None] * self._ke_flat[None, :]).ravel()
        K = sp.coo_matrix((data, (self._rows, self._cols)),
                          shape=(self.mesh.n_dof, self.mesh.n_dof)).tocsc()
        self._lu = spla.splu(K[self.free, :][:, self.free].tocsc())
        return K

    def solve(self, F: np.ndarray) -> np.ndarray:
        if self._lu is None:
            raise RuntimeError("factorize() must be called before solve()")
        u = np.zeros(self.mesh.n_dof, dtype=float)
        u[self.free] = self._lu.solve(F[self.free])
        return u

    def solid_stress(self, u: np.ndarray) -> np.ndarray:
        """Unpenalized centroid stress, shape (n_elem, 6)."""
        return u[self.mesh.edof] @ self.DB.T

    def element_strain_energy_density(self, u: np.ndarray) -> np.ndarray:
        ue = u[self.mesh.edof]
        return np.einsum("ei,ij,ej->e", ue, self.ke, ue)


def von_mises_3d(sigma: np.ndarray) -> np.ndarray:
    q = np.einsum("ei,ij,ej->e", sigma, VM_M_3D, sigma)
    return np.sqrt(np.maximum(q, 0.0))


def d_von_mises_3d(sigma: np.ndarray, vm: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return (sigma @ VM_M_3D) / np.maximum(vm, floor)[:, None]


DIRECTION_AXIS = {"x": 0, "y": 1, "z": 2}


def build_force_3d(mesh: Mesh3D, magnitude: float, distribute_nodes: int,
                   direction: str) -> np.ndarray:
    """Tip load in a named axis, spread over the through-thickness node column.

    ``distribute_nodes`` counts *in-plane* nodes down the right edge, matching the
    2-D convention; the load is always shared across the thickness so that the
    resultant equals ``magnitude`` regardless of ``n_layers``.
    """
    if direction not in DIRECTION_AXIS:
        raise ValueError(f"unknown load direction {direction!r}; use x, y or z")
    axis = DIRECTION_AXIS[direction]
    tol = 1e-9
    on_edge = np.isclose(mesh.coords[:, 0], mesh.L, atol=tol) & (mesh.coords[:, 1] <= mesh.arm + tol)
    nodes = np.flatnonzero(on_edge)
    ys = np.unique(np.round(mesh.coords[nodes, 1], 9))[::-1]
    k = max(1, int(distribute_nodes))
    if k > ys.size:
        raise ValueError(f"cannot distribute over {k} in-plane nodes; edge has {ys.size}")
    chosen = nodes[np.isin(np.round(mesh.coords[nodes, 1], 9), ys[:k])]
    F = np.zeros(mesh.n_dof, dtype=float)
    F[3 * chosen + axis] = -float(magnitude) / chosen.size
    return F
