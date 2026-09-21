"""Structured Q4 finite element analysis for the stress-constrained problem.

Every element on the structured grid is geometrically identical, so the solid
element stiffness matrix and the centroid strain-displacement matrix are formed
once and reused.  The element stiffness routine is the repository's already
verified independent Q4 reference, so this module and the DOLFINx path share a
single definition of the element.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from project.verification.reference_q4 import plane_stress_matrix, q4_element_stiffness


def unit_cell_coords(h: float) -> np.ndarray:
    """Node coordinates of one square cell, node order BL, BR, TR, TL."""
    return np.array([[0.0, 0.0], [h, 0.0], [h, h], [0.0, h]], dtype=float)


def centroid_B(h: float) -> np.ndarray:
    """Strain-displacement matrix at the element centroid (xi = eta = 0)."""
    coords = unit_cell_coords(h)
    dN = 0.25 * np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float
    )
    J = dN.T @ coords
    dN_xy = dN @ np.linalg.inv(J)
    B = np.zeros((3, 8), dtype=float)
    for a in range(4):
        B[0, 2 * a] = dN_xy[a, 0]
        B[1, 2 * a + 1] = dN_xy[a, 1]
        B[2, 2 * a] = dN_xy[a, 1]
        B[2, 2 * a + 1] = dN_xy[a, 0]
    return B


class Q4System:
    """Assembles and solves K(rho) u = F on a fixed structured mesh."""

    def __init__(self, mesh, *, E: float, nu: float):
        self.mesh = mesh
        self.E = float(E)
        self.nu = float(nu)
        h = mesh.geom.h
        self.ke = q4_element_stiffness(
            unit_cell_coords(h), E=E, nu=nu, thickness=mesh.geom.thickness
        )
        self.B = centroid_B(h)
        self.D = plane_stress_matrix(E, nu)
        self.DB = self.D @ self.B                     # (3, 8) solid stress operator

        edof = mesh.edof
        self._rows = np.repeat(edof, 8, axis=1).ravel()
        self._cols = np.tile(edof, (1, 8)).ravel()
        self._ke_flat = self.ke.ravel()

        free = np.setdiff1d(np.arange(mesh.n_dof), mesh.fixed_dofs, assume_unique=False)
        self.free = free
        self._lu = None
        self._scale = None

    def assemble(self, scale: np.ndarray) -> sp.csc_matrix:
        data = (scale[:, None] * self._ke_flat[None, :]).ravel()
        K = sp.coo_matrix(
            (data, (self._rows, self._cols)),
            shape=(self.mesh.n_dof, self.mesh.n_dof),
        ).tocsc()
        return K

    def factorize(self, scale: np.ndarray):
        """Factorize K once; both the state and every adjoint solve reuse it."""
        K = self.assemble(scale)
        Kff = K[self.free, :][:, self.free]
        self._lu = spla.splu(Kff.tocsc())
        self._scale = scale
        return K

    def solve(self, F: np.ndarray) -> np.ndarray:
        if self._lu is None:
            raise RuntimeError("factorize() must be called before solve()")
        u = np.zeros(self.mesh.n_dof, dtype=float)
        u[self.free] = self._lu.solve(F[self.free])
        return u

    def element_displacements(self, u: np.ndarray) -> np.ndarray:
        return u[self.mesh.edof]                       # (n_elem, 8)

    def solid_stress(self, u: np.ndarray) -> np.ndarray:
        """Unpenalized centroid stress sigma_hat = E B u, shape (n_elem, 3)."""
        return self.element_displacements(u) @ self.DB.T

    def element_strain_energy_density(self, u: np.ndarray) -> np.ndarray:
        ue = self.element_displacements(u)
        return np.einsum("ei,ij,ej->e", ue, self.ke, ue)
