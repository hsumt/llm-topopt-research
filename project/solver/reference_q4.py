from __future__ import annotations

import numpy as np


E_MIN = 1.0e-9


def plane_stress_matrix(E: float, nu: float) -> np.ndarray:
    if E <= 0.0 or not (-1.0 < nu < 0.5):
        raise ValueError("invalid plane-stress material parameters")
    return E / (1.0 - nu**2) * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ],
        dtype=float,
    )


def _shape_derivatives(xi: float, eta: float) -> np.ndarray:
    """Return dN/d(xi,eta) for node order BL, BR, TR, TL."""
    return 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [+(1.0 - eta), -(1.0 + xi)],
            [+(1.0 + eta), +(1.0 + xi)],
            [-(1.0 + eta), +(1.0 - xi)],
        ],
        dtype=float,
    )


def q4_element_stiffness(
    coords: np.ndarray,
    *,
    E: float,
    nu: float,
    thickness: float = 1.0,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError("coords must have shape (4,2)")
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    D = plane_stress_matrix(E, nu)
    ke = np.zeros((8, 8), dtype=float)
    gp = 1.0 / np.sqrt(3.0)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            dN_parent = _shape_derivatives(xi, eta)
            J = dN_parent.T @ coords
            detJ = float(np.linalg.det(J))
            if detJ <= 0.0:
                raise ValueError("non-positive Q4 Jacobian")
            dN_xy = dN_parent @ np.linalg.inv(J)
            B = np.zeros((3, 8), dtype=float)
            for a in range(4):
                dN_dx, dN_dy = dN_xy[a]
                B[0, 2 * a] = dN_dx
                B[1, 2 * a + 1] = dN_dy
                B[2, 2 * a] = dN_dy
                B[2, 2 * a + 1] = dN_dx
            ke += float(thickness) * (B.T @ D @ B) * detJ
    return ke


def structured_q4_cantilever(
    *,
    nelx: int,
    nely: int,
    Lx: float,
    Ly: float,
    rho: np.ndarray,
    penal: float,
    E: float = 1.0,
    nu: float = 0.3,
    thickness: float = 1.0,
    load_value: float = -1.0,
) -> dict:
    """Solve a fixed-density clamped cantilever with a center-right nodal load."""
    if nelx <= 0 or nely <= 0 or Lx <= 0.0 or Ly <= 0.0:
        raise ValueError("invalid structured mesh")
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (nely, nelx):
        raise ValueError(f"rho shape {rho.shape}; expected {(nely, nelx)}")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho contains non-finite values")

    nnx = nelx + 1
    nny = nely + 1
    n_nodes = nnx * nny
    coords = np.zeros((n_nodes, 2), dtype=float)
    for j in range(nny):
        for i in range(nnx):
            node = j * nnx + i
            coords[node] = [Lx * i / nelx, Ly * j / nely]

    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    for j in range(nely):
        for i in range(nelx):
            nodes = np.array(
                [
                    j * nnx + i,
                    j * nnx + i + 1,
                    (j + 1) * nnx + i + 1,
                    (j + 1) * nnx + i,
                ],
                dtype=int,
            )
            dofs = np.column_stack((2 * nodes, 2 * nodes + 1)).ravel()
            scale = E_MIN + (1.0 - E_MIN) * float(rho[j, i]) ** float(penal)
            ke = scale * q4_element_stiffness(
                coords[nodes], E=E, nu=nu, thickness=thickness
            )
            K[np.ix_(dofs, dofs)] += ke

    F = np.zeros(ndof, dtype=float)
    if nely % 2 == 0:
        load_nodes = [int((nely // 2) * nnx + nelx)]
    else:
        lower = nely // 2
        load_nodes = [lower * nnx + nelx, (lower + 1) * nnx + nelx]
    for node in load_nodes:
        F[2 * node + 1] += float(load_value) / len(load_nodes)

    fixed = []
    for j in range(nny):
        node = j * nnx
        fixed.extend([2 * node, 2 * node + 1])
    fixed = np.asarray(sorted(fixed), dtype=int)
    all_dofs = np.arange(ndof, dtype=int)
    free = np.setdiff1d(all_dofs, fixed, assume_unique=True)

    U = np.zeros(ndof, dtype=float)
    U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    residual = K @ U - F
    reactions = residual[fixed]
    compliance = float(F @ U)
    strain_energy = float(U @ K @ U)
    return {
        "displacement": U,
        "compliance": compliance,
        "strain_energy": strain_energy,
        "work_energy_relative_error": abs(compliance - strain_energy)
        / max(abs(compliance), abs(strain_energy), 1.0e-30),
        "reaction_x": float(np.sum(reactions[0::2])),
        "reaction_y": float(np.sum(reactions[1::2])),
        "load_resultant_x": float(np.sum(F[0::2])),
        "load_resultant_y": float(np.sum(F[1::2])),
    }
