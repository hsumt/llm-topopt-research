"""Penalized von Mises stress and its derivatives.

Holmberg et al. (2013) Sec. 4.2 and Sec. 5.  The von Mises function is
positively homogeneous of degree one in the stress components, so the stress
penalization eta_S(rho) = rho^(1/2) of Eq. (4) factors straight out:

    sigma_vm_e = eta_S(rho_e) * vm(sigma_hat_e).

That identity is used throughout; it is exact, not an approximation.
"""
from __future__ import annotations

import numpy as np

# Plane-stress von Mises quadratic form: vm^2 = s^T M s for s = [sx, sy, txy].
VM_M = np.array([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)

STRESS_PENALTY_EXPONENT = 0.5


def eta_S(rho: np.ndarray) -> np.ndarray:
    return np.power(rho, STRESS_PENALTY_EXPONENT)


def d_eta_S(rho: np.ndarray) -> np.ndarray:
    return STRESS_PENALTY_EXPONENT * np.power(rho, STRESS_PENALTY_EXPONENT - 1.0)


def von_mises(sigma_hat: np.ndarray) -> np.ndarray:
    """vm of the *unpenalized* centroid stress, shape (n_elem,)."""
    q = np.einsum("ei,ij,ej->e", sigma_hat, VM_M, sigma_hat)
    return np.sqrt(np.maximum(q, 0.0))


def d_von_mises(sigma_hat: np.ndarray, vm: np.ndarray, floor: float = 1.0e-12) -> np.ndarray:
    """d(vm)/d(sigma_hat), shape (n_elem, 3)."""
    safe = np.maximum(vm, floor)[:, None]
    return (sigma_hat @ VM_M) / safe


def penalized_von_mises(rho: np.ndarray, sigma_hat: np.ndarray) -> np.ndarray:
    return eta_S(rho) * von_mises(sigma_hat)
