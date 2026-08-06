"""Compliance objective and SIMP sensitivities."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem import petsc as fem_petsc

from project.topopt.fem.domain import E_MIN
from project.topopt.fem.assembly import epsilon, sigma


def compute_compliance(uh, F_load) -> float:
    """Compute c = F^T U using the same algebraic nodal load vector as FEA."""
    value = float(np.dot(F_load.x.array, uh.x.array))
    if not np.isfinite(value):
        raise FloatingPointError("Compliance is non-finite")
    return value


def compute_sensitivities(
    rho: np.ndarray,
    uh,
    Q,
    penal: float,
    mu: float,
    lmbda: float,
    thickness: float = 1.0,
) -> np.ndarray:
    r"""Return element-integrated SIMP compliance derivatives.

    Equation
    --------
    dc/drho_e = -p (E0-Emin) rho_e^(p-1) integral_{Omega_e}
                epsilon(u):C0:epsilon(u) dOmega.

    Reference
    ---------
    Sigmund (2001), Eq. (4); Bendsøe & Sigmund (2003), Eq. (1.8).

    The DG-0 test function makes each assembled vector entry the *cell
    integral* of the reference strain-energy density. Centroid interpolation
    is not used because it omits the element measure and is not exact for Q1
    quadrilateral strain-energy fields.
    """
    q = ufl.TestFunction(Q)
    energy_form = fem.form(
        ufl.inner(sigma(uh, mu, lmbda), epsilon(uh)) * q * ufl.dx
    )
    energy_vec = fem_petsc.assemble_vector(energy_form)
    energy_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    elemental_energy = energy_vec.array.copy()
    energy_vec.destroy()

    if thickness <= 0.0:
        raise ValueError("thickness must be positive")

    dc = (
        -float(thickness)
        * penal
        * (1.0 - E_MIN)
        * np.power(rho, penal - 1.0)
        * elemental_energy
    )
    if not np.all(np.isfinite(dc)):
        raise FloatingPointError("Non-finite compliance sensitivities")
    return dc
