"""
_objective.py
Created on 5/15/26

Creates the compliance objective
"""
import numpy as np
import ufl
from dolfinx import fem
from _02_FEA._assembly import sigma, epsilon

def compute_compliance(uh, F_load) -> float:
    """
    Equation [1, Eq.(1)]:
        c(rho) = F^T * U  =  U^T * K(rho) * U 
    """
    return float(np.dot(F_load.x.array, uh.x.array))
def compute_sensitivities(
    rho: np.ndarray,
    uh,
    Q,
    penal: float,
    mu: float,
    lmbda: float
) -> np.ndarray:
    """
    Equation: dc/dρ_e = -p * ρ_e^(p-1) * u_e^T * k_0 * u_e
    Reference: Sigmund (2001), Eq.(4); Bendsøe & Sigmund (2003), Eq.(1.8)

    strain_energy_expr = ε(u) : σ(u) per unit volume — the elemental
    strain energy density. For DG-0, one DOF per cell, so interpolation
    collapses to a single point per element (the cell centroid).
    """
    strain_energy_expr = ufl.inner(sigma(uh, mu, lmbda), epsilon(uh))

    se_fn   = fem.Function(Q)
    se_expr = fem.Expression(
        strain_energy_expr,
        Q.element.interpolation_points()   # ← FIXED: call the method
    )
    se_fn.interpolate(se_expr)
    se = se_fn.x.array

    # SIMP sensitivity chain rule — Sigmund (2001) Eq.(4)
    dc = -penal * rho**(penal - 1.0) * se

    return dc

    