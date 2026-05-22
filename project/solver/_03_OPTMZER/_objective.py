"""
_objective.py
Created on 5/15/26

Creates the compliance objective
"""
import numpy as np
import ufl
from dolfinx import fem
from _02_FEA._assembly import sigma, epsilon

def compute_compliance(uh, a, L, bcs) -> float:
    from dolfinx.fem.petsc import assemble_vector
    from dolfinx.fem import form
    b = assemble_vector(form(L))
    # apply BCs to RHS
    return float(np.dot(b.array, uh.x.array))


    """
    Equation [1, Eq.(1)]:
        c(rho) = F^T * U  =  U^T * K(rho) * U 
    """
def compute_sensitivities(
    rho: np.ndarray,
    uh,
    Q,
    penal: float,
    mu: float,
    lmbda: float
) -> np.ndarray:
    
    strain_energy_expr = ufl.inner(sigma(uh, mu, lmbda), epsilon(uh))

    # Project UFL
    se_fn   = fem.Function(Q)
    se_expr = fem.Expression(
        strain_energy_expr,
        Q.element.interpolation_points()
    )
    se_fn.interpolate(se_expr)
    se = se_fn.x.array   

    # SIMP sensitivity chain rule  [1, Eq.(4)]
    dc = -penal * rho**(penal - 1.0) * se   

    return dc

    