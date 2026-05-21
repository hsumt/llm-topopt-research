"""
_domain.py
Created on 5/1/26

Creates the material parameters (shear modulus and lmbda) and the basic stiffness equation
"""

import numpy as np

E_MIN = 1e-9 # p. 121 [1] 99line MATLAB: below EQ (1)


def simp_stiffness(rho: np.ndarray, penal: float) -> np.ndarray:
    """
    SIMP power-law interpolationa of Young's modulus.

    Equation [1, Eq.(1)]:
        E_e(rho_e) = E_min + rho_e^p * (E_0 - E_min)
    """
    return E_MIN + (1.0 - E_MIN) * rho**penal

def get_lame_parameters(E: float, nu: float):
    """
    Compute plane-stress Lamé parameters from E and nu.

    Equations [1, p.127 lines 88-89], [1.1, p.10 Eq (18)]:
        mu     = E / (2*(1 + nu))           shear modulus
        lambda = E*nu / (1 - nu^2)          first Lamé parameter (plane stress)
    → Cantilever problem setup: E = 1.0, nu = 0.3
    """
    mu     = E / (2.0 * (1.0 + nu))
    lmbda  = E * nu / (1.0 - nu**2)
    return mu, lmbda
