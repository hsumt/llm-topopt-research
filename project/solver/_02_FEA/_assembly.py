"""Deterministic finite-element stiffness assembly for 2-D linear elasticity."""

import ufl
from dolfinx import fem

from _01_MSH._domain import simp_stiffness


def epsilon(u):
    """Small-strain tensor: epsilon(u) = sym(grad(u))."""
    return ufl.sym(ufl.grad(u))


def sigma(u, mu, lmbda):
    """Plane-stress constitutive tensor with unit reference Young's modulus."""
    return (
        lmbda * ufl.nabla_div(u) * ufl.Identity(len(u))
        + 2.0 * mu * epsilon(u)
    )


def build_stiffness_form(V, rho_fn, penal, mu, lmbda):
    """Return the SIMP-weighted bilinear stiffness form.

    The external load is *not* represented as a UFL body-force integral. Point
    and edge-resultant loads are assembled directly into a nodal PETSc vector in
    ``_solver.py``. This keeps the load vector used by the equilibrium equation
    identical to the vector used in the compliance objective F^T U.
    """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E_coeff = fem.Function(rho_fn.function_space)
    E_coeff.x.array[:] = simp_stiffness(rho_fn.x.array, penal)
    E_coeff.x.scatter_forward()

    return (
        E_coeff
        * ufl.inner(sigma(u, mu, lmbda), epsilon(v))
        * ufl.dx
    )
