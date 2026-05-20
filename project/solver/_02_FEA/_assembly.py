
"""
References: https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html
"""

import ufl
from dolfinx import fem
from _01_MSH._domain import simp_stiffness
# Strain tensor. Symmetric Gradient
def epsilon(u):
    return ufl.sym(
        ufl.grad(u)
    )  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

# Stress tensor: equivalent to Hooke's law in the terms of the Lame' parameters
def sigma(u, mu, lmbda):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)

def build_weak_form(V, rho_fn, penal, mu, lmbda, F_load):
    u = ufl.TrialFunction(V) #solving for displacement
    v = ufl.TestFunction(V) # test

    E_coeff = fem.Function(rho_fn.function_space) #young's modulus x coefficient
    E_coeff.x.array[:] = simp_stiffness(rho_fn.x.array, penal)

    # stiffness of material in a matrix. strain energy density. the higher E_coeff is the more stiff.
    a = E_coeff * ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * ufl.dx

    # load
    L = ufl.dot(F_load, v) * ufl.dx

    return a, L
