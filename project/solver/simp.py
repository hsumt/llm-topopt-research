import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import dolfinx.fem.petsc as fem_petsc
from dolfinx.io import XDMFFile

import matplotlib.pyplot as plt
import imageio

# E = 1.0,  nu = 0.2
# nelx = 80, nely = 50
# Lx = 1.6,  Ly = 1.0
# BC: fixed left edge
# Load: downward force at right edge middle
def build_mesh(nelx: int, nely: int, Lx: float, Ly: float):
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0,0]), np.array([Lx, Ly])],
        [nelx, nely],
        cell_type=mesh.CellType.quadrilateral,
    )
    return domain
def build_spaces(domain):
    V = fem.functionspace(domain, ("Lagrange", 1, (2,))) #displacement needs to be continuous (Lagrange) across the domain
    Q = fem.functionspace(domain, ("DG", 0)) #discontinuous Galerkin. density does not need to be continuous since every element has a separate density
    return V, Q
E_MIN = 1e-9

def simp_stiffness(rho: np.ndarray, penal: float) -> np.ndarray:
    # SIMP interpolation - Sigmund (2001) Eq. 1 "xe^p"
    # penal (p) penalizes intermediate densities: grey elements get cheap stiffness
    # but cost full material budget, driving the design toward black and white.
    return E_MIN + (1-E_MIN) * rho**penal

def get_lame_parameters(E: float, nu: float):
    # Plane stress formulas
    # p.127 ln 88 and 89 list E = 1, and nu = 0.3. YM and Poisson
    # mu = shear modulus 
    # lmbda (First Lame parameter) = couple normal stresses.
    mu = E / (2 * (1+nu))
    lmbda = E * nu / (1-nu**2)
    return mu, lmbda
def epsilon(u):
    # takes in strain
    return ufl.sym(ufl.grad(u))
def sigma(u, mu, lmbda):
    # takes in displacement, shear modulus, and the coupled normal stresses in order to calculate stress
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2) + 2 * mu * epsilon(u) # creates 2x2 identity matrix


def build_bcs(V, domain):
    def left_edge(x):
        return np.isclose(x[0], 0.0)
    
    dofs = fem.locate_dofs_geometrical(V, left_edge)
    bc = fem.dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), dofs, V)
    return [bc]

def build_load(V, domain, Lx: float, Ly: float):
    # point load of -1 in y at top right corner - Sigmund (2001) line 79
    def top_right(x):
        return np.logical_and(np.isclose(x[0], Lx), np.isclose(x[1], Ly))
    
    # find the DOFs at that point
    dofs = fem.locate_dofs_geometrical(V, top_right)
    

    F = fem.Function(V)
    F.x.array[:] = 0.0
    F.x.array[2 * dofs[0] + 1] = -1.0 #2n+1 for the y dof at node n

    return F


def solve_fea(domain, V, bcs, rho_fn, penal, mu, lmbda, F_load): # https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity.html
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E_coeff = fem.Function(rho_fn.function_space)
    E_coeff.x.array[:] = simp_stiffness(rho_fn.x.array, penal)

    a = ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * ufl.dx
    L = ufl.dot(F_load, v) * ufl.dx


    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "cg",
                                           "pc_type": "hypre"})
    uh = problem.solve()

    return uh

def oc_update(rho, dc, volfrac, nelx, nely): # Sigmund 2001 Appendix (p. 126)
    l1, l2 = 0.0, 1e5
    move = 0.2

    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l1 + l2)

        rho_new = np.maximum(
            0.001,
            np.maximum(
                rho - move,
                np.minimum(
                    1.0,
                    np.minimum(
                        rho + move,
                        rho * np.sqrt(-dc / lmid)
                    )
                )
            )
        )

        if rho_new.sum() - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid

    return rho_new



def main():
    nelx, nely = 80, 50
    Lx, Ly = 1.6, 1.0
    volfrac = 0.4
    penal = 3.0
    rmin = 0.05

    domain = build_mesh(nelx, nely, Lx, Ly)
    V, Q = build_spaces(domain)

    mu, lmbda = get_lame_parameters(1.0, 0.3)

    bcs = build_bcs(V, domain)
    F = build_load(V, domain, Lx, Ly)

