import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import dolfinx.fem.petsc as fem_petsc
from dolfinx.io import XDMFFile

import matplotlib.pyplot as plt
import imageio

# ─────────────────────────────────────────────────────────────
# Mesh + Function Spaces
# ─────────────────────────────────────────────────────────────
def build_mesh(nelx, nely, Lx=1.0, Ly=1.0):
    # Structured rectangle mesh for SIMP topology optimization
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([Lx, Ly])],
        [nelx, nely],
        cell_type=mesh.CellType.quadrilateral,
    )

def build_spaces(domain):
    # V: vector space for displacement (2D)
    # Q: DG0 scalar space for density
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))  # 2D vector
    Q = fem.functionspace(domain, ("DG", 0))  # piecewise constant
    return V, Q

# ─────────────────────────────────────────────────────────────
# Boundary Conditions (Cantilever)
# ─────────────────────────────────────────────────────────────
def build_bcs(V):
    # Fix left edge (x=0) for cantilever
    def left(x): 
        return np.isclose(x[0], 0.0)
    dofs = fem.locate_dofs_geometrical(V, left)
    return [fem.dirichletbc(np.array([0.0, 0.0]), dofs, V)] #glue the left edge!

# ─────────────────────────────────────────────────────────────
# Elasticity
# ─────────────────────────────────────────────────────────────
def epsilon(u):
    # Symmetric gradient: strain tensor
    return ufl.sym(ufl.grad(u))

def sigma(u, mu, lmbda):
    # Linear elastic stress: σ = λ tr(ε) I + 2 μ ε
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2) + 2 * mu * epsilon(u)

def solve_elasticity(domain, V, bcs, rho_fn, penal): #computes how much it bends under load.
    # Solve linear elasticity given density rho_fn
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    E_min = 1e-9
    # SIMP penalization: E = E_min + (ρ^p)*(1-E_min) (that's what the internet said!)
    E = E_min + (1 - E_min) * rho_fn.x.array**penal

    # Map E to function for FEM coefficients
    E_fn = fem.Function(rho_fn.function_space)
    E_fn.x.array[:] = E

    mu0, lmbda0 = 1.0, 0.3
    mu = E_fn * mu0
    lmbda = E_fn * lmbda0

    # Bilinear form (stiffness)
    a = ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * ufl.dx

    # Load: downward force on free end
    T = fem.Constant(domain, np.array([0.0, -1.0]))
    L = ufl.inner(T, v) * ufl.ds

    # Solve linear system using PETSc
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
    return problem.solve()

# ─────────────────────────────────────────────────────────────
# Sensitivity
# ─────────────────────────────────────────────────────────────
def compute_sensitivity(domain, Q, rho_fn, uh, penal): #calculates how much compliance changes for each square
    # Compute compliance sensitivity (dC/dρ)
    rho = rho_fn.x.array
    sed = ufl.inner(sigma(uh, 1.0, 0.3), epsilon(uh))  # strain energy density
    v = ufl.TestFunction(Q)
    L = fem.form(sed * v * ufl.dx)
    vec = fem_petsc.assemble_vector(L)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ce = vec.array.copy()
    dc = -penal * rho**(penal - 1) * ce  # SIMP sensitivity formula

    return dc, ce

# ─────────────────────────────────────────────────────────────
# Helmholtz Filter
# ─────────────────────────────────────────────────────────────
def build_filter(domain, r_min):
    # Create Helmholtz filter matrix solver
    F = fem.functionspace(domain, ("Lagrange", 1))
    u, v = ufl.TrialFunction(F), ufl.TestFunction(F)
    a = (r_min**2 * ufl.dot(ufl.grad(u), ufl.grad(v)) + u*v) * ufl.dx
    A = fem_petsc.assemble_matrix(fem.form(a))
    A.assemble()

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")

    return solver, F

def apply_filter(rho_fn, solver, F): #helmholtz helps to prevent checkboarding. smooth density distribution.
    # Solve (r_min^2 Laplace + I) ρ_smooth = ρ
    u, v = ufl.TrialFunction(F), ufl.TestFunction(F)
    L = rho_fn * v * ufl.dx
    b = fem_petsc.assemble_vector(fem.form(L))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    rho_smooth = fem.Function(F)
    solver.solve(b, rho_smooth.x.petsc_vec)

    rho_filtered = fem.Function(rho_fn.function_space)
    expr = fem.Expression(rho_smooth, rho_fn.function_space.element.interpolation_points())
    rho_filtered.interpolate(expr)

    return rho_filtered

# ─────────────────────────────────────────────────────────────
# OC Update
# ─────────────────────────────────────────────────────────────
def oc_update(rho, dc, volfrac):
    # Optimality criteria density update. Using the sensitivities, we increase material where it helps most and remove material where it doesn’t.
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2 + 1e-8) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        rho_new = np.clip(
            rho * np.sqrt(np.maximum(-dc / lmid, 0)),
            1e-3, 1.0
        )
        if rho_new.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new

# ─────────────────────────────────────────────────────────────
# Main Solver
# ─────────────────────────────────────────────────────────────
def run_simp(nelx=60, nely=20, volfrac=0.4, penal=3.0, r_min=0.04, max_iter=100):
    domain = build_mesh(nelx, nely)
    V, Q = build_spaces(domain)
    bcs = build_bcs(V)

    # Initialize density function
    rho_fn = fem.Function(Q)
    rho_fn.x.array[:] = volfrac

    # Helmholtz filter
    filter_solver, F = build_filter(domain, r_min)

    # XDMF for Paraview output
    xdmf = XDMFFile(domain.comm, "density.xdmf", "w")
    xdmf.write_mesh(domain)

    # GIF frames
    frames = []

    for itr in range(max_iter):
        # Apply density filter
        rho_phys = apply_filter(rho_fn, filter_solver, F)

        # Solve elasticity problem
        uh = solve_elasticity(domain, V, bcs, rho_phys, penal)

        # Compute sensitivity
        dc, ce = compute_sensitivity(domain, Q, rho_phys, uh, penal)
        dc_fn = fem.Function(Q)
        dc_fn.x.array[:] = dc
        dc = apply_filter(dc_fn, filter_solver, F).x.array

        # OC density update
        rho_old = rho_fn.x.array.copy()
        rho_new = oc_update(rho_old, dc, volfrac)
        rho_fn.x.array[:] = rho_new

        # Convergence check
        change = np.linalg.norm(rho_new - rho_old)
        print(f"Iter {itr:3d} | change = {change:.3e}")
        if change < 1e-3:
            break

        rho_fn.name = "density"
        xdmf.write_function(rho_fn, itr)

        rho_grid = rho_fn.x.array.reshape((nely, nelx))
        plt.figure()
        plt.imshow(rho_grid, cmap="gray", origin="lower")
        plt.axis("off")
        plt.title(f"Iter {itr}")
        fname = f"_frame_{itr}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        frames.append(imageio.imread(fname))

    # Close XDMF file
    xdmf.close()

    # Save GIF
    imageio.mimsave("topopt.gif", frames, duration=0.2)

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_simp()