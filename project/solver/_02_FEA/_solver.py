"""
_solver.py Implemented 5/14/26
Initializes the linear problem to solve Finite Element Analysis
"""

from dolfinx.fem.petsc import LinearProblem
from _02_FEA._assembly import build_weak_form

def solve_fea(domain, V, bcs, rho_fn, penal, mu, lmbda, F_load):
    a, L = build_weak_form(V, rho_fn, penal, mu, lmbda, F_load)
    # DOLFINx 0.10.0: petsc_options_prefix is required; keys in petsc_options
    # are unprefixed — PETSc prepends the prefix automatically.
    problem = LinearProblem(
        a, L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": 1e-6,
            "ksp_atol": 1e-10,
            "ksp_max_it": 1000,
            "pc_type": "ilu"
        }
    )
    uh = problem.solve()
    return uh