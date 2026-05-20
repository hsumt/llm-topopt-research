from dolfinx.fem.petsc import LinearProblem
from _02_FEA._solver import build_weak_form

def solve_fea(domain, V, bcs, rho_fn, penal, mu, lmbda, F_load):
    a, L = build_weak_form(V, rho_fn, penal, mu, lmbda, F_load)

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
        # alt petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, https://jsdokken.com/dolfinx-tutorial/chapter4/solvers.html
        
    )
    uh = problem.solve()
    return uh
