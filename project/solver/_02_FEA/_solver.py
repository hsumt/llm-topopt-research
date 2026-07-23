"""Linear-elastic DOLFINx solve with explicit nodal load-vector semantics."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem import petsc as fem_petsc

from _02_FEA._assembly import build_stiffness_form


def solve_fea(domain, V, bcs, rho_fn, penal, mu, lmbda, F_load, thickness=1.0):
    """Solve K(rho) U = F and return displacement plus diagnostics.

    ``F_load`` is a ``fem.Function(V)`` whose coefficient vector is the actual
    algebraic nodal force vector. It is copied directly into the PETSc RHS; it
    is not integrated as a body-force field.
    """
    if domain.comm.size != 1:
        raise NotImplementedError(
            "This verified build currently supports serial execution only. "
            "MPI-safe load assembly, reductions, and output are not yet implemented."
        )

    a = fem.form(build_stiffness_form(
        V, rho_fn, penal, mu, lmbda, thickness=thickness
    ))
    A = fem_petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = F_load.x.petsc_vec.copy()

    # Apply lifting so non-zero Dirichlet values remain mathematically correct.
    fem_petsc.apply_lifting(b, [a], bcs=[bcs])
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    fem_petsc.set_bc(b, bcs)

    uh = fem.Function(V)
    uh.name = "displacement"

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.setTolerances(rtol=1.0e-8, atol=1.0e-12, max_it=2000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    ksp.setFromOptions()
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    reason = int(ksp.getConvergedReason())
    if reason <= 0:
        raise RuntimeError(
            f"PETSc KSP failed to converge; converged reason={reason}, "
            f"iterations={ksp.getIterationNumber()}."
        )

    Au = b.duplicate()
    A.mult(uh.x.petsc_vec, Au)

    residual = Au.copy()
    residual.axpy(-1.0, b)
    rhs_norm = float(b.norm())
    residual_norm = float(residual.norm())
    relative_residual = (
        residual_norm / rhs_norm if rhs_norm > 0.0 else residual_norm
    )

    compliance = float(b.dot(uh.x.petsc_vec))
    strain_energy = float(uh.x.petsc_vec.dot(Au))
    energy_scale = max(abs(compliance), abs(strain_energy), 1.0e-30)
    work_energy_error = abs(compliance - strain_energy) / energy_scale

    diagnostics = {
        "relative_residual": relative_residual,
        "ksp_reason": reason,
        "ksp_iterations": int(ksp.getIterationNumber()),
        "compliance": compliance,
        "strain_energy": strain_energy,
        "work_energy_relative_error": float(work_energy_error),
    }

    if not all(np.isfinite(v) for v in diagnostics.values()):
        raise FloatingPointError(f"Non-finite FEA diagnostics: {diagnostics}")

    residual.destroy()
    Au.destroy()
    b.destroy()
    A.destroy()
    ksp.destroy()

    return uh, diagnostics
