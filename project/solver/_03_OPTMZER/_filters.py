"""Helmholtz density filter and single-field Heaviside projection.

The projection implemented here is the single intermediate projection at eta=0.5.
It is *not* the full eroded/intermediate/dilated robust formulation of Wang,
Lazarov, and Sigmund (2011).
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc


R_MIN_CONVENTION = "cone_equivalent_radius"
R_MIN_TO_R_PDE = 1.0 / (2.0 * np.sqrt(3.0))


def r_pde_from_r_min(r_min: float) -> float:
    """Convert cone-equivalent physical radius to Helmholtz PDE length."""
    if not np.isfinite(r_min) or r_min <= 0.0:
        raise ValueError("r_min must be finite and positive")
    return float(r_min) * R_MIN_TO_R_PDE


def heaviside_projection(rho_tilde, beta, eta=0.5):
    """Smooth Heaviside projection (Wang, Lazarov & Sigmund, 2011)."""
    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (
        np.tanh(beta * eta) + np.tanh(beta * (rho_tilde - eta))
    ) / denom


def heaviside_projection_derivative(rho_tilde, beta, eta=0.5):
    """Derivative of the smooth Heaviside projection with respect to rho_tilde."""
    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    sech2 = 1.0 / np.cosh(beta * (rho_tilde - eta)) ** 2
    return beta * sech2 / denom


def build_helmholtz_filter_CG1(domain, Q, r_min: float):
    r"""Build a reusable Helmholtz PDE density/sensitivity filter.

    Equation
    --------
    -r_pde^2 Laplacian(rho_tilde) + rho_tilde = rho,
    with homogeneous Neumann boundary conditions.

    ``r_min`` is interpreted as the cone-filter-equivalent physical radius.
    Following Lazarov & Sigmund (2011), the PDE length parameter is
    ``r_pde = r_min / (2*sqrt(3))``.

    Returns
    -------
    apply_filter, apply_sensitivity_filter, cell_volumes
    """
    if domain.comm.size != 1:
        raise NotImplementedError(
            "This verified build currently supports the Helmholtz filter in serial only."
        )
    if not np.isfinite(r_min) or r_min <= 0.0:
        raise ValueError("r_min must be finite and positive")

    r_pde = r_pde_from_r_min(r_min)
    V_cg = fem.functionspace(domain, ("Lagrange", 1))

    rho_dg_fn = fem.Function(Q)
    rho_tilde_cg = fem.Function(V_cg)

    phi = ufl.TrialFunction(V_cg)
    psi = ufl.TestFunction(V_cg)

    a_h = (
        r_pde**2 * ufl.inner(ufl.grad(phi), ufl.grad(psi))
        + ufl.inner(phi, psi)
    ) * ufl.dx
    a_form = fem.form(a_h)
    A = fem_petsc.assemble_matrix(a_form)
    A.assemble()

    L_h = ufl.inner(rho_dg_fn, psi) * ufl.dx
    linear_form = fem.form(L_h)
    # This installed DOLFINx API allocates PETSc vectors from compiled linear forms.
    b = fem_petsc.create_vector(linear_form)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=1000)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    solver.setFromOptions()
    solver.setUp()

    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    q_test = ufl.TestFunction(Q)
    vol_form = fem.form(one * q_test * ufl.dx)
    b_vol = fem_petsc.create_vector(vol_form)
    with b_vol.localForm() as b_local:
        b_local.set(0.0)
    fem_petsc.assemble_vector(b_vol, vol_form)
    b_vol.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    cell_volumes = b_vol.array.copy()
    b_vol.destroy()

    if np.any(~np.isfinite(cell_volumes)) or np.any(cell_volumes <= 0.0):
        raise RuntimeError("Invalid DG-0 cell measures in Helmholtz filter")

    # Cell-average projection from CG-1 to DG-0:
    #   rho_bar_e = (1/|Omega_e|) int_{Omega_e} rho_tilde dx.
    # Using cell averages, rather than point interpolation at the DG-0
    # interpolation point, makes the discrete mass-weighted adjoint below
    # mathematically consistent with the forward filter.
    projection_form = fem.form(rho_tilde_cg * q_test * ufl.dx)
    b_projection = fem_petsc.create_vector(projection_form)

    def apply_filter(rho_array: np.ndarray) -> np.ndarray:
        rho_array = np.asarray(rho_array, dtype=float)
        if rho_array.shape != rho_dg_fn.x.array.shape:
            raise ValueError(
                f"Filter input shape {rho_array.shape} does not match DG-0 shape "
                f"{rho_dg_fn.x.array.shape}."
            )
        if not np.all(np.isfinite(rho_array)):
            raise FloatingPointError("Non-finite density supplied to Helmholtz filter")

        rho_dg_fn.x.array[:] = rho_array
        rho_dg_fn.x.scatter_forward()

        with b.localForm() as b_local:
            b_local.set(0.0)
        fem_petsc.assemble_vector(b, linear_form)
        b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )

        solver.solve(b, rho_tilde_cg.x.petsc_vec)
        reason = int(solver.getConvergedReason())
        if reason <= 0:
            raise RuntimeError(
                f"Helmholtz filter KSP failed; reason={reason}, "
                f"iterations={solver.getIterationNumber()}."
            )
        rho_tilde_cg.x.scatter_forward()

        with b_projection.localForm() as projection_local:
            projection_local.set(0.0)
        fem_petsc.assemble_vector(b_projection, projection_form)
        b_projection.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        result = b_projection.array.copy() / cell_volumes
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Helmholtz filter returned non-finite values")
        return result

    def apply_sensitivity_filter(integrated_gradient: np.ndarray) -> np.ndarray:
        r"""Apply the transpose Helmholtz map to an integrated cell gradient.

        For the self-adjoint PDE operator, H^T = H under the mass-weighted
        inner product. Dividing by cell measure converts an integrated DG-0
        gradient to a density before filtering; multiplying afterward restores
        the integrated derivative expected by MMA.
        """
        integrated_gradient = np.asarray(integrated_gradient, dtype=float)
        if integrated_gradient.shape != cell_volumes.shape:
            raise ValueError("Sensitivity shape does not match cell-volume vector")
        density_gradient = integrated_gradient / cell_volumes
        return apply_filter(density_gradient) * cell_volumes

    # Keep PETSc objects alive in closures for the lifetime of the filter.
    return apply_filter, apply_sensitivity_filter, cell_volumes.copy()
