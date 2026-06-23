"""
Implemented Helmholtz filter 5/20/26
_filters.py
Initial Parameter:
-r² ∇²ρ̃ + ρ̃ = ρ
where r = r_min / (2√3) maps the filter radius to the PDE length scale.
Reference: Lazarov & Sigmund (2016), "Filters in topology optimization based on Helmholtz-type differential equations", IJNME Vol. 86, Eq. (4).
The 2√3 factor ensures the Helmholtz filter's influence radius matches the cone filter's radius exactly — this is a derived constant, not a guess.


5/20/2026 Shifted filter from after loop to before loop due to significant artificial greyscaling from smoothing after the fact.

"""

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

# def r_min_elements_to_physical(r_min_elements, Lx, Ly, nelx, nely):
#     dx = float(Lx) / float(nelx)
#     dy = float(Ly) / float(nely)
#     return float(r_min_elements) * min(dx, dy)

# Equation 9 Source [5]
def heaviside_projection(rho_tilde, beta, eta=0.5):
    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (
        np.tanh(beta * eta) + np.tanh(beta * (rho_tilde - eta))
    ) / denom


def heaviside_projection_derivative(rho_tilde, beta, eta=0.5):
    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    sech2 = 1.0 / np.cosh(beta * (rho_tilde - eta))**2
    return beta * sech2 / denom

def build_helmholtz_filter_CG1(domain, Q, r_min: float):
    """
    Helmholtz PDE filter. Solves on CG-1, projects back to DG-0.
    LHS matrix is assembled and factorized once — reused every iteration.

    Reference: Lazarov & Sigmund (2006), IJNME 86:765-781, Eq.(4)
      r_pde = r_min / (2*sqrt(3))  — cone-equivalent radius conversion

    Boundary conditions: homogeneous Neumann (natural BC of weak form).
      r²(∇ρ̃ · n) = 0 on ∂Ω 
      Lazarov & Sigmund (2006), Section 2.1.
    """
    # r_pde = r_min / (2.0 * np.sqrt(3.0))
    r_pde = float(r_min)

    V_cg = fem.functionspace(domain, ("Lagrange", 1))

    rho_dg_fn    = fem.Function(Q)      # source: updated each iteration
    rho_tilde_cg = fem.Function(V_cg)   # PDE solution on CG-1
    rho_tilde_dg = fem.Function(Q)      # filtered output on DG-0

    phi = ufl.TrialFunction(V_cg)
    psi = ufl.TestFunction(V_cg)

    # Bilinear form: r²(∇φ,∇ψ) + (φ,ψ)  — assembled once
    a_h = (r_pde**2 * ufl.inner(ufl.grad(phi), ufl.grad(psi))
           + ufl.inner(phi, psi)) * ufl.dx
    bilinear_form = fem.form(a_h)
    A = fem.petsc.assemble_matrix(bilinear_form)
    A.assemble()

    # Linear form: (ρ_raw, ψ)  — RHS reassembled each iteration
    L_h = ufl.inner(rho_dg_fn, psi) * ufl.dx
    linear_form = fem.form(L_h)
    # DOLFINx 0.10.0: create_vector takes a FunctionSpace, not a Form
    b = fem.petsc.create_vector(linear_form)

    # KSP solver — operator set once, reused every iteration
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)

    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")

    solver.setFromOptions()
    solver.setUp()      # finalizes PC setup before first solve

    # Cell volumes via DG-0 mass lumping: ∫ 1·ψ_e dx = |Ω_e|
    # Used to convert between integrated and density sensitivities
    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    vol_form = fem.form(one * ufl.TestFunction(Q) * ufl.dx)
    # DOLFINx 0.10.0: create_vector takes a FunctionSpace, not a Form
    b_vol = fem.petsc.create_vector(vol_form)
    fem.petsc.assemble_vector(b_vol, vol_form)
    b_vol.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE
    )
    v_array = b_vol.array.copy()   # shape: (n_cells,), value: cell area [m²]

    def apply_filter(rho_array: np.ndarray) -> np.ndarray:
        """
        Solves: -r²Δρ̃ + ρ̃ = ρ  (weak form on CG-1)
        Returns filtered density on DG-0.
        """
        rho_dg_fn.x.array[:] = rho_array

        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(b, linear_form)
        b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )

        solver.solve(b, rho_tilde_cg.x.petsc_vec)
        rho_tilde_cg.x.scatter_forward()

        rho_tilde_dg.interpolate(rho_tilde_cg)
        return rho_tilde_dg.x.array.copy()

    def apply_sensitivity_filter(dc_array: np.ndarray) -> np.ndarray:
        """
        Filters sensitivities through the same self-adjoint Helmholtz operator.
        Chain rule: dC/dρ = H^T (dC/dρ̃) = H (dC/dρ̃)  since H is self-adjoint.

        Volume weighting converts integrated DOF values to density fields
        before filtering, then back after.
        Reference: Lazarov & Sigmund (2016), Section 2.3
        """
        dc_density = dc_array / v_array          # integrated → density
        filtered   = apply_filter(dc_density)    # apply H
        return filtered * v_array                # density → integrated

    return apply_filter, apply_sensitivity_filter