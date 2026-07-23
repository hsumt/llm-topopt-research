"""Numerical verification tests for the corrected SIMP implementation.

Run inside the project DOLFINx environment from the repository root:

    /dolfinx-env/bin/python project/solver/verification_tests.py

These are verification tests, not replacements for benchmark validation against
published or professor-approved reference solutions.
"""

from __future__ import annotations
import sys
from pathlib import Path

# verification_tests.py is in project/solver, while agent is in project/agent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from dolfinx import fem

from _01_MSH._boundaries import build_bcs, build_load
from _01_MSH._domain import get_lame_parameters
from _01_MSH._mesh import build_mesh
from _02_FEA._functionspaces import build_spaces
from _02_FEA._solver import solve_fea
from _03_OPTMZER._filters import (
    R_MIN_CONVENTION,
    build_helmholtz_filter_CG1,
    r_pde_from_r_min,
    heaviside_projection,
    heaviside_projection_derivative,
)
from _03_OPTMZER._objective import compute_sensitivities
from agent._physics import validate


def _assert_close(name, actual, expected, rtol, atol=0.0):
    error = abs(actual - expected)
    limit = atol + rtol * abs(expected)
    if error > limit:
        raise AssertionError(
            f"{name}: actual={actual:.12e}, expected={expected:.12e}, "
            f"abs_error={error:.3e}, limit={limit:.3e}"
        )
    print(f"PASS {name}: relative error={error / max(abs(expected), 1e-30):.3e}")


def build_small_cantilever(nelx=24, nely=8):
    Lx, Ly = 3.0, 1.0
    domain = build_mesh(nelx, nely, Lx, Ly)
    V, Q = build_spaces(domain)
    bcs = build_bcs(V, domain)
    F = build_load(V, domain, Lx, Ly, nely)
    mu, lmbda = get_lame_parameters(1.0, 0.3)
    return domain, V, Q, bcs, F, mu, lmbda, Lx, Ly


def test_work_energy():
    domain, V, Q, bcs, F, mu, lmbda, *_ = build_small_cantilever()
    rho = fem.Function(Q)
    rho.x.array[:] = 0.6
    uh, diag = solve_fea(domain, V, bcs, rho, 3.0, mu, lmbda, F)
    assert diag["ksp_reason"] > 0
    assert diag["relative_residual"] < 1e-6
    assert diag["work_energy_relative_error"] < 1e-8
    print("PASS work-energy and equilibrium diagnostics")


def test_physical_density_sensitivity():
    domain, V, Q, bcs, F, mu, lmbda, *_ = build_small_cantilever()
    n = Q.dofmap.index_map.size_local
    rho0 = np.full(n, 0.55)
    rho_fn = fem.Function(Q)

    rho_fn.x.array[:] = rho0
    uh, base = solve_fea(domain, V, bcs, rho_fn, 3.0, mu, lmbda, F)
    gradient = compute_sensitivities(rho0, uh, Q, 3.0, mu, lmbda)

    rng = np.random.default_rng(7)
    direction = rng.normal(size=n)
    direction /= np.linalg.norm(direction)
    eps = 1e-6

    values = []
    for sign in (+1.0, -1.0):
        rho_fn.x.array[:] = rho0 + sign * eps * direction
        _, diag = solve_fea(domain, V, bcs, rho_fn, 3.0, mu, lmbda, F)
        values.append(diag["compliance"])
    fd = (values[0] - values[1]) / (2.0 * eps)
    analytic = float(np.dot(gradient, direction))
    _assert_close("physical-density directional derivative", analytic, fd, rtol=2e-4)


def test_filter_adjoint_and_full_chain():
    domain, V, Q, bcs, F, mu, lmbda, Lx, Ly = build_small_cantilever()
    n = Q.dofmap.index_map.size_local
    h = min(Lx / 24, Ly / 8)
    apply_filter, apply_sens_filter, cell_volumes = build_helmholtz_filter_CG1(
        domain, Q, 2.5 * h
    )

    rng = np.random.default_rng(11)
    x = rng.uniform(0.25, 0.75, size=n)
    integrated_g = rng.normal(size=n)
    lhs = float(np.dot(integrated_g, apply_filter(x)))
    rhs = float(np.dot(apply_sens_filter(integrated_g), x))
    _assert_close("Helmholtz discrete adjoint", lhs, rhs, rtol=5e-10, atol=1e-11)

    beta, eta = 2.0, 0.5
    rho_tilde = apply_filter(x)
    rho_phys = heaviside_projection(rho_tilde, beta, eta)
    rho_fn = fem.Function(Q)
    rho_fn.x.array[:] = rho_phys
    uh, _ = solve_fea(domain, V, bcs, rho_fn, 3.0, mu, lmbda, F)
    dc_phys = compute_sensitivities(rho_phys, uh, Q, 3.0, mu, lmbda)
    dproj = heaviside_projection_derivative(rho_tilde, beta, eta)
    dc_dx = apply_sens_filter(dc_phys * dproj)

    direction = rng.normal(size=n)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    c = []
    for sign in (+1.0, -1.0):
        xt = x + sign * eps * direction
        rt = apply_filter(xt)
        rp = heaviside_projection(rt, beta, eta)
        rho_fn.x.array[:] = rp
        _, diag = solve_fea(domain, V, bcs, rho_fn, 3.0, mu, lmbda, F)
        c.append(diag["compliance"])
    fd = (c[0] - c[1]) / (2.0 * eps)
    analytic = float(np.dot(dc_dx, direction))
    _assert_close("full filter-projection directional derivative", analytic, fd, rtol=5e-4)

    dg_phys = cell_volumes / cell_volumes.sum()
    dg_dx = apply_sens_filter(dg_phys * dproj)
    vf = []
    for sign in (+1.0, -1.0):
        rt = apply_filter(x + sign * eps * direction)
        rp = heaviside_projection(rt, beta, eta)
        vf.append(float(np.dot(rp, cell_volumes) / cell_volumes.sum()))
    fd_v = (vf[0] - vf[1]) / (2.0 * eps)
    analytic_v = float(np.dot(dg_dx, direction))
    _assert_close("full volume directional derivative", analytic_v, fd_v, rtol=5e-5)



def test_fail_closed_sensitivity_evidence():
    n = 6
    rho = np.full(n, 0.4)
    common = {
        "compliance_history": [1.0],
        "residual_history": [1.0e-10],
        "ksp_reason_history": [2],
        "work_energy_error_history": [1.0e-12],
        "final_dc_drho_phys": -np.ones(n),
        "final_dc_dx_design": -np.ones(n),
        "converged": True,
    }
    malformed = [None, [], np.ones((n, 1)), np.full(n, np.nan), 1.0]
    for evidence in malformed:
        metrics = dict(common)
        metrics["final_dc_drho_phys"] = evidence
        result = validate(metrics, rho, 0.4, n, cell_volumes=np.ones(n))
        if result["passed"]:
            raise AssertionError(
                f"Malformed sensitivity evidence falsely passed: {np.shape(evidence)}"
            )
    print("PASS fail-closed sensitivity evidence gate")


def test_filter_radius_convention():
    if R_MIN_CONVENTION != "cone_equivalent_radius":
        raise AssertionError(f"Unexpected r_min convention: {R_MIN_CONVENTION}")
    _assert_close(
        "cone-equivalent radius conversion",
        r_pde_from_r_min(0.125),
        0.125 / (2.0 * np.sqrt(3.0)),
        rtol=1.0e-14,
    )


def main():
    test_fail_closed_sensitivity_evidence()
    test_filter_radius_convention()
    test_work_energy()
    test_physical_density_sensitivity()
    test_filter_adjoint_and_full_chain()
    print("\nAll numerical verification tests passed.")


if __name__ == "__main__":
    main()
