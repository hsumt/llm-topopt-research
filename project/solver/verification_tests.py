"""Numerical verification tests for the corrected SIMP implementation.

Run inside the project DOLFINx environment from the repository root:

    /dolfinx-env/bin/python -m project.solver.verification_tests

On success, this writes a hash-bound verification manifest. Result packets only
recognize the suite when that manifest still matches the current source files.
These tests do not replace benchmark validation or the mesh-refinement study.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import dolfinx
import numpy as np
from dolfinx import fem
from petsc4py import PETSc

from project.topopt.fem.boundaries import build_bcs, build_load
from project.topopt.fem.domain import get_lame_parameters
from project.topopt.fem.mesh import build_mesh
from project.topopt.fem.functionspaces import build_spaces
from project.topopt.fem.solver import solve_fea
from project.topopt.optimization.filters import (
    R_MIN_CONVENTION,
    build_helmholtz_filter_CG1,
    heaviside_projection,
    heaviside_projection_derivative,
    r_pde_from_r_min,
)
from project.topopt.optimization.objective import compute_sensitivities
from project.solver._verification_manifest import write_manifest
from project.agent._physics import validate
from project.solver.reference_q4 import structured_q4_cantilever
from project.solver.SIMP_MASTER import _termination_status
from project.parser.provenance import (
    canonical_field_paths,
    get_path,
    validate_field_provenance,
)
from project.parser.schema import (
    DefaultedField,
    FieldProvenance,
    ProblemSpec,
)


PASSED_TESTS: list[str] = []


def _pass(name: str, detail: str = ""):
    PASSED_TESTS.append(name)
    print(f"PASS {name}" + (f": {detail}" if detail else ""))


def _assert_close(name, actual, expected, rtol, atol=0.0):
    error = abs(actual - expected)
    limit = atol + rtol * abs(expected)
    if error > limit:
        raise AssertionError(
            f"{name}: actual={actual:.12e}, expected={expected:.12e}, "
            f"abs_error={error:.3e}, limit={limit:.3e}"
        )
    _pass(name, f"relative error={error / max(abs(expected), 1e-30):.3e}")


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
    _, diag = solve_fea(
        domain, V, bcs, rho, 3.0, mu, lmbda, F, thickness=1.0
    )
    assert diag["ksp_reason"] > 0
    assert diag["relative_residual"] < 1e-6
    assert diag["work_energy_relative_error"] < 1e-8
    _pass("work-energy and equilibrium diagnostics")


def test_independent_q4_reference():
    nelx, nely = 12, 4
    Lx, Ly = 3.0, 1.0
    domain, V, Q, bcs, F, mu, lmbda, *_ = build_small_cantilever(nelx, nely)
    rho_value = 0.6
    rho = fem.Function(Q)
    rho.x.array[:] = rho_value
    _, diag = solve_fea(
        domain, V, bcs, rho, 3.0, mu, lmbda, F, thickness=1.0
    )
    reference = structured_q4_cantilever(
        nelx=nelx,
        nely=nely,
        Lx=Lx,
        Ly=Ly,
        rho=np.full((nely, nelx), rho_value),
        penal=3.0,
        E=1.0,
        nu=0.3,
        thickness=1.0,
        load_value=-1.0,
    )
    _assert_close(
        "independent NumPy-Q4 compliance",
        diag["compliance"],
        reference["compliance"],
        rtol=2.0e-9,
        atol=1.0e-10,
    )
    _assert_close(
        "independent Q4 vertical reaction balance",
        reference["reaction_y"],
        -reference["load_resultant_y"],
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_physical_density_sensitivity():
    domain, V, Q, bcs, F, mu, lmbda, *_ = build_small_cantilever()
    n = Q.dofmap.index_map.size_local
    rho0 = np.full(n, 0.55)
    rho_fn = fem.Function(Q)

    rho_fn.x.array[:] = rho0
    uh, _ = solve_fea(
        domain, V, bcs, rho_fn, 3.0, mu, lmbda, F, thickness=1.0
    )
    gradient = compute_sensitivities(
        rho0, uh, Q, 3.0, mu, lmbda, thickness=1.0
    )

    rng = np.random.default_rng(7)
    direction = rng.normal(size=n)
    direction /= np.linalg.norm(direction)
    eps = 1e-6

    values = []
    for sign in (+1.0, -1.0):
        rho_fn.x.array[:] = rho0 + sign * eps * direction
        _, diag = solve_fea(
            domain, V, bcs, rho_fn, 3.0, mu, lmbda, F, thickness=1.0
        )
        values.append(diag["compliance"])
    fd = (values[0] - values[1]) / (2.0 * eps)
    analytic = float(np.dot(gradient, direction))
    _assert_close(
        "physical-density directional derivative", analytic, fd, rtol=2e-4
    )


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
    _assert_close(
        "Helmholtz discrete adjoint", lhs, rhs, rtol=5e-10, atol=1e-11
    )

    beta, eta = 2.0, 0.5
    rho_tilde = apply_filter(x)
    rho_phys = heaviside_projection(rho_tilde, beta, eta)
    rho_fn = fem.Function(Q)
    rho_fn.x.array[:] = rho_phys
    uh, _ = solve_fea(
        domain, V, bcs, rho_fn, 3.0, mu, lmbda, F, thickness=1.0
    )
    dc_phys = compute_sensitivities(
        rho_phys, uh, Q, 3.0, mu, lmbda, thickness=1.0
    )
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
        _, diag = solve_fea(
            domain, V, bcs, rho_fn, 3.0, mu, lmbda, F, thickness=1.0
        )
        c.append(diag["compliance"])
    fd = (c[0] - c[1]) / (2.0 * eps)
    analytic = float(np.dot(dc_dx, direction))
    _assert_close(
        "full filter-projection directional derivative",
        analytic,
        fd,
        rtol=5e-4,
    )

    dg_phys = cell_volumes / cell_volumes.sum()
    dg_dx = apply_sens_filter(dg_phys * dproj)
    vf = []
    for sign in (+1.0, -1.0):
        rt = apply_filter(x + sign * eps * direction)
        rp = heaviside_projection(rt, beta, eta)
        vf.append(float(np.dot(rp, cell_volumes) / cell_volumes.sum()))
    fd_v = (vf[0] - vf[1]) / (2.0 * eps)
    analytic_v = float(np.dot(dg_dx, direction))
    _assert_close(
        "full volume directional derivative", analytic_v, fd_v, rtol=5e-5
    )


def _valid_metrics(n: int) -> dict:
    return {
        "compliance_history": [1.0],
        "residual_history": [1.0e-10],
        "ksp_reason_history": [2],
        "work_energy_error_history": [1.0e-12],
        "final_dc_drho_phys": -np.ones(n),
        "final_dc_dx_design": -np.ones(n),
        "design_converged": True,
        "objective_plateau": False,
        "continuation_complete": True,
        "converged": True,
    }


def test_fail_closed_sensitivity_evidence():
    n = 6
    rho = np.full(n, 0.4)
    malformed = [None, [], np.ones((n, 1)), np.full(n, np.nan), 1.0]
    for evidence in malformed:
        metrics = _valid_metrics(n)
        metrics["final_dc_drho_phys"] = evidence
        result = validate(metrics, rho, 0.4, n, cell_volumes=np.ones(n))
        if result["passed"]:
            raise AssertionError(
                f"Malformed sensitivity evidence falsely passed: {np.shape(evidence)}"
            )
    _pass("fail-closed sensitivity evidence gate")


def test_checkerboard_is_diagnostic_only():
    nely, nelx = 4, 6
    n = nely * nelx
    rho_grid = (np.indices((nely, nelx)).sum(axis=0) % 2).astype(float)
    result = validate(
        _valid_metrics(n),
        rho_grid.ravel(),
        float(rho_grid.mean()),
        n,
        cell_volumes=np.ones(n),
        rho_grid=rho_grid,
    )
    check = result["checks"]["checkerboard_alternating_mode_index"]
    assert check["passed"] is None
    assert check["severity"] == "diagnostic"
    assert result["passed"]
    _pass("checkerboard proxy has no Boolean pass interpretation")


def test_termination_semantics():
    plateau_only = _termination_status(
        iteration=180,
        change=0.062,
        tol_change=0.01,
        objective_plateau=True,
        last_beta_change=150,
        effective_beta_schedule={50: 2.0, 100: 4.0, 150: 8.0},
    )
    assert plateau_only["objective_plateau"]
    assert not plateau_only["design_converged"]
    assert not plateau_only["fully_converged"]

    design = _termination_status(
        iteration=225,
        change=0.005,
        tol_change=0.01,
        objective_plateau=True,
        last_beta_change=200,
        effective_beta_schedule={50: 2.0, 100: 4.0, 150: 8.0, 200: 16.0},
    )
    assert design["design_converged"]
    assert design["continuation_complete"]
    assert design["fully_converged"]
    _pass("objective plateau is separated from design convergence")


def test_parser_provenance_gate():
    spec = ProblemSpec.model_validate(
        {
            "name": "Cantilever Beam",
            "analysis": {
                "formulation": "plane_stress",
                "unit_system": "nondimensional",
                "thickness": 1.0,
                "edge_traction_definition": "line_load",
            },
            "mesh": {"nx": 12, "ny": 4, "Lx": 3.0, "Ly": 1.0},
            "material": {"E": 1.0, "nu": 0.3},
            "loads": [
                {
                    "location": "right_center",
                    "dof": "y",
                    "value": -1.0,
                    "kind": "point_force",
                }
            ],
            "bcs": [
                {"location": "left_edge", "dof": "x", "value": 0.0},
                {"location": "left_edge", "dof": "y", "value": 0.0},
            ],
            "simp": {
                "penal": 3.0,
                "vol_frac": 0.4,
                "r_min": 0.625,
                "max_iter": 250,
                "tol_change": 0.01,
            },
        }
    )
    payload = spec.model_dump()
    defaulted = [
        DefaultedField(
            field_path="simp.r_min",
            default_used=0.625,
            question="What filter radius should I use? Default: 0.625",
        )
    ]
    provenance = []
    for path in canonical_field_paths(spec):
        if path.startswith("analysis."):
            source = "fixed_by_solver_scope"
            evidence = None
        elif path == "simp.r_min":
            source = "defaulted"
            evidence = None
        else:
            source = "explicit"
            evidence = "test fixture"
        provenance.append(
            FieldProvenance(
                field_path=path,
                source=source,
                value=get_path(payload, path),
                evidence=evidence,
                confidence=1.0,
            )
        )
    validate_field_provenance(spec, defaulted, provenance)

    try:
        validate_field_provenance(spec, defaulted, provenance[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("missing field provenance did not fail closed")

    contradictory = [item.model_copy(deep=True) for item in provenance]
    contradictory[0].source = "contradictory"
    contradictory[0].evidence = "conflicting test fixture"
    try:
        validate_field_provenance(spec, defaulted, contradictory)
    except ValueError:
        pass
    else:
        raise AssertionError("contradictory provenance did not fail closed")
    _pass("parser field provenance fails closed")


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
    test_checkerboard_is_diagnostic_only()
    test_termination_semantics()
    test_parser_provenance_gate()
    test_filter_radius_convention()
    test_work_energy()
    test_independent_q4_reference()
    test_physical_density_sensitivity()
    test_filter_adjoint_and_full_chain()

    versions = {
        "python": platform.python_version(),
        "dolfinx": dolfinx.__version__,
        "numpy": np.__version__,
        "petsc": ".".join(str(v) for v in PETSc.Sys.getVersion()),
    }
    path = write_manifest(
        PROJECT_ROOT,
        tests=PASSED_TESTS,
        versions=versions,
    )
    print("\nAll numerical verification tests passed.")
    print(f"Verification manifest written: {path}")


if __name__ == "__main__":
    main()
