"""Deterministic SIMP topology-optimization controller.

Architecture
------------
Natural-language agents may produce/interpret structured data, but this module
owns all finite-element, filtering, projection, sensitivity, optimization, and
verification calculations. The LLM never writes or modifies physics code.

Current verified scope
----------------------
* 2-D small-strain, isotropic linear elasticity
* plane stress
* compliance minimization with one material-volume constraint
* SIMP + Helmholtz PDE filter + one intermediate Heaviside projection
* MMA update
* serial DOLFINx execution

The projection is not the full robust eroded/intermediate/dilated formulation of
Wang, Lazarov, and Sigmund (2011).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

import numpy as np
from dolfinx import fem

from _01_MSH._boundaries import (
    build_bcs,
    build_bcs_mbb,
    build_load,
    build_load_mbb,
)
from _01_MSH._domain import get_lame_parameters
from _01_MSH._mesh import build_mesh
from _02_FEA._functionspaces import build_spaces
from _02_FEA._solver import solve_fea
from _03_OPTMZER._filters import (
    build_helmholtz_filter_CG1,
    heaviside_projection,
    heaviside_projection_derivative,
)
from _03_OPTMZER._MMAupdate import MMAOptimizer
from _03_OPTMZER._objective import compute_compliance, compute_sensitivities
from _04_PPRCS.postprocess import (
    build_cell_perm,
    export_xdmf,
    print_iteration_report,
    save_final_density,
    save_frame,
    save_gif,
    save_summary_slide,
)
from agent._critic import criticize
from agent._physics import validate


CASE = "cantilever"

CASE_PARAMS = {
    "cantilever": {
        "nelx": 80,
        "nely": 50,
        "Lx": 1.6,
        "Ly": 1.0,
        "volfrac": 0.4,
        "penal": 3.0,
        "r_min": 0.05,
        "max_iter": 200,
        "tol_change": 0.01,
        "E": 1.0,
        "nu": 0.3,
    },
    "mbb": {
        "nelx": 120,
        "nely": 40,
        "Lx": 3.0,
        "Ly": 1.0,
        "volfrac": 0.4,
        "penal": 3.0,
        "r_min": 0.06,
        "max_iter": 400,
        "tol_change": 0.01,
        "E": 1.0,
        "nu": 0.3,
    },
}

# Requested continuation schedule. A step is activated only when at least
# BETA_HOLD_ITERATIONS remain afterward; otherwise it would be reported as a
# final beta value without ever being optimized at that value.
REQUESTED_BETA_SCHEDULE = {50: 2.0, 100: 4.0, 150: 8.0, 200: 16.0}
BETA_HOLD_ITERATIONS = 20
ETA = 0.5
RHO_MIN = 1.0e-3


@dataclass
class OptimizationResult:
    rho_fn: fem.Function
    rho_final: np.ndarray
    rho_history: list[np.ndarray]
    frame_paths: list[str]
    metrics: dict
    beta_final: float
    beta_schedule_effective: dict[int, float]
    convergence_reason: str
    cell_volumes: np.ndarray


def _effective_beta_schedule(max_iter: int) -> dict[int, float]:
    latest_usable = max_iter - BETA_HOLD_ITERATIONS
    effective = {
        iteration: beta
        for iteration, beta in REQUESTED_BETA_SCHEDULE.items()
        if iteration <= latest_usable
    }
    omitted = {
        iteration: beta
        for iteration, beta in REQUESTED_BETA_SCHEDULE.items()
        if iteration not in effective
    }
    if omitted:
        print(
            "[Continuation] Not activating late beta steps without a "
            f"{BETA_HOLD_ITERATIONS}-iteration optimization tail: {omitted}. "
            "Increase max_iter to execute them."
        )
    return effective


def _check_problem_setup(domain, bcs, F_load):
    if domain.comm.size != 1:
        raise NotImplementedError(
            "The verified pipeline is serial-only. Run with one MPI rank."
        )
    for i, bc in enumerate(bcs):
        parent, _ = bc.dof_indices()
        if len(parent) == 0:
            raise RuntimeError(f"Boundary condition {i} constrains zero DOFs")
    if np.count_nonzero(F_load.x.array) == 0:
        raise RuntimeError("The algebraic load vector has no non-zero entries")
    if not np.all(np.isfinite(F_load.x.array)):
        raise FloatingPointError("The algebraic load vector is non-finite")


def _physical_density(x_design, apply_filter, beta):
    rho_tilde = np.clip(apply_filter(x_design), RHO_MIN, 1.0)
    rho_phys = np.clip(
        heaviside_projection(rho_tilde, beta, ETA), RHO_MIN, 1.0
    )
    return rho_tilde, rho_phys


def _evaluate_state(
    *,
    x_design,
    beta,
    apply_filter,
    apply_sens_filter,
    rho_fn,
    domain,
    V,
    Q,
    bcs,
    F_load,
    penal,
    mu,
    lmbda,
    cell_volumes,
):
    """Evaluate one design state and return objective, gradients, and diagnostics."""
    rho_tilde, rho_phys = _physical_density(x_design, apply_filter, beta)
    rho_fn.x.array[:] = rho_phys
    rho_fn.x.scatter_forward()

    uh, solve_diag = solve_fea(
        domain, V, bcs, rho_fn, penal, mu, lmbda, F_load
    )
    compliance = float(solve_diag["compliance"])

    # Independent algebraic computation catches load-vector plumbing errors.
    compliance_direct = compute_compliance(uh, F_load)
    mismatch = abs(compliance - compliance_direct) / max(abs(compliance), 1.0e-30)
    if mismatch > 1.0e-12:
        raise RuntimeError(
            f"Compliance mismatch between solver RHS and objective: {mismatch:.3e}"
        )

    dc_drho_phys = compute_sensitivities(
        rho_phys, uh, Q, penal, mu, lmbda
    )
    dproj = heaviside_projection_derivative(rho_tilde, beta, ETA)

    total_volume = float(cell_volumes.sum())
    volfrac_actual = float(np.dot(rho_phys, cell_volumes) / total_volume)
    dg_drho_phys = cell_volumes / total_volume
    dc_dx_design = apply_sens_filter(dc_drho_phys * dproj)
    dg_dx_design = apply_sens_filter(dg_drho_phys * dproj)

    return {
        "rho_tilde": rho_tilde,
        "rho_phys": rho_phys,
        "compliance": compliance,
        "dc_drho_phys": dc_drho_phys,
        "dproj": dproj,
        "volfrac": volfrac_actual,
        "dg_drho_phys": dg_drho_phys,
        "dc_dx_design": dc_dx_design,
        "dg_dx_design": dg_dx_design,
        "solve": solve_diag,
    }


def _replace_final_state_metrics(metrics: dict, state: dict, beta: float):
    """Align the last history entry with the exact final post-update design."""
    metrics["compliance_history"][-1] = float(state["compliance"])
    metrics["volfrac_history"][-1] = float(state["volfrac"])
    metrics["residual_history"][-1] = float(state["solve"]["relative_residual"])
    metrics["ksp_reason_history"][-1] = int(state["solve"]["ksp_reason"])
    metrics["ksp_iterations_history"][-1] = int(state["solve"]["ksp_iterations"])
    metrics["work_energy_error_history"][-1] = float(
        state["solve"]["work_energy_relative_error"]
    )
    metrics["beta_history"][-1] = float(beta)
    metrics["final_dc_drho_phys"] = state["dc_drho_phys"].copy()
    metrics["final_dc_dx_design"] = state["dc_dx_design"].copy()


def _run_optimization(
    *,
    domain,
    V,
    Q,
    bcs,
    F_load,
    p: dict,
    frames_dir: str,
    perm: np.ndarray,
) -> OptimizationResult:
    """Run the deterministic SIMP/MMA loop."""
    _check_problem_setup(domain, bcs, F_load)

    n_cells = Q.dofmap.index_map.size_local
    if n_cells != p["nelx"] * p["nely"]:
        raise RuntimeError(
            "The serial DG-0 cell count does not match nelx*nely."
        )

    rho_fn = fem.Function(Q)
    rho_fn.name = "density"
    apply_filter, apply_sens_filter, cell_volumes = build_helmholtz_filter_CG1(
        domain, Q, float(p["r_min"])
    )

    optimizer = MMAOptimizer(
        n=n_cells, x_min=RHO_MIN, x_max=1.0, move=0.2
    )
    x_design = np.full(n_cells, float(p["volfrac"]), dtype=float)

    beta = 1.0
    beta_schedule = _effective_beta_schedule(int(p["max_iter"]))
    last_beta_change = 0

    metrics = {
        "compliance_history": [],
        "volfrac_history": [],
        "change_history": [],
        "l2_change_history": [],
        "residual_history": [],
        "ksp_reason_history": [],
        "ksp_iterations_history": [],
        "work_energy_error_history": [],
        "beta_history": [],
        "final_dc_drho_phys": None,
        "final_dc_dx_design": None,
        "final_kkt_diagnostics": None,
        "iteration": 0,
        "converged": False,
    }
    rho_history: list[np.ndarray] = []
    frame_paths: list[str] = []
    reason = "max_iter_reached"

    for iteration in range(1, int(p["max_iter"]) + 1):
        if iteration in beta_schedule:
            beta = float(beta_schedule[iteration])
            last_beta_change = iteration
            print(f"  [Projection continuation] beta -> {beta:.1f}")

        state = _evaluate_state(
            x_design=x_design,
            beta=beta,
            apply_filter=apply_filter,
            apply_sens_filter=apply_sens_filter,
            rho_fn=rho_fn,
            domain=domain,
            V=V,
            Q=Q,
            bcs=bcs,
            F_load=F_load,
            penal=float(p["penal"]),
            mu=p["mu"],
            lmbda=p["lmbda"],
            cell_volumes=cell_volumes,
        )

        dc_dx = state["dc_dx_design"]
        dg_dx = state["dg_dx_design"]
        g_val = state["volfrac"] - float(p["volfrac"])

        x_new = optimizer.update(
            x=x_design.copy(),
            f0val=state["compliance"],
            df0dx=dc_dx,
            fval=g_val,
            dfdx=dg_dx,
        )
        if not np.all(np.isfinite(x_new)):
            raise FloatingPointError("MMA returned non-finite design variables")
        x_new = np.clip(x_new, RHO_MIN, 1.0)

        delta = x_new - x_design
        change = float(np.max(np.abs(delta)))
        l2_change = float(np.linalg.norm(delta) / np.sqrt(n_cells))

        metrics["compliance_history"].append(float(state["compliance"]))
        metrics["volfrac_history"].append(float(state["volfrac"]))
        metrics["change_history"].append(change)
        metrics["l2_change_history"].append(l2_change)
        metrics["residual_history"].append(
            float(state["solve"]["relative_residual"])
        )
        metrics["ksp_reason_history"].append(
            int(state["solve"]["ksp_reason"])
        )
        metrics["ksp_iterations_history"].append(
            int(state["solve"]["ksp_iterations"])
        )
        metrics["work_energy_error_history"].append(
            float(state["solve"]["work_energy_relative_error"])
        )
        metrics["beta_history"].append(beta)
        metrics["final_dc_drho_phys"] = state["dc_drho_phys"].copy()
        metrics["final_dc_dx_design"] = state["dc_dx_design"].copy()
        metrics["iteration"] = iteration

        rho_fn.x.array[:] = state["rho_phys"]
        rho_fn.x.scatter_forward()
        print_iteration_report(
            iteration, state["compliance"], state["volfrac"], change
        )
        frame_paths.append(
            save_frame(
                rho_fn,
                p["nelx"],
                p["nely"],
                iteration,
                state["compliance"],
                frames_dir,
                perm,
            )
        )
        rho_history.append(state["rho_phys"].copy())

        # Commit the design update after recording the state that generated it.
        x_design = x_new

        design_converged = change < float(p["tol_change"])
        compliance_converged = False
        if len(metrics["compliance_history"]) >= 21:
            c_now = metrics["compliance_history"][-1]
            c_old = metrics["compliance_history"][-21]
            recent_change = abs(c_now - c_old) / max(abs(c_now), 1.0e-12)
            compliance_converged = recent_change < 5.0e-4

        all_continuation_steps_applied = (
            not beta_schedule or iteration >= max(beta_schedule)
        )
        continuation_settled = (
            iteration - last_beta_change >= BETA_HOLD_ITERATIONS
        )
        may_stop = (
            iteration >= 80
            and all_continuation_steps_applied
            and continuation_settled
        )
        if may_stop and (design_converged or compliance_converged):
            reason = (
                "design_change" if design_converged else "compliance_plateau"
            )
            metrics["converged"] = True
            print(f"\nConverged at iteration {iteration} ({reason})")
            break

    # Exact final re-evaluation: x_design is the last MMA output, whereas the
    # state recorded in the loop was the design that produced that output.
    final_state = _evaluate_state(
        x_design=x_design,
        beta=beta,
        apply_filter=apply_filter,
        apply_sens_filter=apply_sens_filter,
        rho_fn=rho_fn,
        domain=domain,
        V=V,
        Q=Q,
        bcs=bcs,
        F_load=F_load,
        penal=float(p["penal"]),
        mu=p["mu"],
        lmbda=p["lmbda"],
        cell_volumes=cell_volumes,
    )
    _replace_final_state_metrics(metrics, final_state, beta)
    final_g = final_state["volfrac"] - float(p["volfrac"])
    metrics["final_kkt_diagnostics"] = optimizer.kkt_diagnostics(
        x_design,
        final_state["dc_dx_design"],
        final_g,
        final_state["dg_dx_design"],
    )
    rho_final = final_state["rho_phys"].copy()
    rho_history[-1] = rho_final.copy()
    rho_fn.x.array[:] = rho_final
    rho_fn.x.scatter_forward()

    # Overwrite the final frame so image, density history, and final compliance
    # all describe the same state.
    frame_paths[-1] = save_frame(
        rho_fn,
        p["nelx"],
        p["nely"],
        metrics["iteration"],
        final_state["compliance"],
        frames_dir,
        perm,
    )

    return OptimizationResult(
        rho_fn=rho_fn,
        rho_final=rho_final,
        rho_history=rho_history,
        frame_paths=frame_paths,
        metrics=metrics,
        beta_final=beta,
        beta_schedule_effective=beta_schedule,
        convergence_reason=reason,
        cell_volumes=cell_volumes,
    )


def _prepare_output_dir(path: str, clear: bool):
    if clear and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(os.path.join(path, "frames"), exist_ok=True)


def _export_run(result: OptimizationResult, p: dict, out_dir: str, perm, name: str):
    frames_dir = os.path.join(out_dir, "frames")
    gif_path = os.path.join(out_dir, "optimization.gif")
    save_gif(result.frame_paths, gif_path, fps=5)
    export_xdmf(
        p["nelx"],
        p["nely"],
        result.rho_history,
        perm,
        Lx=p["Lx"],
        Ly=p["Ly"],
        output_dir=out_dir,
    )
    save_summary_slide(
        rho_history=result.rho_history,
        compliance_history=result.metrics["compliance_history"],
        volfrac_history=result.metrics["volfrac_history"],
        change_history=result.metrics["change_history"],
        perm=perm,
        nelx=p["nelx"],
        nely=p["nely"],
        volfrac_target=p["volfrac"],
        out_dir=out_dir,
        problem_name=name,
    )
    final_density_path = save_final_density(
        result.rho_fn, p["nelx"], p["nely"], out_dir, perm
    )
    derivative_path = os.path.join(out_dir, "final_derivatives.npz")
    np.savez_compressed(
        derivative_path,
        dc_drho_phys=result.metrics["final_dc_drho_phys"],
        dc_dx_design=result.metrics["final_dc_dx_design"],
    )
    return {
        "summary_png": "summary.png",
        "final_density_png": os.path.basename(final_density_path),
        "optimization_gif": "optimization.gif",
        "xdmf": "topopt.xdmf",
        "h5": "topopt.h5",
        "final_derivatives_npz": os.path.basename(derivative_path),
    }


def _build_hardcoded_case(case: str):
    if case not in CASE_PARAMS:
        raise ValueError(f"Unknown case '{case}'. Choose from {list(CASE_PARAMS)}")
    p = dict(CASE_PARAMS[case])
    domain = build_mesh(p["nelx"], p["nely"], p["Lx"], p["Ly"])
    V, Q = build_spaces(domain)
    mu, lmbda = get_lame_parameters(p["E"], p["nu"])
    p.update(
        mu=mu,
        lmbda=lmbda,
        element_size=min(p["Lx"] / p["nelx"], p["Ly"] / p["nely"]),
    )
    if case == "cantilever":
        bcs = build_bcs(V, domain)
        F_load = build_load(V, domain, p["Lx"], p["Ly"], p["nely"])
    else:
        bcs = build_bcs_mbb(V, domain, p["Lx"], p["Ly"])
        F_load = build_load_mbb(V, domain, p["Lx"], p["Ly"], p["nely"])
    return p, domain, V, Q, bcs, F_load


def _run_main(case: str):
    p, domain, V, Q, bcs, F_load = _build_hardcoded_case(case)
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "_05_OUT", case)
    _prepare_output_dir(out_dir, clear=True)
    perm = build_cell_perm(
        domain, Q, p["nelx"], p["nely"], p["Lx"], p["Ly"]
    )
    result = _run_optimization(
        domain=domain,
        V=V,
        Q=Q,
        bcs=bcs,
        F_load=F_load,
        p=p,
        frames_dir=os.path.join(out_dir, "frames"),
        perm=perm,
    )
    _export_run(result, p, out_dir, perm, case)


def main():
    _run_main(CASE)



def _location_cell_mask(location: str, nelx: int, nely: int) -> np.ndarray:
    """Cells geometrically adjacent to a named boundary load/support region."""
    mask = np.zeros((nely, nelx), dtype=bool)
    mid_x = [nelx // 2] if nelx % 2 else [nelx // 2 - 1, nelx // 2]
    mid_y = [nely // 2] if nely % 2 else [nely // 2 - 1, nely // 2]
    if location == "left_edge": mask[:, 0] = True
    elif location == "right_edge": mask[:, -1] = True
    elif location == "bottom_edge": mask[0, :] = True
    elif location == "top_edge": mask[-1, :] = True
    elif location == "bottom_left": mask[0, 0] = True
    elif location == "bottom_right": mask[0, -1] = True
    elif location == "top_left": mask[-1, 0] = True
    elif location == "top_right": mask[-1, -1] = True
    elif location in {"right_tip", "right_center"}: mask[mid_y, -1] = True
    elif location == "left_center": mask[mid_y, 0] = True
    elif location == "top_center": mask[-1, mid_x] = True
    elif location == "bottom_center": mask[0, mid_x] = True
    else: raise ValueError(f"Unsupported location for cell mask: {location}")
    return mask


def _problem_region_masks(spec, nelx: int, nely: int):
    support = np.zeros((nely, nelx), dtype=bool)
    load = np.zeros((nely, nelx), dtype=bool)
    for bc in spec.bcs:
        support |= _location_cell_mask(bc.location, nelx, nely)
    for force in spec.loads:
        load |= _location_cell_mask(force.location, nelx, nely)
    return support, load


def main_from_spec(spec, parser_usage=None, out_dir=None, run_provenance=None):
    """Run one parser-generated specification and return a structured packet."""
    from _config_bridge import (
        build_bcs_from_spec,
        build_load_from_spec,
        extract_simp_params,
    )

    base = os.path.dirname(os.path.abspath(__file__))
    if out_dir is None:
        out_dir = os.path.join(base, "_05_OUT", "spec")
        clear = True
    else:
        clear = False
    _prepare_output_dir(out_dir, clear=clear)

    p = extract_simp_params(spec)
    domain = build_mesh(p["nelx"], p["nely"], p["Lx"], p["Ly"])
    V, Q = build_spaces(domain)
    mu, lmbda = get_lame_parameters(p["E"], p["nu"])
    p.update(mu=mu, lmbda=lmbda)

    bcs = build_bcs_from_spec(
        spec, V, p["Lx"], p["Ly"], p["nelx"], p["nely"]
    )
    F_load = build_load_from_spec(
        spec, V, p["Lx"], p["Ly"], p["nelx"], p["nely"]
    )
    perm = build_cell_perm(
        domain, Q, p["nelx"], p["nely"], p["Lx"], p["Ly"]
    )

    print("\n=== VERIFIED SPEC ===")
    print(spec.model_dump_json(indent=2))
    print(
        f"r_min={p['r_min']:.6g} cone-equivalent physical units; "
        f"r_pde={p['r_pde']:.6g}; "
        f"span={p['r_min_elements']:.2f} elements"
    )
    print("=====================\n")

    result = _run_optimization(
        domain=domain,
        V=V,
        Q=Q,
        bcs=bcs,
        F_load=F_load,
        p=p,
        frames_dir=os.path.join(out_dir, "frames"),
        perm=perm,
    )
    artifacts = _export_run(result, p, out_dir, perm, spec.name)

    rho_grid = result.rho_final[perm].reshape((p["nely"], p["nelx"]))
    support_mask, load_mask = _problem_region_masks(spec, p["nelx"], p["nely"])
    validation = validate(
        result.metrics,
        result.rho_final,
        p["volfrac"],
        p["nelx"] * p["nely"],
        cell_volumes=result.cell_volumes,
        r_min=p["r_min"],
        element_size=p["element_size"],
        rho_grid=rho_grid,
        support_cell_mask=support_mask,
        load_cell_mask=load_mask,
    )

    final = {
        "iterations": result.metrics["iteration"],
        "converged": result.metrics["converged"],
        "convergence_reason": result.convergence_reason,
        "final_compliance": result.metrics["compliance_history"][-1],
        "initial_compliance": result.metrics["compliance_history"][0],
        "compliance_reduction_pct": round(
            100.0
            * (
                1.0
                - result.metrics["compliance_history"][-1]
                / result.metrics["compliance_history"][0]
            ),
            2,
        ),
        "final_volume_fraction": float(
            np.dot(result.rho_final, result.cell_volumes)
            / result.cell_volumes.sum()
        ),
        "final_change": result.metrics["change_history"][-1],
        "final_l2_change": result.metrics["l2_change_history"][-1],
    }

    critic_input = {
        "problem": spec.name,
        "parsed_spec": spec.model_dump(),
        "run_config": {
            "nelx": p["nelx"],
            "nely": p["nely"],
            "Lx": p["Lx"],
            "Ly": p["Ly"],
            "E": p["E"],
            "nu": p["nu"],
            "kinematics": "2-D small-strain plane stress",
            "execution": "serial",
            "volfrac_target": p["volfrac"],
            "penal": p["penal"],
            "r_min_convention": p["r_min_convention"],
            "r_min": p["r_min"],
            "r_pde": p["r_pde"],
            "r_min_elements": p["r_min_elements"],
            "filter": "-r_pde^2*Laplacian(rho_tilde)+rho_tilde=rho",
            "projection": "single intermediate Heaviside projection, eta=0.5",
            "robust_three_projection": False,
            "beta_schedule_requested": REQUESTED_BETA_SCHEDULE,
            "beta_schedule_effective": result.beta_schedule_effective,
            "beta_final": result.beta_final,
            "steering_enabled": False,
        },
        "final_result": final,
        "validation": validation,
        "artifacts": artifacts,
        "evidence_scope": {
            "topology_image_content_included": False,
            "artifact_filenames_are_visual_evidence": False,
            "spatial_claims_allowed_only_from_deterministic_metrics": True,
        },
    }

    with open(os.path.join(out_dir, "critic_input.json"), "w") as handle:
        json.dump(critic_input, handle, indent=2)

    print("\n--- Deterministic Validation ---")
    print(f"HARD CHECKS PASSED: {validation['passed']}")
    for name, check in validation["checks"].items():
        marker = "OK" if check["passed"] else "FAIL"
        print(
            f"  {marker:4s} [{check['severity']}] {name}: "
            f"{check['value']} (threshold: {check['threshold']})"
        )
    for message in validation["failure_reasons"]:
        print(f"  HARD FAILURE: {message}")
    for message in validation.get("quality_warnings", []):
        print(f"  QUALITY WARNING: {message}")

    if validation["passed"]:
        critic_summary, critic_usage = criticize(critic_input)
        print("\n--- Critic Agent Summary ---")
        print(critic_summary)
    else:
        critic_summary = "Critic Agent not called because hard validation failed."
        critic_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    with open(os.path.join(out_dir, "critic_summary.txt"), "w") as handle:
        handle.write(critic_summary)

    parser_tokens = int(parser_usage["total_tokens"]) if parser_usage else 0
    compute_cost = {
        "optimization_iterations": result.metrics["iteration"],
        "parser_tokens": parser_tokens if parser_usage else None,
        "critic_tokens": int(critic_usage.get("total_tokens", 0)),
        "total_llm_tokens": parser_tokens + int(critic_usage.get("total_tokens", 0)),
    }
    if run_provenance is None:
        run_provenance = {
            "clarification_policy": "unknown",
            "defaulted_fields": [],
            "clarifications_presented": [],
            "user_overrides": [],
            "confirmation_received": False,
        }

    derivative_evidence = {
        "direct_simp_sign_derivative": {
            "key": "final_dc_drho_phys",
            "notation": "dC/d(rho_phys)",
            "shape": list(np.asarray(result.metrics["final_dc_drho_phys"]).shape),
            "artifact": artifacts["final_derivatives_npz"],
        },
        "mma_design_gradient": {
            "key": "final_dc_dx_design",
            "notation": "H^T[(dC/d(rho_phys))*(d(rho_phys)/d(rho_tilde))]",
            "shape": list(np.asarray(result.metrics["final_dc_dx_design"]).shape),
            "artifact": artifacts["final_derivatives_npz"],
        },
    }

    packet = {
        "critic_agent_summary": critic_summary,
        "final_result": final,
        "validation": validation,
        "what_claude_saw": critic_input,
        "artifact_files": artifacts,
        "compute_cost": compute_cost,
        "derivative_evidence": derivative_evidence,
        "interaction_provenance": run_provenance,
        "model_limitations": {
            "point_force": "discrete nodal point force; local stresses at the loaded node are singularity-sensitive",
            "dirichlet_values": "homogeneous displacement constraints only",
            "critic_visual_access": False,
        },
    }
    with open(os.path.join(out_dir, "result_packet.json"), "w") as handle:
        json.dump(packet, handle, indent=2)
    print("\n=== RUN COMPLETE ===")
    print(f"Validation passed: {validation['passed']}")
    print(f"Iterations:        {final['iterations']}")
    print(f"Final compliance:  {final['final_compliance']:.8g}")
    print(f"Final volume:      {final['final_volume_fraction']:.6f}")
    print(f"Output directory:  {out_dir}")
    print("====================")
    return packet


if __name__ == "__main__":
    main()
