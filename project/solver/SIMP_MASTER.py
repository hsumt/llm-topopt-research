"""
SIMP_MASTER.py
Master execute controller.

"""

import os
import numpy as np
from dolfinx import fem

from _01_MSH._mesh           import build_mesh
from _01_MSH._boundaries     import (build_bcs, build_load,
                                     build_bcs_mbb,   build_load_mbb,
                                     build_bcs_michell, build_load_michell)
from _01_MSH._domain         import get_lame_parameters

from _02_FEA._functionspaces import build_spaces
from _02_FEA._solver         import solve_fea

from _03_OPTMZER._filters    import build_helmholtz_filter_CG1
from _03_OPTMZER._objective  import compute_compliance, compute_sensitivities
from _03_OPTMZER._MMAupdate  import MMAOptimizer

from _04_PPRCS.postprocess   import (save_frame, save_gif,
                                     export_xdmf, print_iteration_report,
                                     build_cell_perm)

from agent._steering import steer_code
from agent._physics  import validate
from agent._critic   import criticize


# ---------------------------------------------------------------------------
# SELECT BENCHMARK CASE
# Comment out two, leave one active. Ctrl+/ toggles a line.
# ---------------------------------------------------------------------------
# CASE = "cantilever"
CASE = "mbb"
# DO NOT USE: CASE = "michell"

#
# nelx, nely : mesh resolution (elements)
# Lx, Ly     : domain dimensions
# volfrac    : volume fraction target V* (Sigmund 2001, Eq. 2)
# penal      : SIMP penalisation exponent p (Bendsoe & Sigmund 2003, Eq. 1.4)
# r_min      : Helmholtz filter radius (Lazarov & Sigmund 2016, Eq. 1)
# max_iter   : hard iteration cap
# tol_change : inf-norm convergence threshold on design variable
# E, nu      : Young's modulus, Poisson's ratio (non-dimensionalised)
# ---------------------------------------------------------------------------
CASE_PARAMS = {
    "cantilever": {
        # Sigmund (2001), Struct Multidisc Optim 21:120-127, Section 2
        # 2:1 aspect ratio, center-right load, fully clamped left edge
        "nelx":       80,
        "nely":       50,
        "Lx":         1.6,
        "Ly":         1.0,
        "volfrac":    0.4,
        "penal":      3.0,
        "r_min":      0.05,
        "max_iter":   200,
        "tol_change": 0.01,
        "E":          1.0,
        "nu":         0.3,
    },
    "mbb": {
        # Sigmund (2001), Section 3 — half-symmetry MBB beam
        # 3:1 aspect ratio is the standard benchmark geometry.
        # Half-symmetry: model is the RIGHT half (x in [0, Lx]).
        # Left edge: symmetry plane → u_x = 0 (horizontal locked, vertical free)
        # Bottom-right corner: roller → u_y = 0
        # Load: F_y = -1 at top-LEFT corner (the symmetry edge, x=0, y=Ly)
        "nelx":       120,
        "nely":       40,
        "Lx":         3.0,
        "Ly":         1.0,
        "volfrac":    0.4,
        "penal":      3.0,
        "r_min":      0.06,
        "max_iter":   400,
        "tol_change": 0.01,
        "E":          1.0,
        "nu":         0.3,
    },
    # FORCE-PARKED 
    # Status: stable but not matching expected reference fan toplogy. requires reformulation and literature investigation.
    # 
    # "michell": {
    #     # Michell (1904), Phil. Mag. 8(47):589-597
    #     # Reconstructed benchmark: Bendsoe & Sigmund (2003), Chapter 1
    #     # Square domain; two pin supports at bottom corners; center-top load.
    #     # Analytical solution is the Michell truss — two symmetric fan regions.
    #     "nelx":       200,
    #     "nely":       200,
    #     "Lx":         2.0,
    #     "Ly":         2.0,
    #     "volfrac":    0.12,
    #     "penal":      3.0,
    #     "r_min":      0.015,
    #     "max_iter":   300,
    #     "tol_change": 0.01,
    #     "E":          1.0,
    #     "nu":         0.3,
    # },
}


def _run_main(case: str):
    """
    Shared SIMP loop for all hardcoded benchmark cases.
    No agents. Direct physics verification path.

    Parameters
    ----------
    case : str
        One of "cantilever", "mbb", "michell".
        Must match a key in CASE_PARAMS and map to a valid
        (build_bcs_*, build_load_*) pair below.
    """
    p = CASE_PARAMS[case]

    nelx       = p["nelx"]
    nely       = p["nely"]
    Lx         = p["Lx"]
    Ly         = p["Ly"]
    volfrac    = p["volfrac"]
    penal      = p["penal"]
    r_min      = p["r_min"]
    max_iter   = p["max_iter"]
    tol_change = p["tol_change"]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR  = os.path.join(BASE_DIR, "_05_OUT", case)
    out_dir  = os.path.join(OUT_DIR, "frames")
    gif_path = os.path.join(OUT_DIR, "optimization.gif")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {case.upper()}")
    print(f"  nelx={nelx}  nely={nely}  Lx={Lx}  Ly={Ly}")
    print(f"  volfrac={volfrac}  penal={penal}  r_min={r_min}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. MESH + SPACES
    # ------------------------------------------------------------------
    domain    = build_mesh(nelx, nely, Lx, Ly)
    V, Q      = build_spaces(domain)
    mu, lmbda = get_lame_parameters(p["E"], p["nu"])

    # ------------------------------------------------------------------
    # 2. BOUNDARY CONDITIONS + LOAD
    # ------------------------------------------------------------------
    # Each case uses a dedicated build_bcs_* / build_load_* function.
    # The cantilever uses the existing build_bcs / build_load from
    # _boundaries.py (unchanged, already validated).
    #
    # MBB and Michell use functions that follow the same V.sub(1).collapse()
    # pattern as the working cantilever build_load — no raw DOF index
    # arithmetic — to avoid the 2*i+1 interleave assumption.
    # ------------------------------------------------------------------
    if case == "cantilever":
        bcs    = build_bcs(V, domain)
        F_load = build_load(V, domain, Lx, Ly, nely)

    elif case == "mbb":
        bcs    = build_bcs_mbb(V, domain, Lx, Ly)
        F_load = build_load_mbb(V, domain, Lx, Ly, nely)

    elif case == "michell":
        bcs    = build_bcs_michell(V, domain, Lx, Ly)
        F_load = build_load_michell(V, domain, Lx, Ly, nelx, nely)

    else:
        raise ValueError(f"Unknown case: '{case}'. Choose 'cantilever', 'mbb', or 'michell'.")

    print("--- BC / Load diagnostics ---")
    for i, bc in enumerate(bcs):

        parent, _ = bc.dof_indices()
        n = len(parent)
        print(f"  BC[{i}]  n_constrained_dofs = {n}")
        if n == 0:
            raise RuntimeError(
                f"BC[{i}] has 0 constrained DOFs. "
                f"Check predicate for case '{case}'."
            )

    nnz = int(np.count_nonzero(F_load.x.array))
    fmax = float(F_load.x.array.max())
    fmin = float(F_load.x.array.min())
    print(f"  F_load  nonzero_dofs={nnz}  max={fmax:.6f}  min={fmin:.6f}")
    if nnz == 0:
        raise RuntimeError(
            f"F_load has no nonzero entries. Load was not applied. "
            f"Check load predicate tolerance for case '{case}'."
        )
    print()

    # ------------------------------------------------------------------
    # 4. FILTER + OPTIMIZER SETUP
    # ------------------------------------------------------------------
    domain.topology.create_entities(domain.topology.dim)
    n_cells = Q.dofmap.index_map.size_local
    perm    = build_cell_perm(domain, Q, nelx, nely, Lx, Ly)

    rho_fn      = fem.Function(Q)
    rho_fn.name = "density"

    apply_filter, apply_sens_filter = build_helmholtz_filter_CG1(domain, Q, r_min)
    optimizer   = MMAOptimizer(n=n_cells, x_min=1e-3, x_max=1.0, move=0.2)
    dg_drho     = np.ones(n_cells) / n_cells   # dg/dx_e = 1/n for volume constraint

    # ------------------------------------------------------------------
    # 5. OPTIMIZATION LOOP
    # ------------------------------------------------------------------
    # x_design : design variable (unfiltered) — MMA reads and writes this.
    # rho_fn   : filtered physical density — FEA reads this. NEVER used as
    #            MMA input. (Bug #1 fix.)
    # ------------------------------------------------------------------
    x_design           = np.full(n_cells, volfrac)
    frame_paths        = []
    rho_history        = []
    compliance_history = []

    for iteration in range(1, max_iter + 1):

        # Physical density = filtered design variable
        # Clip to [1e-3, 1.0]: lower bound prevents singular K (void = soft,
        # not zero stiffness). Upper bound is physical material limit.
        rho_tilde = np.clip(apply_filter(x_design), 1e-3, 1.0)
        rho_fn.x.array[:] = rho_tilde

        # FEA: solve K(ρ̃) u = F
        uh = solve_fea(domain, V, bcs, rho_fn, penal, mu, lmbda, F_load)

        # Compliance: C = F^T u  (Sigmund 2001, Eq. 1)
        compliance = compute_compliance(uh, F_load)
        compliance_history.append(float(compliance))

        # Sensitivities dC/dρ̃_e (Sigmund 2001, Eq. 5)
        dc_drho = compute_sensitivities(rho_tilde, uh, Q, penal, mu, lmbda)

        # Chain rule through Helmholtz filter: dC/dx = H^T(dC/dρ̃) = H(dC/dρ̃)
        # H is self-adjoint → applying filter to sensitivity is exact chain rule.
        # (Lazarov & Sigmund 2016, Section 2.3)
        dc_drho_filtered = apply_sens_filter(dc_drho)
        # dc_drho_filtered = apply_filter(dc_drho)

        # Volume constraint: g = (Σ x_e / n) − V* ≤ 0
        volfrac_actual = x_design.sum() / n_cells
        g_val          = volfrac_actual - volfrac

        # MMA update on design variable only
        x_new = optimizer.update(
            x     = x_design.copy(),
            f0val = compliance,
            df0dx = dc_drho_filtered,
            fval  = g_val,
            dfdx  = dg_drho,
        )

        # Compute change BEFORE committing update (Bug #2 fix)
        change   = float(np.max(np.abs(x_new - x_design)))
        x_design = x_new.copy()

        # Re-filter for visualization (so saved frame reflects updated design)
        rho_fn.x.array[:] = np.clip(apply_filter(x_design), 1e-3, 1.0)

        volfrac_report = rho_fn.x.array.sum() / n_cells
        print_iteration_report(iteration, compliance, volfrac_report, change)

        frame_paths.append(
            save_frame(rho_fn, nelx, nely, iteration, compliance, out_dir, perm)
        )
        rho_history.append(rho_fn.x.array.copy())
        # if iteration == 50:
        #     penal = min(penal + 1.0, 3.0)
        #     print(f"  [Continuation] penal → {penal:.1f}")
        # if iteration == 100:
        #     penal = min(penal + 1.0, 3.0)
        #     print(f"  [Continuation] penal → {penal:.1f}")

        # Convergence check 1: compliance plateau.
        # Relative drop in compliance over the last 5 iterations < 1e-4.
        # This fires when the physics has converged even if MMA is still
        # cycling — which it does near the optimum (limit cycling).
        # Without this check, the loop runs to max_iter unnecessarily.
        # Reference: standard practice; see Sigmund (2001), Section 2 notes.
        design_converged = change < tol_change

        compliance_converged = False
        if iteration > 30 and len(compliance_history) >= 21:
            recent_drop = abs(compliance_history[-1] - compliance_history[-21]) / (
                abs(compliance_history[-1]) + 1e-12
            )
            compliance_converged = recent_drop < 5e-4

        if iteration > 80 and (design_converged or compliance_converged):
            reason = "design change" if design_converged else "compliance plateau"
            print(f"\nConverged at iteration {iteration} ({reason})")
            break

    # ------------------------------------------------------------------
    # 6. EXPORT
    # ------------------------------------------------------------------
    save_gif(frame_paths, gif_path, fps=5)
    print(f"\nGIF saved:  {gif_path}")
    export_xdmf(nelx, nely, rho_history, perm, output_dir=OUT_DIR)
    print(f"XDMF saved: {OUT_DIR}")


def main():
    """
    Hardcoded benchmark runner. No agents. Physics verification path.

    To switch benchmark:
        Comment out two CASE = "..." lines at the top of this file,
        leave one active. One Ctrl+/ per line.
    """
    _run_main(CASE)


def main_from_spec(spec):
    """
    Entry point driven by the parser agent.
    Accepts a ProblemSpec Pydantic object, runs the full SIMP loop with agents.
    """
    from _config_bridge import (
        build_bcs_from_spec,
        build_load_from_spec,
        extract_simp_params,
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR  = os.path.join(BASE_DIR, "_05_OUT", "spec")
    out_dir  = os.path.join(OUT_DIR, "frames")
    gif_path = os.path.join(OUT_DIR, "optimization.gif")

    os.makedirs(out_dir, exist_ok=True)

    p  = extract_simp_params(spec)
    Lx = p["Lx"]
    Ly = p["Ly"]

    domain    = build_mesh(p["nelx"], p["nely"], Lx, Ly)
    V, Q      = build_spaces(domain)
    mu, lmbda = get_lame_parameters(p["E"], p["nu"])

    bcs    = build_bcs_from_spec(spec, V, Lx, Ly, p["nelx"], p["nely"])
    F_load = build_load_from_spec(spec, V, Lx, Ly, p["nelx"], p["nely"])

    domain.topology.create_entities(domain.topology.dim)
    n_cells = Q.dofmap.index_map.size_local
    perm    = build_cell_perm(domain, Q, p["nelx"], p["nely"], Lx, Ly)

    rho_fn      = fem.Function(Q)
    rho_fn.name = "density"

    apply_filter, apply_sens_filter = build_helmholtz_filter_CG1(domain, Q, p["r_min"])
    optimizer   = MMAOptimizer(n=n_cells, x_min=1e-3, x_max=1.0, move = 0.2)
    dg_drho     = np.ones(n_cells) / n_cells

    metrics = {
        "compliance_history": [],
        "volfrac_history":    [],
        "change_history":     [],
        "l2_change_history":  [],
        "iteration":          0,
        "converged":          False,
    }
    STEER_EVERY = 20
    live_params = {
        "penal":   p["penal"],
        "r_min":   p["r_min"],
        "volfrac": p["volfrac"],
    }

    x_design    = np.full(n_cells, p["volfrac"])
    frame_paths = []
    rho_history = []

    print("\n=== SPEC DUMP ===")
    print(spec.model_dump_json(indent=2))
    print(f"Lx={Lx}, Ly={Ly}")
    print(f"n BCs: {len(bcs)}")
    print(f"F_load nonzero DOFs: {np.count_nonzero(F_load.x.array)}")
    print(f"F_load max: {F_load.x.array.max():.6f}, min: {F_load.x.array.min():.6f}")
    print("=================\n")

    for iteration in range(1, p["max_iter"] +1):

        rho_tilde = np.clip(apply_filter(x_design), 1e-3, 1.0)
        rho_fn.x.array[:] = rho_tilde

        uh = solve_fea(
            domain, V, bcs, rho_fn,
            live_params["penal"], mu, lmbda, F_load
        )

        compliance = compute_compliance(uh, F_load)

        dc_drho = compute_sensitivities(
            rho_tilde, uh, Q, live_params["penal"], mu, lmbda
        )

        dc_drho_filtered = apply_sens_filter(dc_drho)

        volfrac_actual = x_design.sum() / n_cells
        g_val          = volfrac_actual - live_params["volfrac"]

        x_new = optimizer.update(
            x     = x_design.copy(),
            f0val = compliance,
            df0dx = dc_drho_filtered,
            fval  = g_val,
            dfdx  = dg_drho,
        )

        # Bug #2 fix: compute BEFORE update
        change    = float(np.max(np.abs(x_new - x_design)))
        l2_change = float(np.linalg.norm(x_new - x_design) / np.sqrt(n_cells))

        x_design = x_new.copy()

        rho_fn.x.array[:] = np.clip(apply_filter(x_design), 1e-3, 1.0)

        metrics["compliance_history"].append(float(compliance))
        metrics["volfrac_history"].append(float(volfrac_actual))
        metrics["change_history"].append(float(change))
        metrics["l2_change_history"].append(l2_change)
        metrics["iteration"] = iteration

        volfrac_report = rho_fn.x.array.sum() / n_cells
        print_iteration_report(iteration, compliance, volfrac_report, change)
        frame_paths.append(
            save_frame(rho_fn, p["nelx"], p["nely"],
                       iteration, compliance, out_dir, perm)
        )
        rho_history.append(rho_fn.x.array.copy())

        tol_change = p.get("tol_change", 0.01)

        design_converged = change < tol_change

        compliance_converged = False
        if iteration > 30 and len(metrics["compliance_history"]) >= 21:
            recent_drop = abs(
                metrics["compliance_history"][-1] - metrics["compliance_history"][-21]
            ) / (abs(metrics["compliance_history"][-1]) + 1e-12)
            compliance_converged = recent_drop < 5e-4

        if iteration > 80 and (design_converged or compliance_converged):
            reason = "design change" if design_converged else "compliance plateau"
            print(f"\nConverged at iteration {iteration} ({reason})")
            metrics["converged"] = True
            break

    save_gif(frame_paths, gif_path, fps=5)
    export_xdmf(p["nelx"], p["nely"], rho_history, perm, output_dir=OUT_DIR)
    print(f"GIF saved: {gif_path}")

    from _04_PPRCS.postprocess import save_summary_slide
    save_summary_slide(
        rho_history=rho_history,
        compliance_history=metrics["compliance_history"],
        volfrac_history=metrics["volfrac_history"],
        change_history=metrics["change_history"],
        perm=perm,
        nelx=p["nelx"],
        nely=p["nely"],
        volfrac_target=p["volfrac"],
        out_dir=OUT_DIR,
        problem_name=spec.name,
    )

    val_result = validate(metrics, rho_fn.x.array, p["volfrac"], n_cells)
    print("\n--- Physics Validation ---")
    print(f"PASSED: {val_result['passed']}")
    for name, chk in val_result["checks"].items():
        status = "✓" if chk["passed"] else "✗"
        print(f"  {status} {name}: {chk['value']} (threshold: {chk['threshold']})")
    if val_result["failure_reasons"]:
        for r in val_result["failure_reasons"]:
            print(f"  FAIL: {r}")

    if val_result["passed"]:
        print("\n--- Critic Agent Summary ---")
        summary = criticize(metrics, val_result, spec.name)
        print(summary)
        with open(os.path.join(OUT_DIR, "critic_summary.txt"), "w") as f:
            f.write(summary)
    else:
        print("\nPhysics validation failed — Critic Agent not called.")


if __name__ == "__main__":
    main()