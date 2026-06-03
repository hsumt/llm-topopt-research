"""
SIMP_MASTER.py
Master execute controller.

CHANGED (Bug #1): main() — introduced x_design as a separate array distinct
  from rho_fn.x.array. Previously, apply_filter() overwrote rho_fn.x.array
  (the design variable) with the filtered physical density. MMA then received
  the filtered density as its design input, corrupting sensitivities and update
  steps. x_design is now the canonical design variable; rho_fn holds only the
  filtered physical density used by FEA and visualization.

CHANGED (Bug #2): main_from_spec() step 9 — l2_change_history was computed
  AFTER x_design = x_new.copy(), making x_new - x_design identically zero.
  Moved computation to BEFORE the update.

CHANGED (Bug #7): removed redundant metrics["converged"] assignment after loop.
"""

import os
import numpy as np
from dolfinx import fem

from _01_MSH._mesh           import build_mesh
from _01_MSH._boundaries     import build_bcs, build_load
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


def main():
    """
    Hardcoded cantilever benchmark (Sigmund 2001).
    No agents. Direct physics verification path.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR  = os.path.join(BASE_DIR, "_05_OUT")
    out_dir  = os.path.join(OUT_DIR, "frames")
    gif_path = os.path.join(OUT_DIR, "optimization.gif")

    # -----------------------------------------------------------------------
    # 1. PARAMETERS
    # -----------------------------------------------------------------------
    nelx       = 80
    nely       = 50
    Lx         = 1.6
    Ly         = 1.0
    volfrac    = 0.4
    penal      = 3.0
    r_min      = 0.05
    max_iter   = 100
    tol_change = 0.01

    # -----------------------------------------------------------------------
    # 2. SETUP
    # -----------------------------------------------------------------------
    domain    = build_mesh(nelx, nely, Lx, Ly)
    V, Q      = build_spaces(domain)
    mu, lmbda = get_lame_parameters(1.0, 0.3)
    bcs       = build_bcs(V, domain)
    F_load    = build_load(V, domain, Lx, Ly, nely)   # center-right load (task 1.1)

    domain.topology.create_entities(domain.topology.dim)
    n_cells = Q.dofmap.index_map.size_local
    perm    = build_cell_perm(domain, Q, nelx, nely, Lx, Ly)

    rho_fn = fem.Function(Q)
    rho_fn.name = "density"

    apply_filter, apply_sens_filter = build_helmholtz_filter_CG1(domain, Q, r_min)
    optimizer = MMAOptimizer(n=n_cells, x_min=1e-3, x_max=1.0)
    dg_drho   = np.ones(n_cells) / n_cells

    # -----------------------------------------------------------------------
    # 3. OPTIMIZATION LOOP
    # -----------------------------------------------------------------------
    # CHANGED (Bug #1): x_design is the design variable (unfiltered densities).
    # rho_fn holds apply_filter(x_design) — the physical density seen by FEA.
    # These are SEPARATE arrays. MMA reads/writes x_design only.
    x_design    = np.full(n_cells, volfrac)
    frame_paths = []
    rho_history = []
    compliance_history = []

    for iteration in range(1, max_iter + 1):

        # Physical density = filtered design variable
        rho_tilde = np.clip(apply_filter(x_design), 1e-3, 1.0)
        rho_fn.x.array[:] = rho_tilde

        # FEA on physical density
        uh = solve_fea(domain, V, bcs, rho_fn, penal, mu, lmbda, F_load)

        # Compliance: C = F^T U  (Sigmund 2001, Eq. 1)
        compliance = compute_compliance(uh, F_load)
        compliance_history.append(float(compliance))

        # Sensitivities w.r.t. physical density ρ̃
        dc_drho = compute_sensitivities(rho_tilde, uh, Q, penal, mu, lmbda)

        # Chain rule through filter: dC/dx = H^T (dC/dρ̃) = H(dC/dρ̃)
        # (Lazarov & Sigmund 2016, Section 2.3 — H is self-adjoint)
        dc_drho_filtered = apply_sens_filter(dc_drho)

        # Volume constraint on design variable: g = (Σx_e / n) − V* ≤ 0
        volfrac_actual = x_design.sum() / n_cells
        g_val          = volfrac_actual - volfrac

        # MMA update on design variable
        x_new = optimizer.update(
            x     = x_design.copy(),
            f0val = compliance,
            df0dx = dc_drho_filtered,
            fval  = g_val,
            dfdx  = dg_drho,
        )

        # Inf-norm change on design variable (convergence criterion)
        change   = float(np.max(np.abs(x_new - x_design)))
        x_design = x_new.copy()

        # Update rho_fn for this iteration's visualization
        rho_fn.x.array[:] = np.clip(apply_filter(x_design), 1e-3, 1.0)

        volfrac_report = rho_fn.x.array.sum() / n_cells
        print_iteration_report(iteration, compliance, volfrac_report, change)

        frame_paths.append(
            save_frame(rho_fn, nelx, nely, iteration, compliance, out_dir, perm)
        )
        rho_history.append(rho_fn.x.array.copy())

        compliance_converged = False
        if iteration > 15:
            recent_drop = abs(compliance_history[-1] - compliance_history[-6]) / \
                        (abs(compliance_history[-1]) + 1e-12)
            compliance_converged = recent_drop < 1e-4

        if (change < 0.01 or compliance_converged) and iteration > 10:
            reason = "change" if change < 0.01 else "compliance plateau"
            print(f"\nConverged at iteration {iteration} ({reason})")
            break

    # -----------------------------------------------------------------------
    # 4. EXPORT
    # -----------------------------------------------------------------------
    save_gif(frame_paths, gif_path, fps=5)
    print(f"GIF saved:  {gif_path}")
    export_xdmf(nelx, nely, rho_history, perm, output_dir=OUT_DIR)
    # -----------------------------------------------------------------------
    # EXPORT
    # -----------------------------------------------------------------------


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
    OUT_DIR  = os.path.join(BASE_DIR, "_05_OUT")
    out_dir  = os.path.join(OUT_DIR, "frames")
    gif_path = os.path.join(OUT_DIR, "optimization.gif")

    # CHANGED: extract_simp_params now returns Lx/Ly (schema or default).
    # main_from_spec no longer recomputes them independently.
    p  = extract_simp_params(spec)
    Lx = p["Lx"]
    Ly = p["Ly"]

    domain    = build_mesh(p["nelx"], p["nely"], Lx, Ly)
    coords = domain.geometry.x
    print("y range:", coords[:,1].min(), coords[:,1].max())
    # Find nodes near x=Lx
    right_nodes = coords[np.isclose(coords[:,0], Lx)]
    print("Right edge y values:", np.sort(right_nodes[:,1]))
    V, Q      = build_spaces(domain)
    mu, lmbda = get_lame_parameters(p["E"], p["nu"])

    # CHANGED: pass nelx/nely so center-edge predicates get correct tolerance
    bcs    = build_bcs_from_spec(spec, V, Lx, Ly, p["nelx"], p["nely"])
    print(f"BC 0 n_dofs: {len(bcs[0]._cpp_object.dof_indices())}")
    print(f"BC 1 n_dofs: {len(bcs[1]._cpp_object.dof_indices())}")
    F_load = build_load_from_spec(spec, V, Lx, Ly, p["nelx"], p["nely"])

    domain.topology.create_entities(domain.topology.dim)
    n_cells = Q.dofmap.index_map.size_local
    perm    = build_cell_perm(domain, Q, p["nelx"], p["nely"], Lx, Ly)

    rho_fn = fem.Function(Q)
    rho_fn.name = "density"

    apply_filter, apply_sens_filter = build_helmholtz_filter_CG1(
        domain, Q, p["r_min"]
    )
    optimizer = MMAOptimizer(n=n_cells, x_min=1e-3, x_max=1.0)
    dg_drho   = np.ones(n_cells) / n_cells

    # -----------------------------------------------------------------------
    # Metrics + steering config
    # -----------------------------------------------------------------------
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

    # CHANGED (Bug #1): x_design is the design variable.
    # rho_fn holds apply_filter(x_design) — never used as design variable input.
    x_design = np.full(n_cells, p["volfrac"])

    frame_paths = []
    rho_history = []
    print("\n=== SPEC DUMP ===")
    print(spec.model_dump_json(indent=2))
    print(f"Lx={Lx}, Ly={Ly}")
    print(f"n BCs: {len(bcs)}")
    print(f"F_load nonzero DOFs: {np.count_nonzero(F_load.x.array)}")
    print(f"F_load max: {F_load.x.array.max():.6f}, min: {F_load.x.array.min():.6f}")
    print("=================\n")
    for iteration in range(1, 301):

        # 1. Physical density = filtered design variable
        rho_tilde = np.clip(apply_filter(x_design), 1e-3, 1.0)
        rho_fn.x.array[:] = rho_tilde
        # if iteration == 50:
        #     live_params["penal"] = min(live_params["penal"] + 1.0, 5.0)
        #     print(f"[Continuation] penal → {live_params['penal']}")
        # if iteration == 100:
        #     live_params["penal"] = min(live_params["penal"] + 1.0, 5.0)
        #     print(f"[Continuation] penal → {live_params['penal']}")
        # 2. FEA
        uh = solve_fea(
            domain, V, bcs, rho_fn,
            live_params["penal"], mu, lmbda, F_load
        )

        # 3. Compliance
        compliance = compute_compliance(uh, F_load)

        # 4. Sensitivities w.r.t. physical density ρ̃
        dc_drho = compute_sensitivities(
            rho_tilde, uh, Q, live_params["penal"], mu, lmbda
        )

        # 5. Chain rule through filter back to design space
        dc_drho_filtered = apply_sens_filter(dc_drho)

        # 6. Volume constraint on design variable
        volfrac_actual = x_design.sum() / n_cells
        g_val          = volfrac_actual - live_params["volfrac"]

        # 7. MMA update on design variable
        x_new = optimizer.update(
            x     = x_design.copy(),
            f0val = compliance,
            df0dx = dc_drho_filtered,
            fval  = g_val,
            dfdx  = dg_drho,
        )

        # CHANGED (Bug #2): compute BOTH change metrics BEFORE updating x_design.
        # Previously x_design was set to x_new first, making x_new - x_design = 0.
        change    = float(np.max(np.abs(x_new - x_design)))
        l2_change = float(np.linalg.norm(x_new - x_design) / np.sqrt(n_cells))

        # 8. Commit update
        x_design = x_new.copy()
        # if (not hasattr(optimizer, '_continuation_done') 
        #         and change < 0.02 
        #         and iteration > 50
        #         and live_params["penal"] < 4.0):
        #     live_params["penal"] = 4.0
        #     optimizer._continuation_done = True
        #     print(f"[Continuation] penal 3.0 → 4.0 at iter {iteration} (change < 0.02)")

        # Update rho_fn for visualization (filter the new design variable)
        rho_fn.x.array[:] = np.clip(apply_filter(x_design), 1e-3, 1.0)

        # 9. Record metrics
        metrics["compliance_history"].append(float(compliance))
        metrics["volfrac_history"].append(float(volfrac_actual))
        metrics["change_history"].append(float(change))
        metrics["l2_change_history"].append(l2_change)   # now non-zero
        metrics["iteration"] = iteration

        # 10. Steering agent
        # if iteration % STEER_EVERY == 0:
        #     prev_penal = live_params["penal"]
        #     live_params = steer_code(metrics, live_params)
        #     if live_params["penal"] != prev_penal:
        #         # Reset plateau window so compliance spike doesn't trigger false convergence
        #         metrics["compliance_history"] = metrics["compliance_history"][-3:]
        #         print(f"[Continuation] penal {prev_penal} → {live_params['penal']}, plateau window reset")

        # 11. Report + save
        volfrac_report = rho_fn.x.array.sum() / n_cells
        print_iteration_report(iteration, compliance, volfrac_report, change)
        frame_paths.append(
            save_frame(rho_fn, p["nelx"], p["nely"],
                       iteration, compliance, out_dir, perm)
        )
        rho_history.append(rho_fn.x.array.copy())

        # 12. Convergence
        # CHANGED (Bug #7): removed redundant metrics["converged"] after loop.
        # The flag is set here and only here.
        # TOL_CHANGE     = 0.01
        # TOL_COMPLIANCE = 5e-4   # 0.05% relative improvement over 5 iters

        # compliance_converged = False
        # if iteration > 15 and len(metrics["compliance_history"]) >= 6:
        #     c_hist = metrics["compliance_history"]
        #     recent_drop = abs(c_hist[-1] - c_hist[-6]) / (abs(c_hist[-1]) + 1e-12)
        #     compliance_converged = recent_drop < TOL_COMPLIANCE

        # if (change < TOL_CHANGE or compliance_converged) and iteration > 10:
        #     reason = "change" if change < TOL_CHANGE else "compliance plateau"
        #     print(f"\nConverged at iteration {iteration} ({reason})")
        #     metrics["converged"] = True
        #     break
        TOL_CHANGE = 0.005

        if change < TOL_CHANGE and iteration > 10:
            print(f"\nConverged at iteration {iteration} (change = {change:.6f})")
            metrics["converged"] = True
            break

    # -----------------------------------------------------------------------
    # EXPORT
    # -----------------------------------------------------------------------
    save_gif(frame_paths, gif_path, fps=5)
    export_xdmf(p["nelx"], p["nely"], rho_history, perm, output_dir=OUT_DIR)
    print(f"GIF saved: {gif_path}")

    # ADD THIS:
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
    print(f"GIF saved: {gif_path}")

    # -----------------------------------------------------------------------
    # VALIDATION + CRITIC
    # -----------------------------------------------------------------------
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