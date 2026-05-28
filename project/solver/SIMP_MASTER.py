"""
SIMP_MASTER.py
Master execute controller
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

from agent._steering   import steer_code
from agent._physics import validate
from agent._critic     import criticize
def main():

    # -----------------------------------------------------------------------
    # 0. PATHS — resolved relative to this file, not the working directory
    # -----------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR  = os.path.join(BASE_DIR, "_05_OUT")
    out_dir  = os.path.join(OUT_DIR, "frames")
    gif_path = os.path.join(OUT_DIR, "optimization.gif")

    # -----------------------------------------------------------------------
    # 1. PARAMETERS — matching Sigmund (2001) cantilever benchmark
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
    F_load    = build_load(V, domain, Lx, Ly)

    domain.topology.create_entities(domain.topology.dim)
    n_cells = Q.dofmap.index_map.size_local

    # DOLFINx create_rectangle (quad) uses diagonal cell ordering, not row-major.
    # perm[iy*nelx+ix] = DOF index for spatial cell (ix, iy) — used for visualization.
    perm = build_cell_perm(domain, Q, nelx, nely, Lx, Ly)

    rho_fn = fem.Function(Q)
    rho_fn.name = "density"
    rho_fn.x.array[:] = volfrac

    apply_filter, apply_sens_filter = build_helmholtz_filter_CG1(
        domain, Q, r_min
    )

    optimizer = MMAOptimizer(n=n_cells, x_min=1e-3, x_max=1.0)

    dg_drho = np.ones(n_cells) / n_cells

    # -----------------------------------------------------------------------
    # 3. OPTIMIZATION LOOP
    # -----------------------------------------------------------------------
    frame_paths = []
    rho_history = []
    rho_old     = rho_fn.x.array.copy()

    for iteration in range(1, max_iter + 1):

        rho_tilde = apply_filter(rho_fn.x.array)
        rho_fn.x.array[:] = rho_tilde

        uh = solve_fea(domain, V, bcs, rho_fn, penal, mu, lmbda, F_load)

        compliance = compute_compliance(uh, F_load)

        dc_drho          = compute_sensitivities(rho_tilde, uh, Q, penal, mu, lmbda)
        dc_drho_filtered = apply_sens_filter(dc_drho)

        volfrac_actual = rho_fn.x.array.sum() / n_cells
        g_val          = volfrac_actual - volfrac

        rho_new = optimizer.update(
            x     = rho_fn.x.array.copy(),
            f0val = compliance,
            df0dx = dc_drho_filtered,
            fval  = g_val,
            dfdx  = dg_drho,
        )

        change  = float(np.max(np.abs(rho_new - rho_old)))
        rho_fn.x.array[:] = rho_new
        rho_old = rho_fn.x.array.copy()


        volfrac_report = rho_fn.x.array.sum() / n_cells
        print_iteration_report(iteration, compliance, volfrac_report, change)

        frame_paths.append(
            save_frame(rho_fn, nelx, nely, iteration, compliance, out_dir, perm)
        )
        rho_history.append(rho_fn.x.array.copy())

        if change < tol_change and iteration > 5:
            print(f"\nConverged at iteration {iteration} (change = {change:.6f})")
            break

    # -----------------------------------------------------------------------
    # 4. EXPORT
    # -----------------------------------------------------------------------
    save_gif(frame_paths, gif_path, fps=5)
    print(f"GIF saved:  {gif_path}")

    export_xdmf(nelx, nely, rho_history, perm, output_dir=OUT_DIR)


def main_from_spec(spec):
    """
    Entry point when driven by the parser agent.
    Accepts a ProblemSpec Pydantic object, runs the full SIMP loop.

    Includes agent beta testing
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

    p = extract_simp_params(spec)

    # Geometry: assume unit aspect ratio unless you add Lx/Ly to the schema
    Lx = float(p["nelx"]) / float(p["nely"])
    Ly = 1.0

    domain    = build_mesh(p["nelx"], p["nely"], Lx, Ly)
    V, Q      = build_spaces(domain)
    mu, lmbda = get_lame_parameters(p["E"], p["nu"])
    bcs       = build_bcs_from_spec(spec, V, Lx, Ly)
    F_load    = build_load_from_spec(spec, V, Lx, Ly)

    domain.topology.create_entities(domain.topology.dim)
    n_cells = Q.dofmap.index_map.size_local
    perm = build_cell_perm(domain, Q, p["nelx"], p["nely"], Lx, Ly)

    rho_fn = fem.Function(Q)
    rho_fn.x.array[:] = p["volfrac"]

    apply_filter, apply_sens_filter = build_helmholtz_filter_CG1(
        domain, Q, p["r_min"]
    )
    optimizer = MMAOptimizer(n=n_cells, x_min=1e-3, x_max=1.0)
    dg_drho   = np.ones(n_cells) / n_cells

    frame_paths = []
    rho_history = []
    rho_old     = rho_fn.x.array.copy()


    #-------------------------------
    #[3.5] Opti Agent
    #-------------------------------
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
    x_design = np.full(n_cells, p["volfrac"])
    rho_old  = x_design.copy()

    for iteration in range(1, 301):   # increase to 300

        # 1. Physical density = filtered design variable
        rho_tilde = apply_filter(x_design)
        rho_fn.x.array[:] = rho_tilde

        # 2. FEA
        uh = solve_fea(
            domain, V, bcs, rho_fn,
            live_params["penal"], mu, lmbda, F_load
        )

        # 3. Objective
        compliance = compute_compliance(uh, F_load)

        # 4. Sensitivities w.r.t. physical density
        dc_drho = compute_sensitivities(
            rho_tilde, uh, Q, live_params["penal"], mu, lmbda
        )

        # 5. Chain rule through filter back to design space
        dc_drho_filtered = apply_sens_filter(dc_drho)

        # 6. Volume constraint on design variable
        volfrac_actual = x_design.sum() / n_cells
        g_val = volfrac_actual - live_params["volfrac"]

        # 7. MMA update on design variable
        x_new = optimizer.update(
            x=x_design.copy(),
            f0val=compliance,
            df0dx=dc_drho_filtered,
            fval=g_val,
            dfdx=dg_drho,
        )

        # 8. Change on design variable
        change   = float(np.max(np.abs(x_new - x_design)))
        x_design = x_new.copy()
        rho_fn.x.array[:] = apply_filter(x_design)  # update for visualization

        # 9. Metrics
        metrics["compliance_history"].append(float(compliance))
        metrics["volfrac_history"].append(float(volfrac_actual))
        metrics["change_history"].append(float(change))
        metrics["l2_change_history"].append(
            float(np.linalg.norm(x_new - x_design) / np.sqrt(n_cells))
        )
        metrics["iteration"] = iteration

        # 10. Steering agent
        if iteration % STEER_EVERY == 0:
            print(f"\n[SteeringAgent] Calling at iteration {iteration}...")
            live_params = steer_code(metrics, live_params)
            print(f"[SteeringAgent] Updated: {live_params}")

        # 11. Report + save
        volfrac_report = rho_fn.x.array.sum() / n_cells
        print_iteration_report(iteration, compliance, volfrac_report, change)
        frame_paths.append(
            save_frame(rho_fn, p["nelx"], p["nely"],
                       iteration, compliance, out_dir, perm)
        )
        rho_history.append(rho_fn.x.array.copy())

        # 12. Convergence
        if change < 0.01 and iteration > 10:
            print(f"\nConverged at iteration {iteration}")
            metrics["converged"] = True
            break
    save_gif(frame_paths, gif_path, fps=5)
    export_xdmf(p["nelx"], p["nely"], rho_history, perm, output_dir=OUT_DIR)
    print(f"GIF saved: {gif_path}")
    
    
    
    metrics["converged"] = (change < 0.01)

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
        # Save summary
        with open(os.path.join(OUT_DIR, "critic_summary.txt"), "w") as f:
            f.write(summary)
    else:
        print("\nPhysics validation failed — Critic Agent not called.")



if __name__ == "__main__":
    main()