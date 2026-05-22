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
                                     export_xdmf, print_iteration_report)


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
    r_min      = 0.08 #0.05 attempt 1
    max_iter   = 100
    tol_change = 0.01

    # -----------------------------------------------------------------------
    # 2. SETUP
    # -----------------------------------------------------------------------
    domain    = build_mesh(nelx, nely, Lx, Ly)
    V, Q      = build_spaces(domain)
    mu, lmbda = get_lame_parameters(1.0, 0.3)
    bcs       = build_bcs(V, domain)
    F_load    = build_load(domain, Lx, Ly)

    domain.topology.create_entities(domain.topology.dim)
    n_cells = Q.dofmap.index_map.size_local

    rho_fn = fem.Function(Q)
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
        # rho_fn.x.array[:] = rho_tilde

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
        rho_old = rho_fn.x.array.copy()
        rho_fn.x.array[:] = rho_new

        volfrac_report = rho_fn.x.array.sum() / n_cells
        print_iteration_report(iteration, compliance, volfrac_report, change)

        frame_paths.append(
            save_frame(rho_fn, nelx, nely, iteration, compliance, out_dir)
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

    export_xdmf(nelx, nely, rho_history, output_dir=OUT_DIR)


if __name__ == "__main__":
    main()