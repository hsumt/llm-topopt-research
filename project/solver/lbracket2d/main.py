"""2-D plane-stress L-bracket: level-set minimum compliance with a volume constraint and an
optional p-norm stress constraint (DOLFINx + PyParaLeSTO; FEA over MPI, level set on rank 0).

Entry points: run(config, output, max_iterations), and
    python -m project.solver.lbracket2d.main --config <json> --output <dir> --max-iterations <n>

Ported from mfrto/p_norm_stress_constraint/lbracket/simplified_2D_holes (see PROVENANCE.md).

Original example contributors: A. Guibert, M. Pozzi, M. Bookwala.
"""
from pathlib import Path
import argparse
import json
import time

from .config import LBracket2DConfig


def run(config: LBracket2DConfig, output: Path, max_iterations: int) -> dict:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 500:
        raise ValueError("max_iterations must be an integer between 1 and 500")
    output = Path(output)
    started = time.monotonic()
    # MPI must initialize before dolfinx for mesh partitioning on multiple ranks.
    from mpi4py import MPI
    from pyparalesto.pylsm2d import PyInput, PyLevelSetModule
    from pyparalesto.pyopt import PyOptimizerModule
    from dolfinx import fem, mesh, cpp, io
    from petsc4py import PETSc
    import ufl
    import numpy as np
    from time import process_time
    from .pyfea import linear_solver, sensitivity, sensitivity_compliance
    from .utils import initialize_mapping, distribute_densities, gather_sensitivities
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # FEniCS log level
    cpp.log.set_log_level(cpp.log.LogLevel.ERROR)

    ########################################################################################################################
    # Settings
    ########################################################################################################################

    if rank == 0: print("Start main file... \n \n", flush=True)

    # convention for the coordinate system
    #   x-axis: horizontal, from the left edge of the horizontal arm (x=0) to the right edge (x=lx)
    #   y-axis: vertical, from the bottom of the hotizontal arm (y=0) to the clamped top of the vertical arm (y=ly)
    #   z-axis: thickneess, out of plane (z = lz). A factor to the stiffiness & load via plane stress.

    # Dimensions 
    lx, ly, lz = config.lx_m, config.ly_m, config.thickness_m  # [m] outer square side lengths and plane-stress plate thickness
    cut_length = config.cut_length_m  # [m] side length of the square removed from the top right corner of the original square domain
    arm = lx - cut_length  # [m] width of each arm
    patch_length = config.load_patch_m  # [m] side of the loaded square at the tip of the horizontal arm
    hole_radius = config.hole_radius_m  # [m] radius of the initial holes
    hole_centers = [tuple(c) for c in config.hole_centers_m]  # [m] hole axes
    nelx = int(config.nelx)  # [number of elements]
    nely = int(round(nelx*ly/lx))  # Keep the same scaled element count for the y direction (2D: no elements through the thickness)
    cut_elems, patch_elems = int(round(nelx*cut_length/ly)), int(round(nelx*patch_length/lx))  # The number of elements in the height of the square being cut & the number of elements in the patch being traction loaded
    arm_elems = nelx - cut_elems  # width of each arm in elements
    assert abs(lx/nelx - ly/nely) < 1e-12, "the level set assumes square cells"  # 2D level set assumes square cells
    assert abs(nelx*cut_length/lx - cut_elems) < 1e-9 and abs(nelx*patch_length/lx - patch_elems) < 1e-9, "the cut and the loaded square must fall on whole elements"
    L_volume = float(nelx*nely - cut_elems**2) # area of the L in elements; pyparalesto's 2D level set measures volume as area in elements
    vol_fraction = config.volume_fraction  # volume limit as a fraction of the L
    Sp_limit = config.stress_limit_pa  # [Pa] calibrated p-norm limit; None for a volume-only problem

    # Print number of elements
    if rank == 0:
        print("Number of elements: %d" % (nelx * nely), flush=True)
        print("Number of dofs: %d" % (((nelx+1) * (nely+1))*2), flush=True)

    ########################################################################################################################
    # Material properties (isotropic, from the configuration)
    ########################################################################################################################

    # Mechanical properties
    E_ti, nu_ti = config.youngs_modulus_pa, config.poisson_ratio

    # Lamé parameters
    lmbda_ti = E_ti * nu_ti / (1. + nu_ti) / (1. - 2. * nu_ti)
    mu_ti = E_ti / (2. + 2. * nu_ti)
    lmbda_ti = 2.0 * mu_ti * lmbda_ti / (lmbda_ti + 2.0 * mu_ti) # (2D: plane stress, zero stress through the thickness; the 3D lambda would give plane strain)

    # For numerical stability
    rho_min = config.rho_min

    ########################################################################################################################
    # Initialize FEA
    ########################################################################################################################

    # Mesh
    if rank == 0: print("Start creating the FEA mesh...", end="", flush=True)
    domain = mesh.create_rectangle(comm = comm,
                                   points = ([0., 0.], [lx, ly]),
                                   n = [nelx, nely],
                                   cell_type = mesh.CellType.quadrilateral)
    if rank == 0: print("Done", flush=True)

    # Strain and stress
    tdim = domain.topology.dim
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    def sigma(mu, lmbda, eps):
        return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(tdim)
    def sigma_vm(sig):
        # (2D: plane-stress von Mises from the in-plane components; ufl.dev of the 2x2 tensor would ignore the zero through-thickness stress)
        return ufl.sqrt(sig[0, 0]**2 - sig[0, 0]*sig[1, 1] + sig[1, 1]**2 + 3.0*sig[0, 1]**2)


    # Function spaces for the elasticity problem (linear shape functions)
    V = fem.VectorFunctionSpace(domain, ("CG", 1))
    u_trial, u_test = ufl.TrialFunction(V), ufl.TestFunction(V)
    u = fem.Function(V)
    adj = fem.Function(V)

    # Density space
    D = fem.FunctionSpace(domain, ("DG", 0))
    d = fem.Function(D)
    dCs = fem.Function(D)
    dSp = fem.Function(D)
    vM  = fem.Function(D)

    ########################################################################################################################
    # Definition of subdomains
    ########################################################################################################################

    # helpers for the definition of the subdomains
    tol = 1e-6
    x_corner, y_corner = arm, arm  # [m] the re-entrant corner of the L
    h_patch = patch_elems * ly / nely  # [m] height of the loaded band at the tip


    # Domain for Neumann BC for elasticity PDE
    def MNBC(x):
        end = (x[0] > (lx - tol))
        band = (x[1] > (y_corner - h_patch - tol)) & (x[1] < (y_corner + tol))
        return end & band

    # Domain for Dirichlet BC for elasticity PDE
    def MDBC(x):
        return ((x[1] > (ly - tol)) & (x[0] < (x_corner + tol)))




    ########################################################################################################################
    # Association of subdomains with materials properties
    ########################################################################################################################

    # Functions for maks
    lmbda   = fem.Function(D)
    mu      = fem.Function(D)


    # Lame parameters
    mu.x.array[:] = mu_ti

    lmbda.x.array[:] = lmbda_ti



    ########################################################################################################################
    # Initialize mechanical problem
    ########################################################################################################################

    # Mechanical Dirichlet BC
    facets_mdbc = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, MDBC)
    dofs_mdbc   = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets_mdbc)
    bc_mdbc     = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), dofs_mdbc, V)
    bcs = [bc_mdbc]

    # External load
    facets_mnbc = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, MNBC)
    mt_traction = mesh.meshtags(domain, domain.topology.dim - 1, facets_mnbc, 1)
    force = np.asarray(config.force_n, dtype=float)  # [N] total in-plane force (Fx, Fy) on the tip band
    traction = force / (lz * h_patch)  # [N/m2] traction over the loaded band: thickness times band height
    T = fem.Constant(domain, PETSc.ScalarType(tuple(traction)))




    ########################################################################################################################
    # Initialize variational problem
    ########################################################################################################################

    # Measures
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt_traction, metadata=None)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})  # (2D: pinned on the measure so every form and its derivatives use degree 8, the degree dolfinx estimates for the p-norm)
    applied_load = comm.allreduce(fem.assemble_scalar(fem.form(lz * ufl.sqrt(ufl.inner(T, T)) * ds(1))), op=MPI.SUM)  # (lbracket: integrates the traction over the tagged facets, so every run logs the force actually applied)
    if rank == 0: print("Applied load: %.6f N (target %.1f N)" % (applied_load, float(np.linalg.norm(force))), flush=True)  # (lbracket: logs the force actually applied, to catch facet-tagging errors)

    # P-norm
    p_val = float(config.p_norm)
    p = fem.Constant(domain, PETSc.ScalarType((p_val)))
    vol = fem.Constant(domain, PETSc.ScalarType((lx*ly - cut_length**2)))  # (2D: the p-norm is a mean over the L's area)


    # Mechanical problem
    L = lz * ufl.inner(T, u_test) * ds(1)  # (2D: the thickness turns the area integrals into the plate's forces and compliance)

    a = lz * d * ufl.inner(sigma(mu, lmbda, epsilon(u_trial)), epsilon(u_test)) * dx

    # Functions of interest
    Cs = ufl.action(L, u)
    Sp_tilde = 1/vol * (d * sigma_vm(sigma(mu, lmbda, epsilon(u)))) ** p * dx

    # Compile forms

    bilinear = fem.form(a)
    linear = fem.form(L)
    Cs_form = fem.form(Cs)
    Sp_tilde_form = fem.form(Sp_tilde)


    ########################################################################################################################
    # Initialize LSM
    ########################################################################################################################

    # Level set initialization
    if rank == 0:
        print("Start initialization of the level-set...", end="", flush=True)
        # Parameters
        map_flag = 1 # 0 for least squares, 1 for discrete adjoint
        perturbation = config.perturbation
        move_limit = config.move_limit

        # Initialize the input object
        pyinit = PyInput(nelx, nely, map_flag, perturbation, move_limit)  # (2D: the move limit is set here; get_limits and update ignore the value passed to them)


        pyinit.add_nondesign_void_rectangle(nelx, nely, cut_elems, cut_elems)  # (2D: the binding shrinks the half-widths by 0.1, so only nodes strictly inside the cut are fixed as void)
        pyinit.create_level_set_boundary(arm_elems, nely, 1e-10, cut_elems)  # (2D: zero level set along the cut's vertical inner edge, so that edge is boundary)
        pyinit.create_level_set_boundary(nelx, arm_elems, cut_elems, 1e-10)  # (2D: the same along the cut's horizontal inner edge)

        # Define non-design domains for the boundary where traction is applied
        pyinit.add_nondesign_domain_rectangle(nelx - 0.5*patch_elems, arm_elems - 0.5*patch_elems,
                                              0.5*patch_elems, 0.5*patch_elems)
        # Initial holes: design voids the optimizer can move, merge or close
        for x_hole, y_hole in hole_centers:
            pyinit.add_initial_void_circle(nelx*x_hole/lx, nely*y_hole/ly, nelx*hole_radius/lx)  # (2D: a circle with the same center and radius in elements)


        # Initialize level set object
        pylsm = PyLevelSetModule(pyinit)

        print("Done", flush=True)

    ########################################################################################################################
    # Initialize optimizer
    ########################################################################################################################

    # The optimization problem is set to be 
    #       min     Cs
    #       s.t.    V  <= Vconstraint
    #               Sp <= Sconstraint (only when stress_limit_pa is set)

    if rank == 0:
        # Constraints
        max_cons_vals = np.array([vol_fraction * L_volume] + ([] if Sp_limit is None else [Sp_limit]))

        # Optimizer
        num_cons = len(max_cons_vals)
        curr_cons_vals = np.zeros(num_cons, dtype=np.double)
        opt_algo = 1  # Simplex method 2 || Newton Raphson 0 || MMA NLOPT 1
        pyopt = PyOptimizerModule(num_cons, max_cons_vals, opt_algo)

    # Iterations
    max_iter_opti = max_iterations
    n_iter = 0

    ########################################################################################################################
    # Initialize IO
    ########################################################################################################################

    # IO initialization
    # All ranks agree on one new output directory; existing runs are never overwritten.
    output_path = str(output.resolve())
    creation_error = None
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
            for folder in ("vtk", "states", "stress"):  # (2D: the 2D level set writes VTK)
                (output / folder).mkdir()
            (output / "input.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
            convergence = open(output / "convergence.txt", "w", buffering=1)
            timings = open(output / "timings.txt", "w", buffering=1)
        except Exception as error:
            creation_error = str(error)
    creation_error = comm.bcast(creation_error, root=0)
    if creation_error:
        raise RuntimeError(creation_error)
    comm.barrier()
    loaded_area = comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(1))), op=MPI.SUM)
    last_metrics = start = None
    frames = []

    if rank == 0: print("Start creating the output files...", end="", flush=True)
    with io.XDMFFile(domain.comm, f"{output_path}/states/density.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
    with io.XDMFFile(domain.comm, f"{output_path}/states/displacement.xdmf", "w") as file:
        file.write_mesh(domain)
    with io.XDMFFile(domain.comm, f"{output_path}/stress/stress.xdmf", "w") as file:
        file.write_mesh(domain)
    if rank == 0: print("Done", flush=True)

    ########################################################################################################################
    # Main loop
    ########################################################################################################################

    # Initialize mapping
    if rank == 0: print("Start creating the mapping FEniCSx-Level-set...", end="", flush=True)
    map_to_fenics, map_from_fenics = initialize_mapping(comm, D, [lx, ly], [nelx, nely])
    if rank == 0: print("Done", flush=True)

    # Print headers
    if rank == 0:
        convergence.write("Iteration - Volume - Cs - Sp\n")
        timings.write("Iteration - Elasticity - Sensitivities - Suboptimization - Level-set update\n")

    # Start optimization
    comm.barrier()
    while n_iter < max_iter_opti:
        # Level-set discretization
        if rank == 0:
            dens = pylsm.calculate_element_densities(False)
            volume = pylsm.get_volume()
            last_dens = dens  # (port: kept so the final element densities can be saved for viewing)
            last_bpts = pylsm.get_number_boundary_points()
            if n_iter % 10 == 0:  # (port: element densities every 10 iterations, for the result viewer)
                frames.append((n_iter, np.asarray(dens, dtype=float).reshape(nely, nelx), last_bpts))
            if n_iter == 0:  # (port: stop on an empty start; record the start design)
                assert last_bpts > 0 and volume > 0, (
                    f"Level-set initialization failed: {last_bpts} boundary points, "
                    f"volume {volume:.10f}. The initial design is empty."
                )
                start = {"volume_fraction": volume / L_volume, "boundary_points": last_bpts}
        else:
            dens = None

        # Create density field
        dens_local = distribute_densities(comm, dens, map_to_fenics)
        d.vector.array = rho_min + (1.0 - rho_min) * (dens_local)
        d.x.scatter_forward()


        # Solve mechanical problem
        if rank == 0: 
            print("Start solving mechanical problem...", end="", flush=True)
            start_elasticity = process_time()
        linear_solver(comm, bilinear, linear, bcs, u, "PREONLY", "LU", False)  # (2D: direct solve, exact to round-off; a direct solve takes no initial guess)
        if rank == 0: 
            timing_elasticity = process_time() - start_elasticity
            print("Done", flush=True)
        stress_expr = fem.Expression(sigma_vm(sigma(mu, lmbda, epsilon(u))), D.element.interpolation_points())
        vM.interpolate(stress_expr)

        # Residuals
        if rank == 0: start_sensitivity = process_time()
        R = ufl.action(a, u) - L


        # Mechanical compliance
        if rank == 0: print("Start computing sensitivity structural compliance...", end="", flush=True)
        Cs_local = fem.assemble_scalar(Cs_form)
        Cs_global = comm.allreduce(Cs_local, op=MPI.SUM)
        sensitivity_compliance(u, d, R, dCs)
        if rank == 0: print("Done", flush=True)


        # Stress
        if rank == 0: print("Start computing sensitivity stress constraint...", end="", flush=True)
        Sp_local = fem.assemble_scalar(Sp_tilde_form)
        Sp_global = comm.allreduce(Sp_local, op=MPI.SUM)
        Sp = Sp_global ** (1.0 / p_val)
        sensitivity(comm, bcs, u, d, Sp_tilde, R, adj, dSp)
        dSp.vector.array[:] *= (1.0 / p_val) * Sp**(1.0 - p_val)
        if rank == 0: print("Done", flush=True)

        # Saving data
        # Volume fractions
        with io.XDMFFile(domain.comm, f"{output_path}/states/density.xdmf", "a") as xdmf:
            xdmf.write_function(d, n_iter)
        with io.XDMFFile(domain.comm, f"{output_path}/states/displacement.xdmf", "a") as file:
            file.write_function(u, n_iter)
        with io.XDMFFile(domain.comm, f"{output_path}/stress/stress.xdmf", "a") as file:
            file.write_function(vM, n_iter)

        # Gather sensitivities
        dCs_data = gather_sensitivities(comm, dCs.vector.array, map_from_fenics)
        dSp_data = gather_sensitivities(comm, dSp.vector.array, map_from_fenics)

        # Level set optimization
        if rank == 0:
            print("Start suboptimization...", end="", flush=True)
            # Map the sensitivities
            dJ_bpt = pylsm.map_sensitivities(dCs_data, False)
            dG_bpt = np.zeros((dJ_bpt.shape[0], num_cons)) 
            dV_bpt = pylsm.map_volume_sensitivities()
            dSp_bpt = pylsm.map_sensitivities(dSp_data, False)
            dG_bpt[:, 0:1] = dV_bpt
            if num_cons > 1:
                dG_bpt[:, 1:2] = dSp_bpt
            timing_sensitivity = process_time() - start_sensitivity

            # Save vtk files
            pylsm.write_vtk(n_iter, output_path + "/vtk")  # (2D: writes vtk/level-set_NNNN.vtk; the binding ignores a file name)

            # Get limits
            start_suboptimization = process_time() 
            limits = pylsm.get_limits(move_limit)
            curr_cons_vals[0] = volume 
            if num_cons > 1:
                curr_cons_vals[1] = Sp

            # Solve the optimization problem
            pyopt.set_limits(limits)
            velocities = pyopt.solve(dJ_bpt, dG_bpt, curr_cons_vals, False)
            timing_suboptimization = process_time() - start_suboptimization

            # Update level set
            start_update = process_time() 
            pylsm.update(velocities, move_limit, False)
            timing_update = process_time() - start_update

            # Print convergence iteration
            print("Done", flush=True)
            print("Iteration - Volume - Cs - Sp", flush=True)
            print("%4d %12.4e %12.4e %12.4e" \
                % (n_iter, volume / L_volume, Cs_global, Sp), flush=True)
            last_metrics = {"iteration": n_iter, "volume_fraction": volume / L_volume, "compliance": Cs_global, "stress_pnorm_pa": Sp}
            convergence.write("%4d %12.4e %12.4e %12.4e\n" \
                % (n_iter, volume / L_volume, Cs_global, Sp))
            timings.write("%4d %8.4f %8.4f %8.4f %8.4f\n" \
                % (n_iter, timing_elasticity, timing_sensitivity, timing_suboptimization, timing_update))

        # Update iteration
        n_iter += 1
        comm.barrier()
    if rank == 0:
        convergence.close()
        timings.close()
        # (port: element area fractions of the last evaluated design, row-major nely x nelx, for viewers without HDF5)
        np.save(output / "states" / "density_final.npy", np.asarray(last_dens, dtype=float).reshape(nely, nelx))
        if not frames or frames[-1][0] != n_iter - 1:
            frames.append((n_iter - 1, np.asarray(last_dens, dtype=float).reshape(nely, nelx), last_bpts))
        np.savez_compressed(output / "states" / "density_history.npz", iterations=np.array([f[0] for f in frames]),
                            densities=np.stack([f[1] for f in frames]), boundary_points=np.array([f[2] for f in frames]))
        import dolfinx
        summary = {
            "status": "completed", "convergence": "not assessed",
            "stopping_reason": "iteration_budget_reached",
            "completed_iterations": n_iter, "requested_iterations": max_iterations,
            "mpi_ranks": size, "mesh_shape": [nelx, nely],
            "l_domain_cells": int(L_volume), "start_design": start,
            "loaded_area_m2": float(loaded_area),
            "resultant_force_n": [float(v) for v in force],
            "applied_load_magnitude_n": float(applied_load),
            "runtime_seconds": time.monotonic() - started,
            "dolfinx_version": dolfinx.__version__,
            "last_evaluated_design": last_metrics,
        "level_set": {"library": "pyparalesto (vendored, unmodified)", "level_set_module": "pyparalesto.pylsm2d",
                      "optimizer": "pyparalesto.pyopt, MMA via NLopt (opt_algo 1)",
                      "level_set_updates": n_iter, "final_boundary_points": int(last_bpts)},
            "input": config.to_dict(),
            "artifacts": {
                "convergence": "convergence.txt", "timings": "timings.txt",
                "density": "states/density.xdmf", "displacement": "states/displacement.xdmf",
                "stress": "stress/stress.xdmf", "level_set_vtk": f"vtk/level-set_{n_iter - 1:04d}.vtk",
            "density_final": "states/density_final.npy", "density_history": "states/density_history.npz",
            },
            "notes": [
                "Iteration budget completion does not establish optimization convergence.",
                "Plane stress: the thickness scales stiffness, load and compliance.",
                "Stress constraint uses an area-normalized p-norm, not the maximum local stress.",
                "The five holes are initial design voids that may move, merge or close.",
            ],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    else:
        summary = None
    return comm.bcast(summary, root=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the 2-D plane-stress L-bracket level-set backend")
    parser.add_argument("--config", type=Path, required=True, help="Validated SI JSON configuration")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--max-iterations", type=int, default=1, help="Bounded iteration budget (1-500)")
    args = parser.parse_args(argv)
    if not 1 <= args.max_iterations <= 500:
        parser.error("--max-iterations must be between 1 and 500")
    if args.config.name == ".env" or args.config.suffix == ".env":
        parser.error("Environment files are not solver configurations")
    try:
        config = LBracket2DConfig.from_dict(json.loads(args.config.read_text()))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        run(config, args.output, args.max_iterations)
    except Exception as error:
        # Collective abort prevents a failed MPI rank leaving peers blocked forever.
        import sys
        print(f"L-bracket solver failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        from mpi4py import MPI
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
