'''
    
    lbracket/simplified_3D_holes
    parent: lbracket/simplified_3D
    change: five initial holes from Kambampati, Chung & Kim 2021 (CMAME), Fig. 5: cylinders of radius 0.01 m (5 elements) through the thickness,
            three up the vertical arm and two along the horizontal arm. Volume limit 75% of the L and p = 6 kept from the parent; stress limit recalibrated
    physics: linear elasticity (FEniCSx)
    modified by: N. Jurado

    ########################### ORIGINAL ###########################

    Large-scale thermo-mechanical battery pack level-set topology optimization  
                                AIAA Journal 2024
    
    physics: weakly coupled heat conduction and linear elasticity (FEniCSx)
    optimization: level-set topology optimization (PyParaLeSTO)
    parallelization: FEA using distributed memory parallelization (MPI) and TO on rank 0
    units: SI units (m, N, kg, K, Pa, W...)
    
    contributors: A. Guibert, M. Pozzi, M. Bookwala
    
'''

# Ported 2026-10-02: explicit SI configuration, isolated output, bounded CLI, and run receipt.
from pathlib import Path
import argparse
import json
import time

from .config import LBracket3DConfig


def run(config: LBracket3DConfig, output: Path, max_iterations: int) -> dict:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 500:
        raise ValueError("max_iterations must be an integer between 1 and 500")
    output = Path(output)
    started = time.monotonic()
    # MPI must initialize before dolfinx for mesh partitioning on multiple ranks.
    from mpi4py import MPI
    from pyparalesto.pylsm import PyInput, PyLevelSetModule
    from pyparalesto.pyopt import PyOptimizerModule
    from dolfinx import fem, mesh, cpp, io
    from petsc4py import PETSc
    import ufl
    import numpy as np
    from time import process_time
    from .pyfea import linear_solver, sensitivity, sensitivity_compliance
    from .utils import initialize_mapping, distribute_densities, gather_sensitivities
    # MPI setup
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
    #   z-axis: through the plate thickness

    # Dimensions 
    lx, ly, lz = config.lx_m, config.ly_m, config.thickness_m  # [m] (.10m are the lengths of the sides of the square the L is cut out from, 0.012m is the plate thickness)
    cut_length = config.cut_length_m  # [m] side length of the square removed from the top right corner of the original square domain
    arm = lx - cut_length  # [m] width of each arm
    patch_length = config.load_patch_m  # [m] side of the loaded square at the tip of the horizontal arm
    hole_radius = config.hole_radius_m  # [m] radius of the initial holes (Drawn from Fig. 5 hole diameter taking up about half of the vertical arm width.)
    hole_centers = config.hole_centers_m  # [m] hole axes (holes: Fig. 5, centered in each arm)
    nelx = config.nelx  # [number of elements]
    nely, nelz = int(round(nelx*ly/lx)), int(round(nelx*lz/lx))  # Keep the same scaled element count for the y and z directions.
    cut_elems, patch_elems = int(round(nelx*cut_length/ly)), int(round(nelx*patch_length/lx))  # The number of elements in the height of the square being cut & the number of elements in the patch being traction loaded
    arm_elems = nelx - cut_elems  # width of each arm in elements
    assert min(nelx, nely, nelz) >= 6, "pyparalesto's sensitivity box needs at least 6 cells in every direction"
    assert abs(lx/nelx - ly/nely) < 1e-12 and abs(lx/nelx - lz/nelz) < 1e-12, "the level set assumes cubic cells"
    assert abs(nelx*cut_length/lx - cut_elems) < 1e-9 and abs(nelx*patch_length/lx - patch_elems) < 1e-9, "the cut and the loaded square must fall on whole elements"
    L_volume = float((nelx*nely - cut_elems**2)*nelz)  # volume of the L in elements; pyparalesto measures volume in elements
    vol_fraction = config.volume_fraction  # volume limit as a fraction of the L
    Sp_limit = config.stress_limit_pa  # null explicitly selects volume-only optimization

    # Print number of elements
    if rank == 0:
        print("Number of elements: %d" % (nelx * nely * nelz), flush=True)
        print("Number of dofs: %d" % (((nelx+1) * (nely+1) * (nelz+1))*3), flush=True)
        
    ########################################################################################################################
    # Material properties (titanium, Kambampati, Gray & Kim 2020, Sec. 3.1)
    ########################################################################################################################

    # Mechanical properties titanium
    E_ti, nu_ti = config.youngs_modulus_pa, config.poisson_ratio




    # Lamé parameters
    lmbda_ti = E_ti * nu_ti / (1. + nu_ti) / (1. - 2. * nu_ti)
    mu_ti = E_ti / (2. + 2. * nu_ti)

    # For numerical stability
    rho_min = config.rho_min

    ########################################################################################################################
    # Initialize FEA
    ########################################################################################################################

    # Mesh
    if rank == 0: print("Start creating the FEA mesh...", end="", flush=True)
    domain = mesh.create_box(comm = comm,
                                   points = ([0., 0., 0.], [lx, ly, lz]),
                                   n = [nelx, nely, nelz],
                                   cell_type = mesh.CellType.hexahedron)
    if rank == 0: print("Done", flush=True)

    # Strain and stress
    tdim = domain.topology.dim
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    def sigma(mu, lmbda, eps):
        return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(tdim)
    def sigma_vm(sig):
        sig_dev = ufl.dev(sig)
        return ufl.sqrt(3.0 / 2.0 * ufl.inner(sig_dev, sig_dev))


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
    bc_mdbc     = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0, 0.0)), dofs_mdbc, V)
    bcs = [bc_mdbc]

    # External load
    facets_mnbc = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, MNBC)
    mt_traction = mesh.meshtags(domain, domain.topology.dim - 1, facets_mnbc, 1)
    traction = np.asarray(config.force_n) / (lz * h_patch)
    T = fem.Constant(domain, np.asarray(traction, dtype=PETSc.ScalarType))




    ########################################################################################################################
    # Initialize variational problem
    ########################################################################################################################

    # Measures
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt_traction, metadata=None)
    dx = ufl.Measure("dx", domain=domain, metadata=None)
    applied_load = [comm.allreduce(fem.assemble_scalar(fem.form(T[i] * ds(1))), op=MPI.SUM) for i in range(3)]
    if rank == 0:
        print(f"Applied force: {applied_load} N (target {list(config.force_n)} N)", flush=True)

    # P-norm
    p_val = config.p_norm
    p = fem.Constant(domain, PETSc.ScalarType((p_val)))
    vol = fem.Constant(domain, PETSc.ScalarType(((lx*ly - cut_length**2)*lz)))


    # Mechanical problem
    L = ufl.inner(T, u_test) * ds(1)

    a = d * ufl.inner(sigma(mu, lmbda, epsilon(u_trial)), epsilon(u_test)) * dx

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
        pyinit = PyInput(nelx, nely, nelz, map_flag, perturbation)
        

        pyinit.add_nondesign_void_cuboid(nelx, nely, 0.5*nelz, cut_elems, cut_elems, nelz)  # (lbracket: the square cut is a permanent void through the whole thickness, centered on the grid's top-right corner)

        # Define non-design domains for the boundary where traction is applied
        pyinit.add_nondesign_domain_cuboid(nelx - 0.5*patch_elems, arm_elems - 0.5*patch_elems, 0.5*nelz,
                                           0.5*patch_elems, 0.5*patch_elems, nelz)
        # Initial holes: design voids the optimizer can move, merge or close
        for x_hole, y_hole in hole_centers:
            pyinit.add_initial_void_cylinder(nelx*x_hole/lx, nely*y_hole/ly, 0.5*nelz, nelz, nelx*hole_radius/lx, 0.0, 2)  # (holes: center and radius in elements, half-height nelz spans the thickness, inner radius 0, axis along z)


        # Initialize level set object
        pylsm = PyLevelSetModule(pyinit)
        # Read binary from previous iteration
        # lsto_input_file = "Reach_vol_3650x82x83_MMA/phi_binary/level_set_0085.bin"
        # pylsm.read_binary(lsto_input_file)

        print("Done", flush=True)

    ########################################################################################################################
    # Initialize optimizer
    ########################################################################################################################

    # The optimization problem is set to be 
    #       min     Cs
    #       s.t.    V  <= Vconstraint 
    #               Sp <= Sconstraint

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

    # All ranks agree on one new output directory; existing runs are never overwritten.
    output_path = str(output.resolve())
    creation_error = None
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
            for folder in ("stl", "states", "stress"):
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
    last_metrics = None
    loaded_area = comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(1))), op=MPI.SUM)
    if not np.isclose(loaded_area, lz * h_patch, rtol=1e-8, atol=1e-14):
        raise RuntimeError("Loaded boundary area does not match the declared patch")

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
    map_to_fenics, map_from_fenics = initialize_mapping(comm, D, [lx, ly, lz], [nelx, nely, nelz])
    if rank == 0: print("Done", flush=True)

    # Print headers
    if rank == 0:
        convergence.write("Iteration - Volume - Cs - Sp\n")
        timings.write("Iteration - ElasticityCPU - SensitivitiesCPU - SuboptimizationCPU - LevelSetCPU\n")

    # Start optimization
    comm.barrier()
    while n_iter < max_iter_opti:
        # Level-set discretization
        if rank == 0:
            dens = pylsm.calculate_element_densities(False)
            volume = pylsm.get_volume()
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
        L = ufl.inner(T, u_test) * ds(1)
        a = d * ufl.inner(sigma(mu, lmbda, epsilon(u_trial)), epsilon(u_test)) * dx
        linear_solver(comm, bilinear, linear, bcs, u, "CG", "GAMG", True)
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

        if not np.isfinite(Cs_global) or not np.isfinite(Sp):
            raise RuntimeError("Non-finite compliance or stress; stopping the solver")
        if rank == 0:
            last_metrics = {
                "iteration": n_iter, "volume_fraction": float(volume / L_volume),
                "compliance_j": float(Cs_global), "p_norm_stress_pa": float(Sp),
                "volume_feasible": bool(volume / L_volume <= vol_fraction),
                "stress_feasible": None if Sp_limit is None else bool(Sp <= Sp_limit),
            }

        # Saving data
        # Volume fractions
        with io.XDMFFile(domain.comm, f"{output_path}/states/density.xdmf", "a") as xdmf:
            xdmf.write_function(d, n_iter)
        with io.XDMFFile(domain.comm, f"{output_path}/states/displacement.xdmf", "a") as file:
            file.write_function(u, n_iter)
        with io.XDMFFile(domain.comm, f"{output_path}/stress/stress.xdmf", "a") as file:
            file.write_function(vM, n_iter)

        # Gather sensitivities
        dCs_data = gather_sensitivities(comm, dCs.vector.array[:D.dofmap.index_map.size_local], map_from_fenics)
        dSp_data = gather_sensitivities(comm, dSp.vector.array[:D.dofmap.index_map.size_local], map_from_fenics)

        # Level set optimization
        if rank == 0:
            print("Start suboptimization...", end="", flush=True)
            # Map the sensitivities
            dJ_bpt = pylsm.map_sensitivities(dCs_data, False)
            dG_bpt = np.zeros((dJ_bpt.shape[0], num_cons)) 
            dV_bpt = pylsm.map_volume_sensitivities()
            dSp_bpt = pylsm.map_sensitivities(dSp_data, False)
            dG_bpt[:, 0:1] = dV_bpt
            if Sp_limit is not None:
                dG_bpt[:, 1:2] = dSp_bpt
            timing_sensitivity = process_time() - start_sensitivity
        
            # Save stl files
            pylsm.write_stl(n_iter, output_path + "/stl/", "opt_")
            
            # Save level-set function (binary file)
            
            # Get limits
            start_suboptimization = process_time() 
            limits = pylsm.get_limits(move_limit)
            curr_cons_vals[0] = volume 
            if Sp_limit is not None:
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
        import dolfinx
        summary = {
            "status": "completed", "convergence": "not assessed",
            "stopping_reason": "iteration_budget_reached",
            "completed_iterations": n_iter, "requested_iterations": max_iterations,
            "mpi_ranks": size, "mesh_shape": [nelx, nely, nelz],
            "bounding_box_cells": nelx * nely * nelz,
            "l_domain_cells": int(L_volume),
            "loaded_area_m2": float(loaded_area),
            "resultant_force_n": [float(v) for v in traction * loaded_area],
            "runtime_seconds": time.monotonic() - started,
            "dolfinx_version": dolfinx.__version__,
            "last_evaluated_design": last_metrics,
            "input": config.to_dict(),
            "artifacts": {
                "convergence": "convergence.txt", "timings": "timings.txt",
                "density": "states/density.xdmf", "displacement": "states/displacement.xdmf",
                "stress": "stress/stress.xdmf", "stl": f"stl/opt_{n_iter - 1:04d}.stl",
            },
            "notes": [
                "Iteration budget completion does not establish optimization convergence.",
                "Stress constraint uses a volume-normalized p-norm, not the maximum local stress.",
                "Saved fields and final metrics describe the last evaluated design before its level-set update.",
                "The five holes are initial design voids that may move, merge or close.",
            ],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    else:
        summary = None
    return comm.bcast(summary, root=0)



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the 3-D L-bracket level-set backend")
    parser.add_argument("--config", type=Path, required=True, help="Validated SI JSON configuration")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--max-iterations", type=int, default=1, help="Bounded iteration budget (1-500)")
    args = parser.parse_args(argv)
    if not 1 <= args.max_iterations <= 500:
        parser.error("--max-iterations must be between 1 and 500")
    if args.config.name == ".env" or args.config.suffix == ".env":
        parser.error("Environment files are not solver configurations")
    try:
        config = LBracket3DConfig.from_dict(json.loads(args.config.read_text()))
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
