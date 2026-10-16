"""The 2D L-bracket's initial level set, built with ParaLeSTO alone (no FEM packages).

main.run and the element-form preview both start from build_level_set, so the preview
shows exactly the design the solver starts from (checked against a run's iteration 0).
"""
from .config import LBracket2DConfig


def build_level_set(config: LBracket2DConfig):
    """The ParaLeSTO level-set module (PyLevelSetModule) seeded with the starting design."""
    from pyparalesto.pylsm2d import PyInput, PyLevelSetModule

    lx, ly = config.lx_m, config.ly_m
    nelx, nely = config.mesh_shape
    cut_elems, patch_elems = config.cut_cells, config.patch_cells
    arm_elems = nelx - cut_elems  # width of each arm in elements
    map_flag = 1 # 0 for least squares, 1 for discrete adjoint

    # Initialize the input object
    pyinit = PyInput(nelx, nely, map_flag, config.perturbation, config.move_limit)  # (2D: the move limit is set here; get_limits and update ignore the value passed to them)

    pyinit.add_nondesign_void_rectangle(nelx, nely, cut_elems, cut_elems)  # (2D: the binding shrinks the half-widths by 0.1, so only nodes strictly inside the cut are fixed as void)
    pyinit.create_level_set_boundary(arm_elems, nely, 1e-10, cut_elems)  # (2D: zero level set along the cut's vertical inner edge, so that edge is boundary)
    pyinit.create_level_set_boundary(nelx, arm_elems, cut_elems, 1e-10)  # (2D: the same along the cut's horizontal inner edge)

    # Define non-design domains for the boundary where traction is applied
    pyinit.add_nondesign_domain_rectangle(nelx - 0.5*patch_elems, arm_elems - 0.5*patch_elems,
                                          0.5*patch_elems, 0.5*patch_elems)
    # Initial holes: design voids the optimizer can move, merge or close
    for x_hole, y_hole in config.hole_centers_m:
        pyinit.add_initial_void_circle(nelx*x_hole/lx, nely*y_hole/ly, nelx*config.hole_radius_m/lx)  # (2D: a circle with the same center and radius in elements)

    return PyLevelSetModule(pyinit)


def initial_densities(config: LBracket2DConfig):
    """Element area fractions of the starting design from ParaLeSTO, as a (nely, nelx) array."""
    import numpy as np

    nelx, nely = config.mesh_shape
    pylsm = build_level_set(config)
    densities = np.asarray(pylsm.calculate_element_densities(False), dtype=float).reshape(nely, nelx)
    return densities, int(pylsm.get_number_boundary_points())
