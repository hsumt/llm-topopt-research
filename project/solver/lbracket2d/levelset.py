"""The 2D L-bracket's initial level set, built with ParaLeSTO alone (no FEM packages).

Mirrors the level-set initialization in main.py call for call, so the element-form preview
shows exactly the design the solver starts from (checked against a run's iteration 0).
"""
from .config import LBracket2DConfig


def initial_densities(config: LBracket2DConfig):
    """Element area fractions of the starting design from ParaLeSTO, as a (nely, nelx) array."""
    import numpy as np
    from pyparalesto.pylsm2d import PyInput, PyLevelSetModule

    lx, ly = config.lx_m, config.ly_m
    nelx, nely = config.mesh_shape
    cut_elems = int(round(nelx * config.cut_length_m / ly))
    patch_elems = int(round(nelx * config.load_patch_m / lx))
    arm_elems = nelx - cut_elems
    pyinit = PyInput(nelx, nely, 1, config.perturbation, config.move_limit)
    pyinit.add_nondesign_void_rectangle(nelx, nely, cut_elems, cut_elems)
    pyinit.create_level_set_boundary(arm_elems, nely, 1e-10, cut_elems)
    pyinit.create_level_set_boundary(nelx, arm_elems, cut_elems, 1e-10)
    pyinit.add_nondesign_domain_rectangle(nelx - 0.5 * patch_elems, arm_elems - 0.5 * patch_elems,
                                          0.5 * patch_elems, 0.5 * patch_elems)
    for x_hole, y_hole in config.hole_centers_m:
        pyinit.add_initial_void_circle(nelx * x_hole / lx, nely * y_hole / ly, nelx * config.hole_radius_m / lx)
    pylsm = PyLevelSetModule(pyinit)
    densities = np.asarray(pylsm.calculate_element_densities(False), dtype=float).reshape(nely, nelx)
    return densities, int(pylsm.get_number_boundary_points())
