

import numpy as np
from dolfinx import fem


from _01_MSH._mesh import build_mesh
from _01_MSH._boundaries import build_bcs, build_load
from _01_MSH._domain import get_lame_parameters


from _02_FEA._functionspaces import build_spaces
from _02_FEA._solver         import solve_fea

from _03_OPTMZER._filters   import (
    build_filter_kdtree,
    apply_filter_kdtree,
    apply_sensitivity_filter_kdtree,
)
from _03_OPTMZER._objective  import compute_compliance, compute_sensitivities
from _03_OPTMZER._updateOC  import oc_update
def main():
    # -----------------------------------------------------------------------
    # Parameters — matching Sigmund (2001) cantilever benchmark
    # -----------------------------------------------------------------------
    nelx      = 80       # elements in x                    [1, p.121]
    nely      = 50       # elements in y                    [1, p.121]
    Lx        = 1.6      # domain length x  [m]
    Ly        = 1.0      # domain length y  [m]
    volfrac   = 0.4      # target volume fraction           [1, p.121]
    penal     = 3.0      # SIMP penalisation exponent       [1, Eq.(1)]
    r_min     = 0.05     # filter radius [m] (~2.5 elements for this mesh)
    max_iter  = 100      # maximum iterations
    tol_change = 0.01    # convergence: max(|rho_new - rho_old|) < 1%
    out_dir   = "_05_OUT/frames"


    domain        = build_mesh(nelx, nely, Lx, Ly)
    V, Q          = build_spaces(domain)
    mu, lmbda     = get_lame_parameters(1.0, 0.3)
    bcs           = build_bcs(V, domain)
    F_load        = build_load(V, domain, Lx, Ly)


    