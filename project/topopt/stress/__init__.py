"""Stress-constrained topology optimization, aligned to Holmberg et al. (2013).

Holmberg E., Torstenfelt B., Klarbring A. (2013), "Stress constrained topology
optimization", Struct Multidisc Optim 48:33-47.

This package implements the paper's formulation directly:

* bilinear Q4 elements, one stress evaluation point at the element centroid
  (the superconvergent point for this element)                      -- Sec. 1
* design-variable cone filter rho = W x                             -- Sec. 3, Eq. (2)
* SIMP stiffness penalization eta_K = rho^q, q = 3                  -- Sec. 4.1
* stress penalization eta_S = rho^(1/2)                             -- Sec. 4.2, Eq. (4)
* clustered modified P-norm stress measure, normalized by the limit -- Sec. 5, Eq. (6)
* stress-level and distributed-stress clustering                    -- Sec. 6
* adjoint sensitivities                                             -- Sec. 7
* formulations P1 (min mass s.t. stress), P2 (min compliance s.t.
  stress and mass), P3 (min compliance s.t. mass)                   -- Sec. 2

Nothing here is specific to the L-beam; the geometry lives in ``domain.py``.
"""
