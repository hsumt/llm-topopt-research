"""
_MMAupdate.py
Implemented 5/18/2026 
"""

"""
_MMAupdate.py
Thin wrapper around the canonical Svanberg MMA implementation.

Reference:
  Svanberg, K. (1987). IJNME 24:359-373.
  Python implementation: Deetman (2016), github.com/arjendeetman/GCMMA-MMA-Python
"""

import numpy as np
from _03_OPTMZER._mma import mmasub  # Deetman's canonical implementation


class MMAOptimizer:

    def __init__(self, n: int, m: int = 1,
                 x_min: float = 1e-3, x_max: float = 1.0, move: float = 0.05):
        self.n    = n
        self.m    = m
        self.iter = 0
        self.move = move

        self.xmin = x_min * np.ones((n, 1))
        self.xmax = x_max * np.ones((n, 1))

        # MMA internal state — persists across iterations for asymptote updates
        self.xold1 = x_min * np.ones((n, 1))
        self.xold2 = x_min * np.ones((n, 1))
        self.low   = self.xmin.copy()
        self.upp   = self.xmax.copy()

        # Standard single hard-constraint MMA settings
        # Svanberg (1987): a0=1, a=0, c large, d=0
        self.a0 = 1.0
        self.a  = np.zeros((m, 1))
        self.c  = 1000.0 * np.ones((m, 1))
        self.d  = np.zeros((m, 1))

    def update(self,
               x:     np.ndarray,   # current densities, shape (n,)
               f0val: float,        # compliance
               df0dx: np.ndarray,   # dC/dρ, shape (n,)
               fval:  float,        # volume constraint value: Σρ/n - volfrac
               dfdx:  np.ndarray,   # dg/dρ, shape (n,)
               ) -> np.ndarray:

        self.iter += 1

        # mmasub expects column vectors
        xval  = x.reshape(-1, 1)
        df0   = df0dx.reshape(-1, 1)
        fv    = np.array([[fval]])
        dfdx_ = dfdx.reshape(self.m, self.n)

        xmma, _, _, _, _, _, _, _, _, self.low, self.upp = mmasub(
            self.m, self.n, self.iter,
            xval, self.xmin, self.xmax,
            self.xold1, self.xold2,
            f0val, df0,
            fv, dfdx_,
            self.low, self.upp,
            self.a0, self.a, self.c, self.d,
            move = self.move
        )

        # Shift history
        self.xold2 = self.xold1.copy()
        self.xold1 = xval.copy()

        return xmma.flatten()