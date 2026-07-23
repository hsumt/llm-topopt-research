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
from _03_OPTMZER._mma import kktcheck, mmasub  # Deetman's canonical implementation


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
        self.last_subproblem_solution = None

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

        (xmma, ymma, zmma, lam, xsi, eta, mu, zet, slack,
         self.low, self.upp) = mmasub(
            self.m, self.n, self.iter,
            xval, self.xmin, self.xmax,
            self.xold1, self.xold2,
            f0val, df0,
            fv, dfdx_,
            self.low, self.upp,
            self.a0, self.a, self.c, self.d,
            move=self.move,
        )
        self.last_subproblem_solution = {
            "x": xmma.copy(),
            "y": ymma.copy(),
            "z": zmma.copy(),
            "lam": lam.copy(),
            "xsi": xsi.copy(),
            "eta": eta.copy(),
            "mu": mu.copy(),
            "zet": zet.copy(),
            "s": slack.copy(),
        }

        # Shift history
        self.xold2 = self.xold1.copy()
        self.xold1 = xval.copy()

        return xmma.flatten()

    def kkt_diagnostics(self, x, df0dx, fval, dfdx):
        """Evaluate a diagnostic original-problem KKT residual.

        The multipliers come from the last MMA subproblem while gradients and
        constraint values are re-evaluated at the returned design. Therefore
        this is diagnostic evidence, not a calibrated hard pass/fail gate.
        """
        if self.last_subproblem_solution is None:
            return {"available": False, "reason": "no MMA subproblem solution"}
        q = self.last_subproblem_solution
        residual, norm, maximum = kktcheck(
            self.m, self.n,
            np.asarray(x, dtype=float).reshape(self.n, 1),
            q["y"], q["z"], q["lam"], q["xsi"], q["eta"],
            q["mu"], q["zet"], q["s"],
            self.xmin, self.xmax,
            np.asarray(df0dx, dtype=float).reshape(self.n, 1),
            np.asarray([[fval]], dtype=float),
            np.asarray(dfdx, dtype=float).reshape(self.m, self.n),
            self.a0, self.a, self.c, self.d,
        )
        return {
            "available": True,
            "residual_norm": float(norm),
            "residual_max": float(maximum),
            "residual_size": int(np.asarray(residual).size),
            "interpretation": (
                "diagnostic only; last MMA-subproblem multipliers evaluated "
                "with final re-evaluated objective/constraint gradients"
            ),
        }
