"""MMA driver for several constraints.

The repository's ``MMAOptimizer`` wraps ``mmasub`` for a single scalar
constraint.  The clustered stress formulation needs nc constraints, so this
driver calls the same vendored Svanberg/Deetman ``mmasub`` with a general
constraint vector.  The underlying MMA implementation is unchanged and shared,
so both paths solve subproblems with identical algebra.

Reference: Svanberg, K. (1987), IJNME 24:359-373.
"""
from __future__ import annotations

import numpy as np

from project.topopt.optimization.mma import mmasub


class MultiConstraintMMA:
    def __init__(self, n: int, m: int, *, x_min: float, x_max: float = 1.0, move: float = 0.1):
        self.n = int(n)
        self.m = int(m)
        self.iter = 0
        self.move = float(move)
        self.xmin = x_min * np.ones((self.n, 1))
        self.xmax = x_max * np.ones((self.n, 1))
        self.xold1 = None
        self.xold2 = None
        self.low = self.xmin.copy()
        self.upp = self.xmax.copy()
        self.a0 = 1.0
        self.a = np.zeros((self.m, 1))
        self.c = 1000.0 * np.ones((self.m, 1))
        self.d = np.zeros((self.m, 1))

    def update(self, x, f0val, df0dx, fval, dfdx):
        self.iter += 1
        xval = np.asarray(x, dtype=float).reshape(-1, 1)
        if self.xold1 is None:
            self.xold1 = xval.copy()
            self.xold2 = xval.copy()
        fval = np.asarray(fval, dtype=float).reshape(self.m, 1)
        dfdx = np.asarray(dfdx, dtype=float).reshape(self.m, self.n)
        xmma = mmasub(
            self.m, self.n, self.iter,
            xval, self.xmin, self.xmax,
            self.xold1, self.xold2,
            float(f0val), np.asarray(df0dx, dtype=float).reshape(-1, 1),
            fval, dfdx,
            self.low, self.upp,
            self.a0, self.a, self.c, self.d,
            move=self.move,
        )
        new_x, self.low, self.upp = xmma[0], xmma[-2], xmma[-1]
        self.xold2 = self.xold1.copy()
        self.xold1 = xval.copy()
        return np.asarray(new_x).ravel()
