"""Field rendering for the L-shaped domain.

Elements live on a structured grid restricted to the L, so a field is placed
back onto the bounding grid with the void quadrant left as NaN.
"""
from __future__ import annotations

import numpy as np


def field_to_grid(mesh, values: np.ndarray) -> np.ndarray:
    n = mesh.geom.n_cells_per_side
    h = mesh.geom.h
    grid = np.full((n, n), np.nan)
    ix = np.round(mesh.centroids[:, 0] / h - 0.5).astype(int)
    iy = np.round(mesh.centroids[:, 1] / h - 0.5).astype(int)
    grid[iy, ix] = values
    return grid


def extent(mesh):
    return [0.0, mesh.geom.L, 0.0, mesh.geom.L]
