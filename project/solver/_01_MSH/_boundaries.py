"""
_boundaries.py
Created on 5/1/26

Creates the loads and the BC clamp
"""

import numpy as np
from petsc4py import PETSc
from dolfinx import fem

def build_bcs(V, domain):
    def left_edge(x):
        return np.isclose(x[0], 0.0) #checks if x-coordinate is basically zero

    dofs = fem.locate_dofs_geometrical(V, left_edge)
    bc   = fem.dirichletbc(
        np.array([0.0, 0.0], dtype=PETSc.ScalarType), dofs, V
    ) # Dirchlet BCs clamp the left edge so it can't move x or y
    return [bc]


def build_load(V, domain, Lx: float, Ly: float):
    def bottom_right(x):
        return np.logical_and(
            np.isclose(x[0], Lx),
            np.isclose(x[1], 0.0)
        )

    dofs = fem.locate_dofs_geometrical(V, bottom_right) #locates base index, not the index in the list

    F = fem.Function(V)
    F.x.array[:]                   = 0.0 # clears
    F.x.array[2 * dofs[0] + 1]    = -1.0   # y-DOF at tip node # [Node0_x, Node0_y, Node1_x, Node1_y, Node2_x, Node2_y ...]

    return F