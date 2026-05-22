"""
_boundaries.py
Created on 5/1/26

Creates the loads and the BC clamp
"""
from dolfinx.mesh import locate_entities_boundary, meshtags
import ufl
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


def build_load(domain, Lx: float, Ly: float):
    """
    Distributed downward traction on right edge, total force = -1.
    Much more stable than a point load for FEM.
    """
    fdim = domain.topology.dim - 1
    right_facets = locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], Lx)
    )
    marked = meshtags(domain, fdim, right_facets,
                      np.ones(len(right_facets), dtype=np.int32))
    ds_right = ufl.Measure("ds", domain=domain,
                            subdomain_data=marked, subdomain_id=1)
    return ds_right