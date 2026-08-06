"""
_mesh.py
Created on 5/1/26

Creates the material parameters (shear modulus and lmbda) and the basic stiffness equation
"""# Creates the mesh and elements as quads
import numpy as np
from mpi4py import MPI
from dolfinx import mesh

def build_mesh(nelx: int, nely: int, Lx: float, Ly: float):
    """
    Create a structured quadrilateral mesh on [0, Lx] x [0, Ly] with nelx elements in x and nely elements in y

    """
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([Lx, Ly])],
        [nelx, nely],
        cell_type=mesh.CellType.quadrilateral,
    )
    return domain