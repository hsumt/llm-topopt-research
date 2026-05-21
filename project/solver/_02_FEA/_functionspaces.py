"""
_functionspaces.py 
Implemented 5/11/26
Creates the functions to pass over the mesh
"""
from dolfinx import fem


def build_spaces(domain):
    """
    Reference
    DOLFINx tutorial reference:
        https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity.html
    """

    V = fem.functionspace(domain, ("Lagrange", 1, (2,))) #nodes continuous
    Q = fem.functionspace(domain, ("DG", 0)) #centers of cells

    return V, Q


