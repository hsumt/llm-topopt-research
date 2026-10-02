"""Mapping helpers copied from the simplified_3D_holes L-bracket example.

Original example contributors: A. Guibert, M. Pozzi, M. Bookwala.
Local integration changes are recorded in PROVENANCE.md.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem

def initialize_mapping(comm: MPI.Comm, D: fem.FunctionSpace, l: list, nel: list, rank: int = 0):
    # Extract mesh dimensions
    dim = len(l)
    
    # Get local ranges and global size of array
    dl = np.array(l) / np.array(nel)
    Delta_div = 1 / dl
    numbering_multiplier = np.array([[1], [nel[0]], [nel[0] * nel[1]]])

    imap = D.dofmap.index_map
    imap_bs = D.dofmap.index_map_bs
    local_range = np.asarray(imap.local_range, dtype=np.int32) * imap_bs
    num_of_dofs_current = np.arange(*local_range)
    indices_gathered = comm.gather(num_of_dofs_current, root=rank)

    # Communicate local dof coordinates
    x = D.tabulate_dof_coordinates()[:imap.size_local, :dim]
    x = np.vstack(x) - 0.5 * np.array(dl)

    native_numbering = np.squeeze(np.rint(np.dot((x * Delta_div), numbering_multiplier[:dim, :])).astype(int))
    all_nn = comm.gather(native_numbering, root=rank) # gather all native numberings to one process

    if comm.rank == rank:
        stacked_all_nn = np.hstack(all_nn)
        map_from_fenics = np.argsort(stacked_all_nn)
        map_to_fenics = [stacked_all_nn[indices_gathered[i]] for i in range(comm.size)]
    else:
        map_from_fenics = None
        map_to_fenics = None
    
    return map_to_fenics, map_from_fenics

def distribute_densities(comm: MPI.Comm, dens: list, map_to_fenics: list[np.ndarray], rank: int = 0):
    if comm.rank == rank:   
        # Pass from SIMP to FEniCS
        distribution = {}
        for i in range(comm.size):
            name = 'densities_process_' + str(i)
            distribution[name] = dens[map_to_fenics[i]]
        locals().update(distribution)
    else:
        distribution = None   

    # Broadcast the distribution to the different processors
    distribution = comm.bcast(distribution, root=rank)
    name_local = 'densities_process_' + str(comm.rank)
    dens_local = np.squeeze(distribution[name_local])
    
    return dens_local

def gather_sensitivities(comm: MPI.Comm, sens_local: np.ndarray, map_from_fenics: np.ndarray, rank: int = 0):
    # Gather sensitivites
    sens_gather = comm.gather(sens_local, root=rank)

    # Order sensitivities
    if comm.rank == rank :
        sens = np.hstack(sens_gather)
        sens = sens[map_from_fenics]
    else:
        sens = None
    return sens
