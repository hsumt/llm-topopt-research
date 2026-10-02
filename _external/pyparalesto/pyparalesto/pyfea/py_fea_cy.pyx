#
# Copyright 2021 H Alicia Kim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Define the wrapper classes needed to build OpenMDAO components
"""

from fea.python.py_fea_lib cimport *
from libcpp cimport bool, int
from libcpp.vector cimport vector

# from collections import namedtuple, Iterable

# from mpi4py.libmpi cimport *
# cimport mpi4py.MPI as MPI

from petsc4py.PETSc cimport Vec, PetscVec

# Import numpy
cimport numpy as np
import numpy as np

# def conv_np_to_cpp(self, np.ndarray np_array):
#         num_elem = np_array.shape[0]
#         cdef vector[double] cpp_vec
#         cpp_vec.resize(num_elem)

#         for i in range(num_elem):
#             cpp_vec[i] = np_array[i]

#         return cpp_vec

# def conv_cpp_to_np(self, vector[double] cpp_vec, int num_shape):
#         num_elem = cpp_vec.shape[0]
#         np_array = np.zeros((num_elem, 1), dtype=np.double)
#         for i in range(num_des_vars):
#             np_array[i] = cpp_vec[i]

#         return np_array

# cdef extern from "mpi-compat.h":
#     pass

# cython wrapper class for fea module
cdef class PyInputAnalysis:

    cdef InitializeOpt *this_ptr

    def __cinit__(self, int nelx, int nely, int nelz, double lx, double ly,
                  double lz, int nlvls, double Emax=1.0, double Emin=1e-9,
                  double nu=0.3, double Kmax=1.0, double Kmin=1e-6):
        """
        Initialize PyInputAnalysis and create a C++ InitializeOpt object. The
        PyInputAnalysis class is needed to initialize the PyLinearElasticity
        class.

        Parameters
        ----------
        nelx : int
            Number of elements in the x direction for the FEA grid. Must
            be the same as the finite element mesh. (Will be changed in the
            future to handle different level set and finite element mesh sizes).
        nely : int
            Number of elements in the y direction for the FEA grid. Must
            be the same as the finite element mesh. (Will be changed in the
            future to handle different level set and finite element mesh sizes).
        nelz : int
            Number of elements in the z direction for the FEA grid. Must
            be the same as the finite element mesh. (Will be changed in the
            future to handle different level set and finite element mesh sizes).
        lx : double
            Length of the domain in the x direction. Units of distance.
        ly : double
            Length of the domain in the y direction. Units of distance.
        lz : double
            Length of the domain in the z direction. Units of distance.
        nlvls : int
            Number of multigrid levels for PETSc solver.
        Emax : double
            Elastic modulus of fully solid element.
        Emin : double
            Minimum elastic modulus for void elements.
        nu : double
            Poisson's ratio
        Kmax : double
            Conductivity coefficient of fully solid element.
        Kmin : double
            Minimum conductivity coefficient for void elements
        """
        self.this_ptr = new InitializeOpt()
        self.this_ptr.nelx  = nelx
        self.this_ptr.nely  = nely
        self.this_ptr.nelz  = nelz
        self.this_ptr.lx    = lx
        self.this_ptr.ly    = ly
        self.this_ptr.lz    = lz
        self.this_ptr.nlvls = nlvls
        self.this_ptr.Emax  = Emax
        self.this_ptr.Emin  = Emin
        self.this_ptr.nu    = nu
        self.this_ptr.Kmax  = Kmax
        self.this_ptr.Kmin  = Kmin

    def __dealloc__(self):
        del self.this_ptr

    def add_struct_loads(self, double valx, double valy, double valz, double x,
                         double y, double z, double tolx, double toly,
                         double tolz):
        cdef Force *force_ptr
        force_ptr = new Force(valx, valy, valz, x, y, z, tolx, toly, tolz)
        self.this_ptr.loads.push_back(force_ptr[0])

    def add_adj_loads(self, double valx, double valy, double valz, double x,
                      double y, double z, double tolx, double toly, double tolz):
        cdef Force *force_ptr
        force_ptr = new Force(valx, valy, valz, x, y, z, tolx, toly, tolz)
        self.this_ptr.adjLoads.push_back(force_ptr[0])

    def add_fixed_dofs(self, bool valx, bool valy, bool valz, double x, double y,
                       double z, double tolx, double toly, double tolz):
        cdef FixedDof *fixed_dof_ptr
        fixed_dof_ptr = new FixedDof(valx, valy, valz, x, y, z, tolx, toly, tolz)
        self.this_ptr.fixedDofs.push_back(fixed_dof_ptr[0])

    def add_heat_loads(self, double val, double x, double y, double z,
                       double tolx, double toly, double tolz):
        cdef HeatLoad *heat_ptr
        heat_ptr = new HeatLoad(val, x, y, z, tolx, toly, tolz)
        self.this_ptr.heatLoad.push_back(heat_ptr[0])

    def add_fixed_heat_dofs(self, double val, double x, double y, double z,
                            double tolx, double toly, double tolz):
        cdef HeatLoad *heat_ptr
        heat_ptr = new HeatLoad(val, x, y, z, tolx, toly, tolz)
        self.this_ptr.fixedHeatDof.push_back(heat_ptr[0])

cdef class PyLinearElasticity:
    """Wrapper class for linear elasticity"""

    cdef LinElasticWrapper *this_ptr # pointer to underlying C++ class

    def __cinit__(self, PyInputAnalysis py_input):
        """
        Initialize PyLevelSetModule and create a C++ LevelSetWrapper object.

        Parameters
        ----------
        py_input : PyInputAnalysis
            Class that contains the information needed to intitialize the
            LinearElasticity wrapped class.
        """
        self.this_ptr = new LinElasticWrapper(py_input.this_ptr[0])

    def __dealloc__(self):
        del self.this_ptr

    def solve_state(self, np.ndarray densities, is_print=False, rel_tol=1e-5):
        """
        Solve the state equation for linear elasticity

        Parameters
        ----------
        densities : ndarray
            Element densities
        """
        # Copy numpy array to C++ vector. TODO: change how this is done
        num_elem = densities.shape[0]
        cdef vector[double] cpp_densities
        cpp_densities.resize(num_elem)
        for e in range(num_elem): # loop through elements
            cpp_densities[e] = densities[e]

        # Call underlying C++ functions
        self.this_ptr.SolveState(cpp_densities, is_print, rel_tol)

    def get_state_field(self):
        """
        Get pointer to the FE solution of the state equation
        """
        # return self.this_ptr.GetStateField()
        pass

    def get_state_rhs(self):
        """
        Get pointer to the FE right-hand side (RHS) of the state equation
        """
        # return self.this_ptr.GetStateRhs()
        pass

    def write_displacement_txt(self, int curr_iter = 0, string file_path = "",
                               string file_name = "disp"):
        """
        Writes nodal displacements to a text file (.txt)

        Parameters
        ----------
        curr_iter : int
            Iteration number of the topology optimization cycle.
        file_path : TODO: convert to python string
            Directory to save the file to.
        file_name : TODO: convert to python string
            Name of the text file.
        """
        self.this_ptr.WriteDispToTxt(curr_iter, file_path, file_name)

cdef class PyPoisson:
    """Wrapper class for poisson problem"""

    cdef PoissonWrapper *this_ptr # pointer to underlying C++ class

    def __cinit__(self, PyInputAnalysis py_input):
        """
        Initialize PyLevelSetModule and create a C++ LevelSetWrapper object.

        Parameters
        ----------
        py_input : PyInputAnalysis
            Class that contains the information needed to intitialize the
            Poisson wrapped class.
        """
        self.this_ptr = new PoissonWrapper(py_input.this_ptr[0])

    def __dealloc__(self):
        del self.this_ptr

    def solve_state(self, np.ndarray densities, is_print=False, rel_tol=1e-6):
        """
        Solve the state equation for poisson problem

        Parameters
        ----------
        densities : ndarray
            Element densities
        """
        # Copy numpy array to C++ vector. TODO: change how this is done
        num_elem = densities.shape[0]
        cdef vector[double] cpp_densities
        cpp_densities.resize(num_elem)
        for e in range(num_elem): # loop through elements
            cpp_densities[e] = densities[e]

        # Call underlying C++ functions
        self.this_ptr.SolveState(cpp_densities, is_print, rel_tol)

    def get_state_field(self):
        """
        Get pointer to the FE solution of the state equation
        """
        # return self.this_ptr.GetStateField()
        pass

    def get_state_rhs(self):
        """
        Get pointer to the FE right-hand side (RHS) of the state equation
        """
        # return self.this_ptr.GetStateRhs()
        pass

    def write_temperature_txt(self, int curr_iter = 0, string file_path = "",
                               string file_name = "temp"):
        """
        Writes nodal displacements to a text file (.txt)

        Parameters
        ----------
        curr_iter : int
            Iteration number of the topology optimization cycle.
        file_path : TODO: convert to python string
            Directory to save the file to.
        file_name : TODO: convert to python string
            Name of the text file.
        """
        self.this_ptr.WriteTempToTxt(curr_iter, file_path, file_name)

cdef class PyAnalysisModule:

    cdef PhysicsWrapper *this_ptr

    def __cinit__(self, PyInputAnalysis py_input_analysis):

        self.this_ptr = new PhysicsWrapper(py_input_analysis.this_ptr[0])

    def __dealloc__(self):
        del self.this_ptr

    def calculate_partial_derivatives(self, np.ndarray densities, is_print=False):
        num_des_vars = densities.shape[0]
        cdef vector[double] cpp_densities
        cpp_densities.resize(num_des_vars)
        for i in range(num_des_vars):
            cpp_densities[i] = densities[i]

        cdef vector[double] cpp_elem_sens
        cpp_elem_sens = self.this_ptr.CalculatePartialDerivatives(cpp_densities,
            is_print)

        py_elem_sens = np.zeros((num_des_vars, 1), dtype=np.double)
        for i in range(num_des_vars):
            py_elem_sens[i] = cpp_elem_sens[i]

        return py_elem_sens

    # def assign_volume_fractions(self, np.ndarray densities):
    #     num_des_vars = densities.shape[0]
    #     cdef vector[double] cpp_densities
    #     cpp_densities.resize(num_des_vars)
    #     for i in range(num_des_vars):
    #         cpp_densities[i] = densities[i]

    #     cdef vector[double] cpp_vol_frac
    #     self.this_ptr.AssignVolumeFractions(cpp_densities)

# MPI_PHYS = namedtuple("MPI_Phys", ["calculate_partial_derivatives", "assign_volume_fractions"])

# def mpi_phys():
#     return MPI_PHYS(calculate_partial_derivatives(), assign_volume_fractions())