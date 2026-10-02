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

"""! @file py_lsm_2d_cy.pyx
     @brief Define the wrapper classes needed for the level set module: PyInput 
     and PyLevelSetModule classes.
"""

# cython: c_string_type=unicode, c_string_encoding=utf8

from pyparalesto.py_lsm_2d.py_lsm_2d_lib cimport *
from libcpp cimport bool, int
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string

# Import numpy
cimport numpy as np
import numpy as np
from numpy cimport NPY_DOUBLE

# Import C methods for python
from cpython cimport PyObject, Py_INCREF

cdef class PyInput:
    """! @brief Wrapper class for the input file"""

    cdef InitializeLsm *this_ptr # pointer to underlying C++ class

    def __cinit__(self, int nelx, int nely, int map_flag=0,
                  double perturbation=0.15, double move_limit=0.5):
        """! @brief Constructor."""
        """! Initialize PyInput and create a C++ InitializeLsm object. The 
        PyInput class is needed to initalize the PyLevelSetModule class."""
        """!
        @param nelx
            int. Number of elements in the x direction for the level set grid. 
            Must be the same as the finite element mesh. (Will be changed in the
            future to handle different level set and finite element mesh sizes).
        @param nely
            int. Number of elements in the y direction for the level set grid. 
            Must be the same as the finite element mesh. (Will be changed in the
            future to handle different level set and finite element mesh sizes).
        """
        self.this_ptr = new InitializeLsm()
        self.this_ptr.nelx = nelx
        self.this_ptr.nely = nely
        self.this_ptr.map_flag = map_flag
        self.this_ptr.perturbation = perturbation
        self.this_ptr.move_limit = move_limit

    def __dealloc__(self):
        del self.this_ptr

    def add_initial_void_rectangle(self, double x, double y, double hx, double hy):
        """! @brief Add an initial void of type Hole that represents a rectangle.""" 
        """! This will create a void in the initial design space that 
        will change during the topology optimization. The void may disappear, 
        combine with another void, or change shape."""
        """!
        @param x
            double. X coordinate of the origin of the cuboid
        @param y
            double. Y coordinate of the origin of the cuboid
        @param hx
            double. Half-width of the cuboid in the x direction
        @param hy
            double. Half-width of the cuboid in the y direction
        """
        # The 0.1 is to make sure that the point on the edges are included (matteo).
        cdef shared_ptr[Hole] blob_ptr
        blob_ptr.reset(new Hole(x, y, hx + 0.1, hy + 0.1))
        self.this_ptr.initialVoids.push_back(blob_ptr)

    def add_initial_void_circle(self, double x, double y, double r):
        """! @brief Add an initial void of type Hole that represents a circle."""
        """! This will create a void in the initial design space that 
        will change during the topology optimization. The void may disappear, 
        combine with another void, or change shape."""
        """!
        @param x
            double. X coordinate of the origin of the hole
        @param y
            double. Y coordinate of the origin of the hole
        @param r
            double. Radius
        """
        cdef shared_ptr[Hole] blob_ptr
        blob_ptr.reset(new Hole(x, y, r))
        self.this_ptr.initialVoids.push_back(blob_ptr)

    def add_nondesign_void_rectangle(self, double x, double y, double hx, double hy):
        """! @brief Add a nondesign void of type Hole that represents a rectangle.""" 
        """! This will prevent the design from going into the volume defined by 
        the nondesign void. The region defined by the nondesign void will always 
        be void throughout the topology optimization.
        Note that this function will automatically add the same region as an
        initial void (which is a necessary step) unlike the C++ API where the
        user has to do it manually."""
        """! 
        @param x
            double. X coordinate of the origin of the cuboid
        @param y
            double. Y coordinate of the origin of the cuboid
        @param hx
            double. Half-width of the cuboid in the x direction
        @param hy
            double. Half-width of the cuboid in the y direction
        """
        # The 0.1 is to make sure that the point on the edges are excluded.
        # Otherwise, there will be no boundary points on the edges.
        # In addition, remember to call the "create_level_set_boundary" method
        # to make sure the signed distance values on the edges are set to zero (matteo).
        cdef shared_ptr[Hole] blob_ptr
        blob_ptr.reset(new Hole(x, y, hx - 0.1, hy - 0.1))
        self.this_ptr.initialVoids.push_back(blob_ptr)
        self.this_ptr.domainVoids.push_back(blob_ptr)

    def add_nondesign_void_circle(self, double x, double y, double r):
        """! @brief Add a nondesign void of type Hole that represents a circle.""" 
        """! This will prevent the design from going into the volume defined by 
        the nondesign void. The region defined by the nondesign void will always 
        be void throughout the topology optimization.
        Note that this function will automatically add the same region as an
        initial void (which is a necessary step) unlike the C++ API where the
        user has to do it manually."""
        """! 
        @param x
            double. X coordinate of the origin of the cuboid
        @param y
            double. Y coordinate of the origin of the cuboid
        @param r
            double. Radius
        """
        cdef shared_ptr[Hole] blob_ptr
        blob_ptr.reset(new Hole(x, y, r))
        self.this_ptr.initialVoids.push_back(blob_ptr)
        self.this_ptr.domainVoids.push_back(blob_ptr)

    def add_nondesign_domain_rectangle(self, double x, double y, double hx, double hy):
        """! @brief Add a nondesign domain of type Hole that represents a rectangle.""" 
        """! This will prevent the design from creating a void in the volume 
        defined by the nondesign region. The 'shape' defined by the nondesign 
        region will always remain throughout the topology optimization."""
        """! 
        @param x
            double. X coordinate of the origin of the cuboid
        @param y
            double. Y coordinate of the origin of the cuboid
        @param hx
            double. Half-width of the cuboid in the x direction
        @param hy
            double. Half-width of the cuboid in the y direction
        """
        # The 0.1 is to make sure that the point on the edges are included (matteo).
        cdef shared_ptr[Hole] blob_ptr
        blob_ptr.reset(new Hole(x, y, hx + 0.1, hy + 0.1))
        self.this_ptr.fixedBlobs.push_back(blob_ptr)

    def add_nondesign_domain_circle(self, double x, double y, double r):
        """! @brief Add a nondesign domain of type Hole that represents a circle.""" 
        """! This will prevent the design from creating a void in the volume 
        defined by the nondesign region. The 'shape' defined by the nondesign 
        region will always remain throughout the topology optimization."""
        """! 
        @param x
            double. X coordinate of the origin of the cuboid
        @param y
            double. Y coordinate of the origin of the cuboid
        @param r
            double. Radius
        """
        cdef shared_ptr[Hole] blob_ptr
        blob_ptr.reset(new Hole(x, y, r))
        self.this_ptr.fixedBlobs.push_back(blob_ptr)
    
    def create_level_set_boundary(self, double x, double y, double hx, double hy):
        """! @brief Create a level-set boundary in a region of the domain.""" 
        """! This is done by setting the signed distance values to zero."""
        """! 
        @param x
            double. X coordinate of the origin of the cuboid
        @param y
            double. Y coordinate of the origin of the cuboid
        @param hx
            double. Half-width of the cuboid in the x direction
        @param hy
            double. Half-width of the cuboid in the y direction
        """
        # The 0.1 is to make sure that the point on the edges are included (matteo).
        cdef shared_ptr[Hole] blob_ptr
        blob_ptr.reset(new Hole(x, y, hx + 0.1, hy + 0.1))
        self.this_ptr.boundaryBlobs.push_back(blob_ptr)

cdef class PyLevelSetModule:
    """! @brief Wrapper class for the level set module"""

    cdef LevelSetWrapper *this_ptr # pointer to underlying C++ class

    def __cinit__(self, PyInput py_input):
        """! @brief Initialize PyLevelSetModule and create a C++ LevelSetWrapper 
        object."""
        """!
        @param py_input
            PyInput. Class that contains the information needed to intitialize 
            the level set module.
        """
        self.this_ptr = new LevelSetWrapper(py_input.this_ptr[0])

    def __dealloc__(self):
        del self.this_ptr

    def calculate_element_densities(self, is_print=False):
        """! @brief Get a numpy array of the densities of each element in the 
        level set grid"""
        """!
        @param is_print
            bool. Specifies whether progresses statements for the function will 
            be printed. Useful for debugging.
        @return
            ndarray. Element densities.
        """
        # TODO: Change function so that C++ vector values aren't copied over to
        # a numpy array
        # get C++ vector of element densities
        cdef vector[double] cpp_rho
        cpp_rho = self.this_ptr.CalculateElementDensities(is_print)

        # Copy C++ vector values to numpy array
        nelem = cpp_rho.size()
        py_rho = np.zeros((nelem, 1), dtype=np.double)
        for e in range(nelem):
            py_rho[e, 0] = cpp_rho[e]

        # # set up C++ vector to be converted into numpy array
        # vec_size = cpp_rho.size()
        # cdef double* vec_ptr = &cpp_rho[0]

        # # convert c++ vector into numpy array
        # py_rho = inplace_array_1d(NPY_DOUBLE, vec_size, <void*>vec_ptr, NULL)

        return py_rho

    def map_sensitivities(self, np.ndarray elem_sens, is_print=False):
        """! @brief Maps the element sensitivities to boundary points. Use this 
        function for the objective element sensitivites and for the constraint 
        element sensitivities."""
        """!
        @param elem_sens
            ndarray. The derivative of the objective or constraint function with respect
            to the density of each element of the level set mesh (df/drho). The
            ndarray for the constraint element sensitivities should be of size
            (num boundary points, num constraints).
        @param is_print
            bool. Specifies whether progresses statements for the function will be
            printed. Useful for debugging.
        @return
            ndarray. The sensitivity (derivative) of the objective or constraint
            functions with respect to the boundary points.
        """
        # TODO: Change function so that C++ vector values aren't copied over to
        # a numpy array
        # Set up data structures
        dim = elem_sens.ndim # check if ndarray is 1d or 2d
        num_func = 1
        if dim == 2:
            num_func = elem_sens.shape[1]
        num_bpt  = self.this_ptr.GetNumberBoundaryPoints()
        py_mapped_sens = np.zeros((num_bpt, num_func), dtype=np.double)
        cdef vector[double] cpp_elem_sens
        cdef vector[double] cpp_mapped_sens

        # If interpolating a 1d numpy array, map a single objective/constraint
        if num_func == 1:
            # Copy elem_sens to a C++ vector
            cpp_elem_sens.resize(elem_sens.shape[0])
            for i in range(elem_sens.shape[0]): # loop through elements
                cpp_elem_sens[i] = elem_sens[i]

            # Call the underlying C++ function
            cpp_mapped_sens = self.this_ptr.MapSensitivities(cpp_elem_sens,
                                                             is_print)

            # Copy C++ vector values to numpy array
            for i in range(num_bpt): # loop through boundary points
                py_mapped_sens[i, 0] = cpp_mapped_sens[i]

        # If interpolating a 2d numpy array, map multiple objective/constraints
        elif num_func > 1:
            for nf in range(num_func):
                # Copy elem_sens to a C++ vector
                cpp_elem_sens.resize(elem_sens.shape[0])
                for i in range(elem_sens.shape[0]):
                    cpp_elem_sens[i] = elem_sens[i, nf]

                # Call the underlying C++ function
                cpp_mapped_sens = self.this_ptr.MapSensitivities(cpp_elem_sens,
                                                                 is_print)

                # Copy C++ vector values to numpy array
                for e in range(num_bpt):
                    py_mapped_sens[e, nf] = cpp_mapped_sens[e]

        return py_mapped_sens

    def map_volume_sensitivities(self):
        """! @brief Returns boundary point volume sensitivites"""
        """!
        @return
            ndarray. The sensitivity (derivative) of the the volume constraint 
            with respect to the boundary points.
        """
        # TODO: Change function so that C++ vector values aren't copied over to
        # a numpy array
        # Call the underlying C++ function
        cdef vector[double] cpp_mapped_sens
        cpp_mapped_sens = self.this_ptr.MapVolumeSensitivities()

        # Copy C++ vector values to numpy array
        num_bpt = self.this_ptr.GetNumberBoundaryPoints()
        py_mapped_sens = np.zeros((num_bpt, 1), dtype=np.double)
        for e in range(num_bpt):
            py_mapped_sens[e, 0] = cpp_mapped_sens[e]

        return py_mapped_sens

    def update(self, np.ndarray bpt_vel, double move_limit, is_print=False):
        """! @brief Updates the level set via advection"""
        """!
        @param bpt_vel
            ndarray. Numpy array of velocities for each of the boundary points.
        @param move_limit
            double. Maximum movement of the level set.
        @param is_print
            bool. Specifies whether progresses statements for the function will 
            be printed. Useful for debugging.
        """
        # TODO: Change function so that C++ vector values aren't copied over to
        # a numpy array
        # Copy bpt_vel to a C++ vector
        num_bpt = self.this_ptr.GetNumberBoundaryPoints()
        cdef vector[double] cpp_bpt_vel
        cpp_bpt_vel.resize(num_bpt)
        for i in range(num_bpt):
            cpp_bpt_vel[i] = bpt_vel[i,0] # remove DeprecationWarning (matteo)

        # Call the underlying C++ function
        self.this_ptr.Update(cpp_bpt_vel, is_print)

    def get_volume(self):
        """! @brief Gets the volume represented by the level set."""
        """! The function calculate_element_densities should be called before 
        calling this function."""
        """!
        @return
            double. The volume represented by the level set.
        """
        return self.this_ptr.GetVolume()

    def get_number_boundary_points(self):
        """! @brief Gets the number of boundary points."""
        """! The function calculate_element_densities should be called before 
        calling this function."""
        """!
        @return
            double. The number of boundary points.
        """
        return self.this_ptr.GetNumberBoundaryPoints()

    def get_limits(self, double move_limit):
        """! @brief Gets the lower and upper movement limit for each boundary 
        point."""
        """!
        @return
            ndarray. Rows of the array correspond to the boundary points. The 
            first column is the lower limit and the second column is the upper 
            limit.
        """
        # Call the underlying C++
        cdef vector[double] cpp_upper_limit
        cdef vector[double] cpp_lower_limit
        self.this_ptr.GetLimits(cpp_upper_limit, cpp_lower_limit)

        # Copy C++ vector values to a numpy array
        num_bpt = self.this_ptr.GetNumberBoundaryPoints()
        py_limits = np.zeros((num_bpt, 2), dtype=np.double)
        for i in range(num_bpt):
            py_limits[i, 0] = cpp_lower_limit[i]
            py_limits[i, 1] = cpp_upper_limit[i]

        return py_limits

    def write_densities_txt(self, int curr_iter = 0, str file_path = "",
                            str file_name = "dens"):
        """! @brief Writes element densities to a text file (.txt)"""
        """!
        @param curr_iter
            int. Iteration number of the topology optimization cycle.
        @param file_path
            str. Directory to save the file to.
        @param file_name
            str. Name of the text file.
        """
        self.this_ptr.WriteElementDensitiesToTxt(curr_iter,
            file_path.encode('utf-8'), file_name.encode('utf-8'))

    def write_vtk(self, int curr_iter = 0, str file_path = "",
        str file_name = "opt"):
        """! @brief Writes a vtk file of the surface defined by the level set 
        boundary"""
        """!
        @param curr_iter
            int. Iteration number of the topology optimization cycle.
        @param file_path
            str. Directory to save the file to.
        @param file_name
            str. Name of the text file.
        """
        self.this_ptr.WriteVtk(curr_iter, file_path.encode('utf-8'),
            file_name.encode('utf-8'))

    def write_boundary_limits_vtk(self):
        """! @brief Writes a vtk file of the boundary points limits"""
        self.this_ptr.WriteBoundaryVtk(True)

    def write_boundary_sensitivities_vtk(self):
        """! @brief Writes a vtk file of the boundary points sensitivities"""
        self.this_ptr.WriteBoundaryVtk(False)