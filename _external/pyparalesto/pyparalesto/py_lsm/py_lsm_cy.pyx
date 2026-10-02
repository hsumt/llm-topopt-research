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

"""! @file py_lsm_cy.pyx
     @brief Define the wrapper classes needed for the level set module: PyInput 
     and PyLevelSetModule classes.
"""

# cython: c_string_type=unicode, c_string_encoding=utf8

from pyparalesto.py_lsm.py_lsm_lib cimport *
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

    def __cinit__(self, int nelx, int nely, int nelz, int map_flag=0,
                  double perturbation=0.15):
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
        @param nelz
            int. Number of elements in the z direction for the level set grid. 
            Must be the same as the finite element mesh. (Will be changed in the
            future to handle different level set and finite element mesh sizes).
        """
        self.this_ptr = new InitializeLsm()
        self.this_ptr.nelx = nelx
        self.this_ptr.nely = nely
        self.this_ptr.nelz = nelz
        self.this_ptr.map_flag = map_flag
        self.this_ptr.perturbation = perturbation

    def __dealloc__(self):
        del self.this_ptr

    def add_initial_void_cuboid(self, double x, double y, double z, double hx,
                                double hy, double hz):
        """! @brief Add an initial void of type Cuboid (i.e., a box).""" 
        """! This will create a void (hole) in the initial design space that 
        will change during the topology optimization. The void may disappear, 
        combine with another void, or change shape."""
        """!
        @param x
            double. X coordinate of the origin of the cuboid
        @param y
            double. Y coordinate of the origin of the cuboid
        @param z
            double. Z coordinate of the origin of the cuboid
        @param hx
            double. Half-width of the cuboid in the x direction
        @param hy
            double. Half-width of the cuboid in the y direction
        @param hz
            double. Half-width of the cuboid in the z direction
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new Cuboid(x, y, z, hx, hy, hz))
        self.this_ptr.initialVoids.push_back(blob_ptr)

    def add_initial_void_cutplane(self, double a, double b, double c, double d1,
                                  double d2):
        """! @brief Add an initial void of type CutPlane.""" 
        """! This will create a void (hole) in the initial design space that 
        will change during the topology optimization. The void may disappear, 
        combine with another void, or change shape. Note that a CutPlane 
        represents a region between two parallel planes using equations of the 
        form a*x + b*y + c*z + d = 0."""
        """!
        @param a
            double. X coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param b
            double. Y coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param c
            double. Z coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param d1
            double. Offset for plane 1
        @param d2
            double. Offset for plane 2
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new CutPlane(a, b, c, d1, d2))
        self.this_ptr.initialVoids.push_back(blob_ptr)

    def add_initial_void_cylinder(self, double x, double y, double z, double h,
                                  double r, double r0, int dir_):
        """! @brief Add an initial void of type Cylinder (right circular 
        cylinder)."""
        """! This will create a void (hole) in the initial design space that 
        will change during the topology optimization. The void may disappear, 
        combine with another void, or change shape."""
        """!
        @param x
            double. X coordinate of the origin of the cylinder
        @param y
            double. Y coordinate of the origin of the cylinder
        @param z
            double. Z coordinate of the origin of the cylinder
        @param h
            double. Half of the height of the cylinder
        @param r
            double. Outer radius
        @param r0
            double. Inner radius
        @param dir_
            int. Orientation of the cylinder. For dir = 0 the axis of the cylinder is
            parallel to the x-axis; dir = 1 the axis of the cylinder is parallel
            to the y-axis; dir = 2 the axis of the cylinder is parallel to the
            z-axis.
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new Cylinder(x, y, z, h, r, r0, dir_))
        self.this_ptr.initialVoids.push_back(blob_ptr)

    def add_nondesign_void_cuboid(self, double x, double y, double z, double hx,
                                  double hy, double hz):
        """! @brief Add a nondesign void of type Cuboid (i.e., a box).""" 
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
        @param z
            double. Z coordinate of the origin of the cuboid
        @param hx
            double. Half-width of the cuboid in the x direction
        @param hy
            double. Half-width of the cuboid in the y direction
        @param hz
            double. Half-width of the cuboid in the z direction
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new Cuboid(x, y, z, hx, hy, hz))
        self.this_ptr.initialVoids.push_back(blob_ptr)
        self.this_ptr.domainVoids.push_back(blob_ptr)

    def add_nondesign_void_cutplane(self, double a, double b, double c,
                                    double d1, double d2):
        """! @brief Add a nondesign void of type CutPlane.""" 
        """! This will prevent the design from going into the volume defined by 
        the nondesign void. The region defined by the nondesign void will always 
        be void throughout the topology optimization. Note that a CutPlane 
        represents a region between two parallel planes using equations of the 
        form a*x + b*y + c*z + d = 0.
        Note that this function will automatically add the same region as an
        initial void (which is a necessary step) unlike the C++ API where the
        user has to do it manually."""
        """! 
        @param a
            double. X coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param b
            double. Y coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param c
            double. Z coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param d1
            double. Offset for plane 1
        @param d2
            double. Offset for plane 2
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new CutPlane(a, b, c, d1, d2))
        self.this_ptr.initialVoids.push_back(blob_ptr)
        self.this_ptr.domainVoids.push_back(blob_ptr)

    def add_nondesign_void_cylinder(self, double x, double y, double z, double h,
                                    double r, double r0, int dir_):
        """! @brief Add a nondesign void of type Cylinder (right circular 
        cylinder).""" 
        """! This will prevent the design from going into the volume defined by 
        the nondesign void. The region defined by the nondesign void will always 
        be void throughout the topology optimization.
        Note that this function will automatically add the same region as an
        initial void (which is a necessary step) unlike the C++ API where the
        user has to do it manually."""
        """!
        @param x
            double. X coordinate of the origin of the cylinder
        @param y
            double. Y coordinate of the origin of the cylinder
        @param z
            double. Z coordinate of the origin of the cylinder
        @param h
            double. Half of the height of the cylinder
        @param r
            double. Outer radius
        @param r0
            double. Inner radius
        @param dir_
            int. Orientation of the cylinder. For dir = 0 the axis of the cylinder is
            parallel to the x-axis; dir = 1 the axis of the cylinder is parallel
            to the y-axis; dir = 2 the axis of the cylinder is parallel to the
            z-axis.
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new Cylinder(x, y, z, h, r, r0, dir_))
        self.this_ptr.initialVoids.push_back(blob_ptr)
        self.this_ptr.domainVoids.push_back(blob_ptr)

    def add_nondesign_domain_cuboid(self, double x, double y, double z,
                                    double hx, double hy, double hz):
        """! @brief Add a nondesign region of type Cuboid (i.e., a box).""" 
        """! This will prevent the design from creating a void in the volume 
        defined by the nondesign region. The 'shape' defined by the nondesign 
        region will always remain throughout the topology optimization."""
        """! 
        @param x
            double. X coordinate of the origin of the cuboid
        @param y
            double. Y coordinate of the origin of the cuboid
        @param z
            double. Z coordinate of the origin of the cuboid
        @param hx
            double. Half-width of the cuboid in the x direction
        @param hy
            double. Half-width of the cuboid in the y direction
        @param hz
            double. Half-width of the cuboid in the z direction
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new Cuboid(x, y, z, hx, hy, hz))
        self.this_ptr.fixedBlobs.push_back(blob_ptr)

    def add_nondesign_domain_cutplane(self, double a, double b, double c,
                                      double d1, double d2):
        """! @brief Add a nondesign void of type CutPlane."""
        """! This will prevent the design from creating a void in the volume 
        defined by the nondesign region. The 'shape' defined by the nondesign 
        region will always remain throughout the topology optimization. Note 
        that a CutPlane represents a region between two parallel planes using 
        equations of the form a*x + b*y + c*z + d = 0."""
        """! 
        @param a
            double. X coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param b
            double. Y coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param c
            double. Z coefficient. Note this parameter is the same for both planes
            because they're parallel to each other.
        @param d1
            double. Offset for plane 1
        @param d2
            double. Offset for plane 2
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new CutPlane(a, b, c, d1, d2))
        self.this_ptr.fixedBlobs.push_back(blob_ptr)

    def add_nondesign_domain_cylinder(self, double x, double y, double z,
                                      double h, double r, double r0, int dir_):
        """! @brief Add a nondesign void of type Cylinder (right circular 
        cylinder)."""
        """! This will prevent the design from creating a void in the volume 
        defined by the nondesign region. The 'shape' defined by the nondesign 
        region will always remain throughout the topology optimization."""
        """! 
        @param x
            double. X coordinate of the origin of the cylinder
        @param y
            double. Y coordinate of the origin of the cylinder
        @param z
            double. Z coordinate of the origin of the cylinder
        @param h
            double. Half of the height of the cylinder
        @param r
            double. Outer radius
        @param r0
            double. Inner radius
        @param dir_
            int. Orientation of the cylinder. For dir = 0 the axis of the cylinder is
            parallel to the x-axis; dir = 1 the axis of the cylinder is parallel
            to the y-axis; dir = 2 the axis of the cylinder is parallel to the
            z-axis.
        """
        cdef shared_ptr[Blob] blob_ptr
        blob_ptr.reset(new Cylinder(x, y, z, h, r, r0, dir_))
        self.this_ptr.fixedBlobs.push_back(blob_ptr)

# cdef inplace_array_1d(int nptype, int dim1, void *data_ptr,
#                       PyObject *ptr):
#     """! @brief Wraps a C++ array with a numpy array for later useage."""
#     """! Method comes from TACS:
#     https://github.com/smdogroup/tacs/blob/master/tacs/TACS.pyx"""
#     """!
#     @param nptype
#         int. The data-type of the array.
#     @param dim1
#         int. Length of the 1d array
#     @param data_ptr
#         ptr. Pointer to the data (C++ array) meant to be wrapped
#     @param PyObject
#         ptr. Pointer to the object that owns the data
#     @return
#         ndarray. Numpy array version of the C++ vector
#     """
#     # Set the shape of the array
#     cdef int size = 1
#     cdef np.npy_intp shape[1]
#     cdef np.ndarray ndarray

#     # Set the first entry of the shape array
#     shape[0] = <np.npy_intp>dim1

#     # Create the array itself - Note that this function will not
#     # delete the data once the ndarray goes out of scope
#     ndarray = np.PyArray_SimpleNewFromData(size, shape,
#                                            nptype, data_ptr)

#     # Set the base class who owns the memory
#     if ptr != NULL:
#         ndarray.base = ptr

#     return ndarray

# cdef class VecArray:
#     """! @brief Wrapper class for C++ vectors."""
#     cdef vector[double] vector

#     def __cinit__(self):
#         self.vec_size = 0

#         # if cpp_vec is not None:
#         #     self.vec_size = cpp_vec.size()
#         #     self.ptr = &cpp_vec[0]
#         #     self.ptr.incref()
#         # return

#     def __dealloc__(self):
#         # if self.ptr:
#         #     self.ptr.decref()
#         # return
#         del self.vec_ptr

#     def set_vector(self, vector[double] cpp_vec):
#         """! @brief Assign the C++ vector to be wrapped"""
#         """!
#         @param cpp_vec
#             C++ vector. C++ vector that should be wrapped as a numpy array
#         """
#         self.vec_size = cpp_vec.size()
#         self.ptr = &cpp_vec[0]
#         # self.ptr.incref()

#     def get_array(self):
#         """! @brief Get the numpy array that wraps the C++ vector"""
#         # convert c++ vector into numpy array
#         array = inplace_array_1d(NPY_DOUBLE, self.vec_size, <void*>self.ptr,
#                                  <PyObject*>self)
#         Py_INCREF(self)
#         return array

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
        self.this_ptr.Update(cpp_bpt_vel, move_limit, is_print)

    def get_volume(self):
        """! @brief Gets the volume represented by the level set."""
        """! The function calculate_element_densities should be called before 
        calling this function."""
        """!
        @return
            double. The volume represented by the level set.
        """
        return self.this_ptr.GetVolume()

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
        self.this_ptr.GetLimits(cpp_upper_limit, cpp_lower_limit, move_limit)

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

    def write_stl(self, int curr_iter = 0, str file_path = "",
        str file_name = "opt"):
        """! @brief Writes an stl file of the surface defined by the level set 
        boundary"""
        """!
        @param curr_iter
            int. Iteration number of the topology optimization cycle.
        @param file_path
            str. Directory to save the file to.
        @param file_name
            str. Name of the text file.
        """
        self.this_ptr.WriteStl(curr_iter, file_path.encode('utf-8'),
            file_name.encode('utf-8'))