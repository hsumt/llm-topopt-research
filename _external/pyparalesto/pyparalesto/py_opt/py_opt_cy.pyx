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

"""! @file py_opt_cy.pyx
     @brief Define the wrapper class PyOptimizerModule for the optimization 
     module.
"""

from pyparalesto.py_opt.py_opt_lib cimport *
from libcpp cimport bool, int
from libcpp.vector cimport vector

# Import numpy
cimport numpy as np
import numpy as np

# cython wrapper class for optimization module
cdef class PyOptimizerModule:
    """! @brief Wrapper class for the optimization module"""

    cdef OptimizerWrapper *this_ptr # pointer to underlying C++ class
    cdef int num_cons # number of constraints

    def __cinit__(self, int num_cons, np.ndarray max_cons_vals,
                  int opt_algo = 0, bool is_3d = True):
        """! @brief Initialize PyOptimizerModule and create a C++ 
        OptimizerWrapper object."""
        """!
        @param num_cons
            int. Number of constraints.
        @param max_cons_vals 
            ndarray. The maximum values for each constraint.
        @param opt_algo
            int. Optimization algorithm. 0 for Newton Raphson, 1 for MMA (Method 
            of Moving Asymptotes) using the NLopt library, 2 for Simplex, 3 for Ipopt.
            Note that MMA, Simplex, and Ipopt have not been tested and may cause
            errors if used.
        @param is_3d
            bool. Specifies whether the optimization is for a 3D or 2D problem.
        """
        # TODO: Change function so that C++ vector values aren't copied over to
        # a numpy array
        # Store number of constraints
        self.num_cons = num_cons

        # Copy max_cons_vals to a C++ vector
        cdef vector[double] cpp_vals
        cpp_vals.resize(num_cons)
        for i in range(num_cons):
            cpp_vals[i] = max_cons_vals[i]

        # Call the underlying C++ function
        self.this_ptr = new OptimizerWrapper(num_cons, cpp_vals, opt_algo, is_3d)

    def __dealloc__(self):
        del self.this_ptr

    def solve(self, np.ndarray obj_sens, np.ndarray cons_sens,
              np.ndarray curr_cons_vals, bool is_print = False):
        """! @brief Solves optimization"""
        """!
        @param obj_sens
            ndarray. Sensitivity (derivative) of the objective function with 
            respect to boundary points.
        @param cons_sens
            ndarray. Sensitivity (derivative) of each constraint function with 
            respect to boundary points.
        @param curr_cons_vals
            ndarray. Current value of each constraint.
        @param is_print
            bool. Specifies whether progresses statements for the method will be
            printed. Useful for debugging.
        @return
            ndarray. The velocity of each boundary point.
        """
        # TODO: Change function so that C++ vector values aren't copied over to
        # a numpy array
        # Copy obj_sens to a C++ vector
        num_des_vars = obj_sens.shape[0]
        cdef vector[double] cpp_obj_sens
        cpp_obj_sens.resize(num_des_vars)
        for i in range(num_des_vars):
            cpp_obj_sens[i] = obj_sens[i,0] # remove DeprecationWarning (matteo)

        # Copy cons_sens to a C++ vector
        cdef vector[vector[double]] cpp_cons_sens
        cpp_cons_sens.resize(self.num_cons)
        for j in range(self.num_cons):
            cpp_cons_sens[j].resize(num_des_vars)
            for i in range(num_des_vars):
                cpp_cons_sens[j][i] = cons_sens[i,j]

        # Call the underlying C++ function
        cdef vector[double] bpt_vel
        bpt_vel = self.this_ptr.Solve(cpp_obj_sens, cpp_cons_sens,
                                      curr_cons_vals, is_print)

        # Convert C++ vector to numpy array
        py_bpt_vel = np.zeros((num_des_vars, 1), dtype=np.double)
        for i in range(num_des_vars):
            py_bpt_vel[i] = bpt_vel[i]

        return py_bpt_vel

    def set_limits(self, np.ndarray limits):
        """! @brief Sets the limits for the movement of each boundary point"""
        """!
        @param limits
            ndarray. Lower and upper limit of movement for each boundary point.
        """
        # TODO: Change function so that C++ vector values aren't copied over to
        # a numpy array
        # Copy limits to a C++ vectors
        num_des_vars = limits.shape[0]
        cdef vector[double] lower_limit
        lower_limit.resize(num_des_vars)
        cdef vector[double] upper_limit
        upper_limit.resize(num_des_vars)
        for i in range(num_des_vars):
            lower_limit[i] = limits[i, 0]
            upper_limit[i] = limits[i, 1]

        # Call the underlying C++ function
        self.this_ptr.SetLimits(upper_limit, lower_limit)