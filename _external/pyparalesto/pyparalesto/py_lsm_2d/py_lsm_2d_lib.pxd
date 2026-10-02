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

from libcpp cimport bool, int
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "../../lsm_2d/include/hole.h" namespace "lsm_2d":
    cdef cppclass Hole:
        Hole() except +
        Hole(double, double, double) except +
        Hole(double, double, double, double) except +

cdef extern from "../../lsm_2d/include/initialize_lsm.h" namespace "lsm_2d":
    cdef cppclass InitializeLsm:
        InitializeLsm() except +
        int nelx, nely
        vector[shared_ptr[Hole]] initialVoids, domainVoids, fixedBlobs, boundaryBlobs
        int map_flag
        double perturbation
        double move_limit

cdef extern from "../../lsm_2d/include/level_set_wrapper.h" namespace "lsm_2d":
    cdef cppclass LevelSetWrapper:
        LevelSetWrapper() except +
        LevelSetWrapper(InitializeLsm &) except +
        vector[double] CalculateElementDensities(bool)
        vector[double] MapSensitivities(vector[double], bool)
        vector[double] MapVolumeSensitivities()
        void Update(vector[double], bool)
        double GetVolume()
        int GetNumberBoundaryPoints()
        void GetLimits(vector[double], vector[double])
        void WriteElementDensitiesToTxt (int, string, string)
        void WriteVtk (int, string, string)
        void WriteBoundaryVtk(bool)