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

cdef extern from "../../lsm/include/lsm_3d.h" namespace "para_lsm":
    cdef cppclass Blob:
        Blob() except +
        Blob(double, double, double) except +
        double GetMinDist (vector[double])
        vector[double] RotatePoint (vector[double])
        double x, y, z, theta
        int flipDistanceSign

cdef extern from "../../lsm/include/lsm_3d.h" namespace "para_lsm":
    cdef cppclass Cuboid(Blob):
        Cuboid() except +
        Cuboid(double, double, double, double, double, double) except +
        double hx, hy, hz

cdef extern from "../../lsm/include/lsm_3d.h" namespace "para_lsm":
    cdef cppclass CutPlane(Blob):
        CutPlane() except +
        CutPlane(double, double, double, double, double) except +
        double a, b, c, d1, d2

cdef extern from "../../lsm/include/lsm_3d.h" namespace "para_lsm":
    cdef cppclass Cylinder(Blob):
        Cylinder() except +
        Cylinder(double, double, double, double, double, double, double) except +
        double h, r, r0
        int dir

cdef extern from "../../lsm/include/initialize_lsm.h" namespace "para_lsm":
    cdef cppclass InitializeLsm:
        InitializeLsm() except +
        int nelx, nely, nelz
        vector[shared_ptr[Blob]] initialVoids, domainVoids, fixedBlobs
        int map_flag
        double perturbation

cdef extern from "../../lsm/include/level_set_wrapper.h" namespace "para_lsm":
    cdef cppclass LevelSetWrapper:
        LevelSetWrapper() except +
        LevelSetWrapper(InitializeLsm &) except +
        vector[double] CalculateElementDensities(bool)
        vector[double] MapSensitivities(vector[double], bool)
        vector[double] MapVolumeSensitivities()
        void Update(vector[double], double, bool)
        double GetVolume()
        int GetNumberBoundaryPoints()
        void GetLimits(vector[double], vector[double], double)
        void WriteElementDensitiesToTxt (int, string, string)
        void WriteStl (int, string, string)