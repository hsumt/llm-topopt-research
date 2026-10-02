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
from libcpp.vector cimport vector
from libcpp.string cimport string

from petsc4py.PETSc cimport Vec, PetscVec

cdef extern from "../include/initialize.h" namespace "para_fea":
    cdef cppclass Force:
        Force() except +
        Force(double, double, double, double, double, double, double, double,
              double) except +
        double valx, valy, valz, x, y, z, tolx, toly, tolz

cdef extern from "../include/initialize.h" namespace "para_fea":
    cdef cppclass FixedDof:
        FixedDof() except +
        FixedDof(bool, bool, bool, double, double, double, double, double,
                 double) except +
        bool valx, valy, valz
        double x, y, z, tolx, toly, tolz

cdef extern from "../include/initialize.h" namespace "para_fea":
    cdef cppclass HeatLoad:
        HeatLoad() except +
        HeatLoad(double, double, double, double, double, double,
                double) except +
        double val, x, y, z, tolx, toly, tolz

cdef extern from "../include/initialize.h" namespace "para_fea":
    cdef cppclass InitializeOpt:
        InitializeOpt() except +
        InitializeOpt(bool) except +
        InitializeOpt(bool, bool) except +
        int nelx, nely, nelz
        double lx, ly, lz
        int nlvls
        double Emax, Emin, nu, Kmax, Kmin
        vector[Force] loads, adjLoads
        vector[FixedDof] fixedDofs
        vector[HeatLoad] heatLoad, fixedHeatDof
        int maxItr
        double volCons

cdef extern from "../include/physics_wrapper.h" namespace "para_fea":
    cdef cppclass LinElasticWrapper:
        LinElasticWrapper(InitializeOpt &) except +
        void SolveState(vector[double], bool, double)
        PetscVec GetStateField()
        PetscVec GetStateRhs()
        void WriteDispToTxt(int, string, string)

cdef extern from "../include/physics_wrapper.h" namespace "para_fea":
    cdef cppclass PoissonWrapper:
        PoissonWrapper(InitializeOpt &) except +
        void SolveState(vector[double], bool, double)
        PetscVec GetStateField()
        PetscVec GetStateRhs()
        void WriteTempToTxt(int, string, string)

cdef extern from "../include/physics_wrapper.h" namespace "para_fea":
    cdef cppclass PhysicsWrapper:
        PhysicsWrapper() except +
        PhysicsWrapper(InitializeOpt &) except +
        vector[double] CalculatePartialDerivatives(vector[double], bool)
        void AssignVolumeFractions(vector[double])
        void WriteDispToTxt(int, string, string)
        void WriteTempToTxt(int, string, string)