//
// Copyright 2021 H Alicia Kim
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// A note from Carolina:
// This file defines a default constructor for the InitializeOpt class. It is
// needed for wrapping the class in Cython to be used in Python. This
// implementation file SHOULD NOT be linked if using the library from C++.

#include "initialize.h"

namespace para_fea {

Force ::Force ()
    : valx (0), valy (0), valz (0), x (0), y (0), z (0), tolx (0), toly (0), tolz (0) {}


Force ::Force (double valx_, double valy_, double valz_, double x_, double y_, double z_,
               double tolx_, double toly_, double tolz_)
    : valx (valx_),
      valy (valy_),
      valz (valz_),
      x (x_),
      y (y_),
      z (z_),
      tolx (tolx_),
      toly (toly_),
      tolz (tolz_) {}


FixedDof ::FixedDof (bool valx_, bool valy_, bool valz_, double x_, double y_, double z_,
                     double tolx_, double toly_, double tolz_)
    : valx (valx_),
      valy (valy_),
      valz (valz_),
      x (x_),
      y (y_),
      z (z_),
      tolx (tolx_),
      toly (toly_),
      tolz (tolz_) {}


HeatLoad ::HeatLoad () : val (0), x (0), y (0), z (0), tolx (0), toly (0), tolz (0) {}


HeatLoad ::HeatLoad (double val_, double x_, double y_, double z_, double tolx_, double toly_,
                     double tolz_)
    : val (val_), x (x_), y (y_), z (z_), tolx (tolx_), toly (toly_), tolz (tolz_) {}


InitializeOpt ::InitializeOpt () {}

}  // namespace para_fea