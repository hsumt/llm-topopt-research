//
// Copyright 2020 H Alicia Kim
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

#ifndef SENSITIVITY_H
#define SENSITIVITY_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "eigen3/Eigen/Dense"

/*! \file sensitivity.h
    \brief A file that has the class sensitivity.
*/

namespace para_lsm {

/*! \class Sensitivity
    \brief A class that computes boundary point sensitivities.

    This class is specific to the geometry matching for a helix torus knot.
*/
class Sensitivity {
 public:
  //! Constructor.
  /*! \param r0_
        parameters of the helix

      \param r1_
        parameters of the helix

      \param r2_
        parameters of the helix

      \param p0_
        parameters of the helix

      \param p1_
        parameters of the helix

      \param q0_
        parameters of the helix

      \param q1_
        parameters of the helix

      \param Xc_
        parameters of the helix

      \param Yc_
        parameters of the helix

      \param Zc_
        parameters of the helix
  */
  Sensitivity (double r0_, double r1_, double r2_, int p0_, int p1_, int q0_, int q1_, double Xc_,
               double Yc_, double Zc_);

  //! Get sensitivity
  /*! \param bPoint_
        Boundary point coordinates
  */
  double GetSensitivity (std::vector<double> bPoint_);

 private:
  //! Get the coordinates of the Torus center-line at t
  /*! \param t
        \todo TODO(Carolina): add description
  */
  Eigen::Vector3d GetPointOnTorus (double t);

  //! Get the coordinates of the Torus-knot center-line at t
  /*! \param t
        \todo TODO(Carolina): add description
  */
  Eigen::Vector3d GetPointOnTorusKnot (double t);

  //! Get the coordinates of the Sensitivity center-line at t
  /*! \param t
        \todo TODO(Carolina): add description
  */
  Eigen::Vector3d GetPointOnSensitivity (double t);

  double r0;  //!< parameters of the helix
  double r1;  //!< parameters of the helix
  double r2;  //!< parameters of the helix
  int p0;     //!< parameters of the helix
  int p1;     //!< parameters of the helix
  int q0;     //!< parameters of the helix
  int q1;     //!< parameters of the helix
  double Xc;  //!< parameters of the helix
  double Yc;  //!< parameters of the helix
  double Zc;  //!< parameters of the helix
};

}  // namespace para_lsm

#endif