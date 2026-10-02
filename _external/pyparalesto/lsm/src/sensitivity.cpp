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

#include "sensitivity.h"

namespace para_lsm {

Sensitivity ::Sensitivity (double r0_, double r1_, double r2_, int p0_, int p1_, int q0_, int q1_,
                           double Xc_, double Yc_, double Zc_)
    : r0 (r0_),
      r1 (r1_),
      r2 (r2_),
      p0 (p0_),
      p1 (p1_),
      q0 (q0_),
      q1 (q1_),
      Xc (Xc_),
      Yc (Yc_),
      Zc (Zc_) {}


double Sensitivity ::GetSensitivity (std::vector<double> bPoint_) {
  Eigen::Vector3d bPoint (bPoint_[0], bPoint_[1], bPoint_[2]);
  // find nearest approximate value of t from the current point to the torus
  int nToursKnotDisc = 50;
  double tmin = 0.0;
  double distmin = 1.0e5;
  for (int cT = 0; cT <= nToursKnotDisc; cT++) {  // level-0
    double t = 2 * M_PI / nToursKnotDisc * cT;
    Eigen::Vector3d X0 = GetPointOnTorusKnot (t);
    double dist = (bPoint - X0).norm ();  // compute min distance
    if (dist < distmin) {
      distmin = dist;
      tmin = t;
    }
  }

  int nSensitivityDiscL1 = 1000;
  int nSensitivityDiscL2 = 20000;
  int spacingL1 = 40;
  int spacingL2 = 40;
  double tmin0 = tmin;
  distmin = 1.0e5;
  for (int cT = -spacingL1; cT <= spacingL1; cT++) {  // level-1
    double t = tmin0 + 2 * M_PI / nSensitivityDiscL1 * cT;
    Eigen::Vector3d X0 = GetPointOnSensitivity (t);
    double dist = (bPoint - X0).norm ();
    if (dist < distmin) {
      distmin = dist;
      tmin = t;
    }  // compute min distance
  }    // cT loop

  tmin0 = tmin;
  for (int cT = -spacingL2; cT <= spacingL2; cT++) {  // level-2
    double t = tmin0 + 2 * M_PI / nSensitivityDiscL2 * cT;
    Eigen::Vector3d X0 = GetPointOnSensitivity (t);
    double dist = (bPoint - X0).norm ();
    if (dist < distmin) {
      distmin = dist;
      tmin = t;
    }  // compute min distance
  }    // cT loop

  return -1.0 / (distmin * distmin + 1.0);
}


Eigen::Vector3d Sensitivity ::GetPointOnTorus (double t) {
  Eigen::Vector3d Xc0 (0.0, 0.0, 0.0);

  // define sines and cosines
  double sq0 = std::sin (q0 * t);
  double cq0 = std::cos (q0 * t);

  Xc0[0] = Xc + r0 * cq0;
  Xc0[1] = Yc;
  Xc0[2] = Zc + r0 * sq0;

  return Xc0;
}


Eigen::Vector3d Sensitivity ::GetPointOnTorusKnot (double t) {
  Eigen::Vector3d X0 (0.0, 0.0, 0.0);

  // get point on the torus
  Eigen::Vector3d Xc0 = GetPointOnTorus (t);

  // define sines and cosines
  double sq0 = std::sin (q0 * t);
  double cq0 = std::cos (q0 * t);
  double sp0 = std::sin (p0 * t);
  double cp0 = std::cos (p0 * t);

  // define X0
  X0[0] = r1 * cq0 * cp0;
  X0[1] = r1 * sp0;
  X0[2] = r1 * sq0 * cp0;

  // add X0 to point on torus
  X0 = X0 + Xc0;

  return X0;
}


Eigen::Vector3d Sensitivity ::GetPointOnSensitivity (double t) {
  Eigen::Vector3d X (0.0, 0.0, 0.0);
  Eigen::Vector3d Xt (0.0, 0.0, 0.0);
  Eigen::Vector3d Xr (0.0, 0.0, 0.0);

  // get point on the torus
  Eigen::Vector3d Xc = GetPointOnTorusKnot (t);

  // define sines and cosines
  double sq0 = std::sin (q0 * t);
  double cq0 = std::cos (q0 * t);
  double sp0 = std::sin (p0 * t);
  double cp0 = std::cos (p0 * t);

  // define tangential vector (dX/dt)
  Xt[0] = -q0 * sq0 * r0 - q0 * sq0 * (r1 * cp0) - p0 * cq0 * (r1 * sp0);
  Xt[2] = q0 * cq0 * r0 + q0 * cq0 * (r1 * cp0) - p0 * sq0 * (r1 * sp0);
  Xt[1] = p0 * r1 * cp0;
  Xt = Xt / Xt.norm ();

  // define radial coordinates (dX/dr)
  Xr[0] = cq0 * cp0;
  Xr[2] = sq0 * cp0;
  Xr[1] = sp0;

  // define normal
  Eigen::Vector3d Xn = Xt.cross (Xr);

  // add X0 to point on torus
  X = Xc + r2 * (std::sin (p1 * t) * Xn + std::cos (p1 * t) * Xr);

  return X;
}

}  // namespace para_lsm