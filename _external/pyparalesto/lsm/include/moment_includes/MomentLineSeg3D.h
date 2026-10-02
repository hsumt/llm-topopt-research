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

#ifndef MOMENTLINESEG3D_H
#define MOMENTLINESEG3D_H

#include <vector>

/*! \file MomentLineSeg3D.h
    \brief File that has the class for integrating over line segments for a
    polygon

    \todo TODO(Carolina): Since this file has implementation, it should
    conventionally be a .cpp not a .h file. File structure/extension should be
    changed in the future?
*/

namespace para_lsm {

/*! \class MomentLineSegment
    \brief Class is used to integrate over the line segments for a polygon that
    lies in one of the cartesian coordinate planes.

    Each instance of the class represents a single line segment.
*/
class MomentLineSegment {
 public:
  //! Constructor
  /*! \param p0_
        Start point of the line segment

      \param p1_
        End point of the line segment
  */
  MomentLineSegment (std::vector<double> p0_, std::vector<double> p1_);

  //! Set the normal of the segment with respect to the plane of the polygon
  /*! \todo TODO(Carolina): verify that this description is correct

      \param normal3D
        Normal vector
  */
  void SetSegNormal (std::vector<double> normal3d);

  //! Integrate the test functions over the line
  /*! \param normal3d
        \todo TODO(Carolina): add description of parameter

      \return
        \todo TODO(Carolina): add description
  */
  std::vector<double> Integrate3D (std::vector<double> normal3d);

  //! shape function and the integral
  /*! \param i
        \todo TODO(Carolina): add description

      \param x
        \todo TODO(Carolina): add description
  */
  double getFvalue (int i, double x);

  //! \todo TODO(Carolina): add description for this function
  /*! \param i
        \todo TODO(Carolina): add description

      \param x
        \todo TODO(Carolina): add description
  */
  double getBigFvalue (int i, double x);

  //! \todo TODO(Carolina): add description for this function
  /*! \param i
        \todo TODO(Carolina): add description

      \param x
        \todo TODO(Carolina): add description
  */
  double getBigGvalue (int i, double x);

  //! 5 point Gauss quadrature \todo TODO(Carolina): add more explanation
  std::vector<double> wGauss
      = {0.28444444444, 0.23931433525, 0.23931433525, 0.11846344252, 0.11846344252};

  //! 5 point Gauss quadrature \todo TODO(Carolina): add more explanation
  std::vector<double> xGauss = {0.5, 0.5 - 0.5 * 0.5384693101056831, 0.5 + 0.5 * 0.5384693101056831,
                                0.5 - 0.5 * 0.9061798459386640, 0.5 + 0.5 * 0.9061798459386640};

  std::vector<double> p0;         //!< Start point of the line segment
  std::vector<double> p1;         //!< End point of the line segment
  std::vector<double> segNormal;  //!< segment normal in the plane of the polygon
};


MomentLineSegment ::MomentLineSegment (std::vector<double> p0_, std::vector<double> p1_)
    : p0 (p0_), p1 (p1_) {}


double MomentLineSegment ::getFvalue (int i, double x) {
  return (i == -1) * 1.0 + (i == 0) * x + (i == 1) * x * x;
}


double MomentLineSegment ::getBigFvalue (int i, double x) {
  return (i == -1) * x + (i == 0) * x * x / 2.0 + (i == 1) * x * x * x / 3.0;
}


double MomentLineSegment ::getBigGvalue (int i, double x) {
  return (i == -1) * x * x / 2.0 + (i == 0) * x * x * x / 6.0 + (i == 1) * x * x * x * x / 12.0;
}


void MomentLineSegment ::SetSegNormal (std::vector<double> normal3d) {
  // computes segment normals assuming an anticlockwise arrangement of the
  // points in the polygon
  segNormal[0] = (p1[1] - p0[1]) * normal3d[2] - (p1[2] - p0[2]) * normal3d[1];
  segNormal[1] = (p1[2] - p0[2]) * normal3d[0] - (p1[0] - p0[0]) * normal3d[2];
  segNormal[2] = (p1[0] - p0[0]) * normal3d[1] - (p1[1] - p0[1]) * normal3d[0];

  double norm = std::sqrt (std::pow (segNormal[0], 2) + std::pow (segNormal[1], 2)
                           + std::pow (segNormal[2], 2));
  segNormal[0] /= norm;
  segNormal[1] /= norm;
  segNormal[2] /= norm;
}


std::vector<double> MomentLineSegment ::Integrate3D (std::vector<double> normal3d) {
  std::vector<double> moments;
  double nx = normal3d[0];
  double ny = normal3d[1];
  double nz = normal3d[2];

  if (segNormal.size () == 0) {
    segNormal.resize (3);
    SetSegNormal (normal3d);
  }

  for (int sz = -1; sz <= 1; sz = sz + 1) {
    for (int sy = -1; sy <= 1; sy = sy + 1) {
      for (int sx = -1; sx <= 1; sx = sx + 1) {
        double valuex = 0.0;
        double valuey = 0.0;
        double valuez = 0.0;
        for (int i = 0; i < wGauss.size (); i++) {
          double x = p0[0] * xGauss[i] + p1[0] * (1.0 - xGauss[i]);
          double y = p0[1] * xGauss[i] + p1[1] * (1.0 - xGauss[i]);
          double z = p0[2] * xGauss[i] + p1[2] * (1.0 - xGauss[i]);
          //==============================================================
          valuex += wGauss[i] / 6.0
                    * (nx * getBigGvalue (sx, x) * getFvalue (sy, y) * getFvalue (sz, z)
                       + ny * getBigFvalue (sx, x) * getBigFvalue (sy, y) * getFvalue (sz, z)
                       + nz * getBigFvalue (sx, x) * getFvalue (sy, y) * getBigFvalue (sz, z));
          valuey += wGauss[i] / 6.0
                    * (nx * getBigFvalue (sx, x) * getBigFvalue (sy, y) * getFvalue (sz, z)
                       + ny * getFvalue (sx, x) * getBigGvalue (sy, y) * getFvalue (sz, z)
                       + nz * getFvalue (sx, x) * getBigFvalue (sy, y) * getBigFvalue (sz, z));
          valuez += wGauss[i] / 6.0
                    * (nx * getBigFvalue (sx, x) * getFvalue (sy, y) * getBigFvalue (sz, z)
                       + ny * getFvalue (sx, x) * getBigFvalue (sy, y) * getBigFvalue (sz, z)
                       + nz * getFvalue (sx, x) * getFvalue (sy, y) * getBigGvalue (sz, z));
          //==============================================================
        }
        double norm = std::sqrt (std::pow (p0[0] - p1[0], 2) + std::pow (p0[1] - p1[1], 2)
                                 + std::pow (p0[2] - p1[2], 2));
        valuex *= norm;
        valuey *= norm;
        valuez *= norm;
        moments.push_back (valuex * segNormal[0] + valuey * segNormal[1] + valuez * segNormal[2]);
      }
    }
  }

  return moments;
}

}  // namespace para_lsm

#endif