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

#ifndef MOMENTPOLYGON_H
#define MOMENTPOLYGON_H

#include <math.h>
#include <stdlib.h>

#include <iomanip>
#include <vector>

#include "MomentLineSeg3D.h"
#include "eigen3/Eigen/Dense"

/*! \file MomentPolygon.h
    \brief A file that has the class for level set

    \todo TODO(Carolina): Since this file has implementation, it should
    conventionally be a .cpp not a .h file. File structure/extension should be
    changed in the future?
*/

namespace para_lsm {

/*! \class MomentPolygon
    \brief Class used to compute moments of a polygon
*/
class MomentPolygon {
 public:
  //! Constructor from level set (only works in the x-y plane)
  /*! \param phi_
        Level set function
  */
  MomentPolygon (std::vector<double> phi_);

  //! Constructor from points
  /*! \param points_
        Vector of coordinates of the points

      \param polyNormal_
        \todo TODO(Carolina): add description
  */
  MomentPolygon (std::vector<std::vector<double>> points_, std::vector<double> polyNormal_);

  //! Constructor from triangle points
  /*! \param p0_

      \param p1_

      \param p2_

      \param planeNormal_
  */
  MomentPolygon (Eigen::Vector3d p0_, Eigen::Vector3d p1_, Eigen::Vector3d p2_,
                 Eigen::Vector3d planeNormal_);

  // Add description
  void SetPoints ();

  //! Compute normals of boundary segments
  void SetNormals ();

  //! Compute normals directions outside +ve
  void SetNormalDirections ();

  //! Compute moments
  void ComputeMoments ();

  //! \todo TODO(Carolina): Rotate the segment about an axis?
  /*! \param axis
        Rotation axis. 0, 1, 2 correspond to the x, y, z axis respectively.

      \param nRot
        Number of 180 degree rotations
  */
  void Rotate (int axis, int nRot);

  //! \todo TODO(Carolina): add description
  void SetPolyNormals ();

  //! \todo TODO(Carolina): add description
  /*! \param elemMoments
   */
  Eigen::VectorXd ComputeQuadrature (Eigen::VectorXd elemMoments);

  //! \todo TODO(Carolina): add description
  /*! \param i

      \param x
  */
  double getFunValue (int i, double x);

  //! \todo TODO(Carolina): add description
  /*! \param i

      \param x
  */
  double getLinearShapeFunValue (int i, double x);

  //! Compute moments for a triangle using guassian quadrature points for triangle
  void ComputeTriMoments ();

  //! Compute the equivalent area to volume
  std::vector<double> ComputeTriAreaToVolume ();

  /*! \name Used to define triangle points*/
  ///\{
  Eigen::Vector3d tri0;       //!< vertice of the triangle
  Eigen::Vector3d tri1;       //!< vertice of the triangle
  Eigen::Vector3d tri2;       //!< vertice of the triangle
  Eigen::Vector3d triNormal;  //!< normal of the triangle
  ///\}

  std::vector<double> polyNormal = {0, 0, 1};
  Eigen::VectorXd moments;
  int intersections = 0;
  std::vector<std::vector<double>> points;   //!< vector of boundary points
  std::vector<std::vector<double>> normals;  //!< vector of normals

 private:
  //! Interpolates the location of the zero crossing (boundary) between two points
  /*! \param p0
        Level set (signed distance) value of the first point

      \parm p1
        Level set (signed distance) value of the second point

      \return
        Distance to the zero crossing (boundary) relative to the location of
        p0. E.g. if p0 = -1, p1 = 3, the return value is 1/4; if p0 = 3,
        p1 = -1, the return value is 3/4.
  */
  double Interpolate (double p0, double p1);

  std::vector<double> xGauss = {0.5 - 0.5 * std::sqrt (0.6), 0.5, 0.5 + 0.5 * std::sqrt (0.6)};

  //! Number of quadrature points in each x, y, z direction
  int numQuadPointsPerDim = 3;

  //! Level set function
  std::vector<double> phi;
};


MomentPolygon ::MomentPolygon (std::vector<double> phi_) : phi (phi_) {}


MomentPolygon ::MomentPolygon (std::vector<std::vector<double>> points_,
                               std::vector<double> polyNormal_)
    : points (points_), polyNormal (polyNormal_) {}


MomentPolygon ::MomentPolygon (Eigen::Vector3d tri0_, Eigen::Vector3d tri1_, Eigen::Vector3d tri2_,
                               Eigen::Vector3d triNormal_)
    : tri0 (tri0_), tri1 (tri1_), tri2 (tri2_), triNormal (triNormal_) {}


void MomentPolygon ::SetPoints () {
  bool nonZeroNeighbourFlag = true;

  if (phi[0] == 0.0 && phi[1] == 0.0 && phi[3] == 0.0)
    nonZeroNeighbourFlag = false;
  else
    nonZeroNeighbourFlag = true;

  // y = 0 axis
  if (phi[0] > 0.0 || (phi[0] == 0.0 && nonZeroNeighbourFlag)) {
    points.push_back ({0.0, 0.0, 1.0});
  }
  if (phi[0] * phi[1] < 0.0) {
    intersections++;
    points.push_back ({Interpolate (phi[0], phi[1]), 0.0, 1.0});
  }

  if (phi[1] == 0.0 && phi[2] == 0.0 && phi[0] == 0.0)
    nonZeroNeighbourFlag = false;
  else
    nonZeroNeighbourFlag = true;
  // x = 1 axis
  if (phi[1] > 0.0 || (phi[1] == 0.0 && nonZeroNeighbourFlag)) {
    points.push_back ({1.0, 0.0, 1.0});
  }
  if (phi[1] * phi[2] < 0.0) {
    intersections++;
    points.push_back ({1.0, Interpolate (phi[1], phi[2]), 1.0});
  }

  if (phi[2] == 0.0 && phi[3] == 0.0 && phi[1] == 0.0)
    nonZeroNeighbourFlag = false;
  else
    nonZeroNeighbourFlag = true;
  // y = 1 axis
  if (phi[2] > 0.0 || (phi[2] == 0.0 && nonZeroNeighbourFlag)) {
    points.push_back ({1.0, 1.0, 1.0});
  }
  if (phi[2] * phi[3] < 0.0) {
    intersections++;
    points.push_back ({Interpolate (phi[3], phi[2]), 1.0, 1.0});
  }

  if (phi[3] == 0.0 && phi[2] == 0.0 && phi[0] == 0.0)
    nonZeroNeighbourFlag = false;
  else
    nonZeroNeighbourFlag = true;
  // x = 0 axis
  if (phi[3] > 0.0 || (phi[3] == 0.0 && nonZeroNeighbourFlag)) {
    points.push_back ({0.0, 1.0, 1.0});
  }
  if (phi[3] * phi[0] < 0.0) {
    intersections++;
    points.push_back ({0.0, Interpolate (phi[0], phi[3]), 1.0});
  }
}


double MomentPolygon ::Interpolate (double p0, double p1) {
  return std::abs (p0) / (std::abs (p0 - p1));
}


void MomentPolygon ::SetNormals () {
  normals.resize (points.size ());
  for (int i = 0; i < normals.size (); i++) {
    normals[i].resize (3);
    int j = (i + 1) % normals.size ();  // next index

    normals[i][0] = -(points[i][1] - points[j][1]);
    normals[i][1] = (points[i][0] - points[j][0]);
    // normalize
    double norm = std::sqrt (std::pow (normals[i][0], 2) + std::pow (normals[i][1], 2));

    if (norm >= 1.0e-8) {
      normals[i][0] /= norm;
      normals[i][1] /= norm;
    } else {
      normals[i][0] *= 0.0;
      normals[i][1] = 0.0;
    }
    normals[i][2] = 0.0;
  }

  SetNormalDirections ();
}


void MomentPolygon ::SetPolyNormals () {
  normals.resize (points.size ());
  for (int i = 0; i < normals.size (); i++) {
    normals[i].resize (3);
    int j = (i + 1) % normals.size ();  // next index

    // computes segment normals assuming an anticlockwise arrangement
    normals[i][0] = (points[j][1] - points[i][1]) * polyNormal[2]
                    - (points[j][2] - points[i][2]) * polyNormal[1];
    normals[i][1] = (points[j][2] - points[i][2]) * polyNormal[0]
                    - (points[j][0] - points[i][0]) * polyNormal[2];
    normals[i][2] = (points[j][0] - points[i][0]) * polyNormal[1]
                    - (points[j][1] - points[i][1]) * polyNormal[0];

    double norm = std::sqrt (std::pow (normals[i][0], 2) + std::pow (normals[i][1], 2)
                             + std::pow (normals[i][2], 2));
    // normalize
    if (norm >= 1.0e-8) {
      normals[i][0] /= norm;
      normals[i][1] /= norm;
      normals[i][2] /= norm;
    } else {
      normals[i][0] = 0.0;
      normals[i][1] = 0.0;
      normals[i][2] = 0.0;
    }
  }
  SetNormalDirections ();
}


void MomentPolygon ::Rotate (int axis, int nRot) {
  std::vector<double> rotAxis (3, 0.0);
  Eigen::Vector3d rotOrigin (0.5, 0.5, 0.5);
  rotAxis[axis] = 1.0;
  Eigen::MatrixXd rotMat = Eigen::MatrixXd::Identity (3, 3);

  // define sines and cosines (NOTE: rounding to 0/1 )
  double c = std::round (std::cos (M_PI / 2.0 * nRot));
  double s = std::round (std::sin (M_PI / 2.0 * nRot));
  double ux = rotAxis[0];
  double uy = rotAxis[1];
  double uz = rotAxis[2];

  rotMat (0, 0) = c + ux * ux * (1 - c);
  rotMat (0, 1) = ux * uy * (1 - c) - uz * s;
  rotMat (0, 2) = ux * uz * (1 - c) + uy * s;
  rotMat (1, 0) = ux * uy * (1 - c) + uz * s;
  rotMat (1, 1) = c + uy * uy * (1 - c);
  rotMat (1, 2) = uz * uy * (1 - c) - ux * s;
  rotMat (2, 0) = ux * uz * (1 - c) - uy * s;
  rotMat (2, 1) = uz * uy * (1 - c) + ux * s;
  rotMat (2, 2) = c + uz * uz * (1 - c);

  for (int i = 0; i < points.size (); i++) {
    Eigen::VectorXd normalNew = Eigen::VectorXd::Zero (3);
    normalNew[0] = normals[i][0];
    normalNew[1] = normals[i][1];
    normalNew[2] = normals[i][2];

    Eigen::VectorXd pointNew = Eigen::VectorXd::Zero (3);
    pointNew[0] = points[i][0] - rotOrigin[0];
    pointNew[1] = points[i][1] - rotOrigin[1];
    pointNew[2] = points[i][2] - rotOrigin[2];

    pointNew = rotMat * pointNew + rotOrigin;
    normalNew = rotMat * normalNew;

    points[i] = {pointNew[0], pointNew[1], pointNew[2]};
    normals[i] = {normalNew[0], normalNew[1], normalNew[2]};
  }

  Eigen::VectorXd polyNormalNew = Eigen::VectorXd::Zero (3);
  polyNormalNew[0] = polyNormal[0];
  polyNormalNew[1] = polyNormal[1];
  polyNormalNew[2] = polyNormal[2];
  polyNormalNew = rotMat * polyNormalNew;
  polyNormal = {polyNormalNew[0], polyNormalNew[1], polyNormalNew[2]};
}


void MomentPolygon ::SetNormalDirections () {
  for (int i = 0; i < normals.size (); i++) {
    int j = (i + 1) % normals.size ();  // next index
    // mid points
    double z = 0.5 * (points[i][2] + points[j][2]);
    double y = 0.5 * (points[i][1] + points[j][1]);
    double x = 0.5 * (points[i][0] + points[j][0]);

    double distOld;

    for (int mySign = -1; mySign <= 1; mySign = mySign + 2) {
      double dist2 = 0.0;
      double delta = 1.0e-4;  // some small number
      for (int k = 0; k < points.size (); k++) {
        dist2 += std::pow (x + mySign * delta * normals[i][0], 2)
                 + std::pow (y + mySign * delta * normals[i][1], 2)
                 + std::pow (z + mySign * delta * normals[i][2], 2);
      }
      if (mySign == -1)
        distOld == dist2;
      else if (dist2 < distOld) {
        // wrong normal direction, flip sign
        normals[i][0] *= -1;
        normals[i][1] *= -1;
        normals[i][2] *= -1;
      }
    }

    if (std::isnan (normals[i][0]) || std::isnan (normals[i][1]) || std::isnan (normals[i][2])) {
      normals[i][0] = 0.0;
      normals[i][1] = 0.0;
      normals[i][2] = 0.0;
    }
  }
}


double MomentPolygon ::getFunValue (int i, double x) {
  return (i == -1) * 1.0 + (i == 0) * x + (i == 1) * x * x;
}


double MomentPolygon ::getLinearShapeFunValue (int i, double x) {
  return (i == -1) * (1.0 - x) + (i == 1) * x;
}


#if 1
Eigen::VectorXd MomentPolygon ::ComputeQuadrature (Eigen::VectorXd elemMoments) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero (std::pow (numQuadPointsPerDim, 3),
                                             std::pow (numQuadPointsPerDim, 3));

  // construct A
  int countFun = 0;
  for (int sz = -1; sz <= 1; sz = sz + 1) {
    for (int sy = -1; sy <= 1; sy = sy + 1) {
      for (int sx = -1; sx <= 1; sx = sx + 1) {
        int countGauss = 0;
        for (int k = 0; k < xGauss.size (); k++) {
          for (int j = 0; j < xGauss.size (); j++) {
            for (int i = 0; i < xGauss.size (); i++) {
              A (countFun, countGauss) = getFunValue (sx, xGauss[i]) * getFunValue (sy, xGauss[j])
                                         * getFunValue (sz, xGauss[k]);
              countGauss++;
            }  // k
          }    // j
        }      // i
        countFun++;
      }  // sx
    }    // sy
  }      // sz

  // std::cout <<"Echo 3" << std::endl;

  // Eigen::VectorXd X = A.colPivHouseholderQr().solve(moments);
  Eigen::VectorXd X = A.fullPivLu ().solve (elemMoments);
  // Eigen::VectorXd X = A.householderQr().solve(elemMoments);

  // Eigen::MatrixXd Ainv = A.inverse();
  //
  // // std::cout << "Ainv = " << std::endl;
  // // std::cout << std::setprecision(16) << Ainv << std::endl;
  // for(int i =0; i < 27; i++){
  //   for(int j =0; j < 27; j++){
  //     std::cout << std::setprecision(16) << "Ainv(" << i << ", " << j <<") = " << Ainv(i,j) <<
  //     ";" << std::endl;
  //   }
  // }

  return X;
}
#endif


void MomentPolygon ::ComputeMoments () {
  moments = Eigen::VectorXd::Zero (std::pow (numQuadPointsPerDim, 3));

  for (int i = 0; i < normals.size (); i++) {
    int j = (i + 1) % normals.size ();  // next index
    MomentLineSegment mySeg (points[i], points[j]);
    std::vector<double> segMoment = mySeg.Integrate3D (polyNormal);
    for (int k = 0; k < moments.size (); k++) {
      moments[k] += segMoment[k];
    }
  }
}


void MomentPolygon ::ComputeTriMoments () {
  // Guassian quadrature for triangles
  Eigen::MatrixXd XTriGauss (16, 2);
  XTriGauss << 0.0571041961, 0.06546699455602246, 0.2768430136, 0.05021012321401679, 0.5835904324,
      0.02891208422223085, 0.8602401357, 0.009703785123906346, 0.0571041961, 0.3111645522491480,
      0.2768430136, 0.2386486597440242, 0.5835904324, 0.1374191041243166, 0.8602401357,
      0.04612207989200404, 0.0571041961, 0.6317312516508520, 0.2768430136, 0.4845083266559759,
      0.5835904324, 0.2789904634756834, 0.8602401357, 0.09363778440799593, 0.0571041961,
      0.8774288093439775, 0.2768430136, 0.6729468631859832, 0.5835904324, 0.3874974833777692,
      0.8602401357, 0.1300560791760936;

  Eigen::MatrixXd WTriGauss (16, 1);
  WTriGauss << 0.04713673637581137, 0.07077613579259895, 0.04516809856187617, 0.01084645180365496,
      0.08837017702418863, 0.1326884322074010, 0.08467944903812383, 0.02033451909634504,
      0.08837017702418863, 0.1326884322074010, 0.08467944903812383, 0.02033451909634504,
      0.04713673637581137, 0.07077613579259895, 0.04516809856187617, 0.01084645180365496;

  double triArea = 0.5 * ((tri1 - tri0).cross (tri2 - tri0)).norm ();
  moments = Eigen::VectorXd::Zero (std::pow (numQuadPointsPerDim, 3));

  std::vector<double> dummyPoint (3, 0.0);
  MomentLineSegment mySeg (dummyPoint, dummyPoint);
  int count = 0;
  for (int sz = -1; sz <= 1; sz = sz + 1) {
    for (int sy = -1; sy <= 1; sy = sy + 1) {
      for (int sx = -1; sx <= 1; sx = sx + 1) {
        double mySum = 0.0;
        for (int j = 0; j < 16; j++) {
          double x = tri0[0] + XTriGauss (j, 0) * (tri1[0] - tri0[0])
                     + XTriGauss (j, 1) * (tri2[0] - tri0[0]);
          double y = tri0[1] + XTriGauss (j, 0) * (tri1[1] - tri0[1])
                     + XTriGauss (j, 1) * (tri2[1] - tri0[1]);
          double z = tri0[2] + XTriGauss (j, 0) * (tri1[2] - tri0[2])
                     + XTriGauss (j, 1) * (tri2[2] - tri0[2]);
          double myVal = triNormal[0] * mySeg.getBigFvalue (sx, x) * mySeg.getFvalue (sy, y)
                             * mySeg.getFvalue (sz, z) / 3.0
                         + triNormal[1] * mySeg.getFvalue (sx, x) * mySeg.getBigFvalue (sy, y)
                               * mySeg.getFvalue (sz, z) / 3.0
                         + triNormal[2] * mySeg.getFvalue (sx, x) * mySeg.getFvalue (sy, y)
                               * mySeg.getBigFvalue (sz, z) / 3.0;
          mySum += triArea * WTriGauss (j, 0) * myVal;
        }
        moments[count] += mySum;
        count++;
      }
    }
  }
}


std::vector<double> MomentPolygon ::ComputeTriAreaToVolume () {
  // Guassian quadrature for triangles
  Eigen::MatrixXd XTriGauss (16, 2);
  XTriGauss << 0.0571041961, 0.06546699455602246, 0.2768430136, 0.05021012321401679, 0.5835904324,
      0.02891208422223085, 0.8602401357, 0.009703785123906346, 0.0571041961, 0.3111645522491480,
      0.2768430136, 0.2386486597440242, 0.5835904324, 0.1374191041243166, 0.8602401357,
      0.04612207989200404, 0.0571041961, 0.6317312516508520, 0.2768430136, 0.4845083266559759,
      0.5835904324, 0.2789904634756834, 0.8602401357, 0.09363778440799593, 0.0571041961,
      0.8774288093439775, 0.2768430136, 0.6729468631859832, 0.5835904324, 0.3874974833777692,
      0.8602401357, 0.1300560791760936;

  Eigen::MatrixXd WTriGauss (16, 1);
  WTriGauss << 0.04713673637581137, 0.07077613579259895, 0.04516809856187617, 0.01084645180365496,
      0.08837017702418863, 0.1326884322074010, 0.08467944903812383, 0.02033451909634504,
      0.08837017702418863, 0.1326884322074010, 0.08467944903812383, 0.02033451909634504,
      0.04713673637581137, 0.07077613579259895, 0.04516809856187617, 0.01084645180365496;

  std::vector<double> areatovolume (8, 0);

  int xArr[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
  int yArr[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
  int zArr[8] = {-1, -1, -1, -1, 1, 1, 1, 1};

  double triArea = 0.5 * ((tri1 - tri0).cross (tri2 - tri0)).norm ();

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 16; j++) {
      double x = tri0[0] + XTriGauss (j, 0) * (tri1[0] - tri0[0])
                 + XTriGauss (j, 1) * (tri2[0] - tri0[0]);
      double y = tri0[1] + XTriGauss (j, 0) * (tri1[1] - tri0[1])
                 + XTriGauss (j, 1) * (tri2[1] - tri0[1]);
      double z = tri0[2] + XTriGauss (j, 0) * (tri1[2] - tri0[2])
                 + XTriGauss (j, 1) * (tri2[2] - tri0[2]);

      areatovolume[i] += WTriGauss (j, 0) * triArea * getLinearShapeFunValue (xArr[i], x)
                         * getLinearShapeFunValue (yArr[i], y)
                         * getLinearShapeFunValue (zArr[i], z);
    }  // j
  }    // i

  return areatovolume;
}

}  // namespace para_lsm

#endif