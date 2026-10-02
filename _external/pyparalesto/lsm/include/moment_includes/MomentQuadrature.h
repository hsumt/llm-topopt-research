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

#ifndef MOMENTQUADRATURE_H
#define MOMENTQUADRATURE_H

#include <stdlib.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../boundary.h"
#include "MomentLineSeg3D.h"
#include "MomentPolygon.h"
#include "eigen3/Eigen/Dense"

/*! \file MomentQuadrature.h
    \brief A file that contains the class for computing quadrature

    \todo TODO(Carolina): Since this file has implementation, it should
    conventionally be a .cpp not a .h file. File structure/extension should be
    changed in the future?
*/

namespace para_lsm {

/*! \class MomentQuadrature
    \brief Computes the quadrature fora given cut element

    Phi and a vector of triangles are the inputs
*/
class MomentQuadrature {
 public:
  //! Constructor
  /*! \param boundary_
        Reference to the level set boundary
  */
  MomentQuadrature (Boundary &boundary_);

  //! /todo TODO(Carolina): add description
  /*! \param phi
        Level set function

      \param triaMC

      \param originX

      \param originY

      \param originZ
  */
  Eigen::VectorXd ComputeQuadrature (std::vector<double> phi, std::vector<Triangle> triaMC,
                                     int originX, int originY, int originZ);

  //! /todo TODO(Carolina): add description
  void ComputeElementsQuadrature ();

  //! /todo TODO(Carolina): add description
  /*! \param ignoreBoundingBox
   */
  void ComputeElementsAreaToVolume (bool ignoreBoundingBox = false);

  //! /todo TODO(Carolina): add description
  /*! \param triaMC

      \param originX

      \param originY

      \param originZ
  */
  std::vector<double> ComputeAreaToVolume (std::vector<Triangle> triaMC, int originX, int originY,
                                           int originZ);

  Boundary &boundary;  //!< reference to the level set boundary object
  std::vector<Eigen::VectorXd> elememtsQuadrature;
  double cutOffUpper = 0.999;
  double cutOffLower = 0.001;
  int numQuadPoints = 27;
  std::vector<std::vector<double>> elementsAreaToVolume;
};


void MomentQuadrature ::ComputeElementsQuadrature () {
  int nCells = boundary.levelSet.nCells;
  int nElemX = boundary.levelSet.grid.nelx;
  int nElemY = boundary.levelSet.grid.nely;
  int nElemZ = boundary.levelSet.grid.nelz;
  // resize quadrature vector
  elememtsQuadrature.resize (nCells);

#pragma omp parallel for
  for (int i = 0; i < nElemX; i++) {
    for (int j = 0; j < nElemY; j++) {
      for (int k = 0; k < nElemZ; k++) {
        // cell counter
        int countCell = i + nElemX * j + nElemX * nElemY * k;
        // element volume fraction
        double elemVolFrac = boundary.levelSet.volFractions[countCell];
        elememtsQuadrature[countCell].resize (0);  // reize for memory saving

        if (elemVolFrac <= cutOffUpper && elemVolFrac >= cutOffLower) {
          // compute moments
          elememtsQuadrature[countCell].resize (numQuadPoints);

          // construct the stencil
          Stencil mStencil (boundary.levelSet.grid, i, j, k);
          // get the corner point level set values
          std::vector<double> corner_phi_vals (8, 0);
          corner_phi_vals[0] = mStencil.GetValue (0, 0, 0);
          corner_phi_vals[1] = mStencil.GetValue (1, 0, 0);
          corner_phi_vals[2] = mStencil.GetValue (1, 1, 0);
          corner_phi_vals[3] = mStencil.GetValue (0, 1, 0);
          corner_phi_vals[4] = mStencil.GetValue (0, 0, 1);
          corner_phi_vals[5] = mStencil.GetValue (1, 0, 1);
          corner_phi_vals[6] = mStencil.GetValue (1, 1, 1);
          corner_phi_vals[7] = mStencil.GetValue (0, 1, 1);

          std::vector<Triangle> triaMC;
          int triSize = boundary.triCellIndices[countCell].size ();
          for (int nTri = 0; nTri < triSize; nTri++) {
            triaMC.push_back (boundary.triangles[boundary.triCellIndices[countCell][nTri]]);
          }

          // NOTE: Debugging
          // if(countCell == 1){
          //   std::cout << "triaMC.size() = "<< triaMC.size() << std::endl;
          //   std::cout << ((Eigen::Vector3d) triaMC[0].p[0]).transpose() << std::endl;
          //   std::cout << ((Eigen::Vector3d) triaMC[0].p[1]).transpose() << std::endl;
          //   std::cout << ((Eigen::Vector3d) triaMC[0].p[2]).transpose() << std::endl;
          //   // std::cout << "triaMC.size() = "<< triaMC.size() << std::endl;
          //   // std::cout << ((Eigen::Vector3d) triaMC[1].p[0]).transpose() << std::endl;
          //   // std::cout << ((Eigen::Vector3d) triaMC[1].p[1]).transpose() << std::endl;
          //   // std::cout << ((Eigen::Vector3d) triaMC[1].p[2]).transpose() << std::endl;
          //   for(int i1 = 0; i1 < 8; i1++){
          //     std::cout << corner_phi_vals[i1] << std::endl;
          //   }
          // }

          elememtsQuadrature[countCell] = ComputeQuadrature (corner_phi_vals, triaMC, i, j, k);
        }
      }  // k
    }    // j
  }      // i
}


void MomentQuadrature ::ComputeElementsAreaToVolume (bool ignoreBoundingBox) {
  int nCells = boundary.levelSet.nCells;
  int nElemX = boundary.levelSet.grid.nelx;
  int nElemY = boundary.levelSet.grid.nely;
  int nElemZ = boundary.levelSet.grid.nelz;
  // resize quadrature vector
  elementsAreaToVolume.resize (nCells);

#pragma omp parallel for
  for (int i = 0; i < nElemX; i++) {
    for (int j = 0; j < nElemY; j++) {
      for (int k = 0; k < nElemZ; k++) {
        // cell counter
        int countCell = i + nElemX * j + nElemX * nElemY * k;
        std::vector<Triangle> triaMC;
        int triSize = boundary.triCellIndices[countCell].size ();
        for (int nTri = 0; nTri < triSize; nTri++) {
          Triangle currentTriangle = boundary.triangles[boundary.triCellIndices[countCell][nTri]];
          // get mid point
          Eigen::Vector3d pMid = (Eigen::Vector3d)currentTriangle.p[0]
                                 + (Eigen::Vector3d)currentTriangle.p[1]
                                 + (Eigen::Vector3d)currentTriangle.p[2];
          pMid /= 3.0;
          double distFromBox
              = std::min ({pMid (0), 1.0 * nElemX - pMid (0), pMid (1), 1.0 * nElemY - pMid (1),
                           pMid (2), 1.0 * nElemZ - pMid (2)});
          //
          if ((distFromBox <= 1.0) && ignoreBoundingBox) {
            // this triangle on the bounding box is ignored
          } else {
            // this triangle on the bounding box is considered
            triaMC.push_back (currentTriangle);
          }
          // triaMC.push_back(currentTriangle);
        }
        elementsAreaToVolume[countCell] = ComputeAreaToVolume (triaMC, i, j, k);
        // std::vector<double> tempVector(8,0.0);
        // elementsAreaToVolume[countCell] = tempVector;
      }  // k
    }    // j
  }      // i
}


std::vector<double> MomentQuadrature ::ComputeAreaToVolume (std::vector<Triangle> triaMC,
                                                            int originX, int originY, int originZ) {
  //  area to volume for all triangles in this element
  std::vector<double> triAreaToVolume (8, 0.0);

  // For the triangles
  Eigen::Vector3d origin (1.0 * originX, 1.0 * originY, 1.0 * originZ);
  for (int i = 0; i < triaMC.size (); i++) {
    // std::cout <<" i = " << i << std::endl;
    for (int j = 0; j < 3; j++) {
      Eigen::Vector3d pTemp = (Eigen::Vector3d)triaMC[i].p[j];
      // std::cout << pTemp.transpose() << std::endl;
    }
    Eigen::Vector3d P0 = Eigen::Vector3d (triaMC[i].p[0]) - origin;
    Eigen::Vector3d P1 = Eigen::Vector3d (triaMC[i].p[1]) - origin;
    Eigen::Vector3d P2 = Eigen::Vector3d (triaMC[i].p[2]) - origin;
    Eigen::Vector3d PN = (P1 - P0).cross (P2 - P1);
    PN = PN / PN.norm ();
    MomentPolygon myTria (P0, P1, P2, PN);

    // this is the triangle area to volume for the current triangle
    std::vector<double> triArea2Vol = myTria.ComputeTriAreaToVolume ();

    // for(int k = 0; k < 8; k++) triAreaToVolume[k] += 1.0;
    for (int k = 0; k < 8; k++) triAreaToVolume[k] += triArea2Vol[k];
  }

  return triAreaToVolume;
}


MomentQuadrature ::MomentQuadrature (Boundary &boundary_) : boundary (boundary_) {}


Eigen::VectorXd MomentQuadrature ::ComputeQuadrature (std::vector<double> phi,
                                                      std::vector<Triangle> triaMC, int originX,
                                                      int originY, int originZ) {
  // helper variable for rotating the cube
  Eigen::MatrixXi RMat (6, 2);
  RMat << 0, 0, 0, 1, 0, 2, 0, 3, 1, 1, 1, 3;

  Eigen::MatrixXi IndexMat (6, 4);
  IndexMat << 4, 5, 6, 7, 0, 1, 5, 4, 3, 2, 1, 0, 7, 6, 2, 3, 5, 1, 2, 6, 0, 4, 7, 3;

  // For the polygons on the 6 surfaces of the cube
  Eigen::VectorXd elemMoments = Eigen::VectorXd::Zero (27);
  for (int i = 0; i < 6; i++) {
    MomentPolygon myPolygon (
        {phi[IndexMat (i, 0)], phi[IndexMat (i, 1)], phi[IndexMat (i, 2)], phi[IndexMat (i, 3)]});
    if (phi[IndexMat (i, 0)] <= 0.0 && phi[IndexMat (i, 1)] <= 0.0 && phi[IndexMat (i, 2)] <= 0.0
        && phi[IndexMat (i, 3)] <= 0.0) {
      continue;
    }
    myPolygon.SetPoints ();
    myPolygon.SetNormals ();
    myPolygon.Rotate (RMat (i, 0), RMat (i, 1));
    myPolygon.ComputeMoments ();
    // std::cout << "i = " << i << std::endl;
    // std::cout << myPolygon.moments << std::endl;
    elemMoments += myPolygon.moments;
  }
  // std::cout << "elemMoments = " << std::endl;
  // std::cout << elemMoments << std::endl;
  // For the triangles
  Eigen::Vector3d origin (1.0 * originX, 1.0 * originY, 1.0 * originZ);
  for (int i = 0; i < triaMC.size (); i++) {
    // std::cout <<" i = " << i << std::endl;
    for (int j = 0; j < 3; j++) {
      Eigen::Vector3d pTemp = (Eigen::Vector3d)triaMC[i].p[j];
      // std::cout << pTemp.transpose() << std::endl;
    }
    Eigen::Vector3d P0 = Eigen::Vector3d (triaMC[i].p[0]) - origin;
    Eigen::Vector3d P1 = Eigen::Vector3d (triaMC[i].p[1]) - origin;
    Eigen::Vector3d P2 = Eigen::Vector3d (triaMC[i].p[2]) - origin;
    Eigen::Vector3d PN = (P1 - P0).cross (P2 - P1);
    PN = PN / PN.norm ();
    MomentPolygon myTria (P0, P1, P2, PN);
    myTria.ComputeTriMoments ();
    elemMoments += myTria.moments;
  }

  // Inverse of A hardcoded for efficiency
  // Derived from myTria.CopmuteQuadrature(elemMoments) function
  Eigen::MatrixXd Ainv (27, 27);
  Ainv (0, 0) = 3.234113431290126;
  Ainv (0, 1) = -10.11312655989849;
  Ainv (0, 2) = 7.289799394636491;
  Ainv (0, 3) = -10.1131265598985;
  Ainv (0, 4) = 31.62391517471467;
  Ainv (0, 5) = -22.79532410983537;
  Ainv (0, 6) = 7.289799394636493;
  Ainv (0, 7) = -22.7953241098354;
  Ainv (0, 8) = 16.43145064112477;
  Ainv (0, 9) = -10.1131265598985;
  Ainv (0, 10) = 31.62391517471467;
  Ainv (0, 11) = -22.79532410983539;
  Ainv (0, 12) = 31.62391517471471;
  Ainv (0, 13) = -98.88850940945684;
  Ainv (0, 14) = 71.28135812005445;
  Ainv (0, 15) = -22.79532410983537;
  Ainv (0, 16) = 71.28135812005445;
  Ainv (0, 17) = -51.38141980076739;
  Ainv (0, 18) = 7.289799394636497;
  Ainv (0, 19) = -22.79532410983541;
  Ainv (0, 20) = 16.43145064112479;
  Ainv (0, 21) = -22.79532410983542;
  Ainv (0, 22) = 71.28135812005456;
  Ainv (0, 23) = -51.38141980076777;
  Ainv (0, 24) = 16.43145064112475;
  Ainv (0, 25) = -51.38141980076805;
  Ainv (0, 26) = 37.0370370370363;
  Ainv (1, 0) = -1.457959878927298;
  Ainv (1, 1) = 14.57959878927298;
  Ainv (1, 2) = -14.57959878927298;
  Ainv (1, 3) = 4.559064821967076;
  Ainv (1, 4) = -45.59064821967074;
  Ainv (1, 5) = 45.59064821967071;
  Ainv (1, 6) = -3.286290128224961;
  Ainv (1, 7) = 32.86290128224962;
  Ainv (1, 8) = -32.86290128224953;
  Ainv (1, 9) = 4.559064821967077;
  Ainv (1, 10) = -45.59064821967076;
  Ainv (1, 11) = 45.59064821967074;
  Ainv (1, 12) = -14.25627162401096;
  Ainv (1, 13) = 142.5627162401096;
  Ainv (1, 14) = -142.562716240109;
  Ainv (1, 15) = 10.27628396015359;
  Ainv (1, 16) = -102.7628396015357;
  Ainv (1, 17) = 102.762839601535;
  Ainv (1, 18) = -3.286290128224964;
  Ainv (1, 19) = 32.86290128224963;
  Ainv (1, 20) = -32.86290128224955;
  Ainv (1, 21) = 10.27628396015362;
  Ainv (1, 22) = -102.7628396015359;
  Ainv (1, 23) = 102.7628396015357;
  Ainv (1, 24) = -7.407407407407348;
  Ainv (1, 25) = 74.07407407407354;
  Ainv (1, 26) = -74.07407407407275;
  Ainv (2, 0) = 0.4107862660281205;
  Ainv (2, 1) = -4.466472229374483;
  Ainv (2, 2) = 7.289799394636488;
  Ainv (2, 3) = -1.284535495019201;
  Ainv (2, 4) = 13.96673304495607;
  Ainv (2, 5) = -22.79532410983535;
  Ainv (2, 6) = 0.9259259259259215;
  Ainv (2, 7) = -10.06757717241423;
  Ainv (2, 8) = 16.43145064112477;
  Ainv (2, 9) = -1.284535495019202;
  Ainv (2, 10) = 13.96673304495608;
  Ainv (2, 11) = -22.79532410983536;
  Ainv (2, 12) = 4.016763885312735;
  Ainv (2, 13) = -43.67420683065277;
  Ainv (2, 14) = 71.28135812005455;
  Ainv (2, 15) = -2.89538579054866;
  Ainv (2, 16) = 31.48148148148116;
  Ainv (2, 17) = -51.38141980076765;
  Ainv (2, 18) = 0.9259259259259232;
  Ainv (2, 19) = -10.06757717241424;
  Ainv (2, 20) = 16.43145064112478;
  Ainv (2, 21) = -2.895385790548668;
  Ainv (2, 22) = 31.4814814814813;
  Ainv (2, 23) = -51.38141980076784;
  Ainv (2, 24) = 2.087067877393647;
  Ainv (2, 25) = -22.69265427330558;
  Ainv (2, 26) = 37.03703703703648;
  Ainv (3, 0) = -1.457959878927298;
  Ainv (3, 1) = 4.559064821967078;
  Ainv (3, 2) = -3.286290128224961;
  Ainv (3, 3) = 14.57959878927298;
  Ainv (3, 4) = -45.59064821967076;
  Ainv (3, 5) = 32.8629012822496;
  Ainv (3, 6) = -14.57959878927298;
  Ainv (3, 7) = 45.59064821967078;
  Ainv (3, 8) = -32.86290128224955;
  Ainv (3, 9) = 4.559064821967077;
  Ainv (3, 10) = -14.25627162401095;
  Ainv (3, 11) = 10.27628396015357;
  Ainv (3, 12) = -45.59064821967078;
  Ainv (3, 13) = 142.5627162401096;
  Ainv (3, 14) = -102.7628396015357;
  Ainv (3, 15) = 45.59064821967072;
  Ainv (3, 16) = -142.5627162401093;
  Ainv (3, 17) = 102.7628396015352;
  Ainv (3, 18) = -3.286290128224965;
  Ainv (3, 19) = 10.27628396015362;
  Ainv (3, 20) = -7.407407407407354;
  Ainv (3, 21) = 32.86290128224964;
  Ainv (3, 22) = -102.762839601536;
  Ainv (3, 23) = 74.07407407407341;
  Ainv (3, 24) = -32.86290128224956;
  Ainv (3, 25) = 102.7628396015361;
  Ainv (3, 26) = -74.0740740740729;
  Ainv (4, 0) = 0.6572580256449924;
  Ainv (4, 1) = -6.572580256449924;
  Ainv (4, 2) = 6.572580256449919;
  Ainv (4, 3) = -6.572580256449924;
  Ainv (4, 4) = 65.72580256449925;
  Ainv (4, 5) = -65.72580256449918;
  Ainv (4, 6) = 6.572580256449925;
  Ainv (4, 7) = -65.72580256449925;
  Ainv (4, 8) = 65.72580256449912;
  Ainv (4, 9) = -2.055256792030722;
  Ainv (4, 10) = 20.55256792030722;
  Ainv (4, 11) = -20.55256792030715;
  Ainv (4, 12) = 20.55256792030722;
  Ainv (4, 13) = -205.5256792030722;
  Ainv (4, 14) = 205.5256792030715;
  Ainv (4, 15) = -20.55256792030719;
  Ainv (4, 16) = 205.5256792030719;
  Ainv (4, 17) = -205.5256792030708;
  Ainv (4, 18) = 1.481481481481478;
  Ainv (4, 19) = -14.81481481481475;
  Ainv (4, 20) = 14.8148148148147;
  Ainv (4, 21) = -14.81481481481477;
  Ainv (4, 22) = 148.1481481481475;
  Ainv (4, 23) = -148.148148148147;
  Ainv (4, 24) = 14.81481481481473;
  Ainv (4, 25) = -148.1481481481473;
  Ainv (4, 26) = 148.1481481481461;
  Ainv (5, 0) = -0.1851851851851845;
  Ainv (5, 1) = 2.013515434482847;
  Ainv (5, 2) = -3.286290128224959;
  Ainv (5, 3) = 1.851851851851845;
  Ainv (5, 4) = -20.13515434482849;
  Ainv (5, 5) = 32.86290128224959;
  Ainv (5, 6) = -1.851851851851845;
  Ainv (5, 7) = 20.13515434482848;
  Ainv (5, 8) = -32.86290128224957;
  Ainv (5, 9) = 0.5790771581097348;
  Ainv (5, 10) = -6.296296296296268;
  Ainv (5, 11) = 10.27628396015358;
  Ainv (5, 12) = -5.790771581097349;
  Ainv (5, 13) = 62.96296296296268;
  Ainv (5, 14) = -102.7628396015358;
  Ainv (5, 15) = 5.790771581097339;
  Ainv (5, 16) = -62.96296296296254;
  Ainv (5, 17) = 102.7628396015356;
  Ainv (5, 18) = -0.417413575478733;
  Ainv (5, 19) = 4.538530854661149;
  Ainv (5, 20) = -7.407407407407351;
  Ainv (5, 21) = 4.174135754787324;
  Ainv (5, 22) = -45.38530854661149;
  Ainv (5, 23) = 74.07407407407352;
  Ainv (5, 24) = -4.174135754787315;
  Ainv (5, 25) = 45.38530854661138;
  Ainv (5, 26) = -74.07407407407327;
  Ainv (6, 0) = 0.4107862660281206;
  Ainv (6, 1) = -1.284535495019203;
  Ainv (6, 2) = 0.9259259259259226;
  Ainv (6, 3) = -4.466472229374485;
  Ainv (6, 4) = 13.96673304495609;
  Ainv (6, 5) = -10.06757717241424;
  Ainv (6, 6) = 7.289799394636493;
  Ainv (6, 7) = -22.79532410983539;
  Ainv (6, 8) = 16.43145064112479;
  Ainv (6, 9) = -1.284535495019202;
  Ainv (6, 10) = 4.016763885312743;
  Ainv (6, 11) = -2.895385790548662;
  Ainv (6, 12) = 13.96673304495609;
  Ainv (6, 13) = -43.67420683065283;
  Ainv (6, 14) = 31.48148148148121;
  Ainv (6, 15) = -22.79532410983536;
  Ainv (6, 16) = 71.28135812005476;
  Ainv (6, 17) = -51.38141980076777;
  Ainv (6, 18) = 0.9259259259259238;
  Ainv (6, 19) = -2.895385790548677;
  Ainv (6, 20) = 2.087067877393651;
  Ainv (6, 21) = -10.06757717241425;
  Ainv (6, 22) = 31.48148148148138;
  Ainv (6, 23) = -22.69265427330564;
  Ainv (6, 24) = 16.4314506411248;
  Ainv (6, 25) = -51.38141980076803;
  Ainv (6, 26) = 37.03703703703656;
  Ainv (7, 0) = -0.1851851851851846;
  Ainv (7, 1) = 1.851851851851846;
  Ainv (7, 2) = -1.851851851851845;
  Ainv (7, 3) = 2.013515434482849;
  Ainv (7, 4) = -20.1351543448285;
  Ainv (7, 5) = 20.13515434482847;
  Ainv (7, 6) = -3.286290128224963;
  Ainv (7, 7) = 32.86290128224964;
  Ainv (7, 8) = -32.8629012822496;
  Ainv (7, 9) = 0.5790771581097356;
  Ainv (7, 10) = -5.790771581097358;
  Ainv (7, 11) = 5.790771581097334;
  Ainv (7, 12) = -6.296296296296274;
  Ainv (7, 13) = 62.96296296296276;
  Ainv (7, 14) = -62.96296296296253;
  Ainv (7, 15) = 10.27628396015361;
  Ainv (7, 16) = -102.7628396015361;
  Ainv (7, 17) = 102.7628396015357;
  Ainv (7, 18) = -0.4174135754787335;
  Ainv (7, 19) = 4.174135754787331;
  Ainv (7, 20) = -4.174135754787311;
  Ainv (7, 21) = 4.538530854661158;
  Ainv (7, 22) = -45.38530854661159;
  Ainv (7, 23) = 45.38530854661136;
  Ainv (7, 24) = -7.407407407407382;
  Ainv (7, 25) = 74.07407407407379;
  Ainv (7, 26) = -74.07407407407335;
  Ainv (8, 0) = 0.05217669693484157;
  Ainv (8, 1) = -0.5673163568326441;
  Ainv (8, 2) = 0.9259259259259227;
  Ainv (8, 3) = -0.5673163568326443;
  Ainv (8, 4) = 6.168421299872411;
  Ainv (8, 5) = -10.06757717241424;
  Ainv (8, 6) = 0.9259259259259235;
  Ainv (8, 7) = -10.06757717241425;
  Ainv (8, 8) = 16.4314506411248;
  Ainv (8, 9) = -0.1631574002551381;
  Ainv (8, 10) = 1.774007695784615;
  Ainv (8, 11) = -2.895385790548672;
  Ainv (8, 12) = 1.774007695784615;
  Ainv (8, 13) = -19.28875613230991;
  Ainv (8, 14) = 31.48148148148132;
  Ainv (8, 15) = -2.895385790548679;
  Ainv (8, 16) = 31.48148148148137;
  Ainv (8, 17) = -51.38141980076797;
  Ainv (8, 18) = 0.1176080127709103;
  Ainv (8, 19) = -1.278749964238654;
  Ainv (8, 20) = 2.087067877393661;
  Ainv (8, 21) = -1.278749964238655;
  Ainv (8, 22) = 13.90382706513021;
  Ainv (8, 23) = -22.69265427330573;
  Ainv (8, 24) = 2.087067877393668;
  Ainv (8, 25) = -22.69265427330579;
  Ainv (8, 26) = 37.03703703703682;
  Ainv (9, 0) = -1.457959878927299;
  Ainv (9, 1) = 4.559064821967078;
  Ainv (9, 2) = -3.286290128224963;
  Ainv (9, 3) = 4.55906482196708;
  Ainv (9, 4) = -14.25627162401098;
  Ainv (9, 5) = 10.27628396015361;
  Ainv (9, 6) = -3.286290128224966;
  Ainv (9, 7) = 10.27628396015364;
  Ainv (9, 8) = -7.407407407407354;
  Ainv (9, 9) = 14.57959878927299;
  Ainv (9, 10) = -45.59064821967078;
  Ainv (9, 11) = 32.86290128224963;
  Ainv (9, 12) = -45.59064821967081;
  Ainv (9, 13) = 142.5627162401098;
  Ainv (9, 14) = -102.7628396015359;
  Ainv (9, 15) = 32.86290128224962;
  Ainv (9, 16) = -102.762839601536;
  Ainv (9, 17) = 74.07407407407315;
  Ainv (9, 18) = -14.57959878927299;
  Ainv (9, 19) = 45.5906482196708;
  Ainv (9, 20) = -32.86290128224963;
  Ainv (9, 21) = 45.59064821967081;
  Ainv (9, 22) = -142.5627162401094;
  Ainv (9, 23) = 102.7628396015359;
  Ainv (9, 24) = -32.8629012822496;
  Ainv (9, 25) = 102.7628396015361;
  Ainv (9, 26) = -74.07407407407305;
  Ainv (10, 0) = 0.6572580256449929;
  Ainv (10, 1) = -6.572580256449927;
  Ainv (10, 2) = 6.572580256449924;
  Ainv (10, 3) = -2.055256792030725;
  Ainv (10, 4) = 20.55256792030725;
  Ainv (10, 5) = -20.55256792030721;
  Ainv (10, 6) = 1.481481481481478;
  Ainv (10, 7) = -14.81481481481478;
  Ainv (10, 8) = 14.8148148148147;
  Ainv (10, 9) = -6.572580256449929;
  Ainv (10, 10) = 65.72580256449929;
  Ainv (10, 11) = -65.72580256449925;
  Ainv (10, 12) = 20.55256792030725;
  Ainv (10, 13) = -205.5256792030725;
  Ainv (10, 14) = 205.5256792030719;
  Ainv (10, 15) = -14.81481481481476;
  Ainv (10, 16) = 148.1481481481474;
  Ainv (10, 17) = -148.1481481481466;
  Ainv (10, 18) = 6.572580256449933;
  Ainv (10, 19) = -65.7258025644993;
  Ainv (10, 20) = 65.72580256449922;
  Ainv (10, 21) = -20.55256792030725;
  Ainv (10, 22) = 205.5256792030723;
  Ainv (10, 23) = -205.5256792030719;
  Ainv (10, 24) = 14.81481481481474;
  Ainv (10, 25) = -148.1481481481474;
  Ainv (10, 26) = 148.1481481481464;
  Ainv (11, 0) = -0.1851851851851847;
  Ainv (11, 1) = 2.01351543448285;
  Ainv (11, 2) = -3.286290128224961;
  Ainv (11, 3) = 0.5790771581097357;
  Ainv (11, 4) = -6.29629629629628;
  Ainv (11, 5) = 10.2762839601536;
  Ainv (11, 6) = -0.4174135754787326;
  Ainv (11, 7) = 4.538530854661151;
  Ainv (11, 8) = -7.407407407407357;
  Ainv (11, 9) = 1.851851851851847;
  Ainv (11, 10) = -20.13515434482851;
  Ainv (11, 11) = 32.86290128224962;
  Ainv (11, 12) = -5.790771581097356;
  Ainv (11, 13) = 62.96296296296276;
  Ainv (11, 14) = -102.762839601536;
  Ainv (11, 15) = 4.174135754787317;
  Ainv (11, 16) = -45.38530854661141;
  Ainv (11, 17) = 74.07407407407342;
  Ainv (11, 18) = -1.851851851851849;
  Ainv (11, 19) = 20.13515434482851;
  Ainv (11, 20) = -32.86290128224961;
  Ainv (11, 21) = 5.790771581097357;
  Ainv (11, 22) = -62.96296296296276;
  Ainv (11, 23) = 102.762839601536;
  Ainv (11, 24) = -4.174135754787316;
  Ainv (11, 25) = 45.3853085466114;
  Ainv (11, 26) = -74.07407407407335;
  Ainv (12, 0) = 0.6572580256449931;
  Ainv (12, 1) = -2.055256792030726;
  Ainv (12, 2) = 1.481481481481477;
  Ainv (12, 3) = -6.57258025644993;
  Ainv (12, 4) = 20.55256792030727;
  Ainv (12, 5) = -14.81481481481476;
  Ainv (12, 6) = 6.572580256449932;
  Ainv (12, 7) = -20.55256792030726;
  Ainv (12, 8) = 14.81481481481472;
  Ainv (12, 9) = -6.57258025644993;
  Ainv (12, 10) = 20.55256792030725;
  Ainv (12, 11) = -14.81481481481474;
  Ainv (12, 12) = 65.72580256449932;
  Ainv (12, 13) = -205.5256792030726;
  Ainv (12, 14) = 148.1481481481474;
  Ainv (12, 15) = -65.72580256449925;
  Ainv (12, 16) = 205.5256792030723;
  Ainv (12, 17) = -148.1481481481468;
  Ainv (12, 18) = 6.572580256449934;
  Ainv (12, 19) = -20.55256792030726;
  Ainv (12, 20) = 14.81481481481475;
  Ainv (12, 21) = -65.72580256449932;
  Ainv (12, 22) = 205.5256792030724;
  Ainv (12, 23) = -148.1481481481473;
  Ainv (12, 24) = 65.72580256449923;
  Ainv (12, 25) = -205.5256792030723;
  Ainv (12, 26) = 148.1481481481465;
  Ainv (13, 0) = -0.2962962962962959;
  Ainv (13, 1) = 2.962962962962957;
  Ainv (13, 2) = -2.962962962962953;
  Ainv (13, 3) = 2.962962962962957;
  Ainv (13, 4) = -29.6296296296296;
  Ainv (13, 5) = 29.62962962962952;
  Ainv (13, 6) = -2.962962962962958;
  Ainv (13, 7) = 29.62962962962958;
  Ainv (13, 8) = -29.62962962962947;
  Ainv (13, 9) = 2.962962962962958;
  Ainv (13, 10) = -29.62962962962958;
  Ainv (13, 11) = 29.6296296296295;
  Ainv (13, 12) = -29.62962962962958;
  Ainv (13, 13) = 296.2962962962958;
  Ainv (13, 14) = -296.296296296295;
  Ainv (13, 15) = 29.62962962962955;
  Ainv (13, 16) = -296.2962962962954;
  Ainv (13, 17) = 296.2962962962941;
  Ainv (13, 18) = -2.962962962962962;
  Ainv (13, 19) = 29.62962962962958;
  Ainv (13, 20) = -29.62962962962951;
  Ainv (13, 21) = 29.6296296296296;
  Ainv (13, 22) = -296.2962962962957;
  Ainv (13, 23) = 296.2962962962949;
  Ainv (13, 24) = -29.62962962962953;
  Ainv (13, 25) = 296.2962962962953;
  Ainv (13, 26) = -296.2962962962936;
  Ainv (14, 0) = 0.0834827150957467;
  Ainv (14, 1) = -0.9077061709322319;
  Ainv (14, 2) = 1.481481481481477;
  Ainv (14, 3) = -0.8348271509574668;
  Ainv (14, 4) = 9.077061709322331;
  Ainv (14, 5) = -14.81481481481477;
  Ainv (14, 6) = 0.8348271509574673;
  Ainv (14, 7) = -9.077061709322326;
  Ainv (14, 8) = 14.81481481481475;
  Ainv (14, 9) = -0.834827150957467;
  Ainv (14, 10) = 9.077061709322326;
  Ainv (14, 11) = -14.81481481481476;
  Ainv (14, 12) = 8.348271509574671;
  Ainv (14, 13) = -90.77061709322325;
  Ainv (14, 14) = 148.1481481481476;
  Ainv (14, 15) = -8.34827150957466;
  Ainv (14, 16) = 90.77061709322311;
  Ainv (14, 17) = -148.1481481481473;
  Ainv (14, 18) = 0.8348271509574686;
  Ainv (14, 19) = -9.077061709322331;
  Ainv (14, 20) = 14.81481481481476;
  Ainv (14, 21) = -8.348271509574678;
  Ainv (14, 22) = 90.77061709322325;
  Ainv (14, 23) = -148.1481481481476;
  Ainv (14, 24) = 8.348271509574662;
  Ainv (14, 25) = -90.77061709322311;
  Ainv (14, 26) = 148.1481481481472;
  Ainv (15, 0) = -0.1851851851851849;
  Ainv (15, 1) = 0.579077158109737;
  Ainv (15, 2) = -0.417413575478733;
  Ainv (15, 3) = 2.013515434482851;
  Ainv (15, 4) = -6.296296296296291;
  Ainv (15, 5) = 4.538530854661157;
  Ainv (15, 6) = -3.286290128224965;
  Ainv (15, 7) = 10.27628396015363;
  Ainv (15, 8) = -7.407407407407378;
  Ainv (15, 9) = 1.851851851851848;
  Ainv (15, 10) = -5.790771581097367;
  Ainv (15, 11) = 4.174135754787321;
  Ainv (15, 12) = -20.13515434482852;
  Ainv (15, 13) = 62.96296296296286;
  Ainv (15, 14) = -45.38530854661148;
  Ainv (15, 15) = 32.86290128224963;
  Ainv (15, 16) = -102.7628396015363;
  Ainv (15, 17) = 74.07407407407362;
  Ainv (15, 18) = -1.85185185185185;
  Ainv (15, 19) = 5.79077158109737;
  Ainv (15, 20) = -4.174135754787322;
  Ainv (15, 21) = 20.13515434482852;
  Ainv (15, 22) = -62.96296296296288;
  Ainv (15, 23) = 45.38530854661148;
  Ainv (15, 24) = -32.86290128224963;
  Ainv (15, 25) = 102.7628396015362;
  Ainv (15, 26) = -74.07407407407345;
  Ainv (16, 0) = 0.08348271509574683;
  Ainv (16, 1) = -0.8348271509574682;
  Ainv (16, 2) = 0.8348271509574668;
  Ainv (16, 3) = -0.907706170932233;
  Ainv (16, 4) = 9.077061709322342;
  Ainv (16, 5) = -9.077061709322322;
  Ainv (16, 6) = 1.481481481481481;
  Ainv (16, 7) = -14.81481481481481;
  Ainv (16, 8) = 14.81481481481478;
  Ainv (16, 9) = -0.8348271509574682;
  Ainv (16, 10) = 8.348271509574682;
  Ainv (16, 11) = -8.348271509574657;
  Ainv (16, 12) = 9.077061709322335;
  Ainv (16, 13) = -90.77061709322336;
  Ainv (16, 14) = 90.77061709322309;
  Ainv (16, 15) = -14.81481481481479;
  Ainv (16, 16) = 148.1481481481479;
  Ainv (16, 17) = -148.1481481481475;
  Ainv (16, 18) = 0.8348271509574696;
  Ainv (16, 19) = -8.348271509574689;
  Ainv (16, 20) = 8.34827150957466;
  Ainv (16, 21) = -9.077061709322344;
  Ainv (16, 22) = 90.77061709322339;
  Ainv (16, 23) = -90.77061709322309;
  Ainv (16, 24) = 14.81481481481479;
  Ainv (16, 25) = -148.1481481481479;
  Ainv (16, 26) = 148.1481481481473;
  Ainv (17, 0) = -0.02352160255418206;
  Ainv (17, 1) = 0.2557499928477313;
  Ainv (17, 2) = -0.4174135754787339;
  Ainv (17, 3) = 0.2557499928477313;
  Ainv (17, 4) = -2.780765413026052;
  Ainv (17, 5) = 4.538530854661167;
  Ainv (17, 6) = -0.4174135754787347;
  Ainv (17, 7) = 4.538530854661176;
  Ainv (17, 8) = -7.407407407407398;
  Ainv (17, 9) = 0.2352160255418206;
  Ainv (17, 10) = -2.557499928477314;
  Ainv (17, 11) = 4.174135754787335;
  Ainv (17, 12) = -2.557499928477314;
  Ainv (17, 13) = 27.80765413026049;
  Ainv (17, 14) = -45.38530854661162;
  Ainv (17, 15) = 4.174135754787343;
  Ainv (17, 16) = -45.38530854661169;
  Ainv (17, 17) = 74.07407407407386;
  Ainv (17, 18) = -0.2352160255418214;
  Ainv (17, 19) = 2.557499928477319;
  Ainv (17, 20) = -4.174135754787339;
  Ainv (17, 21) = 2.557499928477319;
  Ainv (17, 22) = -27.80765413026052;
  Ainv (17, 23) = 45.38530854661163;
  Ainv (17, 24) = -4.174135754787346;
  Ainv (17, 25) = 45.3853085466117;
  Ainv (17, 26) = -74.07407407407383;
  Ainv (18, 0) = 0.4107862660281209;
  Ainv (18, 1) = -1.284535495019204;
  Ainv (18, 2) = 0.9259259259259248;
  Ainv (18, 3) = -1.284535495019204;
  Ainv (18, 4) = 4.01676388531276;
  Ainv (18, 5) = -2.895385790548685;
  Ainv (18, 6) = 0.9259259259259249;
  Ainv (18, 7) = -2.895385790548687;
  Ainv (18, 8) = 2.087067877393661;
  Ainv (18, 9) = -4.466472229374488;
  Ainv (18, 10) = 13.96673304495611;
  Ainv (18, 11) = -10.06757717241426;
  Ainv (18, 12) = 13.96673304495611;
  Ainv (18, 13) = -43.67420683065296;
  Ainv (18, 14) = 31.48148148148139;
  Ainv (18, 15) = -10.06757717241426;
  Ainv (18, 16) = 31.48148148148142;
  Ainv (18, 17) = -22.69265427330566;
  Ainv (18, 18) = 7.289799394636496;
  Ainv (18, 19) = -22.79532410983541;
  Ainv (18, 20) = 16.43145064112483;
  Ainv (18, 21) = -22.79532410983541;
  Ainv (18, 22) = 71.28135812005482;
  Ainv (18, 23) = -51.38141980076802;
  Ainv (18, 24) = 16.43145064112482;
  Ainv (18, 25) = -51.38141980076812;
  Ainv (18, 26) = 37.03703703703676;
  Ainv (19, 0) = -0.185185185185185;
  Ainv (19, 1) = 1.851851851851849;
  Ainv (19, 2) = -1.851851851851849;
  Ainv (19, 3) = 0.5790771581097371;
  Ainv (19, 4) = -5.790771581097378;
  Ainv (19, 5) = 5.790771581097365;
  Ainv (19, 6) = -0.4174135754787344;
  Ainv (19, 7) = 4.174135754787342;
  Ainv (19, 8) = -4.174135754787321;
  Ainv (19, 9) = 2.013515434482852;
  Ainv (19, 10) = -20.13515434482853;
  Ainv (19, 11) = 20.13515434482852;
  Ainv (19, 12) = -6.296296296296295;
  Ainv (19, 13) = 62.96296296296291;
  Ainv (19, 14) = -62.96296296296281;
  Ainv (19, 15) = 4.538530854661165;
  Ainv (19, 16) = -45.3853085466116;
  Ainv (19, 17) = 45.38530854661137;
  Ainv (19, 18) = -3.286290128224968;
  Ainv (19, 19) = 32.86290128224967;
  Ainv (19, 20) = -32.86290128224965;
  Ainv (19, 21) = 10.27628396015364;
  Ainv (19, 22) = -102.7628396015363;
  Ainv (19, 23) = 102.7628396015361;
  Ainv (19, 24) = -7.407407407407391;
  Ainv (19, 25) = 74.07407407407386;
  Ainv (19, 26) = -74.07407407407359;
  Ainv (20, 0) = 0.0521766969348417;
  Ainv (20, 1) = -0.5673163568326454;
  Ainv (20, 2) = 0.9259259259259242;
  Ainv (20, 3) = -0.1631574002551385;
  Ainv (20, 4) = 1.774007695784621;
  Ainv (20, 5) = -2.895385790548682;
  Ainv (20, 6) = 0.1176080127709101;
  Ainv (20, 7) = -1.278749964238654;
  Ainv (20, 8) = 2.087067877393662;
  Ainv (20, 9) = -0.5673163568326455;
  Ainv (20, 10) = 6.168421299872421;
  Ainv (20, 11) = -10.06757717241426;
  Ainv (20, 12) = 1.77400769578462;
  Ainv (20, 13) = -19.28875613230996;
  Ainv (20, 14) = 31.48148148148139;
  Ainv (20, 15) = -1.278749964238653;
  Ainv (20, 16) = 13.90382706513019;
  Ainv (20, 17) = -22.6926542733057;
  Ainv (20, 18) = 0.9259259259259253;
  Ainv (20, 19) = -10.06757717241427;
  Ainv (20, 20) = 16.43145064112483;
  Ainv (20, 21) = -2.895385790548686;
  Ainv (20, 22) = 31.48148148148144;
  Ainv (20, 23) = -51.38141980076809;
  Ainv (20, 24) = 2.087067877393666;
  Ainv (20, 25) = -22.69265427330577;
  Ainv (20, 26) = 37.03703703703682;
  Ainv (21, 0) = -0.185185185185185;
  Ainv (21, 1) = 0.5790771581097375;
  Ainv (21, 2) = -0.4174135754787341;
  Ainv (21, 3) = 1.851851851851849;
  Ainv (21, 4) = -5.790771581097383;
  Ainv (21, 5) = 4.174135754787339;
  Ainv (21, 6) = -1.85185185185185;
  Ainv (21, 7) = 5.790771581097377;
  Ainv (21, 8) = -4.174135754787327;
  Ainv (21, 9) = 2.013515434482853;
  Ainv (21, 10) = -6.296296296296294;
  Ainv (21, 11) = 4.538530854661164;
  Ainv (21, 12) = -20.13515434482853;
  Ainv (21, 13) = 62.96296296296295;
  Ainv (21, 14) = -45.38530854661161;
  Ainv (21, 15) = 20.13515434482852;
  Ainv (21, 16) = -62.96296296296288;
  Ainv (21, 17) = 45.38530854661141;
  Ainv (21, 18) = -3.286290128224969;
  Ainv (21, 19) = 10.27628396015364;
  Ainv (21, 20) = -7.407407407407399;
  Ainv (21, 21) = 32.86290128224968;
  Ainv (21, 22) = -102.7628396015363;
  Ainv (21, 23) = 74.07407407407383;
  Ainv (21, 24) = -32.86290128224964;
  Ainv (21, 25) = 102.7628396015362;
  Ainv (21, 26) = -74.07407407407356;
  Ainv (22, 0) = 0.08348271509574695;
  Ainv (22, 1) = -0.8348271509574688;
  Ainv (22, 2) = 0.8348271509574683;
  Ainv (22, 3) = -0.8348271509574687;
  Ainv (22, 4) = 8.348271509574698;
  Ainv (22, 5) = -8.34827150957468;
  Ainv (22, 6) = 0.8348271509574694;
  Ainv (22, 7) = -8.348271509574692;
  Ainv (22, 8) = 8.34827150957466;
  Ainv (22, 9) = -0.9077061709322345;
  Ainv (22, 10) = 9.077061709322345;
  Ainv (22, 11) = -9.077061709322331;
  Ainv (22, 12) = 9.077061709322347;
  Ainv (22, 13) = -90.77061709322345;
  Ainv (22, 14) = 90.77061709322327;
  Ainv (22, 15) = -9.07706170932234;
  Ainv (22, 16) = 90.77061709322331;
  Ainv (22, 17) = -90.77061709322294;
  Ainv (22, 18) = 1.481481481481483;
  Ainv (22, 19) = -14.81481481481482;
  Ainv (22, 20) = 14.8148148148148;
  Ainv (22, 21) = -14.81481481481482;
  Ainv (22, 22) = 148.148148148148;
  Ainv (22, 23) = -148.1481481481478;
  Ainv (22, 24) = 14.8148148148148;
  Ainv (22, 25) = -148.1481481481479;
  Ainv (22, 26) = 148.1481481481474;
  Ainv (23, 0) = -0.02352160255418208;
  Ainv (23, 1) = 0.2557499928477314;
  Ainv (23, 2) = -0.4174135754787342;
  Ainv (23, 3) = 0.2352160255418206;
  Ainv (23, 4) = -2.557499928477317;
  Ainv (23, 5) = 4.174135754787341;
  Ainv (23, 6) = -0.2352160255418209;
  Ainv (23, 7) = 2.557499928477315;
  Ainv (23, 8) = -4.174135754787335;
  Ainv (23, 9) = 0.2557499928477315;
  Ainv (23, 10) = -2.780765413026051;
  Ainv (23, 11) = 4.538530854661167;
  Ainv (23, 12) = -2.557499928477315;
  Ainv (23, 13) = 27.8076541302605;
  Ainv (23, 14) = -45.38530854661164;
  Ainv (23, 15) = 2.557499928477313;
  Ainv (23, 16) = -27.80765413026045;
  Ainv (23, 17) = 45.38530854661153;
  Ainv (23, 18) = -0.4174135754787352;
  Ainv (23, 19) = 4.538530854661177;
  Ainv (23, 20) = -7.407407407407401;
  Ainv (23, 21) = 4.174135754787348;
  Ainv (23, 22) = -45.38530854661171;
  Ainv (23, 23) = 74.07407407407395;
  Ainv (23, 24) = -4.174135754787341;
  Ainv (23, 25) = 45.38530854661164;
  Ainv (23, 26) = -74.07407407407378;
  Ainv (24, 0) = 0.05217669693484174;
  Ainv (24, 1) = -0.1631574002551389;
  Ainv (24, 2) = 0.1176080127709103;
  Ainv (24, 3) = -0.5673163568326456;
  Ainv (24, 4) = 1.774007695784624;
  Ainv (24, 5) = -1.278749964238656;
  Ainv (24, 6) = 0.9259259259259249;
  Ainv (24, 7) = -2.89538579054869;
  Ainv (24, 8) = 2.087067877393668;
  Ainv (24, 9) = -0.5673163568326458;
  Ainv (24, 10) = 1.774007695784622;
  Ainv (24, 11) = -1.278749964238654;
  Ainv (24, 12) = 6.168421299872424;
  Ainv (24, 13) = -19.28875613230999;
  Ainv (24, 14) = 13.90382706513021;
  Ainv (24, 15) = -10.06757717241426;
  Ainv (24, 16) = 31.48148148148146;
  Ainv (24, 17) = -22.69265427330575;
  Ainv (24, 18) = 0.9259259259259256;
  Ainv (24, 19) = -2.89538579054869;
  Ainv (24, 20) = 2.08706787739367;
  Ainv (24, 21) = -10.06757717241427;
  Ainv (24, 22) = 31.48148148148148;
  Ainv (24, 23) = -22.6926542733058;
  Ainv (24, 24) = 16.43145064112483;
  Ainv (24, 25) = -51.38141980076814;
  Ainv (24, 26) = 37.03703703703685;
  Ainv (25, 0) = -0.02352160255418212;
  Ainv (25, 1) = 0.2352160255418211;
  Ainv (25, 2) = -0.2352160255418209;
  Ainv (25, 3) = 0.2557499928477316;
  Ainv (25, 4) = -2.55749992847732;
  Ainv (25, 5) = 2.557499928477315;
  Ainv (25, 6) = -0.4174135754787351;
  Ainv (25, 7) = 4.174135754787351;
  Ainv (25, 8) = -4.174135754787341;
  Ainv (25, 9) = 0.2557499928477318;
  Ainv (25, 10) = -2.557499928477318;
  Ainv (25, 11) = 2.557499928477313;
  Ainv (25, 12) = -2.780765413026054;
  Ainv (25, 13) = 27.80765413026053;
  Ainv (25, 14) = -27.80765413026046;
  Ainv (25, 15) = 4.538530854661175;
  Ainv (25, 16) = -45.38530854661172;
  Ainv (25, 17) = 45.38530854661157;
  Ainv (25, 18) = -0.4174135754787355;
  Ainv (25, 19) = 4.174135754787352;
  Ainv (25, 20) = -4.174135754787344;
  Ainv (25, 21) = 4.53853085466118;
  Ainv (25, 22) = -45.38530854661175;
  Ainv (25, 23) = 45.38530854661165;
  Ainv (25, 24) = -7.407407407407406;
  Ainv (25, 25) = 74.07407407407402;
  Ainv (25, 26) = -74.07407407407381;
  Ainv (26, 0) = 0.006627309450613375;
  Ainv (26, 1) = -0.07205862528668215;
  Ainv (26, 2) = 0.1176080127709106;
  Ainv (26, 3) = -0.07205862528668212;
  Ainv (26, 4) = 0.7834922326926963;
  Ainv (26, 5) = -1.278749964238659;
  Ainv (26, 6) = 0.1176080127709108;
  Ainv (26, 7) = -1.278749964238661;
  Ainv (26, 8) = 2.087067877393673;
  Ainv (26, 9) = -0.07205862528668218;
  Ainv (26, 10) = 0.7834922326926957;
  Ainv (26, 11) = -1.278749964238659;
  Ainv (26, 12) = 0.7834922326926956;
  Ainv (26, 13) = -8.518897997950537;
  Ainv (26, 14) = 13.90382706513025;
  Ainv (26, 15) = -1.27874996423866;
  Ainv (26, 16) = 13.90382706513026;
  Ainv (26, 17) = -22.69265427330582;
  Ainv (26, 18) = 0.1176080127709109;
  Ainv (26, 19) = -1.278749964238662;
  Ainv (26, 20) = 2.087067877393675;
  Ainv (26, 21) = -1.278749964238662;
  Ainv (26, 22) = 13.90382706513028;
  Ainv (26, 23) = -22.69265427330586;
  Ainv (26, 24) = 2.087067877393675;
  Ainv (26, 25) = -22.69265427330587;
  Ainv (26, 26) = 37.03703703703697;

  // compute quadrature
  Eigen::VectorXd elemQuadrature = Ainv * elemMoments;

  return elemQuadrature;
}

}  // namespace para_lsm

#endif