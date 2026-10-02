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

#include "boundary.h"

namespace para_lsm {

// constructor
Boundary ::Boundary (LevelSet3D &levelSet_) : levelSet (levelSet_) {}


// Linear Interpolation function
GridVector Boundary ::LinearInterp (Grid4Vector p4vec1, Grid4Vector p4vec2, double value) {
  GridVector p1 = GridVector (p4vec1);
  GridVector p2 = GridVector (p4vec2);
  Eigen::Vector3d p;

  // if(std::abs(p4vec1.val - p4vec2.val) > 1.0e-6)
  if (p4vec1.val != p4vec2.val) {
    p = (Eigen::Vector3d)p1
        + ((Eigen::Vector3d)p2 - (Eigen::Vector3d)p1) / (p4vec2.val - p4vec1.val)
              * (value - p4vec1.val);
  } else
    p = (Eigen::Vector3d)p1;

  return GridVector (p);
}


void Boundary ::MarchCubes (double isoValue_) {
  ncellsX = levelSet.grid.nelx;
  ncellsY = levelSet.grid.nely;
  ncellsZ = levelSet.grid.nelz;
  isoValue = isoValue_;
  // allocate space for traingles
  triangles.resize (3 * ncellsX * ncellsY * ncellsZ);
  // resize tria list
  triCellIndices.resize (ncellsX * ncellsY * ncellsZ);
  numTriangles = 0;
  int ni, nj;

  // go through all the grid.phi
  for (int i = 0; i < ncellsX; i++) {  // x axis
    ni = i * (ncellsY + 1) * (ncellsZ + 1);
    for (int j = 0; j < ncellsY; j++) {  // y axis
      nj = j * (ncellsZ + 1);
      for (int k = 0; k < ncellsZ; k++) {  // z axiss
        // initialize vertices
        std::vector<Grid4Vector> verts = GetVertices (ni, nj, k, levelSet.grid);
        // get the index
        int cubeIndex = 0;
        for (int n = 0; n < 8; n++) {
          if (verts[n].val <= isoValue) cubeIndex |= (1 << n);
        }
        int ind = i + ncellsX * j + ncellsX * ncellsY * k;
        triCellIndices[ind].resize (0);

        // check if its completely inside or outside
        if (!edgeTable[cubeIndex]) continue;
        std::vector<GridVector> intVerts = GetInterpolatedVertices (cubeIndex, verts);
        // now build the triangles using triTable
        for (int n = 0; triTable[cubeIndex][n] != -1; n += 3) {
          triangles[numTriangles].p[0] = intVerts[triTable[cubeIndex][n + 2]];
          triangles[numTriangles].p[1] = intVerts[triTable[cubeIndex][n + 1]];
          triangles[numTriangles].p[2] = intVerts[triTable[cubeIndex][n]];
          triCellIndices[ind].push_back (numTriangles);
          numTriangles++;
        }
      }  // z
    }    // y
  }      // x
  triangles.resize (numTriangles);
  ComputeMidPoints ();
  perturb_indicies.resize (numTriangles);
  perturb_sensitivities.resize (numTriangles);
}


std::vector<Grid4Vector> Boundary::GetVertices (int ni, int nj, int k, Grid &grid) {
  std::vector<Grid4Vector> verts (8);
  int ind = ni + nj + k;
  verts[0] = levelSet.grid.phi[ind];
  verts[1] = levelSet.grid.phi[ind + (ncellsY + 1) * (ncellsZ + 1)];
  verts[2] = levelSet.grid.phi[ind + (ncellsY + 1) * (ncellsZ + 1) + 1];
  verts[3] = levelSet.grid.phi[ind + 1];
  verts[4] = levelSet.grid.phi[ind + (ncellsZ + 1)];
  verts[5] = levelSet.grid.phi[ind + (ncellsY + 1) * (ncellsZ + 1) + (ncellsZ + 1)];
  verts[6] = levelSet.grid.phi[ind + (ncellsY + 1) * (ncellsZ + 1) + (ncellsZ + 1) + 1];
  verts[7] = levelSet.grid.phi[ind + (ncellsZ + 1) + 1];
  return verts;
}


std::vector<GridVector> Boundary::GetInterpolatedVertices (int cubeIndex,
                                                           std::vector<Grid4Vector> verts) {
  // get linearly interpolated vertices on edges and save into the array
  std::vector<GridVector> intVerts (12);
  if (edgeTable[cubeIndex] & 1) intVerts[0] = LinearInterp (verts[0], verts[1], isoValue);
  if (edgeTable[cubeIndex] & 2) intVerts[1] = LinearInterp (verts[1], verts[2], isoValue);
  if (edgeTable[cubeIndex] & 4) intVerts[2] = LinearInterp (verts[2], verts[3], isoValue);
  if (edgeTable[cubeIndex] & 8) intVerts[3] = LinearInterp (verts[3], verts[0], isoValue);
  if (edgeTable[cubeIndex] & 16) intVerts[4] = LinearInterp (verts[4], verts[5], isoValue);
  if (edgeTable[cubeIndex] & 32) intVerts[5] = LinearInterp (verts[5], verts[6], isoValue);
  if (edgeTable[cubeIndex] & 64) intVerts[6] = LinearInterp (verts[6], verts[7], isoValue);
  if (edgeTable[cubeIndex] & 128) intVerts[7] = LinearInterp (verts[7], verts[4], isoValue);
  if (edgeTable[cubeIndex] & 256) intVerts[8] = LinearInterp (verts[0], verts[4], isoValue);
  if (edgeTable[cubeIndex] & 512) intVerts[9] = LinearInterp (verts[1], verts[5], isoValue);
  if (edgeTable[cubeIndex] & 1024) intVerts[10] = LinearInterp (verts[2], verts[6], isoValue);
  if (edgeTable[cubeIndex] & 2048) intVerts[11] = LinearInterp (verts[3], verts[7], isoValue);

  return intVerts;
}


void Boundary ::ComputeMidPoints () {
  // Compute triangle midpoints
  // Resize boundary_pts
  bPoints.resize (numTriangles);
  bAreas.resize (numTriangles);
  for (int i = 0; i < numTriangles; i++) {
    // save triangle midpoints
    Eigen::Vector3d p0 = triangles[i].p[0];
    Eigen::Vector3d p1 = triangles[i].p[1];
    Eigen::Vector3d p2 = triangles[i].p[2];
    Eigen::Vector3d pMid = (p0 + p1 + p2) / 3.0;

    bPoints[i] = {pMid[0], pMid[1], pMid[2]};
    bAreas[i] = 0.5 * ((p1 - p0).cross (p2 - p0)).norm ();
  }
}


void Boundary ::WriteSTL (std::string file_name) {
  // Writing stl file
  std::ofstream txtfile;
  txtfile.open (file_name);
  txtfile << "solid mysolid" << std::endl;
  for (int i = 0; i < numTriangles; i++) {
    txtfile << "facet normal 0 0 0" << std::endl;
    txtfile << "  outer loop" << std::endl;
    txtfile << "    vertex " << triangles[i].p[0].x << " " << triangles[i].p[0].y << " "
            << triangles[i].p[0].z << " " << std::endl;
    txtfile << "    vertex " << triangles[i].p[1].x << " " << triangles[i].p[1].y << " "
            << triangles[i].p[1].z << " " << std::endl;
    txtfile << "    vertex " << triangles[i].p[2].x << " " << triangles[i].p[2].y << " "
            << triangles[i].p[2].z << " " << std::endl;
    txtfile << "  endloop" << std::endl;
    txtfile << "endfacet" << std::endl;
  }
  txtfile << "endsolid mysolid" << std::endl;
  txtfile.close ();
}


void Boundary ::WritePatch (std::string file_name) {
  // Writing stl file
  std::ofstream txtfile;
  txtfile.open (file_name);

  for (int i = 0; i < numTriangles; i++) {
    txtfile << triangles[i].p[0].x << " " << triangles[i].p[0].y << " " << triangles[i].p[0].z
            << std::endl;
    txtfile << triangles[i].p[1].x << " " << triangles[i].p[1].y << " " << triangles[i].p[1].z
            << std::endl;
    txtfile << triangles[i].p[2].x << " " << triangles[i].p[2].y << " " << triangles[i].p[2].z
            << std::endl;
  }

  txtfile.close ();
}


void Boundary ::Optimize (std::vector<double> &bsens, double moveLimit, double volume,
                          double volCons, int algo) {
  // Initiate myOptimizer
  int nDesVar = numTriangles;
  int nCons = 1;
  MyOptimizer myopt (nDesVar, nCons);  // an optimizer object

  // set bounds
  for (int i = 0; i < nDesVar; i++) {
    double minDist = -1.0e5;  // min distance boundary point is allowed
    double maxDist = 1.0e5;   // max distance boundary point is allowed
    DomainDistance (bPoints[i], minDist, maxDist);
    FixLocDistance (bPoints[i], minDist, maxDist);

    myopt.z_lo[i] = std::max (minDist, -moveLimit);
    myopt.z_up[i] = std::min (maxDist, moveLimit);
    myopt.z_up[i] = std::max (myopt.z_up[i], myopt.z_lo[i]);
    myopt.z_up[i] = std::min (myopt.z_up[i], moveLimit);
    myopt.z[i] = (0.5 * myopt.z_lo[i] + 0.5 * myopt.z_up[i]);
  }

  // objective gradient
  bool nroptFlag = (algo == 0);
  if (nroptFlag) {
    for (int i = 0; i < nDesVar; i++) {
      myopt.objFunGrad[i] = bsens[i];
    }
  } else {
    for (int i = 0; i < nDesVar; i++) {
      myopt.objFunGrad[i] = bsens[i] * bAreas[i];
    }
  }

  // constraint gradient and value
  for (int i = 0; i < nDesVar; i++) {
    myopt.conFunGrad[0][i] = bAreas[i];
  }
  double boxVolume = double (ncellsX * ncellsY * ncellsZ);
  myopt.conMaxVals[0] = volCons * boxVolume - volume;

  // solve the optimzation problem
  if (algo == 0) myopt.SolveWithNR ();
  if (algo == 1) myopt.SolveWithNlopt ();
  if (algo == 2) myopt.SolveWithSimplex ();

  opt_vel.resize (nDesVar, 0.0);
  for (int i = 0; i < nDesVar; i++) {
    opt_vel[i] = std::min (myopt.z[i], myopt.z_up[i]);
    opt_vel[i] = std::max (opt_vel[i], myopt.z_lo[i]);
  }
}


void Boundary ::Optimize (std::vector<double> &objSens, std::vector<std::vector<double>> &conSens,
                          std::vector<double> conMaxVals, std::vector<double> conCurVals,
                          double moveLimit, int algo) {
  // Initiate myOptimizer
  int nDesVar = numTriangles;
  int nCons = conCurVals.size ();
  MyOptimizer myopt (nDesVar, nCons);  // an optimizer object

  // set bounds
  for (int i = 0; i < nDesVar; i++) {
    double minDist = -1.0e5;  // min distance boundary point is allowed
    double maxDist = 1.0e5;   // max distance boundary point is allowed
    DomainDistance (bPoints[i], minDist, maxDist);
    FixLocDistance (bPoints[i], minDist, maxDist);
    myopt.z_lo[i] = std::max (minDist, -moveLimit);
    myopt.z_up[i] = std::min (maxDist, moveLimit);
    myopt.z_up[i] = std::max (myopt.z_up[i], myopt.z_lo[i]);
    myopt.z_up[i] = std::min (myopt.z_up[i], moveLimit);
    myopt.z[i] = (0.5 * myopt.z_lo[i] + 0.5 * myopt.z_up[i]);
  }

  // objective gradient
  bool nroptFlag = (algo <= 0);
  for (int i = 0; i < nDesVar; i++) {
    myopt.objFunGrad[i] = objSens[i];
  }  // NOTE: remove this after debigging

  // constraint gradient and value
  for (int j = 0; j < nCons; j++) {
    for (int i = 0; i < nDesVar; i++) {
      myopt.conFunGrad[j][i] = conSens[j][i];
    }  // NOTE: remove this after debigging}
    myopt.conMaxVals[j] = conMaxVals[j] - conCurVals[j];
  }

  // pass in boundary areas
  myopt.bAreas = bAreas;

  // solve the optimzation problem
  if (algo == 0) myopt.SolveWithNR ();
  if (algo == 1) myopt.SolveWithNlopt ();
  if (algo == 2) myopt.SolveWithSimplex ();
  if (algo == -1) {
    myopt.SolveCustom ();
    if (!myopt.did_nr_converge) {
      std::cout << "Switching to MMA" << std::endl;
      myopt.SolveWithNlopt ();
    }
  }

  opt_vel.resize (nDesVar, 0.0);
  for (int i = 0; i < nDesVar; i++) {
    opt_vel[i] = std::min (myopt.z[i], myopt.z_up[i]);
    opt_vel[i] = std::max (opt_vel[i], myopt.z_lo[i]);
  }
}


void Boundary ::ExtrapolateVelocities () {
  // extrapolates the boundary velocities to nearby grid points
  int nGridPoints = levelSet.nGridPoints;

  // intitizlie the weights and velocity
  std::vector<double> weight (nGridPoints, 0.0);
  std::vector<double> weightedvel (nGridPoints, 0.0);

  // resize grid velocities
  levelSet.gridVel.resize (nGridPoints, 0.0);
  levelSet.isInNarrowBand.resize (nGridPoints);

  // Extrapolate velocities using this helper function
  ExtrapolateWeightedVelocities (weight, weightedvel);

// assign grid velocities and define narrow band
#pragma omp parallel for
  for (int i = 0; i < nGridPoints; i++) {
    if (weight[i] > 1e-7) {
      levelSet.gridVel[i] = weightedvel[i] / weight[i];
      levelSet.isInNarrowBand[i] = true;
    } else {
      levelSet.gridVel[i] = 0.0;
      levelSet.isInNarrowBand[i] = false;
    }
  }
}


void Boundary ::ExtrapolateWeightedVelocities (std::vector<double> &weight,
                                               std::vector<double> &weightedvel) {
  // extrapolate weights and weighted velocities
  double baseVal = 0.001;
  double rootThree = std::sqrt (3.0);
  int hWidth = levelSet.hWidth;
#pragma omp parallel num_threads(8)
  {
    int nGridPoints = levelSet.nGridPoints;
    std::vector<float> weightP (nGridPoints, 0.0);  // floats to save space
    std::vector<float> weightedvelP (nGridPoints, 0.0);
#pragma omp for
    for (int cTri = 0; cTri < numTriangles; cTri++) {
      for (int iRel = -hWidth; iRel <= hWidth; iRel++) {
        for (int jRel = -hWidth; jRel <= hWidth; jRel++) {
          for (int kRel = -hWidth; kRel <= hWidth; kRel++) {
            std::vector<double> mPoint = bPoints[cTri];
            int i = std::floor (mPoint[0] - 0.5) + iRel;
            int j = std::floor (mPoint[1] - 0.5) + jRel;
            int k = std::floor (mPoint[2] - 0.5) + kRel;
            if (levelSet.IsInBounds (i, j, k)) {
              // distance from the boundary point to the grid point
              double dist_sq = std::pow (i - mPoint[0], 2.0);
              dist_sq += std::pow (j - mPoint[1], 2.0);
              dist_sq += std::pow (k - mPoint[2], 2.0);
              // dist  = std::sqrt (dist) ;
              // double wTemp = std::max(rootThree - dist, baseVal);
              double wTemp = 1.0 / (baseVal + dist_sq);
              int ijk = levelSet.GridPtToIndex (i, j, k);
              weightedvelP[ijk] += float (wTemp * opt_vel[cTri]);
              weightP[ijk] += float (wTemp);
            }  // end if
          }    // k
        }      // j
      }        // i
    }          // end of cTri
#pragma omp critical
    for (int i = 0; i < nGridPoints; i++) {
      weightedvel[i] += weightedvelP[i];
      weight[i] += weightP[i];
    }
  }
}


// compute domain distance
void Boundary ::DomainDistance (std::vector<double> bPoint, double &minDist, double &maxDist) {
  maxDist
      = std::min ({bPoint[0], -(bPoint[0] - double (ncellsX)), bPoint[1],
                   -(bPoint[1] - double (ncellsY)), bPoint[2], -(bPoint[2] - double (ncellsZ))});

  int holeCount = levelSet.domainVoids.size ();
  for (int hC = 0; hC < holeCount; hC++) {
    double minDistTemp = levelSet.domainVoids[hC]->GetMinDist (bPoint);
    maxDist = std::min (maxDist, -minDistTemp);
  }
}


// compute distance from fixed locaions
void Boundary ::FixLocDistance (std::vector<double> bPoint, double &minDist, double &maxDist) {
  minDist = -1.0e5;  // some large number
  int holeCount = levelSet.fixedBlobs.size ();
  for (int hC = 0; hC < holeCount; hC++) {
    double minDistTemp = levelSet.fixedBlobs[hC]->GetMinDist (bPoint);
    minDist = std::max (minDist, minDistTemp);
  }

  minDist = std::min (minDist, 0.0);
}


void Boundary ::ComputePerturbationSensitivities (double perturb) {
  // Reize perturbation information values
  perturb_indicies_values.resize (numTriangles);

// Loop through all boundary points
#pragma omp parallel for
  for (int i = 0; i < numTriangles; i++) {
    // define a perturb struct
    PerturbProperties curr_perturb;
    perturb_indicies_values[i].clear ();

    // Define the half-width of the narrow band
    int nb_hwidth = 3;

    // Min and max cordinates in the x direction
    int nelx_mini_0 = std::max (int (round (bPoints[i][0])) - nb_hwidth, 0);
    nelx_mini_0 = std::min (nelx_mini_0, int (ncellsX) - 2 * nb_hwidth);
    int nelx_mini_1 = nelx_mini_0 + 2 * nb_hwidth;

    // Min and max cordinates in the y direction
    int nely_mini_0 = std::max (int (round (bPoints[i][1])) - nb_hwidth, 0);
    nely_mini_0 = std::min (nely_mini_0, int (ncellsY) - 2 * nb_hwidth);
    int nely_mini_1 = nely_mini_0 + 2 * nb_hwidth;

    // Min and max cordinates in the z direction
    int nelz_mini_0 = std::max (int (round (bPoints[i][2])) - nb_hwidth, 0);
    nelz_mini_0 = std::min (nelz_mini_0, int (ncellsZ) - 2 * nb_hwidth);
    int nelz_mini_1 = nelz_mini_0 + 2 * nb_hwidth;

    // Size of the mini grid
    int nelx_mini = nelx_mini_1 - nelx_mini_0;
    int nely_mini = nely_mini_1 - nely_mini_0;
    int nelz_mini = nelz_mini_1 - nelz_mini_0;

    // Create the mini grid and mini level set
    Grid grid_mini (nelx_mini, nely_mini, nelz_mini);
    LevelSet3D level_set_mini (grid_mini);
    level_set_mini.hWidth = nb_hwidth;
    level_set_mini.MakeBox ();
    level_set_mini.volFractions.resize (level_set_mini.nCells, 1.0);

    // Copy the signed distance values to the mini level set
    int counter = 0;
    for (int ix = 0; ix <= nelx_mini; ix++) {
      for (int iy = 0; iy <= nely_mini; iy++) {
        for (int iz = 0; iz <= nelz_mini; iz++) {
          int global_x = nelx_mini_0 + ix;
          int global_y = nely_mini_0 + iy;
          int global_z = nelz_mini_0 + iz;
          int index = levelSet.GridPtToIndex (global_x, global_y, global_z);
          level_set_mini.phi[counter].val = levelSet.phi[index].val;

          counter++;
        }
      }
    }

    // Initialize the mini boundary
    Boundary boundary_mini (level_set_mini);
    boundary_mini.MarchCubes ();

    // Calculate and store the initial volume fractions of the mini grid
    level_set_mini.ComputeVolumeFractions ();
    std::vector<double> vol_mini_init (level_set_mini.nCells, 0.0);
    for (int e = 0; e < level_set_mini.nCells; e++) {
      vol_mini_init[e] = level_set_mini.volFractions[e];
    }

    // Update signed distance depending on distance to perturbed bpoint
    int nnx_mini = nelx_mini + 1;  // number of nodes in the x direction
    int nny_mini = nely_mini + 1;  // number of nodes in the y direction
    int nnz_mini = nelz_mini + 1;  // number of nodes in the z direction
    for (int gpt = 0; gpt < level_set_mini.nGridPoints; gpt++) {
      // Get x,y,z coordinates from the grid point index
      int x_m = int (gpt / (nnz_mini * nny_mini));
      int y_m = int ((gpt - x_m * nnz_mini * nny_mini) / nnz_mini);
      int z_m = gpt - x_m * nnz_mini * nny_mini - y_m * nnz_mini;

      // Calculate distance from the boundary point being perturbed (i loop) to
      // the current grid point (from the mini grid)
      double distance = std::pow (bPoints[i][0] - (x_m + nelx_mini_0), 2)
                        + std::pow (bPoints[i][1] - (y_m + nely_mini_0), 2)
                        + std::pow (bPoints[i][2] - (z_m + nelz_mini_0), 2);
      distance = std::sqrt (distance);

      // Update signed distance using phi = phi - vn*dt
      // Note that dt = 1 and vn (velocity) is determined by the perturbation
      // size and distance from the perturbed boundary point. If the node is
      // more than 1.5 away from the perturbed boundary point, it's signed
      // distance value is not changed.
      level_set_mini.phi[gpt].val -= perturb * std::max (0.0, 1.5 - distance);
    }

    // Calculate the volume fractions of the perturbed mini level set
    boundary_mini.MarchCubes ();
    level_set_mini.ComputeVolumeFractions ();

    // Calculate the change in volume fractions and the sensitivities
    double delta_volume;
    counter = 0;
    for (int iz = 0; iz < nelz_mini; iz++) {
      for (int iy = 0; iy < nely_mini; iy++) {
        for (int ix = 0; ix < nelx_mini; ix++) {
          // Index of the element in the global level set grid
          int global_x = nelx_mini_0 + ix;
          int global_y = nely_mini_0 + iy;
          int global_z = nelz_mini_0 + iz;
          int index = global_x + ncellsX * global_y + ncellsX * ncellsY * global_z;

          // Calculate the change in volume
          delta_volume = vol_mini_init[counter] - level_set_mini.volFractions[counter];

          // Calculate the sensitivity only where the change in area is significant
          if (std::abs (delta_volume) > 1e-3 * perturb) {
            curr_perturb.index = index;
            curr_perturb.value = delta_volume;
            perturb_indicies_values[i].push_back (curr_perturb);
          }

          counter++;
        }
      }
    }
  }
}


// compute boundary sensitivities
void Boundary::ComputeAnalyticalSensitivities (std::vector<double> &bsens, double *sensiArray,
                                               std::vector<int> &volFracToPetscMap,
                                               bool usePetscMap) {
  // Compute the boundary sensitivities using least squares interpolation
  // Resize bsens
  bsens.resize (numTriangles, 0.0);
  int nCells = ncellsX * ncellsY * ncellsZ;

  for (int countCell = 0; countCell < nCells; countCell++) {
    int mappedIndex = usePetscMap * volFracToPetscMap[countCell] + countCell * (1 - usePetscMap);

    int triSize = triCellIndices[countCell].size ();
    for (int nTri = 0; nTri < triSize; nTri++) {
      // assign the corresponding FE sensitivity
      if (levelSet.volFractions[countCell] > 0.2)
        bsens[triCellIndices[countCell][nTri]] = sensiArray[mappedIndex];
    }  // ntRI
  }    // countCell
}


// compute boundary sensitivities
void Boundary ::InterpolateBoundarySensitivities (std::vector<double> &bsens, double *sensiArray,
                                                  std::vector<int> &volFracToPetscMap, int hWidth,
                                                  int weightedVolFrac, bool usePetscMap) {
  // Compute the boundary sensitivities using least squares interpolation
  // Resize bsens
  bsens.resize (numTriangles);
#pragma omp parallel for
  for (int i = 0; i < numTriangles; i++) {
    // get neigboring cells
    std::vector<std::vector<int>> adjCellIndices = GetNearbyCellsInfo (bPoints[i], hWidth);

    // Initialize least squares interpolation matrices
    int nSamples = adjCellIndices.size ();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero (nSamples, 4);
    Eigen::VectorXd B = Eigen::VectorXd::Zero (nSamples);

    // Set up least squares interpolation matrices
    for (int j = 0; j < nSamples; j++) {
      // x,y,z are distance from boundary point to cell centroids
      double x = (-bPoints[i][0] + adjCellIndices[j][0]) + 0.5;
      double y = (-bPoints[i][1] + adjCellIndices[j][1]) + 0.5;
      double z = (-bPoints[i][2] + adjCellIndices[j][2]) + 0.5;
      double dist = std::sqrt ((x * x + y * y) + z * z);

      // Assign weight based on distance and volume fraction of the element
      double weight = std::max (0.001, 3.0 - dist);
      weight = std::min (1.5, weight);
      int ijk = adjCellIndices[j][3];
      // if(levelSet.volFractions[ijk] < 0.2){
      //   weight = 0.001;
      // }
      weight *= (weightedVolFrac * levelSet.volFractions[ijk]) + ((1 - weightedVolFrac) * 1.0);
      A (j, 0) = weight;
      A (j, 1) = weight * x;
      A (j, 2) = weight * y;
      A (j, 3) = weight * z;
      int mappedIndex = usePetscMap * volFracToPetscMap[ijk] + ijk * (1 - usePetscMap);
      B (j) = (weight * sensiArray[mappedIndex])
              * (weightedVolFrac * levelSet.volFractions[ijk] + double (1 - weightedVolFrac));
    }

    // Eigen::MatrixXd AtA = A.transpose()*A ;
    // Eigen::VectorXd BtB = A.transpose()*B ;
    // Solve Least Squares
    Eigen::VectorXd X = A.colPivHouseholderQr ().solve (B);
    if (!std::isnan (X[0])) bsens[i] = X[0];
  }
}


// void Boundary :: InterpolateBoundarySensitivities (std::vector<double> &bsens,
//     double *sensiArray, std::vector<int> &volFracToPetscMap, double perturbation,
//     bool usePetscMap) {
//   // TODO: delete after debugging
//   // std::cout << "\nEntered InterpolateBoundarySensitivities" << std::endl ;

//   // Compute the boundary sensitivities using the perturbation method
//   // Resize bsens
//   bsens.resize (numTriangles) ;
//   // #pragma omp parallel for

//   // // TODO: delete after debugging
//   // std::cout << "   numTriangles    = " << numTriangles << std::endl ;
//   // std::cout << "   bsens size      = " << bsens.size () << std::endl ;
//   // std::cout << "   perturb size    = " << perturb_indicies.size () << std::endl ;

//   for (int i = 0 ; i < numTriangles ; i++) {
//     // Initialize the total change in sensitivity due to the perturbed boundary pt
//     double delta_sens = 0.0 ;

//     // // TODO: delete after debugging
//     // std::cout << "      for i  = " << i << std::endl ;
//     // std::cout << "         j size     = " << perturb_indicies[i].size () << std::endl ;
//     // std::cout << "         sens size  = " << perturb_sensitivities[i].size () << std::endl ;
//     // if (i % 10 == 0) {std::cout << "   [" << i << "]:" << std::endl ;}

//     // For all the appropriate elements in the perturbed mesh
//     for (int j = 0 ; j < perturb_indicies[i].size () ; j++) {
//       // // TODO: delete after debugging
//       // std::cout << "         for j = " << j << std::endl ;

//       // Index of the perturbed element
//       int curr_index  = perturb_indicies[i][j] ;
//       int mappedIndex = usePetscMap*volFracToPetscMap[curr_index] +
//         curr_index*(1 - usePetscMap) ;

//       // // TODO: delete after debugging
//       // std::cout << "            curr_index  = " << curr_index << std::endl ;
//       // std::cout << "            mappedIndex = " << mappedIndex << std::endl ;
//       // if (i % 10 == 0) {
//       //   std::cout << "      mappedIndex  = " << mappedIndex << std::endl ;
//       //   std::cout << "      sensiArray   = " << sensiArray[mappedIndex] << std::endl ;
//       //   std::cout << "      perturbSens  = " << perturb_sensitivities[i][j] << std::endl ;
//       // }

//       // Add the contribution of this element to the total sensitivity
//       delta_sens += sensiArray[mappedIndex]*perturb_sensitivities[i][j] ;
//     }

//     // Calculate the sensitvitity
//     bsens[i] = delta_sens / perturbation ;
//   }
// }


// Helper function for least squares
std::vector<std::vector<int>> Boundary ::GetNearbyCellsInfo (std::vector<double> bPoint,
                                                             int hWidth) {
  // return adjCellIndices of a given boudnary point
  std::vector<std::vector<int>> adjCellIndices;

  int i = int (bPoint[0]);
  int j = int (bPoint[1]);
  int k = int (bPoint[2]);

  for (int iRel = i - hWidth; iRel <= i + hWidth; iRel++) {
    for (int jRel = j - hWidth; jRel <= j + hWidth; jRel++) {
      for (int kRel = k - hWidth; kRel <= k + hWidth; kRel++) {
        if (IsCellInBounds (iRel, jRel, kRel)) {
          // cell counter
          int counter_cell = iRel + ncellsX * jRel + ncellsX * ncellsY * kRel;
          adjCellIndices.push_back ({iRel, jRel, kRel, counter_cell});
        }
      }  // k
    }    // j
  }      // i

  return adjCellIndices;
}


// check if an index of a cell is in bound
bool Boundary ::IsCellInBounds (int i, int j, int k) {
  bool isInBounds = (i >= 0 && j >= 0 && k >= 0 && i < ncellsX && j < ncellsY && k < ncellsZ);

  return isInBounds;
}

}  // namespace para_lsm