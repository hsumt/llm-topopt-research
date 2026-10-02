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

#include "lsm_3d.h"

namespace para_lsm {

// constructor
LevelSet3D ::LevelSet3D (Grid &grid_) : grid (grid_) {
  // set number of elements
  nElemX = grid.nelx;
  nElemY = grid.nely;
  nElemZ = grid.nelz;

  // number of grid points
  nGridPoints = (nElemX + 1) * (nElemY + 1) * (nElemZ + 1);

  // total number of elements (cells)
  nCells = (nElemX) * (nElemY) * (nElemZ);
}


void LevelSet3D ::Update () {
  // Sweep
  SweepVelocity ();

  // estimate number of CFL steps
  int cflCount = AdjustForCFL ();
  // update the level set function
  for (int j = 0; j < cflCount; j++) {
    // compute gradinets
    ComputeGradients ();
// advect
#pragma omp parallel for
    for (int i = 0; i < nGridPoints; i++) {
      phi[i].val += gridVel[i] * gradPhi[i] / (1.0 * cflCount);
    }
  }

  // Make sure the phi is 0 on the domain boundary if it's positive
  for (int i = 0; i < nElemX + 1; i++) {
    for (int j = 0; j < nElemY + 1; j++) {
      for (int k = 0; k < nElemZ + 1; k++) {
        bool is_on_domain
            = (i == 0 || i == nElemX) || (j == 0 || j == nElemY) || (k == 0 || k == nElemZ);
        if (is_on_domain && phi[GridPtToIndex (i, j, k)].val > 0) {
          phi[GridPtToIndex (i, j, k)].val = 0.0;
        }  // end if
      }    // k
    }      // j
  }        // i
}  // end of function


int LevelSet3D ::AdjustForCFL () {
  double max_vel = 0.0;  // initialize
  for (int i = 0; i < nGridPoints; i++) {
    max_vel = std::max (max_vel, std::abs (gridVel[i]));
  }

  double cflValue = 0.9;
  int cflCount = 1 + std::floor (max_vel / cflValue);

  return cflCount;
}


int LevelSet3D ::GridPtToIndex (int x, int y, int z) {
  // returns the index of a grid point in the zyx system (Grid4Vector convention)
  return z + y * (nElemZ + 1) + x * (nElemZ + 1) * (nElemY + 1);
}


std::vector<int> LevelSet3D ::IndexToGridPt (int index) {
  // returns the grid point of an index in the zyx system (Grid4Vector convention)
  std::vector<int> ret (3, 0);
  ret[0] = int (index / ((nElemZ + 1) * (nElemY + 1)));
  ret[1] = int ((index - ret[0] * (nElemZ + 1) * (nElemY + 1)) / (nElemZ + 1));
  ret[2] = index - ret[0] * (nElemZ + 1) * (nElemY + 1) - ret[1] * (nElemZ + 1);
  return ret;
}


void LevelSet3D ::SetGridDimensions (int box_x, int box_y, int box_z) {
  // set the box dimensions and initialize the grid
  nElemX = box_x;
  nElemY = box_y;
  nElemZ = box_z;

  // number of grid points
  nGridPoints = (box_x + 1) * (box_y + 1) * (box_z + 1);

  // number of elements (cells)
  nCells = (box_x) * (box_y) * (box_z);

  // phi = &grid.phi; //
  phi.resize (nGridPoints);

  // set grid properties
  grid.nelx = nElemX;
  grid.nely = nElemY;
  grid.nelz = nElemZ;
}


void LevelSet3D ::MakeBox () {
//  set the signed distance function corresponsing to a box
// Loop through all level set grid points
#pragma omp parallel for
  for (int i = 0; i < nElemX + 1; i++) {
    for (int j = 0; j < nElemY + 1; j++) {
      for (int k = 0; k < nElemZ + 1; k++) {
        int current_index = GridPtToIndex (i, j, k);
        phi[current_index].val = std::min ({i, nElemX - i, j, nElemY - j, k, nElemZ - k});
      }
    }
  }
  MakeInitialHoles ();
}


void LevelSet3D ::MakeInitialHoles () {
  // Loop through all level set grid points
  int holeCount = initialVoids.size ();
#pragma omp parallel for
  for (int i = 0; i < nElemX + 1; i++) {
    for (int j = 0; j < nElemY + 1; j++) {
      for (int k = 0; k < nElemZ + 1; k++) {
        int current_index = GridPtToIndex (i, j, k);
        std::vector<double> point = {1.0 * i, 1.0 * j, 1.0 * k};
        for (int hC = 0; hC < holeCount; hC++) {
          double minDist = initialVoids[hC]->GetMinDist (point);
          phi[current_index].val = std::min (-minDist, phi[current_index].val);
        }
      }
    }
  }
}


void LevelSet3D ::MakeDomainHoles () {
  // Loop through all level set grid points
  int holeCount = domainVoids.size ();
#pragma omp parallel for
  for (int i = 0; i < nElemX + 1; i++) {
    for (int j = 0; j < nElemY + 1; j++) {
      for (int k = 0; k < nElemZ + 1; k++) {
        int current_index = GridPtToIndex (i, j, k);
        std::vector<double> point = {1.0 * i, 1.0 * j, 1.0 * k};
        for (int hC = 0; hC < holeCount; hC++) {
          double minDist = domainVoids[hC]->GetMinDist (point);
          phi[current_index].val = std::min (-minDist, phi[current_index].val);
        }
      }
    }
  }
}


void LevelSet3D ::ComputeVolumeFractions () {
  volFractions.resize (nCells);
#pragma omp parallel for
  for (int i = 0; i < nElemX; i++) {
    for (int j = 0; j < nElemY; j++) {
      for (int k = 0; k < nElemZ; k++) {
        // cell counter
        int countCell = i + nElemX * j + nElemX * nElemY * k;
        // construct the stencil
        Stencil mStencil (grid, i, j, k);
        // Save the indces of all the corner points
        std::vector<double> corner_phi_vals (8, 0);
        corner_phi_vals[0] = mStencil.GetValue (0, 0, 0);
        corner_phi_vals[1] = mStencil.GetValue (0, 0, 1);
        corner_phi_vals[2] = mStencil.GetValue (0, 1, 0);
        corner_phi_vals[3] = mStencil.GetValue (0, 1, 1);
        corner_phi_vals[4] = mStencil.GetValue (1, 0, 0);
        corner_phi_vals[5] = mStencil.GetValue (1, 0, 1);
        corner_phi_vals[6] = mStencil.GetValue (1, 1, 0);
        corner_phi_vals[7] = mStencil.GetValue (1, 1, 1);
        ComputeElementalVolFracs (corner_phi_vals, countCell);
      }  // k
    }    // j
  }      // i
  double volume_temp = 0.0;
#pragma omp parallel for reduction(+ : volume_temp)
  for (int i = 0; i < nCells; i++) {
    volume_temp += volFractions[i];
  }
  volume = volume_temp;
}


void LevelSet3D::ComputeAggregationSensitivities (std::vector<double> &aggSensi, double &maxVal,
                                                  int radius) {
  aggSensi.resize (nCells);
  std::vector<double> vfAgg (nCells);
  std::vector<double> vfAggWeight (nCells);
  maxVal = 0.0;
#pragma omp parallel for
  for (int i = 0; i < nElemX; i++) {
    for (int j = 0; j < nElemY; j++) {
      for (int k = 0; k < nElemZ; k++) {
        // cell counter
        int countCell = i + nElemX * j + nElemX * nElemY * k;
        // initialize volFraction sum and count
        double vfSum = 0.0;
        double vfCount = 0.0;
        for (int kk = std::max (0, k - radius); kk <= std::min (nElemZ - 1, k + radius); kk++) {
          for (int jj = std::max (0, j - radius); jj <= std::min (nElemY - 1, j + radius); jj++) {
            for (int ii = std::max (0, i - radius); ii <= std::min (nElemX - 1, i + radius); ii++) {
              int countTempCell = ii + nElemX * jj + nElemX * nElemY * kk;
              double mydist = std::sqrt (0.0 + (ii - i) * (ii - i) + (jj - j) * (jj - j)
                                         + (kk - k) * (kk - k));

              double wConvo = 0.0;
              if (mydist <= radius) wConvo = 1.0;

              vfCount += wConvo;
              vfSum += volFractions[countTempCell] * wConvo;

            }  // ii
          }    // jj
        }      // kk
        vfAgg[countCell] = vfSum / vfCount;
        vfAggWeight[countCell] = vfCount;

      }  // k
    }    // j
  }      // i

  double G = 0.0;
  double pnorm = 8.0;
  for (int i = 0; i < vfAgg.size (); i++) {
    G += std::pow (vfAgg[i], pnorm);
    maxVal = std::max (vfAgg[i], maxVal);
  }

  double approxVal = std::pow (G, 1.0 / pnorm);
  std::cout << " approx , max = " << approxVal << " , " << maxVal << std::endl;
  double Jfac = std::pow (G, 1.0 / pnorm - 1.0) / pnorm;

#pragma omp parallel for
  for (int i = 0; i < nElemX; i++) {
    for (int j = 0; j < nElemY; j++) {
      for (int k = 0; k < nElemZ; k++) {
        // cell counter
        int countCell = i + nElemX * j + nElemX * nElemY * k;
        // initialize volFraction sum and count
        double sensiSum = 0.0;
        for (int kk = std::max (0, k - radius); kk <= std::min (nElemZ - 1, k + radius); kk++) {
          for (int jj = std::max (0, j - radius); jj <= std::min (nElemY - 1, j + radius); jj++) {
            for (int ii = std::max (0, i - radius); ii <= std::min (nElemX - 1, i + radius); ii++) {
              int countTempCell = ii + nElemX * jj + nElemX * nElemY * kk;
              double mydist = std::sqrt (0.0 + (ii - i) * (ii - i) + (jj - j) * (jj - j)
                                         + (kk - k) * (kk - k));

              if (mydist <= radius) {
                // double sensiWt = pnorm*std::pow(vfAgg[countTempCell] , pnorm - 1.0);
                double sensiWt = pnorm;
                for (int i1 = 0; i1 < std::round (pnorm) - 1; i1++) sensiWt *= vfAgg[countTempCell];
                sensiSum += sensiWt / vfAggWeight[countTempCell];
              }

            }  // ii
          }    // jj
        }      // kk
        aggSensi[countCell] = sensiSum;
      }  // k
    }    // j
  }      // i

#pragma omp parallel for
  for (int i = 0; i < vfAgg.size (); i++) {
    aggSensi[i] *= Jfac * maxVal / approxVal;
  }

  std::cout << "Completed" << std::endl;
}


void LevelSet3D ::ComputeElementalVolFracs (std::vector<double> &corner_phi_vals, int countCell) {
  // compute volume fraction of an element
  double numer = 0.0;
  double denom = 0.0;
  for (int i = 0; i < 8; i++) {
    numer += std::max (0.0, corner_phi_vals[i]);
    denom += std::abs (corner_phi_vals[i]);
  }
  volFractions[countCell] = numer / denom;
}


void LevelSet3D ::InitializeSweep () {
  // Assign large values for grid points that have same sign neighbours
  for (int sign_count = 0; sign_count < 2; sign_count++) {
#pragma omp parallel for
    for (int cGrid = 0; cGrid < nGridPoints; cGrid++) {
      std::vector<int> ijk = IndexToGridPt (cGrid);
      int width = 1;
      bool ignoreSelf = true;

      int i = ijk[0];
      int j = ijk[1];
      int k = ijk[2];

      // construct the min stencil
      MinStencil minStencil (grid, i, j, k, width, ignoreSelf);
      double phix = minStencil.MinValue (0);
      double phiy = minStencil.MinValue (1);
      double phiz = minStencil.MinValue (2);

      // current phi
      double phi0 = minStencil.GetValue ();

      // assign large value if all phi-s are positive
      if (phi0 > 0.0 && phix > 0.0 && phiy > 0.0 && phiz > 0.0) {
        phi[cGrid].val = large_value;
      }
    }  // i

// Flip sign
#pragma omp parallel for
    for (int cGrid = 0; cGrid < nGridPoints; cGrid++) {
      phi[cGrid].val *= -1.0;
    }
  }  // sign_count
}  // end of function


void LevelSet3D ::Reinitialize () {
  // Initialize the process
  InitializeSweep ();

  // Use sweeping to reinitialize
  double small_value = 0.0;
  for (int count = 0; count < 2; count++) {
#pragma omp parallel for
    for (int sweep = 0; sweep < 8; sweep++) {
      int istart, iend, idiff, jstart, jend, jdiff, kstart, kend, kdiff;
      DefineSweepingDirections (sweep, istart, iend, idiff, jstart, jend, jdiff, kstart, kend,
                                kdiff);

      // sweep
      for (int i = istart; (i >= 0 && i <= nElemX); i = i + idiff) {
        for (int j = jstart; (j >= 0 && j <= nElemY); j = j + jdiff) {
          for (int k = kstart; (k >= 0 && k <= nElemZ); k = k + kdiff) {
            if (phi[GridPtToIndex (i, j, k)].val > small_value) {
              int index = GridPtToIndex (i, j, k);
              double phi_old = phi[index].val;
              // solve the ekonal equation for this point
              std::vector<int> ijk = {i, j, k};
              SolveEikonal (ijk);
              // reset the value if it's greater than the older one
              if (phi_old < phi[index].val) phi[index].val = phi_old;
            }  // end of eikonal
          }    // end of k
        }      // end of j
      }        // end of i
    }          // end of sweep

#pragma omp parallel for
    for (int i = 0; i < nGridPoints; i++) phi[i].val *= -1.0;  // flip sign
  }                                                            // end of count
}  // end of function


void LevelSet3D ::SweepVelocity () {
  // Initialize the process
  InitializeSweep ();

  double small_value = 0.0;
  for (int sign_count = 0; sign_count < 2; sign_count++) {
#pragma omp parallel for
    for (int sweep = 0; sweep < 8; sweep++) {
      int istart, iend, idiff, jstart, jend, jdiff, kstart, kend, kdiff;
      DefineSweepingDirections (sweep, istart, iend, idiff, jstart, jend, jdiff, kstart, kend,
                                kdiff);
      // sweep
      for (int i = istart; (i >= 0 && i <= nElemX); i = i + idiff) {
        for (int j = jstart; (j >= 0 && j <= nElemY); j = j + jdiff) {
          for (int k = kstart; (k >= 0 && k <= nElemZ); k = k + kdiff) {
            int index = GridPtToIndex (i, j, k);

            if (phi[index].val > small_value && isInNarrowBand[index]) {
              double phi_old = phi[index].val;
              std::vector<int> ijk = {i, j, k};
              SolveEikonal (ijk);              // solve eikonal at this point
              if (phi[index].val < phi_old) {  // correct characteristic
                phi[index].val = std::min (phi[index].val, double (hWidth));
                UpdateVelocity (i, j, k);
              } else
                phi[GridPtToIndex (i, j, k)].val = phi_old;  // reset
            }                                                // if eikonal
          }                                                  // i
        }                                                    // j
      }                                                      // k
    }                                                        // sweep

// flip sign
#pragma omp parallel for
    for (int i = 0; i < nGridPoints; i++) {
      phi[i].val *= -1.0;
    }
  }  // end of sign_count

  SetNarrowBandPhi ();
}  // end of function


void LevelSet3D ::SetNarrowBandPhi () {
// Set phi at narrow band
#pragma omp parallel for  // reset values outside narrow band
  for (int i = 0; i < nGridPoints; i++) {
    if (!isInNarrowBand[i]) {
      if (phi[i].val > 0.0)
        phi[i].val = 1.0 * hWidth;
      else
        phi[i].val = -1.0 * hWidth;
    }
  }
}


void LevelSet3D ::DefineSweepingDirections (int sweep, int &istart, int &iend, int &idiff,
                                            int &jstart, int &jend, int &jdiff, int &kstart,
                                            int &kend, int &kdiff) {
  // Define sweeping directions
  if (sweep == 0 || sweep == 2 || sweep == 4 || sweep == 6) {
    istart = 0;
    idiff = 1;
    iend = nElemX;
  } else {
    istart = nElemX;
    idiff = -1;
    iend = 0;
  }

  if (sweep == 0 || sweep == 1 || sweep == 4 || sweep == 5) {
    jstart = 0;
    jdiff = 1;
    jend = nElemY;
  } else {
    jstart = nElemY;
    jdiff = -1;
    jend = 0;
  }

  if (sweep == 0 || sweep == 1 || sweep == 2 || sweep == 3) {
    kstart = 0;
    kdiff = 1;
    kend = nElemZ;
  } else {
    kstart = nElemZ;
    kdiff = -1;
    kend = 0;
  }
}  // end of function


void LevelSet3D ::SolveEikonal (std::vector<int> ijk) {
  // solves the eikonal equations for grid points
  int width = 1;
  bool ignoreSelf = true;
  int i = ijk[0];
  int j = ijk[1];
  int k = ijk[2];
  int index = GridPtToIndex (i, j, k);

  // construct the min stencil
  MinStencil minStencil (grid, i, j, k, width, ignoreSelf);
  double phix = minStencil.MinValue (0);
  double phiy = minStencil.MinValue (1);
  double phiz = minStencil.MinValue (2);

  // current phi
  double phi0 = minStencil.GetValue ();

  // solve if all are positive
  if (phi0 > 0.0 && phix > 0.0 && phiy > 0.0 && phiz > 0.0) {
    double a_quad = 3.0;
    double b_quad = -2.0 * ((phix + phiy) + phiz);
    double c_quad = ((phix * phix) + (phiy * phiy)) + ((phiz * phiz) - 1.0);

    // solve the quadratic equation
    if (b_quad * b_quad >= (4 * a_quad) * c_quad) {
      phi[index].val
          = 0.5 * ((-b_quad + std::sqrt ((b_quad * b_quad) - (4 * a_quad) * c_quad)) / a_quad);
    } else {
      phi[index].val = std::min ({phix, phiy, phiz}) + 1.0;
    }
  }  // if
}  // end of function


void LevelSet3D ::UpdateVelocity (int i, int j, int k) {
  // Update velocity at the index
  double ext_weight_x, ext_weight_y, ext_weight_z;
  double ext_vel_x, ext_vel_y, ext_vel_z;
  double small_value = 1.0e-6;

  // construct the vel grad stencil
  VelGradStencil myGradStencil (grid, i, j, k);

  myGradStencil.GetVelGrad (gridVel, 0, ext_weight_x, ext_vel_x);
  myGradStencil.GetVelGrad (gridVel, 1, ext_weight_y, ext_vel_y);
  myGradStencil.GetVelGrad (gridVel, 2, ext_weight_z, ext_vel_z);

  // set up bounds for ext_weights
  ext_weight_x = std::max (small_value, ext_weight_x);
  ext_weight_y = std::max (small_value, ext_weight_y);
  ext_weight_z = std::max (small_value, ext_weight_z);

  // update velocity
  double gridvel = ext_vel_x * ext_weight_x + ext_vel_y * ext_weight_y + ext_vel_z * ext_weight_z;
  gridvel = gridvel / (ext_weight_x + ext_weight_y + ext_weight_z);

  gridVel[GridPtToIndex (i, j, k)] = gridvel;
}  // end of function


void LevelSet3D::SmoothPhi (int box_smooth) {
  // smooth
  std::vector<double> smooth_phi (nGridPoints, 0.0);
  int box_z = nElemZ, box_y = nElemY, box_x = nElemX;
  for (int k = 0; k <= box_z; k++) {
    for (int j = 0; j <= box_y; j++) {
      for (int i = 0; i <= box_x; i++) {
        double count_smooth = 0;
        double delta_phi = 0.0;
        for (int k1 = std::max (0, k - box_smooth); k1 <= std::min (int (box_z), k + box_smooth);
             k1++) {
          for (int j1 = std::max (0, j - box_smooth); j1 <= std::min (int (box_y), j + box_smooth);
               j1++) {
            for (int i1 = std::max (0, i - box_smooth);
                 i1 <= std::min (int (box_x), i + box_smooth); i1++) {
              delta_phi += phi[GridPtToIndex (i1, j1, k1)].val;
              count_smooth = count_smooth + 1.0;
            }
          }
        }
        delta_phi /= count_smooth;
        smooth_phi[GridPtToIndex (i, j, k)] = delta_phi;
        bool is_on_domain = (i == 0 || i == box_x || j == 0 || j == box_y || k == 0 || k == box_z);
        if (is_on_domain && smooth_phi[GridPtToIndex (i, j, k)] > 0)
          smooth_phi[GridPtToIndex (i, j, k)] = 0.0;
      }  // i
    }    // j
  }      // k
  // replace phi with smooth phi
  for (int k = 0; k <= box_z; k++)
    for (int j = 0; j <= box_y; j++)
      for (int i = 0; i <= box_x; i++)
        phi[GridPtToIndex (i, j, k)].val = smooth_phi[GridPtToIndex (i, j, k)];
}  // end of function


void LevelSet3D::WriteSD (std::string file_name) {
  // print phi at grid points
  std::ofstream txtfile;
  txtfile.open (file_name);

  int cut_off_value = 3;
  // loop through grid points
  for (int k = 0; k <= nElemZ; k++) {
    for (int j = 0; j <= nElemY; j++) {
      for (int i = 0; i <= nElemX; i++) {
        if (phi[GridPtToIndex (i, j, k)].val >= cut_off_value) {
          txtfile << int (phi[GridPtToIndex (i, j, k)].val) << "\n";
        } else if (phi[GridPtToIndex (i, j, k)].val <= -cut_off_value) {
          txtfile << int (phi[GridPtToIndex (i, j, k)].val) << "\n";
        } else {
          txtfile << phi[GridPtToIndex (i, j, k)].val << "\n";
        }

      }  // i
    }    // j
  }      // k
  txtfile.close ();
}


void LevelSet3D ::ComputeGradients () {
  // resize gradient vector
  gradPhi.resize (nGridPoints, 1.0);
#pragma omp parallel for
  for (int i1 = 0; i1 < nGridPoints; i1++) {
    if (isInNarrowBand[i1]) {
      int sign = (gridVel[i1] < 0) ? -1 : 1;

      // construct the WENO stencil
      HJWENOStencil mHJWENOStencil (grid, IndexToGridPt (i1)[0], IndexToGridPt (i1)[1],
                                    IndexToGridPt (i1)[2]);

      double grad_x_minus = sign * mHJWENOStencil.GradValue (0, 0);
      double grad_x_plus = sign * mHJWENOStencil.GradValue (0, 1);
      double grad_y_minus = sign * mHJWENOStencil.GradValue (1, 0);
      double grad_y_plus = sign * mHJWENOStencil.GradValue (1, 1);
      double grad_z_minus = sign * mHJWENOStencil.GradValue (2, 0);
      double grad_z_plus = sign * mHJWENOStencil.GradValue (2, 1);

      double grad = 0.0;
      if (grad_x_minus < 0) grad += grad_x_minus * grad_x_minus;
      if (grad_y_minus < 0) grad += grad_y_minus * grad_y_minus;
      if (grad_z_minus < 0) grad += grad_z_minus * grad_z_minus;
      if (grad_x_plus > 0) grad += grad_x_plus * grad_x_plus;
      if (grad_y_plus > 0) grad += grad_y_plus * grad_y_plus;
      if (grad_z_plus > 0) grad += grad_z_plus * grad_z_plus;

      gradPhi[i1] = std::sqrt (grad);

    } else
      gradPhi[i1] = 1.0;
  }
}  // end of function


// check if an index is in bound
bool LevelSet3D ::IsInBounds (int x_index, int y_index, int z_index) {
  bool is_in_bounds = (x_index >= 0 && y_index >= 0 && z_index >= 0 && x_index <= nElemX
                       && y_index <= nElemY && z_index <= nElemZ);

  return is_in_bounds;
}


Blob ::Blob () : x (0.0), y (0.0), z (0.0) {}


Blob ::Blob (double x_, double y_, double z_) : x (x_), y (y_), z (z_) {}


std::vector<double> Blob ::RotatePoint (std::vector<double> point) {
  // define sines and cosines
  double c = std::cos (theta);
  double s = std::sin (theta);

  // normalize axis
  rotAxis.normalize ();
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

  Eigen::VectorXd pointNew = Eigen::VectorXd::Zero (3);
  pointNew[0] = point[0] - rotOrigin[0];
  pointNew[1] = point[1] - rotOrigin[1];
  pointNew[2] = point[2] - rotOrigin[2];

  pointNew = rotMat * pointNew + rotOrigin;
  std::vector<double> pointRet = {pointNew[0], pointNew[1], pointNew[2]};

  return pointRet;
}


Cuboid ::Cuboid (double x_, double y_, double z_, double hx_, double hy_, double hz_)
    : Blob (x_, y_, z_), hx (hx_), hy (hy_), hz (hz_) {}


Cuboid ::Cuboid () : Blob (0.0, 0.0, 0.0), hx (0.0), hy (0.0), hz (0.0) {}


double Cuboid ::GetMinDist (std::vector<double> point) {
  if (theta != 0.0) point = RotatePoint (point);

  double xMax = x + hx;
  double yMax = y + hy;
  double zMax = z + hz;

  double xMin = x - hx;
  double yMin = y - hy;
  double zMin = z - hz;

  double minDist = std::min ({point[0] - xMin, xMax - point[0], point[1] - yMin, yMax - point[1],
                              point[2] - zMin, zMax - point[2]});

  return minDist * flipDistanceSign;
}


CutPlane ::CutPlane () : Blob (0.0, 0.0, 0.0), a (1), b (1), c (1), d1 (0), d2 (0) {}


CutPlane ::CutPlane (double a_, double b_, double c_, double d1_, double d2_)
    : a (a_), b (b_), c (c_), d1 (d1_), d2 (d2_) {}


double CutPlane ::GetMinDist (std::vector<double> point) {
  if (theta != 0.0) point = RotatePoint (point);

  double dist1
      = (a * point[0] + b * point[1] + c * point[2] + d1) / std::sqrt (a * a + b * b + c * c);
  double dist2
      = (a * point[0] + b * point[1] + c * point[2] + d2) / std::sqrt (a * a + b * b + c * c);

  return std::min (dist1, -dist2) * flipDistanceSign;
}


Cylinder ::Cylinder () : Blob (0.0, 0.0, 0.0), h (0), r (0), r0 (0), dir (0) {}


Cylinder ::Cylinder (double x_, double y_, double z_, double h_, double r_, double r0_, int dir_)
    : Blob (x_, y_, z_), h (h_), r (r_), r0 (r0_), dir (dir_) {}


double Cylinder ::GetMinDist (std::vector<double> point) {
  if (theta != 0.0) point = RotatePoint (point);

  int dirArray[3] = {0, 0, 0};
  dirArray[dir] = 1;

  double rMin = std::sqrt ((1 - dirArray[0]) * std::pow (point[0] - x, 2)
                           + (1 - dirArray[1]) * std::pow (point[1] - y, 2)
                           + (1 - dirArray[2]) * std::pow (point[2] - z, 2));

  rMin = std::min (r - rMin, rMin - r0);

  double hMin;
  if (dir == 0) hMin = std::min (point[0] - (x - h), (x + h) - point[0]);
  if (dir == 1) hMin = std::min (point[1] - (y - h), (y + h) - point[1]);
  if (dir == 2) hMin = std::min (point[2] - (z - h), (z + h) - point[2]);

  return std::min (rMin, hMin) * flipDistanceSign;
}


void LevelSet3D ::SetIgnoreElementsStress () {
  ignoreElemStress.resize (nCells);
  // #pragma omp parallel for // NOTE: NOT THREAD SAFE
  for (int i = 0; i < nElemX; i++) {
    for (int j = 0; j < nElemY; j++) {
      for (int k = 0; k < nElemZ; k++) {
        // cell counter
        int countCell = i + nElemX * j + nElemX * nElemY * k;
        // define centroid of cell
        std::vector<double> centroid = {i + 0.5, j + 0.5, k + 0.5};

        double maxDist = -1.0e5;  // some -large number
        int holeCount = fixedBlobs.size ();
        for (int hC = 0; hC < holeCount; hC++) {
          double maxDistTemp = fixedBlobs[hC]->GetMinDist (centroid);
          maxDist = std::max (maxDist, maxDistTemp);
        }  // hole count

        if (maxDist >= 0.0) {
          // this means centroid is inside
          // ignore this element
          ignoreElemStress[countCell] = 1;
        } else {
          ignoreElemStress[countCell] = 0;
        }

      }  // k
    }    // j
  }      // i
}


void LevelSet3D ::WriteGradPhi (int iter, std::string file_name, std::string file_path) {
  // Set up file name
  std::ostringstream name, num;
  num.str ("");
  num.width (4);
  num.fill ('0');
  num << std::right << iter;
  name.str ("");
  name << file_path << file_name << num.str () << ".txt";

  // Set up write object
  std::ofstream w_file (name.str ());
  w_file.precision (15);

  // Write gradients to the object
  for (int i = 0; i < nGridPoints; i++) {
    w_file << gradPhi[i] << "\n";
  }
  w_file.close ();
}

}  // namespace para_lsm