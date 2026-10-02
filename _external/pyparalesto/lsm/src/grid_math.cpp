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

#include "grid_math.h"

namespace para_lsm {

GridVector ::GridVector (double x_, double y_, double z_) : x (x_), y (y_), z (z_) {}


GridVector ::GridVector () : x (0), y (0), z (0) {}


GridVector ::GridVector (const Eigen::Vector3d &Vector3d) {
  x = Vector3d[0];
  y = Vector3d[1];
  z = Vector3d[2];
}


GridVector ::operator Eigen::Vector3d () const {
  Eigen::Vector3d Vector3d;
  Vector3d (0) = x;
  Vector3d (1) = y;
  Vector3d (2) = z;
  return Vector3d;
}


Grid4Vector ::Grid4Vector () : GridVector (), val (0) {}


Grid4Vector ::Grid4Vector (double x_, double y_, double z_, double val_)
    : GridVector (x_, y_, z_), val (val_) {}


Grid4Vector ::operator GridVector () const { return GridVector (x, y, z); }


// constructor
Grid ::Grid (int nelx_, int nely_, int nelz_) : nelx (nelx_), nely (nely_), nelz (nelz_) {
  phi.resize ((nelx + 1) * (nely + 1) * (nelz + 1));
#pragma omp parallel for
  for (int i = 0; i <= nelx; i++) {
    for (int j = 0; j <= nely; j++) {
      for (int k = 0; k <= nelz; k++) {
        int current_index = GridPtToIndex (i, j, k);
        // Assign the current coordinate
        phi[current_index].x = i;
        phi[current_index].y = j;
        phi[current_index].z = k;
        // default to 0
        phi[current_index].val = 0.0;
      }
    }
  }
}


int Grid ::GridPtToIndex (int x, int y, int z) {
  // returns the index of a grid point in the zyx system (Grid4Vector convention)
  return z + y * (nelz + 1) + x * (nelz + 1) * (nely + 1);
}


Stencil ::Stencil (Grid &grid, int iLoc_, int jLoc_, int kLoc_)
    : grid (grid), iLoc (iLoc_), jLoc (jLoc_), kLoc (kLoc_) {
  iMax = grid.nelx;
  jMax = grid.nely;
  kMax = grid.nelz;
}


bool Stencil ::IsInBounds (int iRel, int jRel, int kRel) {
  bool is_in_bounds = (iLoc + iRel >= 0 && jLoc + jRel >= 0 && kLoc + kRel >= 0
                       && iLoc + iRel <= iMax && jLoc + jRel <= jMax && kLoc + kRel <= kMax);

  return is_in_bounds;
}


void Stencil ::CapIndices (int &iRel, int &jRel, int &kRel) {
  if (iLoc + iRel < 0) iRel += -(iLoc + iRel);
  if (iLoc + iRel > iMax) iRel -= iLoc + iRel - iMax;

  if (jLoc + jRel < 0) jRel += -(jLoc + jRel);
  if (jLoc + jRel > jMax) jRel -= jLoc + jRel - jMax;

  if (kLoc + kRel < 0) kRel += -(kLoc + kRel);
  if (kLoc + kRel > kMax) kRel -= kLoc + kRel - kMax;
}


double Stencil ::GetValue (int iRel, int jRel, int kRel) {
  int phi_index = grid.GridPtToIndex (iLoc + iRel, jLoc + jRel, kLoc + kRel);
  double return_value = grid.phi[phi_index].val;

  return return_value;
}


double Stencil ::GetValue (std::vector<double> &phi_other, int iRel, int jRel, int kRel) {
  int phi_index = grid.GridPtToIndex (iLoc + iRel, jLoc + jRel, kLoc + kRel);
  double return_value = phi_other[phi_index];

  return return_value;
}


MinStencil ::MinStencil (Grid &grid, int iLoc_, int jLoc_, int kLoc_, int halfWidth_,
                         bool ignoreSelf_)
    : Stencil (grid, iLoc_, jLoc_, kLoc_), halfWidth (halfWidth_), ignoreSelf (ignoreSelf_) {}


double MinStencil ::MinValue (int direction) {
  double min_value = large_number;
  if (direction == 0) {
    dirArray[0] = 1;
    dirArray[1] = 0;
    dirArray[2] = 0;

  } else if (direction == 1) {
    dirArray[0] = 0;
    dirArray[1] = 1;
    dirArray[2] = 0;

  } else if (direction == 2) {
    dirArray[0] = 0;
    dirArray[1] = 0;
    dirArray[2] = 1;
  }

  // return the minimum value along the direction
  for (int width = -halfWidth; width <= halfWidth; width++) {
    if (!ignoreSelf || width != 0) {
      int i = dirArray[0] * width;
      int j = dirArray[1] * width;
      int k = dirArray[2] * width;

      if (IsInBounds (i, j, k)) {
        min_value = std::min (min_value, GetValue (i, j, k));
      }
    }
  }

  return min_value;
}


HJWENOStencil ::HJWENOStencil (Grid &grid, int iLoc_, int jLoc_, int kLoc_)
    : Stencil (grid, iLoc_, jLoc_, kLoc_) {}


double HJWENOStencil ::GradValue (int direction, bool is_positive) {
  std::vector<double> v (5, 0);
  if (direction == 0) {
    dirArray[0] = 1;
    dirArray[1] = 0;
    dirArray[2] = 0;
  }
  if (direction == 1) {
    dirArray[0] = 0;
    dirArray[1] = 1;
    dirArray[2] = 0;
  }
  if (direction == 2) {
    dirArray[0] = 0;
    dirArray[1] = 0;
    dirArray[2] = 1;
  }

  int v_index = 0;
  for (int width = -3 + is_positive; width < 2 + is_positive; width++) {
    int i0 = dirArray[0] * width;
    int i1 = dirArray[0] * (width + 1);
    int j0 = dirArray[1] * width;
    int j1 = dirArray[1] * (width + 1);
    int k0 = dirArray[2] * width;
    int k1 = dirArray[2] * (width + 1);

    // cap values
    CapIndices (i1, j1, k1);
    CapIndices (i0, j0, k0);

    if (i0 == i1 && j0 == j1 && k0 == k1) {
      i0 = i1 - dirArray[0];
      j0 = j1 - dirArray[1];
      k0 = k1 - dirArray[2];
      CapIndices (i0, j0, k0);
      if (i0 == i1 && j0 == j1 && k0 == k1) {
        i1 = i0 + dirArray[0];
        j1 = j0 + dirArray[1];
        k1 = k0 + dirArray[2];
      }
    }

    double delta_phi = GetValue (i1, j1, k1) - GetValue (i0, j0, k0);
    if (is_positive)
      v[4 - v_index] = delta_phi;
    else
      v[v_index] = delta_phi;
    v_index++;
  }

  double grad = InterpolateGrad (v);
  return grad;
}


// Approximate the gradient using the 5th order Hamilton-Jacobi WENO approximation.
// Taken from pages 34-35 of "Level Set Methods and Dynamic Implicit Surfaces".
// See: http://web.stanford.edu/class/cs237c/Lecture16.pdf
double HJWENOStencil ::InterpolateGrad (std::vector<double> v) {
  double v1 = v[0];
  double v2 = v[1];
  double v3 = v[2];
  double v4 = v[3];
  double v5 = v[4];
  double oneQuarter = 1.0 / 4.0;
  double thirteenTwelths = 13.0 / 12.0;
  double eps = 1e-6;
  // Estimate the smoothness of each stencil
  double s1 = thirteenTwelths * (v1 - 2 * v2 + v3) * (v1 - 2 * v2 + v3)
              + oneQuarter * (v1 - 4 * v2 + 3 * v3) * (v1 - 4 * v2 + 3 * v3);
  double s2 = thirteenTwelths * (v2 - 2 * v3 + v4) * (v2 - 2 * v3 + v4)
              + oneQuarter * (v2 - v4) * (v2 - v4);
  double s3 = thirteenTwelths * (v3 - 2 * v4 + v5) * (v3 - 2 * v4 + v5)
              + oneQuarter * (3 * v3 - 4 * v4 + v5) * (3 * v3 - 4 * v4 + v5);
  // Compute the alpha values for each stencil
  double alpha1 = 0.1 / ((s1 + eps) * (s1 + eps));
  double alpha2 = 0.6 / ((s2 + eps) * (s2 + eps));
  double alpha3 = 0.3 / ((s3 + eps) * (s3 + eps));
  // Calculate the normalised weights
  double totalWeight = alpha1 + alpha2 + alpha3;
  double w1 = alpha1 / totalWeight;
  double w2 = alpha2 / totalWeight;
  double w3 = alpha3 / totalWeight;
  // Sum the three stencil components.
  double grad = w1 * (2 * v1 - 7 * v2 + 11 * v3) + w2 * (5 * v3 - v2 + 2 * v4)
                + w3 * (2 * v3 + 5 * v4 - v5);
  grad *= (1.0 / 6.0);
  return grad;
}


VelGradStencil ::VelGradStencil (Grid &grid, int iLoc_, int jLoc_, int kLoc_, int halfWidth_)
    : Stencil (grid, iLoc_, jLoc_, kLoc_), halfWidth (halfWidth_) {}


void VelGradStencil ::GetVelGrad (std::vector<double> &velGrid, int direction, double &grad,
                                  double &weight) {
  double min_value = large_number;
  if (direction == 0) {
    dirArray[0] = 1;
    dirArray[1] = 0;
    dirArray[2] = 0;
  } else if (direction == 1) {
    dirArray[0] = 0;
    dirArray[1] = 1;
    dirArray[2] = 0;
  } else if (direction == 2) {
    dirArray[0] = 0;
    dirArray[1] = 0;
    dirArray[2] = 1;
  }

  // return the minimum value along the direction
  for (int width = -halfWidth; width <= halfWidth; width++) {
    if (width != 0) {
      int i = dirArray[0] * width;
      int j = dirArray[1] * width;
      int k = dirArray[2] * width;

      if (IsInBounds (i, j, k)) {
        min_value = std::min (min_value, GetValue (i, j, k));
        if (min_value == GetValue (i, j, k)) {
          // use this location for gradients
          grad = GetValue () - GetValue (i, j, k);
          weight = GetValue (velGrid, i, j, k);
        }  // if min val
      }    // is in boudns
    }      // if width
  }        // width loop
}  // end of function

}  // namespace para_lsm