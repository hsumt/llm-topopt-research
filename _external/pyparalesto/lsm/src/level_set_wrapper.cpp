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

#include "level_set_wrapper.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "boundary.h"
#include "eigen3/Eigen/Dense"
#include "grid_math.h"
#include "lsm_3d.h"

namespace para_lsm {

LevelSetWrapper ::LevelSetWrapper () {}


LevelSetWrapper ::LevelSetWrapper (InitializeLsm &lsm_init) {
  nelx = lsm_init.nelx;
  nely = lsm_init.nely;
  nelz = lsm_init.nelz;
  map_flag = lsm_init.map_flag;
  perturbation = lsm_init.perturbation;

  SetUp (lsm_init);
}


LevelSetWrapper ::~LevelSetWrapper () {
  grid_ptr.reset ();
  level_set_ptr.reset ();
  boundary_ptr.reset ();
}


std::vector<double> LevelSetWrapper ::CalculateElementDensities (bool is_print) {
  if (is_print) std::cout << "Starting CalculateElementDensities... ";

  // Compute boundary
  boundary_ptr->MarchCubes ();

  // Compute volume fractions
  level_set_ptr->ComputeVolumeFractions ();

  // Set up volume fraction vector
  int nelem = nelx * nely * nelz;
  std::vector<double> volume_fractions (nelem, 1.0);
  for (int i = 0; i < nelem; i++) {
    volume_fractions[i] = std::max (level_set_ptr->volFractions[i], 0.0);
  }

  // Calculate perturbation sensitvities if using discrete adjoint
  if (map_flag == 1) {
    boundary_ptr->ComputePerturbationSensitivities (perturbation);
  }

  if (is_print) std::cout << "finished." << std::endl;

  return volume_fractions;
}


std::vector<bool> LevelSetWrapper ::GetFixedElementsForStress () {
  level_set_ptr->SetIgnoreElementsStress ();

  return level_set_ptr->ignoreElemStress;
}


std::vector<double> LevelSetWrapper ::MapSensitivities (std::vector<double> elem_sens,
                                                        bool is_print) {
  int num_bpts = boundary_ptr->numTriangles;
  std::vector<double> boundary_sens (num_bpts, 0.0);
  if (is_print) std::cout << "Starting MapSensitivities ";

  if (map_flag == 0) {
    if (is_print) std::cout << "using least squares... ";
    LeastSquareInterpolateBoundarySensitivities (boundary_sens, elem_sens);
  } else if (map_flag == 1) {
    if (is_print) std::cout << "using discrete adjoint... ";
    DiscreteAdjointBoundarySensitivities (boundary_sens, elem_sens);
  } else {
    std::cout << "Unexpected mapping scheme in MapSensitivities."
              << " Check map_flag variable in level set input file." << std::endl;
    exit (EXIT_FAILURE);
  }

  if (is_print) std::cout << "finished." << std::endl;

  return boundary_sens;
}


std::vector<double> LevelSetWrapper ::MapVolumeSensitivities () {
  // Create a vector of ones for boundary point volume sensitivity
  int num_bpts = boundary_ptr->numTriangles;
  std::vector<double> vol_sens (num_bpts, 1.0);

  // Check mapping type
  if (map_flag == 0) {  // Least squares interpolation
    // Volume sensitivity are all ones; do nothing

  } else if (map_flag == 1) {  // Discrete adjoint interpolation
    // Calculate the sensitivities for each of the boundary points
    for (int i = 0; i < num_bpts; i++) {
      // Initialize the total change in sensitivity due to the perturbed boundary pt
      double delta_sens = 0.0;

      // For all the appropriate elements in the perturbed mesh
      int num_perturb = boundary_ptr->perturb_indicies_values[i].size ();
      for (int j = 0; j < num_perturb; j++) {
        // Add the contribution of this element to the total sensitivity
        delta_sens += boundary_ptr->perturb_indicies_values[i][j].value;
      }

      // Calculate the sensitvitity
      vol_sens[i] = std::min (delta_sens / perturbation, 1.5);
      vol_sens[i] = std::max (vol_sens[i], -0.5);
    }

  } else {
    std::cout << "Unexpected mapping scheme in MapSensitivities."
              << " Check map_flag variable in level set input file." << std::endl;
    exit (EXIT_FAILURE);
  }

  return vol_sens;
}


void LevelSetWrapper ::Update (std::vector<double> bpoint_velocities, double move_limit,
                               bool is_print) {
  if (is_print) std::cout << "Starting Update... ";

  // Assign boundary point velocities to the boundary
  boundary_ptr->opt_vel = bpoint_velocities;

  // Update boundary
  boundary_ptr->ExtrapolateVelocities ();
  level_set_ptr->Update ();
  level_set_ptr->MakeDomainHoles ();

  if (is_print) std::cout << "finished." << std::endl;
}


double LevelSetWrapper ::GetVolume () { return level_set_ptr->volume; }


int LevelSetWrapper ::GetNumberBoundaryPoints () { return boundary_ptr->numTriangles; }


void LevelSetWrapper ::GetLimits (std::vector<double> &upper_lim, std::vector<double> &lower_lim,
                                  double move_limit) {
  // Size upper and lower limit vectors to number of boundary points
  int num_bpts = boundary_ptr->numTriangles;
  upper_lim.resize (num_bpts);
  lower_lim.resize (num_bpts);

  // Get lower and upper limits for each boundary point
  for (int i = 0; i < num_bpts; i++) {
    double min_dist = -1.0e5;  // min distance boundary point is allowed
    double max_dist = 1.0e5;   // max distance boundary point is allowed
    std::vector<double> curr_bpoint = boundary_ptr->bPoints[i];
    boundary_ptr->DomainDistance (curr_bpoint, min_dist, max_dist);
    boundary_ptr->FixLocDistance (curr_bpoint, min_dist, max_dist);
    lower_lim[i] = std::max (min_dist, -move_limit);
    upper_lim[i] = std::min (max_dist, move_limit);
    upper_lim[i] = std::max (upper_lim[i], lower_lim[i]);
    upper_lim[i] = std::min (upper_lim[i], move_limit);
  }
}


void LevelSetWrapper ::WriteElementDensitiesToTxt (int iter, std::string file_path,
                                                   std::string file_name) {
  // Set up file name
  std::ostringstream path, num;
  num.str ("");
  num.width (4);
  num.fill ('0');
  num << std::right << iter;
  path.str ("");
  if (file_path.empty ())
    path << file_name << num.str () << ".txt";
  else
    path << file_path << "/" << file_name << num.str () << ".txt";

  // Set up text file
  std::ofstream file (path.str ());
  typedef std::numeric_limits<double> double_limit;
  file.precision (double_limit::max_digits10);

  // Pass element densities to text file
  int nelem = nelx * nely * nelz;
  for (int i = 0; i < nelem; i++) {
    double volume = std::max (level_set_ptr->volFractions[i], 0.0);
    file << volume << "\n";
  }
  file.close ();
}


void LevelSetWrapper ::WriteStl (int curr_iter, std::string file_path, std::string file_name) {
  std::ostringstream path, num;
  num.str ("");
  num.width (4);
  num.fill ('0');
  num << std::right << curr_iter;
  path.str ("");
  if (file_path.empty ())
    path << file_name << num.str () << ".stl";
  else
    path << file_path << "/" << file_name << num.str () << ".stl";

  boundary_ptr->WriteSTL (path.str ());
}


void LevelSetWrapper ::SetUp (InitializeLsm &lsm_init) {
  // Create objects for pointers
  grid_ptr = std::make_unique<Grid> (nelx, nely, nelz);
  level_set_ptr = std::make_unique<LevelSet3D> (*grid_ptr);
  boundary_ptr = std::make_unique<Boundary> (*level_set_ptr);

  // Set up level set object
  level_set_ptr->fixedBlobs = std::move (lsm_init.fixedBlobs);
  level_set_ptr->domainVoids = std::move (lsm_init.domainVoids);
  level_set_ptr->initialVoids = std::move (lsm_init.initialVoids);

  // Set up dimensionality for level set object
  level_set_ptr->hWidth = 4;  // narrowband half-width
  level_set_ptr->MakeBox ();  // initialize signed distance function
  level_set_ptr->volFractions.resize (level_set_ptr->nCells, 1.0);

  // Ignore elements inside fixed blobs--for stress calculation
  level_set_ptr->SetIgnoreElementsStress ();
}


void LevelSetWrapper ::LeastSquareInterpolateBoundarySensitivities (
    std::vector<double> &boundary_sens, std::vector<double> element_sens, int half_width,
    int weighted_vol_frac) {
  int num_bpts = boundary_ptr->numTriangles;
  // #pragma omp parallel for
  for (int i = 0; i < num_bpts; i++) {
    // Get neighboring cells
    std::vector<double> curr_bpoint = boundary_ptr->bPoints[i];
    std::vector<std::vector<int> > adj_cell_indicies
        = boundary_ptr->GetNearbyCellsInfo (curr_bpoint, half_width);

    // Initialize least squares interpolation matrices
    int num_samples = adj_cell_indicies.size ();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero (num_samples, 4);
    Eigen::VectorXd B = Eigen::VectorXd::Zero (num_samples);

    // Set up least squares interpolation matrices
    for (int j = 0; j < num_samples; j++) {
      // Calculate x, y, z distance of boundary point to cell centroid
      double x = (-curr_bpoint[0] + adj_cell_indicies[j][0]) + 0.5;
      double y = (-curr_bpoint[1] + adj_cell_indicies[j][1]) + 0.5;
      double z = (-curr_bpoint[2] + adj_cell_indicies[j][2]) + 0.5;
      double dist = std::sqrt ((x * x + y * y) + z * z);

      // Assign weight based on distance and volume fraction of the element
      double weight = std::max (0.001, 3.0 - dist);
      weight = std::min (1.5, weight);
      int elem_index = adj_cell_indicies[j][3];
      weight *= (weighted_vol_frac * level_set_ptr->volFractions[elem_index])
                + ((1 - weighted_vol_frac) * 1.0);

      // Assign values for interpolation matrices
      A (j, 0) = weight;
      A (j, 1) = weight * x;
      A (j, 2) = weight * y;
      A (j, 3) = weight * z;
      B (j) = (weight * element_sens[elem_index])
              * (weighted_vol_frac * level_set_ptr->volFractions[elem_index]
                 + double (1 - weighted_vol_frac));
    }

    // Solve least squares interpolation
    Eigen::VectorXd X = A.colPivHouseholderQr ().solve (B);
    if (!std::isnan (X[0])) boundary_sens[i] = X[0];
  }
}


void LevelSetWrapper ::DiscreteAdjointBoundarySensitivities (std::vector<double> &boundary_sens,
                                                             std::vector<double> element_sens) {
  int num_bpts = boundary_ptr->numTriangles;

  // Calculate the sensitivities for each of the boundary points
  for (int i = 0; i < num_bpts; i++) {
    // Initialize the total change in sensitivity due to the perturbed boundary pt
    double delta_sens = 0.0;

    // For all the appropriate elements in the perturbed mesh
    int num_perturb = boundary_ptr->perturb_indicies_values[i].size ();
    for (int j = 0; j < num_perturb; j++) {
      // Index of the perturbed element
      int curr_index = boundary_ptr->perturb_indicies_values[i][j].index;

      // Add the contribution of this element to the total sensitivity
      delta_sens += element_sens[curr_index] * boundary_ptr->perturb_indicies_values[i][j].value;
    }

    // Calculate the sensitvitity
    boundary_sens[i] = delta_sens / perturbation;
  }
}


}  // namespace para_lsm