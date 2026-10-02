//
// Copyright 2019 H Alicia Kim
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
#include "initialize_lsm.h"
#include "input_output.h"
#include "level_set.h"
#include "mesh.h"

namespace lsm_2d {

LevelSetWrapper ::LevelSetWrapper () {}

LevelSetWrapper ::LevelSetWrapper (InitializeLsm &lsm_init) {
  nelx = lsm_init.nelx;
  nely = lsm_init.nely;
  map_flag = lsm_init.map_flag;
  perturbation = lsm_init.perturbation;

  SetUp (lsm_init);
}


LevelSetWrapper ::~LevelSetWrapper () {
  grid_ptr.reset ();
  level_set_ptr.reset ();
  boundary_ptr.reset ();
  io_ptr.reset ();
}


std::vector<double> LevelSetWrapper ::CalculateElementDensities (bool is_print) {
  if (is_print) std::cout << "Starting CalculateElementDensities... " << std::flush;

  // Compute boundary
  boundary_ptr->discretise (false, 0);
  if (is_print)
    std::cout << "number of boundary points: " << boundary_ptr->nPoints << "... " << std::flush;

  // Compute volume fractions
  double area = boundary_ptr->computeAreaFractions ();
  if (is_print) std::cout << "total volume: " << area << "... " << std::flush;

  // Set up volume fraction vector
  int nelem = nelx * nely;
  std::vector<double> volume_fractions (nelem, 1.0);
  for (int i = 0; i < nelem; i++) {
    volume_fractions[i] = grid_ptr->elements[i].area;
  }

  // Calculate perturbation sensitvities if using discrete adjoint
  if (map_flag == 1) {
    if (is_print) std::cout << "perturbation: " << perturbation << "... " << std::flush;
    boundary_ptr->computePerturbationSensitivities (perturbation);
  }

  if (is_print) std::cout << "finished." << std::endl;

  return volume_fractions;
}


std::vector<double> LevelSetWrapper ::MapSensitivities (std::vector<double> elem_sens,
                                                        bool is_print) {
  int num_bpts = boundary_ptr->nPoints;
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
  int num_bpts = boundary_ptr->nPoints;
  // std::vector<double> vol_sens (num_bpts, 1.0);
  std::vector<double> vol_sens (num_bpts, -1.0);
  // NOTE: a positive movement of a boundary point shrinks the structure. Therefore, the volume
  // sensitivity is negative (matteo)

  // Check mapping type
  if (map_flag == 0) {  // Least squares interpolation
    // Volume sensitivity are all ones; do nothing

  } else if (map_flag == 1) {  // Discrete adjoint interpolation
    // Calculate the sensitivities for each of the boundary points
    for (int i = 0; i < num_bpts; i++) {
      // Initialize the total change in sensitivity due to the perturbed boundary pt
      double delta_sens = 0.0;

      // For all the appropriate elements in the perturbed mesh
      int num_perturb = boundary_ptr->points[i].perturb_indices.size ();
      for (int j = 0; j < num_perturb; j++) {
        // Add the contribution of this element to the total sensitivity
        delta_sens += boundary_ptr->points[i].perturb_sensitivities[j];
      }

      // Calculate the sensitvitity
      // NOTE: Make sure that the volume sensitivity is between -1.5 and -0.5
      vol_sens[i] = -std::min (delta_sens / perturbation, 1.5);
      vol_sens[i] = std::min (vol_sens[i], -0.5);
    }

  } else {
    std::cout << "Unexpected mapping scheme in MapSensitivities."
              << " Check map_flag variable in level set input file." << std::endl;
    exit (EXIT_FAILURE);
  }

  return vol_sens;
}


void LevelSetWrapper ::Update (std::vector<double> bpoint_velocities, bool is_print) {
  if (is_print) std::cout << "Starting Update... ";

  // Assign boundary point velocities to the boundary
  for (int i = 0; i < boundary_ptr->nPoints; i++) {
    boundary_ptr->points[i].velocity = bpoint_velocities[i];
  }

  // Extend boundary points velocities to all narrow band nodes
  level_set_ptr->computeVelocities (boundary_ptr->points);

  // Compute gradient of the level set within the narrow band
  level_set_ptr->computeGradients ();

  // Update the level-set function
  double time_step = 1.0;
  bool is_reinitialized = level_set_ptr->update (time_step);

  // Reinitialize the signed distance function if it hasn't been already
  if (!is_reinitialized) level_set_ptr->reinitialise ();

  if (is_print) std::cout << "finished." << std::endl;
}


double LevelSetWrapper ::GetVolume () { return boundary_ptr->area; }


int LevelSetWrapper ::GetNumberBoundaryPoints () { return boundary_ptr->nPoints; }


void LevelSetWrapper ::GetLimits (std::vector<double> &upper_lim, std::vector<double> &lower_lim) {
  // Size upper and lower limit vectors to number of boundary points
  int num_bpts = boundary_ptr->nPoints;
  upper_lim.resize (num_bpts);
  lower_lim.resize (num_bpts);

  // Get lower and upper limits for each boundary point
  for (int i = 0; i < num_bpts; i++) {
    // NOTE: why are those limits needed? (matteo)
    double min_dist = -1.0e5;  // min distance boundary point is allowed
    double max_dist = 1.0e5;   // max distance boundary point is allowed
    lower_lim[i] = std::max (min_dist, boundary_ptr->points[i].negativeLimit);
    upper_lim[i] = std::min (max_dist, boundary_ptr->points[i].positiveLimit);
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
  int nelem = nelx * nely;
  for (int i = 0; i < nelem; i++) {
    double volume = grid_ptr->elements[i].area;
    file << volume << "\n";
  }
  file.close ();
}


void LevelSetWrapper ::WriteVtk (int curr_iter, std::string file_path, std::string file_name) {
  io_ptr->saveLevelSetVTK (curr_iter, *level_set_ptr, false, false, file_path);
}


void LevelSetWrapper ::WriteBoundaryVtk (bool write_limits) {
  // Create vector of boundary points
  std::vector<std::vector<double>> BoundaryPoints (boundary_ptr->nPoints);
  for (int i = 0; i < boundary_ptr->nPoints; i++) {
    BoundaryPoints[i] = {boundary_ptr->points[i].coord.x, boundary_ptr->points[i].coord.y};
  }

  // Create vector of neighbors
  std::vector<std::vector<unsigned int>> Neighbors (boundary_ptr->nPoints);
  for (int i = 0; i < boundary_ptr->nPoints; i++) {
    Neighbors[i].reserve (boundary_ptr->points[i].nNeighbours);
    for (int j = 0; j < boundary_ptr->points[i].nNeighbours; j++) {
      Neighbors[i].push_back (boundary_ptr->points[i].neighbours[j]);
    }
  }

  // File name
  std::ostringstream fileName;

  // Check flag value
  std::vector<std::vector<double>> BoundaryValues;
  if (write_limits) {  // write boundary points limits
  fileName << "boundary_limits.vtk";
  BoundaryValues.resize(2);
  BoundaryValues[0].resize(boundary_ptr->nPoints);
  BoundaryValues[1].resize(boundary_ptr->nPoints);
    for (int i = 0; i < boundary_ptr->nPoints; i++) {
      BoundaryValues[0][i] = boundary_ptr->points[i].negativeLimit;
      BoundaryValues[1][i] = boundary_ptr->points[i].positiveLimit;
    }
  } else {  // write boundary points limits TODO: change this to write boundary points sensitivities
  fileName << "boundary_sensitivities.vtk";
  BoundaryValues.resize(2);
  BoundaryValues[0].resize(boundary_ptr->nPoints);
  BoundaryValues[1].resize(boundary_ptr->nPoints);
    for (int i = 0; i < boundary_ptr->nPoints; i++) {
      BoundaryValues[0][i] = boundary_ptr->points[i].negativeLimit;
      BoundaryValues[1][i] = boundary_ptr->points[i].positiveLimit;
    }
  }

  // Save file
  io_ptr->BoundaryVTK (fileName, BoundaryPoints, BoundaryValues, Neighbors);
}


void LevelSetWrapper ::SetUp (InitializeLsm &lsm_init) {
  // Create level-set mesh
  bool is_periodic = false;
  grid_ptr = std::make_shared<Mesh> (nelx, nely, is_periodic);

  // Create the vector of initial voids
  int n_init_voids = lsm_init.initialVoids.size ();
  std::vector<Hole> initial_voids (n_init_voids);
  for (int i = 0; i < n_init_voids; i++) {
    initial_voids[i] = *lsm_init.initialVoids[i];
  }

  // Level-set function
  double band_width = 6;
  bool is_fixed_domain = false;  // whether the domain boundary is fixed
  level_set_ptr = std::make_shared<LevelSet> (*grid_ptr, initial_voids, lsm_init.move_limit,
                                              band_width, is_fixed_domain);
  level_set_ptr->reinitialise ();

  // Loop over nondesign voids
  int n_voids = lsm_init.domainVoids.size ();
  for (int i = 0; i < n_voids; i++) {
    if (lsm_init.domainVoids[i]->is_square_hole) {
      level_set_ptr->killNodes ({lsm_init.domainVoids[i]->coord1, lsm_init.domainVoids[i]->coord2});
    } else {
      level_set_ptr->killNodes (lsm_init.domainVoids[i]->coord, lsm_init.domainVoids[i]->r);
    }
  }

  // Loop over nondesign voids
  int n_blobs = lsm_init.fixedBlobs.size ();
  for (int i = 0; i < n_blobs; i++) {
    if (lsm_init.fixedBlobs[i]->is_square_hole) {
      level_set_ptr->fixNodes ({lsm_init.fixedBlobs[i]->coord1, lsm_init.fixedBlobs[i]->coord2});
    } else {
      level_set_ptr->fixNodes (lsm_init.fixedBlobs[i]->coord, lsm_init.fixedBlobs[i]->r);
    }
  }

  // Loop over level-set boundaries: this creates level-set boundaries in the specified region
  int n_boundaries = lsm_init.boundaryBlobs.size ();
  for (int i = 0; i < n_boundaries; i++) {
    level_set_ptr->createLevelSetBoundary (
        {lsm_init.boundaryBlobs[i]->coord1, lsm_init.boundaryBlobs[i]->coord2});
  }

  // Reinitialize only if the number of fixed regions is greater than 0
  if ((n_voids + n_blobs + n_boundaries) > 0) {
    level_set_ptr->reinitialise ();
  }

  // Create boundary
  boundary_ptr = std::make_shared<Boundary> (*level_set_ptr);

  // Set up number of elements in the boundary
  // NOTE: these values are required for the boundary perturbation (matteo)
  boundary_ptr->nelx = level_set_ptr->mesh.width;
  boundary_ptr->nely = level_set_ptr->mesh.height;

  // Create input/output object
  io_ptr = std::make_shared<InputOutput> ();
}


// Helper function for least squares
std::vector<std::vector<int>> LevelSetWrapper ::GetNearbyCellsInfo (std::vector<double> bPoint,
                                                                    int hWidth) {
  // return adjCellIndices of a given boudnary point
  std::vector<std::vector<int>> adjCellIndices;
  adjCellIndices.reserve (std::pow (2 * hWidth + 1, 2));

  int i = int (bPoint[0]);
  int j = int (bPoint[1]);

  for (int iRel = i - hWidth; iRel <= i + hWidth; iRel++) {
    for (int jRel = j - hWidth; jRel <= j + hWidth; jRel++) {
      if (IsCellInBounds (iRel, jRel)) {
        // cell counter
        int counter_cell = iRel + boundary_ptr->nelx * jRel;
        adjCellIndices.push_back ({iRel, jRel, counter_cell});
      }
    }  // j
  }    // i

  return adjCellIndices;
}


// check if an index of a cell is in bound
bool LevelSetWrapper ::IsCellInBounds (int i, int j) {
  bool isInBounds = (i >= 0 && j >= 0 && i < boundary_ptr->nelx && j < boundary_ptr->nely);

  return isInBounds;
}


void LevelSetWrapper ::LeastSquareInterpolateBoundarySensitivities (
    std::vector<double> &boundary_sens, std::vector<double> element_sens, int half_width,
    int weighted_vol_frac) {
  int num_bpts = boundary_ptr->nPoints;
  // #pragma omp parallel for
  for (int i = 0; i < num_bpts; i++) {
    // Get neighboring cells
    std::vector<double> curr_bpoint
        = {boundary_ptr->points[i].coord.x, boundary_ptr->points[i].coord.y};
    std::vector<std::vector<int>> adj_cell_indicies = GetNearbyCellsInfo (curr_bpoint, half_width);

    // Initialize least squares interpolation matrices
    int num_samples = adj_cell_indicies.size ();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero (num_samples, 3);
    Eigen::VectorXd B = Eigen::VectorXd::Zero (num_samples);

    // Set up least squares interpolation matrices
    for (int j = 0; j < num_samples; j++) {
      // Calculate x, y distance of boundary point to cell centroid
      double x = (-curr_bpoint[0] + adj_cell_indicies[j][0]) + 0.5;
      double y = (-curr_bpoint[1] + adj_cell_indicies[j][1]) + 0.5;
      double dist = std::sqrt (x * x + y * y);

      // Assign weight based on distance and volume fraction of the element
      double weight = std::max (0.001, 3.0 - dist);
      weight = std::min (1.5, weight);
      int elem_index = adj_cell_indicies[j][2];
      weight *= (weighted_vol_frac * grid_ptr->elements[elem_index].area)
                + ((1 - weighted_vol_frac) * 1.0);

      // Assign values for interpolation matrices
      A (j, 0) = weight;
      A (j, 1) = weight * x;
      A (j, 2) = weight * y;
      B (j) = (weight * element_sens[elem_index])
              * (weighted_vol_frac * grid_ptr->elements[elem_index].area
                 + double (1 - weighted_vol_frac));
    }

    // Solve least squares interpolation
    Eigen::VectorXd X = A.colPivHouseholderQr ().solve (B);
    // if (!std::isnan (X[0])) boundary_sens[i] = X[0];
    if (!std::isnan (X[0])) boundary_sens[i] = -X[0];
    // NOTE: a positive movement of a boundary point shrinks the structure. Therefore, the volume
    // sensitivity is negative (matteo)
  }
}


void LevelSetWrapper ::DiscreteAdjointBoundarySensitivities (std::vector<double> &boundary_sens,
                                                             std::vector<double> element_sens) {
  int num_bpts = boundary_ptr->nPoints;

  // Calculate the sensitivities for each of the boundary points
  for (int i = 0; i < num_bpts; i++) {
    // Initialize the total change in sensitivity due to the perturbed boundary pt
    double delta_sens = 0.0;

    // For all the appropriate elements in the perturbed mesh
    int num_perturb = boundary_ptr->points[i].perturb_indices.size ();
    for (int j = 0; j < num_perturb; j++) {
      // Index of the perturbed element
      int curr_index = boundary_ptr->points[i].perturb_indices[j];

      // Add the contribution of this element to the total sensitivity
      delta_sens += element_sens[curr_index] * boundary_ptr->points[i].perturb_sensitivities[j];
    }

    // Calculate the sensitvitity
    // NOTE: the convention for this implementation of the level-set method is that a positive
    // movement of a boundary points is directed towards the inside of the domain. For this reason,
    // the perturbation sensitivity is negative (the domain shrinks). However, the values stored in
    // perturb_sensitivities represent the absolute volume variation due to the perturbation.
    // Therefore, we use a minus sign to get the correct sensitivity. (matteo)
    boundary_sens[i] = -delta_sens / perturbation;
  }
}


}  // namespace lsm_2d