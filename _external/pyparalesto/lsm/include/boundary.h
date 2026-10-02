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

#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <vector>

#include "eigen3/Eigen/Dense"
#include "grid_math.h"
#include "lsm_3d.h"
#include "mc_table.h"
#include "optimize.h"

/*! \file boundary.h
    \brief A class for handling the boundary
*/

namespace para_lsm {

/*! \struct Triangle
    \brief Struct for storing triangles of the surface mesh
*/
struct Triangle {
  GridVector p[3];  //!< vector of vertices
};


/*! \struct PerturbProperties
    \brief Struct containing the index and sensitivity of an element
*/
struct PerturbProperties {
  int index;     //!< index of the element
  double value;  //!< sensitivity of element density wrt a bpt perturbation
};

/*! \class Boundary
    \brief A class that handles the boundary for a given level set function
*/
class Boundary {
 public:
  //! Constructor
  /*! \param &levelSet_
        A reference to the level set function
  */
  Boundary (LevelSet3D &levelSet_);

  //! Creates surface mesh using marching cubes algorithm
  /*! \param isoValue_
        The iso surface value
  */
  void MarchCubes (double isoValue_ = 0.0);

  //! Optimize the boundary velocities for a single (volume) constraint
  /*! \param &bsens
        Reference to the boundary sensitivities

      \param moveLimit
        Maximum allowable limit for the design variable

      \param volume
        volume of the topology

      \param volCons
        volume constraint

      \param algo
        Optimization algorithm. 0 for Newton Raphson, 1 for MMA (Method of
        Moving Asymptotes) using the NLopt library, 2 for Simplex. Default is
        0, Newton Raphson.
  */
  void Optimize (std::vector<double> &bsens, double moveLimit, double volume, double volCons,
                 int algo = 0);

  //! Optimize the boundary velocities for multiple constraints
  /*! \param objSens
        Reference to the boundary objective sensitivities

      \param conSens
        Reference to the boundary constraint sensitivities

      \param conMaxVals
        Maximum values for the constraints

      \param conCurVals
        Current values of the constraints

      \param moveLimit
        Maximum allowable limit for the design variable

      \param algo
        Optimization algorithm. 0 for Newton Raphson, 1 for MMA (Method of
        Moving Asymptotes) using the NLopt library, 2 for Simplex.
  */
  void Optimize (std::vector<double> &objSens, std::vector<std::vector<double>> &conSens,
                 std::vector<double> conMaxVals, std::vector<double> conCurVals, double moveLimit,
                 int algo);

  //! Extrapolates velocities from boundary points to grid points
  void ExtrapolateVelocities ();

  //! Interpolate boundary sensitivitites using least squares
  /*! \param &bsens
        Reference to vector of boundary sensitivities

      \param sensiArray
        \todo TODO(carolina): figure out what this parameter is

      \param &volFracToPetscMap
        Vector of indexes for element volume fractions as they're stored by
        using PETSc

      \param hWidth
        Half-width of the cube volume, centered at the boundary point. Used to
        select elements that are close enough to the boundary point to be
        considered in the least squares interpolation.

      \param weightedVolFrac
        \todo TODO(carolina): figure out what this parameter is

      \param usePetscMap
        Use the mapping used by PETSc if true. True by default.
  */
  void InterpolateBoundarySensitivities (std::vector<double> &bsens, double *sensiArray,
                                         std::vector<int> &volFracToPetscMap, int hWidth,
                                         int weightedVolFrac = 1, bool usePetscMap = 1);

  //! Interpolate boundary sensitivitites using the perturbation method
  /*! \param &bsens
        Reference to vector of boundary sensitivities

      \param sensiArray
        \todo TODO(carolina): figure out what this parameter is

      \param &volFracToPetscMap
        Vector of indexes for element volume fractions as they're stored by
        using PETSc

      \param hWidth
        Half-width of the cube volume, centered at the boundary point. Used to
        select elements that are close enough to the boundary point to be
        considered in the least squares interpolation.

      \param usePetscMap
        Use the mapping used by PETSc if true. True by default.
  */
  // void InterpolateBoundarySensitivities (std::vector<double> &bsens,
  //   double *sensiArray, std::vector<int> &volFracToPetscMap,
  //   double perturbation, bool usePetscMap = 1) ;

  //! Compute boundary sensitivitites
  /*! \param &bsens
        Reference to vector of boundary sensitivities

      \param sensiArray
        \todo TODO(carolina): figure out what this parameter is

      \param &volFracToPetscMap
        Vector of indexes for element volume fractions as they're stored by
        using PETSc

      \param usePetscMap
        Use the mapping used by PETSc if true. True by default.
  */
  void ComputeAnalyticalSensitivities (std::vector<double> &bsens, double *sensiArray,
                                       std::vector<int> &volFracToPetscMap, bool usePetscMap = 1);

  //! Helper function for least squares
  /*! \param bPoint
        Coordinates of the boundary point.

      \param hWidth
        Half-width of the cube volume, centered at a boundary point. If a cell
        (element) is within the volume, it is considered nearby.

      \return
        A vector that contains the relative distances (x,y,z) and index of
        the nearby cells (elements), stored in that order. Distances are
        relative to the boundary point.
  */
  std::vector<std::vector<int>> GetNearbyCellsInfo (std::vector<double> bPoint, int hWidth);

  //! Check if an index of a cell is in bound
  /*! \param i
        Index of the cell (element) in the i-th (x) direction.

      \param j
        Index of the cell (element) in the j-th (y) direction.

      \param k
        Index of the cell (element) in the j-th (z) direction.

      \return
        True if the cell is in the bounds of the grid, False otherwise.
  */
  bool IsCellInBounds (int i, int j, int k);

  //! Compute the shortest distance from the point to fixed surfaces
  /*! \param bPoint
        Boundary point coordinates

      \param minDist
        Reference to the minimum (negative) distance from the point to a fixed
        surface.

      \param maxDist
        Reference to the minimum (positive) distance from the point to a fixed
        surface.
  */
  void FixLocDistance (std::vector<double> bPoint, double &minDist, double &maxDist);

  //! Compute the shortest distance from the point to domain boundaries
  /*! \param bPoint
        Boundary point coordinates

      \param minDist
        Reference to the minimum (negative) distance from the point to a
        domain boundary.

      \param maxDist
        Reference to the minimum (positive) distance from the point to a
        domain boundary.
  */
  void DomainDistance (std::vector<double> bPoint, double &minDist, double &maxDist);

  // Added by Carolina 2020 October 21
  //! Calculate the sensitivity of element density with respect to a boundary
  //! point perturbation
  /*! See Kambampati et al (2021) "A discrete adjoint based level set topology
      optimization method for stress constraints" for explanation of
      sensitivity calculation via perturbing the level set (specifically Sec
      3.2.1). https://doi.org/10.1016/j.cma.2020.113563

      \param perturb
        Size of the perturbation to a boundary point to calculate the
        sensitivity

      \warning This method is still in-progress and does not work properly
  */
  void ComputePerturbationSensitivities (double perturb);


  /*! \name Write output files*/
  ///\{

  //! Write design to stl
  /*! \param file_name
        Name of the stl file
  */
  void WriteSTL (std::string file_name = "mystlfile.stl");

  //! Write patch
  /*! Write surface in a file format so that it can be visualized using MATLAB

      \param file_name
        Name of the stl file
  */
  void WritePatch (std::string file_name = "patchfile.txt");
  ///\}

  //! A reference to the level set function
  LevelSet3D &levelSet;

  //! Number of triangles for the surface mesh.
  /*! Also equivalent to the number of boundary points.*/
  int numTriangles;

  //! Vector of triangles for the triangular surface mesh
  std::vector<Triangle> triangles;

  //! Indices list of triangles to cells
  /*! Size of outer vector is the number of grid cells (elements)
      (nelx*nely*nelz) and the size of the inner vector is the number of
      triangles that lie inside the cell. The elements of the inner vector are
      the indicies of the triangles as listed in the vector triangles.
  */
  std::vector<std::vector<int>> triCellIndices;

  //! Vector of boundary point coordinates
  std::vector<std::vector<double>> bPoints;

  //! Vector of boundary areas for each boundary point
  std::vector<double> bAreas;

  //! Vector of optimum velocities for each boundary point
  std::vector<double> opt_vel;

  //! List of indicies of the cells (elements of the grid) that are perturbed
  //! by each boundary point
  /*! Size of outer vector corresponds to the number of boundary points. Size
      of the inner vector is the number of cells perturbed by the boundary
      point. The elements of the inner vector are the indicies of the cells.*/
  std::vector<std::vector<int>> perturb_indicies;

  //! Sensitivities of the perturbed cells (elements of the grid)
  /*! Size of outer vector corresponds to the number of boundary points. Size
      of the inner vector is the number of cells perturbed by the boundary
      point. The elements of the inner vector are the sensitivities of the
      cells.*/
  std::vector<std::vector<double>> perturb_sensitivities;

  //! Perturbation sensitivites (sensitivity of elem density wrt bpt perturbation)
  /*! Size of outer vector is the number of boundary points. Size of inner
      vector is the number of elements affected by the particular boundary
      point being perturbed. PerturbProperties contains the index of the
      affected element and its sensitivity value.
  */
  std::vector<std::vector<PerturbProperties>> perturb_indicies_values;

 private:
  //! Interpolation function for computing the boundary. Helper function for
  //! MarchCubes.
  /*! Given two points (p4vec1 and p4vec2), this function will find the
      coordinate between the two points, whose level set (signed distance)
      value corresponds to the given parameter.

      \param p4vec1
        Grid4Vector for node 1

      \param p4vec2
        Grid4Vector for node 2

      \param value
        Level set function value for the interpolation.

      \return
        The coordinates of the point between p4vec1 and p4vec that has a level
        set function value that corresponds to the parameter value.
  */
  GridVector LinearInterp (Grid4Vector p4vec1, Grid4Vector p4vec2, double value);

  //! Helper function for MarchCubes.
  /*! Interpolates the vertices of the iso-surface for a single cell (element).

      \param cubeIndex
        Index for the Marching Cubes edgeTable that corresponds to the cell
        (element) of interest.

      \param verts
        Vector of vertices (coordinates and level set function values) of the
        cell (element) of interest.

      \return
        Vector of coordinates for the vertices of the iso-surface.
  */
  std::vector<GridVector> GetInterpolatedVertices (int cubeIndex, std::vector<Grid4Vector> verts);

  //! Helper function for MarchCubes. Gets the vertices of a cell (element).
  /*! \param ni
        Index of cell along the x direction

      \param nj
        Index of cell along the y direction

      \param k
        Index of cell along the z direction

      \param &grid
        Reference to the grid
  */
  std::vector<Grid4Vector> GetVertices (int ni, int nj, int k, Grid &grid);

  //! Helper function for ExtrapolateVelocities
  /*! \param &weight
        Reference to a vector of weights for each grid point.

      \param &weightedvel
        Reference to a vector of weighted velocity values for each grid point.
  */
  void ExtrapolateWeightedVelocities (std::vector<double> &weight,
                                      std::vector<double> &weightedvel);

  //! Compute midpoints of triangles
  void ComputeMidPoints ();

  int ncellsX;      //!< Number of cells (elements) in the x direction.
  int ncellsY;      //!< Number of cells (elements) in the y direction.
  int ncellsZ;      //!< Number of cells (elements) in the z direction.
  double isoValue;  //!< Value for the isosurface. Zero by default.
};

}  // namespace para_lsm

#endif