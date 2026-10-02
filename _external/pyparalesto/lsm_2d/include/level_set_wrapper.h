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

#ifndef LEVELSETWRAPPER_H
#define LEVELSETWRAPPER_H

#include <memory>
#include <string>
#include <vector>

#include "boundary.h"
#include "initialize_lsm.h"
#include "input_output.h"
#include "level_set.h"
#include "mesh.h"

/*! \file level_set_wrapper.h
    \brief A file that contains the class for wrapping the level set method
    of the topology optimization framework into a single object.
*/

namespace lsm_2d {
/*! \class LevelSetWrapper
    \brief Wrapper for all the level set functionality into a single object
*/
class LevelSetWrapper {
 public:
  using GridPtr = std::shared_ptr<Mesh>;
  using LevelSetPtr = std::shared_ptr<LevelSet>;
  using BoundaryPtr = std::shared_ptr<Boundary>;
  using InputOutputPtr = std::shared_ptr<InputOutput>;

  //! Default constructor
  LevelSetWrapper ();

  //! Constructor
  /*! \param lsm_init
        Initializtion object for the level set input file
  */
  LevelSetWrapper (InitializeLsm &lsm_init);

  //! Destructor
  ~LevelSetWrapper ();

  //! Calculate the element densities (area fractions) of the level set mesh
  /*! \param is_print
        Specifies whether progress statements for the method will be printed
        during runtime. Useful for debugging.
  */
  std::vector<double> CalculateElementDensities (bool is_print = false);

  //! Map element sensitvities to boundary points
  /*! Use this method for the objective element sensitivites and each of the
      constraint element sensitivities

      \param elem_sens
        The partial derivative of the objective or constraint function with
        respect to the density of each element of the level set mesh (df/drho).

      \param is_print
        Prints out progress statements during method runtime. Useful for
        debugging.

      \note If you need the boundary point sensitivities for a volume
      constraint use the MapVolumeSensitivities function instead.
  */
  std::vector<double> MapSensitivities (std::vector<double> elem_sens, bool is_print = false);

  //! Return boundary point volume sensitivities
  std::vector<double> MapVolumeSensitivities ();

  //! Updates the level set via advection
  /*! \param bpoint_velocity
        Vector of velocities for each of the boundary points

      \param is_print
        Prints out progress statements during method runtime. Useful for
        debugging.
  */
  void Update (std::vector<double> bpoint_velocity, bool is_print = false);

  //! Get volume represented by the level set
  /*! CalculateElementDensities should be called before using this method
   */
  double GetVolume ();

  //! Get the number of boundary points that help define the discrete boundary
  /*! CalculateElementDensities should be called before using this method
   */
  int GetNumberBoundaryPoints ();

  //! Get the limits for the movement of each boundary point
  /*! \param upper_lim
        Vector of the upper limit of movement for each boundary point

      \param lower_lim
        Vector of the lower limit of movement for each boundary point
  */
  void GetLimits (std::vector<double> &upper_lim, std::vector<double> &lower_lim);

  //! Writes element densities to a text file (.txt)
  /*! \param iteration
        Iteration number of the topology optimization cycle.

      \param file_path
        Directory to save the file to.

      \param file_name
        Name of the text file.

      \todo TODO(carolina): Move to wrapper_io class?
  */
  void WriteElementDensitiesToTxt (int iter = 0, std::string file_path = "",
                                   std::string file_name = "dens");

  //! Writes stl file of the level set boundary for the current iteration
  /*! \param iteration
        Iteration number of the topology optimization cycle.

      \param file_path
        Directory to save the file to.

      \param file_name
        Name of the text file.

      \todo TODO(carolina): Move to wrapper_io class?
  */
  void WriteVtk (int iter = 0, std::string file_path = "", std::string file_name = "opt_");

  //! Writes vtk file of the boundary points
  /*! \param write_limits
        Flag to decide what to save: true for limits, false for sensitivities.
  */
  void WriteBoundaryVtk (bool write_limits = true);

 private:
  //! Helper function for setting up the grid, level set, and boundary
  /*! \param lsm_init
        Initializtion object for the level set input file
  */
  void SetUp (InitializeLsm &lsm_init);

  //! Helper function for least squares interpolation of boundary sensitivities
  /*! \param bPoint
        Reference to the vector of boundary sensitivities

      \param hWidth
        Half-width of square region to sample points from for interpolation
  */
  std::vector<std::vector<int>> GetNearbyCellsInfo (std::vector<double> bPoint, int hWidth);

  //! Function to check if an index of a cell is in bound
  /*! \param i
        Index of the cell in the x direction

      \param j
        Index of the cell in the y direction
  */
  bool IsCellInBounds (int i, int j);

  //! Least squares interpolation of boundary sensitivities
  /*! \param boundary_sens
        Reference to the vector of boundary sensitivities

      \param element_sens
        Vector of sensitivities with respect to element densities

      \param half_width
        Half-width of square region to sample points from for interpolation

      \param weighted_vol_frac
  */
  void LeastSquareInterpolateBoundarySensitivities (std::vector<double> &boundary_sens,
                                                    std::vector<double> element_sens,
                                                    int half_width = 2, int weighted_vol_frac = 1);

  //! Discrete adjoint calculation of boundary sensitivities
  /*! \param boundary_sens
        Reference to the vector of boundary sensitivities

      \param element_sens
        Vector of sensitivities with respect to element densities
  */
  void DiscreteAdjointBoundarySensitivities (std::vector<double> &boundary_sens,
                                             std::vector<double> element_sens);

  //! Indicates which scheme should be used for mapping the element
  //! sensitivities to the boundary points.
  /*! 0 indicates least squares interpolation.
      1 indicates discrete adjoint.
  */
  int map_flag;

  double perturbation;  //!< Size of perturbation for sensitivity mapping

  /*! \name Dimensionality*/
  ///\{
  int nelx;  //!< number of elements in the x direction
  int nely;  //!< number of elements in the y direction
  ///\}

  /*! \name Pointers to level set objects*/
  ///\{
  GridPtr grid_ptr;           //!< A pointer to the discrete grid for the level set
  LevelSetPtr level_set_ptr;  //!< A pointer to the level set
  BoundaryPtr boundary_ptr;   //!< A pointer to the discretized boundary
  InputOutputPtr io_ptr;      //!< A pointer to the input/output object
                              ///\}
};

}  // namespace lsm_2d

#endif