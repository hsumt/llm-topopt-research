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

#ifndef LSM_3D_H
#define LSM_3D_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "eigen3/Eigen/Dense"
#include "grid_math.h"

/*! \file lsm_3d.h
    \brief A file that has the class for level set
*/

//! \todo TODO(carolina): Probably move all the shape classes to a separate file

namespace para_lsm {

/*! \class Blob
    \brief An abstract class for representing arbitrary shapes
*/
class Blob {
 public:
  //! Default onstructor
  Blob ();

  //! Constructor
  /*! \param x_
        x coordinate of the origin

      \param y_
        y coordinate of the origin

      \param z_
        z coordinate of the origin
  */
  Blob (double x_, double y_, double z_);

  //! Get min distance from a point to boundary
  /*! \param point
        coordinates of a point of interest
  */
  virtual double GetMinDist (std::vector<double> point) = 0;

  //! Function that rotates a point about the axis and by the angle
  /*! \param point
        coordinate of the point of interest

      \return
        coordinate of the rotated point
  */
  std::vector<double> RotatePoint (std::vector<double> point);

  double x;                  //!< x coordinate of the origin of the blob
  double y;                  //!< y coordinate of the origin of the blob
  double z;                  //!< z coordinate of the origin of the blob
  double theta = 0.0;        //!< rotation angle
  int flipDistanceSign = 1;  //!< reverse the sign of the min distance

  Eigen::MatrixXd rotMat = Eigen::MatrixXd::Identity (3, 3);  //!< Rotation matrix
  Eigen::VectorXd rotAxis = Eigen::VectorXd::Zero (3);        //!< axis of Rotation
  Eigen::VectorXd rotOrigin = Eigen::VectorXd::Zero (3);      //!< axis of Rotation
};


/*! \class Cuboid
    \brief A derived class for representing a cubeoid
*/
class Cuboid : public Blob {
 public:
  //! Constructor
  Cuboid ();

  //! Constructor
  /*! \param x_
        x coordinate of the origin

      \param y_
        y coordinate of the origin

      \param z_
        z coordinate of the origin

      \param hx_
        half-width of cuboid in the x direction

      \param hx_
        half-width of cuboid in the y direction

      \param hx_
        half-width of cuboid in the z direction
  */
  Cuboid (double x_, double y_, double z_, double hx_, double hy_, double hz_);

  //! get min distance from a point to boundary
  /*! \param point
        coordinates of a point of interest
  */
  double GetMinDist (std::vector<double> point);

  double hx;  //!< half-width of cuboid in the x direction
  double hy;  //!< half-width of cuboid in the y direction
  double hz;  //!< half-width of cuboid in the z direction
};


/*! \class CutPlane
    \brief A derived class for representing a region between two parallel planes
*/
class CutPlane : public Blob {
 public:
  //! Default constructor
  CutPlane ();

  //! Constructor
  /*! Defines two parallel planes using the equation a*x + b*y + c*z + d = 0

      \param a_
        x coefficient

      \param b_
        y coefficient

      \param c_
        z coefficient

      \param d1_
        offset for plane 1

      \param d2_
        offset for plane 2
  */
  CutPlane (double a_, double b_, double c_, double d1_, double d2_);

  //! get min distance from a point to boundary
  /*! \param point
        coordinates of a point of interest
  */
  double GetMinDist (std::vector<double> point);

  // constants used in equation of a plane a*x + b*y + c*z + d = 0
  double a;   //!< x coefficient for plane equation
  double b;   //!< y coefficient for plane equation
  double c;   //!< z coefficient for plane equation
  double d1;  //!< offset for plane 1 equation
  double d2;  //!< offset for plane 2 equation
};


/*! \class Cylinder
    \brief A derived class for representing a right circular cylinder
*/
class Cylinder : public Blob {
 public:
  //! Constructor
  Cylinder ();

  //! Constructor
  /*! \param x_
        x coordinate of the origin

      \param y_
        y coordinate of the origin

      \param z_
        z coordinate of the origin

      \param h_
        half of the height

      \param r_
        Outer radius

      \param r0_
        Inner radius

      \param dir_
        Orientation of the cylinder. For dir = 0 the axis of the cylinder is
        parallel to the x-axis; dir = 1 the axis of the cylinder is parallel
        to the y-axis; dir = 2 the axis of the cylinder is parallel to the
        z-axis.
  */
  Cylinder (double x_, double y_, double z_, double h_, double r_, double r0_, int dir_);

  //! get min distance from a point to boundary
  /*! \param point
        coordinates of a point of interest
   */
  double GetMinDist (std::vector<double> point);

  double h;   //!< half of the height
  double r;   //!< outer radius
  double r0;  //!< inner radius
  int dir;    //!< direction/orientation
};


/*! \class LevelSet3D
    \brief A class that handles the level set function on the grid
*/
class LevelSet3D {
 public:
  using BlobPtr = std::shared_ptr<Blob>;

  //! Constructor.
  /*! \param grid_
        A reference to the grid
  */
  LevelSet3D (Grid &grid_);

  //! Sets grid dimensions
  /*! \param box_x
        Grid dimension in x

      \param box_y
        Grid dimension in y

      \param box_z
        Grid dimension in z
   */
  void SetGridDimensions (int box_x, int box_y, int box_z);

  //! Initializes the signed distance function for a box
  void MakeBox ();

  //! Recomputes the signed distance
  void Reinitialize ();

  //! Advects the level set function under a velocity field
  void Update ();

  //! Smooths the level set funtion
  /*! \param box_smooth
        half width of the box around each grid point for smoothing
  */
  void SmoothPhi (int box_smooth);

  //! compute volume fractions
  void ComputeVolumeFractions ();

  //! Compute aggregation sensitivities
  /*! \param &aggSensi
        \todo TODO(Carolina): add description

      \param &maxVal
        \todo TODO(Carolina): add description

      \param radius
        \todo TODO(Carolina): add description
  */
  void ComputeAggregationSensitivities (std::vector<double> &aggSensi, double &maxVal, int radius);

  //! Helper function
  /*! \param &vfAgg
        \todo TODO(Carolina): add description

      \param &vfAggWeight
        \todo TODO(Carolina): add description

      \param &aggSensi
        \todo TODO(Carolina): add description

      \param radius
        \todo TODO(Carolina): add description
  */
  void DifferntiateAggVf (std::vector<double> &vfAgg, std::vector<double> &vfAggWeight,
                          std::vector<double> &aggSensi, double radius);

  //! Write the signed distance to a text file
  /*! \param file_name
        File name for outputing the signed distance
  */
  void WriteSD (std::string file_name = "SDatgridpoints.txt");

  //! Check if an index is within the bounds of the grid
  /*! \param x_index
        x location of the point of interest

      \param y_index
        y location of the point of interest

      \param z_index
        z location of the point of interest
  */
  bool IsInBounds (int x_index, int y_index, int z_index);

  //! Returns the index of a grid point in the zyx system
  /*! \param x
        x location of the point of interest

      \param y
        y location of the point of interest

      \param z
        z location of the point of interest

      \return
        Vector index of the grid point (Grid4Vector convention) as stored in
        the phi vector of the grid
   */
  int GridPtToIndex (int x, int y, int z);

  //! initialize the reinitialize subroutine
  void InitializeSweep ();

  //! Make some initial holes (voids) in the topology
  void MakeDomainHoles ();

  //! Ignore certain elements' contribution to stress
  void SetIgnoreElementsStress ();

  //! Write out the norm of the gradient of the signed distance function to
  //! a text file
  /*! \param iter
        Iteration of the main optimization.

      \param file_name
        Name of the text file.

      \param file_path
        Directory that the text file should be saved in. A forward slash
        should be included at the end of the file path.
   */
  void WriteGradPhi (int iter = 0, std::string file_name = "grad_phi", std::string file_path = "");

  Grid &grid;           //!< reference to the level set grid
  int hWidth = 3;       //!< half width of the narrowband
  int nGridPoints;      //!< Number of grid points
  int nCells;           //!< Number of cells (elements/voxels)
  double volume = 0.0;  //!< volume represented by the level set

  std::vector<double> volFractions;          //!< vector of element volume fractions
  std::vector<Grid4Vector> &phi = grid.phi;  //!< reference to signed distance
  std::vector<BlobPtr> initialVoids;         //!< vector of pointers to initial holes
  std::vector<BlobPtr> domainVoids;          //!< vector of pointers to the domain holes
  std::vector<BlobPtr> fixedBlobs;           //!< vector of pointers to the fixed holes

  /// Grid velocities that are extrapolated from the optimum velocities
  std::vector<double> gridVel;

  /// bool vector that tells if a point is in narrow band
  std::vector<bool> isInNarrowBand;

  /// Vector that determines which elements should be ignored
  std::vector<bool> ignoreElemStress;

 private:
  //! Adjust the velocities on the grid for the CFL value
  /*! \return
        number of CFL steps
  */
  int AdjustForCFL ();

  //! Set phi at narrow band
  void SetNarrowBandPhi ();

  //! returns the grid point of an index the zyx system
  /*! \param index
        Vector index of the grid point (Grid4Vector convention) as stored in
        the phi vector of the grid

      \return
        Coordinate of the grid point in the zyx system (Grid4Vector convention)
        \todo TODO(carolina): check if coordinate is (x,y,z) or (z,y,x)
  */
  std::vector<int> IndexToGridPt (int index);

  //! computes gradients of grid points in the narrow band
  void ComputeGradients ();

  //! Solves the Eikonal equation for the point
  /*! \param indices_xyz
        Contains the (x,y,z) location of the point of interest
  */
  void SolveEikonal (std::vector<int> indices_xyz);

  //! Define the sweeping directions
  void DefineSweepingDirections (int sweep, int &istart, int &iend, int &idiff, int &jstart,
                                 int &jend, int &jdiff, int &kstart, int &kend, int &kdiff);

  //! Updates velocity at a grid point
  /*! \param xin
        x coordinate of the grid point

      \param yin
        y coordinate of the grid point

      \param zin
        z coordinate of the grid point
  */
  void UpdateVelocity (int xin, int yin, int zin);

  //! Sweep velocity using Fast Sweeping Method (FSM)
  void SweepVelocity ();

  //! Compute volume fraction of an element
  /*! \param corner_phi_vals
        level set function values at the corners

      \param countCell
        the index of element
  */
  void ComputeElementalVolFracs (std::vector<double> &corner_phi_vals, int countCell);

  //! Make some initial holes (voids) in the topology
  void MakeInitialHoles ();

  int nElemX;                   //!< Number of elements x direction
  int nElemY;                   //!< Number of elements Y direction
  int nElemZ;                   //!< Number of elements Z direction
  double large_value = 1.0e10;  //!< Arbitrary large value
  std::vector<double> gradPhi;  //!< gradients on the grid
};

}  // namespace para_lsm

#endif