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

#ifndef GRID_MATH_H
#define GRID_MATH_H

#include <math.h>

#include <iostream>
#include <vector>

#include "eigen3/Eigen/Dense"

/*! \file grid_math.h
    \brief A file that encompasses the classes grid handling and math for
    stencils.
*/

namespace para_lsm {

/*! \class GridVector
    \brief A class used for storing a point on the grid.
*/
class GridVector {
 public:
  //! Default constructor
  GridVector ();

  //! Constructor
  /*! \param x_
        The x location of the point

      \param y_
        The y location of the point

      \param z_
        The z location of the point
  */
  GridVector (double x_, double y_, double z_);

  //! Constructor
  /*! \param &Vector3d
        Reference to a vector of the (x,y,z) coordinates of the point.
  */
  GridVector (const Eigen::Vector3d &Vector3d);

  //! Operator to convert a point as a Vector3d vector
  operator Eigen::Vector3d () const;

  double x;  //!< x location of the point
  double y;  //!< y location of the point
  double z;  //!< z location of the point
};


/*! \class Grid4Vector
    \brief A derived class that can store the location and a scalar value of a
    point.
*/
class Grid4Vector : public GridVector {
 public:
  //! Default constructor
  Grid4Vector ();

  //! Constructor
  /*! \param x_
        The x location of the point

      \param y_
        The y location of the point

      \param z_
        The z location of the point

      \param val_
        Scalar value of the point. Usually the signed distance value.
  */
  Grid4Vector (double x_, double y_, double z_, double val_);

  //! operator
  operator GridVector () const;

  double val;  //!< Value of a scalar value associated with the point
};


/*! \class Grid
    \brief A class that stores the grid
*/
class Grid {
 public:
  //! Constructor
  /*! \param nelx_
        The number of elements in x

      \param nely_
        The number of elements in y

      \param nelz_
        The number of elements in z
  */
  Grid (int nelx_, int nely_, int nelz_);

  //! function to convert grid point to index
  /*! \param x_
        The x location of the point

      \param y_
        The y location of the point

      \param z_
        The z location of the point
  */
  int GridPtToIndex (int x, int y, int z);

  int nelx;                      //!< number of elements in x
  int nely;                      //!< number of elements in y
  int nelz;                      //!< number of elements in z
  std::vector<Grid4Vector> phi;  //!< level set function (signed distance)
};


/*! \class Stencil
    \brief A Base class for calculating spatial gradients at a point
*/
class Stencil {
 public:
  //! Constructor
  /*! \param &grid
        Reference to the grid

      \param iLoc_
        Origin of the stencil in the i-th direction

      \param jLoc_
        Origin of the stencil in the j-th direction

      \param kLoc_
        Origin of the stencil in the k-th direction
  */
  Stencil (Grid &grid, int iLoc_, int jLoc_, int kLoc_);

  //! Boolean function that tells if a point is in bounds
  /*! \param iRel
        Location of the point relative to the origin of the stencil in the
        i-th direction

      \param jRel
        Location of the point relative to the origin of the stencil in the
        j-th direction

      \param kRel
        Location of the point relative to the origin of the stencil in the
        k-th direction

      \return
        True if point is in the bounds of the grid, False otherwise.
  */
  bool IsInBounds (int iRel = 0, int jRel = 0, int kRel = 0);

  //! Function that caps indices to the bounds of the grid
  /*! \param &iRel
        Reference to the location of the point relative to the origin of the
        stencil in the i-th direction

      \param &jRel
        Reference to the location of the point relative to the origin of the
        stencil in the j-th direction

      \param &kRel
        Reference to the location of the point relative to the origin of the
        stencil in the k-th direction
  */
  void CapIndices (int &iRel, int &jRel, int &kRel);

  //! Function that returns the grid value of a point given its the relative
  //! location
  /*! \param iRel
        Location of the point relative to the origin of the stencil in the
        i-th direction

      \param jRel
        Location of the point relative to the origin of the stencil in the
        j-th direction

      \param kRel
        Location of the point relative to the origin of the stencil in the
        k-th direction

      \return
        Value of level set (signed distance) function at the point
  */
  double GetValue (int iRel = 0, int jRel = 0, int kRel = 0);

  //! Function that returns a value (stored in another vector) of a point
  /*! \param &phi_other
        Reference to the other vector that stores the values of interest

      \param iRel
        Location of the point relative to the origin of the stencil in the
        i-th direction

      \param jRel
        Location of the point relative to the origin of the stencil in the
        j-th direction

      \param kRel
        Location of the point relative to the origin of the stencil in the
        k-th direction
  */
  double GetValue (std::vector<double> &phi_other, int iRel = 0, int jRel = 0, int kRel = 0);

  int iLoc;                      //!< Stencil origin in i-th direction
  int jLoc;                      //!< Stencil origin in j-th direction
  int kLoc;                      //!< Stencil origin in k-th direction
  int iMax;                      //!< Positive bound in i-th direction
  int jMax;                      //!< Positive bound in j-th direction
  int kMax;                      //!< Positive bound in k-th direction
  double large_number = 1.0e20;  //!< Arbitrary large value
  Grid &grid;                    //!< reference to the grid
};


/*! \class MinStencil
    \brief A derived class for calculating spatial gradients
*/
class MinStencil : public Stencil {
 public:
  //! Constructor
  /*! \param &grid
        Reference to the grid

      \param iLoc_
        Origin of the stencil in the i-th direction

      \param jLoc_
        Origin of the stencil in the j-th direction

      \param kLoc_
        Origin of the stencil in the k-th direction

      \param halfWidth_
        Half of the width of the stencil

      \param ignoreSelf_
        If True, ignores the stencil origin for computing min
  */
  MinStencil (Grid &grid, int iLoc_, int jLoc_, int kLoc_, int halfWidth_, bool ignoreSelf_);

  //! Function that computes min value in a given direction
  /*! \param direction
        direction = 0,1,2 for x,y,z directions, respectively
  */
  double MinValue (int direction);

  int halfWidth;                //!< half of the width of the stencil
  bool ignoreSelf;              //!< ignores the stencil origin for computing min
  int dirArray[3] = {0, 0, 0};  //!< direction of search
};


/*! \class HJWENOStencil
    \brief A derived class for calculating spatial gradients at a point

    Uses the Hamilton-Jacobi Weighted Essentially Non-Oscillating (HJWENO)
    scheme.
*/
class HJWENOStencil : public Stencil {
 public:
  //! Constructor
  /*! \param &grid
        Reference to the grid

      \param iLoc_
        Origin of the stencil in the i-th direction

      \param jLoc_
        Origin of the stencil in the j-th direction

      \param kLoc_
        Origin of the stencil in the k-th direction
  */
  HJWENOStencil (Grid &grid, int iLoc_, int jLoc_, int kLoc_);

  //! Function that return the gradient value for a given direction
  /*! \param direction
        0,1,2 for x,y,z directions

      \param is_positive
        False for -ve and True for +ve directions

     \return
        gradient value in that direction
  */
  double GradValue (int direction, bool is_positive);

 private:
  //! Function that returns the gradient value in a given direction
  /*! \param v
        A vector of finite differences

      \return
        Gradient value for the given finite differences
  */
  double InterpolateGrad (std::vector<double> v);

  int dirArray[3] = {0, 0, 0};  //!< direction of search
};


/*! \class VelGradStencil
    \brief A derived class for calculating the gradients of the velocity at a
    point
*/
class VelGradStencil : public Stencil {
 public:
  //! Constructor
  /*! \param &grid
        Reference to the grid

      \param iLoc_
        Origin of the stencil in the i-th direction

      \param jLoc_
        Origin of the stencil in the j-th direction

      \param kLoc_
        Origin of the stencil in the k-th direction

      \param halfWidth_
        Halfwidth of the stencil
  */
  VelGradStencil (Grid &grid, int iLoc_, int jLoc_, int kLoc_, int halfWidth_ = 1);

  //! Returns the velocity gradient value and weights for a given direction
  /*! \param &velGrid
        Reference to the velocity

      \param direction
        0,1 for -ve and +ve directions

      \param &grad
        Reference to the gradient

      \param &weight
        Reference to the weight
  */
  void GetVelGrad (std::vector<double> &velGrid, int direction, double &grad, double &weight);

  int halfWidth;                //!< halfwidth of the stencil
  int dirArray[3] = {0, 0, 0};  //!< direction of the gradient
};

}  // namespace para_lsm

#endif