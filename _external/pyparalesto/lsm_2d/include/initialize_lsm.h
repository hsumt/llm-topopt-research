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

#ifndef INITIALIZELSM_H
#define INITIALIZELSM_H

#include <memory>
#include <vector>

#include "hole.h"

/*! \file initialize_lsm.h
    \brief A file that contains the class needed for declaring level set objects
*/

namespace lsm_2d {

/*! \class InitializeLsm
    \brief Serves as an input file type object for the level set method settings
    and defining voids in the level set.
*/
class InitializeLsm {
 public:
  using BlobPtr = std::shared_ptr<Hole>;

  //! Default constructor
  InitializeLsm ();

  /*! \name Dimensionality*/
  ///\{
  int nelx;  //!< number of elements in the x direction
  int nely;  //!< number of elements in the y direction
  ///\}

  /*! \name Initialization of voids*/
  ///\{
  std::vector<BlobPtr> initialVoids;   //!< vector of pointers to the initial voids
  std::vector<BlobPtr> domainVoids;    //!< vector of pointers to the nondesign voids
  std::vector<BlobPtr> fixedBlobs;     //!< vector of pointers to the nondesign domains
  std::vector<BlobPtr> boundaryBlobs;  //!< vector of pointers to the level-set boundaries
  ///\}

  /*! \name Level set method parameters*/
  double move_limit;  //!< maximum distance that a node can move in a single iteration

  /*! \name Sensitivity mapping parameters*/
  ///\{
  /*! \brief Type of sensitivity mapping
      0 for least squares, 1 for discrete adjoint
  */
  int map_flag = 1;
  double perturbation;  //!< Size of perturbation for sensitivity calculation
                        ///\}
};

}  // namespace lsm_2d

#endif