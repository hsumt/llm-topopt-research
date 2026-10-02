//
// Copyright 2021 H Alicia Kim
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

#ifndef OPTIMIZERWRAPPER_H
#define OPTIMIZERWRAPPER_H

#include <vector>

/*! \file optimizer_wrapper.h
    \brief A file that contains the classes for creating an optimization module
    in the level set topology optimization framework
*/

namespace para_lsm {

/*! \class OptimizerWrapper
    \brief Wrapper for in-house optimization functionality
*/
class OptimizerWrapper {
 public:
  //! Constructor
  /*! \param num_cons
        Number of constraints for the optimization.

      \param max_cons_vals
        Vector of maximum constraint values.

      \param opt_algo
        Optimization algorithm. 0 for Newton Raphson, 1 for MMA (Method of
        Moving Asymptotes) using the NLopt library, 2 for Simplex, 3 for Ipopt.
        \warning MMA, Simplex, and Ipopt have not been fully tested and may cause errors if used.

      \param is_3d
        Boolean for whether the problem is 3D or 2D.
  */
  OptimizerWrapper (int num_cons, std::vector<double> max_cons_vals, int opt_algo = 0,
                    bool is_3d = true);

  //! Solve optimization
  /*! Solves single objective multi-constraint optimization.

      \param obj_sens
        Vector of the objective sensitivities for each boundary point.

      \param cons_sens
        Size of the outer vector is the number of constraints. Each element of
        the outer vector is a vector sensitivites for each constraint. The
        size of each of the inner vectors is the number of boundary points.

      \param curr_cons_vals
        Vector of current constraint values.

      \param is_print
        Prints out progress statements during method runtime. Useful for
        debugging.

      \return
        Vector of optimal velocities for each boundary point.
  */
  std::vector<double> Solve (std::vector<double> obj_sens,
                             std::vector<std::vector<double> > cons_sens,
                             std::vector<double> curr_cons_vals, bool is_print = false);

  //! Set the limits for the movement of each boundary point
  /*! \param upper_lim
        Vector of the upper limit of movement for each boundary point

      \param lower_lim
        Vector of the lower limit of movement for each boundary point
  */
  void SetLimits (std::vector<double> upper_lim, std::vector<double> lower_lim);

 private:
  int num_cons;                       //!< number of constraints
  std::vector<double> max_cons_vals;  //!< vector of maximum constraint values
  std::vector<double> upper_lim;      //!< upper movement limit for each bpt
  std::vector<double> lower_lim;      //!< lower movement limit for each bpt

  //! Optimization algorithm.
  /*! 0 for Newton Raphson, 1 for MMA (Method of Moving Asymptotes) using the
      NLopt library, 2 for Simplex, 3 for Ipopt.*/
  int opt_algo;

  //! Boolean for whether the problem is 3D or 2D
  bool is_3d;
};

}  // namespace para_lsm

#endif