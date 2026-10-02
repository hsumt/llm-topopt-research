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

#include "optimizer_wrapper.h"

#include <vector>

#include "optimize.h"

namespace para_lsm {

OptimizerWrapper ::OptimizerWrapper (int num_cons_, std::vector<double> max_cons_vals_,
                                     int opt_algo_, bool is_3d_)
    : num_cons (num_cons_), max_cons_vals (max_cons_vals_), opt_algo (opt_algo_), is_3d (is_3d_) {}


std::vector<double> OptimizerWrapper ::Solve (std::vector<double> obj_sens,
                                              std::vector<std::vector<double> > cons_sens,
                                              std::vector<double> curr_cons_vals, bool is_print) {
  // Initiate myOptimizer
  int num_des_var = obj_sens.size ();
  MyOptimizer my_opt (num_des_var, num_cons);

  // set bounds
  for (int i = 0; i < num_des_var; i++) {
    my_opt.z_lo[i] = lower_lim[i];
    my_opt.z_up[i] = upper_lim[i];
    my_opt.z[i] = (0.5 * my_opt.z_lo[i] + 0.5 * my_opt.z_up[i]);
  }

  // Objective gradient
  for (int i = 0; i < num_des_var; i++) my_opt.objFunGrad[i] = obj_sens[i];

  // Constraint gradients
  for (int j = 0; j < num_cons; j++) {
    for (int i = 0; i < num_des_var; i++) {
      my_opt.conFunGrad[j][i] = cons_sens[j][i];
    }
    my_opt.conMaxVals[j] = max_cons_vals[j] - curr_cons_vals[j];
  }

  // The Newton Raphson solver is based on the assumption that a positive boundary point
  // displacement will always result in a positive volume change. This is not true for 2D problems.
  // Therefore, some adjustments need to be made to use Newton Raphson for 2D. (matteo)
  if (!is_3d && opt_algo == 0) {
    // First possibility: use Simplex instead of Newton Raphson
    // std::cout << "Newton Raphson is not available for 2D problems. "
    //           << "Switching to Nlopt MMA." << std::endl;
    // opt_algo = 2;

    // Second possibility: use Newton Raphson but with some modifications.
    // The modifications are as follows:
    // 1) Revert the sign of the objective and constraint gradients.
    // 2) Switch and revert the sign of the upper and lower bounds.
    // 3) Switch the sign of the optimum velocity values (after optimization).

    // 1) Revert the sign of the objective and constraint gradients.
    for (int i = 0; i < num_des_var; i++) my_opt.objFunGrad[i] *= -1;
    for (int j = 0; j < num_cons; j++) {
      for (int i = 0; i < num_des_var; i++) {
        my_opt.conFunGrad[j][i] *= -1;
      }
    }

    // 2) Switch and revert the sign of the upper and lower bounds.
    for (int i = 0; i < num_des_var; i++) {
      double temp = my_opt.z_lo[i];
      my_opt.z_lo[i] = -my_opt.z_up[i];
      my_opt.z_up[i] = -temp;
      my_opt.z[i] = (0.5 * my_opt.z_lo[i] + 0.5 * my_opt.z_up[i]);
    }
  }

  // Solve the optimization problem
  if (opt_algo == 0) my_opt.SolveWithNR ();
  if (opt_algo == 1) my_opt.SolveWithNlopt ();
  if (opt_algo == 2) my_opt.SolveWithSimplex ();
  // if (opt_algo == 3) my_opt.SolveWithIpopt ();

  // Assign optimum velocity values
  std::vector<double> opt_vel (num_des_var, 0.0);
  for (int i = 0; i < num_des_var; i++) {
    opt_vel[i] = my_opt.z[i];
  }

  // 3) Switch the sign of the optimum velocity values (after optimization).
  if (!is_3d && opt_algo == 0) {
    for (int i = 0; i < num_des_var; i++) opt_vel[i] *= -1;
  }

  return opt_vel;
}


void OptimizerWrapper ::SetLimits (std::vector<double> upper_lim_, std::vector<double> lower_lim_) {
  upper_lim = upper_lim_;
  lower_lim = lower_lim_;
}

}  // namespace para_lsm