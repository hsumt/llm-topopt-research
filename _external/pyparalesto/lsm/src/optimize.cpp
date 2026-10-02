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

#include "optimize.h"

namespace para_lsm {

MyOptimizer ::MyOptimizer (int nDesVar_, int nCons_) : nDesVar (nDesVar_), nCons (nCons_) {
  // resize vectors
  z_lo.resize (nDesVar, 0.0);
  z_up.resize (nDesVar, 1.0);
  z.resize (nDesVar, 0.0);

  // resize gradients
  objFunGrad.resize (nDesVar, 0.0);
  conFunGrad.resize (nCons);

  for (int i = 0; i < nCons; i++) {
    conFunGrad[i].resize (nDesVar, 0.0);
  }

  conMaxVals.resize (nCons, 0.0);
}


// helper function for Newton-Raphson
double MyOptimizer ::EstimateTargetConstraintNR () {
  double targetCons;
  gamma = 0.5;
  double min_change = 0.0;
  double max_change = 0.0;
  for (int j = 0; j < nDesVar; j++) {
    if (conFunGrad[0][j] > 0) {
      min_change += gamma * conFunGrad[0][j] * z_lo[j];
      max_change += gammaMax * conFunGrad[0][j] * z_up[j];
    } else {
      min_change += gamma * conFunGrad[0][j] * z_up[j];
      max_change += gammaMax * conFunGrad[0][j] * z_lo[j];
    }
  }  // j

  targetCons = std::max (min_change, conMaxVals[0]);
  // targetCons = std::min (max_change, targetCons) ;

  return targetCons;
}


// helper function for Newton-Raphson
void MyOptimizer ::EstimateTargetConstraintNRmulti () {
  targetConsVec.resize (nCons);
  isActive.resize (nCons);

  for (int i = 0; i < nCons; i++) {
    double min_change = 0.0;
    double max_change = 0.0;
    double total_min_change = 0.0;
    double total_max_change = 0.0;

    for (int j = 0; j < nDesVar; j++) {
      if (conFunGrad[i][j] > 0) {
        min_change += gamma * conFunGrad[i][j] * z_lo[j];
        max_change += gamma * conFunGrad[i][j] * z_up[j];

        total_min_change += conFunGrad[i][j] * z_lo[j];
        total_max_change += conFunGrad[i][j] * z_up[j];
      } else {
        min_change += gamma * conFunGrad[i][j] * z_up[j];
        max_change += gamma * conFunGrad[i][j] * z_lo[j];

        total_min_change += conFunGrad[i][j] * z_up[j];
        total_max_change += conFunGrad[i][j] * z_lo[j];
      }
    }

    if (total_max_change < conMaxVals[i]) {
      isActive[i] = 0;
      targetConsVec[i] = 0.0;
    } else {
      isActive[i] = 1;
      targetConsVec[i] = std::max (min_change, conMaxVals[i]);
    }
  }
}


// Helper function for Newton-Raphson
Eigen::VectorXd MyOptimizer ::FOfLambdaNRMulti (std::vector<double> &lambda, double lambdag) {
  Eigen::VectorXd f_of_lambda = Eigen::VectorXd::Zero (nCons);
  for (int i = 0; i < nCons; i++) {
    f_of_lambda[i] = -targetConsVec[i];
  }

  for (int j = 0; j < nDesVar; j++) {
    double zj = lambdag * z_lo[j];

    for (int i = 0; i < nCons; i++) {
      if (i == 0) {
        zj += lambda[i] * objFunGrad[j];
      } else {
        zj += lambda[i] * conFunGrad[i][j] / (1.0e-8 + bAreas[j]);
      }
    }

    zj = std::max (zj, z_lo[j]);
    zj = std::min (zj, z_up[j]);
    z[j] = zj;
    for (int i = 0; i < nCons; i++) {
      if (isActive[i]) f_of_lambda[i] += conFunGrad[i][j] * z[j];
    }
  }

  return f_of_lambda;
}


// Helper function for Newton-Raphson
void MyOptimizer ::UpdateLambdaNRMulti (std::vector<double> &lambda, double delta_lambda,
                                        double lambdag) {
  // construct jacobian using finite difference
  Eigen::MatrixXd Jac = Eigen::MatrixXd::Zero (nCons, nCons);
  Eigen::VectorXd F0 = FOfLambdaNRMulti (lambda);

  for (int i = 0; i < nCons; i++) {
    if (isActive[i]) {
      std::vector<double> lambdaNew = lambda;
      lambdaNew[i] += delta_lambda;
      Jac.col (i) = FOfLambdaNRMulti (lambdaNew);
      lambdaNew[i] -= 2.0 * delta_lambda;
      Jac.col (i) -= FOfLambdaNRMulti (lambdaNew);
      Jac.col (i) /= (delta_lambda * 2);
    } else {
      Jac (i, i) = 1.0;
    }
  }

  Eigen::VectorXd deltaLam = Jac.colPivHouseholderQr ().solve (F0);

  // update lambda
  if (std::isnan (deltaLam.norm ())) {
    std::cout << "Newton-Raphson did not converge!" << std::endl;
  } else
    for (int i = 0; i < nCons; i++) lambda[i] -= deltaLam[i];
}


// Helper function for Newton-Raphson
double MyOptimizer ::FOfLambdaNR (double lambda, double targetCons) {
  double f_of_lambda = -targetCons;
  for (int j = 0; j < nDesVar; j++) {
    double zj = z_lo[j] + lambda * objFunGrad[j];
    zj = std::max (zj, z_lo[j]);
    zj = std::min (zj, z_up[j]);
    f_of_lambda += conFunGrad[0][j] * zj;
  }

  return f_of_lambda;
}


// Helper function for Newton-Raphson
double MyOptimizer ::NormalizeObjectiveGradients () {
  // find absolute max
  double abs_max = std::abs (objFunGrad[0]);
  for (int j = 0; j < nDesVar; j++) {
    abs_max = std::max (abs_max, (objFunGrad[j]));
  }

  // normalize
  for (int j = 0; j < nDesVar; j++) {
    objFunGrad[j] /= abs_max;
  }

  return abs_max;
}


// Helper function for Newton-Raphson
std::vector<double> MyOptimizer ::NormalizeConstraintGradients (int nSkip) {
  std::vector<double> conGradArgMax (nCons);

  for (int i = 0; i < nCons; i++) {
    // find absolute max
    double abs_max = std::abs (conFunGrad[i][0]);
    for (int j = 0; j < nDesVar; j++) {
      abs_max = std::max (abs_max, (conFunGrad[i][j]));
    }
    // normalize
    if (i == nSkip) abs_max = 1.0;      // adjust for skip number
    if (abs_max == 0.0) abs_max = 1.0;  // divide by 0 error
    for (int j = 0; j < nDesVar; j++) {
      conFunGrad[i][j] /= abs_max;
    }
    conGradArgMax[i] = abs_max;

    conMaxVals[i] /= abs_max;
  }

  return conGradArgMax;
}


void MyOptimizer ::SolveWithNR () {
  // NOTE: Only for minimizing obj function s.t. one volume constraint

  NormalizeObjectiveGradients ();
  double targetCons = EstimateTargetConstraintNR ();

  double lambda = 0.0;
  double delta_lambda = -0.001;
  for (int nr_iter = 0; nr_iter <= 100; nr_iter++) {
    // use the Newton-Raphson method to update lambda
    double f_of_lambda = FOfLambdaNR (lambda, targetCons);
    double f1_of_lambda = FOfLambdaNR (lambda + delta_lambda, targetCons);
    double f_dash_of_lambda = (f1_of_lambda - f_of_lambda) / delta_lambda;
    if (std::abs (f_dash_of_lambda) < 1.0e-16) {
      std::cout << "f_dash_of_lambda = " << f_dash_of_lambda << std::endl;
      break;
    }
    if (std::abs (f_of_lambda) < 0.001 * std::abs (targetCons)) {
      break;
    }
    // update lammbda
    lambda = lambda - (f_of_lambda / f_dash_of_lambda);
  }

  // cap the optimized velocity
  for (int j = 0; j < nDesVar; j++) {
    double zj = z_lo[j] + (lambda)*objFunGrad[j];
    zj = std::max (zj, z_lo[j]);
    zj = std::min (zj, z_up[j]);
    z[j] = zj;

    if (z_lo[j] > z_up[j]) {
      std::cout << " bound error: " << z_lo[j] << " " << z_up[j] << std::endl;
    }
  }
}


void MyOptimizer ::SolveWithSimplex () {
  double objGradArgMax = NormalizeObjectiveGradients ();  // normalize

  std::vector<double> conGradArgMax = NormalizeConstraintGradients ();  // normalize

  // Set up GLP (GNU Linear Programming module)
  glp_prob *linear_programming;

  int num_constraints = nCons;
  int num_des_vars = nDesVar;

  //  Initialise the arrays of the A matrix
  // int i_array[1+num_constraints*num_des_vars], j_array[1+num_constraints*num_des_vars];
  // double array_values[1+num_constraints*num_des_vars];

  std::vector<int> i_array (1 + num_constraints * num_des_vars, 0);
  std::vector<int> j_array (1 + num_constraints * num_des_vars, 0);
  std::vector<double> array_values (1 + num_constraints * num_des_vars, 0.0);

  // create problem
  linear_programming = glp_create_prob ();
  glp_add_cols (linear_programming, num_des_vars);
  glp_add_rows (linear_programming, num_constraints);

  // specify we are minimizing the objective funtion
  glp_set_obj_dir (linear_programming, GLP_MIN);

  // turn off output
  glp_term_out (GLP_OFF);

  for (int i = 1; i <= num_constraints; ++i) {
    for (int j = 1; j <= num_des_vars; ++j) {
      // cout << "i,j =" << i << ", " << j << endl;
      i_array[(i - 1) * num_des_vars + j] = i;
      j_array[(i - 1) * num_des_vars + j] = j;
      array_values[(i - 1) * num_des_vars + j] = conFunGrad[i - 1][j - 1];

      // set bounds and objective gradients
      if (i == 1) {
        if (z_lo[j - 1] >= z_up[j - 1]) {
          // std::cout << " bound error: " <<  z_lo[j-1] << " " << z_up[j-1] << std::endl;
          z_lo[j - 1] = z_up[j - 1] - 1.0e-6;
        }

        glp_set_col_bnds (linear_programming, j, GLP_DB, z_lo[j - 1], z_up[j - 1]);
        glp_set_obj_coef (linear_programming, j, objFunGrad[j - 1]);
      }
    }  // j

    // compute min and max change
    double min_change = 0.0;
    double max_change = 0.0;
    double gamma = 0.5;
    for (int j = 1; j <= num_des_vars; j++) {
      if (conFunGrad[i - 1][j - 1] > 0) {
        min_change += gamma * conFunGrad[i - 1][j - 1] * z_lo[j - 1];
        max_change += gamma * conFunGrad[i - 1][j - 1] * z_up[j - 1];
      } else {
        max_change += gamma * conFunGrad[i - 1][j - 1] * z_lo[j - 1];
        min_change += gamma * conFunGrad[i - 1][j - 1] * z_up[j - 1];
      }  // if loop
    }    // j

    double change = std::max (min_change, conMaxVals[i - 1]);
    // change = std::min(change , max_change);
    // std::cout << "change = " << change << std::endl;
    // std::cout << "min_change = " << min_change << std::endl;
    // std::cout << " conMaxVals[i-1] = " <<  conMaxVals[i-1] << std::endl;

    // set row bounds
    glp_set_row_bnds (linear_programming, i, GLP_UP, 0.0, change);
  }  // i

  std::cout << "solving simplex ..." << std::endl;

  // solve problem
  int *i_arrayP = &i_array[0];
  int *j_arrayP = &j_array[0];
  double *array_valuesP = &array_values[0];
  glp_load_matrix (linear_programming, num_constraints * num_des_vars, i_arrayP, j_arrayP,
                   array_valuesP);
  glp_smcp parm;
  glp_init_smcp (&parm);
  parm.tm_lim = 1000;  // time limit in milli seconds
  // parm.meth = GLP_DUAL;
  glp_simplex (linear_programming, &parm);

  std::cout << "done." << std::endl;

  // assign the optimum velocities
  double velsum = 0.0;
  double delta_obj = 0.0;
  std::vector<double> delta_cons (nCons, 0.0);
  for (int j = 1; j <= num_des_vars; ++j) {
    z[j - 1] = glp_get_col_prim (linear_programming, j);
    velsum += std::abs (z[j - 1]);
    delta_obj += objFunGrad[j - 1] * z[j - 1];
    // velsum += std::abs( conFunGrad[0][j-1]);
    for (int i = 0; i < nCons; i++) delta_cons[i] += conFunGrad[i][j - 1] * z[j - 1];
  }
  std::cout << "velsum = " << velsum << std::endl;
  std::cout << "delta_obj = " << delta_obj * objGradArgMax << std::endl;
  for (int i = 0; i < nCons; i++) {
    std::cout << "delta_cons[" << i << "] = " << delta_cons[i] * conGradArgMax[i] << std::endl;
  }

  /* housekeeping */
  glp_delete_prob (linear_programming);
  glp_free_env ();
}


double MyOptimizer ::myfunc (const std::vector<double> &x, std::vector<double> &grad, void *data) {
  nlopt_cons_grad_data *d = reinterpret_cast<nlopt_cons_grad_data *> (data);
  double max_cons = d->max_cons;
  std::vector<double> grad_data = d->grad_data;

  double return_value = 0.0;
  if (!grad.empty ()) {
    for (int i = 0; i < grad_data.size (); i++) {
      grad[i] = grad_data[i];
      return_value += grad[i] * x[i];
    }
  }
  return return_value;
}


double MyOptimizer ::myconstraint (const std::vector<double> &x, std::vector<double> &grad,
                                   void *data) {
  nlopt_cons_grad_data *d = reinterpret_cast<nlopt_cons_grad_data *> (data);
  double max_cons = d->max_cons;
  std::vector<double> grad_data = d->grad_data;

  double return_value = -max_cons;
  if (!grad.empty ()) {
    for (int i = 0; i < grad_data.size (); i++) {
      grad[i] = grad_data[i];
      return_value += grad[i] * x[i];
    }
  }
  return return_value;
}


void MyOptimizer ::SolveWithNlopt () {
  double objGradArgMax = NormalizeObjectiveGradients ();                // normalize
  std::vector<double> conGradArgMax = NormalizeConstraintGradients ();  // normalize
  int num_constraints = nCons;
  int num_des_vars = nDesVar;

  // nlopt::opt opt(nlopt::LD_CCSAQ, num_des_vars); //(NOTE: not working?)
  nlopt::opt opt (nlopt::LD_MMA, num_des_vars);
  // nlopt::opt opt(nlopt::LN_SBPLX, num_des_vars);

  // BOUNDS
  opt.set_lower_bounds (z_lo);
  opt.set_upper_bounds (z_up);

  // OBJECTIVE FUNCTION
  nlopt_cons_grad_data obj_data;
  obj_data.max_cons = 0.0;
  obj_data.grad_data = objFunGrad;
  opt.set_min_objective (MyOptimizer::myfunc, &obj_data);

  // CONSTRAINT Gradients
  std::vector<nlopt_cons_grad_data> constraint_data (num_constraints);
  std::vector<double> max_change (num_constraints, 0.0);
  std::vector<double> min_change (num_constraints, 0.0);

  for (int i = 0; i < num_constraints; i++) {
    constraint_data[i].grad_data = conFunGrad[i];

    double min_change = 0.0;
    double max_change = 0.0;
    for (int j = 0; j < num_des_vars; j++) {
      if (conFunGrad[i][j] > 0) {
        min_change += gamma * conFunGrad[i][j] * z_lo[j];
        max_change += gamma * conFunGrad[i][j] * z_up[j];
      } else {
        min_change += gamma * conFunGrad[i][j] * z_up[j];
        max_change += gamma * conFunGrad[i][j] * z_lo[j];
      }
    }

    constraint_data[i].max_cons = std::max (min_change, conMaxVals[i]);
    // constraint_data[i].max_cons = std::min(max_change , constraint_data[i].max_cons);

    opt.add_inequality_constraint (MyOptimizer::myconstraint, &constraint_data[i], 1e-8);
  }

  opt.set_xtol_rel (1e-6);
  double minf;
  opt.set_maxtime (5);  // 5 seconds

  try {
    nlopt::result result = opt.optimize (z, minf);
    // std::cout << "found minimum at f(" << z[0] << "," << z[1] << ") = "
    //     << std::setprecision(10) << minf << std::endl;
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what () << std::endl;
  }

  // cap the optimized velocity
  for (int j = 0; j < nDesVar; j++) {
    double zj = z[j];
    zj = std::max (zj, z_lo[j]);
    zj = std::min (zj, z_up[j]);
    z[j] = zj;
  }

  // assign the optimum velocities
  double velsum = 0.0;
  double delta_obj = 0.0;
  std::vector<double> delta_cons (nCons, 0.0);
  for (int j = 1; j <= num_des_vars; ++j) {
    velsum += std::abs (z[j - 1]);
    delta_obj += objFunGrad[j - 1] * z[j - 1];
    // velsum += std::abs( conFunGrad[0][j-1]);
    for (int i = 0; i < nCons; i++) delta_cons[i] += conFunGrad[i][j - 1] * z[j - 1];
  }
  std::cout << "velsum = " << velsum << std::endl;
  std::cout << "delta_obj = " << delta_obj * objGradArgMax << std::endl;
  for (int i = 0; i < nCons; i++) {
    std::cout << "delta_cons[" << i << "] = " << delta_cons[i] * conGradArgMax[i] << std::endl;
  }
}


void MyOptimizer ::SolveCustom () {
  // normalize objective and constraints (skip the volume constraint)
  double objGradArgMax = NormalizeObjectiveGradients ();                 // normalize
  std::vector<double> conGradArgMax = NormalizeConstraintGradients (0);  // normalize

  // initialize lambda
  std::vector<double> lambda (nCons, 0.0);

  // solve NR
  double delta_obj = SolveCustomAux (lambda);
  std::cout << "delta_obj = " << delta_obj << std::endl;
  for (int i = 0; i < nCons; i++) {
    if (isActive[i]) {
      std::cout << "Further reducing constraint " << i << std::endl;
      double conMaxValsBkp = conMaxVals[i];
      conMaxVals[i] = -1.0e10;  // set to a really large negative number
      // solve again
      std::vector<double> lambdaNew (nCons, 0.0);
      double delta_obj_new = SolveCustomAux (lambdaNew);

      std::cout << "delta_obj_new = " << delta_obj_new << std::endl;

      if (delta_obj_new < delta_obj) {
        delta_obj_new = delta_obj;
        lambda = lambdaNew;
      }
      conMaxVals[i] = conMaxValsBkp;
    }
  }

  delta_obj = 0.0;
  std::vector<double> delta_cons (nCons, 0.0);
  // cap the optimized velocity
  for (int j = 0; j < nDesVar; j++) {
    double zj = z_lo[j];
    for (int i = 0; i < nCons; i++) {
      if (i == 0) {
        zj += lambda[i] * objFunGrad[j];
      } else {
        zj += lambda[i] * conFunGrad[i][j] / (1.0e-8 + bAreas[j]);
      }
    }
    zj = std::max (zj, z_lo[j]);
    zj = std::min (zj, z_up[j]);
    z[j] = zj;
    delta_obj += objFunGrad[j] * z[j];
    // velsum += std::abs( conFunGrad[0][j-1]);
    for (int i = 0; i < nCons; i++) delta_cons[i] += conFunGrad[i][j] * z[j];
  }

  EstimateTargetConstraintNRmulti ();
  std::cout << "final delta_obj = " << delta_obj << std::endl;
  std::cout << "final dimensional delta_obj = " << delta_obj * objGradArgMax << std::endl;
  for (int i = 0; i < nCons; i++) {
    std::cout << "final delta_cons[" << i << "] = " << delta_cons[i] * conGradArgMax[i]
              << "; limit = " << targetConsVec[i] * conGradArgMax[i] << std::endl;
  }
}


// helper function
double MyOptimizer ::SolveCustomAux (std::vector<double> &lambda) {
  // estimate min and max and active constraints
  EstimateTargetConstraintNRmulti ();
  // The objective is to find F0(lambda) = 0 vector
  Eigen::VectorXd F0 = FOfLambdaNRMulti (lambda);
  for (int i = 0; i < nCons; i++) {
    std::cout << "constraint " << i << " is active ? " << int (isActive[i]) << std::endl;
  }
  std::cout << "F0.norm() = " << F0.norm () << std::endl;

  int max_nr_iter = 40;
  double delta_lambda = -0.1;
  double lambdag = 1.0;
  for (int nr_iter = 0; nr_iter <= max_nr_iter; nr_iter++) {
    UpdateLambdaNRMulti (lambda, delta_lambda, lambdag);
    Eigen::VectorXd Fnew = FOfLambdaNRMulti (lambda);
    std::cout << "Fnew.norm() = " << Fnew.norm () << std::endl;
    std::cout << "Fnew.minCoeff() = " << Fnew.minCoeff () << std::endl;
    // std::cout << "Fnew = " << Fnew << std::endl;
    if (Fnew.norm () <= 1.0e-4) {
      did_nr_converge = true;
      break;
    }
  }

  // cap the optimized velocity
  for (int j = 0; j < nDesVar; j++) {
    double zj = z_lo[j];
    for (int i = 0; i < nCons; i++) {
      if (i == 0) {
        zj += lambda[i] * objFunGrad[j];
      } else {
        zj += lambda[i] * conFunGrad[i][j] / (1.0e-8 + bAreas[j]);
      }
    }
    zj = std::max (zj, z_lo[j]);
    zj = std::min (zj, z_up[j]);
    z[j] = zj;
  }
  double delta_obj = 0.0;
  for (int j = 1; j <= nDesVar; ++j) delta_obj += objFunGrad[j - 1] * z[j - 1];

  return delta_obj;
}


// Solve with Ipopt
// void MyOptimizer ::SolveWithIpopt () {
//   // Create a new instance of IpoptApplication
//   Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory ();

//   // Initialize IpoptApplication
//   InitializeIpopt (app);

//   // Create a new instance of the sub optimal problem
//   Ipopt::SmartPtr<Ipopt::TNLP> nlp
//       = new OptimizeIpopt (z, z_lo, z_up, objFunGrad, conFunGrad, conMaxVals);

//   // Ask Ipopt to solve the problem
//   Ipopt::ApplicationReturnStatus status = app->OptimizeTNLP (nlp);

//   // Check the return status of Ipopt
//   if (status != Ipopt::ApplicationReturnStatus::Solve_Succeeded)
//     std::cout << "\n*** WARNING: the problem has not converged (code " << status << ")."
//               << std::endl;
// }


// Helper function for Ipopt
// void MyOptimizer ::InitializeIpopt (Ipopt::SmartPtr<Ipopt::IpoptApplication> app) {
//   // Numeric options
//   app->Options ()->SetNumericValue ("tol", 1e-8);
//   app->Options ()->SetIntegerValue ("max_iter", 500);

//   // Linear solver: ma27 (default), ma57, ma77, ma86, ma97, paradiso, paradisomkl, spral, wsm,
//   // mumps, custom
//   app->Options ()->SetStringValue ("linear_solver", "mumps");

//   // Specify thet we are going to deal with a LP problem
//   app->Options ()->SetStringValue ("hessian_approximation", "limited-memory");
//   app->Options ()->SetStringValue ("jac_c_constant", "yes");
//   app->Options ()->SetStringValue ("jac_d_constant", "yes");

//   // Output options
//   app->Options ()->SetIntegerValue ("print_level", 0);
//   // app->Options()->SetIntegerValue("file_print_level", 4);
//   // app->Options()->SetStringValue("output_file", std::string(getenv("HOME")) +
//   // "/OpenLSTO/ipopt/ipopt.out");

//   // Technique used for scaling the NLP: none, user-scaling, gradient-based (default),
//   // equilibration-based
//   // app->Options ()->SetStringValue ("nlp_scaling_method", "equilibration-based");

//   // Indicates whether to do Mehrotra's predictor-corrector algorithm: yes, no (default)
//   app->Options ()->SetStringValue ("mehrotra_algorithm", "yes");

//   // Controls whether to enable automatic scaling in MA57: yes, no (default)
//   // app->Options ()->SetStringValue ("ma57_automatic_scaling", "yes");

//   // Initialize the IpoptApplication
//   Ipopt::ApplicationReturnStatus status = app->Initialize ();

//   // Check if everything worked
//   if (status != Ipopt::ApplicationReturnStatus::Solve_Succeeded)
//     throw std::runtime_error ("*** Error during Ipopt initialization!\n");
// }

}  // namespace para_lsm