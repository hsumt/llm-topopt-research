//
// Copyright 2023 H Alicia Kim
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

#include "optimize_ipopt.h"

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace para_lsm {
// Constructor
OptimizeIpopt::OptimizeIpopt (std::vector<double> &z, std::vector<double> &z_lo,
                              std::vector<double> &z_up, std::vector<double> &objFunGrad,
                              std::vector<std::vector<double>> &conFunGrad,
                              std::vector<double> conMaxVals)
    : z (z),
      z_lo (z_lo),
      z_up (z_up),
      objFunGrad (objFunGrad),
      conFunGrad (conFunGrad),
      conMaxVals (conMaxVals) {
  // Number of design variables
  nDesVar = z.size ();

  // Number of constraints
  nCons = conMaxVals.size ();

  // Scale constraints
  ScaleConstraints ();
}

// Method to scale constraints
void OptimizeIpopt::ScaleConstraints () {
  // Loop on constraits
  for (unsigned int i = 0; i < nCons; i++) {
    if (conMaxVals[i] < 0) {
      // Min constraint change
      double min = 0;

      // Integrate over design variables
      for (unsigned int j = 0; j < nDesVar; j++) {
        min += std::min (conFunGrad[i][j] * z_lo[j], conFunGrad[i][j] * z_up[j]);
      }
      // Scaling
      min *= gamma;

      // Constraint can't be reached at this iteration. Relax it.
      conMaxVals[i] = std::max (conMaxVals[i], min);
    }
  }
}

// Default destructor
OptimizeIpopt::~OptimizeIpopt () {}

// Method to return some info about the nlp
bool OptimizeIpopt::get_nlp_info (
    Ipopt::Index &n,             // number of x
    Ipopt::Index &m,             // number of g
    Ipopt::Index &nnz_jac_g,     // number of non zero elements of J
    Ipopt::Index &nnz_h_lag,     // number of non zero elements of H
    IndexStyleEnum &index_style  // style of the indexing (C starts from 0)
) {
  n = nDesVar;
  m = nCons;
  nnz_jac_g = n * m;
  nnz_h_lag = 0;
  index_style = TNLP::C_STYLE;

  return true;
}

// Method to return the number of non-linear variables
Ipopt::Index OptimizeIpopt::get_number_of_nonlinear_variables () { return 0; }

// Method to return the bounds for the nlp
bool OptimizeIpopt::get_bounds_info (Ipopt::Index n,  // number of x
                                     Number *x_l,     // lower bound for x
                                     Number *x_u,     // upper bound for x
                                     Ipopt::Index m,  // number of g
                                     Number *g_l,     // lower bound for g
                                     Number *g_u      // upper bound for g
) {
  // Lower and upper bound for x
  for (Ipopt::Index i = 0; i < n; i++) {
    x_l[i] = z_lo[i];
    x_u[i] = z_up[i];
  }

  // Bounds for constraints
  for (Ipopt::Index j = 0; j < m; j++) {
    g_l[j] = -2e19; // this is to specify that there is no lower bound
    g_u[j] = conMaxVals[j];
  }

  return true;
}

// Method to return the starting point for the algorithm
bool OptimizeIpopt::get_starting_point (
    Ipopt::Index n,    // number of x
    bool init_x,       // if TRUE this method must provide an initial value for x
    Number *x,         // initial value for x
    bool init_z,       // if TRUE this method must provide an initial value for z
    Number *z_l,       // lower bound for z
    Number *z_u,       // upper bound for z
    Ipopt::Index m,    // number of g
    bool init_lambda,  // if TRUE this method must provide an initial value for lambda
    Number *lambda     // initial value for lambda
) {
  // We provide only the initial value of x
  assert (init_x == true);
  assert (init_z == false);
  assert (init_lambda == false);

  // Starting point
  for (Ipopt::Index i = 0; i < n; i++) {
    x[i] = 0;
  }

  return true;
}

// Method to return the value of the objective function f
bool OptimizeIpopt::eval_f (Ipopt::Index n,   // number of x
                            const Number *x,  // value of x
                            bool new_x,  // FALSE if any evaluation method (eval_*) was previously
                                         // called with the same values in x
                            Number &obj_value  // value of f
) {
  obj_value = 0;
  for (Ipopt::Index i = 0; i < n; i++) {
    obj_value += objFunGrad[i] * x[i];
  }

  return true;
}

// Method to return the value of the gradient of the objective function
bool OptimizeIpopt::eval_grad_f (Ipopt::Index n,   // number of x
                                 const Number *x,  // value of x
                                 bool new_x,       // FALSE if any evaluation method (eval_*) was
                                                   // previously called with the same values in x
                                 Number *grad_f    // value of df
) {
  for (Ipopt::Index i = 0; i < n; i++) {
    grad_f[i] = objFunGrad[i];
  }

  return true;
}

// Method to return the value of the constraints
bool OptimizeIpopt::eval_g (Ipopt::Index n,   // number of x
                            const Number *x,  // value of x
                            bool new_x,  // FALSE if any evaluation method (eval_*) was previously
                                         // called with the same values in x
                            Ipopt::Index m,  // number of g
                            Number *g        // value of g
) {
  // Loop on constraint to assign initial values
  for (Ipopt::Index j = 0; j < m; j++) {
    g[j] = 0;
  }

  // Loop on constraints
  for (Ipopt::Index j = 0; j < m; j++) {
    // Loop on design variables
    for (Ipopt::Index i = 0; i < n; i++) {
      g[j] += conFunGrad[j][i] * x[i];
    }
  }

  return true;
}

/* Method to return:
    1) The structure of the constraint's jacobian if "values" is NULL
    2) The values of the constraint's jacobian if "values" is not NULL
*/
bool OptimizeIpopt::eval_jac_g (
    Ipopt::Index n,   // number of x
    const Number *x,  // value of x
    bool new_x,       // FALSE if any evaluation method (eval_*) was previously called with the same
                      // values in x
    Ipopt::Index m,   // number of g
    Ipopt::Index nnz_jac_g,  // number of non zero elements of J
    Ipopt::Index *
        iRow,  // first call: vector of row indices of the non zero elements of J, later calls: NULL
    Ipopt::Index *jCol,  // first call: vector of column indices of the non zero elements of J,
                         // later calls: NULL
    Number *values       // first call: NULL, later calls: values of the non zero elements of J
) {
  if (values == NULL) {
    // Return the structure of J: m rows and n columns
    Ipopt::Index idx = 0;
    for (Ipopt::Index j = 0; j < m; j++) {
      for (Ipopt::Index i = 0; i < n; i++) {
        iRow[idx] = j;
        jCol[idx] = i;
        idx++;
      }
    }
    assert (idx == m * n);
  } else {
    // Return the values of J
    Ipopt::Index idx = 0;
    for (Ipopt::Index j = 0; j < m; j++) {
      for (Ipopt::Index i = 0; i < n; i++) {
        values[idx] = conFunGrad[j][i];
        idx++;
      }
    }
    assert (idx == m * n);
  }

  return true;
}

// This method is called when the algorithm is complete so the TNLP can store/write the solution
void OptimizeIpopt::finalize_solution (
    SolverReturn status,              // status of the algorithm
    Ipopt::Index n,                   // number of x
    const Number *x,                  // optimal value of x
    const Number *z_l,                // final lower value of z
    const Number *z_u,                // final upper bound of z
    Ipopt::Index m,                   // number of g
    const Number *g,                  // final values of g
    const Number *lambda,             // final values lambda
    Number obj_value,                 // residual of f
    const IpoptData *ip_data,         // provided for expert users - not my case I think
    IpoptCalculatedQuantities *ip_cp  // provided for expert users - not my case I think
) {
  // Store optimal solution
  for (Ipopt::Index i = 0; i < n; i++) z[i] = x[i];
}
}  // namespace para_lsm