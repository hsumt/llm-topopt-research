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

#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <glpk.h>

#include <cmath>
// #include <coin-or/IpIpoptApplication.hpp>
#include <iomanip>
#include <iostream>
#include <nlopt.hpp>
#include <vector>

#include "eigen3/Eigen/Dense"
// #include "optimize_ipopt.h"

/*! \file optimize.h
    \brief A file that has the class for optimization
*/

namespace para_lsm {

/*! \class MyOptimizer
    \brief A class that is a wrapper for the optimization
*/
class MyOptimizer {
 public:
  //! constructor
  /*! \param nDesVar_
        Number of design variables

      \param nCons_
        Number of constraints
  */
  MyOptimizer (int nDesVar_, int nCons_);

  /*! \name Solvers*/
  ///\{

  //! Solves the optimization using newtwon-raphson
  /*! \note Can only be used for minimizing an objective function subject to a
      single volume constraint.
  */
  void SolveWithNR ();

  //! Solves the optimization using simplex
  void SolveWithSimplex ();

  //! Solves the optimization using MMA from the NLopt library.
  /*! See
     https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa
      for details
  */
  void SolveWithNlopt ();

  //! Solves the optimization using Ipopt
  // void SolveWithIpopt ();

  //! Custom solver for when nothing else works
  void SolveCustom ();
  ///\}

  //! Helper function for SolveCustom
  /*! \param &lambda
        Reference to vector of lambda values

      \return
        Objective function value
  */
  double SolveCustomAux (std::vector<double> &lambda);

  //! Evaluate objective function value?
  /*! /todo TODO(Carolina): verify description

      \param x
        Reference to a vector of design variable values? /todo TODO(Carolina):
        check that this description is correct.

      \param grad
        Reference to gradient vector? /todo TODO(Carolina): check that this
        description is correct.

      \param data
        \todo TODO(Carolina): figure out what this parameter is.

      \return
        Objective function value? /todo TODO(Carolina): check that this is
        correct
  */
  static double myfunc (const std::vector<double> &x, std::vector<double> &grad, void *data);

  //! Evaluate constraint function value?
  /*! /todo TODO(Carolina): verify description

      \param x
        Reference to a vector of design variable values? /todo TODO(Carolina):
        check that this description is correct.

      \param grad
        Reference to gradient vector? /todo TODO(Carolina): check that this
        description is correct.

      \param data
        \todo TODO(Carolina): figure out what this parameter is.

      \return
        Constraint function value? /todo TODO(Carolina): check that this is
        correct.
  */
  static double myconstraint (const std::vector<double> &x, std::vector<double> &grad, void *data);

  //! Helper function for Newton-Raphson
  /*! \param lambda
        \todo TODO(Carolina): figure out what this parameter is

      \param lambdag
        \todo TODO(Carolina): figure out what this parameter is
  */
  Eigen::VectorXd FOfLambdaNRMulti (std::vector<double> &lambda, double lambdag = 1.0);

  //! Helper function for Newton-Raphson
  /*! \param lambda
        \todo TODO(Carolina): figure out what this parameter is

      \param delta_lambda
        \todo TODO(Carolina): figure out what this parameter is

      \param lambdag
        \todo TODO(Carolina): figure out what this parameter is
  */
  void UpdateLambdaNRMulti (std::vector<double> &lambda, double delta_lambda = -1.0e-2,
                            double lambdag = 1.0);

  // Some constants for
  double gamma = 0.5;
  double gammaMax = 0.25;

  int nDesVar;                                  //!< number of design variables
  int nCons;                                    //!< number of constraints
  std::vector<double> z;                        //!< design variable
  std::vector<double> z_up;                     //!< upper bounds of the design variables
  std::vector<double> z_lo;                     //!< lower bounds of the design variables
  std::vector<double> objFunGrad;               //!< objective function gradient
  std::vector<std::vector<double>> conFunGrad;  //!< constraint function gradient
  std::vector<double> conMaxVals;               //!< Constraint maximum values

  std::vector<double> bAreas;
  std::vector<bool> isActive;
  std::vector<double> targetConsVec;

  bool did_nr_converge = false;

  /*! \struct nlopt_cons_grad_data
      \brief nlopt data members
  */
  struct nlopt_cons_grad_data {
    double max_cons;
    std::vector<double> grad_data;
  };

 private:
  //! Helper function for SolveWithNR
  /*! \param lambda
        \todo TODO(Carolina): add description

      \param targetCons
        \todo TODO(Carolina): add description

      \return
        \todo TODO(Carolina): add description
  */
  double FOfLambdaNR (double lambda, double targetCons);

  //! Helper function for SolveWithNR
  /*! \return
        \todo TODO(Carolina): add description
  */
  double EstimateTargetConstraintNR ();

  //! Helper function for SolveWithNR for multiple constraints
  void EstimateTargetConstraintNRmulti ();

  //! helper function
  /*! \return
        \todo TODO(Carolina): add description
  */
  double NormalizeObjectiveGradients ();

  //! helper function
  /*! \param nSkip
        \todo TODO(Carolina): add description

      \return
        \todo TODO(Carolina): add description
  */
  std::vector<double> NormalizeConstraintGradients (int nSkip = -1);

  //! helper function to initialize Ipopt
  // void InitializeIpopt (Ipopt::SmartPtr<Ipopt::IpoptApplication> app);
};

}  // namespace para_lsm

#endif