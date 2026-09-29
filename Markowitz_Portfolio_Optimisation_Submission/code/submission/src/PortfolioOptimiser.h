#pragma once
#include "Matrix.h"
#include "LinearSolver.h"

// solves the Markowitz mean-variance optimisation problem
// minimise w'*Sigma*w subject to r_bar'*w = rP and e'*w = 1
//
// the KKT conditions give a (N+2) x (N+2) linear system which we solve
// using the Schur complement approach since the full KKT matrix is indefinite
class PortfolioOptimiser {
public:
    PortfolioOptimiser(const Matrix& sigma,
                       const Matrix& meanReturns,
                       const LinearSolver& solver);

    // returns the optimal weight vector for a given target return
    Matrix optimise(double targetReturn) const;

private:
    const Matrix& sigma_;
    const Matrix& meanReturns_;
    const LinearSolver& solver_;
    int N_;
};
