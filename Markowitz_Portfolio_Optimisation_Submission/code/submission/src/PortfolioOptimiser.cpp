#include "PortfolioOptimiser.h"
#include "ConjugateGradientSolver.h"
#include "LinearSystemSolver.h"
#include <stdexcept>

PortfolioOptimiser::PortfolioOptimiser(const Matrix& sigma,
                                       const Matrix& meanReturns,
                                       const LinearSolver& solver)
    : sigma_(sigma), meanReturns_(meanReturns),
      solver_(solver), N_(meanReturns.rows())
{}

Matrix PortfolioOptimiser::optimise(double targetReturn) const {
    // the KKT system for Markowitz is:
    //   [ Sigma  -r_bar  -e ] [ w   ]   [  0  ]
    //   [ -r_bar'  0      0 ] [ lam ] = [ -rP ]
    //   [ -e'      0      0 ] [ mu  ]   [ -1  ]
    //
    // Q is indefinite so we can't apply CG to the full system directly.
    // instead use Schur complement: apply CG only to Sigma (positive definite)
    // then solve a tiny 2x2 system for the Lagrange multipliers

    Matrix e = onesVector(N_);

    // solve Sigma*A = r_bar and Sigma*B = e using CG
    Matrix A = solver_.solve(sigma_, meanReturns_);
    Matrix B = solver_.solve(sigma_, e);

    // build the 2x2 Schur system and solve for lambda and mu
    Matrix S(2, 2);
    S(0,0) = dot(meanReturns_, A);
    S(0,1) = dot(meanReturns_, B);
    S(1,0) = dot(e, A);
    S(1,1) = dot(e, B);

    Matrix rhs(2, 1);
    rhs(0,0) = targetReturn;
    rhs(1,0) = 1.0;

    Matrix lagrange = LinearSystemSolver::solve(S, rhs);
    double lambda = lagrange(0,0);
    double mu     = lagrange(1,0);

    // recover weights: w = lambda*A + mu*B
    Matrix w(N_, 1, 0.0);
    for (int i = 0; i < N_; ++i)
        w(i,0) = lambda * A(i,0) + mu * B(i,0);

    return w;
}
