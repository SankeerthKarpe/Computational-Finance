#pragma once
#include "LinearSolver.h"
#include "Matrix.h"

// implements the conjugate gradient method from Algorithm 1 in the coursework
// only works for symmetric positive definite matrices
// we use this to solve Sigma*x = b inside the portfolio optimiser
class ConjugateGradientSolver : public LinearSolver {
public:
    // tol is the convergence tolerance epsilon from the spec (default 1e-6)
    explicit ConjugateGradientSolver(double tol = 1e-6, int maxIter = 10000);

    Matrix solve(const Matrix& Q, const Matrix& b) const override;

    int iterationsUsed() const { return itersUsed_; }

private:
    double tol_;
    int maxIter_;
    mutable int itersUsed_;
};
