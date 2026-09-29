#pragma once
#include "Matrix.h"

// abstract base class for linear solvers
// by depending on this interface rather than a concrete class,
// we can swap in different solvers without changing the optimiser or backtester
class LinearSolver {
public:
    virtual ~LinearSolver() = default;
    virtual Matrix solve(const Matrix& A, const Matrix& b) const = 0;
};
