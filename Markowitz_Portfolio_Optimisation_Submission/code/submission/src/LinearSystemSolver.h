#pragma once
#include "Matrix.h"

// solves Ax = b using Gaussian elimination with partial pivoting
// used for the small 2x2 Schur complement system in the portfolio optimiser
class LinearSystemSolver {
public:
    static Matrix solve(const Matrix& A, const Matrix& b);
};
