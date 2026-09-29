#include "ConjugateGradientSolver.h"
#include <cmath>
#include <stdexcept>

ConjugateGradientSolver::ConjugateGradientSolver(double tol, int maxIter)
    : tol_(tol), maxIter_(maxIter), itersUsed_(0)
{}

Matrix ConjugateGradientSolver::solve(const Matrix& Q, const Matrix& b) const {
    int n = b.rows();

    if (Q.rows() != n || Q.cols() != n || b.cols() != 1)
        throw std::invalid_argument("CG: Q must be n x n and b must be n x 1");

    // start from the zero vector
    Matrix x(n, 1, 0.0);

    // initialise residual and search direction as in Algorithm 1
    Matrix s = b;   // s0 = b - Q*x0 = b since x0 = 0
    Matrix p = s;

    double ss = dot(s, s);
    itersUsed_ = 0;

    // already converged before we start
    if (ss <= tol_)
        return x;

    while (itersUsed_ < maxIter_) {
        Matrix Qp  = Q * p;
        double pQp = dot(p, Qp);

        // if the denominator is basically zero we can't continue
        if (std::abs(pQp) < 1e-15) break;

        double alpha = ss / pQp;

        // update x and residual
        for (int i = 0; i < n; ++i)
            x(i, 0) += alpha * p(i, 0);

        for (int i = 0; i < n; ++i)
            s(i, 0) -= alpha * Qp(i, 0);

        double ss_new = dot(s, s);

        // check convergence: stop when s^T s <= epsilon
        if (ss_new <= tol_) {
            ss = ss_new;
            ++itersUsed_;
            break;
        }

        double beta = ss_new / ss;

        // update search direction
        for (int i = 0; i < n; ++i)
            p(i, 0) = s(i, 0) + beta * p(i, 0);

        ss = ss_new;
        ++itersUsed_;
    }

    return x;
}
