#include "LinearSystemSolver.h"
#include <cmath>
#include <stdexcept>
#include <vector>

Matrix LinearSystemSolver::solve(const Matrix& A, const Matrix& b) {
    int n = A.rows();
    if (A.cols() != n || b.rows() != n || b.cols() != 1)
        throw std::invalid_argument("LinearSystemSolver: dimension mismatch");

    // copy into plain arrays so we can modify them during elimination
    std::vector<std::vector<double>> a(n, std::vector<double>(n));
    std::vector<double> rhs(n);
    for (int i = 0; i < n; ++i) {
        rhs[i] = b(i, 0);
        for (int j = 0; j < n; ++j)
            a[i][j] = A(i, j);
    }

    // forward elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        int pivotRow = col;
        double maxVal = std::abs(a[col][col]);
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(a[row][col]) > maxVal) {
                maxVal = std::abs(a[row][col]);
                pivotRow = row;
            }
        }
        if (maxVal < 1e-14)
            throw std::runtime_error("LinearSystemSolver: singular matrix");

        if (pivotRow != col) {
            std::swap(a[col], a[pivotRow]);
            std::swap(rhs[col], rhs[pivotRow]);
        }

        for (int row = col + 1; row < n; ++row) {
            double factor = a[row][col] / a[col][col];
            for (int j = col; j < n; ++j)
                a[row][j] -= factor * a[col][j];
            rhs[row] -= factor * rhs[col];
        }
    }

    // back substitution
    Matrix x(n, 1, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = rhs[i];
        for (int j = i + 1; j < n; ++j)
            sum -= a[i][j] * x(j, 0);
        x(i, 0) = sum / a[i][i];
    }
    return x;
}
