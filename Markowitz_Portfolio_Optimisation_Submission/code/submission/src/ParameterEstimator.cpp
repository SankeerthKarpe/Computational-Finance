#include "ParameterEstimator.h"

ParameterEstimator::ParameterEstimator(const Matrix& data)
    : data_(data),
      T_(data.rows()),
      N_(data.cols()),
      meanReturns_(N_, 1, 0.0),
      covMatrix_(N_, N_, 0.0)
{}

void ParameterEstimator::estimate() {
    // compute mean return for each asset over the in-sample window
    // formula: r_bar_i = (1/n) * sum of r_{i,k} for k = 1 to n
    for (int i = 0; i < N_; ++i) {
        double sum = 0.0;
        for (int k = 0; k < T_; ++k)
            sum += data_(k, i);
        meanReturns_(i, 0) = sum / T_;
    }

    // compute sample covariance matrix
    // formula: Sigma_{ij} = 1/(n-1) * sum of (r_ik - r_bar_i)(r_jk - r_bar_j)
    // only compute upper triangle then copy to lower since the matrix is symmetric
    for (int i = 0; i < N_; ++i) {
        for (int j = i; j < N_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < T_; ++k) {
                double di = data_(k, i) - meanReturns_(i, 0);
                double dj = data_(k, j) - meanReturns_(j, 0);
                sum += di * dj;
            }
            double cov = sum / (T_ - 1);
            covMatrix_(i, j) = cov;
            covMatrix_(j, i) = cov;
        }
    }
}

Matrix ParameterEstimator::getShrinkageCov(double delta) const {
    // shrinkage estimator: blend sample covariance with diagonal target
    // Sigma_shrink = (1 - delta) * Sigma + delta * F
    // F is just the diagonal of Sigma (variances only, no cross terms)
    // this reduces the noise in the off-diagonal entries which are unreliable
    // with only 100 observations for 83 assets

    Matrix F(N_, N_, 0.0);
    for (int i = 0; i < N_; ++i)
        F(i, i) = covMatrix_(i, i);

    Matrix shrink(N_, N_, 0.0);
    for (int i = 0; i < N_; ++i)
        for (int j = 0; j < N_; ++j)
            shrink(i, j) = (1.0 - delta) * covMatrix_(i, j)
                         + delta         * F(i, j);

    return shrink;
}
