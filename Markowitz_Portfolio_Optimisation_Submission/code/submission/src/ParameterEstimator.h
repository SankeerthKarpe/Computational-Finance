#pragma once
#include "Matrix.h"

// estimates mean returns and covariance matrix from a window of return data
// the data matrix has shape T x N (rows = days, cols = assets)
class ParameterEstimator {
public:
    explicit ParameterEstimator(const Matrix& data);

    // run the estimation - must call this before getting results
    void estimate();

    const Matrix& getMeanReturns() const { return meanReturns_; }
    const Matrix& getCovMatrix()   const { return covMatrix_;   }

    // shrinkage covariance: blends sample cov with diagonal target
    // Sigma_shrink = (1-delta)*Sigma + delta*F
    // helps with stability when n is small relative to N
    Matrix getShrinkageCov(double delta) const;

private:
    const Matrix& data_;
    int T_;  // number of time periods
    int N_;  // number of assets

    Matrix meanReturns_;
    Matrix covMatrix_;
};
