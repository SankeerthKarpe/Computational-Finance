#include "Backtester.h"
#include "ParameterEstimator.h"
#include "PortfolioOptimiser.h"
#include <cmath>
// #include <fstream> -- replaced with FILE*
#include <iostream>
#include <stdexcept>

Backtester::Backtester(const Matrix&       allData,
                       const LinearSolver& solver,
                       int inSample,
                       int oosDays)
    : allData_(allData),
      solver_(solver),
      T_(allData.rows()),
      N_(allData.cols()),
      inSample_(inSample),
      oosDays_(oosDays),
      numFolds_(0)
{
    if (inSample_ <= 1)
        throw std::invalid_argument("Backtester: in-sample size must be > 1");
    if (oosDays_ <= 0)
        throw std::invalid_argument("Backtester: out-of-sample size must be > 0");
    if (T_ < inSample_ + oosDays_)
        throw std::invalid_argument("Backtester: not enough data for one fold");
}

Matrix Backtester::extractWindow(int startRow, int numRows) const {
    if (startRow < 0 || startRow + numRows > T_)
        throw std::out_of_range("Backtester::extractWindow: index out of bounds");
    Matrix w(numRows, N_);
    for (int r = 0; r < numRows; r++)
        for (int c = 0; c < N_; c++)
            w(r, c) = allData_(startRow + r, c);
    return w;
}

void Backtester::runImpl(const std::vector<double>& targetReturns,
                         double shrinkageDelta) {
    int numTargets = (int)targetReturns.size();

    targetReturns_ = targetReturns;
    foldOosReturns_.clear();
    foldOosVariances_.clear();
    ewOosReturns_.clear();
    ewOosVariances_.clear();
    numFolds_ = 0;

    for (int start = 0;
         start + inSample_ + oosDays_ <= T_;
         start += oosDays_)
    {
        // In-sample: estimate parameters
        Matrix isData = extractWindow(start, inSample_);
        ParameterEstimator isEst(isData);
        isEst.estimate();

        const Matrix isMean = isEst.getMeanReturns();
        const Matrix isCov  = (shrinkageDelta > 0.0)
                            ? isEst.getShrinkageCov(shrinkageDelta)
                            : isEst.getCovMatrix();

        // Out-of-sample: realised parameters
        Matrix oosData = extractWindow(start + inSample_, oosDays_);
        ParameterEstimator oosEst(oosData);
        oosEst.estimate();
        const Matrix oosMean = oosEst.getMeanReturns();
        const Matrix oosCov  = oosEst.getCovMatrix();

        // Optimise using IN-SAMPLE parameters only
        PortfolioOptimiser opt(isCov, isMean, solver_);

        std::vector<double> foldRet(numTargets, 0.0);
        std::vector<double> foldVar(numTargets, 0.0);

        for (int t = 0; t < numTargets; ++t) {
            Matrix w = opt.optimise(targetReturns[t]);

            // OOS return: r_bar_oos^T * w
            double ret = 0.0;
            for (int i = 0; i < N_; ++i)
                ret += oosMean(i, 0) * w(i, 0);

            // OOS variance: w^T * Sigma_oos * w
            double var = 0.0;
            for (int i = 0; i < N_; ++i) {
                double tmp = 0.0;
                for (int j = 0; j < N_; ++j)
                    tmp += oosCov(i, j) * w(j, 0);
                var += w(i, 0) * tmp;
            }

            foldRet[t] = ret;
            foldVar[t] = var;
        }

        foldOosReturns_.push_back(foldRet);
        foldOosVariances_.push_back(foldVar);

        // Equal-weight benchmark
        const double ewW = 1.0 / N_;
        double ewRet = 0.0, ewVar = 0.0;
        for (int i = 0; i < N_; ++i)
            ewRet += ewW * oosMean(i, 0);
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                ewVar += ewW * oosCov(i, j) * ewW;

        ewOosReturns_.push_back(ewRet);
        ewOosVariances_.push_back(ewVar);

        ++numFolds_;
    }

    std::cout << "Backtest complete: " << numFolds_ << " folds.\n";
}

void Backtester::run(const std::vector<double>& targetReturns) {
    runImpl(targetReturns, 0.0);
}

void Backtester::runShrinkage(const std::vector<double>& targetReturns,
                               double delta) {
    runImpl(targetReturns, delta);
}

std::vector<BacktestResult> Backtester::aggregateResults() const {
    if (numFolds_ == 0)
        throw std::runtime_error("Backtester: call run() before aggregateResults()");

    int numTargets = (int)targetReturns_.size();
    std::vector<BacktestResult> results(numTargets);

    for (int t = 0; t < numTargets; ++t) {
        double sumRet = 0.0, sumVar = 0.0;
        for (int f = 0; f < numFolds_; ++f) {
            sumRet += foldOosReturns_[f][t];
            sumVar += foldOosVariances_[f][t];
        }
        const double avgRet = sumRet / numFolds_;
        const double avgVar = sumVar / numFolds_;

        // Maximum drawdown: peak cumulative return minus trough
        double peak = 0.0, cumRet = 0.0, maxDD = 0.0;
        for (int f = 0; f < numFolds_; ++f) {
            cumRet += foldOosReturns_[f][t];
            if (cumRet > peak) peak = cumRet;
            double dd = peak - cumRet;
            if (dd > maxDD) maxDD = dd;
        }

        BacktestResult& r = results[t];
        r.targetReturn = targetReturns_[t];
        r.oosReturn    = avgRet;
        r.oosVariance  = avgVar;
        r.oosStdDev    = std::sqrt(std::max(avgVar, 0.0));
        r.sharpeRatio  = (r.oosStdDev > 1e-12) ? (avgRet / r.oosStdDev) : 0.0;
        r.maxDrawdown  = -maxDD;  // reported as negative (loss)
    }
    return results;
}

BacktestResult Backtester::equalWeightBenchmark() const {
    if (numFolds_ == 0)
        throw std::runtime_error("Backtester: call run() before equalWeightBenchmark()");

    double sumRet = 0.0, sumVar = 0.0;
    double peak = 0.0, cumRet = 0.0, maxDD = 0.0;
    for (int f = 0; f < numFolds_; ++f) {
        sumRet  += ewOosReturns_[f];
        sumVar  += ewOosVariances_[f];
        cumRet  += ewOosReturns_[f];
        if (cumRet > peak) peak = cumRet;
        double dd = peak - cumRet;
        if (dd > maxDD) maxDD = dd;
    }

    BacktestResult r;
    r.targetReturn = -1.0;
    r.oosReturn    = sumRet / numFolds_;
    r.oosVariance  = sumVar / numFolds_;
    r.oosStdDev    = std::sqrt(std::max(r.oosVariance, 0.0));
    r.sharpeRatio  = (r.oosStdDev > 1e-12) ? (r.oosReturn / r.oosStdDev) : 0.0;
    r.maxDrawdown  = -maxDD;
    return r;
}

void Backtester::saveResultsToCSV(const std::string& filename) const {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Backtester: cannot open %s\n", filename.c_str());
        return;
    }

    fprintf(f, "fold,target_return,oos_return,oos_variance\n");
    int numTargets = (int)targetReturns_.size();
    for (int fold = 0; fold < numFolds_; ++fold)
        for (int t = 0; t < numTargets; ++t)
            fprintf(f, "%d,%.8f,%.8f,%.8f\n",
                    fold,
                    targetReturns_[t],
                    foldOosReturns_[fold][t],
                    foldOosVariances_[fold][t]);

    fflush(f);
    fclose(f);
    printf("Per-fold results saved to %s\n", filename.c_str());
    fflush(stdout);
}
