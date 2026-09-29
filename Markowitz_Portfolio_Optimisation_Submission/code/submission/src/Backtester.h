#pragma once
#include "Matrix.h"
#include "LinearSolver.h"
#include <vector>
#include <string>

// holds the average OOS performance for one portfolio
struct BacktestResult {
    double targetReturn;   // target return used in optimisation
    double oosReturn;      // average realised return out of sample
    double oosVariance;    // average realised variance out of sample
    double oosStdDev;      // square root of variance
    double sharpeRatio;    // return / stddev (no risk free rate)
    double maxDrawdown;    // worst peak to trough loss over all folds
};

// runs the rolling walk-forward backtest
// splits data into 100-day in-sample and 12-day out-of-sample windows
// slides forward by 12 days each time for 50 folds total
class Backtester {
public:
    Backtester(const Matrix& allData,
               const LinearSolver& solver,
               int inSample = 100,
               int oosDays  = 12);

    // vanilla run using sample covariance
    void run(const std::vector<double>& targetReturns);

    // shrinkage run - uses Sigma_shrink = (1-delta)*Sigma + delta*F
    void runShrinkage(const std::vector<double>& targetReturns, double delta);

    // returns averaged results across all folds
    std::vector<BacktestResult> aggregateResults() const;

    // equal weight 1/N benchmark for comparison
    BacktestResult equalWeightBenchmark() const;

    // saves per-fold results to a CSV file for plotting
    void saveResultsToCSV(const std::string& filename) const;

private:
    Matrix extractWindow(int startRow, int numRows) const;
    void runImpl(const std::vector<double>& targetReturns, double shrinkageDelta);

    const Matrix& allData_;
    const LinearSolver& solver_;
    int T_, N_;
    int inSample_, oosDays_;
    int numFolds_;

    std::vector<double> targetReturns_;
    std::vector<std::vector<double>> foldOosReturns_;
    std::vector<std::vector<double>> foldOosVariances_;
    std::vector<double> ewOosReturns_;
    std::vector<double> ewOosVariances_;
};
