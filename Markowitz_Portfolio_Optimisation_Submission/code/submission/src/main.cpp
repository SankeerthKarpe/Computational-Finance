// main.cpp
// Markowitz portfolio optimisation with rolling backtest
// Runs vanilla and shrinkage versions and prints results

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "csv_reader.h"
#include "Matrix.h"
#include "ConjugateGradientSolver.h"
#include "Backtester.h"

// creates 20 evenly spaced target returns from 0% to 10%
static std::vector<double> buildTargets(int n, double minT, double maxT) {
    std::vector<double> t(n);
    for (int i = 0; i < n; ++i)
        t[i] = minT + i * (maxT - minT) / (n - 1);
    return t;
}

// prints the side by side results table to the terminal
static void printTable(const std::vector<BacktestResult>& vanilla,
                       const std::vector<BacktestResult>& shrink,
                       const BacktestResult& ew) {
    const int W = 13;
    std::cout << "\n=== RESULTS (50 folds, per-period figures) ===\n";
    std::cout << std::setw(10) << "Target%"
              << std::setw(W)  << "V.Ret%"
              << std::setw(W)  << "V.Std%"
              << std::setw(W)  << "V.Sharpe"
              << std::setw(W)  << "V.MaxDD%"
              << std::setw(W)  << "S.Ret%"
              << std::setw(W)  << "S.Std%"
              << std::setw(W)  << "S.Sharpe"
              << std::setw(W)  << "S.MaxDD%"
              << "\n" << std::string(10 + W * 8, '-') << "\n";

    std::cout << std::fixed << std::setprecision(4);
    for (size_t t = 0; t < vanilla.size(); ++t) {
        const auto& v = vanilla[t];
        const auto& s = shrink[t];
        std::cout << std::setw(9)  << v.targetReturn * 100 << "%"
                  << std::setw(W)  << v.oosReturn    * 100 << "%"
                  << std::setw(W)  << v.oosStdDev    * 100 << "%"
                  << std::setw(W)  << v.sharpeRatio
                  << std::setw(W)  << v.maxDrawdown  * 100 << "%"
                  << std::setw(W)  << s.oosReturn    * 100 << "%"
                  << std::setw(W)  << s.oosStdDev    * 100 << "%"
                  << std::setw(W)  << s.sharpeRatio
                  << std::setw(W)  << s.maxDrawdown  * 100 << "%"
                  << "\n";
    }
    std::cout << std::string(10 + W * 8, '-') << "\n";
    std::cout << std::setw(9)  << "EW BM"
              << std::setw(W)  << ew.oosReturn  * 100 << "%"
              << std::setw(W)  << ew.oosStdDev  * 100 << "%"
              << std::setw(W)  << ew.sharpeRatio
              << std::setw(W)  << ew.maxDrawdown * 100 << "%"
              << "\n\n";
    std::cout.flush();
}

// saves the aggregate results to a CSV file
// using FILE* instead of ofstream because ofstream crashes on Windows
static void writeAggregateCSV(const std::string& filename,
                               const std::vector<BacktestResult>& vanilla,
                               const std::vector<BacktestResult>& shrink,
                               const BacktestResult& ew) {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) { std::cerr << "Cannot open " << filename << "\n"; return; }

    fprintf(f, "method,target_return,oos_return,oos_variance,oos_stddev,sharpe_ratio,max_drawdown\n");

    for (const auto& r : vanilla)
        fprintf(f, "vanilla,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                r.targetReturn, r.oosReturn, r.oosVariance,
                r.oosStdDev, r.sharpeRatio, r.maxDrawdown);

    for (const auto& r : shrink)
        fprintf(f, "shrinkage,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                r.targetReturn, r.oosReturn, r.oosVariance,
                r.oosStdDev, r.sharpeRatio, r.maxDrawdown);

    fprintf(f, "benchmark,-1,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            ew.oosReturn, ew.oosVariance, ew.oosStdDev,
            ew.sharpeRatio, ew.maxDrawdown);

    fflush(f);
    fclose(f);
    printf("Aggregate results saved to %s\n", filename.c_str());
    fflush(stdout);
}

int main(int argc, char* argv[]) {
    const std::string dataFile = (argc > 1) ? argv[1] : "asset_returns.csv";

    // load the returns data
    printf("Loading: %s\n", dataFile.c_str());
    fflush(stdout);
    Matrix* allData = new Matrix(loadCSVMatrix(dataFile));

    // 20 target returns from 0% to 10%
    const std::vector<double> targets = buildTargets(20, 0.0, 0.10);

    // single CG solver shared by both backtests
    ConjugateGradientSolver* cgSolver = new ConjugateGradientSolver(1e-6, 10000);

    // run vanilla Markowitz backtest
    printf("\nVanilla Markowitz...\n"); fflush(stdout);
    Backtester* btVanilla = new Backtester(*allData, *cgSolver);
    btVanilla->run(targets);
    std::vector<BacktestResult> vanilla = btVanilla->aggregateResults();
    BacktestResult ew = btVanilla->equalWeightBenchmark();
    btVanilla->saveResultsToCSV("fold_results_vanilla.csv");
    delete btVanilla;
    printf("Vanilla done.\n"); fflush(stdout);

    // run shrinkage Markowitz backtest (delta = 0.4 worked well in testing)
    printf("\nShrinkage Markowitz (delta=0.4)...\n"); fflush(stdout);
    Backtester* btShrink = new Backtester(*allData, *cgSolver);
    btShrink->runShrinkage(targets, 0.4);
    std::vector<BacktestResult> shrink = btShrink->aggregateResults();
    btShrink->saveResultsToCSV("fold_results_shrinkage.csv");
    delete btShrink;
    printf("Shrinkage done.\n"); fflush(stdout);

    // print and save results
    printTable(vanilla, shrink, ew);
    writeAggregateCSV("aggregate_results.csv", vanilla, shrink, ew);

    printf("\nDone.\n"); fflush(stdout);

    // using _Exit to avoid a crash on Windows during cleanup
    _Exit(0);
}
