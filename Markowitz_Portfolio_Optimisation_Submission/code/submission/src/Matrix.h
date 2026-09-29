#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>

// general N x M matrix class
// stores data in a flat vector on the heap to avoid stack overflow
// with large matrices like the 83x83 covariance matrix
class Matrix {
public:
    Matrix(int rows, int cols, double val = 0.0);
    Matrix() : rows_(0), cols_(0) {}

    // element access with bounds checking
    double& operator()(int i, int j);
    const double& operator()(int i, int j) const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    friend Matrix operator*(double scalar, const Matrix& m);

    Matrix transpose() const;
    void print() const;

private:
    int rows_, cols_;
    std::vector<double> data_;
};

// dot product of two column vectors
double dot(const Matrix& a, const Matrix& b);

// helper to create zero and ones vectors
Matrix zeroVector(int n);
Matrix onesVector(int n);
