#include "Matrix.h"
#include <cmath>
#include <iomanip>

Matrix::Matrix(int rows, int cols, double val)
    : rows_(rows), cols_(cols), data_(rows * cols, val)
{}

double& Matrix::operator()(int i, int j) {
    if (i < 0 || i >= rows_ || j < 0 || j >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[i * cols_ + j];
}

const double& Matrix::operator()(int i, int j) const {
    if (i < 0 || i >= rows_ || j < 0 || j >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[i * cols_ + j];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; i++)
        result.data_[i] = data_[i] + other.data_[i];
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions don't match for subtraction");
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
        result.data_[i] = data_[i] - other.data_[i];
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    Matrix result(rows_, other.cols_, 0.0);
    for (int i = 0; i < rows_; ++i)
        for (int k = 0; k < cols_; ++k)
            for (int j = 0; j < other.cols_; ++j)
                result(i,j) += (*this)(i,k) * other(k,j);
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
        result.data_[i] = data_[i] * scalar;
    return result;
}

Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (int i = 0; i < rows_; ++i)
        for (int j = 0; j < cols_; ++j)
            result(j,i) = (*this)(i,j);
    return result;
}

void Matrix::print() const {
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j)
            std::cout << std::setw(12) << std::setprecision(6) << (*this)(i,j);
        std::cout << "\n";
    }
}

double dot(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != 1 || b.cols() != 1)
        throw std::invalid_argument("dot() needs two column vectors of the same size");
    double sum = 0.0;
    for (int i = 0; i < a.rows(); ++i)
        sum += a(i,0) * b(i,0);
    return sum;
}

Matrix zeroVector(int n) {
    return Matrix(n, 1, 0.0);
}

Matrix onesVector(int n) {
    return Matrix(n, 1, 1.0);
}
