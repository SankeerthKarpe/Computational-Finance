#pragma once
#include "Matrix.h"
#include <string>
#include <vector>

// reads asset returns from a CSV file
// returns a 2D vector of doubles (safe on Windows)
std::vector<std::vector<double>> loadCSVRaw(const std::string& filename);

// convenience wrapper that returns a Matrix directly
Matrix loadCSVMatrix(const std::string& filename);
