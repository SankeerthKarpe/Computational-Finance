#include "csv_reader.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cstdlib>

std::vector<std::vector<double>> loadCSVRaw(const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f)
        throw std::runtime_error("Cannot open: " + filename);

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buf = new char[fsize + 1];
    if (fread(buf, 1, fsize, f) != static_cast<size_t>(fsize))
        throw std::runtime_error("File read error: " + filename);
    fclose(f);
    buf[fsize] = '\0';

    std::vector<std::vector<double>> rows;
    rows.reserve(800);

    char* p   = buf;
    char* end = buf + fsize;

    while (p < end) {
        char* lineEnd = p;
        while (lineEnd < end && *lineEnd != '\n' && *lineEnd != '\r')
            ++lineEnd;

        std::vector<double> row;
        row.reserve(90);

        char* q = p;
        while (q <= lineEnd) {
            while (q < lineEnd && (*q == ' ' || *q == '\t')) ++q;

            char* fieldEnd = q;
            while (fieldEnd < lineEnd && *fieldEnd != ',') ++fieldEnd;

            if (fieldEnd > q) {
                char tmp[64];
                int len = (int)(fieldEnd - q);
                if (len < 63) {
                    memcpy(tmp, q, len);
                    tmp[len] = '\0';
                    char* endptr;
                    double val = strtod(tmp, &endptr);
                    if (endptr != tmp)
                        row.push_back(val);
                }
            }
            q = fieldEnd + 1;
        }

        if (!row.empty())
            rows.push_back(row);

        while (lineEnd < end && (*lineEnd == '\n' || *lineEnd == '\r'))
            ++lineEnd;
        p = lineEnd;
    }

    delete[] buf;

    if (rows.empty())
        throw std::runtime_error("No data in: " + filename);

    std::cout << "Loaded: " << rows.size() << " periods x "
              << rows[0].size() << " assets\n";
    std::cout.flush();
    return rows;
}

// Fixed: construct Matrix directly on the stack -- no unnecessary heap allocation
Matrix loadCSVMatrix(const std::string& filename) {
    auto rows = loadCSVRaw(filename);
    int T = (int)rows.size();
    int N = (int)rows[0].size();
    Matrix data(T, N);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n)
            data(t, n) = rows[t][n];
    return data;
}
