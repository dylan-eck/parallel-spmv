#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <stdlib.h>
#include <time.h>

#define EIGEN_MPL2_ONLY
#include "./eigen-3.4.0/Eigen/Sparse"
#include "./eigen-3.4.0/Eigen/Dense"

struct Element {
    long row;
    long col;
    double value;

    bool operator<(const Element &other) const {
        return row < other.row;
    }
};

struct Matrix {
    long numRows;
    long numCols;
    long numVals;
    std::vector<Element> elements;

    Element& operator[](long row) {
        return elements[row];
    }
};

void load_matrix_template(const char* filepath, Matrix& matrix) {
    std::string line;
    std::ifstream inputFile;
    inputFile.open(filepath);

    // consume comments at start of file
    while (std::getline(inputFile, line) && line[0] == '%');

    char *end = nullptr;
    matrix.numRows = std::strtol(line.c_str(), &end, 10);
    matrix.numCols = std::strtol(end, &end, 10);
    matrix.numVals = std::strtol(end, &end, 10);

    matrix.elements.reserve(matrix.numVals * 2);

    while (std::getline(inputFile, line)) {
        Element elem;
        elem.row = std::strtol(line.c_str(), &end, 10) - 1;
        elem.col = std::strtol(end, &end, 10) - 1;
        elem.value = (double)rand() / RAND_MAX * 4.8f + 0.1f;

        matrix.elements.push_back(elem);
    }
    std::sort(matrix.elements.begin(), matrix.elements.end());
    inputFile.close();
}

Eigen::SparseMatrix<double> ref_format(const Matrix& matrix) {
    std::vector< Eigen::Triplet<double> > triplets;
    triplets.reserve(matrix.numRows);
    for (const Element& element : matrix.elements) {
        triplets.push_back(Eigen::Triplet<double>(element.row, element.col, element.value));
    }

    Eigen::SparseMatrix<double> ref_mat(matrix.numRows, matrix.numCols);
    ref_mat.setFromTriplets(triplets.begin(), triplets.end());
    return ref_mat;
}

int main(int argc, char const *argv[])
{
    std::srand(std::time(NULL));
    std::rand();

    Matrix matrix;
    load_matrix_template("./NLR.mtx", matrix);
    Eigen::SparseMatrix<double> ref_matrix = ref_format(matrix);

    Matrix vec;
    vec.numRows = matrix.numCols;
    vec.numCols = 1;
    vec.elements = std::vector<Element>(vec.numRows);

    Eigen::VectorXd ref_vec(matrix.numCols);

    for (int i = 0; i < vec.numRows; i++) {
        vec[i] = Element{i, 0, (double)rand() / RAND_MAX * 4.8f + 0.1f};
        ref_vec(i) = vec[i].value;
    }

    Matrix res;
    res.numRows = matrix.numCols;
    res.numCols = 1;
    res.elements = std::vector<Element>(res.numRows);

    Eigen::VectorXd ref_res(matrix.numCols);
    
    ref_res = ref_matrix * ref_vec;

    auto startTimePoint = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < matrix.elements.size(); i++) {
        res[matrix[i].row].value += matrix[i].value * vec[matrix[i].col].value;
    }

    auto endTimePoint = std::chrono::high_resolution_clock::now();

    auto tstart = std::chrono::time_point_cast<std::chrono::microseconds>(startTimePoint).time_since_epoch().count();
    auto tend = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();

    float duration = (tend - tstart) * 0.001f;
    std::cout << "multiplication execution time: " << duration << "ms" << std::endl;

    bool answer_correct = true;
    for (int i = 0; i < vec.numRows; i++) {
        if (std::abs(res[i].value - ref_res(i)) > 0.1f) {
            std::cout << "result is not correct (row " << i << "): " << res[i].value << " != " << ref_res(i) << std::endl;
            answer_correct = false;
            break;
        }
    }

    if (answer_correct) {
        std::cout << "result is correct" << std::endl;
    }

    return EXIT_SUCCESS;
}