#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <stdlib.h>
#include <time.h>

#define EIGEN_MPL2_ONLY // only use portions of Eigen that have an MPL2 license 
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <omp.h>

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
    std::vector<long> row_indexes;
};

void load_matrix_template(const char* filepath, Matrix& matrix) {
    std::string line;
    std::ifstream inputFile;
    inputFile.open(filepath);
    char *end = nullptr;

    // consume comments at start of file
    while (std::getline(inputFile, line) && line[0] == '%');

    matrix.numRows = std::strtol(line.c_str(), &end, 10);
    matrix.numCols = std::strtol(end, &end, 10);
    matrix.numVals = std::strtol(end, &end, 10);

    matrix.elements = std::vector<Element>(matrix.numVals);
    matrix.row_indexes = std::vector<long>(matrix.numRows + 1);

    for (Element& elem : matrix.elements) {
        std::getline(inputFile, line);
        elem.row = std::strtol(line.c_str(), &end, 10) - 1;
        elem.col = std::strtol(end, &end, 10) - 1;
        elem.value = (double)rand() / RAND_MAX * 4.8f + 0.1f;
    }
    std::sort(matrix.elements.begin(), matrix.elements.end());

    long row_index = 0;
    int ecount = 0;
    long total = 0;
    for (int i = 0; i < matrix.numVals; ++i) {
        Element& elem = matrix.elements[i];
        if (elem.row != row_index) {
            for (int j = row_index + 1; j < elem.row; ++j) {
                matrix.row_indexes[j] = total + ecount;
            }
            row_index = elem.row;
            total += ecount;
            matrix.row_indexes[row_index] = total;
            ecount = 0;
        }
        ecount += 1;
    }
    matrix.row_indexes[matrix.numRows] = matrix.numVals;

    // for (const auto& e : matrix.elements) {
    //     std::cout << e.row << " " << e.col << " " << e.value << std::endl;
    // }
    // std::cout << std::endl;

    // std::cout << "0 1 2 3 4 5 6 7" << std::endl;
    // for (const auto& idx : matrix.row_indexes) {
    //     std::cout << idx << " ";
    // }
    // std::cout << std::endl << std::endl;

    inputFile.close();
}

std::mutex m;
void t_func(
    long s_idx,
    long e_idx,
    const Matrix& matrix,
    const std::vector<double>& vector,
    std::vector<double>& result
    ) {


    for (int i = s_idx; i < e_idx; ++i) {
        long rstart = matrix.row_indexes[i];
        long rend = matrix.row_indexes[i + 1];
        if (rstart == rend) continue;

        double accum = 0;
        for (int j = rstart; j < rend; ++j) {
            double a = matrix.elements[j].value;
            double b = vector[matrix.elements[j].col];
            accum += a * b;
        }
        result[i] = accum;
    }
}

void parallel_spmv(Matrix& matrix, std::vector<double>& vector, std::vector<double>& result) {
    int concurrency = std::thread::hardware_concurrency();
    // int n_threads = std::floor((0.8 * concurrency) / 2) * 2;
    int n_threads = 2;
    long chunk_size = matrix.numRows / n_threads;

    std::vector<std::thread> threads(n_threads);

    for (int i = 0; i < n_threads; i++) {
        long s_idx = i * chunk_size;
        long e_idx;
        if (i < n_threads - 1) {
            e_idx = s_idx + chunk_size;
        } else {
            e_idx = matrix.numRows;
    }

        threads[i] = std::thread(t_func, s_idx, e_idx, matrix, vector, std::ref(result));
    }

    for (int i = 0; i < n_threads; i++) {
        threads[i].join();
    }

    // for (int i = 0; i < matrix.numRows; ++i) {
    //     long rstart = matrix.row_indexes[i];
    //     long rend = matrix.row_indexes[i + 1];
    //     if (rstart == rend) continue;

    //     double accum = 0;
    //     for (int j = rstart; j < rend; ++j) {
    //         double a = matrix.elements[j].value;
    //         double b = vector[matrix.elements[j].col];
    //         accum += a * b;
    //     }
    //     result[i] = accum;
    // }
}

bool verify_result(
    const Matrix& matrix,
    std::vector<double>& vector,
    std::vector<double>& result) {

    std::vector< Eigen::Triplet<double> > triplets;
    triplets.reserve(matrix.numVals);
    for (const Element& element : matrix.elements) {
        triplets.push_back(
            Eigen::Triplet<double>(element.row, element.col, element.value)
        );
    }

    Eigen::SparseMatrix<double> ref_mat(matrix.numRows, matrix.numCols);
    ref_mat.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd ref_vec(matrix.numCols);
    for (int i = 0; i < vector.size(); ++i) {
        ref_vec(i) = vector[i];
    }

    Eigen::VectorXd ref_res = ref_mat * ref_vec;

    for (int i = 0; i < vector.size(); ++i) {
        if (std::abs(result[i] - ref_res(i)) > 0.1f) {
            return false;
        }
    }
    return true;
}

int main(int argc, char const *argv[])
{
    std::srand(std::time(NULL));
    std::rand();

    const char *input_files[3] = {
        "NLR.mtx",
        "channel-500x100x100-b050.mtx",
        "delaunay_n19.mtx"
    };

    float exec_times[3]{};

    for (int i = 0; i < 3; i++) {
        Matrix matrix;
        load_matrix_template(input_files[i], matrix);

        std::vector<double> vector(matrix.numCols);
        std::generate(
            vector.begin(),
            vector.end(),
            []() {return (double)rand() / RAND_MAX * 4.8f + 0.1f;}
        );

        std::vector<double> result(matrix.numCols);
        std::fill(result.begin(), result.end(), 0.0f);

        auto startTimePoint = std::chrono::high_resolution_clock::now();

        parallel_spmv(matrix, vector, result);

        auto endTimePoint = std::chrono::high_resolution_clock::now();

        auto tstart =
            std::chrono::time_point_cast<std::chrono::microseconds>(startTimePoint)
            .time_since_epoch()
            .count();

        auto tend =
            std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint)
            .time_since_epoch()
            .count();

        float duration = (tend - tstart) * 0.001f;
        exec_times[i] = duration;

        bool correct = verify_result(matrix, vector, result);
        if (!correct) {
            throw std::runtime_error(
                "matrix multiplication produced incorrect result"
            );
        }
    }

    for (int i = 0; i < 3; i++) {
        std::cout
            << std::setw(30)
            << std::left
            << input_files[i];

        std::cout
            << std::setw(6)
            << std::right
            << std::fixed
            << std::setprecision(0)
            << exec_times[i] << "ms"
            << std::endl;
    }

    return EXIT_SUCCESS;
}
