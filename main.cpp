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

std::mutex m;
void t_func() {
    m.lock();
    std::cout << "hello from thread " << std::this_thread::get_id() << std::endl;
    m.unlock();
}

void parallel_spmv(Matrix& matrix, Matrix& vector, Matrix& result) {
    int concurrency = std::thread::hardware_concurrency();
    int n_threads = std::floor((0.8 * concurrency) / 2) * 2;
    std::cout << "using " << n_threads << " threads" << std::endl;

    std::vector<std::thread> threads(n_threads);

    for (int i = 0; i < n_threads; i++) {
        threads[i] = std::thread(t_func);
    }

    for (int i = 0; i < n_threads; i++) {
        threads[i].join();
    }

    for (int i = 0; i < matrix.numVals; i++) {
        result[matrix[i].row].value += matrix[i].value * vector[matrix[i].col].value;
    }
}

bool verify_result(
    const Matrix& matrix,
    Matrix& vector,
    Matrix& result) {

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
    for (int i = 0; i < vector.numRows; i++) {
        ref_vec(i) = vector[i].value;
    }

    Eigen::VectorXd ref_res = ref_mat * ref_vec;

    for (int i = 0; i < vector.numVals; i++) {
        if (std::abs(result[i].value - ref_res(i)) > 0.1f) {
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

        Matrix vec;
        vec.numRows = matrix.numCols;
        vec.numCols = 1;
        vec.numVals = matrix.numCols;
        vec.elements = std::vector<Element>(vec.numVals);

        for (int i = 0; i < vec.numRows; i++) {
            vec[i] = Element{i, 0, (double)rand() / RAND_MAX * 4.8f + 0.1f};
        }

        Matrix res;
        res.numRows = matrix.numCols;
        res.numCols = 1;
        res.numRows = matrix.numCols;
        res.elements = std::vector<Element>(res.numRows);

        auto startTimePoint = std::chrono::high_resolution_clock::now();

        parallel_spmv(matrix, vec, res);

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

        bool correct = verify_result(matrix, vec, res);
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
