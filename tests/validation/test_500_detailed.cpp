#include <chrono>
#include <cstdlib>
#include <iostream>

#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/logging/logger.h"
#include "gap/solver/lu_solver_interface.h"

using namespace gap;
using namespace gap::core;

// Generate random sparse matrix for testing
SparseMatrix generateRandomSparseMatrix(int size, double sparsity_ratio = 0.1) {
    SparseMatrix matrix;
    matrix.num_rows = size;
    matrix.num_cols = size;

    // Pre-calculate approximate nnz
    int estimated_nnz = static_cast<int>(size * size * sparsity_ratio);
    matrix.values.reserve(estimated_nnz);
    matrix.col_idx.reserve(estimated_nnz);
    matrix.row_ptr.resize(size + 1, 0);

    int nnz = 0;
    for (int row = 0; row < size; ++row) {
        matrix.row_ptr[row] = nnz;
        for (int col = 0; col < size; ++col) {
            double rand_val = static_cast<double>(rand()) / RAND_MAX;
            if (rand_val < sparsity_ratio || row == col) {
                double real_part = static_cast<double>(rand()) / RAND_MAX * 10.0 - 5.0;
                double imag_part = static_cast<double>(rand()) / RAND_MAX * 10.0 - 5.0;
                if (row == col) {
                    real_part += 10.0;  // Make diagonal dominant
                }
                matrix.values.emplace_back(real_part, imag_part);
                matrix.col_idx.push_back(col);
                ++nnz;
            }
        }
    }
    matrix.row_ptr[size] = nnz;
    matrix.nnz = nnz;

    return matrix;
}

int main() {
    // Configure logger to DEBUG level
    auto& logger = logging::global_logger;
    logger.configure(logging::LogLevel::DEBUG, logging::LogOutput::CONSOLE);
    logger.setComponent("TestMain");

    // Verify debug logging is enabled
    LOG_DEBUG(logger, "=== DEBUG logging enabled successfully ===");

    std::cout << "\n=== Detailed Timing Analysis for 500x500 Matrix ===" << std::endl;
    std::cout << "Generating 500x500 sparse matrix...\n" << std::endl;

    srand(42);  // Fixed seed for reproducibility
    auto matrix = generateRandomSparseMatrix(500, 0.05);

    std::cout << "Matrix: " << matrix.num_rows << "x" << matrix.num_cols << std::endl;
    std::cout << "Non-zeros: " << matrix.nnz << std::endl;
    std::cout << "Sparsity: " << (100.0 * matrix.nnz / (matrix.num_rows * matrix.num_cols)) << "%\n"
              << std::endl;

    // Generate RHS
    ComplexVector rhs(matrix.num_rows);
    for (int i = 0; i < matrix.num_rows; ++i) {
        rhs[i] = Complex(1.0, 0.0);
    }

    std::cout << "\n--- CPU LU Solver ---" << std::endl;
    auto cpu_solver = BackendFactory::create_lu_solver(BackendType::CPU);

    auto cpu_factor_start = std::chrono::high_resolution_clock::now();
    if (!cpu_solver->factorize(matrix)) {
        std::cerr << "CPU factorization failed!" << std::endl;
        return 1;
    }
    auto cpu_factor_end = std::chrono::high_resolution_clock::now();
    double cpu_factor_ms =
        std::chrono::duration<double, std::milli>(cpu_factor_end - cpu_factor_start).count();

    auto cpu_solve_start = std::chrono::high_resolution_clock::now();
    auto cpu_solution = cpu_solver->solve(rhs);
    auto cpu_solve_end = std::chrono::high_resolution_clock::now();
    double cpu_solve_us =
        std::chrono::duration<double, std::micro>(cpu_solve_end - cpu_solve_start).count();

    std::cout << "\nCPU Total Factor Time: " << cpu_factor_ms << " ms" << std::endl;
    std::cout << "CPU Solve Time: " << cpu_solve_us << " μs\n" << std::endl;

    std::cout << "\n--- GPU QR Solver (with detailed timing) ---" << std::endl;
    auto gpu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    std::cout << "\n** Factorization Phase **" << std::endl;
    auto gpu_factor_start = std::chrono::high_resolution_clock::now();
    if (!gpu_solver->factorize(matrix)) {
        std::cerr << "GPU factorization failed!" << std::endl;
        return 1;
    }
    auto gpu_factor_end = std::chrono::high_resolution_clock::now();
    double gpu_factor_ms =
        std::chrono::duration<double, std::milli>(gpu_factor_end - gpu_factor_start).count();

    std::cout << "\n** Solve Phase **" << std::endl;
    auto gpu_solve_start = std::chrono::high_resolution_clock::now();
    auto gpu_solution = gpu_solver->solve(rhs);
    auto gpu_solve_end = std::chrono::high_resolution_clock::now();
    double gpu_solve_us =
        std::chrono::duration<double, std::micro>(gpu_solve_end - gpu_solve_start).count();

    std::cout << "\nGPU Total Factor Time: " << gpu_factor_ms << " ms" << std::endl;
    std::cout << "GPU Solve Time: " << gpu_solve_us << " μs" << std::endl;

    // Verify correctness
    double max_error = 0.0;
    for (size_t i = 0; i < cpu_solution.size(); ++i) {
        double error = std::abs(cpu_solution[i] - gpu_solution[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    std::cout << "Correctness: " << (max_error < 1e-8 ? "PASS ✓" : "FAIL ✗") << std::endl;

    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "CPU Factor: " << cpu_factor_ms << " ms" << std::endl;
    std::cout << "GPU Factor: " << gpu_factor_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_factor_ms / gpu_factor_ms) << "x" << std::endl;
    std::cout << "\nCPU Solve: " << cpu_solve_us << " μs" << std::endl;
    std::cout << "GPU Solve: " << gpu_solve_us << " μs" << std::endl;
    std::cout << "Speedup: " << (cpu_solve_us / gpu_solve_us) << "x" << std::endl;

    return 0;
}
