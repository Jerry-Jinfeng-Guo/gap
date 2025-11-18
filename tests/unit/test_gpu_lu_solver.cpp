#include <iostream>

#include "gap/core/backend_factory.h"

#include "test_framework.h"

using namespace gap;
using namespace gap::core;
using namespace gap::solver;

void test_gpu_lu_solver_large_matrix() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    // Create larger sparse matrix
    SparseMatrix matrix;
    matrix.num_rows = 10;
    matrix.num_cols = 10;
    matrix.nnz = 28;  // Tridiagonal matrix with some extra elements

    // Build tridiagonal matrix pattern
    matrix.row_ptr.resize(11);
    matrix.col_idx.clear();
    matrix.values.clear();

    int nnz_count = 0;
    matrix.row_ptr[0] = 0;

    for (int i = 0; i < 10; ++i) {
        // Add diagonal element
        matrix.col_idx.push_back(i);
        matrix.values.push_back(Complex(4.0, 0.0));
        nnz_count++;

        // Add off-diagonal elements
        if (i > 0) {
            matrix.col_idx.push_back(i - 1);
            matrix.values.push_back(Complex(-1.0, 0.0));
            nnz_count++;
        }
        if (i < 9) {
            matrix.col_idx.push_back(i + 1);
            matrix.values.push_back(Complex(-1.0, 0.0));
            nnz_count++;
        }

        matrix.row_ptr[i + 1] = nnz_count;
    }

    matrix.nnz = nnz_count;

    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);
    ASSERT_TRUE(lu_solver->is_factorized());

    // Test solve with this matrix
    ComplexVector rhs(10, Complex(1.0, 0.0));
    ComplexVector solution = lu_solver->solve(rhs);
    ASSERT_EQ(10, solution.size());
}

void test_gpu_lu_solver_memory_management() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    // Test multiple factorizations to check memory management
    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    for (int iter = 0; iter < 3; ++iter) {
        SparseMatrix matrix;
        matrix.num_rows = 5;
        matrix.num_cols = 5;
        matrix.nnz = 13;

        // Create a simple pentadiagonal pattern
        matrix.row_ptr = {0, 3, 6, 9, 12, 13};
        matrix.col_idx = {0, 1, 2, 0, 1, 2, 1, 2, 3, 2, 3, 4, 4};
        matrix.values = {Complex(3.0, 0.0),  Complex(-1.0, 0.0), Complex(-0.5, 0.0),
                         Complex(-1.0, 0.0), Complex(4.0, 0.0),  Complex(-1.5, 0.0),
                         Complex(-0.5, 0.0), Complex(3.5, 0.0),  Complex(-1.0, 0.0),
                         Complex(-1.0, 0.0), Complex(3.0, 0.0),  Complex(-0.8, 0.0),
                         Complex(2.0, 0.0)};

        bool success = lu_solver->factorize(matrix);
        ASSERT_TRUE(success);

        ComplexVector rhs(5, Complex(1.0, static_cast<Float>(iter)));
        ComplexVector solution = lu_solver->solve(rhs);
        ASSERT_EQ(5, solution.size());
    }
}

void test_gpu_vs_cpu_lu_solver_correctness() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "=== Testing GPU vs CPU LU Solver Correctness ===" << std::endl;

    // Create both CPU and GPU solvers
    auto cpu_solver = BackendFactory::create_lu_solver(BackendType::CPU);
    auto gpu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    // Test 1: Small tridiagonal matrix (5x5)
    {
        std::cout << "Test 1: 5x5 tridiagonal matrix" << std::endl;
        SparseMatrix matrix;
        matrix.num_rows = 5;
        matrix.num_cols = 5;
        matrix.row_ptr = {0, 2, 4, 6, 8, 9};
        matrix.col_idx = {0, 1, 1, 2, 2, 3, 3, 4, 4};
        matrix.values = {Complex(2.0, 0.0),  Complex(-1.0, 0.0), Complex(2.0, 0.0),
                         Complex(-1.0, 0.0), Complex(2.0, 0.0),  Complex(-1.0, 0.0),
                         Complex(2.0, 0.0),  Complex(-1.0, 0.0), Complex(2.0, 0.0)};
        matrix.nnz = 9;

        cpu_solver->factorize(matrix);
        gpu_solver->factorize(matrix);

        ComplexVector rhs = {Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0),
                             Complex(2.0, 0.0), Complex(1.0, 0.0)};

        ComplexVector cpu_solution = cpu_solver->solve(rhs);
        ComplexVector gpu_solution = gpu_solver->solve(rhs);

        ASSERT_EQ(cpu_solution.size(), gpu_solution.size());

        Float max_diff = 0.0;
        for (size_t i = 0; i < cpu_solution.size(); ++i) {
            Float diff = std::abs(cpu_solution[i] - gpu_solution[i]);
            max_diff = std::max(max_diff, diff);
        }

        std::cout << "  Max difference: " << max_diff << std::endl;
        ASSERT_TRUE(max_diff < 1e-10);
    }

    // Test 2: Larger matrix with complex values (10x10)
    {
        std::cout << "Test 2: 10x10 matrix with complex values" << std::endl;
        SparseMatrix matrix;
        matrix.num_rows = 10;
        matrix.num_cols = 10;
        matrix.row_ptr.resize(11);
        matrix.col_idx.clear();
        matrix.values.clear();

        int nnz_count = 0;
        matrix.row_ptr[0] = 0;

        for (int i = 0; i < 10; ++i) {
            // Diagonal
            matrix.col_idx.push_back(i);
            matrix.values.push_back(Complex(5.0, 0.5));
            nnz_count++;

            // Off-diagonal
            if (i > 0) {
                matrix.col_idx.push_back(i - 1);
                matrix.values.push_back(Complex(-1.0, -0.1));
                nnz_count++;
            }
            if (i < 9) {
                matrix.col_idx.push_back(i + 1);
                matrix.values.push_back(Complex(-1.0, 0.1));
                nnz_count++;
            }

            matrix.row_ptr[i + 1] = nnz_count;
        }
        matrix.nnz = nnz_count;

        cpu_solver->factorize(matrix);
        gpu_solver->factorize(matrix);

        ComplexVector rhs(10, Complex(1.0, 0.5));

        ComplexVector cpu_solution = cpu_solver->solve(rhs);
        ComplexVector gpu_solution = gpu_solver->solve(rhs);

        ASSERT_EQ(cpu_solution.size(), gpu_solution.size());

        Float max_diff = 0.0;
        for (size_t i = 0; i < cpu_solution.size(); ++i) {
            Float diff = std::abs(cpu_solution[i] - gpu_solution[i]);
            max_diff = std::max(max_diff, diff);
        }

        std::cout << "  Max difference: " << max_diff << std::endl;
        ASSERT_TRUE(max_diff < 1e-10);
    }

    // Test 3: Dense-like pattern (more fill)
    {
        std::cout << "Test 3: 7x7 matrix with denser pattern" << std::endl;
        SparseMatrix matrix;
        matrix.num_rows = 7;
        matrix.num_cols = 7;
        matrix.row_ptr = {0, 3, 7, 11, 15, 19, 23, 26};
        matrix.col_idx = {0, 1, 2,     // row 0
                          0, 1, 2, 3,  // row 1
                          0, 1, 2, 3,  // row 2
                          1, 2, 3, 4,  // row 3
                          2, 3, 4, 5,  // row 4
                          3, 4, 5, 6,  // row 5
                          4, 5, 6};    // row 6
        matrix.values = {
            Complex(4.0, 0.0),   Complex(-1.0, 0.2), Complex(-0.5, 0.0), Complex(-1.0, -0.2),
            Complex(5.0, 0.0),   Complex(-1.0, 0.3), Complex(-0.5, 0.0), Complex(-0.5, 0.0),
            Complex(-1.0, -0.3), Complex(5.0, 0.0),  Complex(-1.0, 0.1), Complex(-0.5, 0.0),
            Complex(-1.0, -0.1), Complex(5.0, 0.0),  Complex(-1.0, 0.2), Complex(-0.5, 0.0),
            Complex(-1.0, -0.2), Complex(5.0, 0.0),  Complex(-1.0, 0.0), Complex(-0.5, 0.0),
            Complex(-1.0, 0.0),  Complex(5.0, 0.0),  Complex(-1.0, 0.0), Complex(-0.5, 0.0),
            Complex(-1.0, 0.0),  Complex(4.0, 0.0)};
        matrix.nnz = 26;

        cpu_solver->factorize(matrix);
        gpu_solver->factorize(matrix);

        ComplexVector rhs = {Complex(1.0, 0.1), Complex(2.0, 0.2), Complex(3.0, 0.3),
                             Complex(4.0, 0.4), Complex(3.0, 0.3), Complex(2.0, 0.2),
                             Complex(1.0, 0.1)};

        ComplexVector cpu_solution = cpu_solver->solve(rhs);
        ComplexVector gpu_solution = gpu_solver->solve(rhs);

        ASSERT_EQ(cpu_solution.size(), gpu_solution.size());

        Float max_diff = 0.0;
        for (size_t i = 0; i < cpu_solution.size(); ++i) {
            Float diff = std::abs(cpu_solution[i] - gpu_solution[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > 1e-10) {
                std::cout << "    Element " << i << ": CPU=" << cpu_solution[i]
                          << " GPU=" << gpu_solution[i] << " diff=" << diff << std::endl;
            }
        }

        std::cout << "  Max difference: " << max_diff << std::endl;
        ASSERT_TRUE(max_diff < 1e-9);  // Slightly relaxed tolerance for denser matrix
    }

    std::cout << "✓ All GPU vs CPU correctness tests passed!" << std::endl;
}
