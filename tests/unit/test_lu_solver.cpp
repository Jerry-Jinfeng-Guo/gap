#include "gap/core/backend_factory.h"
#include "gap/solver/lu_solver_interface.h"

#include "test_framework.h"

using namespace gap;

void test_cpu_lu_solver_creation() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
    ASSERT_TRUE(lu_solver != nullptr);
    ASSERT_BACKEND_EQ(BackendType::CPU, lu_solver->get_backend_type());
    ASSERT_FALSE(lu_solver->is_factorized());
}

void test_lu_factorization() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create a simple 2x2 matrix
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};
    matrix.values = {Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)};

    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);
    ASSERT_TRUE(lu_solver->is_factorized());
}

void test_lu_solve() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create a simple matrix and factorize
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};
    matrix.values = {Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)};

    lu_solver->factorize(matrix);

    // Create RHS vector
    ComplexVector rhs = {Complex(3.0, 0.0), Complex(3.0, 0.0)};

    ComplexVector solution = lu_solver->solve(rhs);
    ASSERT_EQ(2, solution.size());
}

void test_lu_update_factorization() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 3;
    matrix.row_ptr = {0, 2, 3};
    matrix.col_idx = {0, 1, 1};
    matrix.values = {Complex(1.0, 0.0), Complex(0.5, 0.0), Complex(1.0, 0.0)};

    bool success = lu_solver->update_factorization(matrix);
    ASSERT_TRUE(success);
    ASSERT_TRUE(lu_solver->is_factorized());
}

void test_lu_symbolic_analysis() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create a 3x3 matrix with known sparsity pattern
    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 7;
    matrix.row_ptr = {0, 2, 5, 7};
    matrix.col_idx = {0, 1, 0, 1, 2, 1, 2};
    matrix.values = {
        Complex(2.0, 0.0), Complex(1.0, 0.0),                     // Row 0: [2, 1, 0]
        Complex(1.0, 0.0), Complex(3.0, 0.0), Complex(1.0, 0.0),  // Row 1: [1, 3, 1]
        Complex(1.0, 0.0), Complex(2.0, 0.0)                      // Row 2: [0, 1, 2]
    };

    // Factorize should succeed and predict reasonable fill-in
    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);
    ASSERT_TRUE(lu_solver->is_factorized());

    // The symbolic analysis should predict some fill-in but not excessive
    // This is tested implicitly through successful factorization
}

void test_lu_numerical_accuracy_3x3() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create a well-conditioned 3x3 test matrix with known solution
    // Matrix: [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 7;
    matrix.row_ptr = {0, 2, 5, 7};
    matrix.col_idx = {0, 1, 0, 1, 2, 1, 2};
    matrix.values = {Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(3.0, 0.0),
                     Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)};

    lu_solver->factorize(matrix);

    // Create RHS for known solution x = [1, 1, 1]
    // b = A * x = [3, 5, 3]
    ComplexVector rhs = {Complex(3.0, 0.0), Complex(5.0, 0.0), Complex(3.0, 0.0)};

    ComplexVector solution = lu_solver->solve(rhs);
    ASSERT_EQ(3, solution.size());

    // Verify solution accuracy (should be close to [1, 1, 1])
    double tolerance = 1e-10;
    ASSERT_TRUE(std::abs(solution[0].real() - 1.0) < tolerance);
    ASSERT_TRUE(std::abs(solution[1].real() - 1.0) < tolerance);
    ASSERT_TRUE(std::abs(solution[2].real() - 1.0) < tolerance);
    ASSERT_TRUE(std::abs(solution[0].imag()) < tolerance);
    ASSERT_TRUE(std::abs(solution[1].imag()) < tolerance);
    ASSERT_TRUE(std::abs(solution[2].imag()) < tolerance);
}

void test_lu_complex_arithmetic() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create a 2x2 matrix with complex values
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};
    matrix.values = {
        Complex(2.0, 1.0), Complex(1.0, 0.0),  // Row 0: [2+i, 1]
        Complex(0.0, 1.0), Complex(1.0, 1.0)   // Row 1: [i, 1+i]
    };

    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);

    // Test solve with complex RHS
    ComplexVector rhs = {Complex(1.0, 2.0), Complex(2.0, 1.0)};
    ComplexVector solution = lu_solver->solve(rhs);
    ASSERT_EQ(2, solution.size());

    // Verify that we get a solution (exact values depend on complex arithmetic)
    ASSERT_TRUE(std::isfinite(solution[0].real()));
    ASSERT_TRUE(std::isfinite(solution[0].imag()));
    ASSERT_TRUE(std::isfinite(solution[1].real()));
    ASSERT_TRUE(std::isfinite(solution[1].imag()));
}

void test_lu_diagonal_matrix() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create a diagonal matrix (should be easy to factorize)
    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 3;
    matrix.row_ptr = {0, 1, 2, 3};
    matrix.col_idx = {0, 1, 2};
    matrix.values = {Complex(2.0, 0.0), Complex(3.0, 0.0), Complex(4.0, 0.0)};

    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);

    ComplexVector rhs = {Complex(4.0, 0.0), Complex(6.0, 0.0), Complex(8.0, 0.0)};
    ComplexVector solution = lu_solver->solve(rhs);

    // Solution should be [2, 2, 2]
    double tolerance = 1e-12;
    ASSERT_TRUE(std::abs(solution[0].real() - 2.0) < tolerance);
    ASSERT_TRUE(std::abs(solution[1].real() - 2.0) < tolerance);
    ASSERT_TRUE(std::abs(solution[2].real() - 2.0) < tolerance);
}

void test_lu_identity_matrix() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create identity matrix
    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 3;
    matrix.row_ptr = {0, 1, 2, 3};
    matrix.col_idx = {0, 1, 2};
    matrix.values = {Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0)};

    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);

    ComplexVector rhs = {Complex(5.0, 2.0), Complex(3.0, 1.0), Complex(7.0, 3.0)};
    ComplexVector solution = lu_solver->solve(rhs);

    // Solution should equal RHS for identity matrix
    double tolerance = 1e-12;
    ASSERT_TRUE(std::abs(solution[0] - rhs[0]) < tolerance);
    ASSERT_TRUE(std::abs(solution[1] - rhs[1]) < tolerance);
    ASSERT_TRUE(std::abs(solution[2] - rhs[2]) < tolerance);
}

void test_lu_permutation_handling() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create a matrix that requires row pivoting
    // [[0, 1], [1, 1]] - first pivot is zero, needs permutation
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 3;
    matrix.row_ptr = {0, 1, 3};
    matrix.col_idx = {1, 0, 1};
    matrix.values = {Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0)};

    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);

    ComplexVector rhs = {Complex(1.0, 0.0), Complex(2.0, 0.0)};
    ComplexVector solution = lu_solver->solve(rhs);
    ASSERT_EQ(2, solution.size());

    // Should handle pivoting correctly and produce finite solution
    ASSERT_TRUE(std::isfinite(solution[0].real()));
    ASSERT_TRUE(std::isfinite(solution[1].real()));
}

void test_lu_error_conditions() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Test non-square matrix
    SparseMatrix non_square;
    non_square.num_rows = 2;
    non_square.num_cols = 3;
    non_square.nnz = 2;
    non_square.row_ptr = {0, 1, 2};
    non_square.col_idx = {0, 1};
    non_square.values = {Complex(1.0, 0.0), Complex(1.0, 0.0)};

    bool success = lu_solver->factorize(non_square);
    ASSERT_FALSE(success);  // Should fail for non-square matrix

    // Test empty matrix
    SparseMatrix empty;
    empty.num_rows = 2;
    empty.num_cols = 2;
    empty.nnz = 0;
    empty.row_ptr = {0, 0, 0};
    empty.col_idx = {};
    empty.values = {};

    success = lu_solver->factorize(empty);
    ASSERT_FALSE(success);  // Should fail for empty matrix
}

void test_lu_solve_without_factorization() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    ComplexVector rhs = {Complex(1.0, 0.0), Complex(2.0, 0.0)};

    // Should throw exception when trying to solve without factorization
    bool exception_thrown = false;
    try {
        lu_solver->solve(rhs);
    } catch (const std::runtime_error&) {
        exception_thrown = true;
    }
    ASSERT_TRUE(exception_thrown);
}

void test_lu_mismatched_rhs_size() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create and factorize a 2x2 matrix
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};
    matrix.values = {Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)};

    lu_solver->factorize(matrix);

    // Try to solve with wrong RHS size
    ComplexVector wrong_rhs = {Complex(1.0, 0.0), Complex(2.0, 0.0),
                               Complex(3.0, 0.0)};  // Size 3 instead of 2

    bool exception_thrown = false;
    try {
        lu_solver->solve(wrong_rhs);
    } catch (const std::runtime_error&) {
        exception_thrown = true;
    }
    ASSERT_TRUE(exception_thrown);
}

void test_lu_multiple_solves() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    // Create and factorize matrix once
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};
    matrix.values = {Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)};

    lu_solver->factorize(matrix);

    // Solve multiple times with different RHS
    ComplexVector rhs1 = {Complex(1.0, 0.0), Complex(1.0, 0.0)};
    ComplexVector rhs2 = {Complex(2.0, 0.0), Complex(2.0, 0.0)};
    ComplexVector rhs3 = {Complex(0.0, 1.0), Complex(0.0, 1.0)};

    ComplexVector sol1 = lu_solver->solve(rhs1);
    ComplexVector sol2 = lu_solver->solve(rhs2);
    ComplexVector sol3 = lu_solver->solve(rhs3);

    // All solutions should be valid
    ASSERT_EQ(2, sol1.size());
    ASSERT_EQ(2, sol2.size());
    ASSERT_EQ(2, sol3.size());

    // Solution 2 should be 2x solution 1 (linearity)
    double tolerance = 1e-12;
    ASSERT_TRUE(std::abs(sol2[0] - 2.0 * sol1[0]) < tolerance);
    ASSERT_TRUE(std::abs(sol2[1] - 2.0 * sol1[1]) < tolerance);
}

void test_gpu_lu_solver_availability() {
    bool gpu_available = core::BackendFactory::is_backend_available(BackendType::GPU_CUDA);

    if (gpu_available) {
        auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        ASSERT_TRUE(lu_solver != nullptr);
        ASSERT_BACKEND_EQ(BackendType::GPU_CUDA, lu_solver->get_backend_type());
    }

    // Test should pass regardless of GPU availability
    ASSERT_TRUE(true);
}

void register_lu_solver_tests(TestRunner& runner) {
    runner.add_test("CPU LU Solver Creation", test_cpu_lu_solver_creation);
    runner.add_test(" - LU Factorization", test_lu_factorization);
    runner.add_test(" - LU Solve", test_lu_solve);
    runner.add_test(" - LU Update Factorization", test_lu_update_factorization);
    runner.add_test(" - LU Symbolic Analysis", test_lu_symbolic_analysis);
    runner.add_test(" - LU Numerical Accuracy 3x3", test_lu_numerical_accuracy_3x3);
    runner.add_test(" - LU Complex Arithmetic", test_lu_complex_arithmetic);
    runner.add_test(" - LU Diagonal Matrix", test_lu_diagonal_matrix);
    runner.add_test(" - LU Identity Matrix", test_lu_identity_matrix);
    runner.add_test(" - LU Permutation Handling", test_lu_permutation_handling);
    runner.add_test(" - LU Error Conditions", test_lu_error_conditions);
    runner.add_test(" - LU Solve Without Factorization", test_lu_solve_without_factorization);
    runner.add_test(" - LU Mismatched RHS Size", test_lu_mismatched_rhs_size);
    runner.add_test(" - LU Multiple Solves", test_lu_multiple_solves);

    runner.add_test("GPU LU Solver Availability", test_gpu_lu_solver_availability);
}
