#include "../unit/test_main.cpp"
#include "gap/solver/lu_solver_interface.h"
#include "gap/core/backend_factory.h"

using namespace gap;

void test_cpu_lu_solver_creation() {
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
    ASSERT_TRUE(lu_solver != nullptr);
    ASSERT_EQ(BackendType::CPU, lu_solver->get_backend_type());
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
    matrix.values = {
        Complex(2.0, 0.0), Complex(1.0, 0.0),
        Complex(1.0, 0.0), Complex(2.0, 0.0)
    };
    
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
    matrix.values = {
        Complex(2.0, 0.0), Complex(1.0, 0.0),
        Complex(1.0, 0.0), Complex(2.0, 0.0)
    };
    
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
    matrix.values = {
        Complex(1.0, 0.0), Complex(0.5, 0.0),
        Complex(1.0, 0.0)
    };
    
    bool success = lu_solver->update_factorization(matrix);
    ASSERT_TRUE(success);
    ASSERT_TRUE(lu_solver->is_factorized());
}

void test_gpu_lu_solver_availability() {
    bool gpu_available = core::BackendFactory::is_backend_available(BackendType::GPU_CUDA);
    
    if (gpu_available) {
        auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        ASSERT_TRUE(lu_solver != nullptr);
        ASSERT_EQ(BackendType::GPU_CUDA, lu_solver->get_backend_type());
    }
    
    // Test should pass regardless of GPU availability
    ASSERT_TRUE(true);
}

int main() {
    TestRunner runner;
    
    runner.add_test("CPU LU Solver Creation", test_cpu_lu_solver_creation);
    runner.add_test("LU Factorization", test_lu_factorization);
    runner.add_test("LU Solve", test_lu_solve);
    runner.add_test("LU Update Factorization", test_lu_update_factorization);
    runner.add_test("GPU LU Solver Availability", test_gpu_lu_solver_availability);
    
    runner.run_all();
    
    return runner.get_failed_count();
}