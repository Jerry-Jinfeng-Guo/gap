#include "../unit/test_main.cpp"
#include "gap/core/backend_factory.h"

using namespace gap;

void test_gpu_lu_solver_large_matrix() {
    if (!core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }
    
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    
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
            matrix.col_idx.push_back(i-1);
            matrix.values.push_back(Complex(-1.0, 0.0));
            nnz_count++;
        }
        if (i < 9) {
            matrix.col_idx.push_back(i+1);
            matrix.values.push_back(Complex(-1.0, 0.0));
            nnz_count++;
        }
        
        matrix.row_ptr[i+1] = nnz_count;
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
    if (!core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }
    
    // Test multiple factorizations to check memory management
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    
    for (int iter = 0; iter < 3; ++iter) {
        SparseMatrix matrix;
        matrix.num_rows = 5;
        matrix.num_cols = 5;
        matrix.nnz = 13;
        
        // Create a simple pentadiagonal pattern
        matrix.row_ptr = {0, 3, 6, 9, 12, 13};
        matrix.col_idx = {0, 1, 2, 0, 1, 2, 1, 2, 3, 2, 3, 4, 4};
        matrix.values = {
            Complex(3.0, 0.0), Complex(-1.0, 0.0), Complex(-0.5, 0.0),
            Complex(-1.0, 0.0), Complex(4.0, 0.0), Complex(-1.5, 0.0),
            Complex(-0.5, 0.0), Complex(3.5, 0.0), Complex(-1.0, 0.0),
            Complex(-1.0, 0.0), Complex(3.0, 0.0), Complex(-0.8, 0.0),
            Complex(2.0, 0.0)
        };
        
        bool success = lu_solver->factorize(matrix);
        ASSERT_TRUE(success);
        
        ComplexVector rhs(5, Complex(1.0, static_cast<double>(iter)));
        ComplexVector solution = lu_solver->solve(rhs);
        ASSERT_EQ(5, solution.size());
    }
}

int main() {
    TestRunner runner;
    
    runner.add_test("GPU LU Solver Large Matrix", test_gpu_lu_solver_large_matrix);
    runner.add_test("GPU LU Solver Memory Management", test_gpu_lu_solver_memory_management);
    
    runner.run_all();
    
    return runner.get_failed_count();
}