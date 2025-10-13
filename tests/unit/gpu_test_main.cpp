#include "gap/core/backend_factory.h"

#include "../unit/test_main.cpp"

using namespace gap;
using namespace gap::core;
using namespace gap::solver;

// Forward declarations of test functions
void test_gpu_admittance_functionality();
void test_gpu_lu_solver_functionality();
void test_gpu_powerflow_functionality();
void test_gpu_lu_solver_large_matrix();
void test_gpu_lu_solver_memory_management();
void test_gpu_powerflow_convergence();
void test_gpu_powerflow_different_configs();

int main() {
    std::cout << "Running GPU tests..." << std::endl;

    try {
        // GPU Admittance tests
        test_gpu_admittance_functionality();
        test_gpu_lu_solver_functionality();
        test_gpu_powerflow_functionality();

        // GPU LU Solver tests
        test_gpu_lu_solver_large_matrix();
        test_gpu_lu_solver_memory_management();

        // GPU Power Flow tests
        test_gpu_powerflow_convergence();
        test_gpu_powerflow_different_configs();

        std::cout << "All GPU tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}