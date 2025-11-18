#include <iostream>

#include "gap/core/backend_factory.h"

using namespace gap;
using namespace gap::core;
using namespace gap::solver;

// Forward declarations of test functions
void test_gpu_admittance_functionality();
void test_gpu_vs_cpu_admittance_simple();
void test_gpu_vs_cpu_admittance_pgm_network();
void test_gpu_admittance_update_small_batch();
void test_gpu_admittance_update_large_batch();
void test_gpu_lu_solver_functionality();
void test_gpu_powerflow_functionality();
void test_gpu_lu_solver_large_matrix();
void test_gpu_lu_solver_memory_management();
void test_gpu_vs_cpu_lu_solver_correctness();
void test_gpu_powerflow_convergence();
void test_gpu_powerflow_different_configs();
void test_gpu_vs_cpu_powerflow();

int main() {
    std::cout << "Running GPU tests..." << std::endl;

    try {
        // GPU Admittance tests
        test_gpu_admittance_functionality();
        test_gpu_vs_cpu_admittance_simple();
        test_gpu_vs_cpu_admittance_pgm_network();
        test_gpu_admittance_update_small_batch();
        test_gpu_admittance_update_large_batch();

        test_gpu_lu_solver_functionality();
        test_gpu_powerflow_functionality();

        // GPU LU Solver tests
        test_gpu_lu_solver_large_matrix();
        test_gpu_lu_solver_memory_management();
        test_gpu_vs_cpu_lu_solver_correctness();

        // GPU Power Flow tests
        test_gpu_powerflow_convergence();
        test_gpu_powerflow_different_configs();
        test_gpu_vs_cpu_powerflow();

        std::cout << "\nAll GPU tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}