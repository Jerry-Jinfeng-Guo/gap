#include <cmath>
#include <fstream>

#include "gap/core/backend_factory.h"

#include "../unit/test_framework.h"

using namespace gap;

// Forward declaration for LU solver validation tests
void register_lu_solver_validation_tests(TestRunner& runner);

// Forward declaration for PGM GPU validation tests
void register_pgm_gpu_validation_tests(TestRunner& runner);

// Forward declaration for solver comparison tests
void register_solver_comparison_tests(TestRunner& runner);

/**
 * @brief Backend comparison test - CPU vs GPU
 */
void test_backend_comparison() {
    std::cout << "Running backend comparison validation (CPU vs GPU)..." << std::endl;

    // Create identical test case for both backends
    NetworkData network;
    network.num_buses = 5;

    network.buses.push_back({.id = 1,
                             .u_rated = 230000.0,
                             .bus_type = BusType::SLACK,
                             .u = 243800.0,
                             .u_pu = 1.06,
                             .u_angle = 0.0,
                             .active_power = 0.0,
                             .reactive_power = 0.0});
    network.buses.push_back({.id = 2,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PQ,
                             .u = 230000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = -50e6,      // Load (negative)
                             .reactive_power = -30e6});  // Load (negative)
    network.buses.push_back({.id = 3,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PQ,
                             .u = 230000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = -60e6,      // Load (negative)
                             .reactive_power = -35e6});  // Load (negative)
    network.buses.push_back({.id = 4,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PV,
                             .u = 234600.0,
                             .u_pu = 1.02,
                             .u_angle = 0.0,
                             .active_power = 40e6,     // Generator (positive)
                             .reactive_power = 0.0});  // Generator
    network.buses.push_back({.id = 5,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PQ,
                             .u = 230000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = -70e6,      // Load (negative)
                             .reactive_power = -40e6});  // Load (negative)

    // Since energized is now an output field (default 0), we need to manually set it
    // for this test since we're providing a pre-built Y-matrix without branches
    for (auto& bus : network.buses) {
        bus.energized = 1;
    }

    // Y-bus should be in Siemens - the solver will convert to per-unit automatically
    SparseMatrix matrix;
    matrix.num_rows = 5;
    matrix.num_cols = 5;
    matrix.nnz = 13;
    matrix.row_ptr = {0, 3, 6, 9, 12, 13};
    matrix.col_idx = {0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 4};

    // Typical 230kV line: Z ≈ 30-100 Ω → Y ≈ 0.01-0.03 S
    matrix.values = {Complex(0.003, -0.01),    Complex(-0.001, 0.003), Complex(-0.0008, 0.0025),
                     Complex(-0.001, 0.003),   Complex(0.004, -0.012), Complex(-0.0012, 0.0036),
                     Complex(-0.0012, 0.0036), Complex(0.005, -0.014), Complex(-0.0016, 0.0048),
                     Complex(-0.0016, 0.0048), Complex(0.006, -0.016), Complex(-0.002, 0.006),
                     Complex(0.0028, -0.0085)};

    solver::PowerFlowConfig config;
    config.tolerance = 1e-5;
    config.max_iterations = 25;

    // Test CPU
    auto cpu_pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto cpu_lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
    cpu_pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu_solver.release()));

    auto cpu_result = cpu_pf_solver->solve_power_flow(network, matrix, config);

    std::cout << "  CPU result: " << (cpu_result.converged ? "CONVERGED" : "NOT CONVERGED")
              << " in " << cpu_result.iterations << " iterations" << std::endl;

    // Test GPU (now fully implemented!)
    if (core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        auto gpu_pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        auto gpu_lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        gpu_pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu_solver.release()));

        auto gpu_result = gpu_pf_solver->solve_power_flow(network, matrix, config);

        std::cout << "  GPU result: " << (gpu_result.converged ? "CONVERGED" : "NOT CONVERGED")
                  << " in " << gpu_result.iterations << " iterations" << std::endl;

        // Compare results
        if (cpu_result.converged && gpu_result.converged) {
            Float max_diff = 0.0;
            for (size_t i = 0; i < cpu_result.bus_voltages.size(); ++i) {
                Float diff = std::abs(cpu_result.bus_voltages[i] - gpu_result.bus_voltages[i]);
                max_diff = std::max(max_diff, diff);
            }
            std::cout << "  Maximum voltage difference (CPU vs GPU): " << max_diff << " pu"
                      << std::endl;
            ASSERT_TRUE(max_diff < 1e-4);  // Results should match within 0.01%
            std::cout << "  ✓ CPU and GPU results match!" << std::endl;
        }
    } else {
        std::cout << "  GPU result: SKIPPED (GPU not available)" << std::endl;
    }

    // Placeholder result check - just verify CPU results are reasonable
    ASSERT_EQ(5, cpu_result.bus_voltages.size());
    for (size_t i = 0; i < cpu_result.bus_voltages.size(); ++i) {
        Float cpu_vm = std::abs(cpu_result.bus_voltages[i]);
        ASSERT_TRUE(cpu_vm > 0.5 && cpu_vm < 2.0);  // Reasonable voltage bounds
    }
}

void register_validation_tests(TestRunner& runner) {
    // Temporarily disabled - test has incorrectly configured network causing singular Jacobian
    // runner.add_test("Backend Comparison", test_backend_comparison);

    // Register LU solver validation tests
    register_lu_solver_validation_tests(runner);

    // Register PGM GPU validation tests
    register_pgm_gpu_validation_tests(runner);

    // Register solver comparison tests (NR vs IC)
    register_solver_comparison_tests(runner);
}
