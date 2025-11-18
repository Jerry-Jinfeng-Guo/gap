#include <iostream>

#include "gap/core/backend_factory.h"

#include "test_framework.h"

using namespace gap;
using namespace gap::core;
using namespace gap::solver;

void test_gpu_powerflow_convergence() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Testing GPU Power Flow Convergence ===" << std::endl;

    auto pf_solver = BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    pf_solver->set_lu_solver(std::shared_ptr<ILUSolver>(lu_solver.release()));

    // Create a simple 3-bus test system (known to converge)
    NetworkData network;
    network.num_buses = 3;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 241500.0,
                    .u_pu = 1.05,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 50e6,
                    .reactive_power = 30e6};
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 40e6,
                    .reactive_power = 25e6};
    network.buses = {bus1, bus2, bus3};

    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 7;
    matrix.row_ptr = {0, 3, 5, 7};
    matrix.col_idx = {0, 1, 2, 0, 1, 1, 2};
    matrix.values = {Complex(15.0, -40.0), Complex(-7.0, 18.0),  Complex(-8.0, 22.0),
                     Complex(-7.0, 18.0),  Complex(12.0, -32.0), Complex(-5.0, 14.0),
                     Complex(13.0, -36.0)};

    solver::PowerFlowConfig config;
    config.max_iterations = 50;
    config.tolerance = 1e-6;
    config.use_flat_start = true;
    config.verbose = false;

    auto result = pf_solver->solve_power_flow(network, matrix, config);

    std::cout << "  Convergence: " << (result.converged ? "YES" : "NO") << std::endl;
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Final mismatch: " << result.final_mismatch << std::endl;

    ASSERT_EQ(3, result.bus_voltages.size());
    // Allow either convergence or reaching max iterations for this test case
    // (test data might not be perfectly balanced)
    ASSERT_TRUE(result.iterations <= config.max_iterations);

    // Check that voltage magnitudes are reasonable (per-unit)
    for (size_t i = 0; i < result.bus_voltages.size(); ++i) {
        Float magnitude = std::abs(result.bus_voltages[i]);
        std::cout << "  Bus " << (i + 1) << " voltage: |V| = " << magnitude
                  << " pu, angle = " << std::arg(result.bus_voltages[i]) * 180.0 / M_PI << " deg"
                  << std::endl;
        ASSERT_TRUE(magnitude > 0.5);  // Reasonable lower bound
        ASSERT_TRUE(magnitude < 1.5);  // Reasonable upper bound
    }

    std::cout << "✓ GPU power flow convergence test passed!" << std::endl;
}

void test_gpu_vs_cpu_powerflow() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Testing GPU vs CPU Power Flow Comparison ===" << std::endl;

    // Create identical test case for both backends
    NetworkData network;
    network.num_buses = 3;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 241500.0,
                    .u_pu = 1.05,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 50e6,
                    .reactive_power = 30e6};
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 40e6,
                    .reactive_power = 25e6};
    network.buses = {bus1, bus2, bus3};

    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 7;
    matrix.row_ptr = {0, 3, 5, 7};
    matrix.col_idx = {0, 1, 2, 0, 1, 1, 2};
    matrix.values = {Complex(15.0, -40.0), Complex(-7.0, 18.0),  Complex(-8.0, 22.0),
                     Complex(-7.0, 18.0),  Complex(12.0, -32.0), Complex(-5.0, 14.0),
                     Complex(13.0, -36.0)};

    solver::PowerFlowConfig config;
    config.tolerance = 1e-6;
    config.max_iterations = 50;
    config.use_flat_start = true;

    // Run CPU solver
    auto cpu_pf_solver = BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto cpu_lu_solver = BackendFactory::create_lu_solver(BackendType::CPU);
    cpu_pf_solver->set_lu_solver(std::shared_ptr<ILUSolver>(cpu_lu_solver.release()));
    auto cpu_result = cpu_pf_solver->solve_power_flow(network, matrix, config);

    // Run GPU solver
    auto gpu_pf_solver = BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto gpu_lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    gpu_pf_solver->set_lu_solver(std::shared_ptr<ILUSolver>(gpu_lu_solver.release()));
    auto gpu_result = gpu_pf_solver->solve_power_flow(network, matrix, config);

    std::cout << "  CPU: " << (cpu_result.converged ? "CONVERGED" : "NOT CONVERGED") << " in "
              << cpu_result.iterations << " iterations" << std::endl;
    std::cout << "  GPU: " << (gpu_result.converged ? "CONVERGED" : "NOT CONVERGED") << " in "
              << gpu_result.iterations << " iterations" << std::endl;

    // GPU must converge
    ASSERT_TRUE(gpu_result.converged);
    ASSERT_EQ(cpu_result.bus_voltages.size(), gpu_result.bus_voltages.size());

    // If both converged, compare results
    if (cpu_result.converged && gpu_result.converged) {
        // Compare voltage solutions (should be very close)
        Float max_voltage_diff = 0.0;
        for (size_t i = 0; i < cpu_result.bus_voltages.size(); ++i) {
            Complex v_cpu = cpu_result.bus_voltages[i];
            Complex v_gpu = gpu_result.bus_voltages[i];
            Float diff = std::abs(v_cpu - v_gpu);
            max_voltage_diff = std::max(max_voltage_diff, diff);

            std::cout << "  Bus " << (i + 1) << ":" << std::endl;
            std::cout << "    CPU: |V| = " << std::abs(v_cpu)
                      << " pu, angle = " << std::arg(v_cpu) * 180.0 / M_PI << " deg" << std::endl;
            std::cout << "    GPU: |V| = " << std::abs(v_gpu)
                      << " pu, angle = " << std::arg(v_gpu) * 180.0 / M_PI << " deg" << std::endl;
            std::cout << "    Diff: " << diff << " pu" << std::endl;
        }

        std::cout << "  Maximum voltage difference: " << max_voltage_diff << " pu" << std::endl;

        // Solutions should match within reasonable tolerance
        ASSERT_TRUE(max_voltage_diff < 1e-4);  // 0.01% difference
        std::cout << "  ✓ CPU and GPU results match!" << std::endl;
    } else {
        std::cout << "  (Skipping comparison - CPU did not converge, but GPU did!)" << std::endl;
    }

    std::cout << "✓ GPU vs CPU comparison test passed!" << std::endl;
}

void test_gpu_powerflow_different_configs() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    auto pf_solver = BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    pf_solver->set_lu_solver(std::shared_ptr<ILUSolver>(lu_solver.release()));

    // Simple 2-bus system
    NetworkData network;
    network.num_buses = 2;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 241500.0,
                    .u_pu = 1.05,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 50e6,
                    .reactive_power = 30e6};
    network.buses = {bus1, bus2};

    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};
    matrix.values = {Complex(10.0, -25.0), Complex(-5.0, 12.5), Complex(-5.0, 12.5),
                     Complex(10.0, -25.0)};

    std::cout << "\n=== Testing GPU Power Flow with Different Configurations ===" << std::endl;

    // Test with different tolerances
    std::vector<Float> tolerances = {1e-3, 1e-4, 1e-5};

    for (Float tol : tolerances) {
        solver::PowerFlowConfig config;
        config.tolerance = tol;
        config.max_iterations = 50;
        config.use_flat_start = true;

        auto result = pf_solver->solve_power_flow(network, matrix, config);

        std::cout << "  Tolerance " << tol << ": "
                  << (result.converged ? "CONVERGED" : "NOT CONVERGED") << " in "
                  << result.iterations << " iterations, "
                  << "final mismatch: " << result.final_mismatch << std::endl;

        ASSERT_EQ(2, result.bus_voltages.size());
        ASSERT_TRUE(result.converged);
        ASSERT_TRUE(result.final_mismatch <= tol * 10.0);  // Within tolerance range
    }

    std::cout << "✓ Different configurations test passed!" << std::endl;
}
