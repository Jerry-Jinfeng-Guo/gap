#include <cmath>
#include <fstream>

#include "gap/core/backend_factory.h"

#include "../unit/test_framework.h"

using namespace gap;

// Forward declaration for LU solver validation tests
void register_lu_solver_validation_tests(TestRunner& runner);

/**
 * @brief IEEE 14-bus test case validation
 */
void test_ieee14_bus_case() {
    std::cout << "Running IEEE 14-bus test case validation..." << std::endl;

    // Create IEEE 14-bus system data (simplified)
    NetworkData network;
    network.num_buses = 14;
    network.num_branches = 20;

    // Create buses (simplified data for testing framework)
    network.buses.push_back({.id = 1,
                             .u_rated = 132000.0,
                             .bus_type = BusType::SLACK,
                             .energized = 1,
                             .u = 139920.0,
                             .u_pu = 1.060,
                             .u_angle = 0.0,
                             .active_power = 0.0,
                             .reactive_power = 0.0});  // Slack bus
    network.buses.push_back({.id = 2,
                             .u_rated = 132000.0,
                             .bus_type = BusType::PV,
                             .energized = 1,
                             .u = 137940.0,
                             .u_pu = 1.045,
                             .u_angle = 0.0,
                             .active_power = 40e6,
                             .reactive_power = 42.4e6});  // PV bus (generator)
    network.buses.push_back({.id = 3,
                             .u_rated = 132000.0,
                             .bus_type = BusType::PV,
                             .energized = 1,
                             .u = 133320.0,
                             .u_pu = 1.010,
                             .u_angle = 0.0,
                             .active_power = 0.0,
                             .reactive_power = 23.4e6});  // PV bus (generator)
    network.buses.push_back({.id = 4,
                             .u_rated = 132000.0,
                             .bus_type = BusType::PQ,
                             .energized = 1,
                             .u = 132000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = 47.8e6,
                             .reactive_power = -3.9e6});  // PQ bus
    network.buses.push_back({.id = 5,
                             .u_rated = 132000.0,
                             .bus_type = BusType::PQ,
                             .energized = 1,
                             .u = 132000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = 7.6e6,
                             .reactive_power = 1.6e6});  // PQ bus

    // Add remaining buses (PQ type)
    for (int i = 6; i <= 14; ++i) {
        BusData bus = {.id = i,
                       .u_rated = 132000.0,
                       .bus_type = BusType::PQ,
                       .energized = 1,
                       .u = 132000.0,
                       .u_pu = 1.0,
                       .u_angle = 0.0,
                       .active_power = 0.0,
                       .reactive_power = 0.0};
        if (i == 6) {
            bus.active_power = 11.2e6;   // 11.2 MW
            bus.reactive_power = 7.5e6;  // 7.5 MVAr
        }
        if (i == 9) {
            bus.active_power = 29.5e6;    // 29.5 MW
            bus.reactive_power = 16.6e6;  // 16.6 MVAr
        }
        if (i == 10) {
            bus.active_power = 9.0e6;    // 9.0 MW
            bus.reactive_power = 5.8e6;  // 5.8 MVAr
        }
        if (i == 11) {
            bus.active_power = 3.5e6;    // 3.5 MW
            bus.reactive_power = 1.8e6;  // 1.8 MVAr
        }
        if (i == 12) {
            bus.active_power = 6.1e6;    // 6.1 MW
            bus.reactive_power = 1.6e6;  // 1.6 MVAr
        }
        if (i == 13) {
            bus.active_power = 13.5e6;   // 13.5 MW
            bus.reactive_power = 5.8e6;  // 5.8 MVAr
        }
        if (i == 14) {
            bus.active_power = 14.9e6;   // 14.9 MW
            bus.reactive_power = 5.0e6;  // 5.0 MVAr
        }
        network.buses.push_back(bus);
    }

    // Create a simplified admittance matrix for IEEE 14-bus
    SparseMatrix matrix;
    matrix.num_rows = 14;
    matrix.num_cols = 14;
    matrix.nnz = 60;  // Approximate for IEEE 14-bus

    // Initialize with dummy values (in real implementation, this would be calculated)
    matrix.row_ptr.resize(15);
    for (int i = 0; i <= 14; ++i) {
        matrix.row_ptr[i] = i * 4;  // Assuming avg 4 connections per bus
    }

    matrix.col_idx.resize(60);
    matrix.values.resize(60);
    for (int i = 0; i < 60; ++i) {
        matrix.col_idx[i] = i % 14;              // Dummy pattern
        matrix.values[i] = Complex(5.0, -15.0);  // Dummy admittance values
    }

    // Test CPU backend
    {
        auto pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
        auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
        pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(lu_solver.release()));

        solver::PowerFlowConfig config;
        config.tolerance = 1e-4;
        config.max_iterations = 30;
        config.verbose = false;

        auto result = pf_solver->solve_power_flow(network, matrix, config);

        ASSERT_EQ(14, result.bus_voltages.size());

        // Check voltage magnitude bounds (typical for power systems)
        for (size_t i = 0; i < result.bus_voltages.size(); ++i) {
            double vm = std::abs(result.bus_voltages[i]);
            ASSERT_TRUE(vm > 0.8);  // Lower bound
            ASSERT_TRUE(vm < 1.2);  // Upper bound

            // Slack bus should maintain its specified voltage
            if (i == 0) {
                ASSERT_NEAR(1.06, vm, 0.1);
            }
        }

        std::cout << "  CPU backend: " << (result.converged ? "CONVERGED" : "NOT CONVERGED")
                  << " in " << result.iterations << " iterations" << std::endl;
    }

    // Test GPU backend if available
    if (core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        auto pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(lu_solver.release()));

        solver::PowerFlowConfig config;
        config.tolerance = 1e-4;
        config.max_iterations = 30;
        config.verbose = false;

        auto result = pf_solver->solve_power_flow(network, matrix, config);

        ASSERT_EQ(14, result.bus_voltages.size());

        std::cout << "  GPU backend: " << (result.converged ? "CONVERGED" : "NOT CONVERGED")
                  << " in " << result.iterations << " iterations" << std::endl;
    }
}

/**
 * @brief Simple 3-bus validation test
 */
void test_simple_3bus_case() {
    std::cout << "Running simple 3-bus validation case..." << std::endl;

    NetworkData network;
    network.num_buses = 3;

    // Create 3-bus system: 1 generator, 2 loads
    network.buses.push_back({.id = 1,
                             .u_rated = 230000.0,
                             .bus_type = BusType::SLACK,
                             .energized = 1,
                             .u = 241500.0,
                             .u_pu = 1.05,
                             .u_angle = 0.0,
                             .active_power = 0.0,
                             .reactive_power = 0.0});  // Slack
    network.buses.push_back({.id = 2,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PQ,
                             .energized = 1,
                             .u = 230000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = 100e6,
                             .reactive_power = 50e6});  // Load
    network.buses.push_back({.id = 3,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PQ,
                             .energized = 1,
                             .u = 230000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = 80e6,
                             .reactive_power = 40e6});  // Load

    // Create simple Y-matrix (star configuration)
    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 6;
    matrix.row_ptr = {0, 2, 4, 6};
    matrix.col_idx = {0, 1, 0, 1, 0, 2};
    matrix.values = {
        Complex(15.0, -45.0), Complex(-5.0, 15.0),   // Row 0
        Complex(-5.0, 15.0),  Complex(20.0, -50.0),  // Row 1
        Complex(-10.0, 30.0), Complex(18.0, -42.0)   // Row 2
    };

    auto pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
    pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(lu_solver.release()));

    solver::PowerFlowConfig config;
    config.tolerance = 1e-6;
    config.max_iterations = 20;

    auto result = pf_solver->solve_power_flow(network, matrix, config);

    ASSERT_EQ(3, result.bus_voltages.size());

    // Slack bus voltage should remain close to specified
    double slack_vm = std::abs(result.bus_voltages[0]);
    ASSERT_NEAR(1.05, slack_vm, 0.05);

    // Load bus voltages should be lower due to voltage drop
    for (int i = 1; i < 3; ++i) {
        double vm = std::abs(result.bus_voltages[i]);
        ASSERT_TRUE(vm < slack_vm);  // Should be lower than slack
        ASSERT_TRUE(vm > 0.85);      // But not too low
    }

    std::cout << "  3-bus case: " << (result.converged ? "CONVERGED" : "NOT CONVERGED") << " in "
              << result.iterations << " iterations" << std::endl;
}

/**
 * @brief Backend comparison test
 */
void test_backend_comparison() {
    if (!core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping backend comparison" << std::endl;
        return;
    }

    std::cout << "Running CPU vs GPU backend comparison..." << std::endl;

    // Create identical test case for both backends
    NetworkData network;
    network.num_buses = 5;

    network.buses.push_back({.id = 1,
                             .u_rated = 230000.0,
                             .bus_type = BusType::SLACK,
                             .energized = 1,
                             .u = 243800.0,
                             .u_pu = 1.06,
                             .u_angle = 0.0,
                             .active_power = 0.0,
                             .reactive_power = 0.0});
    network.buses.push_back({.id = 2,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PQ,
                             .energized = 1,
                             .u = 230000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = 50e6,
                             .reactive_power = 30e6});
    network.buses.push_back({.id = 3,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PQ,
                             .energized = 1,
                             .u = 230000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = 60e6,
                             .reactive_power = 35e6});
    network.buses.push_back({.id = 4,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PV,
                             .energized = 1,
                             .u = 234600.0,
                             .u_pu = 1.02,
                             .u_angle = 0.0,
                             .active_power = -40e6,
                             .reactive_power = 0.0});  // Generator
    network.buses.push_back({.id = 5,
                             .u_rated = 230000.0,
                             .bus_type = BusType::PQ,
                             .energized = 1,
                             .u = 230000.0,
                             .u_pu = 1.0,
                             .u_angle = 0.0,
                             .active_power = 70e6,
                             .reactive_power = 40e6});

    SparseMatrix matrix;
    matrix.num_rows = 5;
    matrix.num_cols = 5;
    matrix.nnz = 13;
    matrix.row_ptr = {0, 3, 6, 9, 12, 13};
    matrix.col_idx = {0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 4};
    matrix.values = {Complex(8.0, -24.0), Complex(-3.0, 9.0),   Complex(-2.0, 6.0),
                     Complex(-3.0, 9.0),  Complex(12.0, -30.0), Complex(-4.0, 10.0),
                     Complex(-4.0, 10.0), Complex(15.0, -35.0), Complex(-5.0, 12.0),
                     Complex(-5.0, 12.0), Complex(18.0, -40.0), Complex(-6.0, 15.0),
                     Complex(8.0, -21.0)};

    solver::PowerFlowConfig config;
    config.tolerance = 1e-5;
    config.max_iterations = 25;

    // Test CPU
    auto cpu_pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto cpu_lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
    cpu_pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu_solver.release()));

    auto cpu_result = cpu_pf_solver->solve_power_flow(network, matrix, config);

    // Test GPU
    auto gpu_pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto gpu_lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    gpu_pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu_solver.release()));

    auto gpu_result = gpu_pf_solver->solve_power_flow(network, matrix, config);

    // Both should produce similar results
    ASSERT_EQ(cpu_result.bus_voltages.size(), gpu_result.bus_voltages.size());

    // Compare voltage magnitudes (allowing for some numerical difference)
    for (size_t i = 0; i < cpu_result.bus_voltages.size(); ++i) {
        double cpu_vm = std::abs(cpu_result.bus_voltages[i]);
        double gpu_vm = std::abs(gpu_result.bus_voltages[i]);

        // Allow 1% difference between backends
        double relative_error = std::abs(cpu_vm - gpu_vm) / cpu_vm;
        ASSERT_TRUE(relative_error < 0.01);
    }

    std::cout << "  CPU result: " << (cpu_result.converged ? "CONVERGED" : "NOT CONVERGED")
              << " in " << cpu_result.iterations << " iterations" << std::endl;
    std::cout << "  GPU result: " << (gpu_result.converged ? "CONVERGED" : "NOT CONVERGED")
              << " in " << gpu_result.iterations << " iterations" << std::endl;
}

void register_validation_tests(TestRunner& runner) {
    runner.add_test("IEEE 14-bus Test Case", test_ieee14_bus_case);
    runner.add_test("Simple 3-bus Case", test_simple_3bus_case);
    runner.add_test("Backend Comparison", test_backend_comparison);

    // Register LU solver validation tests
    register_lu_solver_validation_tests(runner);
}
