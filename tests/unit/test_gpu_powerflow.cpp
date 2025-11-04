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

    auto pf_solver = BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    pf_solver->set_lu_solver(std::shared_ptr<ILUSolver>(lu_solver.release()));

    // Create a small test system
    NetworkData network;
    network.num_buses = 4;

    // Create buses: 1 slack, 2 PQ, 1 PV
    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 241500.0,
                    .u_pu = 1.05,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};  // Slack
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 120e6,
                    .reactive_power = 80e6};  // PQ
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,
                    .reactive_power = 60e6};  // PQ
    BusData bus4 = {.id = 4,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PV,
                    .energized = 1,
                    .u = 234600.0,
                    .u_pu = 1.02,
                    .u_angle = 0.0,
                    .active_power = -80e6,
                    .reactive_power = 0.0};  // PV (generator)

    network.buses = {bus1, bus2, bus3, bus4};

    // Create dummy admittance matrix (4x4)
    SparseMatrix matrix;
    matrix.num_rows = 4;
    matrix.num_cols = 4;
    matrix.nnz = 10;  // Assuming some sparsity

    matrix.row_ptr = {0, 3, 5, 8, 10};
    matrix.col_idx = {0, 1, 3, 0, 1, 1, 2, 3, 2, 3};
    matrix.values = {Complex(5.0, -15.0), Complex(-2.0, 6.0),  Complex(-1.0, 3.0),
                     Complex(-2.0, 6.0),  Complex(8.0, -20.0), Complex(-3.0, 8.0),
                     Complex(6.0, -18.0), Complex(-1.5, 4.0),  Complex(-1.5, 4.0),
                     Complex(4.0, -12.0)};

    solver::PowerFlowConfig config;
    config.max_iterations = 20;
    config.tolerance = 1e-4;
    config.use_flat_start = true;
    config.verbose = false;

    auto result = pf_solver->solve_power_flow(network, matrix, config);

    ASSERT_EQ(4, result.bus_voltages.size());
    ASSERT_TRUE(result.iterations <= config.max_iterations);

    // Check that voltage magnitudes are reasonable
    for (auto const& voltage : result.bus_voltages) {
        double magnitude = std::abs(voltage);
        ASSERT_TRUE(magnitude > 0.5);  // Not too low
        ASSERT_TRUE(magnitude < 2.0);  // Not too high
    }
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

    // Test with different tolerances
    std::vector<double> tolerances = {1e-3, 1e-4, 1e-5};

    for (double tol : tolerances) {
        solver::PowerFlowConfig config;
        config.tolerance = tol;
        config.max_iterations = 15;

        auto result = pf_solver->solve_power_flow(network, matrix, config);

        ASSERT_EQ(2, result.bus_voltages.size());
        // Stricter tolerance should require more iterations (if not converged immediately)
        // This is more of a behavioral test
    }
}
