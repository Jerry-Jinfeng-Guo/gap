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
    network.base_mva = 100.0;

    // Create buses: 1 slack, 2 PQ, 1 PV
    BusData bus1 = {1, 1.05, 0.0, 0.0, 0.0, BusType::SLACK};  // Slack
    BusData bus2 = {2, 1.0, 0.0, 120.0, 80.0, BusType::PQ};   // PQ
    BusData bus3 = {3, 1.0, 0.0, 100.0, 60.0, BusType::PQ};   // PQ
    BusData bus4 = {4, 1.02, 0.0, -80.0, 0.0, BusType::PV};   // PV (generator)

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
    for (const auto& voltage : result.bus_voltages) {
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
    network.base_mva = 100.0;

    BusData bus1 = {1, 1.05, 0.0, 0.0, 0.0, BusType::SLACK};
    BusData bus2 = {2, 1.0, 0.0, 50.0, 30.0, BusType::PQ};
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
