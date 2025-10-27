#include <iostream>

#include "gap/core/backend_factory.h"

#include "test_framework.h"

using namespace gap;
using namespace gap::core;
using namespace gap::solver;

void test_gpu_admittance_functionality() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    auto admittance = BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
    ASSERT_TRUE(admittance != nullptr);

    NetworkData network;
    network.num_buses = 5;
    network.num_branches = 4;

    auto matrix = admittance->build_admittance_matrix(network);
    ASSERT_TRUE(matrix != nullptr);
    ASSERT_EQ(5, matrix->num_rows);
}

void test_gpu_lu_solver_functionality() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    ASSERT_TRUE(lu_solver != nullptr);

    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 9;
    matrix.row_ptr = {0, 3, 6, 9};
    matrix.col_idx = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    matrix.values = {Complex(4.0, 0.0),  Complex(-1.0, 0.0), Complex(-1.0, 0.0),
                     Complex(-1.0, 0.0), Complex(4.0, 0.0),  Complex(-1.0, 0.0),
                     Complex(-1.0, 0.0), Complex(-1.0, 0.0), Complex(4.0, 0.0)};

    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);
}

void test_gpu_powerflow_functionality() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    auto pf_solver = BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    ASSERT_TRUE(pf_solver != nullptr);
    pf_solver->set_lu_solver(std::shared_ptr<ILUSolver>(lu_solver.release()));

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
                    .active_power = 100e6,
                    .reactive_power = 50e6};
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 80e6,
                    .reactive_power = 40e6};
    network.buses = {bus1, bus2, bus3};

    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;

    solver::PowerFlowConfig config;
    config.max_iterations = 5;

    auto result = pf_solver->solve_power_flow(network, matrix, config);
    ASSERT_EQ(3, result.bus_voltages.size());
}
