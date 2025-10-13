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
    network.base_mva = 100.0;

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
    network.base_mva = 100.0;

    BusData bus1 = {1, 1.05, 0.0, 0.0, 0.0, 2};
    BusData bus2 = {2, 1.0, 0.0, 100.0, 50.0, 0};
    BusData bus3 = {3, 1.0, 0.0, 80.0, 40.0, 0};
    network.buses = {bus1, bus2, bus3};

    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;

    solver::PowerFlowConfig config;
    config.max_iterations = 5;

    auto result = pf_solver->solve_power_flow(network, matrix, config);
    ASSERT_EQ(3, result.bus_voltages.size());
}
