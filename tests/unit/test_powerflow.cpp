#include "gap/core/backend_factory.h"
#include "gap/solver/powerflow_interface.h"

#include "test_framework.h"

using namespace gap;

void test_cpu_powerflow_creation() {
    auto pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    ASSERT_TRUE(pf_solver != nullptr);
    ASSERT_BACKEND_EQ(BackendType::CPU, pf_solver->get_backend_type());
}

void test_powerflow_config() {
    solver::PowerFlowConfig config;
    config.tolerance = 1e-8;
    config.max_iterations = 100;
    config.use_flat_start = false;
    config.acceleration_factor = 1.6;
    config.verbose = true;

    ASSERT_NEAR(1e-8, config.tolerance, 1e-10);
    ASSERT_EQ(100, config.max_iterations);
    ASSERT_FALSE(config.use_flat_start);
    ASSERT_NEAR(1.6, config.acceleration_factor, 1e-6);
    ASSERT_TRUE(config.verbose);
}

void test_mismatch_calculation() {
    auto pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);

    // Create dummy network data
    NetworkData network;
    network.num_buses = 2;
    network.base_mva = 100.0;

    BusData bus1 = {1, 1.0, 0.0, 0.0, 0.0, 2};     // Slack bus
    BusData bus2 = {2, 1.0, 0.0, 100.0, 50.0, 0};  // PQ bus
    network.buses = {bus1, bus2};

    // Create dummy admittance matrix
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;

    // Create dummy voltage vector
    ComplexVector voltages = {Complex(1.0, 0.0), Complex(0.95, -0.1)};

    auto mismatches = pf_solver->calculate_mismatches(network, voltages, matrix);
    ASSERT_TRUE(mismatches.size() > 0);
}

void test_powerflow_solve_simple() {
    auto pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

    pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(lu_solver.release()));

    // Create simple test case
    NetworkData network;
    network.num_buses = 2;
    network.base_mva = 100.0;

    BusData bus1 = {1, 1.05, 0.0, 0.0, 0.0, 2};   // Slack bus
    BusData bus2 = {2, 1.0, 0.0, 50.0, 30.0, 0};  // PQ bus
    network.buses = {bus1, bus2};

    // Create dummy admittance matrix
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};
    matrix.values = {Complex(10.0, -20.0), Complex(-5.0, 10.0), Complex(-5.0, 10.0),
                     Complex(10.0, -20.0)};

    solver::PowerFlowConfig config;
    config.max_iterations = 10;
    config.tolerance = 1e-4;

    auto result = pf_solver->solve_power_flow(network, matrix, config);

    ASSERT_EQ(2, result.bus_voltages.size());
    ASSERT_TRUE(result.iterations <= config.max_iterations);
}

void test_gpu_powerflow_availability() {
    bool gpu_available = core::BackendFactory::is_backend_available(BackendType::GPU_CUDA);

    if (gpu_available) {
        auto pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        ASSERT_TRUE(pf_solver != nullptr);
        ASSERT_BACKEND_EQ(BackendType::GPU_CUDA, pf_solver->get_backend_type());
    }

    // Test should pass regardless of GPU availability
    ASSERT_TRUE(true);
}

void register_powerflow_tests(TestRunner& runner) {
    runner.add_test("CPU PowerFlow Creation", test_cpu_powerflow_creation);
    runner.add_test("PowerFlow Config", test_powerflow_config);
    runner.add_test("Mismatch Calculation", test_mismatch_calculation);
    runner.add_test("PowerFlow Solve Simple", test_powerflow_solve_simple);
    runner.add_test("GPU PowerFlow Availability", test_gpu_powerflow_availability);
}
