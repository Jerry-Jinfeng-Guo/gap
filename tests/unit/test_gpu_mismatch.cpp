#include <cmath>
#include <iostream>

#include "gap/core/backend_factory.h"
#include "gap/solver/powerflow_interface.h"

#include "test_framework.h"

using namespace gap;

namespace test_helpers {

// Helper function to create a simple 2-bus test network
NetworkData create_simple_2bus_network() {
    NetworkData network;
    network.num_buses = 2;

    // Bus 1: Slack bus
    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 241500.0,
                    .u_pu = 1.05,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};

    // Bus 2: Load bus (PQ)
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};

    network.buses = {bus1, bus2};

    // Add load appliance
    ApplianceData load = {
        .id = 10,
        .node = 2,
        .status = 1,
        .type = ApplianceType::LOADGEN,
        .p_specified = -50e6,  // 50 MW load
        .q_specified = -30e6   // 30 MVAr load
    };

    network.appliances = {load};

    return network;
}

// Helper function to create a simple admittance matrix
SparseMatrix create_simple_admittance_matrix() {
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;

    // CSR format
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};

    // Simple symmetric admittance matrix
    matrix.values = {
        Complex(10.0, -20.0),  // Y11
        Complex(-5.0, 10.0),   // Y12
        Complex(-5.0, 10.0),   // Y21
        Complex(10.0, -20.0)   // Y22
    };

    return matrix;
}

// Helper to create 3-bus network
NetworkData create_3bus_network() {
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
                    .bus_type = BusType::PV,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 80e6,
                    .reactive_power = 0.0};

    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 218500.0,
                    .u_pu = 0.95,
                    .u_angle = -0.1,
                    .active_power = 0.0,
                    .reactive_power = 0.0};

    network.buses = {bus1, bus2, bus3};

    ApplianceData gen = {.id = 10,
                         .node = 2,
                         .status = 1,
                         .type = ApplianceType::SOURCE,
                         .p_specified = 80e6,
                         .q_specified = 0.0};

    ApplianceData load = {.id = 11,
                          .node = 3,
                          .status = 1,
                          .type = ApplianceType::LOADGEN,
                          .p_specified = -100e6,
                          .q_specified = -50e6};

    network.appliances = {gen, load};

    return network;
}

SparseMatrix create_3bus_admittance_matrix() {
    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 9;
    matrix.row_ptr = {0, 3, 6, 9};
    matrix.col_idx = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    matrix.values = {Complex(15.0, -30.0), Complex(-5.0, 10.0),  Complex(-10.0, 20.0),
                     Complex(-5.0, 10.0),  Complex(20.0, -25.0), Complex(-15.0, 15.0),
                     Complex(-10.0, 20.0), Complex(-15.0, 15.0), Complex(25.0, -35.0)};

    return matrix;
}

}  // namespace test_helpers

// Test 1: Unit test CPU mismatch calculation directly
void test_cpu_mismatch_calculation_unit() {
    std::cout << "\n=== Test: CPU Mismatch Calculation (Unit Test) ===" << std::endl;

    // Create solver
    auto cpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto cpu_lu = core::BackendFactory::create_lu_solver(BackendType::CPU);
    cpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu.release()));

    // Create test network
    auto network = test_helpers::create_simple_2bus_network();
    auto matrix = test_helpers::create_simple_admittance_matrix();

    // Test voltages (known state)
    ComplexVector voltages = {
        Complex(1.05, 0.0),  // Slack bus at 1.05 p.u.
        Complex(1.0, -0.01)  // PQ bus at 1.0 p.u., slight angle
    };

    // Call calculate_mismatches directly (unit test!)
    auto mismatches = cpu_solver->calculate_mismatches(network, voltages, matrix);

    std::cout << "  Number of mismatches: " << mismatches.size() << std::endl;

    // For 2-bus system with 1 slack + 1 PQ: should have 2 mismatches (P and Q for PQ bus)
    ASSERT_EQ(2, mismatches.size());

    // Mismatches should be finite numbers
    for (size_t i = 0; i < mismatches.size(); ++i) {
        std::cout << "  Mismatch[" << i << "] = " << mismatches[i] << std::endl;
        ASSERT_TRUE(std::isfinite(mismatches[i]));
    }

    std::cout << "✓ CPU mismatch calculation unit test passed!" << std::endl;
}

// Test 2: Unit test GPU mismatch calculation directly
void test_gpu_mismatch_calculation_unit() {
    if (!core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Test: GPU Mismatch Calculation (Unit Test) ===" << std::endl;

    // Create GPU solver
    auto gpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto gpu_lu = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    gpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu.release()));

    // Create test network
    auto network = test_helpers::create_simple_2bus_network();
    auto matrix = test_helpers::create_simple_admittance_matrix();

    // Test voltages (known state)
    ComplexVector voltages = {
        Complex(1.05, 0.0),  // Slack bus at 1.05 p.u.
        Complex(1.0, -0.01)  // PQ bus at 1.0 p.u., slight angle
    };

    // Call calculate_mismatches directly (unit test!)
    auto mismatches = gpu_solver->calculate_mismatches(network, voltages, matrix);

    std::cout << "  Number of mismatches: " << mismatches.size() << std::endl;

    // For 2-bus system with 1 slack + 1 PQ: should have 2 mismatches (P and Q for PQ bus)
    ASSERT_EQ(2, mismatches.size());

    // Mismatches should be finite numbers
    for (size_t i = 0; i < mismatches.size(); ++i) {
        std::cout << "  Mismatch[" << i << "] = " << mismatches[i] << std::endl;
        ASSERT_TRUE(std::isfinite(mismatches[i]));
    }

    // GPU mismatches should NOT all be zero (current bug)
    bool all_zero = true;
    for (auto m : mismatches) {
        if (std::abs(m) > 1e-10) {
            all_zero = false;
            break;
        }
    }

    if (all_zero) {
        std::cout << "  *** BUG DETECTED: All GPU mismatches are zero! ***" << std::endl;
        std::cout << "  This indicates calculate_mismatches() is not properly implemented for GPU"
                  << std::endl;
    }

    std::cout << "✓ GPU mismatch calculation unit test completed" << std::endl;
}

// Test 3: Compare CPU vs GPU mismatch calculation results
void test_cpu_vs_gpu_mismatch_comparison() {
    if (!core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Test: CPU vs GPU Mismatch Comparison ===" << std::endl;

    // Create solvers
    auto cpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto cpu_lu = core::BackendFactory::create_lu_solver(BackendType::CPU);
    cpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu.release()));

    auto gpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto gpu_lu = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    gpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu.release()));

    // Create test network
    auto network = test_helpers::create_simple_2bus_network();
    auto matrix = test_helpers::create_simple_admittance_matrix();

    // Test voltages
    ComplexVector voltages = {Complex(1.05, 0.0), Complex(1.0, -0.01)};

    // Calculate mismatches on both backends
    auto cpu_mismatches = cpu_solver->calculate_mismatches(network, voltages, matrix);
    auto gpu_mismatches = gpu_solver->calculate_mismatches(network, voltages, matrix);

    std::cout << "  CPU mismatches: " << cpu_mismatches.size() << std::endl;
    std::cout << "  GPU mismatches: " << gpu_mismatches.size() << std::endl;

    ASSERT_EQ(cpu_mismatches.size(), gpu_mismatches.size());

    // Compare values
    Float max_diff = 0.0;
    std::cout << "\n  Detailed comparison:" << std::endl;
    std::cout << "  Index | CPU Mismatch  | GPU Mismatch  | Difference" << std::endl;
    std::cout << "  " << std::string(60, '-') << std::endl;

    for (size_t i = 0; i < cpu_mismatches.size(); ++i) {
        Float diff = std::abs(cpu_mismatches[i] - gpu_mismatches[i]);
        max_diff = std::max(max_diff, diff);

        printf("  %5zu | %13.6e | %13.6e | %13.6e\n", i, cpu_mismatches[i], gpu_mismatches[i],
               diff);
    }

    std::cout << "  " << std::string(60, '-') << std::endl;
    std::cout << "  Max difference: " << max_diff << std::endl;

    // Tolerance for comparison (should be very small for same inputs)
    Float tolerance = 1e-6;

    if (max_diff > tolerance) {
        std::cout << "  *** MISMATCH DIFFERENCE EXCEEDS TOLERANCE ***" << std::endl;
        std::cout << "  This indicates GPU mismatch calculation has a bug!" << std::endl;
    } else {
        std::cout << "  CPU and GPU mismatches match within tolerance" << std::endl;
    }

    std::cout << "✓ CPU vs GPU mismatch comparison completed" << std::endl;
}

// Test 4: Test with 3-bus network including PV bus
void test_mismatch_with_pv_bus() {
    if (!core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Test: Mismatch with PV Bus ===" << std::endl;

    auto cpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto cpu_lu = core::BackendFactory::create_lu_solver(BackendType::CPU);
    cpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu.release()));

    auto gpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto gpu_lu = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    gpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu.release()));

    auto network = test_helpers::create_3bus_network();
    auto matrix = test_helpers::create_3bus_admittance_matrix();

    ComplexVector voltages = {
        Complex(1.05, 0.0),   // Slack bus
        Complex(1.0, 0.0),    // PV bus
        Complex(0.95, -0.05)  // PQ bus
    };

    auto cpu_mismatches = cpu_solver->calculate_mismatches(network, voltages, matrix);
    auto gpu_mismatches = gpu_solver->calculate_mismatches(network, voltages, matrix);

    std::cout << "  CPU mismatches: " << cpu_mismatches.size() << std::endl;
    std::cout << "  GPU mismatches: " << gpu_mismatches.size() << std::endl;

    // 3-bus system: 1 slack (0 equations) + 1 PV (1 P equation) + 1 PQ (2 equations) = 3 total
    ASSERT_EQ(3, cpu_mismatches.size());
    ASSERT_EQ(3, gpu_mismatches.size());

    // Compare
    Float max_diff = 0.0;
    for (size_t i = 0; i < cpu_mismatches.size(); ++i) {
        Float diff = std::abs(cpu_mismatches[i] - gpu_mismatches[i]);
        max_diff = std::max(max_diff, diff);
        std::cout << "  Mismatch[" << i << "]: CPU=" << cpu_mismatches[i]
                  << ", GPU=" << gpu_mismatches[i] << ", diff=" << diff << std::endl;
    }

    std::cout << "  Max difference: " << max_diff << std::endl;
    std::cout << "✓ PV bus mismatch test completed" << std::endl;
}

void register_gpu_mismatch_tests(TestRunner& runner) {
    runner.add_test("CPU Mismatch - Unit Test", test_cpu_mismatch_calculation_unit);
    runner.add_test("GPU Mismatch - Unit Test", test_gpu_mismatch_calculation_unit);
    runner.add_test("CPU vs GPU Mismatch - Comparison", test_cpu_vs_gpu_mismatch_comparison);
    runner.add_test("Mismatch with PV Bus", test_mismatch_with_pv_bus);
}
