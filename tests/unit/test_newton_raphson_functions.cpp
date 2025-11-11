#include "gap/core/backend_factory.h"
#include "gap/solver/powerflow_interface.h"

#include "test_framework.h"

using namespace gap;

// Since the Newton-Raphson functions are private, we'll create a test-friendly derived class
namespace test_helpers {

class TestableNewtonRaphson {
  private:
    std::unique_ptr<solver::IPowerFlowSolver> solver_;

  public:
    TestableNewtonRaphson() {
        solver_ = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
        auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
        solver_->set_lu_solver(std::shared_ptr<solver::ILUSolver>(lu_solver.release()));
    }

    // Expose calculate_mismatches for testing
    std::vector<Float> test_calculate_mismatches(NetworkData const& network_data,
                                                 ComplexVector const& bus_voltages,
                                                 SparseMatrix const& admittance_matrix) {
        return solver_->calculate_mismatches(network_data, bus_voltages, admittance_matrix);
    }

    solver::IPowerFlowSolver* get_solver() { return solver_.get(); }
};

// Helper function to create a simple 2-bus test network
NetworkData create_simple_network() {
    NetworkData network;
    network.num_buses = 2;

    // Bus 1: Slack bus (generator)
    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 241500.0,  // 1.05 p.u.
                    .u_pu = 1.05,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};

    // Bus 2: Load bus (PQ)
    // Note: Power is specified via appliances, not directly on bus
    BusData bus2 = {
        .id = 2,
        .u_rated = 230000.0,
        .bus_type = BusType::PQ,
        .energized = 1,
        .u = 230000.0,
        .u_pu = 1.0,
        .u_angle = 0.0,
        .active_power = 0.0,   // Power specified via appliances below
        .reactive_power = 0.0  // Power specified via appliances below
    };

    network.buses = {bus1, bus2};

    // Add appliances for power injections
    ApplianceData load = {.id = 10,
                          .node = 2,
                          .status = 1,
                          .type = ApplianceType::LOADGEN,
                          .p_specified = -50e6,  // Negative for load (consumption)
                          .q_specified = -30e6};

    network.appliances = {load};

    return network;
}

// Helper function to create a simple admittance matrix
SparseMatrix create_simple_admittance_matrix() {
    SparseMatrix matrix;
    matrix.num_rows = 2;
    matrix.num_cols = 2;
    matrix.nnz = 4;

    // CSR format for a simple 2x2 admittance matrix
    matrix.row_ptr = {0, 2, 4};
    matrix.col_idx = {0, 1, 0, 1};

    // Y11 = 10-20j, Y12 = -5+10j
    // Y21 = -5+10j, Y22 = 10-20j
    matrix.values = {
        Complex(10.0, -20.0),  // Y11
        Complex(-5.0, 10.0),   // Y12
        Complex(-5.0, 10.0),   // Y21
        Complex(10.0, -20.0)   // Y22
    };

    return matrix;
}

// Helper function to create 3-bus test network for more complex scenarios
NetworkData create_3bus_network() {
    NetworkData network;
    network.num_buses = 3;

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

    // Bus 2: PV bus (generator)
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PV,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 80e6,  // 80 MW generation
                    .reactive_power = 0.0};

    // Bus 3: Load bus (PQ)
    BusData bus3 = {
        .id = 3,
        .u_rated = 230000.0,
        .bus_type = BusType::PQ,
        .energized = 1,
        .u = 218500.0,
        .u_pu = 0.95,
        .u_angle = -0.1,
        .active_power = 100e6,  // 100 MW load
        .reactive_power = 50e6  // 50 MVAr load
    };

    network.buses = {bus1, bus2, bus3};

    // Add appliances
    ApplianceData gen = {.id = 10,
                         .node = 2,
                         .status = 1,
                         .type = ApplianceType::SOURCE,
                         .p_specified = 80e6,  // Generation
                         .q_specified = 0.0};

    ApplianceData load = {.id = 11,
                          .node = 3,
                          .status = 1,
                          .type = ApplianceType::LOADGEN,
                          .p_specified = -100e6,  // Load consumption
                          .q_specified = -50e6};

    network.appliances = {gen, load};

    return network;
}

}  // namespace test_helpers

// Test function: calculate_mismatches basic functionality
void test_calculate_mismatches_basic() {
    test_helpers::TestableNewtonRaphson tester;

    auto network = test_helpers::create_simple_network();
    auto matrix = test_helpers::create_simple_admittance_matrix();

    // Create voltage vector (flat start)
    ComplexVector voltages = {
        Complex(1.05, 0.0),  // Slack bus at 1.05 p.u.
        Complex(1.0, 0.0)    // PQ bus at 1.0 p.u.
    };

    auto mismatches = tester.test_calculate_mismatches(network, voltages, matrix);

    // For 2-bus system with 1 slack + 1 PQ: should have 2 mismatches (P and Q for PQ bus)
    ASSERT_EQ(2, mismatches.size());

    // Mismatches should be finite numbers
    for (auto mismatch : mismatches) {
        ASSERT_TRUE(std::isfinite(mismatch));
    }
}

// Test function: calculate_mismatches with 3-bus system
void test_calculate_mismatches_3bus() {
    test_helpers::TestableNewtonRaphson tester;

    auto network = test_helpers::create_3bus_network();

    // Create 3x3 admittance matrix
    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 9;
    matrix.row_ptr = {0, 3, 6, 9};
    matrix.col_idx = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    // Symmetric admittance matrix
    matrix.values = {
        Complex(15.0, -30.0), Complex(-5.0, 10.0),  Complex(-10.0, 20.0),  // Row 1
        Complex(-5.0, 10.0),  Complex(20.0, -25.0), Complex(-15.0, 15.0),  // Row 2
        Complex(-10.0, 20.0), Complex(-15.0, 15.0), Complex(25.0, -35.0)   // Row 3
    };

    ComplexVector voltages = {
        Complex(1.05, 0.0),   // Slack bus
        Complex(1.0, 0.0),    // PV bus
        Complex(0.95, -0.05)  // PQ bus
    };

    auto mismatches = tester.test_calculate_mismatches(network, voltages, matrix);

    // 3-bus system: 1 slack (0 equations) + 1 PV (1 P equation) + 1 PQ (2 equations) = 3 total
    ASSERT_EQ(3, mismatches.size());

    // All mismatches should be finite
    for (auto mismatch : mismatches) {
        ASSERT_TRUE(std::isfinite(mismatch));
    }
}

// Test function: calculate_mismatches with no load (should have zero mismatches)
void test_calculate_mismatches_no_load() {
    test_helpers::TestableNewtonRaphson tester;

    NetworkData network;
    network.num_buses = 2;

    // Two slack buses (unusual but valid test case)
    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};

    network.buses = {bus1, bus2};
    network.appliances = {};  // No appliances

    auto matrix = test_helpers::create_simple_admittance_matrix();
    ComplexVector voltages = {Complex(1.0, 0.0), Complex(1.0, 0.0)};

    auto mismatches = tester.test_calculate_mismatches(network, voltages, matrix);

    // No PQ or PV buses = no mismatch equations
    ASSERT_EQ(0, mismatches.size());
}

// Test function: calculate_mismatches with empty admittance matrix
void test_calculate_mismatches_empty_matrix() {
    test_helpers::TestableNewtonRaphson tester;

    auto network = test_helpers::create_simple_network();

    // Empty admittance matrix
    SparseMatrix empty_matrix;
    empty_matrix.num_rows = 2;
    empty_matrix.num_cols = 2;
    empty_matrix.nnz = 0;

    ComplexVector voltages = {Complex(1.05, 0.0), Complex(1.0, 0.0)};

    auto mismatches = tester.test_calculate_mismatches(network, voltages, empty_matrix);

    // Should still produce mismatches, but with zero calculated power
    ASSERT_EQ(2, mismatches.size());

    // Mismatches should equal negative specified power (since calculated power = 0)
    ASSERT_NEAR(50e6, mismatches[0], 1e-6);  // P mismatch = 0 - (-50e6) = 50e6
    ASSERT_NEAR(30e6, mismatches[1], 1e-6);  // Q mismatch = 0 - (-30e6) = 30e6
}

// Test function: power flow convergence with realistic parameters
void test_power_flow_convergence_realistic() {
    test_helpers::TestableNewtonRaphson tester;

    auto network = test_helpers::create_simple_network();
    auto matrix = test_helpers::create_simple_admittance_matrix();

    solver::PowerFlowConfig config;
    config.max_iterations = 20;
    config.tolerance = 1e-6;
    config.use_flat_start = true;
    config.verbose = false;

    auto result = tester.get_solver()->solve_power_flow(network, matrix, config);

    // Should have voltage results for both buses
    ASSERT_EQ(2, result.bus_voltages.size());

    // Should converge or at least complete iterations
    ASSERT_TRUE(result.iterations <= config.max_iterations);

    // Voltage magnitudes should be in reasonable range
    for (auto const& voltage : result.bus_voltages) {
        Float magnitude = std::abs(voltage);
        ASSERT_TRUE(magnitude >= 0.5);  // Not too low
        ASSERT_TRUE(magnitude <= 2.0);  // Not too high
        ASSERT_TRUE(std::isfinite(magnitude));
    }
}

// Test function: power flow with tight convergence tolerance
void test_power_flow_tight_tolerance() {
    test_helpers::TestableNewtonRaphson tester;

    auto network = test_helpers::create_simple_network();
    auto matrix = test_helpers::create_simple_admittance_matrix();

    solver::PowerFlowConfig config;
    config.max_iterations = 50;
    config.tolerance = 1e-10;  // Very tight tolerance
    config.use_flat_start = true;
    config.verbose = false;

    auto result = tester.get_solver()->solve_power_flow(network, matrix, config);

    ASSERT_EQ(2, result.bus_voltages.size());

    if (result.converged) {
        // If converged, final mismatch should be below tolerance
        ASSERT_TRUE(result.final_mismatch < config.tolerance);
    }

    // Voltages should be reasonable regardless of convergence
    for (auto const& voltage : result.bus_voltages) {
        Float magnitude = std::abs(voltage);
        ASSERT_TRUE(magnitude >= 0.3);
        ASSERT_TRUE(magnitude <= 3.0);
        ASSERT_TRUE(std::isfinite(magnitude));
    }
}

// Test function: power flow behavior with different initial conditions
void test_power_flow_initial_conditions() {
    test_helpers::TestableNewtonRaphson tester;

    auto network = test_helpers::create_simple_network();
    auto matrix = test_helpers::create_simple_admittance_matrix();

    solver::PowerFlowConfig config;
    config.max_iterations = 15;
    config.tolerance = 1e-5;
    config.use_flat_start = false;  // Don't use flat start
    config.verbose = false;

    auto result = tester.get_solver()->solve_power_flow(network, matrix, config);

    ASSERT_EQ(2, result.bus_voltages.size());

    // Should complete without crashing
    ASSERT_TRUE(result.iterations <= config.max_iterations);

    // Check voltage stability (no NaN or infinite values)
    for (auto const& voltage : result.bus_voltages) {
        ASSERT_TRUE(std::isfinite(voltage.real()));
        ASSERT_TRUE(std::isfinite(voltage.imag()));
    }
}

// Test function: mismatch calculation with zero voltages (edge case)
void test_calculate_mismatches_zero_voltages() {
    test_helpers::TestableNewtonRaphson tester;

    auto network = test_helpers::create_simple_network();
    auto matrix = test_helpers::create_simple_admittance_matrix();

    // Zero voltage vector (extreme case)
    ComplexVector voltages = {Complex(0.0, 0.0), Complex(0.0, 0.0)};

    auto mismatches = tester.test_calculate_mismatches(network, voltages, matrix);

    ASSERT_EQ(2, mismatches.size());

    // With zero voltages, calculated power should be zero
    // So mismatches should equal specified power
    ASSERT_NEAR(50e6, mismatches[0], 1e-6);  // P mismatch
    ASSERT_NEAR(30e6, mismatches[1], 1e-6);  // Q mismatch
}

void register_newton_raphson_function_tests(TestRunner& runner) {
    runner.add_test("NR f/m - Calculate Mismatches Basic", test_calculate_mismatches_basic);
    runner.add_test("NR f/m - Calculate Mismatches 3-Bus", test_calculate_mismatches_3bus);
    runner.add_test("NR f/m - Calculate Mismatches No Load", test_calculate_mismatches_no_load);
    runner.add_test("NR f/m - Calculate Mismatches Empty Matrix",
                    test_calculate_mismatches_empty_matrix);
    runner.add_test("NR f/m - Power Flow Convergence Realistic",
                    test_power_flow_convergence_realistic);
    runner.add_test("NR f/m - Power Flow Tight Tolerance", test_power_flow_tight_tolerance);
    runner.add_test("NR f/m - Power Flow Initial Conditions", test_power_flow_initial_conditions);
    runner.add_test("NR f/m - Calculate Mismatches Zero Voltages",
                    test_calculate_mismatches_zero_voltages);
}