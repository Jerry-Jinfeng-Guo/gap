/**
 * @file debug_simple_2bus.cpp
 * @brief Debug program for the simple 2-bus test case
 *
 * This program replicates the exact logic used in the Python bindings
 * to debug the simple_2bus test case and compare against PGM reference.
 */

#include <cmath>
#include <iomanip>
#include <iostream>

#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/logging/logger.h"
#include "gap/solver/powerflow_interface.h"

using namespace gap;

/**
 * @brief Print complex voltage in engineering format
 */
void print_voltage(int bus_id, Complex voltage, double u_rated) {
    double magnitude = std::abs(voltage);
    double angle_rad = std::arg(voltage);
    double angle_deg = angle_rad * 180.0 / M_PI;
    double u_volt = magnitude * u_rated;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Bus " << bus_id << ": " << magnitude << "∠" << angle_deg << "° pu"
              << " (" << u_volt / 1000.0 << " kV)" << std::endl;
}

int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << "GAP Simple 2-Bus Debug Program" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << std::endl;

    std::cout << "📋 Test Case: Simple 2-Bus System" << std::endl;
    std::cout << "  Base: 100 MVA, 230 kV" << std::endl;
    std::cout << "  Bus 1: Source, 1.05 pu" << std::endl;
    std::cout << "  Bus 2: Load, 100 MW + j50 MVAR" << std::endl;
    std::cout << "  Line: 0.01 + j0.1 pu (5.29 + j52.9 Ω)" << std::endl;
    std::cout << std::endl;

    std::cout << "📊 PGM Reference Solution:" << std::endl;
    std::cout << "  Bus 1: 1.0439∠-0.53° pu" << std::endl;
    std::cout << "  Bus 2: 0.9780∠-5.87° pu" << std::endl;
    std::cout << std::endl;

    try {
        // System parameters
        const double base_voltage = 230000.0;                                      // 230 kV
        const double base_power = 100e6;                                           // 100 MVA
        const double base_impedance = (base_voltage * base_voltage) / base_power;  // 529 Ω

        // Create network data (matching Python bindings format)
        NetworkData network_data;
        network_data.num_buses = 2;
        network_data.num_branches = 1;
        network_data.num_appliances = 0;

        // Bus 1: Slack bus at 1.05 pu
        BusData bus1;
        bus1.id = 1;
        bus1.u_rated = base_voltage;
        bus1.bus_type = BusType::SLACK;
        bus1.energized = true;
        bus1.u_pu = 1.05;           // Initial voltage
        bus1.u_angle = 0.0;         // Initial angle
        bus1.active_power = 0.0;    // Will be calculated
        bus1.reactive_power = 0.0;  // Will be calculated

        // Bus 2: PQ bus with load
        BusData bus2;
        bus2.id = 2;
        bus2.u_rated = base_voltage;
        bus2.bus_type = BusType::PQ;
        bus2.energized = true;
        bus2.u_pu = 1.0;              // Flat start
        bus2.u_angle = 0.0;           // Flat start
        bus2.active_power = -100e6;   // 100 MW load (negative for load)
        bus2.reactive_power = -50e6;  // 50 MVAR load (negative for load)

        network_data.buses = {bus1, bus2};

        // Branch: Line from bus 1 to bus 2
        BranchData branch;
        branch.id = 1;
        branch.from_bus = 1;
        branch.to_bus = 2;
        branch.branch_type = BranchType::LINE;
        branch.status = true;
        branch.r1 = 5.29;  // Resistance in Ohms
        branch.x1 = 52.9;  // Reactance in Ohms
        branch.b1 = 0.0;   // Susceptance in Siemens

        network_data.branches = {branch};

        std::cout << "🔧 Building Network Components..." << std::endl;
        std::cout << "  Buses: " << network_data.num_buses << std::endl;
        std::cout << "  Branches: " << network_data.num_branches << std::endl;
        std::cout << "  Base impedance: " << base_impedance << " Ω" << std::endl;
        std::cout << std::endl;

        // Create solver components
        std::cout << "🔧 Creating Solver Components..." << std::endl;
        auto admittance_builder = core::BackendFactory::create_admittance_backend(BackendType::CPU);
        auto powerflow_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
        auto lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);

        // Set LU solver
        powerflow_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(lu_solver.release()));
        std::cout << "  ✓ Admittance builder created" << std::endl;
        std::cout << "  ✓ Power flow solver created" << std::endl;
        std::cout << "  ✓ LU solver created and assigned" << std::endl;
        std::cout << std::endl;

        // Build admittance matrix
        std::cout << "🔧 Building Admittance Matrix..." << std::endl;
        auto admittance_matrix = admittance_builder->build_admittance_matrix(network_data);
        std::cout << "  ✓ Matrix size: " << admittance_matrix->num_rows << "x"
                  << admittance_matrix->num_cols << std::endl;
        std::cout << "  ✓ Non-zeros: " << admittance_matrix->nnz << std::endl;
        std::cout << std::endl;

        // Configure power flow solver
        solver::PowerFlowConfig config;
        config.tolerance = 1e-4;      // Relaxed tolerance
        config.max_iterations = 100;  // More iterations
        config.verbose = true;        // Enable verbose output
        config.use_flat_start = true;

        std::cout << "⚙️  Power Flow Configuration:" << std::endl;
        std::cout << "  Tolerance: " << config.tolerance << std::endl;
        std::cout << "  Max iterations: " << config.max_iterations << std::endl;
        std::cout << "  Flat start: " << (config.use_flat_start ? "Yes" : "No") << std::endl;
        std::cout << std::endl;

        // Solve power flow
        std::cout << "🚀 Running Newton-Raphson Power Flow..." << std::endl;
        std::cout << "======================================================" << std::endl;
        auto result = powerflow_solver->solve_power_flow(network_data, *admittance_matrix, config);
        std::cout << "======================================================" << std::endl;
        std::cout << std::endl;

        // Print results
        std::cout << "📊 GAP Solution Results:" << std::endl;
        std::cout << "  Status: " << (result.converged ? "✅ CONVERGED" : "❌ NOT CONVERGED")
                  << std::endl;
        std::cout << "  Iterations: " << result.iterations << std::endl;
        std::cout << "  Final mismatch: " << std::scientific << std::setprecision(2)
                  << result.final_mismatch << std::endl;
        std::cout << std::endl;

        if (result.bus_voltages.size() >= 2) {
            std::cout << "Bus Voltages:" << std::endl;
            for (size_t i = 0; i < result.bus_voltages.size(); ++i) {
                print_voltage(i + 1, result.bus_voltages[i], base_voltage);
            }
            std::cout << std::endl;

            // Compare with PGM reference
            std::cout << "📊 Comparison with PGM Reference:" << std::endl;
            std::cout << "------------------------------------------------------" << std::endl;

            // Bus 1 comparison
            double gap_mag_1 = std::abs(result.bus_voltages[0]);
            double gap_ang_1 = std::arg(result.bus_voltages[0]) * 180.0 / M_PI;
            double ref_mag_1 = 1.0439;
            double ref_ang_1 = -0.53;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Bus 1:" << std::endl;
            std::cout << "  GAP:       " << gap_mag_1 << "∠" << gap_ang_1 << "°" << std::endl;
            std::cout << "  PGM Ref:   " << ref_mag_1 << "∠" << ref_ang_1 << "°" << std::endl;
            std::cout << "  Mag Error: " << std::abs(gap_mag_1 - ref_mag_1) << " ("
                      << (std::abs(gap_mag_1 - ref_mag_1) / ref_mag_1 * 100.0) << "%)" << std::endl;
            std::cout << "  Ang Error: " << std::abs(gap_ang_1 - ref_ang_1) << "°" << std::endl;
            std::cout << std::endl;

            // Bus 2 comparison
            double gap_mag_2 = std::abs(result.bus_voltages[1]);
            double gap_ang_2 = std::arg(result.bus_voltages[1]) * 180.0 / M_PI;
            double ref_mag_2 = 0.9780;
            double ref_ang_2 = -5.87;
            std::cout << "Bus 2:" << std::endl;
            std::cout << "  GAP:       " << gap_mag_2 << "∠" << gap_ang_2 << "°" << std::endl;
            std::cout << "  PGM Ref:   " << ref_mag_2 << "∠" << ref_ang_2 << "°" << std::endl;
            std::cout << "  Mag Error: " << std::abs(gap_mag_2 - ref_mag_2) << " ("
                      << (std::abs(gap_mag_2 - ref_mag_2) / ref_mag_2 * 100.0) << "%)" << std::endl;
            std::cout << "  Ang Error: " << std::abs(gap_ang_2 - ref_ang_2) << "°" << std::endl;
            std::cout << std::endl;

            // Validation
            bool mag_1_ok = std::abs(gap_mag_1 - ref_mag_1) < 0.01;  // 1% tolerance
            bool ang_1_ok = std::abs(gap_ang_1 - ref_ang_1) < 1.0;   // 1 degree tolerance
            bool mag_2_ok = std::abs(gap_mag_2 - ref_mag_2) < 0.01;
            bool ang_2_ok = std::abs(gap_ang_2 - ref_ang_2) < 1.0;

            if (mag_1_ok && ang_1_ok && mag_2_ok && ang_2_ok) {
                std::cout << "✅ VALIDATION PASSED - Results match PGM reference!" << std::endl;
            } else {
                std::cout << "❌ VALIDATION FAILED - Results differ from PGM reference"
                          << std::endl;
                std::cout << "   Bus 1: " << (mag_1_ok && ang_1_ok ? "✅ OK" : "❌ FAIL")
                          << std::endl;
                std::cout << "   Bus 2: " << (mag_2_ok && ang_2_ok ? "✅ OK" : "❌ FAIL")
                          << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << "======================================================" << std::endl;

        return result.converged ? 0 : 1;

    } catch (std::exception const& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
