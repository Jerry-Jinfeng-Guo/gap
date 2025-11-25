/**
 * @file test_gpu_state_capture.cpp
 * @brief Test GPU solver with state capture enabled
 *
 * This test runs the GPU solver with state capture enabled to observe
 * internal states (voltages, currents, power injections, mismatches) after
 * each Newton-Raphson iteration.
 */

#include <cmath>
#include <iomanip>
#include <iostream>

#include "gap/admittance/admittance_interface.h"
#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/logging/logger.h"
#include "gap/solver/powerflow_interface.h"

using namespace gap;
using namespace gap::solver;

void print_iteration_state(const solver::IterationState& state, int num_buses_to_print = 5) {
    std::cout << "\n=== Iteration " << state.iteration << " ===" << std::endl;
    std::cout << "Max mismatch: " << std::scientific << std::setprecision(6) << state.max_mismatch
              << std::endl;

    int n = std::min(num_buses_to_print, static_cast<int>(state.voltages.size()));

    std::cout << "\nVoltages (first " << n << " buses):" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "  Bus " << i << ": " << std::fixed << std::setprecision(6)
                  << state.voltages[i].real() << " + " << state.voltages[i].imag() << "j"
                  << "  (mag=" << std::abs(state.voltages[i])
                  << ", ang=" << std::arg(state.voltages[i]) * 180.0 / M_PI << "°)" << std::endl;
    }

    std::cout << "\nCurrents (first " << n << " buses):" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "  Bus " << i << ": " << std::scientific << std::setprecision(6)
                  << state.currents[i].real() << " + " << state.currents[i].imag() << "j"
                  << std::endl;
    }

    std::cout << "\nPower Injections (first " << n << " buses):" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "  Bus " << i << ": P=" << std::fixed << std::setprecision(6)
                  << state.power_injections[i].real() << " Q=" << state.power_injections[i].imag()
                  << " p.u." << std::endl;
    }

    int m = std::min(10, static_cast<int>(state.mismatches.size()));
    std::cout << "\nMismatches (first " << m << "):" << std::endl;
    for (int i = 0; i < m; ++i) {
        std::cout << "  Mismatch[" << i << "]: " << std::scientific << std::setprecision(6)
                  << state.mismatches[i] << std::endl;
    }
}

int main(int /*argc*/, char** /*argv*/) {
    auto& logger = logging::global_logger;
    logger.setComponent("GPUStateCapture");
    logger.configure(logging::LogLevel::INFO, logging::LogOutput::CONSOLE);

    std::cout << "========================================" << std::endl;
    std::cout << "GPU State Capture Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // Create a simple 3-bus network
        std::cout << "\nCreating 3-bus test network..." << std::endl;
        NetworkData network;
        network.num_buses = 3;

        // Bus 1: Slack
        network.buses.push_back({.id = 1,
                                 .u_rated = 230000.0,
                                 .bus_type = BusType::SLACK,
                                 .energized = 1,
                                 .u = 230000.0,
                                 .u_pu = 1.0,
                                 .u_angle = 0.0,
                                 .active_power = 0.0,
                                 .reactive_power = 0.0});

        // Bus 2: PV
        network.buses.push_back({
            .id = 2,
            .u_rated = 230000.0,
            .bus_type = BusType::PV,
            .energized = 1,
            .u = 230000.0,
            .u_pu = 1.0,
            .u_angle = 0.0,
            .active_power = 50e6,
            .reactive_power = 0.0  // 50 MW generation
        });

        // Bus 3: PQ
        network.buses.push_back({
            .id = 3,
            .u_rated = 230000.0,
            .bus_type = BusType::PQ,
            .energized = 1,
            .u = 230000.0,
            .u_pu = 1.0,
            .u_angle = 0.0,
            .active_power = -30e6,
            .reactive_power = -20e6  // 30 MW + 20 MVAr load
        });

        // Add branches
        network.branches.push_back(
            {.id = 1, .from_bus = 1, .from_status = 1, .to_bus = 2, .to_status = 1, .status = 1});
        network.branches.push_back(
            {.id = 2, .from_bus = 2, .from_status = 1, .to_bus = 3, .to_status = 1, .status = 1});
        network.branches.push_back(
            {.id = 3, .from_bus = 1, .from_status = 1, .to_bus = 3, .to_status = 1, .status = 1});

        std::cout << "Network created:" << std::endl;
        std::cout << "  Buses: " << network.num_buses << std::endl;
        std::cout << "  Branches: " << network.branches.size() << std::endl;

        // Build admittance matrix
        std::cout << "\nBuilding admittance matrix..." << std::endl;
        auto gpu_admittance =
            core::BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
        auto y_matrix = gpu_admittance->build_admittance_matrix(network);

        // Configure power flow
        solver::PowerFlowConfig config;
        config.tolerance = 1e-6;
        config.max_iterations = 20;
        config.acceleration_factor = 1.0;
        config.use_flat_start = true;
        config.verbose = true;
        config.base_power = 100e6;  // 100 MVA

        // Create GPU solver
        std::cout << "\nCreating GPU solver..." << std::endl;
        auto gpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        auto gpu_lu = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        gpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu.release()));

        // Enable state capture
        gpu_solver->enable_state_capture(true);

        // Solve
        std::cout << "\nSolving power flow with state capture enabled..." << std::endl;
        auto result = gpu_solver->solve_power_flow(network, *y_matrix, config);

        std::cout << "\n========================================" << std::endl;
        std::cout << "Solution Result" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Converged: " << (result.converged ? "YES" : "NO") << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Final mismatch: " << std::scientific << result.final_mismatch << std::endl;

        // Print captured states
        auto const& states = gpu_solver->get_iteration_states();
        std::cout << "\n========================================" << std::endl;
        std::cout << "Captured " << states.size() << " iteration states" << std::endl;
        std::cout << "========================================" << std::endl;

        for (auto const& state : states) {
            print_iteration_state(state);
        }

        return result.converged ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
