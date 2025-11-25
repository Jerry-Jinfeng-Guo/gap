/**
 * @file debug_gpu_mismatch.cpp
 * @brief Debug GPU mismatch calculation using state capture
 *
 * Uses state capture to compare GPU vs CPU mismatch calculations
 * on the same failing test case.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "gap/admittance/admittance_interface.h"
#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/logging/logger.h"
#include "gap/solver/powerflow_interface.h"

using namespace gap;
using namespace gap::solver;

void print_comparison(std::string const& label, double cpu_val, double gpu_val,
                      bool /*is_complex*/ = false) {
    double abs_diff = std::abs(cpu_val - gpu_val);

    std::cout << std::setw(30) << std::left << label << ": " << std::scientific
              << std::setprecision(6) << "CPU=" << std::setw(13) << cpu_val
              << " GPU=" << std::setw(13) << gpu_val << " diff=" << std::setw(13) << abs_diff;

    if (abs_diff > 1e-6) {
        std::cout << " *** LARGE DIFFERENCE ***";
    }
    std::cout << std::endl;
}

void compare_iteration_states(IterationState const& cpu_state, IterationState const& gpu_state) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Iteration " << cpu_state.iteration << " Comparison" << std::endl;
    std::cout << "========================================" << std::endl;

    print_comparison("Max Mismatch", cpu_state.max_mismatch, gpu_state.max_mismatch);

    int n = std::min(5, static_cast<int>(cpu_state.voltages.size()));
    std::cout << "\nVoltage Comparison (first " << n << " buses):" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::string label = "  V[" + std::to_string(i) + "] mag";
        print_comparison(label, std::abs(cpu_state.voltages[i]), std::abs(gpu_state.voltages[i]));

        label = "  V[" + std::to_string(i) + "] ang";
        print_comparison(label, std::arg(cpu_state.voltages[i]) * 180.0 / M_PI,
                         std::arg(gpu_state.voltages[i]) * 180.0 / M_PI);
    }

    std::cout << "\nCurrent Comparison (first " << n << " buses):" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::string label = "  I[" + std::to_string(i) + "] real";
        print_comparison(label, cpu_state.currents[i].real(), gpu_state.currents[i].real());

        label = "  I[" + std::to_string(i) + "] imag";
        print_comparison(label, cpu_state.currents[i].imag(), gpu_state.currents[i].imag());
    }

    std::cout << "\nPower Injection Comparison (first " << n << " buses):" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::string label = "  S[" + std::to_string(i) + "] P";
        print_comparison(label, cpu_state.power_injections[i].real(),
                         gpu_state.power_injections[i].real());

        label = "  S[" + std::to_string(i) + "] Q";
        print_comparison(label, cpu_state.power_injections[i].imag(),
                         gpu_state.power_injections[i].imag());
    }

    int m = std::min(10, static_cast<int>(cpu_state.mismatches.size()));
    std::cout << "\nMismatch Comparison (first " << m << "):" << std::endl;
    for (int i = 0; i < m; ++i) {
        std::string label = "  Δ[" + std::to_string(i) + "]";
        print_comparison(label, cpu_state.mismatches[i], gpu_state.mismatches[i]);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <network_json>" << std::endl;
        std::cerr << "Example: " << argv[0] << " tests/validation/pgm/4_node_radial_two_pvs.json"
                  << std::endl;
        return 1;
    }

    auto& logger = logging::global_logger;
    logger.setComponent("MismatchDebug");

    try {
        std::cout << "========================================" << std::endl;
        std::cout << "GPU Mismatch Debug" << std::endl;
        std::cout << "========================================" << std::endl;

        // Load network from JSON
        std::cout << "\nLoading network: " << argv[1] << std::endl;
        std::ifstream file(argv[1]);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open file: " << argv[1] << std::endl;
            return 1;
        }

        // Parse JSON and create network
        // NOTE: This needs proper JSON parsing - for now use io module
        auto io_module = core::BackendFactory::create_io_module();
        if (!io_module) {
            std::cerr << "ERROR: Failed to create IO module" << std::endl;
            return 1;
        }

        auto network = io_module->read_network_data(argv[1]);
        std::cout << "Network loaded: " << network.num_buses << " buses" << std::endl;

        // Build admittance matrix
        auto admittance_builder = core::BackendFactory::create_admittance_backend(BackendType::CPU);
        auto y_matrix = admittance_builder->build_admittance_matrix(network);

        // Configure power flow
        PowerFlowConfig config;
        config.tolerance = 1e-6;
        config.max_iterations = 3;  // Only run 3 iterations for debugging
        config.acceleration_factor = 1.0;
        config.use_flat_start = true;
        config.verbose = false;
        config.base_power = 100e6;  // 100 MVA

        // ============================================================
        // Run CPU solver with state capture
        // ============================================================
        std::cout << "\n--- Running CPU Solver ---" << std::endl;
        auto cpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
        auto cpu_lu = core::BackendFactory::create_lu_solver(BackendType::CPU);
        cpu_solver->set_lu_solver(std::shared_ptr<ILUSolver>(cpu_lu.release()));

        // Enable state capture via interface
        cpu_solver->enable_state_capture(true);

        auto cpu_result = cpu_solver->solve_power_flow(network, *y_matrix, config);
        std::cout << "CPU Result: " << (cpu_result.converged ? "CONVERGED" : "DIVERGED") << " in "
                  << cpu_result.iterations << " iterations"
                  << ", max_mismatch=" << std::scientific << cpu_result.final_mismatch << std::endl;

        // ============================================================
        // Run GPU solver with state capture
        // ============================================================
        std::cout << "\n--- Running GPU Solver ---" << std::endl;
        auto gpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        if (!gpu_solver) {
            std::cerr << "ERROR: Failed to load GPU solver" << std::endl;
            return 1;
        }

        auto gpu_lu = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        gpu_solver->set_lu_solver(std::shared_ptr<ILUSolver>(gpu_lu.release()));

        // Enable state capture via interface
        gpu_solver->enable_state_capture(true);

        auto gpu_result = gpu_solver->solve_power_flow(network, *y_matrix, config);
        std::cout << "GPU Result: " << (gpu_result.converged ? "CONVERGED" : "DIVERGED") << " in "
                  << gpu_result.iterations << " iterations"
                  << ", max_mismatch=" << std::scientific << gpu_result.final_mismatch << std::endl;

        // ============================================================
        // Compare captured states
        // ============================================================
        auto const& cpu_states = cpu_solver->get_iteration_states();
        auto const& gpu_states = gpu_solver->get_iteration_states();

        if (cpu_states.empty()) {
            std::cout << "\nWARNING: CPU solver has no states captured" << std::endl;
            std::cout << "Only showing GPU states:" << std::endl;

            for (auto const& state : gpu_states) {
                std::cout << "\n=== GPU Iteration " << state.iteration << " ===" << std::endl;
                std::cout << "Max mismatch: " << std::scientific << state.max_mismatch << std::endl;

                size_t n = (3 < state.voltages.size()) ? 3 : state.voltages.size();
                std::cout << "Voltages (first " << n << "):" << std::endl;
                for (size_t i = 0; i < n; ++i) {
                    std::cout << "  V[" << i << "]: |V|=" << std::abs(state.voltages[i]) << " ∠"
                              << std::arg(state.voltages[i]) * 180.0 / M_PI << "°" << std::endl;
                }

                size_t m = (5 < state.mismatches.size()) ? 5 : state.mismatches.size();
                std::cout << "Mismatches (first " << m << "):" << std::endl;
                for (size_t i = 0; i < m; ++i) {
                    std::cout << "  Δ[" << i << "]: " << state.mismatches[i] << std::endl;
                }
            }
        } else {
            std::cout << "\n========================================" << std::endl;
            std::cout << "CPU vs GPU State Comparison" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "CPU captured " << cpu_states.size() << " states" << std::endl;
            std::cout << "GPU captured " << gpu_states.size() << " states" << std::endl;

            size_t min_states =
                (cpu_states.size() < gpu_states.size()) ? cpu_states.size() : gpu_states.size();
            for (size_t i = 0; i < min_states; ++i) {
                compare_iteration_states(cpu_states[i], gpu_states[i]);
            }
        }

        return 0;

    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
