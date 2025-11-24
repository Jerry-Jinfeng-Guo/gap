/**
 * @file debug_state_compare.cpp
 * @brief Debug utility to compare CPU vs GPU solver iteration states
 *
 * This utility runs both CPU and GPU solvers on a test case with state
 * capture enabled, then compares the iteration states to identify where
 * the GPU solver diverges.
 */

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/io/io_interface.h"
#include "gap/logging/logger.h"
#include "gap/solver/powerflow_interface.h"

using namespace gap;
namespace fs = std::filesystem;

void print_complex(const Complex& c) {
    std::cout << std::fixed << std::setprecision(6) << "(" << c.real() << ", " << c.imag() << ")";
}

void compare_iteration_states(const std::vector<solver::IterationState>& cpu_states,
                              const std::vector<solver::IterationState>& gpu_states) {
    std::cout << "\n========== ITERATION STATE COMPARISON ==========\n";
    std::cout << "CPU iterations: " << cpu_states.size() << "\n";
    std::cout << "GPU iterations: " << gpu_states.size() << "\n\n";

    size_t min_iter = std::min(cpu_states.size(), gpu_states.size());

    for (size_t i = 0; i < min_iter; i++) {
        const auto& cpu = cpu_states[i];
        const auto& gpu = gpu_states[i];

        std::cout << "---------- Iteration " << cpu.iteration << " ----------\n";
        std::cout << "CPU max_mismatch: " << std::scientific << std::setprecision(6)
                  << cpu.max_mismatch << "\n";
        std::cout << "GPU max_mismatch: " << std::scientific << std::setprecision(6)
                  << gpu.max_mismatch << "\n";

        // Check if sizes match
        if (cpu.voltages.size() != gpu.voltages.size()) {
            std::cout << "ERROR: Voltage vector size mismatch!\n";
            std::cout << "  CPU: " << cpu.voltages.size() << " GPU: " << gpu.voltages.size()
                      << "\n";
            continue;
        }

        // Compare voltages
        double max_v_diff = 0.0;
        int max_v_idx = -1;
        for (size_t j = 0; j < cpu.voltages.size(); j++) {
            double diff = std::abs(cpu.voltages[j] - gpu.voltages[j]);
            if (diff > max_v_diff) {
                max_v_diff = diff;
                max_v_idx = j;
            }
        }

        std::cout << "Max voltage difference: " << std::scientific << max_v_diff << " at bus "
                  << max_v_idx << "\n";

        if (max_v_diff > 1e-6) {
            std::cout << "  CPU voltage[" << max_v_idx << "]: ";
            print_complex(cpu.voltages[max_v_idx]);
            std::cout << "\n";
            std::cout << "  GPU voltage[" << max_v_idx << "]: ";
            print_complex(gpu.voltages[max_v_idx]);
            std::cout << "\n";
        }

        // Compare currents
        if (cpu.currents.size() > 0 && gpu.currents.size() > 0) {
            double max_i_diff = 0.0;
            int max_i_idx = -1;
            for (size_t j = 0; j < std::min(cpu.currents.size(), gpu.currents.size()); j++) {
                double diff = std::abs(cpu.currents[j] - gpu.currents[j]);
                if (diff > max_i_diff) {
                    max_i_diff = diff;
                    max_i_idx = j;
                }
            }

            std::cout << "Max current difference: " << std::scientific << max_i_diff << " at bus "
                      << max_i_idx << "\n";

            if (max_i_diff > 1e-6) {
                std::cout << "  CPU current[" << max_i_idx << "]: ";
                print_complex(cpu.currents[max_i_idx]);
                std::cout << "\n";
                std::cout << "  GPU current[" << max_i_idx << "]: ";
                print_complex(gpu.currents[max_i_idx]);
                std::cout << "\n";
            }
        }

        // Compare mismatches
        double max_m_diff = 0.0;
        int max_m_idx = -1;
        for (size_t j = 0; j < std::min(cpu.mismatches.size(), gpu.mismatches.size()); j++) {
            double diff = std::abs(cpu.mismatches[j] - gpu.mismatches[j]);
            if (diff > max_m_diff) {
                max_m_diff = diff;
                max_m_idx = j;
            }
        }

        std::cout << "Max mismatch difference: " << std::scientific << max_m_diff << " at index "
                  << max_m_idx << "\n";

        if (max_m_diff > 1e-6) {
            std::cout << "  CPU mismatch[" << max_m_idx << "]: " << cpu.mismatches[max_m_idx]
                      << "\n";
            std::cout << "  GPU mismatch[" << max_m_idx << "]: " << gpu.mismatches[max_m_idx]
                      << "\n";
        }

        std::cout << "\n";

        // If we see significant divergence, stop here
        if (max_v_diff > 1e-3 || max_m_diff > 1e-3) {
            std::cout << "SIGNIFICANT DIVERGENCE DETECTED AT ITERATION " << i << "!\n";
            std::cout << "Stopping comparison here.\n";
            break;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_data_directory>\n";
        std::cerr << "Example: " << argv[0]
                  << " tests/pgm_validation/test_data/radial_3feeder_8nodepf\n";
        return 1;
    }

    fs::path test_dir(argv[1]);
    if (!fs::exists(test_dir)) {
        std::cerr << "Error: Test directory does not exist: " << test_dir << "\n";
        return 1;
    }

    try {
        // Load test case
        std::cout << "Loading test case from: " << test_dir << "\n";
        fs::path input_path = test_dir / "input.json";

        auto io = core::BackendFactory::create_io_module();
        auto network = io->read_network_data(input_path.string());

        std::cout << "Network loaded: " << network.num_buses << " buses, " << network.num_branches
                  << " branches\n";

        // Setup configuration
        solver::PowerFlowConfig config;
        config.tolerance = 1e-8;
        config.max_iterations = 50;
        config.verbose = false;

        // Run CPU solver with state capture
        std::cout << "\n========== Running CPU Solver ==========\n";
        auto cpu_admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);
        auto cpu_matrix = cpu_admittance->build_admittance_matrix(network);

        auto cpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
        auto cpu_lu = core::BackendFactory::create_lu_solver(BackendType::CPU);
        cpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu.release()));
        cpu_solver->enable_state_capture(true);

        auto cpu_result = cpu_solver->solve_power_flow(network, *cpu_matrix, config);
        const auto& cpu_states = cpu_solver->get_iteration_states();

        std::cout << "CPU Result: " << (cpu_result.converged ? "CONVERGED" : "DIVERGED") << "\n";
        std::cout << "Iterations: " << cpu_result.iterations << "\n";
        std::cout << "Final mismatch: " << std::scientific << cpu_result.final_mismatch << "\n";
        std::cout << "Captured " << cpu_states.size() << " iteration states\n";

        // Run GPU solver with state capture
        std::cout << "\n========== Running GPU Solver ==========\n";
        auto gpu_admittance =
            core::BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
        auto gpu_matrix = gpu_admittance->build_admittance_matrix(network);

        auto gpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        auto gpu_lu = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        gpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu.release()));
        gpu_solver->enable_state_capture(true);

        auto gpu_result = gpu_solver->solve_power_flow(network, *gpu_matrix, config);
        const auto& gpu_states = gpu_solver->get_iteration_states();

        std::cout << "GPU Result: " << (gpu_result.converged ? "CONVERGED" : "DIVERGED") << "\n";
        std::cout << "Iterations: " << gpu_result.iterations << "\n";
        std::cout << "Final mismatch: " << std::scientific << gpu_result.final_mismatch << "\n";
        std::cout << "Captured " << gpu_states.size() << " iteration states\n";

        // Compare iteration states
        compare_iteration_states(cpu_states, gpu_states);

        // Detailed final voltage comparison
        if (cpu_result.converged && gpu_result.converged) {
            std::cout << "\n========== DETAILED FINAL VOLTAGE COMPARISON ==========\n";
            std::cout << "Both solvers converged. Comparing all bus voltages...\n\n";

            const auto& cpu_final = cpu_states.back();
            const auto& gpu_final = gpu_states.back();

            std::cout << std::setw(6) << "Bus" << std::setw(18) << "CPU Mag (pu)" << std::setw(18)
                      << "GPU Mag (pu)" << std::setw(16) << "Mag Diff" << std::setw(18)
                      << "CPU Ang (deg)" << std::setw(18) << "GPU Ang (deg)" << std::setw(16)
                      << "Ang Diff (deg)" << "\n";
            std::cout << std::string(126, '-') << "\n";

            double max_mag_diff = 0.0, max_ang_diff = 0.0;
            int max_mag_bus = -1, max_ang_bus = -1;
            bool all_match = true;
            const double mag_tol = 1e-6;  // 0.0001%
            const double ang_tol = 1e-4;  // 0.0057 degrees

            for (size_t i = 0; i < cpu_final.voltages.size(); i++) {
                double cpu_mag = std::abs(cpu_final.voltages[i]);
                double gpu_mag = std::abs(gpu_final.voltages[i]);
                double cpu_ang = std::arg(cpu_final.voltages[i]) * 180.0 / M_PI;
                double gpu_ang = std::arg(gpu_final.voltages[i]) * 180.0 / M_PI;

                double mag_diff = std::abs(cpu_mag - gpu_mag);
                double ang_diff = std::abs(cpu_ang - gpu_ang);

                if (mag_diff > max_mag_diff) {
                    max_mag_diff = mag_diff;
                    max_mag_bus = i;
                }
                if (ang_diff > max_ang_diff) {
                    max_ang_diff = ang_diff;
                    max_ang_bus = i;
                }

                if (mag_diff > mag_tol || ang_diff > ang_tol) {
                    all_match = false;
                }

                std::cout << std::setw(6) << i << std::fixed << std::setprecision(10)
                          << std::setw(18) << cpu_mag << std::setw(18) << gpu_mag << std::scientific
                          << std::setprecision(4) << std::setw(16) << mag_diff << std::fixed
                          << std::setprecision(8) << std::setw(18) << cpu_ang << std::setw(18)
                          << gpu_ang << std::scientific << std::setprecision(4) << std::setw(16)
                          << ang_diff << "\n";
            }

            std::cout << std::string(126, '-') << "\n";
            std::cout << "Maximum magnitude difference: " << std::scientific << std::setprecision(4)
                      << max_mag_diff << " at bus " << max_mag_bus << "\n";
            std::cout << "Maximum angle difference:     " << std::scientific << std::setprecision(4)
                      << max_ang_diff << " deg (" << max_ang_diff * M_PI / 180.0 << " rad) at bus "
                      << max_ang_bus << "\n";

            if (all_match) {
                std::cout << "\n✓ ALL VOLTAGES MATCH within tolerance (mag: " << mag_tol
                          << ", ang: " << ang_tol << " rad)\n";
                return 0;
            } else {
                std::cout << "\n✗ SOME VOLTAGES DIFFER beyond tolerance\n";
                return 1;
            }
        } else if (!cpu_result.converged && !gpu_result.converged) {
            std::cout << "\n⚠ Both solvers failed to converge\n";
            return 1;
        } else {
            std::cout << "\n✗ CONVERGENCE MISMATCH: CPU "
                      << (cpu_result.converged ? "converged" : "diverged") << ", GPU "
                      << (gpu_result.converged ? "converged" : "diverged") << "\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
