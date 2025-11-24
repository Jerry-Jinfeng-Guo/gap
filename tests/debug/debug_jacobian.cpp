/**
 * @file debug_jacobian.cpp
 * @brief Debug utility to examine Jacobian and corrections in detail
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

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_data_directory>\n";
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

        // Setup configuration - do 2 iterations to see the effect
        solver::PowerFlowConfig config;
        config.tolerance = 1e-8;
        config.max_iterations = 2;  // Do 2 iterations to see voltage change
        config.verbose = true;

        // Run GPU solver
        std::cout << "\n========== Running GPU Solver (1 iteration) ==========\n";
        auto gpu_admittance =
            core::BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
        auto gpu_matrix = gpu_admittance->build_admittance_matrix(network);

        auto gpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        auto gpu_lu = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        gpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu.release()));
        gpu_solver->enable_state_capture(true);

        auto gpu_result = gpu_solver->solve_power_flow(network, *gpu_matrix, config);
        const auto& gpu_states = gpu_solver->get_iteration_states();

        std::cout << "\nGPU Iteration 0 state:\n";
        if (gpu_states.size() > 0) {
            std::cout << "  Max mismatch: " << gpu_states[0].max_mismatch << "\n";
            std::cout << "  First 5 voltages:\n";
            for (size_t i = 0; i < std::min(size_t(5), gpu_states[0].voltages.size()); i++) {
                std::cout << "    V[" << i << "] = " << gpu_states[0].voltages[i] << "\n";
            }
        }

        if (gpu_states.size() > 1) {
            std::cout << "\nGPU Iteration 1 state (after first correction):\n";
            std::cout << "  Max mismatch: " << gpu_states[1].max_mismatch << "\n";
            std::cout << "  First 5 voltages:\n";
            for (size_t i = 0; i < std::min(size_t(5), gpu_states[1].voltages.size()); i++) {
                std::cout << "    V[" << i << "] = " << gpu_states[1].voltages[i] << "\n";
            }
        }

        // Run CPU solver for comparison
        std::cout << "\n========== Running CPU Solver (1 iteration) ==========\n";
        auto cpu_admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);
        auto cpu_matrix = cpu_admittance->build_admittance_matrix(network);

        auto cpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
        auto cpu_lu = core::BackendFactory::create_lu_solver(BackendType::CPU);
        cpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu.release()));
        cpu_solver->enable_state_capture(true);

        auto cpu_result = cpu_solver->solve_power_flow(network, *cpu_matrix, config);
        const auto& cpu_states = cpu_solver->get_iteration_states();

        std::cout << "\nCPU Iteration 0 state:\n";
        if (cpu_states.size() > 0) {
            std::cout << "  Max mismatch: " << cpu_states[0].max_mismatch << "\n";
            std::cout << "  First 5 voltages:\n";
            for (size_t i = 0; i < std::min(size_t(5), cpu_states[0].voltages.size()); i++) {
                std::cout << "    V[" << i << "] = " << cpu_states[0].voltages[i] << "\n";
            }
        }

        if (cpu_states.size() > 1) {
            std::cout << "\nCPU Iteration 1 state (after first correction):\n";
            std::cout << "  Max mismatch: " << cpu_states[1].max_mismatch << "\n";
            std::cout << "  First 5 voltages:\n";
            for (size_t i = 0; i < std::min(size_t(5), cpu_states[1].voltages.size()); i++) {
                std::cout << "    V[" << i << "] = " << cpu_states[1].voltages[i] << "\n";
            }
        }

        // Compare voltage changes
        if (cpu_states.size() > 1 && gpu_states.size() > 1) {
            std::cout << "\n========== Voltage Change Comparison ==========\n";
            for (size_t i = 0; i < std::min(size_t(10), cpu_states[0].voltages.size()); i++) {
                auto cpu_delta = cpu_states[1].voltages[i] - cpu_states[0].voltages[i];
                auto gpu_delta = gpu_states[1].voltages[i] - gpu_states[0].voltages[i];
                auto diff = cpu_delta - gpu_delta;

                if (std::abs(diff) > 1e-6) {
                    std::cout << "  Bus " << i << ":\n";
                    std::cout << "    CPU ΔV = " << cpu_delta << "\n";
                    std::cout << "    GPU ΔV = " << gpu_delta << "\n";
                    std::cout << "    Diff   = " << diff << "\n";
                }
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
