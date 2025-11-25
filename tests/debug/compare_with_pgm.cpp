/**
 * @file compare_with_pgm.cpp
 * @brief Compare CPU and GPU solver results against PGM reference outputs
 */

#include <cmath>
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

struct PGMNodeResult {
    int id;
    double u_pu;
    double u_angle;  // radians
};

std::vector<PGMNodeResult> load_pgm_results(std::string const& output_path) {
    std::vector<PGMNodeResult> results;
    std::ifstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open PGM output file: " + output_path);
    }

    std::string line;
    bool in_node_array = false;
    bool in_node_object = false;
    PGMNodeResult current_node{};

    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line.find("\"node\"") != std::string::npos) {
            in_node_array = true;
            continue;
        }

        if (!in_node_array) continue;

        if (line.find("{") != std::string::npos && line.find("}") == std::string::npos) {
            in_node_object = true;
            current_node = PGMNodeResult{};
            continue;
        }

        if (in_node_object) {
            if (line.find("\"id\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value = value.substr(0, value.find_first_of(",}"));
                current_node.id = std::stoi(value);
            } else if (line.find("\"u_pu\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value = value.substr(0, value.find_first_of(",}"));
                current_node.u_pu = std::stod(value);
            } else if (line.find("\"u_angle\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value = value.substr(0, value.find_first_of(",}"));
                current_node.u_angle = std::stod(value);
            }

            if (line.find("}") != std::string::npos) {
                in_node_object = false;
                results.push_back(current_node);
            }
        }

        if (line.find("]") != std::string::npos && in_node_array) {
            break;
        }
    }

    return results;
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
        fs::path input_path = test_dir / "input.json";
        fs::path output_path = test_dir / "output.json";

        if (!fs::exists(output_path)) {
            std::cerr << "Warning: No PGM reference output found at: " << output_path << "\n";
            std::cerr << "Skipping PGM comparison.\n";
            return 0;
        }

        std::cout << "Loading test case: " << test_dir.filename() << "\n";

        auto io = core::BackendFactory::create_io_module();
        auto network = io->read_network_data(input_path.string());

        std::cout << "Network: " << network.num_buses << " buses, " << network.num_branches
                  << " branches\n";

        // Load PGM reference results
        auto pgm_results = load_pgm_results(output_path.string());
        std::cout << "PGM reference: " << pgm_results.size() << " bus results\n\n";

        // Solver configuration
        solver::PowerFlowConfig config;
        config.tolerance = 1e-8;
        config.max_iterations = 50;
        config.verbose = false;

        // === CPU Solver ===
        std::cout << "========== CPU Solver ==========\n";
        auto cpu_admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);
        auto cpu_matrix = cpu_admittance->build_admittance_matrix(network);

        auto cpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
        auto cpu_lu = core::BackendFactory::create_lu_solver(BackendType::CPU);
        cpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu.release()));

        auto cpu_result = cpu_solver->solve_power_flow(network, *cpu_matrix, config);
        std::cout << "Status: " << (cpu_result.converged ? "CONVERGED" : "DIVERGED") << "\n";
        std::cout << "Iterations: " << cpu_result.iterations << "\n\n";

        // === GPU Solver ===
        std::cout << "========== GPU Solver ==========\n";
        auto gpu_admittance =
            core::BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
        auto gpu_matrix = gpu_admittance->build_admittance_matrix(network);

        auto gpu_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        auto gpu_lu = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        gpu_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu.release()));

        auto gpu_result = gpu_solver->solve_power_flow(network, *cpu_matrix, config);
        std::cout << "Status: " << (gpu_result.converged ? "CONVERGED" : "DIVERGED") << "\n";
        std::cout << "Iterations: " << gpu_result.iterations << "\n\n";

        // === Comparison with PGM ===
        std::cout << "========== Comparison with PGM Reference ==========\n\n";
        std::cout << std::setw(6) << "Bus" << std::setw(14) << "PGM Mag" << std::setw(14)
                  << "CPU Mag" << std::setw(14) << "GPU Mag" << std::setw(12) << "CPU Err"
                  << std::setw(12) << "GPU Err" << std::setw(14) << "PGM Ang(deg)" << std::setw(14)
                  << "CPU Ang(deg)" << std::setw(14) << "GPU Ang(deg)" << std::setw(12) << "CPU Err"
                  << std::setw(12) << "GPU Err" << "\n";
        std::cout << std::string(154, '-') << "\n";

        double cpu_max_mag_err = 0.0, gpu_max_mag_err = 0.0;
        double cpu_max_ang_err = 0.0, gpu_max_ang_err = 0.0;
        bool cpu_matches_pgm = true, gpu_matches_pgm = true;

        const double MAG_TOLERANCE = 5e-6;      // 5 micro-pu
        const double ANGLE_TOLERANCE = 0.0001;  // ~0.0057 degrees

        for (auto const& pgm : pgm_results) {
            if (pgm.id >= static_cast<int>(cpu_result.bus_voltages.size())) continue;

            double cpu_mag = std::abs(cpu_result.bus_voltages[pgm.id]);
            double cpu_ang = std::arg(cpu_result.bus_voltages[pgm.id]);
            double cpu_ang_deg = cpu_ang * 180.0 / M_PI;

            double gpu_mag = std::abs(gpu_result.bus_voltages[pgm.id]);
            double gpu_ang = std::arg(gpu_result.bus_voltages[pgm.id]);
            double gpu_ang_deg = gpu_ang * 180.0 / M_PI;

            double pgm_ang_deg = pgm.u_angle * 180.0 / M_PI;

            double cpu_mag_err = std::abs(cpu_mag - pgm.u_pu);
            double gpu_mag_err = std::abs(gpu_mag - pgm.u_pu);
            double cpu_ang_err = std::abs(cpu_ang - pgm.u_angle);
            double gpu_ang_err = std::abs(gpu_ang - pgm.u_angle);

            cpu_max_mag_err = std::max(cpu_max_mag_err, cpu_mag_err);
            gpu_max_mag_err = std::max(gpu_max_mag_err, gpu_mag_err);
            cpu_max_ang_err = std::max(cpu_max_ang_err, cpu_ang_err);
            gpu_max_ang_err = std::max(gpu_max_ang_err, gpu_ang_err);

            if (cpu_mag_err > MAG_TOLERANCE || cpu_ang_err > ANGLE_TOLERANCE) {
                cpu_matches_pgm = false;
            }
            if (gpu_mag_err > MAG_TOLERANCE || gpu_ang_err > ANGLE_TOLERANCE) {
                gpu_matches_pgm = false;
            }

            std::cout << std::setw(6) << pgm.id << std::fixed << std::setprecision(10)
                      << std::setw(14) << pgm.u_pu << std::setw(14) << cpu_mag << std::setw(14)
                      << gpu_mag << std::scientific << std::setprecision(2) << std::setw(12)
                      << cpu_mag_err << std::setw(12) << gpu_mag_err << std::fixed
                      << std::setprecision(8) << std::setw(14) << pgm_ang_deg << std::setw(14)
                      << cpu_ang_deg << std::setw(14) << gpu_ang_deg << std::scientific
                      << std::setprecision(2) << std::setw(12) << cpu_ang_err << std::setw(12)
                      << gpu_ang_err << "\n";
        }

        std::cout << std::string(154, '-') << "\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "CPU: Max magnitude error = " << std::scientific << cpu_max_mag_err
                  << " pu, Max angle error = " << cpu_max_ang_err << " rad ("
                  << cpu_max_ang_err * 180.0 / M_PI << " deg)\n";
        std::cout << "GPU: Max magnitude error = " << std::scientific << gpu_max_mag_err
                  << " pu, Max angle error = " << gpu_max_ang_err << " rad ("
                  << gpu_max_ang_err * 180.0 / M_PI << " deg)\n\n";

        std::cout << "Tolerances: Magnitude = " << MAG_TOLERANCE
                  << " pu, Angle = " << ANGLE_TOLERANCE << " rad ("
                  << ANGLE_TOLERANCE * 180.0 / M_PI << " deg)\n\n";

        if (cpu_matches_pgm && gpu_matches_pgm) {
            std::cout << "✓ Both CPU and GPU results match PGM reference within tolerance\n";
            return 0;
        } else {
            if (!cpu_matches_pgm) {
                std::cout << "⚠ CPU results differ from PGM reference\n";
            }
            if (!gpu_matches_pgm) {
                std::cout << "⚠ GPU results differ from PGM reference\n";
            }
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
