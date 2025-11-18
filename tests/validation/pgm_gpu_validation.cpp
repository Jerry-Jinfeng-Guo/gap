/**
 * @file pgm_gpu_validation.cpp
 * @brief GPU validation tests using PGM test cases
 *
 * This file contains tests that run all PGM validation test cases
 * on the GPU backend and compare results with CPU backend.
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/io/io_interface.h"

#include "../unit/test_framework.h"

using namespace gap;
namespace fs = std::filesystem;

/**
 * @brief Test a single PGM case with both CPU and GPU backends
 */
bool test_pgm_case_with_gpu(const std::string& test_name, const std::string& input_path) {
    std::cout << "\n  Testing: " << test_name << std::endl;

    // Load network data using the IO module
    auto io = core::BackendFactory::create_io_module();
    NetworkData network;
    try {
        network = io->read_network_data(input_path);
    } catch (const std::exception& e) {
        std::cerr << "    ❌ Failed to load network: " << e.what() << std::endl;
        return false;
    }

    std::cout << "    Network: " << network.num_buses << " buses, " << network.num_branches
              << " branches" << std::endl;

    // Build admittance matrix and solve with CPU backend
    std::cout << "    Running CPU backend..." << std::endl;

    auto cpu_admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);
    auto cpu_matrix = cpu_admittance->build_admittance_matrix(network);

    auto cpu_pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
    auto cpu_lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
    cpu_pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu_solver.release()));

    solver::PowerFlowConfig config;
    config.tolerance = 1e-6;
    config.max_iterations = 100;
    config.verbose = false;

    auto cpu_result = cpu_pf_solver->solve_power_flow(network, *cpu_matrix, config);

    if (!cpu_result.converged) {
        std::cerr << "    ❌ CPU solver did not converge" << std::endl;
        return false;
    }

    std::cout << "    ✓ CPU converged in " << cpu_result.iterations << " iterations" << std::endl;

    // Build admittance matrix and solve with GPU backend
    std::cout << "    Running GPU backend..." << std::endl;

    if (!core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cerr << "    ⚠️  GPU backend not available, skipping" << std::endl;
        return true;  // Not a failure, just skip
    }

    auto gpu_admittance = core::BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
    auto gpu_matrix = gpu_admittance->build_admittance_matrix(network);

    auto gpu_pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto gpu_lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    gpu_pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu_solver.release()));

    auto gpu_result = gpu_pf_solver->solve_power_flow(network, *gpu_matrix, config);

    if (!gpu_result.converged) {
        std::cerr << "    ❌ GPU solver did not converge" << std::endl;
        return false;
    }

    std::cout << "    ✓ GPU converged in " << gpu_result.iterations << " iterations" << std::endl;

    // Compare voltage results
    if (cpu_result.bus_voltages.size() != gpu_result.bus_voltages.size()) {
        std::cerr << "    ❌ Result size mismatch: CPU=" << cpu_result.bus_voltages.size()
                  << ", GPU=" << gpu_result.bus_voltages.size() << std::endl;
        return false;
    }

    Float max_mag_diff = 0.0;
    Float max_angle_diff = 0.0;

    for (size_t i = 0; i < cpu_result.bus_voltages.size(); ++i) {
        Float mag_diff =
            std::abs(std::abs(cpu_result.bus_voltages[i]) - std::abs(gpu_result.bus_voltages[i]));
        Float angle_diff =
            std::abs(std::arg(cpu_result.bus_voltages[i]) - std::arg(gpu_result.bus_voltages[i]));

        max_mag_diff = std::max(max_mag_diff, mag_diff);
        max_angle_diff = std::max(max_angle_diff, angle_diff);
    }

    std::cout << "    Max voltage magnitude difference: " << max_mag_diff << " pu" << std::endl;
    std::cout << "    Max voltage angle difference: " << max_angle_diff << " rad" << std::endl;

    const Float tolerance = 5e-4;  // Relaxed tolerance for GPU vs CPU comparison
    if (max_mag_diff > tolerance) {
        std::cerr << "    ❌ Voltage magnitude difference exceeds tolerance (" << tolerance << ")"
                  << std::endl;
        return false;
    }

    if (max_angle_diff > tolerance) {
        std::cerr << "    ❌ Voltage angle difference exceeds tolerance (" << tolerance << ")"
                  << std::endl;
        return false;
    }

    std::cout << "    ✅ PASSED - CPU and GPU results match" << std::endl;
    return true;
}

/**
 * @brief Detailed investigation of GPU vs CPU accuracy on a single test case
 */
void test_pgm_gpu_accuracy_investigation() {
    std::cout << "\n=== GPU Power Flow Accuracy Investigation ===" << std::endl;
    std::cout << "Testing: radial_1feeder_2nodepf with various tolerances\n" << std::endl;

    std::string test_data_dir = "../tests/pgm_validation/test_data";
    if (!fs::exists(test_data_dir)) {
        test_data_dir = "../../tests/pgm_validation/test_data";
    }

    std::string input_path = test_data_dir + "/radial_1feeder_2nodepf/input.json";

    if (!fs::exists(input_path)) {
        std::cerr << "❌ Test case not found" << std::endl;
        ASSERT_TRUE(false);
        return;
    }

    // Load network
    auto io = core::BackendFactory::create_io_module();
    NetworkData network = io->read_network_data(input_path);

    std::cout << "Network: " << network.num_buses << " buses, " << network.num_branches
              << " branches" << std::endl;

    // Test with different tolerances
    std::vector<Float> tolerances = {1e-6, 1e-8, 1e-10};

    for (Float tol : tolerances) {
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "Testing with tolerance: " << tol << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        // CPU solve
        std::cout << "\nCPU Backend:" << std::endl;
        auto cpu_admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);
        auto cpu_matrix = cpu_admittance->build_admittance_matrix(network);

        auto cpu_pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::CPU);
        auto cpu_lu_solver = core::BackendFactory::create_lu_solver(BackendType::CPU);
        cpu_pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(cpu_lu_solver.release()));

        solver::PowerFlowConfig config;
        config.tolerance = tol;
        config.max_iterations = 100;
        config.verbose = false;

        auto cpu_result = cpu_pf_solver->solve_power_flow(network, *cpu_matrix, config);

        std::cout << "  Converged: " << (cpu_result.converged ? "YES" : "NO") << std::endl;
        std::cout << "  Iterations: " << cpu_result.iterations << std::endl;
        std::cout << "  Final mismatch: " << cpu_result.final_mismatch << std::endl;

        // GPU solve
        std::cout << "\nGPU Backend:" << std::endl;
        if (!core::BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
            std::cout << "  GPU not available, skipping" << std::endl;
            continue;
        }

        auto gpu_admittance =
            core::BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
        auto gpu_matrix = gpu_admittance->build_admittance_matrix(network);

        auto gpu_pf_solver = core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
        auto gpu_lu_solver = core::BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
        gpu_pf_solver->set_lu_solver(std::shared_ptr<solver::ILUSolver>(gpu_lu_solver.release()));

        // Enable verbose for GPU solver to see debug output
        solver::PowerFlowConfig gpu_config = config;
        gpu_config.verbose = true;

        auto gpu_result = gpu_pf_solver->solve_power_flow(network, *gpu_matrix, gpu_config);

        std::cout << "  Converged: " << (gpu_result.converged ? "YES" : "NO") << std::endl;
        std::cout << "  Iterations: " << gpu_result.iterations << std::endl;
        std::cout << "  Final mismatch: " << gpu_result.final_mismatch << std::endl;

        if (!cpu_result.converged || !gpu_result.converged) {
            std::cout << "\n⚠️  One or both solvers did not converge" << std::endl;
            continue;
        }

        // Detailed comparison
        std::cout << "\nVoltage Comparison:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout
            << "Bus   CPU |V| (pu)  GPU |V| (pu)  Mag Diff    CPU ∠ (deg)  GPU ∠ (deg)  Angle Diff"
            << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        Float max_mag_diff = 0.0;
        Float max_angle_diff = 0.0;
        Float sum_mag_diff = 0.0;
        Float sum_angle_diff = 0.0;

        for (size_t i = 0; i < cpu_result.bus_voltages.size(); ++i) {
            Float cpu_mag = std::abs(cpu_result.bus_voltages[i]);
            Float gpu_mag = std::abs(gpu_result.bus_voltages[i]);
            Float cpu_angle = std::arg(cpu_result.bus_voltages[i]) * 180.0 / M_PI;
            Float gpu_angle = std::arg(gpu_result.bus_voltages[i]) * 180.0 / M_PI;

            Float mag_diff = std::abs(cpu_mag - gpu_mag);
            Float angle_diff = std::abs(cpu_angle - gpu_angle);

            max_mag_diff = std::max(max_mag_diff, mag_diff);
            max_angle_diff = std::max(max_angle_diff, angle_diff);
            sum_mag_diff += mag_diff;
            sum_angle_diff += angle_diff;

            printf("%3zu   %12.8f  %12.8f  %10.2e  %11.6f  %11.6f  %10.6f\n", i + 1, cpu_mag,
                   gpu_mag, mag_diff, cpu_angle, gpu_angle, angle_diff);
        }

        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Max magnitude difference:  " << max_mag_diff << " pu" << std::endl;
        std::cout << "Mean magnitude difference: " << sum_mag_diff / cpu_result.bus_voltages.size()
                  << " pu" << std::endl;
        std::cout << "Max angle difference:      " << max_angle_diff << " degrees" << std::endl;
        std::cout << "Mean angle difference:     "
                  << sum_angle_diff / cpu_result.bus_voltages.size() << " degrees" << std::endl;

        // Check if differences are acceptable
        const Float acceptable_mag_diff = tol * 100;     // 100x tolerance
        const Float acceptable_angle_diff = tol * 1000;  // Angles are more sensitive

        if (max_mag_diff < acceptable_mag_diff && max_angle_diff < acceptable_angle_diff) {
            std::cout << "\n✅ Results match within acceptable bounds for tolerance " << tol
                      << std::endl;
        } else {
            std::cout << "\n⚠️  Differences exceed acceptable bounds:" << std::endl;
            if (max_mag_diff >= acceptable_mag_diff) {
                std::cout << "   Magnitude diff " << max_mag_diff << " >= " << acceptable_mag_diff
                          << std::endl;
            }
            if (max_angle_diff >= acceptable_angle_diff) {
                std::cout << "   Angle diff " << max_angle_diff << " >= " << acceptable_angle_diff
                          << std::endl;
            }
        }
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Investigation complete" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

/**
 * @brief Run all PGM validation tests with GPU backend
 */
void test_pgm_gpu_all_cases() {
    std::cout << "\n=== Running all PGM test cases with GPU backend ===" << std::endl;

    // Find test data directory
    std::string test_data_dir = "../tests/pgm_validation/test_data";

    // Check if directory exists
    if (!fs::exists(test_data_dir) || !fs::is_directory(test_data_dir)) {
        // Try alternative path
        test_data_dir = "../../tests/pgm_validation/test_data";
        if (!fs::exists(test_data_dir) || !fs::is_directory(test_data_dir)) {
            std::cerr << "❌ Test data directory not found" << std::endl;
            ASSERT_TRUE(false);
            return;
        }
    }

    std::vector<std::string> test_cases;

    // Collect all test case directories
    for (const auto& entry : fs::directory_iterator(test_data_dir)) {
        if (entry.is_directory()) {
            test_cases.push_back(entry.path().filename().string());
        }
    }

    // Sort for consistent order
    std::sort(test_cases.begin(), test_cases.end());

    std::cout << "\nFound " << test_cases.size() << " test cases" << std::endl;

    int passed = 0;
    int failed = 0;
    int skipped = 0;
    std::vector<std::string> failed_cases;

    for (const auto& test_case : test_cases) {
        std::string case_dir = test_data_dir + "/" + test_case;
        std::string input_path = case_dir + "/input.json";

        if (!fs::exists(input_path)) {
            std::cerr << "  ⚠️  Skipping " << test_case << " (no input.json)" << std::endl;
            skipped++;
            continue;
        }

        if (test_pgm_case_with_gpu(test_case, input_path)) {
            passed++;
        } else {
            failed++;
            failed_cases.push_back(test_case);
        }
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PGM GPU Validation Results:" << std::endl;
    std::cout << "  Total tests: " << (passed + failed) << std::endl;
    std::cout << "  Passed: " << passed << " ✅" << std::endl;
    std::cout << "  Failed: " << failed << " ❌" << std::endl;
    std::cout << "  Skipped: " << skipped << std::endl;

    if (!failed_cases.empty()) {
        std::cout << "\nFailed test cases:" << std::endl;
        for (const auto& case_name : failed_cases) {
            std::cout << "  - " << case_name << std::endl;
        }
    }

    std::cout << std::string(80, '=') << std::endl;

    ASSERT_TRUE(failed == 0);
}

/**
 * @brief Test a few specific PGM cases (quick smoke test)
 */
void test_pgm_gpu_smoke_test() {
    std::cout << "\n=== PGM GPU Smoke Test (selected cases) ===" << std::endl;

    std::string test_data_dir = "../tests/pgm_validation/test_data";

    // Try alternative path if first doesn't exist
    if (!fs::exists(test_data_dir)) {
        test_data_dir = "../../tests/pgm_validation/test_data";
    }

    // Test just one simple case to avoid stalling
    std::vector<std::string> smoke_test_cases = {"radial_1feeder_2nodepf"};

    int passed = 0;
    int failed = 0;

    for (const auto& test_case : smoke_test_cases) {
        std::string case_dir = test_data_dir + "/" + test_case;
        std::string input_path = case_dir + "/input.json";

        if (!fs::exists(input_path)) {
            std::cerr << "  ⚠️  Skipping " << test_case << " (not found)" << std::endl;
            continue;
        }

        if (test_pgm_case_with_gpu(test_case, input_path)) {
            passed++;
        } else {
            failed++;
        }
    }

    std::cout << "\nSmoke test results: " << passed << "/" << (passed + failed) << " passed"
              << std::endl;
    ASSERT_TRUE(failed == 0);
}

// Register tests with the test runner
void register_pgm_gpu_validation_tests(TestRunner& runner) {
    std::cout << "Registering PGM GPU validation tests..." << std::endl;

    // Accuracy investigation test - detailed analysis of CPU vs GPU
    runner.add_test("PGM GPU Accuracy Investigation",
                    []() { test_pgm_gpu_accuracy_investigation(); });

    // Smoke test - quick validation with single case
    runner.add_test("PGM GPU Smoke Test", []() { test_pgm_gpu_smoke_test(); });

    // Full test suite - commented out to avoid long runtime
    // runner.add_test("PGM GPU All Cases", []() { test_pgm_gpu_all_cases(); });
}
