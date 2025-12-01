/**
 * @file test_main_cli.cpp
 * @brief Tests for main.cpp command-line interface
 */

#include <cstring>
#include <vector>

#include "gap/core/types.h"
#include "gap/solver/powerflow_interface.h"

#include "test_framework.h"

using namespace gap;

// Forward declare functions from main.cpp that we want to test
// Note: In production, these would be in a separate testable module

/**
 * @brief Mock configuration structure matching main.cpp
 */
struct AppConfig {
    std::string input_file;
    std::string output_file;
    std::string update_file;
    BackendType backend_type = BackendType::CPU;
    solver::PowerFlowConfig pf_config;
    solver::BatchPowerFlowConfig batch_config;
    bool verbose = false;
    bool benchmark = false;
    bool batch_mode = false;
};

/**
 * @brief Mock argument parser (mirrors main.cpp logic)
 */
AppConfig parse_test_arguments(int argc, char* argv[]) {
    AppConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                config.input_file = argv[++i];
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                config.output_file = argv[++i];
            }
        } else if (arg == "-u" || arg == "--update") {
            if (i + 1 < argc) {
                config.update_file = argv[++i];
            }
        } else if (arg == "-b" || arg == "--backend") {
            if (i + 1 < argc) {
                std::string backend_str = argv[++i];
                if (backend_str == "cpu") {
                    config.backend_type = BackendType::CPU;
                } else if (backend_str == "gpu") {
                    config.backend_type = BackendType::GPU_CUDA;
                }
            }
        } else if (arg == "-t" || arg == "--tolerance") {
            if (i + 1 < argc) {
                config.pf_config.tolerance = std::stod(argv[++i]);
            }
        } else if (arg == "-m" || arg == "--max-iter") {
            if (i + 1 < argc) {
                config.pf_config.max_iterations = std::stoi(argv[++i]);
            }
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
            config.pf_config.verbose = true;
        } else if (arg == "--benchmark") {
            config.benchmark = true;
        } else if (arg == "--flat-start") {
            config.pf_config.use_flat_start = true;
        } else if (arg == "--batch") {
            config.batch_mode = true;
        } else if (arg == "--reuse-ybus") {
            config.batch_config.reuse_y_bus_factorization = true;
        } else if (arg == "--warm-start") {
            config.batch_config.warm_start = true;
        }
    }

    // Initialize batch config from base config
    config.batch_config.base_config = config.pf_config;
    config.batch_config.verbose_summary = config.verbose;

    return config;
}

/**
 * @brief Helper to create argv from string arguments
 */
class ArgvHelper {
  private:
    std::vector<char*> argv_;
    std::vector<std::string> args_;

  public:
    ArgvHelper(std::vector<std::string> const& args) : args_(args) {
        argv_.reserve(args_.size());
        for (auto& arg : args_) {
            argv_.push_back(const_cast<char*>(arg.c_str()));
        }
    }

    char** argv() { return argv_.data(); }
    int argc() { return static_cast<int>(argv_.size()); }
};

// ============================================================================
// Test Cases
// ============================================================================

void test_single_power_flow_basic_args() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_EQ(config.input_file, "input.json");
    ASSERT_EQ(config.output_file, "output.json");
    ASSERT_FALSE(config.batch_mode);
    ASSERT_EQ(config.backend_type, BackendType::CPU);
}

void test_single_power_flow_with_backend() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json", "-b", "cpu"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_EQ(config.backend_type, BackendType::CPU);
}

void test_single_power_flow_with_gpu_backend() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json", "-b", "gpu"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_EQ(config.backend_type, BackendType::GPU_CUDA);
}

void test_single_power_flow_with_tolerance() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json", "-t", "1e-8"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_NEAR(config.pf_config.tolerance, 1e-8, 1e-10);
}

void test_single_power_flow_with_max_iter() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json", "-m", "100"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_EQ(config.pf_config.max_iterations, 100);
}

void test_single_power_flow_with_verbose() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json", "-v"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_TRUE(config.verbose);
    ASSERT_TRUE(config.pf_config.verbose);
}

void test_single_power_flow_with_benchmark() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json", "--benchmark"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_TRUE(config.benchmark);
}

void test_single_power_flow_with_flat_start() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json", "--flat-start"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_TRUE(config.pf_config.use_flat_start);
}

void test_batch_calculation_basic_args() {
    ArgvHelper args(
        {"gap_main", "-i", "input.json", "-u", "update.json", "-o", "output.json", "--batch"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_EQ(config.input_file, "input.json");
    ASSERT_EQ(config.update_file, "update.json");
    ASSERT_EQ(config.output_file, "output.json");
    ASSERT_TRUE(config.batch_mode);
}

void test_batch_calculation_with_reuse_ybus() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-u", "update.json", "-o", "output.json",
                     "--batch", "--reuse-ybus"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_TRUE(config.batch_mode);
    ASSERT_TRUE(config.batch_config.reuse_y_bus_factorization);
}

void test_batch_calculation_with_warm_start() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-u", "update.json", "-o", "output.json",
                     "--batch", "--warm-start"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_TRUE(config.batch_mode);
    ASSERT_TRUE(config.batch_config.warm_start);
}

void test_batch_calculation_all_options() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-u", "update.json", "-o", "output.json",
                     "--batch", "-b", "gpu", "-t", "1e-8", "-m", "100", "-v", "--benchmark",
                     "--reuse-ybus", "--warm-start"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_TRUE(config.batch_mode);
    ASSERT_EQ(config.input_file, "input.json");
    ASSERT_EQ(config.update_file, "update.json");
    ASSERT_EQ(config.output_file, "output.json");
    ASSERT_EQ(config.backend_type, BackendType::GPU_CUDA);
    ASSERT_NEAR(config.pf_config.tolerance, 1e-8, 1e-10);
    ASSERT_EQ(config.pf_config.max_iterations, 100);
    ASSERT_TRUE(config.verbose);
    ASSERT_TRUE(config.benchmark);
    ASSERT_TRUE(config.batch_config.reuse_y_bus_factorization);
    ASSERT_TRUE(config.batch_config.warm_start);
}

void test_batch_config_inherits_from_base_config() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-u", "update.json", "-o", "output.json",
                     "--batch", "-t", "1e-7", "-m", "75"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    // Base config should be copied to batch config
    ASSERT_NEAR(config.batch_config.base_config.tolerance, 1e-7, 1e-10);
    ASSERT_EQ(config.batch_config.base_config.max_iterations, 75);
}

void test_batch_verbose_sets_verbose_summary() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-u", "update.json", "-o", "output.json",
                     "--batch", "-v"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_TRUE(config.verbose);
    ASSERT_TRUE(config.batch_config.verbose_summary);
}

void test_long_form_arguments() {
    ArgvHelper args({"gap_main", "--input", "input.json", "--output", "output.json", "--update",
                     "update.json", "--batch", "--backend", "cpu", "--tolerance", "1e-6",
                     "--max-iter", "50", "--verbose"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_EQ(config.input_file, "input.json");
    ASSERT_EQ(config.update_file, "update.json");
    ASSERT_EQ(config.output_file, "output.json");
    ASSERT_TRUE(config.batch_mode);
    ASSERT_EQ(config.backend_type, BackendType::CPU);
    ASSERT_NEAR(config.pf_config.tolerance, 1e-6, 1e-10);
    ASSERT_EQ(config.pf_config.max_iterations, 50);
    ASSERT_TRUE(config.verbose);
}

void test_mixed_short_long_arguments() {
    ArgvHelper args({"gap_main", "-i", "input.json", "--output", "output.json", "-u", "update.json",
                     "--batch", "-b", "gpu", "-v"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_EQ(config.input_file, "input.json");
    ASSERT_EQ(config.update_file, "update.json");
    ASSERT_EQ(config.output_file, "output.json");
    ASSERT_TRUE(config.batch_mode);
    ASSERT_EQ(config.backend_type, BackendType::GPU_CUDA);
    ASSERT_TRUE(config.verbose);
}

void test_default_values() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    // Check defaults
    ASSERT_FALSE(config.batch_mode);
    ASSERT_FALSE(config.verbose);
    ASSERT_FALSE(config.benchmark);
    ASSERT_EQ(config.backend_type, BackendType::CPU);
    ASSERT_FALSE(config.pf_config.verbose);
    // Note: reuse_y_bus_factorization defaults to true for performance
    ASSERT_TRUE(config.batch_config.reuse_y_bus_factorization);
    ASSERT_FALSE(config.batch_config.warm_start);
}

void test_batch_without_update_file() {
    ArgvHelper args({"gap_main", "-i", "input.json", "-o", "output.json", "--batch"});

    auto config = parse_test_arguments(args.argc(), args.argv());

    ASSERT_TRUE(config.batch_mode);
    ASSERT_TRUE(config.update_file.empty());
    // In real main.cpp, validation would catch this and throw an error
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "Running Main CLI Tests...\n" << std::endl;

    // Single power flow tests
    run_test("Single power flow basic args", test_single_power_flow_basic_args);
    run_test("Single power flow with backend", test_single_power_flow_with_backend);
    run_test("Single power flow with GPU backend", test_single_power_flow_with_gpu_backend);
    run_test("Single power flow with tolerance", test_single_power_flow_with_tolerance);
    run_test("Single power flow with max iterations", test_single_power_flow_with_max_iter);
    run_test("Single power flow with verbose", test_single_power_flow_with_verbose);
    run_test("Single power flow with benchmark", test_single_power_flow_with_benchmark);
    run_test("Single power flow with flat start", test_single_power_flow_with_flat_start);

    // Batch calculation tests
    run_test("Batch calculation basic args", test_batch_calculation_basic_args);
    run_test("Batch calculation with reuse Y-bus", test_batch_calculation_with_reuse_ybus);
    run_test("Batch calculation with warm start", test_batch_calculation_with_warm_start);
    run_test("Batch calculation all options", test_batch_calculation_all_options);
    run_test("Batch config inherits from base config", test_batch_config_inherits_from_base_config);
    run_test("Batch verbose sets verbose summary", test_batch_verbose_sets_verbose_summary);

    // Argument format tests
    run_test("Long form arguments", test_long_form_arguments);
    run_test("Mixed short/long arguments", test_mixed_short_long_arguments);

    // Default and edge case tests
    run_test("Default values", test_default_values);
    run_test("Batch without update file", test_batch_without_update_file);

    print_test_summary();
    return get_failed_count();
}
