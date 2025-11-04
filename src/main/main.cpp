#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/logging/logger.h"

using namespace gap;

/**
 * @brief Configuration structure for the application
 */
struct AppConfig {
    std::string input_file;
    std::string output_file;
    BackendType backend_type = BackendType::CPU;
    solver::PowerFlowConfig pf_config;
    bool verbose = false;
    bool benchmark = false;
};

/**
 * @brief Print usage information
 */
void print_usage(char const* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "\nOPTIONS:\n"
              << "  -i, --input FILE      Input file path (required)\n"
              << "  -o, --output FILE     Output file path (required)\n"
              << "  -b, --backend TYPE    Backend type: cpu, gpu (default: cpu)\n"
              << "  -t, --tolerance VAL   Convergence tolerance (default: 1e-6)\n"
              << "  -m, --max-iter NUM    Maximum iterations (default: 50)\n"
              << "  -v, --verbose         Enable verbose output\n"
              << "  --benchmark           Enable benchmarking\n"
              << "  --flat-start          Use flat start initialization\n"
              << "  -h, --help            Show this help message\n"
              << "\nEXAMPLES:\n"
              << "  " << program_name << " -i network.json -o results.json\n"
              << "  " << program_name << " -i network.json -o results.json -b gpu -v\n"
              << std::endl;
}

/**
 * @brief Parse command line arguments
 */
AppConfig parse_arguments(int argc, char* argv[]) {
    AppConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                config.input_file = argv[++i];
            } else {
                throw std::invalid_argument("Missing input file argument");
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                config.output_file = argv[++i];
            } else {
                throw std::invalid_argument("Missing output file argument");
            }
        } else if (arg == "-b" || arg == "--backend") {
            if (i + 1 < argc) {
                std::string backend_str = argv[++i];
                if (backend_str == "cpu") {
                    config.backend_type = BackendType::CPU;
                } else if (backend_str == "gpu") {
                    config.backend_type = BackendType::GPU_CUDA;
                } else {
                    throw std::invalid_argument("Invalid backend type: " + backend_str);
                }
            } else {
                throw std::invalid_argument("Missing backend argument");
            }
        } else if (arg == "-t" || arg == "--tolerance") {
            if (i + 1 < argc) {
                config.pf_config.tolerance = std::stod(argv[++i]);
            } else {
                throw std::invalid_argument("Missing tolerance argument");
            }
        } else if (arg == "-m" || arg == "--max-iter") {
            if (i + 1 < argc) {
                config.pf_config.max_iterations = std::stoi(argv[++i]);
            } else {
                throw std::invalid_argument("Missing max iterations argument");
            }
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
            config.pf_config.verbose = true;
        } else if (arg == "--benchmark") {
            config.benchmark = true;
        } else if (arg == "--flat-start") {
            config.pf_config.use_flat_start = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    // Validate required arguments
    if (config.input_file.empty()) {
        throw std::invalid_argument("Input file is required");
    }
    if (config.output_file.empty()) {
        throw std::invalid_argument("Output file is required");
    }

    return config;
}

/**
 * @brief Main power flow calculation function
 */
int run_power_flow(AppConfig const& config) {
    auto& logger = gap::logging::global_logger;
    logger.setComponent("GAP");

    try {
        // Check if requested backend is available
        if (!core::BackendFactory::is_backend_available(config.backend_type)) {
            LOG_ERROR(logger, "Requested backend is not available");
            return 1;
        }

        if (config.verbose) {
            LOG_INFO(logger, "GAP Power Flow Calculator");
            LOG_INFO(logger, "Configuration:");
            LOG_INFO(logger, "  Input file: {}", config.input_file);
            LOG_INFO(logger, "  Output file: {}", config.output_file);
            LOG_INFO(logger, "  Backend: {}",
                     (config.backend_type == BackendType::CPU ? "CPU" : "GPU"));
            LOG_INFO(logger, "  Tolerance: {}", config.pf_config.tolerance);
            LOG_INFO(logger, "  Max iterations: {}", config.pf_config.max_iterations);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Create backend instances
        auto io_module = core::BackendFactory::create_io_module();
        auto admittance_backend =
            core::BackendFactory::create_admittance_backend(config.backend_type);
        std::shared_ptr<solver::ILUSolver> lu_solver(
            core::BackendFactory::create_lu_solver(config.backend_type).release());
        auto powerflow_solver = core::BackendFactory::create_powerflow_solver(config.backend_type);

        // Read network data
        if (config.verbose) {
            LOG_INFO(logger, "Reading network data...");
        }
        auto network_data = io_module->read_network_data(config.input_file);

        // Build admittance matrix
        if (config.verbose) {
            LOG_INFO(logger, "Building admittance matrix...");
        }
        auto admittance_matrix = admittance_backend->build_admittance_matrix(network_data);

        // Configure power flow solver
        powerflow_solver->set_lu_solver(lu_solver);

        // Solve power flow
        if (config.verbose) {
            LOG_INFO(logger, "Solving power flow...");
        }
        auto result =
            powerflow_solver->solve_power_flow(network_data, *admittance_matrix, config.pf_config);

        // Write results
        if (config.verbose) {
            LOG_INFO(logger, "Writing results...");
        }
        io_module->write_results(config.output_file, result.bus_voltages, result.converged,
                                 result.iterations);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Print summary
        std::cout << "\nPower Flow Solution Summary:" << std::endl;
        std::cout << "  Status: " << (result.converged ? "CONVERGED" : "NOT CONVERGED")
                  << std::endl;
        std::cout << "  Iterations: " << result.iterations << std::endl;
        std::cout << "  Final mismatch: " << result.final_mismatch << std::endl;
        std::cout << "  Bus voltages: " << result.bus_voltages.size() << std::endl;

        if (config.benchmark) {
            std::cout << "  Execution time: " << duration.count() << " ms" << std::endl;
        }

        return result.converged ? 0 : 1;

    } catch (std::exception const& e) {
        LOG_ERROR(logger, "Error: {}", e.what());
        return 1;
    }
}

/**
 * @brief Main entry point
 */
int main(int argc, char* argv[]) {
    try {
        // Show available backends
        auto available_backends = core::BackendFactory::get_available_backends();
        if (argc == 1) {
            std::cout << "GAP Power Flow Calculator\n" << std::endl;
            std::cout << "Available backends: ";
            for (auto backend : available_backends) {
                std::cout << (backend == BackendType::CPU ? "CPU " : "GPU ");
            }
            std::cout << std::endl << std::endl;
            print_usage(argv[0]);
            return 0;
        }

        // Parse command line arguments
        auto config = parse_arguments(argc, argv);

        // Run power flow calculation
        return run_power_flow(config);

    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}
