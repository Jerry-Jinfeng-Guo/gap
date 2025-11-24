// Save Jacobian matrix to file for external verification
#include <fstream>
#include <iomanip>
#include <iostream>

#include "gap/core/types.h"
#include "gap/logging/logger.h"
#include "gap/solver/powerflow/cpu/cpu_newton_raphson.h"
#include "gap/solver/powerflow/gpu/gpu_newton_raphson.h"
#include "gap/system/system.h"
#include "gap/system/system_io.h"

using namespace gap;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test_case_file>" << std::endl;
        return 1;
    }

    auto& logger = logging::global_logger;
    logger.setMinLevel(logging::LogLevel::INFO);

    try {
        std::string test_file = argv[1];
        System system;

        if (!system_io::load_system_from_file(test_file, system)) {
            std::cerr << "Failed to load system from: " << test_file << std::endl;
            return 1;
        }

        std::cout << "System loaded: " << system.meta.name << std::endl;
        std::cout << "  Buses: " << system.buses.size() << std::endl;

        // CPU Solver
        solver::powerflow::CPUNewtonRaphson cpu_solver(system);

        // Initialize with flat start
        cpu_solver.flat_start();

        // Build Jacobian (one iteration to populate it)
        cpu_solver.solve();

        // Access internal Jacobian matrix
        // We need to add a public method to expose the Jacobian
        // For now, just run one iteration

        std::cout << "CPU solver completed first iteration\n";

    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
