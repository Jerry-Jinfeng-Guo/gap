#include <cmath>
#include <filesystem>
#include <iostream>

#include "gap/core/backend_factory.h"
#include "gap/io/io_interface.h"

#include "test_framework.h"

using namespace gap;
using namespace gap::core;
using namespace gap::solver;

// Helper function to find data files
std::string find_data_file(const std::string& relative_path) {
    std::string cwd = std::filesystem::current_path().string();
    std::vector<std::string> candidates = {relative_path, "../" + relative_path,
                                           "../../" + relative_path};

    if (cwd.find("/build") != std::string::npos) {
        std::string project_root = cwd;
        size_t build_pos = project_root.find("/build");
        if (build_pos != std::string::npos) {
            project_root = project_root.substr(0, build_pos);
            candidates.push_back(project_root + "/" + relative_path);
        }
    }

    for (auto const& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return relative_path;
}

// Helper function to compare two complex numbers
bool complex_near(const Complex& a, const Complex& b, double tol = 1e-10) {
    return std::abs(a.real() - b.real()) < tol && std::abs(a.imag() - b.imag()) < tol;
}

// Helper function to compare two sparse matrices
bool matrices_equal(const SparseMatrix& cpu_matrix, const SparseMatrix& gpu_matrix,
                    double tol = 1e-10) {
    if (cpu_matrix.num_rows != gpu_matrix.num_rows || cpu_matrix.num_cols != gpu_matrix.num_cols ||
        cpu_matrix.nnz != gpu_matrix.nnz) {
        std::cout << "Matrix dimensions differ: CPU (" << cpu_matrix.num_rows << "x"
                  << cpu_matrix.num_cols << ", nnz=" << cpu_matrix.nnz << ") vs GPU ("
                  << gpu_matrix.num_rows << "x" << gpu_matrix.num_cols << ", nnz=" << gpu_matrix.nnz
                  << ")" << std::endl;
        return false;
    }

    if (cpu_matrix.row_ptr != gpu_matrix.row_ptr) {
        std::cout << "Row pointers differ" << std::endl;
        return false;
    }

    if (cpu_matrix.col_idx != gpu_matrix.col_idx) {
        std::cout << "Column indices differ" << std::endl;
        return false;
    }

    for (size_t i = 0; i < cpu_matrix.values.size(); ++i) {
        if (!complex_near(cpu_matrix.values[i], gpu_matrix.values[i], tol)) {
            std::cout << "Value mismatch at index " << i << ": CPU (" << cpu_matrix.values[i].real()
                      << " + " << cpu_matrix.values[i].imag() << "j) vs GPU ("
                      << gpu_matrix.values[i].real() << " + " << gpu_matrix.values[i].imag() << "j)"
                      << std::endl;
            return false;
        }
    }

    return true;
}

void test_gpu_admittance_functionality() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    auto admittance = BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
    ASSERT_TRUE(admittance != nullptr);

    NetworkData network;
    network.num_buses = 5;
    network.num_branches = 4;

    auto matrix = admittance->build_admittance_matrix(network);
    ASSERT_TRUE(matrix != nullptr);
    ASSERT_EQ(5, matrix->num_rows);
}

void test_gpu_vs_cpu_admittance_simple() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Testing GPU vs CPU admittance matrix (simple network) ===" << std::endl;

    // Create a simple 3-bus network
    NetworkData network;
    network.num_buses = 3;
    network.num_branches = 2;

    BusData bus1 = {.id = 0,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 241500.0,
                    .u_pu = 1.05,
                    .u_angle = 0.0};
    BusData bus2 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0};
    BusData bus3 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0};
    network.buses = {bus1, bus2, bus3};

    BranchData branch1 = {.id = 0,
                          .from_bus = 0,
                          .to_bus = 1,
                          .status = true,
                          .r1 = 0.01,
                          .x1 = 0.1,
                          .g1 = 0.0,
                          .b1 = 0.05};
    BranchData branch2 = {.id = 1,
                          .from_bus = 1,
                          .to_bus = 2,
                          .status = true,
                          .r1 = 0.02,
                          .x1 = 0.2,
                          .g1 = 0.0,
                          .b1 = 0.05};
    network.branches = {branch1, branch2};

    // Build matrices with both backends
    auto cpu_backend = BackendFactory::create_admittance_backend(BackendType::CPU);
    auto gpu_backend = BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);

    auto cpu_matrix = cpu_backend->build_admittance_matrix(network);
    auto gpu_matrix = gpu_backend->build_admittance_matrix(network);

    std::cout << "CPU matrix: " << cpu_matrix->num_rows << "x" << cpu_matrix->num_cols
              << ", nnz=" << cpu_matrix->nnz << std::endl;
    std::cout << "GPU matrix: " << gpu_matrix->num_rows << "x" << gpu_matrix->num_cols
              << ", nnz=" << gpu_matrix->nnz << std::endl;

    ASSERT_TRUE(matrices_equal(*cpu_matrix, *gpu_matrix));
    std::cout << "✓ Matrices match!" << std::endl;
}

void test_gpu_vs_cpu_admittance_pgm_network() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Testing GPU vs CPU admittance matrix (PGM network) ===" << std::endl;

    // Load a PGM network file
    std::string network_file = find_data_file("data/pgm/network_1.json");
    auto io = BackendFactory::create_io_module();

    NetworkData network;
    try {
        network = io->read_network_data(network_file);
    } catch (const std::exception& e) {
        std::cout << "Could not load " << network_file << ": " << e.what() << std::endl;
        std::cout << "Skipping PGM network test" << std::endl;
        return;
    }

    std::cout << "Loaded network: " << network.num_buses << " buses, " << network.num_branches
              << " branches, " << network.appliances.size() << " appliances" << std::endl;

    // Build matrices with both backends
    auto cpu_backend = BackendFactory::create_admittance_backend(BackendType::CPU);
    auto gpu_backend = BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);

    auto cpu_matrix = cpu_backend->build_admittance_matrix(network);
    auto gpu_matrix = gpu_backend->build_admittance_matrix(network);

    std::cout << "CPU matrix: " << cpu_matrix->num_rows << "x" << cpu_matrix->num_cols
              << ", nnz=" << cpu_matrix->nnz << std::endl;
    std::cout << "GPU matrix: " << gpu_matrix->num_rows << "x" << gpu_matrix->num_cols
              << ", nnz=" << gpu_matrix->nnz << std::endl;

    ASSERT_TRUE(matrices_equal(*cpu_matrix, *gpu_matrix));
    std::cout << "✓ Matrices match!" << std::endl;
}

void test_gpu_lu_solver_functionality() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);
    ASSERT_TRUE(lu_solver != nullptr);

    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 9;
    matrix.row_ptr = {0, 3, 6, 9};
    matrix.col_idx = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    matrix.values = {Complex(4.0, 0.0),  Complex(-1.0, 0.0), Complex(-1.0, 0.0),
                     Complex(-1.0, 0.0), Complex(4.0, 0.0),  Complex(-1.0, 0.0),
                     Complex(-1.0, 0.0), Complex(-1.0, 0.0), Complex(4.0, 0.0)};

    bool success = lu_solver->factorize(matrix);
    ASSERT_TRUE(success);
}

void test_gpu_powerflow_functionality() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    auto pf_solver = BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA);
    auto lu_solver = BackendFactory::create_lu_solver(BackendType::GPU_CUDA);

    ASSERT_TRUE(pf_solver != nullptr);
    pf_solver->set_lu_solver(std::shared_ptr<ILUSolver>(lu_solver.release()));

    NetworkData network;
    network.num_buses = 3;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 241500.0,
                    .u_pu = 1.05,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,
                    .reactive_power = 50e6};
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 80e6,
                    .reactive_power = 40e6};
    network.buses = {bus1, bus2, bus3};

    SparseMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;

    solver::PowerFlowConfig config;
    config.max_iterations = 5;

    auto result = pf_solver->solve_power_flow(network, matrix, config);
    ASSERT_EQ(3, result.bus_voltages.size());
}
