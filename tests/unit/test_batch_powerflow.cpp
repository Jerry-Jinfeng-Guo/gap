#include <filesystem>

#include "gap/admittance/admittance_interface.h"
#include "gap/core/backend_factory.h"
#include "gap/io/io_interface.h"
#include "gap/solver/lu_solver_interface.h"
#include "gap/solver/powerflow_interface.h"

#include "test_framework.h"

using namespace gap;
using namespace gap::solver;
using namespace gap::core;

// Helper function to find data files in various relative paths
static std::string find_data_file(std::string const& relative_path) {
    std::vector<std::string> candidates = {
        relative_path,            // Direct path from project root
        "../" + relative_path,    // From build directory
        "../../" + relative_path  // From build/bin directory
    };

    for (auto const& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    // If not found, return the original path and let error occur
    return relative_path;
}

void test_batch_results_match_individual() {
    // Load test network
    auto io_module = BackendFactory::create_io_module();
    auto base_network = io_module->read_network_data(find_data_file("data/pgm/network_1.json"));

    // Build admittance matrix
    auto admittance = BackendFactory::create_admittance_backend(BackendType::CPU);
    auto y_matrix = admittance->build_admittance_matrix(base_network);

    // Create CPU IC solver
    auto cpu_ic_solver = BackendFactory::create_powerflow_solver(
        BackendType::CPU, PowerFlowMethod::ITERATIVE_CURRENT);
    cpu_ic_solver->set_lu_solver(BackendFactory::create_lu_solver(BackendType::CPU));

    // Create 5 scenarios with varying loads
    std::vector<NetworkData> scenarios;
    std::vector<double> load_factors = {0.8, 0.9, 1.0, 1.1, 1.2};

    for (double factor : load_factors) {
        NetworkData scenario = base_network;
        for (auto& appliance : scenario.appliances) {
            appliance.p_specified *= factor;
            appliance.q_specified *= factor;
        }
        scenarios.push_back(scenario);
    }

    PowerFlowConfig config;
    config.tolerance = 1e-6;
    config.max_iterations = 50;
    config.verbose = false;

    // Solve individually
    std::vector<PowerFlowResult> individual_results;
    for (auto const& scenario : scenarios) {
        individual_results.push_back(cpu_ic_solver->solve_power_flow(scenario, *y_matrix, config));
    }

    // Solve as batch
    BatchPowerFlowConfig batch_config;
    batch_config.base_config = config;
    batch_config.reuse_y_bus_factorization = true;
    batch_config.verbose_summary = true;

    auto batch_result = cpu_ic_solver->solve_power_flow_batch(scenarios, *y_matrix, batch_config);

    // Verify results match
    ASSERT_EQ(batch_result.results.size(), scenarios.size());
    ASSERT_EQ(batch_result.converged_count, static_cast<int>(scenarios.size()));
    ASSERT_EQ(batch_result.failed_count, 0);

    for (size_t i = 0; i < scenarios.size(); ++i) {
        ASSERT_TRUE(batch_result.results[i].converged);
        ASSERT_EQ(batch_result.results[i].iterations, individual_results[i].iterations);
        ASSERT_NEAR(batch_result.results[i].final_mismatch, individual_results[i].final_mismatch,
                    1e-8);

        // Check voltage results match
        for (size_t bus = 0; bus < batch_result.results[i].bus_voltages.size(); ++bus) {
            double diff = std::abs(batch_result.results[i].bus_voltages[bus] -
                                   individual_results[i].bus_voltages[bus]);
            ASSERT_TRUE(diff < 1e-8);
        }
    }
}

void test_batch_performance_improvement() {
    // Load test network
    auto io_module = BackendFactory::create_io_module();
    auto base_network = io_module->read_network_data(find_data_file("data/pgm/network_1.json"));

    auto admittance = BackendFactory::create_admittance_backend(BackendType::CPU);
    auto y_matrix = admittance->build_admittance_matrix(base_network);

    auto cpu_ic_solver = BackendFactory::create_powerflow_solver(
        BackendType::CPU, PowerFlowMethod::ITERATIVE_CURRENT);
    cpu_ic_solver->set_lu_solver(BackendFactory::create_lu_solver(BackendType::CPU));

    // Create 20 scenarios
    std::vector<NetworkData> scenarios;
    for (int i = 0; i < 20; ++i) {
        NetworkData scenario = base_network;
        double factor = 0.7 + 0.03 * i;
        for (auto& appliance : scenario.appliances) {
            appliance.p_specified *= factor;
            appliance.q_specified *= factor;
        }
        scenarios.push_back(scenario);
    }

    PowerFlowConfig config;
    config.tolerance = 1e-6;
    config.max_iterations = 50;
    config.verbose = false;

    // Time batch solve
    BatchPowerFlowConfig batch_config;
    batch_config.base_config = config;
    batch_config.reuse_y_bus_factorization = true;
    batch_config.verbose_summary = false;

    auto batch_result = cpu_ic_solver->solve_power_flow_batch(scenarios, *y_matrix, batch_config);

    std::cout << "  Batch solve completed: " << scenarios.size() << " scenarios\n";
    std::cout << "  Total time: " << batch_result.total_solve_time_ms << " ms\n";
    std::cout << "  Avg time per scenario: " << batch_result.avg_solve_time_ms << " ms\n";

    ASSERT_TRUE(batch_result.converged_count == static_cast<int>(scenarios.size()));
}

void test_batch_empty_scenarios() {
    auto io_module = BackendFactory::create_io_module();
    auto base_network = io_module->read_network_data(find_data_file("data/pgm/network_1.json"));

    auto admittance = BackendFactory::create_admittance_backend(BackendType::CPU);
    auto y_matrix = admittance->build_admittance_matrix(base_network);

    auto cpu_ic_solver = BackendFactory::create_powerflow_solver(
        BackendType::CPU, PowerFlowMethod::ITERATIVE_CURRENT);
    cpu_ic_solver->set_lu_solver(BackendFactory::create_lu_solver(BackendType::CPU));

    std::vector<NetworkData> empty_scenarios;
    BatchPowerFlowConfig batch_config;

    auto batch_result =
        cpu_ic_solver->solve_power_flow_batch(empty_scenarios, *y_matrix, batch_config);

    ASSERT_EQ(batch_result.results.size(), 0);
    ASSERT_EQ(batch_result.converged_count, 0);
    ASSERT_EQ(batch_result.failed_count, 0);
}

void test_batch_nr_solver() {
    auto io_module = BackendFactory::create_io_module();
    auto base_network = io_module->read_network_data(find_data_file("data/pgm/network_1.json"));

    auto admittance = BackendFactory::create_admittance_backend(BackendType::CPU);
    auto y_matrix = admittance->build_admittance_matrix(base_network);

    auto cpu_nr_solver = BackendFactory::create_powerflow_solver(BackendType::CPU);
    cpu_nr_solver->set_lu_solver(BackendFactory::create_lu_solver(BackendType::CPU));

    std::vector<NetworkData> scenarios;
    for (int i = 0; i < 5; ++i) {
        NetworkData scenario = base_network;
        double factor = 0.8 + 0.1 * i;
        for (auto& appliance : scenario.appliances) {
            appliance.p_specified *= factor;
            appliance.q_specified *= factor;
        }
        scenarios.push_back(scenario);
    }

    BatchPowerFlowConfig batch_config;
    batch_config.base_config.tolerance = 1e-6;
    batch_config.base_config.max_iterations = 50;
    batch_config.verbose_summary = true;

    auto batch_result = cpu_nr_solver->solve_power_flow_batch(scenarios, *y_matrix, batch_config);

    ASSERT_EQ(batch_result.results.size(), scenarios.size());
    ASSERT_EQ(batch_result.converged_count, static_cast<int>(scenarios.size()));
    ASSERT_EQ(batch_result.failed_count, 0);

    for (auto const& result : batch_result.results) {
        ASSERT_TRUE(result.converged);
        ASSERT_TRUE(result.final_mismatch < 1e-6);
    }
}

void register_batch_tests(TestRunner& runner) {
    runner.add_test("Batch results match individual solves", test_batch_results_match_individual);
    runner.add_test("Batch solver performance check", test_batch_performance_improvement);
    runner.add_test("Batch handles empty scenarios", test_batch_empty_scenarios);
    runner.add_test("CPU Newton-Raphson batch solver", test_batch_nr_solver);
}
