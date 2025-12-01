/**
 * @file solver_bindings.cpp
 * @brief Power flow solver bindings for GAP
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>

#include "gap/admittance/admittance_interface.h"
#include "gap/core/backend_factory.h"
#include "gap/solver/powerflow_interface.h"

namespace py = pybind11;
using namespace gap;
using namespace gap::solver;

void init_solver_bindings(pybind11::module& m) {
    // Power flow configuration
    py::class_<PowerFlowConfig>(m, "PowerFlowConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &PowerFlowConfig::tolerance,
                       "Convergence tolerance (default: 1e-6)")
        .def_readwrite("max_iterations", &PowerFlowConfig::max_iterations,
                       "Maximum number of iterations (default: 50)")
        .def_readwrite("use_flat_start", &PowerFlowConfig::use_flat_start,
                       "Use flat start for voltages (default: true)")
        .def_readwrite("acceleration_factor", &PowerFlowConfig::acceleration_factor,
                       "Acceleration factor (default: 1.4)")
        .def_readwrite("verbose", &PowerFlowConfig::verbose,
                       "Enable verbose output (default: false)")
        .def("__repr__", [](PowerFlowConfig const& c) {
            return "<PowerFlowConfig tolerance=" + std::to_string(c.tolerance) +
                   " max_iter=" + std::to_string(c.max_iterations) + ">";
        });

    // Power flow results
    py::class_<PowerFlowResult>(m, "PowerFlowResult")
        .def(py::init<>())
        .def_readwrite("bus_voltages", &PowerFlowResult::bus_voltages, "Bus voltage phasors")
        .def_readwrite("converged", &PowerFlowResult::converged, "Convergence status")
        .def_readwrite("iterations", &PowerFlowResult::iterations, "Number of iterations")
        .def_readwrite("final_mismatch", &PowerFlowResult::final_mismatch, "Final mismatch norm")
        .def_readwrite("bus_injections", &PowerFlowResult::bus_injections,
                       "Calculated bus injections")
        .def("__repr__", [](PowerFlowResult const& r) {
            return "<PowerFlowResult converged=" + std::string(r.converged ? "True" : "False") +
                   " iterations=" + std::to_string(r.iterations) +
                   " buses=" + std::to_string(r.bus_voltages.size()) + ">";
        });

    // Batch power flow configuration
    py::class_<BatchPowerFlowConfig>(m, "BatchPowerFlowConfig")
        .def(py::init<>())
        .def_readwrite("base_config", &BatchPowerFlowConfig::base_config,
                       "Base power flow configuration")
        .def_readwrite("reuse_y_bus_factorization",
                       &BatchPowerFlowConfig::reuse_y_bus_factorization,
                       "Reuse Y-bus factorization across scenarios (default: true)")
        .def_readwrite("warm_start", &BatchPowerFlowConfig::warm_start,
                       "Use previous solution as starting point (default: false)")
        .def_readwrite("verbose_summary", &BatchPowerFlowConfig::verbose_summary,
                       "Print batch statistics summary (default: false)")
        .def("__repr__", [](BatchPowerFlowConfig const& c) {
            return "<BatchPowerFlowConfig reuse_factorization=" +
                   std::string(c.reuse_y_bus_factorization ? "True" : "False") +
                   " warm_start=" + std::string(c.warm_start ? "True" : "False") + ">";
        });

    // Batch power flow results
    py::class_<BatchPowerFlowResult>(m, "BatchPowerFlowResult")
        .def(py::init<>())
        .def_readwrite("results", &BatchPowerFlowResult::results,
                       "Individual power flow results for each scenario")
        .def_readwrite("total_iterations", &BatchPowerFlowResult::total_iterations,
                       "Total iterations across all scenarios")
        .def_readwrite("total_solve_time_ms", &BatchPowerFlowResult::total_solve_time_ms,
                       "Total solve time in milliseconds")
        .def_readwrite("avg_solve_time_ms", &BatchPowerFlowResult::avg_solve_time_ms,
                       "Average solve time per scenario in milliseconds")
        .def_readwrite("converged_count", &BatchPowerFlowResult::converged_count,
                       "Number of scenarios that converged")
        .def_readwrite("failed_count", &BatchPowerFlowResult::failed_count,
                       "Number of scenarios that failed")
        .def("__repr__", [](BatchPowerFlowResult const& r) {
            return "<BatchPowerFlowResult scenarios=" + std::to_string(r.results.size()) +
                   " converged=" + std::to_string(r.converged_count) +
                   " failed=" + std::to_string(r.failed_count) +
                   " avg_time=" + std::to_string(r.avg_solve_time_ms) + "ms>";
        });

    // Ultra-minimal API using only basic Python types (lists and numbers)
    m.def(
        "solve_simple_power_flow",
        [](std::vector<std::vector<double>> const& bus_data,
           std::vector<std::vector<double>> const& branch_data, double tolerance = 1e-6,
           int max_iterations = 50, bool verbose = false, double base_power = 100e6,
           std::string backend = "cpu") -> std::vector<std::vector<double>> {
            try {
                // Convert basic types to internal structures
                NetworkData network_data;

                // Parse bus data
                network_data.buses.reserve(bus_data.size());
                for (size_t i = 0; i < bus_data.size(); ++i) {
                    if (bus_data[i].size() < 7) {
                        throw std::runtime_error(
                            "Bus data must have at least 7 elements: [id, u_rated, bus_type, u_pu, "
                            "u_angle, p, q]");
                    }

                    BusData bus;
                    bus.id = static_cast<int>(bus_data[i][0]);
                    bus.u_rated = bus_data[i][1];
                    bus.bus_type = static_cast<BusType>(static_cast<int>(bus_data[i][2]));
                    bus.energized = true;
                    bus.u_pu = bus_data[i][3];
                    bus.u_angle = bus_data[i][4];
                    bus.active_power = bus_data[i][5];
                    bus.reactive_power = bus_data[i][6];

                    network_data.buses.push_back(bus);
                }

                // Parse branch data
                network_data.branches.reserve(branch_data.size());
                for (size_t i = 0; i < branch_data.size(); ++i) {
                    if (branch_data[i].size() < 5) {
                        throw std::runtime_error(
                            "Branch data must have at least 5 elements: [id, from_bus, to_bus, r, "
                            "x]");
                    }

                    BranchData branch;
                    branch.id = static_cast<int>(branch_data[i][0]);
                    branch.from_bus = static_cast<int>(branch_data[i][1]);
                    branch.to_bus = static_cast<int>(branch_data[i][2]);
                    branch.status = true;
                    branch.r1 = branch_data[i][3];
                    branch.x1 = branch_data[i][4];
                    branch.b1 = (branch_data[i].size() > 5) ? branch_data[i][5] : 0.0;

                    network_data.branches.push_back(branch);
                }

                network_data.num_buses = network_data.buses.size();
                network_data.num_branches = network_data.branches.size();
                network_data.num_appliances = 0;

                // Create solver configuration
                PowerFlowConfig config;
                config.tolerance = tolerance;
                config.max_iterations = max_iterations;
                config.verbose = verbose;
                config.base_power = base_power;

                // Determine backend type
                BackendType backend_type = BackendType::CPU;
                if (backend == "gpu" || backend == "GPU" || backend == "cuda" ||
                    backend == "CUDA") {
                    backend_type = BackendType::GPU_CUDA;
                }

                // Check if backend is available
                if (!core::BackendFactory::is_backend_available(backend_type)) {
                    throw std::runtime_error("Requested backend '" + backend +
                                             "' is not available");
                }

                // Create solver and solve
                auto solver = core::BackendFactory::create_powerflow_solver(backend_type);
                if (!solver) {
                    throw std::runtime_error("Failed to create solver");
                }

                // Create LU solver backend
                std::shared_ptr<solver::ILUSolver> lu_solver(
                    core::BackendFactory::create_lu_solver(backend_type).release());
                if (!lu_solver) {
                    throw std::runtime_error("Failed to create LU solver");
                }

                // Set LU solver on the powerflow solver
                solver->set_lu_solver(lu_solver);

                auto admittance_builder =
                    core::BackendFactory::create_admittance_backend(backend_type);
                if (!admittance_builder) {
                    throw std::runtime_error("Failed to create admittance builder");
                }

                auto admittance_matrix = admittance_builder->build_admittance_matrix(network_data);
                if (!admittance_matrix) {
                    throw std::runtime_error("Failed to build admittance matrix");
                }

                auto result = solver->solve_power_flow(network_data, *admittance_matrix, config);

                // Convert result to basic Python types
                std::vector<std::vector<double>> output;
                for (auto const& voltage : result.bus_voltages) {
                    output.push_back(
                        {voltage.real(), voltage.imag(), std::abs(voltage), std::arg(voltage)});
                }

                // Add metadata as last row: [converged, iterations, final_mismatch, 0]
                output.push_back({
                    result.converged ? 1.0 : 0.0, static_cast<double>(result.iterations),
                    result.final_mismatch,
                    0.0  // reserved
                });

                return output;

            } catch (std::exception const& e) {
                throw std::runtime_error(std::string("Simple power flow solution failed: ") +
                                         e.what());
            }
        },
        "Solve power flow using only basic Python types (ultra-minimal API)", py::arg("bus_data"),
        py::arg("branch_data"), py::arg("tolerance") = 1e-6, py::arg("max_iterations") = 50,
        py::arg("verbose") = false, py::arg("base_power") = 100e6, py::arg("backend") = "cpu",
        R"(
          Solve power flow using only basic Python lists and numbers.
          
          Args:
              bus_data: List of bus data, each bus: [id, u_rated, bus_type, u_pu, u_angle, p_load, q_load]
                       id: Bus ID (can be 0-based or 1-based, auto-detected)
                       u_rated: Rated voltage in Volts (e.g., 11000.0)  
                       bus_type: 0=PQ, 1=PV, 2=SLACK
                       u_pu: Initial voltage magnitude in p.u. (e.g., 1.0)
                       u_angle: Initial voltage angle in radians (e.g., 0.0)
                       p_load: Active power load in Watts (negative for loads)
                       q_load: Reactive power load in VAr (negative for loads)
              branch_data: List of branch data, each branch: [id, from_bus, to_bus, r, x, b(optional)]
                       id: Branch ID (can be any integer)
                       from_bus: Source bus ID (must match ID scheme in bus_data)
                       to_bus: Destination bus ID (must match ID scheme in bus_data)
                       r: Resistance in Ohms
                       x: Reactance in Ohms
                       b: Susceptance in Siemens (optional, default: 0.0)
              tolerance: Convergence tolerance (default: 1e-6)
              max_iterations: Maximum iterations (default: 50)
              verbose: Enable verbose output (default: False)
              base_power: Base power for per-unit system in VA (default: 100e6 = 100 MVA)
              backend: Computation backend - 'cpu' or 'gpu' (default: 'cpu')
          
          Returns:
              List of voltage results, each voltage: [real, imag, magnitude, angle_rad]
              Last row contains metadata: [converged(0/1), iterations, final_mismatch, 0]
          
          Note:
              Bus IDs are auto-detected as 0-based or 1-based. Both conventions work seamlessly.
          )");

    // Batch power flow solve function
    m.def(
        "solve_simple_power_flow_batch",
        [](std::vector<std::vector<double>> const& bus_data,
           std::vector<std::vector<double>> const& branch_data,
           std::vector<std::vector<std::vector<double>>> const& scenarios_data,
           double tolerance = 1e-6, int max_iterations = 50, bool verbose = false,
           double base_power = 100e6, std::string backend = "cpu",
           bool reuse_y_bus_factorization = true, bool warm_start = false,
           bool verbose_summary = false)
            -> std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<double>> {
            try {
                // Convert basic types to internal structures (base network)
                NetworkData base_network;

                // Parse bus data (same as single solve)
                base_network.buses.reserve(bus_data.size());
                for (size_t i = 0; i < bus_data.size(); ++i) {
                    if (bus_data[i].size() < 7) {
                        throw std::runtime_error(
                            "Bus data must have at least 7 elements: [id, u_rated, bus_type, "
                            "u_pu, u_angle, p, q]");
                    }

                    BusData bus;
                    bus.id = static_cast<int>(bus_data[i][0]);
                    bus.u_rated = bus_data[i][1];
                    bus.bus_type = static_cast<BusType>(static_cast<int>(bus_data[i][2]));
                    bus.energized = true;
                    bus.u_pu = bus_data[i][3];
                    bus.u_angle = bus_data[i][4];
                    bus.active_power = bus_data[i][5];
                    bus.reactive_power = bus_data[i][6];

                    base_network.buses.push_back(bus);
                }

                // Parse branch data (same as single solve)
                base_network.branches.reserve(branch_data.size());
                for (size_t i = 0; i < branch_data.size(); ++i) {
                    if (branch_data[i].size() < 5) {
                        throw std::runtime_error(
                            "Branch data must have at least 5 elements: [id, from_bus, to_bus, r, "
                            "x]");
                    }

                    BranchData branch;
                    branch.id = static_cast<int>(branch_data[i][0]);
                    branch.from_bus = static_cast<int>(branch_data[i][1]);
                    branch.to_bus = static_cast<int>(branch_data[i][2]);
                    branch.status = true;
                    branch.r1 = branch_data[i][3];
                    branch.x1 = branch_data[i][4];
                    branch.b1 = (branch_data[i].size() > 5) ? branch_data[i][5] : 0.0;

                    base_network.branches.push_back(branch);
                }

                base_network.num_buses = base_network.buses.size();
                base_network.num_branches = base_network.branches.size();
                base_network.num_appliances = 0;

                // Create scenarios by updating loads
                std::vector<NetworkData> scenarios;
                scenarios.reserve(scenarios_data.size());

                for (auto const& scenario : scenarios_data) {
                    NetworkData scenario_network = base_network;  // Copy base network

                    // Update bus loads for this scenario
                    // Each scenario: list of [bus_id, p_load, q_load]
                    for (auto const& load_update : scenario) {
                        if (load_update.size() < 3) {
                            throw std::runtime_error(
                                "Load update must have 3 elements: [bus_id, p_load, q_load]");
                        }

                        int bus_id = static_cast<int>(load_update[0]);
                        double p_load = load_update[1];
                        double q_load = load_update[2];

                        // Find and update the bus
                        bool found = false;
                        for (auto& bus : scenario_network.buses) {
                            if (bus.id == bus_id) {
                                bus.active_power = p_load;
                                bus.reactive_power = q_load;
                                found = true;
                                break;
                            }
                        }

                        if (!found) {
                            throw std::runtime_error("Bus ID " + std::to_string(bus_id) +
                                                     " not found in network");
                        }
                    }

                    scenarios.push_back(scenario_network);
                }

                // Create solver configuration
                PowerFlowConfig base_config;
                base_config.tolerance = tolerance;
                base_config.max_iterations = max_iterations;
                base_config.verbose = verbose;
                base_config.base_power = base_power;

                BatchPowerFlowConfig batch_config;
                batch_config.base_config = base_config;
                batch_config.reuse_y_bus_factorization = reuse_y_bus_factorization;
                batch_config.warm_start = warm_start;
                batch_config.verbose_summary = verbose_summary;

                // Determine backend type
                BackendType backend_type = BackendType::CPU;
                if (backend == "gpu" || backend == "GPU" || backend == "cuda" ||
                    backend == "CUDA") {
                    backend_type = BackendType::GPU_CUDA;
                }

                // Check if backend is available
                if (!core::BackendFactory::is_backend_available(backend_type)) {
                    throw std::runtime_error("Requested backend '" + backend +
                                             "' is not available");
                }

                // Create solver
                auto solver = core::BackendFactory::create_powerflow_solver(backend_type);
                if (!solver) {
                    throw std::runtime_error("Failed to create solver");
                }

                // Create LU solver backend
                std::shared_ptr<solver::ILUSolver> lu_solver(
                    core::BackendFactory::create_lu_solver(backend_type).release());
                if (!lu_solver) {
                    throw std::runtime_error("Failed to create LU solver");
                }

                solver->set_lu_solver(lu_solver);

                // Create admittance matrix for base network
                auto admittance_builder =
                    core::BackendFactory::create_admittance_backend(backend_type);
                if (!admittance_builder) {
                    throw std::runtime_error("Failed to create admittance builder");
                }

                auto admittance_matrix = admittance_builder->build_admittance_matrix(base_network);
                if (!admittance_matrix) {
                    throw std::runtime_error("Failed to build admittance matrix");
                }

                // Solve batch
                auto batch_result =
                    solver->solve_power_flow_batch(scenarios, *admittance_matrix, batch_config);

                // Convert results to Python types
                std::vector<std::vector<std::vector<double>>> all_voltages;
                all_voltages.reserve(batch_result.results.size());

                for (auto const& result : batch_result.results) {
                    std::vector<std::vector<double>> scenario_voltages;
                    for (auto const& voltage : result.bus_voltages) {
                        scenario_voltages.push_back(
                            {voltage.real(), voltage.imag(), std::abs(voltage), std::arg(voltage)});
                    }
                    all_voltages.push_back(scenario_voltages);
                }

                // Create statistics vector
                std::vector<double> stats = {static_cast<double>(batch_result.total_iterations),
                                             batch_result.total_solve_time_ms,
                                             batch_result.avg_solve_time_ms,
                                             static_cast<double>(batch_result.converged_count),
                                             static_cast<double>(batch_result.failed_count)};

                return std::make_tuple(all_voltages, stats);

            } catch (std::exception const& e) {
                throw std::runtime_error(std::string("Batch power flow solution failed: ") +
                                         e.what());
            }
        },
        "Solve batch power flow using only basic Python types", py::arg("bus_data"),
        py::arg("branch_data"), py::arg("scenarios_data"), py::arg("tolerance") = 1e-6,
        py::arg("max_iterations") = 50, py::arg("verbose") = false, py::arg("base_power") = 100e6,
        py::arg("backend") = "cpu", py::arg("reuse_y_bus_factorization") = true,
        py::arg("warm_start") = false, py::arg("verbose_summary") = false,
        R"(
          Solve multiple power flow scenarios efficiently with Y-bus factorization caching.
          
          Args:
              bus_data: Base network bus data (same format as solve_simple_power_flow)
              branch_data: Branch data (same format as solve_simple_power_flow)
              scenarios_data: List of scenarios, each scenario is a list of load updates
                             Each update: [bus_id, p_load, q_load]
              tolerance: Convergence tolerance (default: 1e-6)
              max_iterations: Maximum iterations per scenario (default: 50)
              verbose: Enable verbose output (default: False)
              base_power: Base power for per-unit system in VA (default: 100 MVA)
              backend: Computation backend - 'cpu' or 'gpu' (default: 'cpu')
              reuse_y_bus_factorization: Reuse Y-bus factorization across scenarios (default: True)
              warm_start: Use previous solution as starting point (default: False)
              verbose_summary: Print batch statistics summary (default: False)
          
          Returns:
              Tuple of (voltages, statistics):
                - voltages: List of voltage results per scenario, each containing
                           list of [real, imag, magnitude, angle_rad] per bus
                - statistics: [total_iterations, total_time_ms, avg_time_ms, 
                              converged_count, failed_count]
          
          Performance:
              With reuse_y_bus_factorization=True, the Y-bus matrix is factorized once
              and reused across all scenarios, providing significant speedup for 
              time-series and contingency analysis.
          )");
}