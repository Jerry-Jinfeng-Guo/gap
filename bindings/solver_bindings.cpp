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
        .def("__repr__", [](const PowerFlowConfig& c) {
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
        .def("__repr__", [](const PowerFlowResult& r) {
            return "<PowerFlowResult converged=" + std::string(r.converged ? "True" : "False") +
                   " iterations=" + std::to_string(r.iterations) +
                   " buses=" + std::to_string(r.bus_voltages.size()) + ">";
        });

    // Ultra-minimal API using only basic Python types (lists and numbers)
    m.def(
        "solve_simple_power_flow",
        [](const std::vector<std::vector<double>>& bus_data,
           const std::vector<std::vector<double>>& branch_data, double tolerance = 1e-6,
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
                for (const auto& voltage : result.bus_voltages) {
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

            } catch (const std::exception& e) {
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
}