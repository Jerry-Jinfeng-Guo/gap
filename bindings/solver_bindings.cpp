/**
 * @file solver_bindings.cpp
 * @brief Power flow solver bindings for GAP
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

    // Abstract power flow solver interface
    py::class_<IPowerFlowSolver>(m, "IPowerFlowSolver")
        .def("solve_power_flow", &IPowerFlowSolver::solve_power_flow,
             "Solve power flow using Newton-Raphson method", py::arg("network_data"),
             py::arg("admittance_matrix"), py::arg("config") = PowerFlowConfig{})
        .def("calculate_mismatches", &IPowerFlowSolver::calculate_mismatches,
             "Calculate power mismatches", py::arg("network_data"), py::arg("bus_voltages"),
             py::arg("admittance_matrix"))
        .def("get_backend_type", &IPowerFlowSolver::get_backend_type, "Get backend execution type");

    // Factory functions for solver creation
    m.def(
        "create_cpu_newton_raphson",
        []() { return core::BackendFactory::create_powerflow_solver(BackendType::CPU); },
        "Create CPU Newton-Raphson power flow solver");

#if GAP_CUDA_AVAILABLE
    m.def(
        "create_gpu_newton_raphson",
        []() { return core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA); },
        "Create GPU (CUDA) Newton-Raphson power flow solver");
#endif

    // Convenience solver classes that can be directly instantiated
    // Note: These will be defined in separate CPU/GPU specific binding files
}