/**
 * @file gap_bindings.cpp
 * @brief Main pybind11 bindings for GAP Power Flow Calculator
 *
 * This file provides Python bindings for the GAP (GPU-Accelerated Power Flow) calculator.
 * Supports both CPU and GPU (CUDA) backends with runtime backend selection.
 */

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/solver/powerflow_interface.h"

// Forward declarations for modular bindings
void init_core_types(pybind11::module& m);
void init_solver_bindings(pybind11::module& m);
void init_io_bindings(pybind11::module& m);

#if GAP_CUDA_AVAILABLE
void init_cuda_bindings(pybind11::module& m);
#endif

namespace py = pybind11;
using namespace gap;

PYBIND11_MODULE(gap_solver, m) {
    m.doc() = "GAP Power Flow Calculator - Python Bindings";

    // Module metadata
    m.attr("__version__") = py::str(VERSION_INFO);

#if GAP_CUDA_AVAILABLE
    m.attr("__cuda_available__") = py::bool_(true);
#else
    m.attr("__cuda_available__") = py::bool_(false);
#endif

    // Initialize core type bindings
    init_core_types(m);

    // Initialize solver bindings
    init_solver_bindings(m);

    // Initialize I/O bindings
    init_io_bindings(m);

#if GAP_CUDA_AVAILABLE
    // Initialize CUDA-specific bindings if available
    init_cuda_bindings(m);
#endif

    // Utility functions
    m.def(
        "get_available_backends", []() { return core::BackendFactory::get_available_backends(); },
        "Get list of available computation backends");

    m.def(
        "is_cuda_available",
        []() {
#if GAP_CUDA_AVAILABLE
            return core::BackendFactory::is_backend_available(BackendType::GPU_CUDA);
#else
            return false;
#endif
        },
        "Check if CUDA backend is available");

    m.def(
        "create_solver",
        [](BackendType backend) { return core::BackendFactory::create_powerflow_solver(backend); },
        "Create a power flow solver for the specified backend", py::arg("backend"));
}