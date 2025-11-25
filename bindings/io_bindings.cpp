/**
 * @file io_bindings.cpp
 * @brief I/O module bindings for GAP
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gap/core/backend_factory.h"
#include "gap/io/io_interface.h"

namespace py = pybind11;
using namespace gap;
using namespace gap::io;

void init_io_bindings(pybind11::module& m) {
    // I/O interface
    py::class_<IIOModule>(m, "IIOModule")
        .def("read_network_data", &IIOModule::read_network_data, "Read network data from file",
             py::arg("filename"))
        .def("write_results", &IIOModule::write_results, "Write power flow results to file",
             py::arg("filename"), py::arg("bus_voltages"), py::arg("converged"),
             py::arg("iterations"));

    // Factory function for I/O module
    m.def(
        "create_io_module", []() { return core::BackendFactory::create_io_module(); },
        "Create I/O module instance");

    // Convenience functions for direct file operations
    m.def(
        "load_network_from_json",
        [](std::string const& filename) {
            auto io_module = core::BackendFactory::create_io_module();
            return io_module->read_network_data(filename);
        },
        "Load network data from JSON file", py::arg("filename"));

    m.def(
        "save_results_to_json",
        [](std::string const& filename, ComplexVector const& bus_voltages, bool converged,
           int iterations) {
            auto io_module = core::BackendFactory::create_io_module();
            io_module->write_results(filename, bus_voltages, converged, iterations);
        },
        "Save power flow results to JSON file", py::arg("filename"), py::arg("bus_voltages"),
        py::arg("converged"), py::arg("iterations"));
}