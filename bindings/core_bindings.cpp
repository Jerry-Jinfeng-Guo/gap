/**
 * @file core_bindings.cpp
 * @brief Core type bindings for GAP
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gap/core/types.h"

namespace py = pybind11;
using namespace gap;

void init_core_types(pybind11::module& m) {
    // Enumerations
    py::enum_<BusType>(m, "BusType")
        .value("PQ", BusType::PQ, "Load bus (P and Q specified)")
        .value("PV", BusType::PV, "Generator bus (P and V specified)")
        .value("SLACK", BusType::SLACK, "Slack/reference bus (V and angle specified)")
        .export_values();

    py::enum_<ApplianceType>(m, "ApplianceType")
        .value("SOURCE", ApplianceType::SOURCE, "Source appliance")
        .value("LOADGEN", ApplianceType::LOADGEN, "Load/generator appliance")
        .value("SHUNT", ApplianceType::SHUNT, "Shunt appliance (capacitor/reactor)")
        .export_values();

    py::enum_<BranchType>(m, "BranchType")
        .value("LINE", BranchType::LINE, "Transmission/distribution line")
        .value("TRAFO", BranchType::TRAFO, "Power transformer")
        .value("GENERIC", BranchType::GENERIC, "Generic branch")
        .export_values();

    py::enum_<BackendType>(m, "BackendType")
        .value("CPU", BackendType::CPU, "CPU backend")
        .value("GPU_CUDA", BackendType::GPU_CUDA, "CUDA GPU backend")
        .export_values();

    // Data structures
    py::class_<BusData>(m, "BusData")
        .def(py::init<>())
        .def_readwrite("id", &BusData::id, "Bus ID")
        .def_readwrite("u_rated", &BusData::u_rated, "Rated voltage (V)")
        .def_readwrite("bus_type", &BusData::bus_type, "Bus type")
        .def_readwrite("energized", &BusData::energized, "Bus connected to source")
        .def_readwrite("u", &BusData::u, "Voltage magnitude (V)")
        .def_readwrite("u_pu", &BusData::u_pu, "Voltage magnitude (p.u.)")
        .def_readwrite("u_angle", &BusData::u_angle, "Voltage angle (rad)")
        .def_readwrite("active_power", &BusData::active_power, "Active power injection (W)")
        .def_readwrite("reactive_power", &BusData::reactive_power, "Reactive power injection (VAr)")
        .def("__repr__", [](const BusData& b) {
            return "<BusData id=" + std::to_string(b.id) + " u_rated=" + std::to_string(b.u_rated) +
                   "V>";
        });

    py::class_<BranchData>(m, "BranchData")
        .def(py::init<>())
        .def_readwrite("id", &BranchData::id, "Branch ID")
        .def_readwrite("from_bus", &BranchData::from_bus, "From bus ID")
        .def_readwrite("to_bus", &BranchData::to_bus, "To bus ID")
        .def_readwrite("branch_type", &BranchData::branch_type, "Branch type")
        .def_readwrite("status", &BranchData::status, "Branch status")
        .def_readwrite("r1", &BranchData::r1, "Positive-sequence resistance (Ω)")
        .def_readwrite("x1", &BranchData::x1, "Positive-sequence reactance (Ω)")
        .def_readwrite("g1", &BranchData::g1, "Positive-sequence conductance (S)")
        .def_readwrite("b1", &BranchData::b1, "Positive-sequence susceptance (S)")
        .def_readwrite("k", &BranchData::k, "Off-nominal ratio (tap)")
        .def_readwrite("theta", &BranchData::theta, "Angle shift (rad)")
        .def_readwrite("sn", &BranchData::sn, "Rated power (VA)")
        .def_readwrite("i_n", &BranchData::i_n, "Rated current (A)")
        .def("set_from_pgm_capacitive_params", &BranchData::set_from_pgm_capacitive_params,
             "Set admittance from PGM capacitive parameters", py::arg("c1"), py::arg("tan1"),
             py::arg("frequency") = 50.0)
        .def("__repr__", [](const BranchData& b) {
            return "<BranchData id=" + std::to_string(b.id) + " " + std::to_string(b.from_bus) +
                   "->" + std::to_string(b.to_bus) + ">";
        });

    py::class_<ApplianceData>(m, "ApplianceData")
        .def(py::init<>())
        .def_readwrite("id", &ApplianceData::id, "Appliance ID")
        .def_readwrite("node", &ApplianceData::node, "Connected bus/node ID")
        .def_readwrite("status", &ApplianceData::status, "Connection status")
        .def_readwrite("type", &ApplianceData::type, "Appliance type")
        .def_readwrite("p", &ApplianceData::p, "Active power (W)")
        .def_readwrite("q", &ApplianceData::q, "Reactive power (VAr)")
        .def_readwrite("u_ref", &ApplianceData::u_ref, "Reference voltage (p.u.)")
        .def_readwrite("u_ref_angle", &ApplianceData::u_ref_angle, "Reference voltage angle (rad)")
        .def_readwrite("p_specified", &ApplianceData::p_specified, "Specified active power (W)")
        .def_readwrite("q_specified", &ApplianceData::q_specified, "Specified reactive power (VAr)")
        .def_readwrite("g1", &ApplianceData::g1, "Positive-sequence conductance (S)")
        .def_readwrite("b1", &ApplianceData::b1, "Positive-sequence susceptance (S)")
        .def("__repr__", [](const ApplianceData& a) {
            return "<ApplianceData id=" + std::to_string(a.id) + " node=" + std::to_string(a.node) +
                   ">";
        });

    py::class_<NetworkData>(m, "NetworkData")
        .def(py::init<>())
        .def_readwrite("buses", &NetworkData::buses, "List of bus data")
        .def_readwrite("branches", &NetworkData::branches, "List of branch data")
        .def_readwrite("appliances", &NetworkData::appliances, "List of appliance data")
        .def_readwrite("num_buses", &NetworkData::num_buses, "Number of buses")
        .def_readwrite("num_branches", &NetworkData::num_branches, "Number of branches")
        .def_readwrite("num_appliances", &NetworkData::num_appliances, "Number of appliances")
        .def_readwrite("version", &NetworkData::version, "PGM format version")
        .def_readwrite("type", &NetworkData::type, "Dataset type identifier")
        .def_readwrite("is_batch", &NetworkData::is_batch, "Batch calculation flag")
        .def("validate_pgm_compliance", &NetworkData::validate_pgm_compliance,
             "Validate PGM compliance and component consistency")
        .def("__repr__", [](const NetworkData& n) {
            return "<NetworkData buses=" + std::to_string(n.num_buses) +
                   " branches=" + std::to_string(n.num_branches) +
                   " appliances=" + std::to_string(n.num_appliances) + ">";
        });
}