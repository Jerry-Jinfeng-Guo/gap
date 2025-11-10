/**
 * @file cuda_bindings.cpp
 * @brief CUDA-specific bindings for GAP (only compiled when CUDA is available)
 */

#if GAP_CUDA_AVAILABLE

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// CUDA headers
#include <cuda_runtime.h>

#include "gap/core/backend_factory.h"
#include "gap/solver/powerflow_interface.h"

namespace py = pybind11;
using namespace gap;
using namespace gap::solver;

void init_cuda_bindings(pybind11::module& m) {
    // CUDA-specific utility functions
    m.def(
        "get_cuda_device_count",
        []() {
            int device_count = 0;
            cudaError_t error = cudaGetDeviceCount(&device_count);
            if (error != cudaSuccess) {
                return 0;
            }
            return device_count;
        },
        "Get number of available CUDA devices");

    m.def(
        "get_cuda_device_properties",
        [](int device_id = 0) {
            cudaDeviceProp prop;
            cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
            if (error != cudaSuccess) {
                throw std::runtime_error("Failed to get CUDA device properties");
            }

            py::dict properties;
            properties["name"] = std::string(prop.name);
            properties["major"] = prop.major;
            properties["minor"] = prop.minor;
            properties["total_memory"] = prop.totalGlobalMem;
            properties["max_threads_per_block"] = prop.maxThreadsPerBlock;
            properties["multiprocessor_count"] = prop.multiProcessorCount;
            properties["warp_size"] = prop.warpSize;

            return properties;
        },
        "Get CUDA device properties", py::arg("device_id") = 0);

    m.def(
        "set_cuda_device",
        [](int device_id) {
            cudaError_t error = cudaSetDevice(device_id);
            if (error != cudaSuccess) {
                throw std::runtime_error("Failed to set CUDA device");
            }
        },
        "Set active CUDA device", py::arg("device_id"));

    // GPU-specific factory functions
    m.def(
        "create_gpu_newton_raphson_direct",
        []() { return core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA); },
        "Create GPU Newton-Raphson power flow solver (direct instantiation)");

    // CUDA-specific convenience class (using py::class_ with a holder type to avoid conflicts)
    m.def(
        "GPUNewtonRaphson",
        []() { return core::BackendFactory::create_powerflow_solver(BackendType::GPU_CUDA); },
        "Create GPU Newton-Raphson power flow solver");

    // CUDA memory management utilities (if needed for large networks)
    m.def(
        "cuda_memory_info",
        []() {
            size_t free_mem, total_mem;
            cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
            if (error != cudaSuccess) {
                throw std::runtime_error("Failed to get CUDA memory info");
            }

            py::dict mem_info;
            mem_info["free"] = free_mem;
            mem_info["total"] = total_mem;
            mem_info["used"] = total_mem - free_mem;

            return mem_info;
        },
        "Get CUDA memory usage information");
}

#endif  // GAP_CUDA_AVAILABLE