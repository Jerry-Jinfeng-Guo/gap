#include "gap/core/backend_factory.h"

#include <dlfcn.h>

#include <iostream>
#include <stdexcept>

// Include CUDA headers only if CUDA is available at compile time
#if GAP_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

// Include concrete implementations for static linking
#include "../admittance/cpu/cpu_admittance_matrix.cpp"
#include "../io/json_io.cpp"
#include "../solver/lu/cpu/cpu_lu_solver.cpp"
#include "../solver/powerflow/cpu/cpu_iterative_current.cpp"
#include "../solver/powerflow/cpu/cpu_newton_raphson.cpp"

namespace gap::core {

std::unique_ptr<io::IIOModule> BackendFactory::create_io_module() {
    // For now, always return the JSON IO module
    return std::make_unique<io::JsonIOModule>();
}

std::unique_ptr<admittance::IAdmittanceMatrix> BackendFactory::create_admittance_backend(
    BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:
            return std::make_unique<admittance::CPUAdmittanceMatrix>();

        case BackendType::GPU_CUDA: {
            // Try to load GPU backend dynamically
            void* handle = load_backend_library("libgap_admittance_gpu.so");
            if (handle) {
                // Get factory function from shared library
                typedef admittance::IAdmittanceMatrix* (*CreateFunc)();
                CreateFunc create_func = (CreateFunc)dlsym(handle, "create_gpu_admittance_matrix");
                if (create_func) {
                    return std::unique_ptr<admittance::IAdmittanceMatrix>(create_func());
                }
            }
            throw std::runtime_error("GPU admittance backend not available");
        }

        default:
            throw std::invalid_argument("Unknown backend type");
    }
}

std::unique_ptr<solver::ILUSolver> BackendFactory::create_lu_solver(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:
            return std::make_unique<solver::CPULUSolver>();

        case BackendType::GPU_CUDA: {
            // Try to load GPU backend dynamically
            void* handle = load_backend_library("libgap_lu_gpu.so");
            if (handle) {
                typedef solver::ILUSolver* (*CreateFunc)();
                CreateFunc create_func = (CreateFunc)dlsym(handle, "create_gpu_lu_solver");
                if (create_func) {
                    return std::unique_ptr<solver::ILUSolver>(create_func());
                }
            }
            throw std::runtime_error("GPU LU solver backend not available");
        }

        default:
            throw std::invalid_argument("Unknown backend type");
    }
}

std::unique_ptr<solver::IPowerFlowSolver> BackendFactory::create_powerflow_solver(
    BackendType backend_type, PowerFlowMethod method) {
    // Handle Newton-Raphson method
    if (method == PowerFlowMethod::NEWTON_RAPHSON) {
        switch (backend_type) {
            case BackendType::CPU:
                return std::make_unique<solver::CPUNewtonRaphson>();

            case BackendType::GPU_CUDA: {
                // Try to load GPU backend dynamically
                void* handle = load_backend_library("libgap_powerflow_gpu.so");
                if (handle) {
                    typedef solver::IPowerFlowSolver* (*CreateFunc)();
                    CreateFunc create_func =
                        (CreateFunc)dlsym(handle, "create_gpu_powerflow_solver");
                    if (create_func) {
                        return std::unique_ptr<solver::IPowerFlowSolver>(create_func());
                    }
                }
                throw std::runtime_error("GPU power flow solver backend not available");
            }

            default:
                throw std::invalid_argument("Unknown backend type");
        }
    }

    // Handle Iterative Current method
    else if (method == PowerFlowMethod::ITERATIVE_CURRENT) {
        switch (backend_type) {
            case BackendType::CPU:
                return std::make_unique<solver::CPUIterativeCurrent>();

            case BackendType::GPU_CUDA: {
                // Try to load GPU IC solver dynamically
                void* handle = load_backend_library("libgap_powerflow_gpu.so");
                if (handle) {
                    typedef solver::IPowerFlowSolver* (*CreateFunc)();
                    CreateFunc create_func =
                        (CreateFunc)dlsym(handle, "create_gpu_ic_powerflow_solver");
                    if (create_func) {
                        return std::unique_ptr<solver::IPowerFlowSolver>(create_func());
                    }
                }
                throw std::runtime_error("GPU iterative current solver backend not available");
            }

            default:
                throw std::invalid_argument("Unknown backend type");
        }
    }

    throw std::invalid_argument("Unknown power flow method");
}

bool BackendFactory::is_backend_available(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:
            return true;  // CPU backend is always available

        case BackendType::GPU_CUDA:
            return check_cuda_availability();

        default:
            return false;
    }
}

std::vector<BackendType> BackendFactory::get_available_backends() {
    std::vector<BackendType> backends;

    // CPU is always available
    backends.push_back(BackendType::CPU);

    // Check for CUDA availability
    if (check_cuda_availability()) {
        backends.push_back(BackendType::GPU_CUDA);
    }

    return backends;
}

void* BackendFactory::load_backend_library(std::string const& library_name) {
    void* handle = dlopen(library_name.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot load library " << library_name << ": " << dlerror() << std::endl;
        return nullptr;
    }
    return handle;
}

bool BackendFactory::check_cuda_availability() {
#if GAP_CUDA_AVAILABLE
    // Check if CUDA runtime is available
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess || device_count == 0) {
        // For debugging: uncomment the line below to see CUDA error
        // std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // Check if at least one device has compute capability >= 6.0
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if (error == cudaSuccess && prop.major >= 6) {
            return true;
        }
    }

    return false;
#else
    // CUDA not available during compilation
    return false;
#endif
}

}  // namespace gap::core