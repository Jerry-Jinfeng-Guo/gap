/**
 * @file test_kernel_runner.cu
 * @brief Executable to run in-library GPU kernel unit tests
 *
 * This executable calls the kernel tests that are compiled directly into
 * the gap_powerflow_gpu library. These tests can only be enabled at compile
 * time using -DGAP_ENABLE_KERNEL_TESTS=ON.
 */

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#ifdef GAP_ENABLE_KERNEL_TESTS
#include "gap/solver/gpu_newton_raphson_kernels.h"
#endif

int main() {
#ifdef GAP_ENABLE_KERNEL_TESTS
    std::cout << "=== Running In-Library GPU Kernel Tests ===" << std::endl;
    std::cout << std::endl;

    // Initialize CUDA runtime explicitly
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "ERROR: No CUDA devices found or CUDA error: " << cudaGetErrorString(err)
                  << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Force CUDA context creation
    void* dummy;
    cudaMalloc(&dummy, 4);
    cudaFree(dummy);

    std::cout << "CUDA runtime initialized successfully" << std::endl;
    std::cout << std::endl;

    gap::solver::nr_kernels::kernel_tests::run_all_kernel_tests();

    std::cout << std::endl;
    std::cout << "=== Kernel Tests Complete ===" << std::endl;
    return 0;
#else
    std::cerr << "ERROR: Kernel tests not enabled at compile time." << std::endl;
    std::cerr << "Rebuild with: cmake -DGAP_ENABLE_KERNEL_TESTS=ON" << std::endl;
    return 1;
#endif
}
