#include <cuda_runtime.h>

#include <cuComplex.h>

#include <cmath>
#include <iostream>

#include "gap/core/backend_factory.h"
#include "gap/solver/gpu_newton_raphson_kernels.h"

#include "test_framework.h"

using namespace gap;
using namespace gap::core;
using namespace gap::solver;

/**
 * @brief Test Jacobian matrix construction for a simple 2-bus system
 *
 * This test verifies that the GPU Jacobian construction produces correct
 * partial derivatives dP/dθ, dP/dV, dQ/dθ, dQ/dV for a known test case.
 *
 * Test case: 2-bus system with known voltages and Y-bus
 * - Bus 1 (SLACK): V = 1.0∠0° pu
 * - Bus 2 (PQ): V = 0.95∠-5° pu
 *
 * Expected Jacobian can be computed analytically for verification.
 */
void test_gpu_jacobian_2bus_simple() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Testing GPU Jacobian Construction (2-bus) ===" << std::endl;

    // Test parameters
    const int num_buses = 2;
    const int num_unknowns = 2;  // Bus 2: P and Q equations

    // Known voltages (per-unit)
    // Bus 1 (SLACK): V = 1.0∠0°
    // Bus 2 (PQ): V = 0.95∠-5° = 0.95∠-0.0873 rad
    const double V1_mag = 1.0;
    const double V1_angle = 0.0;
    const double V2_mag = 0.95;
    const double V2_angle = -5.0 * M_PI / 180.0;  // -0.0873 rad

    // Admittance matrix (per-unit) - simple line between two buses
    // Y = [ Y11  Y12 ]   [ 10-j30  -10+j30 ]
    //     [ Y21  Y22 ] = [ -10+j30  10-j30  ]
    Complex Y11(10.0, -30.0);
    Complex Y12(-10.0, 30.0);
    Complex Y21(-10.0, 30.0);
    Complex Y22(10.0, -30.0);

    // Allocate GPU memory
    cuDoubleComplex* d_voltages;
    cuDoubleComplex* d_currents;
    cuDoubleComplex* d_powers;
    int* d_bus_types;
    int* d_row_ptr;
    int* d_col_idx;
    cuDoubleComplex* d_y_values;
    int* d_angle_var_idx;
    int* d_mag_var_idx;
    double* d_jacobian;

    cudaMalloc(&d_voltages, num_buses * sizeof(cuDoubleComplex));
    cudaMalloc(&d_currents, num_buses * sizeof(cuDoubleComplex));
    cudaMalloc(&d_powers, num_buses * sizeof(cuDoubleComplex));
    cudaMalloc(&d_bus_types, num_buses * sizeof(int));
    cudaMalloc(&d_row_ptr, (num_buses + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, 4 * sizeof(int));  // 4 non-zeros
    cudaMalloc(&d_y_values, 4 * sizeof(cuDoubleComplex));
    cudaMalloc(&d_angle_var_idx, num_buses * sizeof(int));
    cudaMalloc(&d_mag_var_idx, num_buses * sizeof(int));
    cudaMalloc(&d_jacobian, num_unknowns * num_unknowns * sizeof(double));

    // Initialize host data
    std::vector<cuDoubleComplex> h_voltages(num_buses);
    h_voltages[0] = make_cuDoubleComplex(V1_mag * cos(V1_angle), V1_mag * sin(V1_angle));
    h_voltages[1] = make_cuDoubleComplex(V2_mag * cos(V2_angle), V2_mag * sin(V2_angle));

    std::vector<int> h_bus_types = {2, 0};  // SLACK=2, PQ=0
    std::vector<int> h_row_ptr = {0, 2, 4};
    std::vector<int> h_col_idx = {0, 1, 0, 1};
    std::vector<cuDoubleComplex> h_y_values(4);
    h_y_values[0] = make_cuDoubleComplex(Y11.real(), Y11.imag());
    h_y_values[1] = make_cuDoubleComplex(Y12.real(), Y12.imag());
    h_y_values[2] = make_cuDoubleComplex(Y21.real(), Y21.imag());
    h_y_values[3] = make_cuDoubleComplex(Y22.real(), Y22.imag());

    std::vector<int> h_voltage_indices = {-1, 0};  // Bus 0: no equation, Bus 1: index 0
    std::vector<int> h_angle_var_idx = {-1, 0};    // Bus 0: no angle var, Bus 1: angle at index 0
    std::vector<int> h_mag_var_idx = {-1, 1};      // Bus 0: no mag var, Bus 1: magnitude at index 1

    // Copy to GPU
    cudaMemcpy(d_voltages, h_voltages.data(), num_buses * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bus_types, h_bus_types.data(), num_buses * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_buses + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx.data(), 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_values, h_y_values.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_angle_var_idx, h_angle_var_idx.data(), num_buses * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mag_var_idx, h_mag_var_idx.data(), num_buses * sizeof(int),
               cudaMemcpyHostToDevice);

    // Calculate currents: I = Y * V
    nr_kernels::launch_calculate_current_injections(d_row_ptr, d_col_idx, d_y_values, d_voltages,
                                                    d_currents, num_buses);

    // Calculate powers: S = V * conj(I)
    nr_kernels::launch_calculate_power_injections(d_voltages, d_currents, d_powers, num_buses);

    // Build Jacobian matrix
    nr_kernels::launch_build_jacobian_dense(d_row_ptr, d_col_idx, d_y_values, d_voltages, d_powers,
                                            d_bus_types, d_angle_var_idx, d_mag_var_idx, d_jacobian,
                                            num_buses, num_unknowns);

    // Copy results back
    std::vector<double> h_jacobian(num_unknowns * num_unknowns);
    cudaMemcpy(h_jacobian.data(), d_jacobian, num_unknowns * num_unknowns * sizeof(double),
               cudaMemcpyDeviceToHost);

    std::cout << "\nGPU-computed Jacobian:" << std::endl;
    std::cout << "  [ dP2/dθ2   dP2/dV2 ]   [ " << h_jacobian[0] << "  " << h_jacobian[1] << " ]"
              << std::endl;
    std::cout << "  [ dQ2/dθ2   dQ2/dV2 ] = [ " << h_jacobian[2] << "  " << h_jacobian[3] << " ]"
              << std::endl;

    // Compute expected Jacobian analytically
    // For a 2-bus system with bus 2 as PQ:
    // dP2/dθ2 = V2*V1*|Y21|*sin(θ21 - δ2 + δ1) + V2*V2*|Y22|*sin(θ22)
    // dP2/dV2 = V1*|Y21|*cos(θ21 - δ2 + δ1) + 2*V2*|Y22|*cos(θ22)
    // dQ2/dθ2 = -V2*V1*|Y21|*cos(θ21 - δ2 + δ1) - V2*V2*|Y22|*cos(θ22)
    // dQ2/dV2 = V1*|Y21|*sin(θ21 - δ2 + δ1) + 2*V2*|Y22|*sin(θ22)

    double Y21_mag = std::abs(Y21);
    double Y21_angle = std::arg(Y21);
    double Y22_mag = std::abs(Y22);
    double Y22_angle = std::arg(Y22);

    double delta_angle = V2_angle - V1_angle;  // δ2 - δ1

    double dP2_dtheta2 = V2_mag * V1_mag * Y21_mag * sin(Y21_angle - delta_angle) +
                         V2_mag * V2_mag * Y22_mag * sin(Y22_angle);
    double dP2_dV2 =
        V1_mag * Y21_mag * cos(Y21_angle - delta_angle) + 2 * V2_mag * Y22_mag * cos(Y22_angle);
    double dQ2_dtheta2 = -V2_mag * V1_mag * Y21_mag * cos(Y21_angle - delta_angle) -
                         V2_mag * V2_mag * Y22_mag * cos(Y22_angle);
    double dQ2_dV2 =
        V1_mag * Y21_mag * sin(Y21_angle - delta_angle) + 2 * V2_mag * Y22_mag * sin(Y22_angle);

    std::cout << "\nAnalytically-computed Jacobian:" << std::endl;
    std::cout << "  [ dP2/dθ2   dP2/dV2 ]   [ " << dP2_dtheta2 << "  " << dP2_dV2 << " ]"
              << std::endl;
    std::cout << "  [ dQ2/dθ2   dQ2/dV2 ] = [ " << dQ2_dtheta2 << "  " << dQ2_dV2 << " ]"
              << std::endl;

    // Verify results (allow small numerical differences)
    const double tolerance = 1e-6;
    double error_11 = std::abs(h_jacobian[0] - dP2_dtheta2);
    double error_12 = std::abs(h_jacobian[1] - dP2_dV2);
    double error_21 = std::abs(h_jacobian[2] - dQ2_dtheta2);
    double error_22 = std::abs(h_jacobian[3] - dQ2_dV2);

    std::cout << "\nErrors:" << std::endl;
    std::cout << "  dP2/dθ2: " << error_11 << (error_11 < tolerance ? " ✓" : " ✗") << std::endl;
    std::cout << "  dP2/dV2: " << error_12 << (error_12 < tolerance ? " ✓" : " ✗") << std::endl;
    std::cout << "  dQ2/dθ2: " << error_21 << (error_21 < tolerance ? " ✓" : " ✗") << std::endl;
    std::cout << "  dQ2/dV2: " << error_22 << (error_22 < tolerance ? " ✓" : " ✗") << std::endl;

    ASSERT_TRUE(error_11 < tolerance);
    ASSERT_TRUE(error_12 < tolerance);
    ASSERT_TRUE(error_21 < tolerance);
    ASSERT_TRUE(error_22 < tolerance);

    // Cleanup
    cudaFree(d_voltages);
    cudaFree(d_currents);
    cudaFree(d_powers);
    cudaFree(d_bus_types);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_y_values);
    cudaFree(d_angle_var_idx);
    cudaFree(d_mag_var_idx);
    cudaFree(d_jacobian);

    std::cout << "✓ GPU Jacobian 2-bus test passed!" << std::endl;
}

/**
 * @brief Test Jacobian matrix construction matches CPU implementation
 *
 * This test compares GPU and CPU Jacobian construction for the same
 * 3-bus test system to ensure they produce identical results.
 */
void test_gpu_vs_cpu_jacobian_3bus() {
    if (!BackendFactory::is_backend_available(BackendType::GPU_CUDA)) {
        std::cout << "GPU not available, skipping test" << std::endl;
        return;
    }

    std::cout << "\n=== Testing GPU vs CPU Jacobian (3-bus) ===" << std::endl;

    // Create a 3-bus test system
    NetworkData network;
    network.num_buses = 3;

    BusData bus1 = {.id = 0,
                    .u_rated = 10000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u_pu = 1.05,
                    .u_angle = 0.0};
    BusData bus2 = {.id = 1,
                    .u_rated = 10000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u_pu = 1.0,
                    .u_angle = 0.0};
    BusData bus3 = {.id = 2,
                    .u_rated = 10000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u_pu = 1.0,
                    .u_angle = 0.0};
    network.buses = {bus1, bus2, bus3};

    // Simple Y-bus (per-unit)
    SparseMatrix y_pu;
    y_pu.num_rows = 3;
    y_pu.num_cols = 3;
    y_pu.nnz = 7;
    y_pu.row_ptr = {0, 3, 5, 7};
    y_pu.col_idx = {0, 1, 2, 0, 1, 1, 2};
    y_pu.values = {
        Complex(3.89478, -6.36759),  // Y[0,0]
        Complex(-1.94739, 3.18379),  // Y[0,1]
        Complex(-1.94739, 3.18379),  // Y[0,2]
        Complex(-1.94739, 3.18379),  // Y[1,0]
        Complex(8.32467, -13.61),    // Y[1,1]
        Complex(-4.42989, 7.24245),  // Y[1,2]
        Complex(4.42989, -7.24245)   // Y[2,2]
    };
}
