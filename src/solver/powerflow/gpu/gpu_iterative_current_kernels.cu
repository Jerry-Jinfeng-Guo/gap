#include <cuda_runtime.h>

#include <cuComplex.h>
#include <device_launch_parameters.h>

#include "gap/core/types.h"

namespace gap::solver::ic_kernels {

// ============================================================================
// CUDA Kernels for Iterative Current Power Flow Solver
// ============================================================================

/**
 * @brief Initialize voltages to flat start (1.0 + 0.0j for all buses)
 */
__global__ void initialize_voltages_flat_kernel(cuDoubleComplex* d_voltages, int num_buses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_buses) {
        d_voltages[idx] = make_cuDoubleComplex(1.0, 0.0);
    }
}

/**
 * @brief Copy current voltages to old voltages (for convergence checking)
 */
__global__ void copy_voltages_kernel(cuDoubleComplex const* __restrict__ src,
                                     cuDoubleComplex* __restrict__ dst, int num_buses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_buses) {
        dst[idx] = src[idx];
    }
}

/**
 * @brief Calculate currents using sparse matrix-vector multiplication: I = Y * V
 *
 * Each thread calculates one row (one bus current)
 */
__global__ void calculate_currents_kernel(cuDoubleComplex const* __restrict__ ybus_values,
                                          int const* __restrict__ ybus_col_idx,
                                          int const* __restrict__ ybus_row_ptr,
                                          cuDoubleComplex const* __restrict__ voltages,
                                          cuDoubleComplex* __restrict__ currents, int num_buses) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_buses) {
        // Get the start and end indices for this row in CSR format
        int row_start = ybus_row_ptr[row];
        int row_end = ybus_row_ptr[row + 1];

        // Accumulate: I[row] = sum(Y[row,col] * V[col])
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = ybus_col_idx[idx];
            cuDoubleComplex y_val = ybus_values[idx];
            cuDoubleComplex v_val = voltages[col];

            // Complex multiplication: sum += y_val * v_val
            sum = cuCadd(sum, cuCmul(y_val, v_val));
        }

        currents[row] = sum;
    }
}

/**
 * @brief Update voltages using Iterative Current method
 *
 * V_new = V_old + (I_specified - I_calculated) / Y_diagonal
 */
__global__ void update_voltages_kernel(cuDoubleComplex const* __restrict__ ybus_values,
                                       int const* __restrict__ ybus_col_idx,
                                       int const* __restrict__ ybus_row_ptr,
                                       cuDoubleComplex const* __restrict__ i_specified,
                                       cuDoubleComplex const* __restrict__ currents,
                                       cuDoubleComplex* __restrict__ voltages, int num_buses) {
    int bus = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus < num_buses) {
        // Find diagonal element of Y-bus for this bus
        cuDoubleComplex y_diag = make_cuDoubleComplex(1.0, 0.0);  // Default

        int row_start = ybus_row_ptr[bus];
        int row_end = ybus_row_ptr[bus + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = ybus_col_idx[idx];
            if (col == bus) {
                y_diag = ybus_values[idx];
                break;
            }
        }

        // Iterative Current formula: V_new = V_old + (I_specified - I_calculated) / Y_diagonal
        cuDoubleComplex i_spec = i_specified[bus];
        cuDoubleComplex i_calc = currents[bus];
        cuDoubleComplex i_mismatch = cuCsub(i_spec, i_calc);
        cuDoubleComplex delta_v = cuCdiv(i_mismatch, y_diag);

        // Update voltage: V_new = V_old + delta_V
        voltages[bus] = cuCadd(voltages[bus], delta_v);
    }
}

/**
 * @brief Enforce slack bus constraint
 *
 * Set slack bus voltage to fixed reference value (typically 1.0 + 0j)
 */
__global__ void enforce_slack_bus_kernel(cuDoubleComplex* __restrict__ voltages, int slack_bus_idx,
                                         cuDoubleComplex slack_voltage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only first thread needs to do this
    if (idx == 0) {
        voltages[slack_bus_idx] = slack_voltage;
    }
}

/**
 * @brief Calculate max voltage change for convergence checking
 */
__global__ void calculate_voltage_change_kernel(cuDoubleComplex const* __restrict__ old_voltages,
                                                cuDoubleComplex const* __restrict__ new_voltages,
                                                double* __restrict__ max_change, int num_buses) {
    // Shared memory for block-level reduction
    __shared__ double shared_max[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread calculates its voltage magnitude change
    double local_max = 0.0;
    if (idx < num_buses) {
        double old_mag = cuCabs(old_voltages[idx]);
        double new_mag = cuCabs(new_voltages[idx]);
        local_max = fabs(new_mag - old_mag);
    }

    // Store in shared memory
    shared_max[tid] = local_max;
    __syncthreads();

    // Block-level reduction to find max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // First thread writes block result
    if (tid == 0) {
        atomicMax((unsigned long long*)max_change, __double_as_longlong(shared_max[0]));
    }
}

/**
 * @brief Simple test kernel to verify CUDA kernel launches work
 */
__global__ void test_kernel(double* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] = d_data[idx] * 2.0;  // Simple operation
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void launch_initialize_voltages_flat(cuDoubleComplex* d_voltages, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    initialize_voltages_flat_kernel<<<numBlocks, blockSize>>>(d_voltages, num_buses);

    cudaDeviceSynchronize();
}

void launch_copy_voltages(cuDoubleComplex const* d_src, cuDoubleComplex* d_dst, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    copy_voltages_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, num_buses);

    cudaDeviceSynchronize();
}

void launch_calculate_currents(cuDoubleComplex const* d_ybus_values, int const* d_ybus_col_idx,
                               int const* d_ybus_row_ptr, cuDoubleComplex const* d_voltages,
                               cuDoubleComplex* d_currents, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    calculate_currents_kernel<<<numBlocks, blockSize>>>(
        d_ybus_values, d_ybus_col_idx, d_ybus_row_ptr, d_voltages, d_currents, num_buses);

    cudaDeviceSynchronize();
}

void launch_update_voltages(cuDoubleComplex const* d_ybus_values, int const* d_ybus_col_idx,
                            int const* d_ybus_row_ptr, cuDoubleComplex const* d_i_specified,
                            cuDoubleComplex const* d_currents, cuDoubleComplex* d_voltages,
                            int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    update_voltages_kernel<<<numBlocks, blockSize>>>(d_ybus_values, d_ybus_col_idx, d_ybus_row_ptr,
                                                     d_i_specified, d_currents, d_voltages,
                                                     num_buses);

    cudaDeviceSynchronize();
}

void launch_enforce_slack_bus(cuDoubleComplex* d_voltages, int slack_bus_idx,
                              cuDoubleComplex slack_voltage) {
    enforce_slack_bus_kernel<<<1, 1>>>(d_voltages, slack_bus_idx, slack_voltage);

    cudaDeviceSynchronize();
}

void launch_calculate_voltage_change(cuDoubleComplex const* d_old_voltages,
                                     cuDoubleComplex const* d_new_voltages, double* d_max_change,
                                     int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * sizeof(double);

    calculate_voltage_change_kernel<<<numBlocks, blockSize, sharedMemSize>>>(
        d_old_voltages, d_new_voltages, d_max_change, num_buses);

    cudaDeviceSynchronize();
}

void launch_test_kernel(double* d_data, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    test_kernel<<<numBlocks, blockSize>>>(d_data, n);

    cudaDeviceSynchronize();
}

}  // namespace gap::solver::ic_kernels
