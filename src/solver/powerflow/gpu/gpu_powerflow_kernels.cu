#include <cuda_runtime.h>

#include <cuComplex.h>
#include <device_launch_parameters.h>

#include "gap/core/types.h"

namespace gap::solver::gpu_kernels {

/**
 * @brief CUDA kernel to calculate current injections: I = Y * V
 *
 * Uses CSR sparse matrix-vector multiplication
 * Each thread computes one row (one bus current injection)
 */
__global__ void calculate_current_injections_kernel(const int* __restrict__ row_ptr,
                                                    const int* __restrict__ col_idx,
                                                    const cuDoubleComplex* __restrict__ y_values,
                                                    const cuDoubleComplex* __restrict__ voltages,
                                                    cuDoubleComplex* __restrict__ currents,
                                                    int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        int row_start = row_ptr[bus_idx];
        int row_end = row_ptr[bus_idx + 1];

        for (int j = row_start; j < row_end; ++j) {
            int col = col_idx[j];
            cuDoubleComplex y_ij = y_values[j];
            cuDoubleComplex v_j = voltages[col];

            // sum += Y[i,j] * V[j]
            sum = cuCadd(sum, cuCmul(y_ij, v_j));
        }

        currents[bus_idx] = sum;
    }
}

/**
 * @brief CUDA kernel to calculate power injections: S = V * conj(I)
 *
 * Also calculates power mismatches for PQ and PV buses
 */
__global__ void calculate_power_mismatches_kernel(
    const cuDoubleComplex* __restrict__ voltages, const cuDoubleComplex* __restrict__ currents,
    const cuDoubleComplex* __restrict__ specified_power,
    const int* __restrict__ bus_types,  // 0=SLACK, 1=PQ, 2=PV
    double* __restrict__ mismatches, int num_buses,
    int* __restrict__ mismatch_indices  // Maps bus -> mismatch vector index
) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        int bus_type = bus_types[bus_idx];

        // Skip slack bus
        if (bus_type == 0) return;

        // Calculate S_calc = V * conj(I)
        cuDoubleComplex v = voltages[bus_idx];
        cuDoubleComplex i_conj = cuConj(currents[bus_idx]);
        cuDoubleComplex s_calc = cuCmul(v, i_conj);

        // Get specified power
        cuDoubleComplex s_spec = specified_power[bus_idx];

        // Get mismatch index
        int mismatch_idx = mismatch_indices[bus_idx];

        if (bus_type == 1) {  // PQ bus
            // P mismatch
            mismatches[mismatch_idx] = cuCreal(s_spec) - cuCreal(s_calc);
            // Q mismatch
            mismatches[mismatch_idx + 1] = cuCimag(s_spec) - cuCimag(s_calc);
        } else if (bus_type == 2) {  // PV bus
            // Only P mismatch (Q is not specified)
            mismatches[mismatch_idx] = cuCreal(s_spec) - cuCreal(s_calc);
        }
    }
}

/**
 * @brief CUDA kernel to compute partial derivatives for Jacobian matrix
 *
 * Calculates dS/dV for Newton-Raphson Jacobian
 * This is a simplified version - full implementation needs careful handling
 * of the sparsity pattern
 */
__global__ void calculate_jacobian_diagonal_kernel(
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const cuDoubleComplex* __restrict__ y_values, const cuDoubleComplex* __restrict__ voltages,
    const cuDoubleComplex* __restrict__ currents, const int* __restrict__ bus_types,
    cuDoubleComplex* __restrict__ jacobian_diag, int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        int bus_type = bus_types[bus_idx];

        // Skip slack bus
        if (bus_type == 0) {
            jacobian_diag[bus_idx] = make_cuDoubleComplex(1.0, 0.0);
            return;
        }

        cuDoubleComplex v_i = voltages[bus_idx];
        cuDoubleComplex i_i = currents[bus_idx];

        // Get Y_ii (diagonal element)
        cuDoubleComplex y_ii = make_cuDoubleComplex(0.0, 0.0);
        int row_start = row_ptr[bus_idx];
        int row_end = row_ptr[bus_idx + 1];

        for (int j = row_start; j < row_end; ++j) {
            if (col_idx[j] == bus_idx) {
                y_ii = y_values[j];
                break;
            }
        }

        // dS/dV diagonal: dS_i/dV_i = Y_ii * V_i + I_i
        jacobian_diag[bus_idx] = cuCadd(cuCmul(y_ii, v_i), i_i);
    }
}

/**
 * @brief CUDA kernel to calculate off-diagonal Jacobian elements
 *
 * For each non-zero Y_ij, calculate dS_i/dV_j = Y_ij * V_i
 */
__global__ void calculate_jacobian_offdiag_kernel(const int* __restrict__ row_ptr,
                                                  const int* __restrict__ col_idx,
                                                  const cuDoubleComplex* __restrict__ y_values,
                                                  const cuDoubleComplex* __restrict__ voltages,
                                                  cuDoubleComplex* __restrict__ jacobian_values,
                                                  int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        cuDoubleComplex v_i = voltages[bus_idx];

        int row_start = row_ptr[bus_idx];
        int row_end = row_ptr[bus_idx + 1];

        for (int j = row_start; j < row_end; ++j) {
            int col = col_idx[j];

            if (col != bus_idx) {  // Off-diagonal
                cuDoubleComplex y_ij = y_values[j];
                // dS_i/dV_j = Y_ij * V_i
                jacobian_values[j] = cuCmul(y_ij, v_i);
            } else {  // Will be handled by diagonal kernel
                jacobian_values[j] = make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }
}

/**
 * @brief CUDA kernel to update voltages: V_new = V_old + alpha * deltaV
 */
__global__ void update_voltages_kernel(
    cuDoubleComplex* __restrict__ voltages, const cuDoubleComplex* __restrict__ delta_v,
    const int* __restrict__ bus_types,
    const int* __restrict__ voltage_indices,  // Maps bus -> delta_v index
    double acceleration_factor, int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        int bus_type = bus_types[bus_idx];

        // Skip slack bus (voltage is fixed)
        if (bus_type == 0) return;

        int delta_idx = voltage_indices[bus_idx];
        if (delta_idx < 0) return;  // Invalid index

        cuDoubleComplex delta = delta_v[delta_idx];
        cuDoubleComplex v_old = voltages[bus_idx];

        // Scale by acceleration factor
        delta = make_cuDoubleComplex(cuCreal(delta) * acceleration_factor,
                                     cuCimag(delta) * acceleration_factor);

        // Update voltage
        voltages[bus_idx] = cuCadd(v_old, delta);

        // For PV buses, maintain voltage magnitude
        if (bus_type == 2) {
            cuDoubleComplex v_new = voltages[bus_idx];
            double mag = cuCabs(v_new);
            double target_mag = cuCabs(v_old);  // Keep original magnitude

            if (mag > 1e-10) {
                double scale = target_mag / mag;
                voltages[bus_idx] =
                    make_cuDoubleComplex(cuCreal(v_new) * scale, cuCimag(v_new) * scale);
            }
        }
    }
}

/**
 * @brief CUDA kernel to find maximum mismatch (for convergence check)
 */
__global__ void reduce_max_mismatch_kernel(const double* __restrict__ mismatches,
                                           double* __restrict__ partial_max, int num_mismatches) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < num_mismatches) ? fabs(mismatches[idx]) : 0.0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Initialize voltage vector with flat start
 */
__global__ void initialize_voltages_flat_start_kernel(
    cuDoubleComplex* __restrict__ voltages, const int* __restrict__ bus_types,
    const double* __restrict__ specified_magnitudes, int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        int bus_type = bus_types[bus_idx];

        if (bus_type == 0 || bus_type == 2) {  // SLACK or PV
            // Use specified magnitude
            double mag = specified_magnitudes[bus_idx];
            voltages[bus_idx] = make_cuDoubleComplex(mag, 0.0);
        } else {  // PQ bus
            // Flat start: 1.0 + 0j
            voltages[bus_idx] = make_cuDoubleComplex(1.0, 0.0);
        }
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void launch_calculate_current_injections(const int* d_row_ptr, const int* d_col_idx,
                                         const cuDoubleComplex* d_y_values,
                                         const cuDoubleComplex* d_voltages,
                                         cuDoubleComplex* d_currents, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    calculate_current_injections_kernel<<<numBlocks, blockSize>>>(
        d_row_ptr, d_col_idx, d_y_values, d_voltages, d_currents, num_buses);

    cudaDeviceSynchronize();
}

void launch_calculate_power_mismatches(const cuDoubleComplex* d_voltages,
                                       const cuDoubleComplex* d_currents,
                                       const cuDoubleComplex* d_specified_power,
                                       const int* d_bus_types, double* d_mismatches, int num_buses,
                                       int* d_mismatch_indices) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    calculate_power_mismatches_kernel<<<numBlocks, blockSize>>>(
        d_voltages, d_currents, d_specified_power, d_bus_types, d_mismatches, num_buses,
        d_mismatch_indices);

    cudaDeviceSynchronize();
}

void launch_update_voltages(cuDoubleComplex* d_voltages, const cuDoubleComplex* d_delta_v,
                            const int* d_bus_types, const int* d_voltage_indices,
                            double acceleration_factor, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    update_voltages_kernel<<<numBlocks, blockSize>>>(
        d_voltages, d_delta_v, d_bus_types, d_voltage_indices, acceleration_factor, num_buses);

    cudaDeviceSynchronize();
}

void launch_initialize_flat_start(cuDoubleComplex* d_voltages, const int* d_bus_types,
                                  const double* d_specified_magnitudes, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    initialize_voltages_flat_start_kernel<<<numBlocks, blockSize>>>(
        d_voltages, d_bus_types, d_specified_magnitudes, num_buses);

    cudaDeviceSynchronize();
}

}  // namespace gap::solver::gpu_kernels
