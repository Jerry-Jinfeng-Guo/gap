#include <cuda_runtime.h>

#include <cuComplex.h>
#include <device_launch_parameters.h>

#include <stdexcept>

#include "gap/core/types.h"

namespace gap::solver::nr_kernels {

/**
 * @brief CUDA kernel to calculate current injections: I = Y * V
 *
 * Uses CSR sparse matrix-vector multiplication
 * Each thread computes one row (one bus current injection)
 */
__global__ void calculate_current_injections_kernel(int const* __restrict__ row_ptr,
                                                    int const* __restrict__ col_idx,
                                                    cuDoubleComplex const* __restrict__ y_values,
                                                    cuDoubleComplex const* __restrict__ voltages,
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
 * Supports ZIP load model:
 * - const_pq (0): S(V) = S_rated (constant power)
 * - const_y (1):  S(V) = S_rated * |V|^2 (constant impedance)
 * - const_i (2):  S(V) = S_rated * |V| (constant current)
 */
__global__ void calculate_power_mismatches_kernel(
    cuDoubleComplex const* __restrict__ voltages, cuDoubleComplex const* __restrict__ currents,
    cuDoubleComplex const* __restrict__ specified_power_rated,  // S_rated (at nominal voltage)
    int const* __restrict__ bus_types,                          // 0=PQ, 1=PV, 2=SLACK
    int const* __restrict__ load_types,                         // 0=const_pq, 1=const_y, 2=const_i
    double* __restrict__ mismatches, int num_buses,
    int* __restrict__ mismatch_indices  // Maps bus -> mismatch vector index
) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        int bus_type = bus_types[bus_idx];

        // Skip slack bus (BusType::SLACK = 2)
        if (bus_type == 2) return;

        // Calculate S_calc = V * conj(I)
        cuDoubleComplex v = voltages[bus_idx];
        cuDoubleComplex i_conj = cuConj(currents[bus_idx]);
        cuDoubleComplex s_calc = cuCmul(v, i_conj);

        // Get rated power and apply ZIP model scaling
        cuDoubleComplex s_rated = specified_power_rated[bus_idx];
        cuDoubleComplex s_spec = s_rated;  // Default: const_pq

        // Apply ZIP load model scaling based on voltage
        int load_type = load_types[bus_idx];
        double v_mag = cuCabs(v);

        switch (load_type) {
            case 0:  // const_pq: S(V) = S_rated (no scaling)
                break;
            case 1:  // const_y: S(V) = S_rated * |V|^2 (constant impedance)
                s_spec = make_cuDoubleComplex(cuCreal(s_rated) * v_mag * v_mag,
                                              cuCimag(s_rated) * v_mag * v_mag);
                break;
            case 2:  // const_i: S(V) = S_rated * |V| (constant current)
                s_spec = make_cuDoubleComplex(cuCreal(s_rated) * v_mag, cuCimag(s_rated) * v_mag);
                break;
        }

        // Get mismatch index
        int mismatch_idx = mismatch_indices[bus_idx];

        if (bus_type == 0) {  // PQ bus (BusType::PQ = 0)
            // P mismatch: ΔP = P_calculated - P_specified (Newton-Raphson convention)
            mismatches[mismatch_idx] = cuCreal(s_calc) - cuCreal(s_spec);
            // Q mismatch: ΔQ = Q_calculated - Q_specified
            mismatches[mismatch_idx + 1] = cuCimag(s_calc) - cuCimag(s_spec);
        } else if (bus_type == 1) {  // PV bus (BusType::PV = 1)
            // Only P mismatch (Q is not specified)
            mismatches[mismatch_idx] = cuCreal(s_calc) - cuCreal(s_spec);
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
    int const* __restrict__ row_ptr, int const* __restrict__ col_idx,
    cuDoubleComplex const* __restrict__ y_values, cuDoubleComplex const* __restrict__ voltages,
    cuDoubleComplex const* __restrict__ currents, int const* __restrict__ bus_types,
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
 * @brief CUDA kernel to build full Jacobian matrix for Newton-Raphson power flow
 *
 * Jacobian structure:
 * J = [ ∂P/∂θ   ∂P/∂|V| ]
 *     [ ∂Q/∂θ   ∂Q/∂|V| ]
 *
 * Uses dense matrix format for simplicity with cuDSS solver
 * Each thread computes one row of the Jacobian
 */
__global__ void build_jacobian_dense_kernel(
    int const* __restrict__ y_row_ptr, int const* __restrict__ y_col_idx,
    cuDoubleComplex const* __restrict__ y_values, cuDoubleComplex const* __restrict__ voltages,
    cuDoubleComplex const* __restrict__ powers,  // Calculated S = V * conj(I)
    int const* __restrict__ bus_types,
    int const* __restrict__ angle_var_idx,  // Maps bus_id -> angle variable index (-1 if slack)
    int const* __restrict__ mag_var_idx,  // Maps bus_id -> magnitude variable index (-1 if not PQ)
    double* __restrict__ jacobian,        // Output: Dense Jacobian matrix (row-major)
    int num_buses, int num_vars) {
    int eq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eq_idx >= num_vars) return;

    // Determine which bus and equation type
    int bus_i = -1;
    bool is_q_equation = false;
    int current_eq = 0;

    for (int i = 0; i < num_buses; ++i) {
        if (bus_types[i] == 2) continue;  // Skip slack bus (SLACK=2)

        if (current_eq == eq_idx) {
            bus_i = i;
            break;
        }
        current_eq++;

        if (bus_types[i] == 0) {  // PQ bus (PQ=0) - add Q equation
            if (current_eq == eq_idx) {
                bus_i = i;
                is_q_equation = true;
                break;
            }
            current_eq++;
        }
    }

    if (bus_i < 0 || bus_i >= num_buses) return;

    cuDoubleComplex Vi = voltages[bus_i];
    double Vi_mag = cuCabs(Vi);
    if (Vi_mag < 1e-9) Vi_mag = 1.0;
    double theta_i = atan2(cuCimag(Vi), cuCreal(Vi));

    cuDoubleComplex Si = powers[bus_i];
    double Pi = cuCreal(Si);
    double Qi = cuCimag(Si);

    // Get Y_ii
    double G_ii = 0.0, B_ii = 0.0;
    int row_start = y_row_ptr[bus_i];
    int row_end = y_row_ptr[bus_i + 1];

    for (int idx = row_start; idx < row_end; ++idx) {
        if (y_col_idx[idx] == bus_i) {
            G_ii = cuCreal(y_values[idx]);
            B_ii = cuCimag(y_values[idx]);
            break;
        }
    }

    double* jac_row = &jacobian[eq_idx * num_vars];

    if (!is_q_equation) {
        // P equation
        for (int j = 0; j < num_buses; ++j) {
            int theta_j_idx = angle_var_idx[j];
            if (theta_j_idx >= 0) {
                double deriv;
                if (j == bus_i) {
                    deriv = -Qi - B_ii * Vi_mag * Vi_mag;
                } else {
                    cuDoubleComplex Vj = voltages[j];
                    double Vj_mag = cuCabs(Vj);
                    double theta_j = atan2(cuCimag(Vj), cuCreal(Vj));
                    double theta_ij = theta_i - theta_j;

                    double G_ij = 0.0, B_ij = 0.0;
                    for (int idx = row_start; idx < row_end; ++idx) {
                        if (y_col_idx[idx] == j) {
                            G_ij = cuCreal(y_values[idx]);
                            B_ij = cuCimag(y_values[idx]);
                            break;
                        }
                    }

                    deriv = Vi_mag * Vj_mag * (G_ij * sin(theta_ij) - B_ij * cos(theta_ij));
                }
                jac_row[theta_j_idx] = deriv;
            }
        }

        for (int j = 0; j < num_buses; ++j) {
            int V_j_idx = mag_var_idx[j];
            if (V_j_idx >= 0) {
                double deriv;
                if (j == bus_i) {
                    deriv = (Pi / Vi_mag) + G_ii * Vi_mag;
                } else {
                    cuDoubleComplex Vj = voltages[j];
                    double theta_j = atan2(cuCimag(Vj), cuCreal(Vj));
                    double theta_ij = theta_i - theta_j;

                    double G_ij = 0.0, B_ij = 0.0;
                    for (int idx = row_start; idx < row_end; ++idx) {
                        if (y_col_idx[idx] == j) {
                            G_ij = cuCreal(y_values[idx]);
                            B_ij = cuCimag(y_values[idx]);
                            break;
                        }
                    }

                    deriv = Vi_mag * (G_ij * cos(theta_ij) + B_ij * sin(theta_ij));
                }
                jac_row[V_j_idx] = deriv;
            }
        }

    } else {
        // Q equation
        for (int j = 0; j < num_buses; ++j) {
            int theta_j_idx = angle_var_idx[j];
            if (theta_j_idx >= 0) {
                double deriv;
                if (j == bus_i) {
                    deriv = Pi - G_ii * Vi_mag * Vi_mag;
                } else {
                    cuDoubleComplex Vj = voltages[j];
                    double Vj_mag = cuCabs(Vj);
                    double theta_j = atan2(cuCimag(Vj), cuCreal(Vj));
                    double theta_ij = theta_i - theta_j;

                    double G_ij = 0.0, B_ij = 0.0;
                    for (int idx = row_start; idx < row_end; ++idx) {
                        if (y_col_idx[idx] == j) {
                            G_ij = cuCreal(y_values[idx]);
                            B_ij = cuCimag(y_values[idx]);
                            break;
                        }
                    }

                    deriv = -Vi_mag * Vj_mag * (G_ij * cos(theta_ij) + B_ij * sin(theta_ij));
                }
                jac_row[theta_j_idx] = deriv;
            }
        }

        for (int j = 0; j < num_buses; ++j) {
            int V_j_idx = mag_var_idx[j];
            if (V_j_idx >= 0) {
                double deriv;
                if (j == bus_i) {
                    deriv = (Qi / Vi_mag) - B_ii * Vi_mag;
                } else {
                    cuDoubleComplex Vj = voltages[j];
                    double theta_j = atan2(cuCimag(Vj), cuCreal(Vj));
                    double theta_ij = theta_i - theta_j;

                    double G_ij = 0.0, B_ij = 0.0;
                    for (int idx = row_start; idx < row_end; ++idx) {
                        if (y_col_idx[idx] == j) {
                            G_ij = cuCreal(y_values[idx]);
                            B_ij = cuCimag(y_values[idx]);
                            break;
                        }
                    }

                    deriv = Vi_mag * (G_ij * sin(theta_ij) - B_ij * cos(theta_ij));
                }
                jac_row[V_j_idx] = deriv;
            }
        }
    }
}

/**
 * @brief CUDA kernel to calculate off-diagonal Jacobian elements
 *
 * For each non-zero Y_ij, calculate dS_i/dV_j = Y_ij * V_i
 */
__global__ void calculate_jacobian_offdiag_kernel(int const* __restrict__ row_ptr,
                                                  int const* __restrict__ col_idx,
                                                  cuDoubleComplex const* __restrict__ y_values,
                                                  cuDoubleComplex const* __restrict__ voltages,
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
    cuDoubleComplex* __restrict__ voltages, cuDoubleComplex const* __restrict__ delta_v,
    int const* __restrict__ bus_types,
    int const* __restrict__ voltage_indices,  // Maps bus -> delta_v index
    double acceleration_factor, int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        int bus_type = bus_types[bus_idx];

        // Skip slack bus (voltage is fixed) - SLACK=2
        if (bus_type == 2) return;

        int delta_idx = voltage_indices[bus_idx];
        if (delta_idx < 0) return;  // Invalid index

        cuDoubleComplex delta = delta_v[delta_idx];
        cuDoubleComplex v_old = voltages[bus_idx];

        // Scale by acceleration factor
        delta = make_cuDoubleComplex(cuCreal(delta) * acceleration_factor,
                                     cuCimag(delta) * acceleration_factor);

        // Update voltage
        voltages[bus_idx] = cuCadd(v_old, delta);

        // For PV buses (PV=1), maintain voltage magnitude
        if (bus_type == 1) {
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
__global__ void reduce_max_mismatch_kernel(double const* __restrict__ mismatches,
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
    cuDoubleComplex* __restrict__ voltages, int const* __restrict__ bus_types,
    double const* __restrict__ specified_magnitudes, int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        int bus_type = bus_types[bus_idx];

        // BusType enum: PQ=0, PV=1, SLACK=2
        if (bus_type == 1 || bus_type == 2) {  // PV or SLACK bus
            // Use specified magnitude (V is specified for PV and SLACK)
            double mag = specified_magnitudes[bus_idx];
            voltages[bus_idx] = make_cuDoubleComplex(mag, 0.0);
        } else {  // PQ bus (bus_type == 0)
            // Flat start: 0.98 + 0j per-unit (slightly lower for load buses, matches CPU)
            voltages[bus_idx] = make_cuDoubleComplex(0.98, 0.0);
        }
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void launch_calculate_current_injections(int const* d_row_ptr, int const* d_col_idx,
                                         cuDoubleComplex const* d_y_values,
                                         cuDoubleComplex const* d_voltages,
                                         cuDoubleComplex* d_currents, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    calculate_current_injections_kernel<<<numBlocks, blockSize>>>(
        d_row_ptr, d_col_idx, d_y_values, d_voltages, d_currents, num_buses);

    cudaDeviceSynchronize();
}

void launch_calculate_power_mismatches(cuDoubleComplex const* d_voltages,
                                       cuDoubleComplex const* d_currents,
                                       cuDoubleComplex const* d_specified_power,
                                       int const* d_bus_types, int const* d_load_types,
                                       double* d_mismatches, int num_buses,
                                       int* d_mismatch_indices) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    calculate_power_mismatches_kernel<<<numBlocks, blockSize>>>(
        d_voltages, d_currents, d_specified_power, d_bus_types, d_load_types, d_mismatches,
        num_buses, d_mismatch_indices);

    cudaDeviceSynchronize();
}

void launch_update_voltages(cuDoubleComplex* d_voltages, cuDoubleComplex const* d_delta_v,
                            int const* d_bus_types, int const* d_voltage_indices,
                            double acceleration_factor, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    update_voltages_kernel<<<numBlocks, blockSize>>>(
        d_voltages, d_delta_v, d_bus_types, d_voltage_indices, acceleration_factor, num_buses);

    cudaDeviceSynchronize();
}

void launch_initialize_flat_start(cuDoubleComplex* d_voltages, int const* d_bus_types,
                                  double const* d_specified_magnitudes, int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    initialize_voltages_flat_start_kernel<<<numBlocks, blockSize>>>(
        d_voltages, d_bus_types, d_specified_magnitudes, num_buses);

    cudaDeviceSynchronize();
}

void launch_build_jacobian_dense(int const* d_y_row_ptr, int const* d_y_col_idx,
                                 cuDoubleComplex const* d_y_values,
                                 cuDoubleComplex const* d_voltages, cuDoubleComplex const* d_powers,
                                 int const* d_bus_types, int const* d_angle_var_idx,
                                 int const* d_mag_var_idx, double* d_jacobian, int num_buses,
                                 int num_vars) {
    int blockSize = 256;
    int numBlocks = (num_vars + blockSize - 1) / blockSize;

    build_jacobian_dense_kernel<<<numBlocks, blockSize>>>(
        d_y_row_ptr, d_y_col_idx, d_y_values, d_voltages, d_powers, d_bus_types, d_angle_var_idx,
        d_mag_var_idx, d_jacobian, num_buses, num_vars);

    cudaDeviceSynchronize();
}

/**
 * @brief Helper to calculate power injections: S = V * conj(I)
 * This is needed before building the Jacobian
 */
__global__ void calculate_power_injections_kernel(cuDoubleComplex const* __restrict__ voltages,
                                                  cuDoubleComplex const* __restrict__ currents,
                                                  cuDoubleComplex* __restrict__ powers,
                                                  int num_buses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_buses) {
        cuDoubleComplex v = voltages[idx];
        cuDoubleComplex i_conj = cuConj(currents[idx]);
        powers[idx] = cuCmul(v, i_conj);
    }
}

void launch_calculate_power_injections(cuDoubleComplex const* d_voltages,
                                       cuDoubleComplex const* d_currents, cuDoubleComplex* d_powers,
                                       int num_buses) {
    int blockSize = 256;
    int numBlocks = (num_buses + blockSize - 1) / blockSize;

    calculate_power_injections_kernel<<<numBlocks, blockSize>>>(d_voltages, d_currents, d_powers,
                                                                num_buses);

    cudaDeviceSynchronize();
}

// ===================================================================================

}  // namespace gap::solver::nr_kernels
