#pragma once

#include <cuComplex.h>

namespace gap::solver::gpu_kernels {

/**
 * @brief Launch CUDA kernel to calculate current injections I = Y * V
 */
void launch_calculate_current_injections(int const* d_row_ptr, int const* d_col_idx,
                                         cuDoubleComplex const* d_y_values,
                                         cuDoubleComplex const* d_voltages,
                                         cuDoubleComplex* d_currents, int num_buses);

/**
 * @brief Launch CUDA kernel to calculate power mismatches
 */
void launch_calculate_power_mismatches(cuDoubleComplex const* d_voltages,
                                       cuDoubleComplex const* d_currents,
                                       cuDoubleComplex const* d_specified_power,
                                       int const* d_bus_types, double* d_mismatches, int num_buses,
                                       int* d_mismatch_indices);

/**
 * @brief Launch CUDA kernel to update voltages with correction
 */
void launch_update_voltages(cuDoubleComplex* d_voltages, cuDoubleComplex const* d_delta_v,
                            int const* d_bus_types, int const* d_voltage_indices,
                            double acceleration_factor, int num_buses);

/**
 * @brief Launch CUDA kernel to initialize voltages with flat start
 */
void launch_initialize_flat_start(cuDoubleComplex* d_voltages, int const* d_bus_types,
                                  double const* d_specified_magnitudes, int num_buses);

/**
 * @brief Launch CUDA kernel to build Jacobian matrix (dense format)
 *
 * Builds the full Newton-Raphson Jacobian matrix on GPU
 * Output is in row-major dense format for compatibility with dense solvers
 */
void launch_build_jacobian_dense(int const* d_y_row_ptr, int const* d_y_col_idx,
                                 cuDoubleComplex const* d_y_values,
                                 cuDoubleComplex const* d_voltages, cuDoubleComplex const* d_powers,
                                 int const* d_bus_types, int const* d_angle_var_idx,
                                 int const* d_mag_var_idx, double* d_jacobian, int num_buses,
                                 int num_vars);

/**
 * @brief Launch CUDA kernel to calculate power injections S = V * conj(I)
 */
void launch_calculate_power_injections(cuDoubleComplex const* d_voltages,
                                       cuDoubleComplex const* d_currents, cuDoubleComplex* d_powers,
                                       int num_buses);

}  // namespace gap::solver::gpu_kernels
