#pragma once

#include <cuComplex.h>

namespace gap::solver::ic_kernels {

/**
 * @brief Launch CUDA kernel to initialize voltages with flat start
 */
void launch_initialize_voltages_flat(cuDoubleComplex* d_voltages, int num_buses);

/**
 * @brief Launch CUDA kernel to copy voltages
 */
void launch_copy_voltages(cuDoubleComplex const* d_src, cuDoubleComplex* d_dst, int num_buses);

/**
 * @brief Launch CUDA kernel to calculate currents: I = Y * V
 */
void launch_calculate_currents(cuDoubleComplex const* d_ybus_values, int const* d_ybus_col_idx,
                               int const* d_ybus_row_ptr, cuDoubleComplex const* d_voltages,
                               cuDoubleComplex* d_currents, int num_buses);

/**
 * @brief Launch CUDA kernel to update voltages using Iterative Current method
 */
void launch_update_voltages(cuDoubleComplex const* d_ybus_values, int const* d_ybus_col_idx,
                            int const* d_ybus_row_ptr, cuDoubleComplex const* d_i_specified,
                            cuDoubleComplex const* d_currents, cuDoubleComplex* d_voltages,
                            int num_buses);

/**
 * @brief Launch CUDA kernel to enforce slack bus constraint
 */
void launch_enforce_slack_bus(cuDoubleComplex* d_voltages, int slack_bus_idx,
                              cuDoubleComplex slack_voltage);

/**
 * @brief Launch CUDA kernel to calculate max voltage change for convergence
 */
void launch_calculate_voltage_change(cuDoubleComplex const* d_old_voltages,
                                     cuDoubleComplex const* d_new_voltages, double* d_max_change,
                                     int num_buses);

/**
 * @brief Launch test kernel for verification
 */
void launch_test_kernel(double* d_data, int n);

}  // namespace gap::solver::ic_kernels
