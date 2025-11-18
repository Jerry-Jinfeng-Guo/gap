#include <cuda_runtime.h>

#include <cuComplex.h>

#include <cstring>
#include <stdexcept>

#include "gap/core/types.h"

namespace gap {

// ============================================================================
// GPUSparseMatrix Implementation
// ============================================================================

void GPUSparseMatrix::allocate_device(int rows, int cols, int nonzeros) {
    // Free any existing allocations
    free_device();

    num_rows = rows;
    num_cols = cols;
    nnz = nonzeros;

    // Allocate device memory
    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nonzeros * sizeof(int));
    cudaMalloc(&d_values, nonzeros * sizeof(cuDoubleComplex));

    owns_device_memory = true;
}

void GPUSparseMatrix::free_device() {
    if (owns_device_memory) {
        if (d_row_ptr) cudaFree(d_row_ptr);
        if (d_col_idx) cudaFree(d_col_idx);
        if (d_values) cudaFree(d_values);
    }

    d_row_ptr = nullptr;
    d_col_idx = nullptr;
    d_values = nullptr;
    num_rows = 0;
    num_cols = 0;
    nnz = 0;
}

void GPUSparseMatrix::sync_from_device() {
    if (!d_row_ptr || !d_col_idx || !d_values) {
        return;  // No device data to sync
    }

    // Allocate host shadow if needed
    h_row_ptr.resize(num_rows + 1);
    h_col_idx.resize(nnz);
    h_values.resize(nnz);

    // Copy from device to host
    cudaMemcpy(h_row_ptr.data(), d_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), d_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert cuDoubleComplex to Complex
    std::vector<cuDoubleComplex> temp(nnz);
    cudaMemcpy(temp.data(), d_values, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nnz; ++i) {
        h_values[i] = Complex(cuCreal(temp[i]), cuCimag(temp[i]));
    }

    has_host_shadow = true;
}

void GPUSparseMatrix::sync_to_device() {
    if (!has_host_shadow) {
        throw std::runtime_error("No host shadow data to sync to device");
    }

    // Allocate device if needed
    if (!d_row_ptr) {
        allocate_device(num_rows, num_cols, nnz);
    }

    // Copy from host to device
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);

    // Convert Complex to cuDoubleComplex
    std::vector<cuDoubleComplex> temp(nnz);
    for (int i = 0; i < nnz; ++i) {
        temp[i] = make_cuDoubleComplex(h_values[i].real(), h_values[i].imag());
    }

    cudaMemcpy(d_values, temp.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
}

SparseMatrix GPUSparseMatrix::get_host_copy() {
    sync_from_device();

    SparseMatrix result;
    result.num_rows = num_rows;
    result.num_cols = num_cols;
    result.nnz = nnz;
    result.row_ptr = h_row_ptr;
    result.col_idx = h_col_idx;
    result.values = h_values;

    return result;
}

// ============================================================================
// GPUPowerFlowData Implementation
// ============================================================================

void GPUPowerFlowData::allocate_device(int buses, int unknowns) {
    // Free any existing allocations
    free_device();

    num_buses = buses;
    num_unknowns = unknowns;

    // Allocate device memory
    cudaMalloc(&d_voltages, buses * sizeof(cuDoubleComplex));
    cudaMalloc(&d_power_injections, buses * sizeof(cuDoubleComplex));
    cudaMalloc(&d_mismatches, unknowns * sizeof(double));
    cudaMalloc(&d_rhs, unknowns * sizeof(cuDoubleComplex));
    cudaMalloc(&d_solution, unknowns * sizeof(cuDoubleComplex));
    cudaMalloc(&d_bus_types, buses * sizeof(int));

    owns_device_memory = true;
}

void GPUPowerFlowData::free_device() {
    if (owns_device_memory) {
        if (d_voltages) cudaFree(d_voltages);
        if (d_power_injections) cudaFree(d_power_injections);
        if (d_mismatches) cudaFree(d_mismatches);
        if (d_rhs) cudaFree(d_rhs);
        if (d_solution) cudaFree(d_solution);
        if (d_bus_types) cudaFree(d_bus_types);
    }

    d_voltages = nullptr;
    d_power_injections = nullptr;
    d_mismatches = nullptr;
    d_rhs = nullptr;
    d_solution = nullptr;
    d_bus_types = nullptr;

    num_buses = 0;
    num_unknowns = 0;
}

void GPUPowerFlowData::sync_voltages_from_device() {
    if (!d_voltages) return;

    h_voltages.resize(num_buses);
    std::vector<cuDoubleComplex> temp(num_buses);
    cudaMemcpy(temp.data(), d_voltages, num_buses * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_buses; ++i) {
        h_voltages[i] = Complex(cuCreal(temp[i]), cuCimag(temp[i]));
    }

    has_host_shadow = true;
}

void GPUPowerFlowData::sync_mismatches_from_device() {
    if (!d_mismatches) return;

    h_mismatches.resize(num_unknowns);
    cudaMemcpy(h_mismatches.data(), d_mismatches, num_unknowns * sizeof(double),
               cudaMemcpyDeviceToHost);
}

void GPUPowerFlowData::sync_solution_from_device() {
    if (!d_solution) return;

    h_solution.resize(num_unknowns);
    std::vector<cuDoubleComplex> temp(num_unknowns);
    cudaMemcpy(temp.data(), d_solution, num_unknowns * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_unknowns; ++i) {
        h_solution[i] = Complex(cuCreal(temp[i]), cuCimag(temp[i]));
    }
}

GPUPowerFlowData::StateSnapshot GPUPowerFlowData::get_state_snapshot() {
    sync_voltages_from_device();
    sync_mismatches_from_device();

    StateSnapshot snapshot;
    snapshot.voltages = h_voltages;
    snapshot.mismatches = h_mismatches;

    // Calculate max mismatch
    snapshot.max_mismatch = 0.0;
    for (Float m : h_mismatches) {
        snapshot.max_mismatch = std::max(snapshot.max_mismatch, std::abs(m));
    }
    snapshot.iteration = 0;  // Caller should set this

    return snapshot;
}

}  // namespace gap
