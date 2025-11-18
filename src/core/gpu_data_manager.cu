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
    free_device();  // Clean up any existing allocation

    num_rows = rows;
    num_cols = cols;
    nnz = nonzeros;

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
    owns_device_memory = false;
}

void GPUSparseMatrix::sync_from_device() {
    if (!d_row_ptr || !d_col_idx || !d_values) {
        throw std::runtime_error("Cannot sync from device: device memory not allocated");
    }

    // Allocate host shadow if needed
    h_row_ptr.resize(num_rows + 1);
    h_col_idx.resize(nnz);
    h_values.resize(nnz);

    // Copy from device to host
    cudaMemcpy(h_row_ptr.data(), d_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), d_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // Copy complex values
    std::vector<cuDoubleComplex> temp(nnz);
    cudaMemcpy(temp.data(), d_values, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Convert cuDoubleComplex to Complex
    for (int i = 0; i < nnz; ++i) {
        h_values[i] = Complex(cuCreal(temp[i]), cuCimag(temp[i]));
    }

    has_host_shadow = true;
}

void GPUSparseMatrix::sync_to_device() {
    if (!d_row_ptr || !d_col_idx || !d_values) {
        throw std::runtime_error("Cannot sync to device: device memory not allocated");
    }

    if (!has_host_shadow) {
        throw std::runtime_error("Cannot sync to device: no host shadow available");
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
    // Sync from device if we don't have a host shadow
    if (!has_host_shadow) {
        sync_from_device();
    }

    // Create and return a SparseMatrix
    SparseMatrix matrix;
    matrix.num_rows = num_rows;
    matrix.num_cols = num_cols;
    matrix.nnz = nnz;
    matrix.row_ptr = h_row_ptr;
    matrix.col_idx = h_col_idx;
    matrix.values = h_values;

    return matrix;
}

// ============================================================================
// GPUPowerFlowData Implementation
// ============================================================================

void GPUPowerFlowData::allocate_device(int buses, int unknowns) {
    free_device();  // Clean up any existing allocation

    num_buses = buses;
    num_unknowns = unknowns;

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
    owns_device_memory = false;
}

void GPUPowerFlowData::sync_voltages_from_device() {
    if (!d_voltages) {
        throw std::runtime_error("Cannot sync voltages: device memory not allocated");
    }

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
    if (!d_mismatches) {
        throw std::runtime_error("Cannot sync mismatches: device memory not allocated");
    }

    h_mismatches.resize(num_unknowns);
    cudaMemcpy(h_mismatches.data(), d_mismatches, num_unknowns * sizeof(double),
               cudaMemcpyDeviceToHost);

    has_host_shadow = true;
}

void GPUPowerFlowData::sync_solution_from_device() {
    if (!d_solution) {
        throw std::runtime_error("Cannot sync solution: device memory not allocated");
    }

    h_solution.resize(num_unknowns);

    std::vector<cuDoubleComplex> temp(num_unknowns);
    cudaMemcpy(temp.data(), d_solution, num_unknowns * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_unknowns; ++i) {
        h_solution[i] = Complex(cuCreal(temp[i]), cuCimag(temp[i]));
    }

    has_host_shadow = true;
}

GPUPowerFlowData::StateSnapshot GPUPowerFlowData::get_state_snapshot() {
    StateSnapshot snapshot;

    // Sync all relevant data from device
    sync_voltages_from_device();
    sync_mismatches_from_device();

    snapshot.voltages = h_voltages;
    snapshot.mismatches = h_mismatches;

    // Calculate max mismatch
    snapshot.max_mismatch = 0.0;
    for (Float mismatch : h_mismatches) {
        snapshot.max_mismatch = std::max(snapshot.max_mismatch, std::abs(mismatch));
    }

    snapshot.iteration = 0;  // Will be set by caller

    return snapshot;
}

}  // namespace gap
