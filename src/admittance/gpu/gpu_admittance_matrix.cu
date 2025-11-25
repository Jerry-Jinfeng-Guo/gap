#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "gap/admittance/admittance_interface.h"
#include "gap/logging/logger.h"

// Device-side triplet structure for COO format
struct DeviceTriplet {
    int row;
    int col;
    cuDoubleComplex value;

    __device__ __host__ DeviceTriplet() : row(0), col(0), value(make_cuDoubleComplex(0.0, 0.0)) {}
    __device__ __host__ DeviceTriplet(int r, int c, cuDoubleComplex v) : row(r), col(c), value(v) {}
};

// Comparison functor for sorting triplets by (row, col)
struct TripletComparator {
    __device__ __host__ bool operator()(const DeviceTriplet& a, const DeviceTriplet& b) const {
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    }
};

// CUDA kernel: Compute branch admittances in parallel
__global__ void compute_branch_admittances_kernel(const gap::BranchData* branches, int num_branches,
                                                  const int* id_to_idx, int max_bus_id,
                                                  DeviceTriplet* triplets, int* valid_branches,
                                                  cuDoubleComplex* diagonal_accumulator) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_branches) return;

    const gap::BranchData& branch = branches[idx];

    // Check if branch is active
    if (!branch.status) {
        valid_branches[idx] = 0;
        return;
    }

    // Map bus IDs to indices
    int from_bus = -1, to_bus = -1;
    if (branch.from_bus >= 0 && branch.from_bus <= max_bus_id) {
        from_bus = id_to_idx[branch.from_bus];
    }
    if (branch.to_bus >= 0 && branch.to_bus <= max_bus_id) {
        to_bus = id_to_idx[branch.to_bus];
    }

    if (from_bus < 0 || to_bus < 0) {
        valid_branches[idx] = 0;
        return;
    }

    valid_branches[idx] = 1;

    // Calculate branch admittance: Y = 1/(R + jX)
    double z_magnitude_sq = branch.r1 * branch.r1 + branch.x1 * branch.x1;
    cuDoubleComplex series_admittance;

    if (z_magnitude_sq > 1e-12) {
        series_admittance =
            make_cuDoubleComplex(branch.r1 / z_magnitude_sq, -branch.x1 / z_magnitude_sq);
    } else {
        series_admittance = make_cuDoubleComplex(0.0, 0.0);
    }

    cuDoubleComplex branch_admittance =
        cuCadd(series_admittance, make_cuDoubleComplex(branch.g1, 0.0));
    cuDoubleComplex shunt_admittance = make_cuDoubleComplex(0.0, branch.b1 / 2.0);

    // Accumulate diagonal elements atomically
    cuDoubleComplex diagonal_contrib = cuCadd(branch_admittance, shunt_admittance);
    atomicAdd(&diagonal_accumulator[from_bus].x, diagonal_contrib.x);
    atomicAdd(&diagonal_accumulator[from_bus].y, diagonal_contrib.y);
    atomicAdd(&diagonal_accumulator[to_bus].x, diagonal_contrib.x);
    atomicAdd(&diagonal_accumulator[to_bus].y, diagonal_contrib.y);

    // Store off-diagonal triplets (negative series admittance)
    cuDoubleComplex neg_series = make_cuDoubleComplex(-series_admittance.x, -series_admittance.y);
    triplets[2 * idx] = DeviceTriplet(from_bus, to_bus, neg_series);
    triplets[2 * idx + 1] = DeviceTriplet(to_bus, from_bus, neg_series);
}

// CUDA kernel: Add shunt appliances to diagonal
__global__ void accumulate_shunt_appliances_kernel(const gap::ApplianceData* appliances,
                                                   int num_appliances, const int* id_to_idx,
                                                   int max_bus_id,
                                                   cuDoubleComplex* diagonal_accumulator) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_appliances) return;

    const gap::ApplianceData& appliance = appliances[idx];

    if (appliance.type == gap::ApplianceType::SHUNT) {
        if (appliance.node >= 0 && appliance.node <= max_bus_id) {
            int bus_idx = id_to_idx[appliance.node];
            if (bus_idx >= 0) {
                cuDoubleComplex shunt = make_cuDoubleComplex(appliance.g1, appliance.b1);
                atomicAdd(&diagonal_accumulator[bus_idx].x, shunt.x);
                atomicAdd(&diagonal_accumulator[bus_idx].y, shunt.y);
            }
        }
    }
}

// Device helper function: Find CSR element index for (row, col)
__device__ int find_csr_element(const int* row_ptr, const int* col_idx, int row, int col) {
    int start = row_ptr[row];
    int end = row_ptr[row + 1];

    for (int idx = start; idx < end; ++idx) {
        if (col_idx[idx] == col) {
            return idx;
        }
    }
    return -1;  // Element not found
}

// CUDA kernel: Update admittance matrix elements in parallel
__global__ void update_admittance_matrix_kernel(const gap::BranchData* branch_changes,
                                                int num_changes, const int* row_ptr,
                                                const int* col_idx, cuDoubleComplex* values,
                                                int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_changes) return;

    const gap::BranchData& branch = branch_changes[idx];
    int from_bus = branch.from_bus;
    int to_bus = branch.to_bus;

    // Validate bus indices
    if (from_bus < 0 || from_bus >= num_rows || to_bus < 0 || to_bus >= num_rows) {
        return;
    }

    // Calculate branch admittance change
    double z_magnitude_sq = branch.r1 * branch.r1 + branch.x1 * branch.x1;
    cuDoubleComplex series_admittance_change = make_cuDoubleComplex(0.0, 0.0);

    if (z_magnitude_sq > 1e-12) {
        series_admittance_change =
            make_cuDoubleComplex(branch.r1 / z_magnitude_sq, -branch.x1 / z_magnitude_sq);
    }

    cuDoubleComplex total_admittance_change =
        cuCadd(series_admittance_change, make_cuDoubleComplex(branch.g1, branch.b1));

    // Handle branch status (out of service = subtract admittance)
    if (!branch.status) {
        total_admittance_change =
            make_cuDoubleComplex(-total_admittance_change.x, -total_admittance_change.y);
        series_admittance_change =
            make_cuDoubleComplex(-series_admittance_change.x, -series_admittance_change.y);
    }

    cuDoubleComplex shunt_change = make_cuDoubleComplex(0.0, branch.b1 / 2.0);
    if (!branch.status) {
        shunt_change = make_cuDoubleComplex(-shunt_change.x, -shunt_change.y);
    }

    cuDoubleComplex diagonal_change = cuCadd(total_admittance_change, shunt_change);

    // Update diagonal element (from_bus, from_bus)
    int diag_from_idx = find_csr_element(row_ptr, col_idx, from_bus, from_bus);
    if (diag_from_idx >= 0) {
        atomicAdd(&values[diag_from_idx].x, diagonal_change.x);
        atomicAdd(&values[diag_from_idx].y, diagonal_change.y);
    }

    // Update diagonal element (to_bus, to_bus)
    int diag_to_idx = find_csr_element(row_ptr, col_idx, to_bus, to_bus);
    if (diag_to_idx >= 0) {
        atomicAdd(&values[diag_to_idx].x, diagonal_change.x);
        atomicAdd(&values[diag_to_idx].y, diagonal_change.y);
    }

    // Update off-diagonal element (from_bus, to_bus)
    int off_diag_ft_idx = find_csr_element(row_ptr, col_idx, from_bus, to_bus);
    if (off_diag_ft_idx >= 0) {
        atomicAdd(&values[off_diag_ft_idx].x, -series_admittance_change.x);
        atomicAdd(&values[off_diag_ft_idx].y, -series_admittance_change.y);
    }

    // Update off-diagonal element (to_bus, from_bus)
    int off_diag_tf_idx = find_csr_element(row_ptr, col_idx, to_bus, from_bus);
    if (off_diag_tf_idx >= 0) {
        atomicAdd(&values[off_diag_tf_idx].x, -series_admittance_change.x);
        atomicAdd(&values[off_diag_tf_idx].y, -series_admittance_change.y);
    }
}

namespace gap::admittance {

class GPUAdmittanceMatrix : public IAdmittanceMatrix {
  private:
    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;
    bool initialized_ = false;

    void initialize_cuda() {
        if (initialized_) return;

        cudaError_t cuda_status = cudaSetDevice(0);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device");
        }

        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }

        cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle_);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            cublasDestroy(cublas_handle_);
            throw std::runtime_error("Failed to create cuSPARSE handle");
        }

        initialized_ = true;
    }

  public:
    GPUAdmittanceMatrix() { initialize_cuda(); }

    ~GPUAdmittanceMatrix() {
        if (initialized_) {
            cusparseDestroy(cusparse_handle_);
            cublasDestroy(cublas_handle_);
        }
    }

    std::unique_ptr<SparseMatrix> build_admittance_matrix(
        NetworkData const& network_data) override {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto& logger = gap::logging::global_logger;
        logger.setComponent("GPUAdmittanceMatrix");
        LOG_INFO(logger, "Building admittance matrix on GPU");
        LOG_INFO(logger, "  Number of buses:", network_data.num_buses);
        LOG_INFO(logger, "  Number of branches:", network_data.num_branches);
        LOG_INFO(logger, "  Number of appliances:", network_data.appliances.size());

        // Handle empty network case
        if (network_data.num_buses == 0) {
            auto matrix = std::make_unique<SparseMatrix>();
            matrix->num_rows = 0;
            matrix->num_cols = 0;
            matrix->nnz = 0;
            matrix->row_ptr = {0};
            return matrix;
        }

        // === Step 1: Bus ID mapping (on host) ===
        int min_bus_id = std::numeric_limits<int>::max();
        int max_bus_id = -1;

        for (auto const& bus : network_data.buses) {
            min_bus_id = std::min(min_bus_id, bus.id);
            max_bus_id = std::max(max_bus_id, bus.id);
        }

        // Handle case where no buses exist
        if (max_bus_id < 0) {
            auto matrix = std::make_unique<SparseMatrix>();
            matrix->num_rows = network_data.num_buses;
            matrix->num_cols = network_data.num_buses;
            matrix->nnz = 0;
            matrix->row_ptr.resize(network_data.num_buses + 1, 0);
            return matrix;
        }

        LOG_INFO(logger, "  Bus ID range: [", min_bus_id, ",", max_bus_id, "]");

        // Create ID-to-index mapping
        std::vector<int> h_id_to_idx(max_bus_id + 1, -1);
        for (size_t idx = 0; idx < network_data.buses.size(); ++idx) {
            int bus_id = network_data.buses[idx].id;
            if (bus_id >= 0 && bus_id <= max_bus_id) {
                h_id_to_idx[bus_id] = static_cast<int>(idx);
            }
        }

        // === Step 2: Transfer data to device ===
        thrust::device_vector<BranchData> d_branches(network_data.branches.begin(),
                                                     network_data.branches.end());
        thrust::device_vector<ApplianceData> d_appliances(network_data.appliances.begin(),
                                                          network_data.appliances.end());
        thrust::device_vector<int> d_id_to_idx(h_id_to_idx.begin(), h_id_to_idx.end());
        thrust::device_vector<cuDoubleComplex> d_diagonal(network_data.num_buses,
                                                          make_cuDoubleComplex(0.0, 0.0));

        // Allocate space for off-diagonal triplets (2 per branch)
        int max_triplets = 2 * network_data.num_branches;
        thrust::device_vector<DeviceTriplet> d_triplets(max_triplets);
        thrust::device_vector<int> d_valid_branches(network_data.num_branches, 0);

        // === Step 3: Compute branch admittances in parallel ===
        int threads_per_block = 256;
        int num_blocks = (network_data.num_branches + threads_per_block - 1) / threads_per_block;

        if (network_data.num_branches > 0) {
            compute_branch_admittances_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(d_branches.data()), network_data.num_branches,
                thrust::raw_pointer_cast(d_id_to_idx.data()), max_bus_id,
                thrust::raw_pointer_cast(d_triplets.data()),
                thrust::raw_pointer_cast(d_valid_branches.data()),
                thrust::raw_pointer_cast(d_diagonal.data()));

            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA kernel launch failed: ") +
                                         cudaGetErrorString(cuda_status));
            }
        }

        // === Step 4: Accumulate shunt appliances ===
        if (network_data.appliances.size() > 0) {
            int app_blocks =
                (network_data.appliances.size() + threads_per_block - 1) / threads_per_block;
            accumulate_shunt_appliances_kernel<<<app_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(d_appliances.data()), network_data.appliances.size(),
                thrust::raw_pointer_cast(d_id_to_idx.data()), max_bus_id,
                thrust::raw_pointer_cast(d_diagonal.data()));

            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA appliance kernel failed: ") +
                                         cudaGetErrorString(cuda_status));
            }
        }

        cudaDeviceSynchronize();

        // === Step 5: Copy diagonal back and create diagonal triplets ===
        thrust::host_vector<cuDoubleComplex> h_diagonal = d_diagonal;
        std::vector<DeviceTriplet> h_diagonal_triplets(network_data.num_buses);
        for (int i = 0; i < network_data.num_buses; ++i) {
            h_diagonal_triplets[i] = DeviceTriplet(i, i, h_diagonal[i]);
        }

        // Copy diagonal triplets to device and append
        int diagonal_offset = max_triplets;
        d_triplets.resize(max_triplets + network_data.num_buses);
        thrust::copy(h_diagonal_triplets.begin(), h_diagonal_triplets.end(),
                     d_triplets.begin() + diagonal_offset);

        // === Step 6: Sort triplets by (row, col) using Thrust ===
        // Use stable_sort to match CPU std::ranges::stable_sort and ensure deterministic ordering
        LOG_INFO(logger, "  Sorting triplets on GPU");
        thrust::stable_sort(d_triplets.begin(), d_triplets.end(), TripletComparator());

        // === Step 7: Convert COO to CSR format ===
        // Note: Like CPU implementation, duplicate (row,col) entries are NOT merged
        // Copy sorted triplets back to host for CSR construction
        thrust::host_vector<DeviceTriplet> h_triplets = d_triplets;

        auto matrix = std::make_unique<SparseMatrix>();
        matrix->num_rows = network_data.num_buses;
        matrix->num_cols = network_data.num_buses;

        // Pre-allocate CSR arrays
        matrix->col_idx.reserve(h_triplets.size());
        matrix->values.reserve(h_triplets.size());
        matrix->row_ptr.reserve(network_data.num_buses + 1);

        int current_row = 0;
        matrix->row_ptr.push_back(0);

        for (auto const& triplet : h_triplets) {
            // Skip invalid entries (from inactive branches)
            if (triplet.row < 0 || triplet.col < 0) continue;

            // Fill row_ptr for any empty rows
            while (current_row < triplet.row) {
                matrix->row_ptr.push_back(static_cast<int>(matrix->col_idx.size()));
                current_row++;
            }

            matrix->col_idx.push_back(triplet.col);
            matrix->values.push_back(Complex(cuCreal(triplet.value), cuCimag(triplet.value)));
        }

        // Fill remaining row_ptr entries
        while (current_row < network_data.num_buses) {
            matrix->row_ptr.push_back(static_cast<int>(matrix->col_idx.size()));
            current_row++;
        }

        matrix->nnz = static_cast<int>(matrix->values.size());

        auto end_time = std::chrono::high_resolution_clock::now();
        double build_time_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();

        LOG_INFO(logger, "  Matrix constructed with", matrix->nnz, "non-zero elements");
        LOG_INFO(logger, "  GPU build time:", build_time_ms, "ms");

        return matrix;
    }

    std::unique_ptr<SparseMatrix> update_admittance_matrix(
        SparseMatrix const& matrix, std::vector<BranchData> const& branch_changes) override {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("GPUAdmittanceMatrix");
        LOG_INFO(logger, "Updating admittance matrix on GPU");
        LOG_INFO(logger, "  Branch changes:", branch_changes.size());

        // Create a copy of the matrix
        auto updated_matrix = std::make_unique<SparseMatrix>(matrix);

        // For GPU: For small number of updates, CPU approach is more efficient
        // Only use GPU if many branches change simultaneously (>100)
        if (branch_changes.size() < 100) {
            LOG_INFO(logger, "  Using CPU path for small update (", branch_changes.size(),
                     "changes)");

            // Process updates on CPU (same as CPU backend)
            for (auto const& branch_change : branch_changes) {
                int from_bus = branch_change.from_bus;
                int to_bus = branch_change.to_bus;

                // Validate bus indices
                if (from_bus < 0 || from_bus >= updated_matrix->num_rows || to_bus < 0 ||
                    to_bus >= updated_matrix->num_rows) {
                    LOG_WARN(logger, "  Skipping branch update with invalid indices:", from_bus,
                             "→", to_bus);
                    continue;
                }

                // Calculate branch admittance change
                double z_magnitude_sq =
                    branch_change.r1 * branch_change.r1 + branch_change.x1 * branch_change.x1;
                Complex series_admittance_change(0.0, 0.0);
                if (z_magnitude_sq > 1e-12) {
                    series_admittance_change = Complex(branch_change.r1 / z_magnitude_sq,
                                                       -branch_change.x1 / z_magnitude_sq);
                }

                Complex total_admittance_change =
                    series_admittance_change + Complex(branch_change.g1, branch_change.b1);

                if (!branch_change.status) {
                    // Branch going out of service - subtract admittance
                    total_admittance_change = -total_admittance_change;
                    series_admittance_change = -series_admittance_change;
                }

                Complex shunt_change(0.0, branch_change.b1 / 2.0);
                if (!branch_change.status) {
                    shunt_change = -shunt_change;
                }

                // Update diagonal elements for from_bus
                for (int idx = updated_matrix->row_ptr[from_bus];
                     idx < updated_matrix->row_ptr[from_bus + 1]; ++idx) {
                    if (updated_matrix->col_idx[idx] == from_bus) {
                        updated_matrix->values[idx] += total_admittance_change + shunt_change;
                        break;
                    }
                }

                // Update diagonal elements for to_bus
                for (int idx = updated_matrix->row_ptr[to_bus];
                     idx < updated_matrix->row_ptr[to_bus + 1]; ++idx) {
                    if (updated_matrix->col_idx[idx] == to_bus) {
                        updated_matrix->values[idx] += total_admittance_change + shunt_change;
                        break;
                    }
                }

                // Update off-diagonal element (from_bus, to_bus)
                for (int idx = updated_matrix->row_ptr[from_bus];
                     idx < updated_matrix->row_ptr[from_bus + 1]; ++idx) {
                    if (updated_matrix->col_idx[idx] == to_bus) {
                        updated_matrix->values[idx] -= series_admittance_change;
                        break;
                    }
                }

                // Update off-diagonal element (to_bus, from_bus)
                for (int idx = updated_matrix->row_ptr[to_bus];
                     idx < updated_matrix->row_ptr[to_bus + 1]; ++idx) {
                    if (updated_matrix->col_idx[idx] == from_bus) {
                        updated_matrix->values[idx] -= series_admittance_change;
                        break;
                    }
                }
            }

            LOG_INFO(logger, "  Matrix update completed on CPU");
        } else {
            // GPU path: Use parallel kernel for large batch updates
            LOG_INFO(logger, "  Using GPU path for large update (", branch_changes.size(),
                     "changes)");

            // Transfer matrix CSR structure to device
            thrust::device_vector<int> d_row_ptr(updated_matrix->row_ptr.begin(),
                                                 updated_matrix->row_ptr.end());
            thrust::device_vector<int> d_col_idx(updated_matrix->col_idx.begin(),
                                                 updated_matrix->col_idx.end());

            // Convert Complex values to cuDoubleComplex
            std::vector<cuDoubleComplex> h_values(updated_matrix->values.size());
            for (size_t i = 0; i < updated_matrix->values.size(); ++i) {
                h_values[i] = make_cuDoubleComplex(updated_matrix->values[i].real(),
                                                   updated_matrix->values[i].imag());
            }

            LOG_INFO(logger, "  Initial h_values[78] = (", cuCreal(h_values[78]), ",",
                     cuCimag(h_values[78]), ")");

            thrust::device_vector<cuDoubleComplex> d_values(h_values.begin(), h_values.end());

            // Transfer branch changes to device
            thrust::device_vector<BranchData> d_branch_changes(branch_changes.begin(),
                                                               branch_changes.end());

            // Launch kernel
            int threads_per_block = 256;
            int num_blocks = (branch_changes.size() + threads_per_block - 1) / threads_per_block;

            update_admittance_matrix_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(d_branch_changes.data()),
                static_cast<int>(branch_changes.size()), thrust::raw_pointer_cast(d_row_ptr.data()),
                thrust::raw_pointer_cast(d_col_idx.data()),
                thrust::raw_pointer_cast(d_values.data()), updated_matrix->num_rows);

            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA update kernel failed: ") +
                                         cudaGetErrorString(cuda_status));
            }

            cudaDeviceSynchronize();

            // Copy updated values back to host
            thrust::host_vector<cuDoubleComplex> h_values_updated = d_values;

            LOG_INFO(logger, "  Device values size:", d_values.size());
            LOG_INFO(logger, "  Host values size:", h_values_updated.size());
            LOG_INFO(logger, "  Matrix values size:", updated_matrix->values.size());

            // Sample some values for debugging
            if (h_values_updated.size() > 78) {
                LOG_INFO(logger, "  h_values_updated[78] = (", cuCreal(h_values_updated[78]), ",",
                         cuCimag(h_values_updated[78]), ")");
            }

            for (size_t i = 0; i < updated_matrix->values.size(); ++i) {
                updated_matrix->values[i] =
                    Complex(cuCreal(h_values_updated[i]), cuCimag(h_values_updated[i]));
            }

            LOG_INFO(logger, "  Matrix update completed on GPU");
        }

        return updated_matrix;
    }

    BackendType get_backend_type() const noexcept override { return BackendType::GPU_CUDA; }
};

}  // namespace gap::admittance

// C-style interface for dynamic loading
extern "C" {
gap::admittance::IAdmittanceMatrix* create_gpu_admittance_matrix() {
    return new gap::admittance::GPUAdmittanceMatrix();
}

void destroy_gpu_admittance_matrix(gap::admittance::IAdmittanceMatrix* instance) {
    delete instance;
}
}
