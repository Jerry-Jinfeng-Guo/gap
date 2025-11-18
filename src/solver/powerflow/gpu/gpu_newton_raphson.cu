#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "gap/logging/logger.h"
#include "gap/solver/gpu_powerflow_kernels.h"
#include "gap/solver/powerflow_interface.h"

namespace gap::solver {

class GPUNewtonRaphson : public IPowerFlowSolver {
  private:
    std::shared_ptr<ILUSolver> lu_solver_;
    cublasHandle_t cublas_handle_;
    bool initialized_ = false;

    // GPU-resident data (persistent across iterations)
    GPUSparseMatrix gpu_admittance_;
    GPUPowerFlowData gpu_data_;

    // Index mapping arrays on device
    int* d_mismatch_indices_ = nullptr;  // bus_id -> mismatch vector index
    int* d_voltage_indices_ = nullptr;   // bus_id -> voltage correction index
    double* d_specified_magnitudes_ = nullptr;
    cuDoubleComplex* d_currents_ = nullptr;  // Current injections buffer

    // Cached network info
    int num_buses_ = 0;
    int num_unknowns_ = 0;
    bool data_initialized_ = false;

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

        initialized_ = true;
    }

    void cleanup_gpu_data() {
        gpu_admittance_.free_device();
        gpu_data_.free_device();

        if (d_mismatch_indices_) cudaFree(d_mismatch_indices_);
        if (d_voltage_indices_) cudaFree(d_voltage_indices_);
        if (d_specified_magnitudes_) cudaFree(d_specified_magnitudes_);
        if (d_currents_) cudaFree(d_currents_);

        d_mismatch_indices_ = nullptr;
        d_voltage_indices_ = nullptr;
        d_specified_magnitudes_ = nullptr;
        d_currents_ = nullptr;

        data_initialized_ = false;
    }

    void initialize_gpu_data(NetworkData const& network_data,
                             SparseMatrix const& admittance_matrix) {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("GPUNewtonRaphson");

        LOG_INFO(logger, "Initializing GPU-resident data structures");

        // Clean up any existing data
        cleanup_gpu_data();

        num_buses_ = network_data.num_buses;

        // Validate inputs
        if (num_buses_ == 0 || network_data.buses.empty()) {
            throw std::runtime_error("Invalid network data: no buses");
        }

        if (admittance_matrix.nnz == 0 || admittance_matrix.row_ptr.empty()) {
            throw std::runtime_error("Invalid admittance matrix: empty or uninitialized");
        }

        LOG_INFO(logger, "  Admittance matrix: ", admittance_matrix.num_rows, "x",
                 admittance_matrix.num_cols, ", nnz=", admittance_matrix.nnz);

        // Count unknowns and create index mappings
        std::vector<int> h_mismatch_indices(num_buses_, -1);
        std::vector<int> h_voltage_indices(num_buses_, -1);
        std::vector<double> h_magnitudes(num_buses_, 1.0);
        std::vector<int> h_bus_types(num_buses_);

        int mismatch_idx = 0;
        int voltage_idx = 0;

        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            const auto& bus = network_data.buses[i];
            int bus_type_val = static_cast<int>(bus.bus_type);
            h_bus_types[i] = bus_type_val;
            h_magnitudes[i] = bus.u_pu;

            if (bus.bus_type == BusType::SLACK) {
                // Slack bus: no unknowns
                continue;
            } else if (bus.bus_type == BusType::PQ) {
                // PQ bus: P and Q unknowns
                h_mismatch_indices[i] = mismatch_idx;
                h_voltage_indices[i] = voltage_idx;
                mismatch_idx += 2;  // P and Q
                voltage_idx += 2;   // Real and imag parts
            } else if (bus.bus_type == BusType::PV) {
                // PV bus: only P unknown
                h_mismatch_indices[i] = mismatch_idx;
                h_voltage_indices[i] = voltage_idx;
                mismatch_idx += 1;  // Only P
                voltage_idx += 1;   // Only angle (magnitude fixed)
            }
        }

        num_unknowns_ = mismatch_idx;

        LOG_INFO(logger, "  Number of buses:", num_buses_);
        LOG_INFO(logger, "  Number of unknowns:", num_unknowns_);

        // Allocate GPU memory for admittance matrix
        gpu_admittance_.allocate_device(admittance_matrix.num_rows, admittance_matrix.num_cols,
                                        admittance_matrix.nnz);

        // Copy admittance matrix to GPU
        cudaMemcpy(gpu_admittance_.d_row_ptr, admittance_matrix.row_ptr.data(),
                   (num_buses_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_admittance_.d_col_idx, admittance_matrix.col_idx.data(),
                   admittance_matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);

        // Convert Complex to cuDoubleComplex
        std::vector<cuDoubleComplex> cu_values(admittance_matrix.nnz);
        for (int i = 0; i < admittance_matrix.nnz; ++i) {
            cu_values[i] = make_cuDoubleComplex(admittance_matrix.values[i].real(),
                                                admittance_matrix.values[i].imag());
        }
        cudaMemcpy(gpu_admittance_.d_values, cu_values.data(),
                   admittance_matrix.nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        // Allocate power flow data structures
        gpu_data_.allocate_device(num_buses_, num_unknowns_);

        // Allocate index mapping arrays
        cudaMalloc(&d_mismatch_indices_, num_buses_ * sizeof(int));
        cudaMalloc(&d_voltage_indices_, num_buses_ * sizeof(int));
        cudaMalloc(&d_specified_magnitudes_, num_buses_ * sizeof(double));
        cudaMalloc(&d_currents_, num_buses_ * sizeof(cuDoubleComplex));

        // Copy index mappings to device
        cudaMemcpy(d_mismatch_indices_, h_mismatch_indices.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_voltage_indices_, h_voltage_indices.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_specified_magnitudes_, h_magnitudes.data(), num_buses_ * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_data_.d_bus_types, h_bus_types.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Copy power injections to device
        // Aggregate power from buses and appliances
        std::vector<cuDoubleComplex> h_power_inj(num_buses_, make_cuDoubleComplex(0.0, 0.0));

        // First, power specified directly on buses
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            const auto& bus = network_data.buses[i];
            h_power_inj[i] = make_cuDoubleComplex(bus.active_power, bus.reactive_power);
        }

        // Add power from appliances connected to buses
        for (const auto& appliance : network_data.appliances) {
            if (appliance.status == 1 && (appliance.type == ApplianceType::SOURCE ||
                                          appliance.type == ApplianceType::LOADGEN)) {
                // Find bus index for this appliance
                for (size_t i = 0; i < network_data.buses.size(); ++i) {
                    if (network_data.buses[i].id == appliance.node) {
                        cuDoubleComplex existing = h_power_inj[i];
                        h_power_inj[i] =
                            make_cuDoubleComplex(cuCreal(existing) + appliance.p_specified,
                                                 cuCimag(existing) + appliance.q_specified);
                        break;
                    }
                }
            }
        }

        cudaMemcpy(gpu_data_.d_power_injections, h_power_inj.data(),
                   num_buses_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        data_initialized_ = true;

        LOG_INFO(logger, "GPU data initialization complete");
        LOG_INFO(logger, "  Admittance matrix: ", admittance_matrix.nnz, " non-zeros");
    }

  public:
    GPUNewtonRaphson() { initialize_cuda(); }

    ~GPUNewtonRaphson() {
        cleanup_gpu_data();
        if (initialized_) {
            cublasDestroy(cublas_handle_);
        }
    }

    PowerFlowResult solve_power_flow(NetworkData const& network_data,
                                     SparseMatrix const& admittance_matrix,
                                     PowerFlowConfig const& config) override {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("GPUNewtonRaphson");

        LOG_INFO(logger, "Starting GPU Newton-Raphson power flow solver");
        LOG_INFO(logger, "  Tolerance:", config.tolerance);
        LOG_INFO(logger, "  Max iterations:", config.max_iterations);

        PowerFlowResult result;
        result.bus_voltages.resize(network_data.num_buses);

        // Initialize or update GPU data
        if (!data_initialized_ || num_buses_ != network_data.num_buses) {
            initialize_gpu_data(network_data, admittance_matrix);
        }

        // Initialize voltages on GPU
        if (config.use_flat_start) {
            LOG_INFO(logger, "Using flat start initialization");
            gpu_kernels::launch_initialize_flat_start(gpu_data_.d_voltages, gpu_data_.d_bus_types,
                                                      d_specified_magnitudes_, num_buses_);
        } else {
            // TODO: Use previous solution if available
            gpu_kernels::launch_initialize_flat_start(gpu_data_.d_voltages, gpu_data_.d_bus_types,
                                                      d_specified_magnitudes_, num_buses_);
        }

        // Newton-Raphson iterations (all on GPU)
        for (int iter = 0; iter < config.max_iterations; ++iter) {
            // Step 1: Calculate current injections I = Y * V
            gpu_kernels::launch_calculate_current_injections(
                gpu_admittance_.d_row_ptr, gpu_admittance_.d_col_idx, gpu_admittance_.d_values,
                gpu_data_.d_voltages, d_currents_, num_buses_);

            // Step 2: Calculate power mismatches
            gpu_kernels::launch_calculate_power_mismatches(
                gpu_data_.d_voltages, d_currents_, gpu_data_.d_power_injections,
                gpu_data_.d_bus_types, gpu_data_.d_mismatches, num_buses_, d_mismatch_indices_);

            // Step 3: Check convergence (sync mismatches to host for now)
            gpu_data_.sync_mismatches_from_device();

            double max_mismatch = 0.0;
            for (double mismatch : gpu_data_.h_mismatches) {
                max_mismatch = std::max(max_mismatch, std::abs(mismatch));
            }

            result.final_mismatch = max_mismatch;

            if (config.verbose) {
                LOG_INFO(logger, "  Iteration", iter + 1, "- Max mismatch:", max_mismatch);
            }

            if (max_mismatch < config.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                LOG_INFO(logger, "Converged in", iter + 1, "iterations");
                break;
            }

            // Step 4: Solve correction equations (Jacobian * delta_V = mismatches)
            // TODO: Build Jacobian on GPU and use GPU solver
            // For now, this is a simplified placeholder

            // Step 5: Update voltages on GPU
            // TODO: Use actual correction from solver
            // Placeholder: small update

            if (iter == config.max_iterations - 1) {
                LOG_INFO(logger, "Failed to converge after", config.max_iterations, "iterations");
            }
        }

        // Copy final voltages back to host
        gpu_data_.sync_voltages_from_device();
        result.bus_voltages = gpu_data_.h_voltages;

        if (!result.converged) {
            result.iterations = config.max_iterations;
        }

        return result;
    }

    void set_lu_solver(std::shared_ptr<ILUSolver> lu_solver) override {
        lu_solver_ = lu_solver;
        LOG_INFO(gap::logging::global_logger, "GPU LU solver backend set");
    }

    /**
     * @brief Get current state snapshot for debugging
     */
    GPUPowerFlowData::StateSnapshot get_debug_snapshot() {
        if (!data_initialized_) {
            throw std::runtime_error("GPU data not initialized");
        }
        return gpu_data_.get_state_snapshot();
    }

    /**
     * @brief Get admittance matrix copy for verification
     */
    SparseMatrix get_admittance_copy() {
        if (!data_initialized_) {
            throw std::runtime_error("GPU data not initialized");
        }
        return gpu_admittance_.get_host_copy();
    }

    std::vector<Float> calculate_mismatches(
        NetworkData const& network_data, [[maybe_unused]] ComplexVector const& bus_voltages,
        [[maybe_unused]] SparseMatrix const& admittance_matrix) override {
        // This method is deprecated - mismatches are calculated on GPU
        auto& logger = gap::logging::global_logger;
        LOG_INFO(logger, "Warning: calculate_mismatches called on GPU solver (deprecated)");

        std::vector<Float> mismatches;
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            if (network_data.buses[i].bus_type != BusType::SLACK) {
                mismatches.push_back(0.0);
                if (network_data.buses[i].bus_type == BusType::PQ) {
                    mismatches.push_back(0.0);
                }
            }
        }
        return mismatches;
    }

    BackendType get_backend_type() const noexcept override { return BackendType::GPU_CUDA; }
};

}  // namespace gap::solver

// C-style interface for dynamic loading
extern "C" {
gap::solver::IPowerFlowSolver* create_gpu_powerflow_solver() {
    return new gap::solver::GPUNewtonRaphson();
}

void destroy_gpu_powerflow_solver(gap::solver::IPowerFlowSolver* instance) { delete instance; }
}
