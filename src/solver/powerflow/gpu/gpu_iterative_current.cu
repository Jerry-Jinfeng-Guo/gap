#include <cuda.h>
#include <cuda_runtime.h>

#include <cuComplex.h>

#include <iostream>
#include <memory>
#include <vector>

#include "gap/core/backend_factory.h"
#include "gap/logging/logger.h"
#include "gap/solver/gpu_iterative_current_kernels.h"
#include "gap/solver/powerflow_interface.h"

namespace gap::solver {

// ============================================================================
// GPU Iterative Current Solver Implementation
// ============================================================================

class GPUIterativeCurrent : public IPowerFlowSolver {
  private:
    std::shared_ptr<ILUSolver> lu_solver_;
    bool initialized_ = false;

    // GPU memory for IC solver
    cuDoubleComplex* d_voltages_ = nullptr;      // Current voltages
    cuDoubleComplex* d_old_voltages_ = nullptr;  // Previous iteration voltages
    cuDoubleComplex* d_currents_ = nullptr;      // Current injections (calculated)
    cuDoubleComplex* d_i_specified_ = nullptr;   // Specified currents (from loads/gens)
    double* d_max_change_ = nullptr;             // Max voltage change for convergence

    // Y-bus matrix in CSR format on GPU
    cuDoubleComplex* d_ybus_values_ = nullptr;  // Non-zero values
    int* d_ybus_col_idx_ = nullptr;             // Column indices
    int* d_ybus_row_ptr_ = nullptr;             // Row pointers
    int ybus_nnz_ = 0;                          // Number of non-zeros

    int num_buses_ = 0;
    int slack_bus_idx_ = 0;  // Index of slack bus (default: bus 0)
    bool data_allocated_ = false;
    bool ybus_allocated_ = false;

    gap::logging::Logger& logger = gap::logging::global_logger;

    void initialize_cuda() {
        if (initialized_) {
            return;
        }

        LOG_DEBUG(logger, "Initializing CUDA...");

        // Simple initialization - just set device (do NOT call cudaDeviceReset!)
        cudaError_t status = cudaSetDevice(0);
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device: " +
                                     std::string(cudaGetErrorString(status)));
        }

        initialized_ = true;
        LOG_DEBUG(logger, "CUDA initialized successfully");
    }

    void verify_kernel_capability() {
        // Quick verification that kernel launches work
        const int n = 4;
        double* d_data = nullptr;

        cudaError_t err = cudaMalloc(&d_data, n * sizeof(double));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed in verification");
        }

        std::vector<double> h_data = {1.0, 2.0, 3.0, 4.0};
        cudaMemcpy(d_data, h_data.data(), n * sizeof(double), cudaMemcpyHostToDevice);

        ic_kernels::launch_test_kernel(d_data, n);

        cudaFree(d_data);

        LOG_DEBUG(logger, "Kernel launch capability verified");
    }

    void cleanup_gpu_memory() {
        if (d_voltages_) {
            cudaFree(d_voltages_);
            d_voltages_ = nullptr;
        }
        if (d_old_voltages_) {
            cudaFree(d_old_voltages_);
            d_old_voltages_ = nullptr;
        }
        if (d_currents_) {
            cudaFree(d_currents_);
            d_currents_ = nullptr;
        }
        if (d_i_specified_) {
            cudaFree(d_i_specified_);
            d_i_specified_ = nullptr;
        }
        if (d_max_change_) {
            cudaFree(d_max_change_);
            d_max_change_ = nullptr;
        }
        data_allocated_ = false;
    }

    void cleanup_ybus_memory() {
        if (d_ybus_values_) {
            cudaFree(d_ybus_values_);
            d_ybus_values_ = nullptr;
        }
        if (d_ybus_col_idx_) {
            cudaFree(d_ybus_col_idx_);
            d_ybus_col_idx_ = nullptr;
        }
        if (d_ybus_row_ptr_) {
            cudaFree(d_ybus_row_ptr_);
            d_ybus_row_ptr_ = nullptr;
        }
        ybus_allocated_ = false;
        ybus_nnz_ = 0;
    }

    void allocate_gpu_memory(int num_buses) {
        LOG_DEBUG(logger, "Allocating GPU memory for", num_buses, "buses");

        // Free existing if size changed
        if (data_allocated_ && num_buses_ != num_buses) {
            cleanup_gpu_memory();
        }

        if (data_allocated_ && num_buses_ == num_buses) {
            LOG_DEBUG(logger, "  Memory already allocated");
            return;
        }

        num_buses_ = num_buses;

        cudaError_t err;
        err = cudaMalloc(&d_voltages_, num_buses * sizeof(cuDoubleComplex));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate d_voltages_: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&d_old_voltages_, num_buses * sizeof(cuDoubleComplex));
        if (err != cudaSuccess) {
            cleanup_gpu_memory();
            throw std::runtime_error("Failed to allocate d_old_voltages_: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&d_currents_, num_buses * sizeof(cuDoubleComplex));
        if (err != cudaSuccess) {
            cleanup_gpu_memory();
            throw std::runtime_error("Failed to allocate d_currents_: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&d_i_specified_, num_buses * sizeof(cuDoubleComplex));
        if (err != cudaSuccess) {
            cleanup_gpu_memory();
            throw std::runtime_error("Failed to allocate d_i_specified_: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&d_max_change_, sizeof(double));
        if (err != cudaSuccess) {
            cleanup_gpu_memory();
            throw std::runtime_error("Failed to allocate d_max_change_: " +
                                     std::string(cudaGetErrorString(err)));
        }

        data_allocated_ = true;
        LOG_DEBUG(logger, "GPU memory allocated successfully");
    }

    void allocate_ybus_memory(SparseMatrix const& ybus) {
        LOG_DEBUG(logger, "Allocating Y-bus GPU memory -", ybus.num_rows, "x", ybus.num_cols,
                  "with", ybus.nnz, "non-zeros");

        // Free existing if size changed
        if (ybus_allocated_ && ybus_nnz_ != ybus.nnz) {
            cleanup_ybus_memory();
        }

        if (ybus_allocated_ && ybus_nnz_ == ybus.nnz) {
            LOG_DEBUG(logger, "  Y-bus already allocated, reusing");
            return;
        }

        ybus_nnz_ = ybus.nnz;

        cudaError_t err;

        // Allocate Y-bus values
        err = cudaMalloc(&d_ybus_values_, ybus.nnz * sizeof(cuDoubleComplex));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate d_ybus_values_: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // Allocate column indices
        err = cudaMalloc(&d_ybus_col_idx_, ybus.nnz * sizeof(int));
        if (err != cudaSuccess) {
            cleanup_ybus_memory();
            throw std::runtime_error("Failed to allocate d_ybus_col_idx_: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // Allocate row pointers (num_rows + 1)
        err = cudaMalloc(&d_ybus_row_ptr_, (ybus.num_rows + 1) * sizeof(int));
        if (err != cudaSuccess) {
            cleanup_ybus_memory();
            throw std::runtime_error("Failed to allocate d_ybus_row_ptr_: " +
                                     std::string(cudaGetErrorString(err)));
        }

        ybus_allocated_ = true;
        LOG_DEBUG(logger, "Y-bus GPU memory allocated successfully");
    }

    void transfer_ybus_to_gpu(SparseMatrix const& ybus) {
        if (!ybus_allocated_) {
            throw std::runtime_error("Y-bus GPU memory not allocated");
        }

        LOG_DEBUG(logger, "Transferring Y-bus to GPU");

        cudaError_t err;

        // Convert Complex to cuDoubleComplex and copy values
        std::vector<cuDoubleComplex> h_values(ybus.nnz);
        for (int i = 0; i < ybus.nnz; ++i) {
            h_values[i] = make_cuDoubleComplex(ybus.values[i].real(), ybus.values[i].imag());
        }

        err = cudaMemcpy(d_ybus_values_, h_values.data(), ybus.nnz * sizeof(cuDoubleComplex),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy Y-bus values to GPU: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // Copy column indices
        err = cudaMemcpy(d_ybus_col_idx_, ybus.col_idx.data(), ybus.nnz * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy Y-bus col_idx to GPU: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // Copy row pointers
        err = cudaMemcpy(d_ybus_row_ptr_, ybus.row_ptr.data(), (ybus.num_rows + 1) * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy Y-bus row_ptr to GPU: " +
                                     std::string(cudaGetErrorString(err)));
        }

        LOG_DEBUG(logger, "Y-bus transferred to GPU successfully");
    }

    void initialize_voltages_flat_start() {
        if (!data_allocated_) {
            throw std::runtime_error("GPU memory not allocated");
        }

        LOG_DEBUG(logger, "Initializing voltages to flat start on GPU");

        ic_kernels::launch_initialize_voltages_flat(d_voltages_, num_buses_);

        LOG_DEBUG(logger, "Voltages initialized on GPU");
    }

    void save_voltages_for_convergence() {
        if (!data_allocated_) {
            throw std::runtime_error("GPU memory not allocated");
        }

        ic_kernels::launch_copy_voltages(d_voltages_, d_old_voltages_, num_buses_);
    }

    double check_convergence() {
        if (!data_allocated_) {
            throw std::runtime_error("GPU memory not allocated");
        }

        // Zero out max_change on device
        cudaMemset(d_max_change_, 0, sizeof(double));

        ic_kernels::launch_calculate_voltage_change(d_voltages_, d_old_voltages_, d_max_change_,
                                                    num_buses_);

        // Copy result back to host
        double max_change = 0.0;
        cudaError_t err =
            cudaMemcpy(&max_change, d_max_change_, sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy max_change to host: " +
                                     std::string(cudaGetErrorString(err)));
        }

        return max_change;
    }

    void calculate_currents() {
        if (!data_allocated_ || !ybus_allocated_) {
            throw std::runtime_error("GPU memory or Y-bus not allocated");
        }

        ic_kernels::launch_calculate_currents(d_ybus_values_, d_ybus_col_idx_, d_ybus_row_ptr_,
                                              d_voltages_, d_currents_, num_buses_);
    }

    void update_voltages() {
        if (!data_allocated_ || !ybus_allocated_) {
            throw std::runtime_error("GPU memory or Y-bus not allocated");
        }

        ic_kernels::launch_update_voltages(d_ybus_values_, d_ybus_col_idx_, d_ybus_row_ptr_,
                                           d_i_specified_, d_currents_, d_voltages_, num_buses_);
    }

    void enforce_slack_bus() {
        if (!data_allocated_) {
            throw std::runtime_error("GPU memory not allocated");
        }

        // Set slack bus to reference voltage (1.0 + 0j)
        cuDoubleComplex slack_voltage = make_cuDoubleComplex(1.0, 0.0);

        ic_kernels::launch_enforce_slack_bus(d_voltages_, slack_bus_idx_, slack_voltage);
    }

    void setup_specified_currents(NetworkData const& network_data, PowerFlowConfig const& config) {
        LOG_DEBUG(logger, "Setting up specified currents from load/generation data");

        // Initialize all specified currents to zero
        std::vector<cuDoubleComplex> h_i_specified(num_buses_, make_cuDoubleComplex(0.0, 0.0));

        // Calculate I_specified from loads and generators: I = (P - jQ)* / V*
        // For now, use flat start voltage (1.0 + 0j) for initial calculation
        for (const auto& appliance : network_data.appliances) {
            if (appliance.status == 0) continue;  // Skip disabled appliances

            int bus_idx = appliance.node - 1;  // Convert to 0-based index
            if (bus_idx < 0 || bus_idx >= num_buses_) continue;

            // For PQ loads/generators: I = (P - jQ)* / V*
            // With V = 1.0 + 0j: I* = (P - jQ) / 1.0, so I = (P + jQ)
            // Sign convention: Loads have negative P, so current is negative (flowing out)
            double p_pu = appliance.p_specified / config.base_power;
            double q_pu = appliance.q_specified / config.base_power;

            // I = (S / V)* = ((P + jQ) / V)* = (P + jQ) for V=1.0+0j
            // Sign: negative P (load) gives negative real current (flowing out of bus)
            cuDoubleComplex i_bus = make_cuDoubleComplex(-p_pu, -q_pu);
            h_i_specified[bus_idx] = cuCadd(h_i_specified[bus_idx], i_bus);
        }

        // Transfer to GPU
        cudaError_t err = cudaMemcpy(d_i_specified_, h_i_specified.data(),
                                     num_buses_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy I_specified to device: " +
                                     std::string(cudaGetErrorString(err)));
        }

        LOG_DEBUG(logger, "Specified currents transferred to GPU");
    }

  public:
    explicit GPUIterativeCurrent(std::shared_ptr<ILUSolver> lu_solver = nullptr)
        : lu_solver_(lu_solver) {
        initialize_cuda();
        logger.setComponent("GPUIterativeCurrent");
    }

    ~GPUIterativeCurrent() {
        cleanup_gpu_memory();
        cleanup_ybus_memory();
    }

    void set_lu_solver(std::shared_ptr<ILUSolver> solver) override {
        lu_solver_ = solver;
        LOG_INFO(logger, "GPU LU solver backend set");
    }

    PowerFlowResult solve_power_flow(NetworkData const& network_data,
                                     SparseMatrix const& admittance_matrix,
                                     PowerFlowConfig const& config) override {
        LOG_INFO(logger, "Starting GPU Iterative Current power flow solver");
        LOG_INFO(logger, "  Tolerance:", config.tolerance);
        LOG_INFO(logger, "  Max iterations:", config.max_iterations);
        LOG_INFO(logger, "  Number of buses:", network_data.num_buses);

        // Step 1: Allocate GPU memory for this network size
        allocate_gpu_memory(network_data.num_buses);

        // Step 1b: Allocate and transfer Y-bus to GPU
        allocate_ybus_memory(admittance_matrix);
        transfer_ybus_to_gpu(admittance_matrix);

        // Step 2: Initialize voltages on GPU to flat start
        initialize_voltages_flat_start();

        // Step 2b: Set slack bus index (bus 0 by default)
        slack_bus_idx_ = 0;
        LOG_DEBUG(logger, "Slack bus set to bus", slack_bus_idx_);

        // Step 2c: Calculate and transfer specified currents from loads/generators
        setup_specified_currents(network_data, config);

        // Step 3-7: Iteration loop
        int iteration = 0;
        double max_change = 0.0;
        bool converged = false;

        for (iteration = 0; iteration < config.max_iterations; ++iteration) {
            // Step 3: Save current voltages for convergence check
            save_voltages_for_convergence();

            // Step 4: Calculate new currents using I = Y * V
            calculate_currents();

            // Step 5: Update voltages using Iterative Current method
            update_voltages();

            // Step 6: Enforce slack bus constraint
            enforce_slack_bus();

            // Step 7: Check convergence
            max_change = check_convergence();

            if (config.verbose && iteration % 10 == 0) {
                LOG_INFO(logger, "  Iteration", iteration, "- Max voltage change:", max_change);
            }

            if (max_change < config.tolerance) {
                converged = true;
                break;
            }
        }

        if (converged) {
            LOG_INFO(logger, "Converged in", iteration, "iterations");
        } else {
            LOG_INFO(logger, "Failed to converge after", config.max_iterations, "iterations");
        }
        LOG_INFO(logger, "  Final max voltage change:", max_change);

        // Copy final voltages back to host
        std::vector<cuDoubleComplex> h_voltages(network_data.num_buses);
        cudaError_t err =
            cudaMemcpy(h_voltages.data(), d_voltages_,
                       network_data.num_buses * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy voltages from device: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // Prepare result
        PowerFlowResult result;
        result.converged = converged;
        result.iterations = iteration;
        result.final_mismatch = max_change;
        result.bus_voltages.resize(network_data.num_buses);

        // Convert cuDoubleComplex to Complex
        for (int i = 0; i < network_data.num_buses; ++i) {
            result.bus_voltages[i] = Complex(cuCreal(h_voltages[i]), cuCimag(h_voltages[i]));
        }

        return result;
    }

    BatchPowerFlowResult solve_power_flow_batch(std::vector<NetworkData> const& network_scenarios,
                                                SparseMatrix const& admittance_matrix,
                                                BatchPowerFlowConfig const& config) override {
        LOG_INFO(logger, "Starting GPU Iterative Current batch power flow solver");
        LOG_INFO(logger, "  Number of scenarios:", network_scenarios.size());

        BatchPowerFlowResult batch_result;

        // Solve each scenario with dummy results
        for (size_t i = 0; i < network_scenarios.size(); ++i) {
            auto result =
                solve_power_flow(network_scenarios[i], admittance_matrix, config.base_config);
            batch_result.results.push_back(result);
            batch_result.total_iterations += result.iterations;
            if (result.converged) {
                batch_result.converged_count++;
            }
        }

        batch_result.total_solve_time_ms = 0.0;
        batch_result.avg_solve_time_ms = 0.0;

        LOG_INFO(logger, "Batch power flow complete -", batch_result.converged_count, "of",
                 network_scenarios.size(), "scenarios converged");
        return batch_result;
    }

    std::vector<Float> calculate_mismatches(NetworkData const& /*network_data*/,
                                            ComplexVector const& /*bus_voltages*/,
                                            SparseMatrix const& /*admittance_matrix*/) override {
        return std::vector<Float>();
    }

    BackendType get_backend_type() const noexcept override { return BackendType::GPU_CUDA; }

    void enable_state_capture(bool /*enable*/) override {
        // Not implemented in shell
    }

    std::vector<IterationState> const& get_iteration_states() const override {
        static std::vector<IterationState> empty;
        return empty;
    }

    void clear_iteration_states() override {
        // Not implemented in shell
    }
};

// C-style interface for dynamic loading
extern "C" {
gap::solver::IPowerFlowSolver* create_gpu_ic_powerflow_solver() {
    return new gap::solver::GPUIterativeCurrent();
}

void destroy_gpu_ic_powerflow_solver(gap::solver::IPowerFlowSolver* instance) { delete instance; }
}

}  // namespace gap::solver
