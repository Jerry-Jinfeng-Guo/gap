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

// Forward declarations for GPU IC specific kernels
namespace gpu_ic_kernels {
__global__ void calculate_ic_current_injections_kernel(
    cuDoubleComplex const* __restrict__ voltages,
    cuDoubleComplex const* __restrict__ specified_powers, int const* __restrict__ bus_types,
    int const* __restrict__ load_types, double const* __restrict__ u_rated,
    cuDoubleComplex* __restrict__ currents, int num_buses);

__global__ void check_voltage_convergence_kernel(cuDoubleComplex const* __restrict__ old_voltages,
                                                 cuDoubleComplex const* __restrict__ new_voltages,
                                                 double* __restrict__ max_change, int num_buses);

__global__ void enforce_slack_voltage_kernel(cuDoubleComplex* __restrict__ voltages,
                                             int const* __restrict__ bus_types,
                                             double const* __restrict__ u_pu, int num_buses);
}  // namespace gpu_ic_kernels

class GPUIterativeCurrent : public IPowerFlowSolver {
  private:
    std::shared_ptr<ILUSolver> lu_solver_;
    cublasHandle_t cublas_handle_;
    bool initialized_ = false;

    // State capture for debugging
    bool capture_states_ = false;
    std::vector<IterationState> iteration_states_;

    // GPU-resident data (persistent across iterations)
    GPUSparseMatrix gpu_admittance_;
    GPUPowerFlowData gpu_data_;

    // GPU arrays for iterative current method
    cuDoubleComplex* d_currents_ = nullptr;          // Current injections
    cuDoubleComplex* d_old_voltages_ = nullptr;      // Previous iteration voltages
    double* d_max_change_ = nullptr;                 // Convergence metric
    cuDoubleComplex* d_specified_powers_ = nullptr;  // Bus power injections
    int* d_bus_types_ = nullptr;                     // Bus types (PQ, PV, SLACK)
    int* d_load_types_ = nullptr;                    // ZIP load model types
    double* d_u_rated_ = nullptr;                    // Rated voltages
    double* d_u_pu_ = nullptr;                       // Per-unit voltages for slack buses

    // Cached network info
    int num_buses_ = 0;
    bool data_initialized_ = false;

    // Y-bus factorization caching
    bool y_bus_factorized_ = false;
    SparseMatrix cached_y_bus_;

    gap::logging::Logger& logger = gap::logging::global_logger;

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

        if (d_currents_) cudaFree(d_currents_);
        if (d_old_voltages_) cudaFree(d_old_voltages_);
        if (d_max_change_) cudaFree(d_max_change_);
        if (d_specified_powers_) cudaFree(d_specified_powers_);
        if (d_bus_types_) cudaFree(d_bus_types_);
        if (d_load_types_) cudaFree(d_load_types_);
        if (d_u_rated_) cudaFree(d_u_rated_);
        if (d_u_pu_) cudaFree(d_u_pu_);

        d_currents_ = nullptr;
        d_old_voltages_ = nullptr;
        d_max_change_ = nullptr;
        d_specified_powers_ = nullptr;
        d_bus_types_ = nullptr;
        d_load_types_ = nullptr;
        d_u_rated_ = nullptr;
        d_u_pu_ = nullptr;

        data_initialized_ = false;
    }

    void allocate_gpu_arrays(int num_buses) {
        if (num_buses_ == num_buses && data_initialized_) {
            return;  // Already allocated
        }

        cleanup_gpu_data();
        num_buses_ = num_buses;

        // Allocate GPU memory
        cudaMalloc(&d_currents_, num_buses * sizeof(cuDoubleComplex));
        cudaMalloc(&d_old_voltages_, num_buses * sizeof(cuDoubleComplex));
        cudaMalloc(&d_max_change_, sizeof(double));
        cudaMalloc(&d_specified_powers_, num_buses * sizeof(cuDoubleComplex));
        cudaMalloc(&d_bus_types_, num_buses * sizeof(int));
        cudaMalloc(&d_load_types_, num_buses * sizeof(int));
        cudaMalloc(&d_u_rated_, num_buses * sizeof(double));
        cudaMalloc(&d_u_pu_, num_buses * sizeof(double));

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("GPU memory allocation failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        data_initialized_ = true;
    }

    void upload_network_data(NetworkData const& network_data, PowerFlowConfig const& config) {
        // Prepare host arrays
        std::vector<cuDoubleComplex> h_specified_powers(num_buses_);
        std::vector<int> h_bus_types(num_buses_);
        std::vector<int> h_load_types(num_buses_);
        std::vector<double> h_u_rated(num_buses_);
        std::vector<double> h_u_pu(num_buses_);

        double base_power = config.base_power;

        for (int i = 0; i < num_buses_; ++i) {
            auto const& bus = network_data.buses[i];

            // Bus type
            h_bus_types[i] = static_cast<int>(bus.bus_type);

            // Rated voltage
            h_u_rated[i] = bus.u_rated;

            // Per-unit voltage for slack bus
            h_u_pu[i] = bus.u_pu;

            // Calculate specified power at this bus from appliances
            double p_total = 0.0;
            double q_total = 0.0;

            // Add appliance contributions
            for (auto const& appliance : network_data.appliances) {
                if (appliance.node == bus.id && appliance.status == 1) {
                    if (appliance.type == ApplianceType::LOADGEN) {
                        p_total += appliance.p_specified;
                        q_total += appliance.q_specified;
                    }
                }
            }

            // Convert to per-unit
            double p_pu = p_total / base_power;
            double q_pu = q_total / base_power;

            h_specified_powers[i] = make_cuDoubleComplex(p_pu, q_pu);

            // Determine load type from first matching appliance
            h_load_types[i] = 0;  // Default: const_pq
            for (auto const& appliance : network_data.appliances) {
                if (appliance.node == bus.id && appliance.status == 1 &&
                    appliance.type == ApplianceType::LOADGEN) {
                    h_load_types[i] = static_cast<int>(appliance.load_gen_type);
                    break;
                }
            }
        }

        // Upload to GPU
        cudaMemcpy(d_specified_powers_, h_specified_powers.data(),
                   num_buses_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bus_types_, h_bus_types.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_load_types_, h_load_types.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_u_rated_, h_u_rated.data(), num_buses_ * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_u_pu_, h_u_pu.data(), num_buses_ * sizeof(double), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
    }

    void initialize_voltages_flat_start(NetworkData const& network_data,
                                        cuDoubleComplex* d_voltages) {
        // Initialize on CPU and upload
        std::vector<cuDoubleComplex> h_voltages(num_buses_);

        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            if (network_data.buses[i].bus_type == BusType::SLACK ||
                network_data.buses[i].bus_type == BusType::PV) {
                h_voltages[i] = make_cuDoubleComplex(network_data.buses[i].u_pu, 0.0);
            } else {
                h_voltages[i] = make_cuDoubleComplex(0.98, 0.0);  // PQ buses slightly lower
            }
        }

        cudaMemcpy(d_voltages, h_voltages.data(), num_buses_ * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    void factorize_y_bus_if_needed(SparseMatrix const& y_matrix, NetworkData const& network_data) {
        // Check if Y-bus has changed or not yet factorized
        bool needs_factorization = !y_bus_factorized_;

        if (!needs_factorization && cached_y_bus_.nnz == y_matrix.nnz) {
            // Quick check: compare structure and values
            for (size_t i = 0; i < y_matrix.values.size(); ++i) {
                if (std::abs(cached_y_bus_.values[i] - y_matrix.values[i]) > 1e-14) {
                    needs_factorization = true;
                    break;
                }
            }
        } else {
            needs_factorization = true;
        }

        if (needs_factorization) {
            LOG_INFO(logger, "  Factorizing Y-bus matrix on GPU (one-time cost)");

            if (!lu_solver_) {
                throw std::runtime_error("LU solver not initialized");
            }

            // Create modified Y-bus with source admittance added to diagonal
            SparseMatrix y_modified = y_matrix;

            // Add source admittance to diagonal for slack buses (PGM approach)
            Complex y_source(1.0, -10.0);  // High imaginary part for stiff source

            for (int i = 0; i < network_data.num_buses; ++i) {
                if (network_data.buses[i].bus_type == BusType::SLACK) {
                    // Find diagonal element in sparse matrix
                    for (int idx = y_modified.row_ptr[i]; idx < y_modified.row_ptr[i + 1]; ++idx) {
                        if (y_modified.col_idx[idx] == i) {
                            y_modified.values[idx] += y_source;
                            break;
                        }
                    }
                }
            }

            // Factorize the modified Y-bus
            lu_solver_->factorize(y_modified);

            // Cache the Y-bus
            cached_y_bus_ = y_modified;
            y_bus_factorized_ = true;
        }
    }

  public:
    explicit GPUIterativeCurrent(std::shared_ptr<ILUSolver> lu_solver = nullptr)
        : lu_solver_(lu_solver) {
        initialize_cuda();
        logger.setComponent("GPUIterativeCurrent");
    }

    ~GPUIterativeCurrent() {
        cleanup_gpu_data();

        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
    }

    void set_lu_solver(std::shared_ptr<ILUSolver> solver) override { lu_solver_ = solver; }

    PowerFlowResult solve_power_flow(NetworkData const& network_data,
                                     SparseMatrix const& admittance_matrix,
                                     PowerFlowConfig const& config) override {
        LOG_INFO(logger, "Starting GPU iterative current power flow solution");
        LOG_DEBUG(logger, "  Number of buses:", network_data.num_buses);
        LOG_DEBUG(logger, "  Tolerance:", config.tolerance);
        LOG_DEBUG(logger, "  Max iterations:", config.max_iterations);
        LOG_DEBUG(logger, "  Base power:", config.base_power / 1e6, "MVA");

        // Per-unit system normalization
        Float v_base = network_data.buses[0].u_rated;
        Float base_impedance = (v_base * v_base) / config.base_power;

        LOG_INFO(logger, "  Base voltage:", v_base / 1e3, "kV");
        LOG_INFO(logger, "  Base impedance:", base_impedance, "Ohms");

        // Create per-unit admittance matrix
        SparseMatrix y_pu = admittance_matrix;
        for (size_t i = 0; i < y_pu.values.size(); ++i) {
            y_pu.values[i] = admittance_matrix.values[i] * base_impedance;
        }

        PowerFlowResult result;
        result.bus_voltages.resize(network_data.num_buses);

        // Allocate GPU arrays
        allocate_gpu_arrays(network_data.num_buses);

        // Upload network data to GPU
        upload_network_data(network_data, config);

        // Allocate voltages array
        cuDoubleComplex* d_voltages = nullptr;
        cudaMalloc(&d_voltages, num_buses_ * sizeof(cuDoubleComplex));

        // Initialize voltages (flat start)
        if (config.use_flat_start) {
            LOG_DEBUG(logger, "  Using flat start initialization");
            initialize_voltages_flat_start(network_data, d_voltages);
        }

        // Clear previous iteration states if capturing
        if (capture_states_) {
            iteration_states_.clear();
        }

        // Factorize Y-bus once (cached for subsequent solves with same topology)
        factorize_y_bus_if_needed(y_pu, network_data);

        auto total_start = std::chrono::high_resolution_clock::now();

        // Iterative current power flow iterations
        for (int iter = 0; iter < config.max_iterations; ++iter) {
            if (config.verbose) {
                LOG_DEBUG(logger, "  Iteration", (iter + 1));
            }

            // Step 1: Calculate current injections I = conj(S / V) for loads
            int block_size = 256;
            int grid_size = (num_buses_ + block_size - 1) / block_size;

            gpu_ic_kernels::calculate_ic_current_injections_kernel<<<grid_size, block_size>>>(
                d_voltages, d_specified_powers_, d_bus_types_, d_load_types_, d_u_pu_, d_currents_,
                num_buses_);

            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Current injection kernel failed: " +
                                         std::string(cudaGetErrorString(err)));
            }

            // Step 2: Solve linear system Y * V = I for new voltages on GPU
            std::vector<Complex> h_currents(num_buses_);
            cudaMemcpy(h_currents.data(), d_currents_, num_buses_ * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost);

            std::vector<Complex> h_new_voltages = lu_solver_->solve(h_currents);

            // Upload new voltages to GPU
            cudaMemcpy(d_old_voltages_, d_voltages, num_buses_ * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_voltages, reinterpret_cast<cuDoubleComplex*>(h_new_voltages.data()),
                       num_buses_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

            // Step 3: Enforce slack bus voltage constraints
            gpu_ic_kernels::enforce_slack_voltage_kernel<<<grid_size, block_size>>>(
                d_voltages, d_bus_types_, d_u_pu_, num_buses_);

            cudaDeviceSynchronize();

            // Step 4: Check convergence - max voltage change
            double h_max_change = 0.0;
            cudaMemset(d_max_change_, 0, sizeof(double));

            gpu_ic_kernels::check_voltage_convergence_kernel<<<grid_size, block_size>>>(
                d_old_voltages_, d_voltages, d_max_change_, num_buses_);

            cudaMemcpy(&h_max_change, d_max_change_, sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            result.final_mismatch = h_max_change;

            // Capture iteration state if enabled
            if (capture_states_) {
                IterationState state;
                state.iteration = iter;
                state.max_mismatch = h_max_change;

                // Download voltages and currents for state capture
                std::vector<Complex> h_voltages(num_buses_);
                cudaMemcpy(h_voltages.data(), d_voltages, num_buses_ * sizeof(cuDoubleComplex),
                           cudaMemcpyDeviceToHost);
                state.voltages = h_voltages;

                cudaMemcpy(h_currents.data(), d_currents_, num_buses_ * sizeof(cuDoubleComplex),
                           cudaMemcpyDeviceToHost);
                state.currents = h_currents;

                iteration_states_.push_back(state);
            }

            if (h_max_change < config.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                LOG_INFO(logger, "  Converged in", (iter + 1), "iterations");
                break;
            }
        }

        if (!result.converged) {
            LOG_WARN(logger, "  Failed to converge after", config.max_iterations, "iterations");
            result.iterations = config.max_iterations;
        }

        // Download final voltages from GPU
        cudaMemcpy(result.bus_voltages.data(), d_voltages, num_buses_ * sizeof(cuDoubleComplex),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_voltages);

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();

        if (config.verbose || !result.converged) {
            LOG_INFO(logger, "\n=== GPU IC SOLUTION SUMMARY ===");
            LOG_INFO(logger, "Convergence:", (result.converged ? "YES" : "NO"));
            LOG_INFO(logger, "Iterations:", result.iterations);
            LOG_INFO(logger, "Final mismatch:", result.final_mismatch);
            LOG_INFO(logger, "Total time:", total_time, "ms");
        }

        cudaDeviceSynchronize();
        return result;
    }

    BatchPowerFlowResult solve_power_flow_batch(std::vector<NetworkData> const& network_scenarios,
                                                SparseMatrix const& admittance_matrix,
                                                BatchPowerFlowConfig const& config) override {
        logger.setComponent("GPUIterativeCurrent::Batch");
        LOG_INFO(logger, "Starting GPU batch power flow solution");
        LOG_INFO(logger, "  Number of scenarios:", network_scenarios.size());
        LOG_INFO(logger, "  Reuse Y-bus factorization:", config.reuse_y_bus_factorization);

        if (network_scenarios.empty()) {
            LOG_WARN(logger, "Empty scenario list provided");
            return BatchPowerFlowResult{};
        }

        // Validate all scenarios have same topology
        int num_buses = network_scenarios[0].num_buses;
        for (size_t i = 1; i < network_scenarios.size(); ++i) {
            if (network_scenarios[i].num_buses != num_buses) {
                throw std::runtime_error("All scenarios must have same network topology");
            }
        }

        BatchPowerFlowResult batch_result;
        batch_result.results.reserve(network_scenarios.size());

        auto batch_start = std::chrono::high_resolution_clock::now();

        // Pre-factorize Y-bus if caching is enabled
        if (config.reuse_y_bus_factorization) {
            y_bus_factorized_ = false;
            LOG_INFO(logger, "Pre-factorizing Y-bus for GPU batch (one-time cost)");
            factorize_y_bus_if_needed(admittance_matrix, network_scenarios[0]);
            LOG_INFO(logger, "Y-bus factorization complete, will be reused for all",
                     network_scenarios.size(), "scenarios on GPU");
        }

        // Solve each scenario
        for (size_t i = 0; i < network_scenarios.size(); ++i) {
            auto const& network = network_scenarios[i];

            if (config.base_config.verbose || config.verbose_summary) {
                LOG_INFO(logger, "Solving GPU scenario", i + 1, "of", network_scenarios.size());
            }

            auto result = solve_power_flow(network, admittance_matrix, config.base_config);

            batch_result.results.push_back(result);
            batch_result.total_iterations += result.iterations;
            if (result.converged) {
                batch_result.converged_count++;
            } else {
                batch_result.failed_count++;
            }
        }

        auto batch_end = std::chrono::high_resolution_clock::now();
        batch_result.total_solve_time_ms =
            std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
        batch_result.avg_solve_time_ms =
            batch_result.total_solve_time_ms / network_scenarios.size();

        if (config.verbose_summary) {
            LOG_INFO(logger, "=== GPU Batch Solution Summary ===");
            LOG_INFO(logger, "  Total scenarios:", network_scenarios.size());
            LOG_INFO(logger, "  Converged:", batch_result.converged_count);
            LOG_INFO(logger, "  Failed:", batch_result.failed_count);
            LOG_INFO(logger, "  Total iterations:", batch_result.total_iterations);
            LOG_INFO(logger, "  Avg iterations per scenario:",
                     batch_result.total_iterations / static_cast<double>(network_scenarios.size()));
            LOG_INFO(logger, "  Total time:", batch_result.total_solve_time_ms, "ms");
            LOG_INFO(logger, "  Avg time per scenario:", batch_result.avg_solve_time_ms, "ms");
        }

        return batch_result;
    }

    std::vector<Float> calculate_mismatches(NetworkData const& /*network_data*/,
                                            ComplexVector const& /*bus_voltages*/,
                                            SparseMatrix const& /*admittance_matrix*/) override {
        return std::vector<Float>();
    }

    BackendType get_backend_type() const noexcept override { return BackendType::GPU_CUDA; }

    void enable_state_capture(bool enable) override { capture_states_ = enable; }

    std::vector<IterationState> const& get_iteration_states() const override {
        return iteration_states_;
    }

    void clear_iteration_states() override { iteration_states_.clear(); }
};

// CUDA Kernel Implementations
namespace gpu_ic_kernels {

/**
 * @brief Calculate current injections using iterative current method
 *
 * For slack buses: I = Y_source * U_ref
 * For load/gen buses: I = conj(S / V) with ZIP model support
 */
__global__ void calculate_ic_current_injections_kernel(
    cuDoubleComplex const* __restrict__ voltages,
    cuDoubleComplex const* __restrict__ specified_powers, int const* __restrict__ bus_types,
    int const* __restrict__ load_types, double const* __restrict__ u_pu,
    cuDoubleComplex* __restrict__ currents, int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        // Slack bus (BusType::SLACK = 2)
        if (bus_types[bus_idx] == 2) {
            // I_inj = Y_source * U_ref
            cuDoubleComplex u_ref = make_cuDoubleComplex(u_pu[bus_idx], 0.0);
            cuDoubleComplex y_source = make_cuDoubleComplex(1.0, -10.0);
            currents[bus_idx] = cuCmul(y_source, u_ref);
            return;
        }

        cuDoubleComplex s_spec = specified_powers[bus_idx];
        cuDoubleComplex v = voltages[bus_idx];

        // Check for zero power
        double s_mag = cuCabs(s_spec);
        if (s_mag < 1e-14) {
            currents[bus_idx] = make_cuDoubleComplex(0.0, 0.0);
            return;
        }

        // Avoid division by zero
        double v_mag = cuCabs(v);
        if (v_mag < 1e-10) {
            v = make_cuDoubleComplex(1e-10, 0.0);
            v_mag = 1e-10;
        }

        int load_type = load_types[bus_idx];

        // Calculate current based on ZIP load model
        // 0 = const_pq, 1 = const_y, 2 = const_i
        cuDoubleComplex current;

        if (load_type == 0) {
            // Constant power: I = conj(S / V)
            cuDoubleComplex s_over_v = cuCdiv(s_spec, v);
            current = cuConj(s_over_v);
        } else if (load_type == 1) {
            // Constant impedance: I = conj(S) * V
            current = cuCmul(cuConj(s_spec), v);
        } else {  // load_type == 2
            // Constant current: I = conj(S * |V| / V)
            cuDoubleComplex s_vmag = cuCmul(s_spec, make_cuDoubleComplex(v_mag, 0.0));
            cuDoubleComplex s_vmag_over_v = cuCdiv(s_vmag, v);
            current = cuConj(s_vmag_over_v);
        }

        currents[bus_idx] = current;
    }
}

/**
 * @brief Check voltage convergence by finding maximum voltage change
 */
__global__ void check_voltage_convergence_kernel(cuDoubleComplex const* __restrict__ old_voltages,
                                                 cuDoubleComplex const* __restrict__ new_voltages,
                                                 double* __restrict__ max_change, int num_buses) {
    __shared__ double shared_max[256];

    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double local_max = 0.0;

    if (bus_idx < num_buses) {
        cuDoubleComplex old_v = old_voltages[bus_idx];
        cuDoubleComplex new_v = new_voltages[bus_idx];
        cuDoubleComplex diff = cuCsub(new_v, old_v);
        local_max = cuCabs(diff);
    }

    shared_max[tid] = local_max;
    __syncthreads();

    // Reduction to find maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(max_change),
                  __double_as_longlong(shared_max[0]));
    }
}

/**
 * @brief Enforce slack bus voltage constraints
 */
__global__ void enforce_slack_voltage_kernel(cuDoubleComplex* __restrict__ voltages,
                                             int const* __restrict__ bus_types,
                                             double const* __restrict__ u_pu, int num_buses) {
    int bus_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bus_idx < num_buses) {
        // Slack bus (BusType::SLACK = 2)
        if (bus_types[bus_idx] == 2) {
            voltages[bus_idx] = make_cuDoubleComplex(u_pu[bus_idx], 0.0);
        }
    }
}

}  // namespace gpu_ic_kernels

// Factory function for dynamic loading
extern "C" {
__attribute__((visibility("default"))) IPowerFlowSolver* create_gpu_ic_powerflow_solver() {
    return new GPUIterativeCurrent();
}
}

}  // namespace gap::solver
