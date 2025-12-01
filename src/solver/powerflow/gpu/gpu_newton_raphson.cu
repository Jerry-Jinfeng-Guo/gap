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

    // State capture for debugging
    bool capture_states_ = false;
    std::vector<IterationState> iteration_states_;

    // GPU-resident data (persistent across iterations)
    GPUSparseMatrix gpu_admittance_;
    GPUPowerFlowData gpu_data_;

    // Index mapping arrays on device
    int* d_mismatch_indices_ = nullptr;  // bus_id -> mismatch vector index
    int* d_voltage_indices_ = nullptr;   // bus_id -> voltage correction index (for complex update)
    int* d_angle_var_idx_ = nullptr;     // bus_id -> angle variable index in Jacobian
    int* d_mag_var_idx_ = nullptr;       // bus_id -> magnitude variable index in Jacobian
    double* d_specified_magnitudes_ = nullptr;
    int* d_load_types_ = nullptr;  // Load model types (ZIP model: 0=const_pq, 1=const_y, 2=const_i)
    cuDoubleComplex* d_currents_ = nullptr;  // Current injections buffer
    cuDoubleComplex* d_powers_ = nullptr;    // Power injections S = V * conj(I)
    double* d_jacobian_ = nullptr;           // Jacobian matrix (dense, row-major)
    double* d_corrections_ = nullptr;        // Voltage corrections from solver

    // Cached network info
    int num_buses_ = 0;
    int num_unknowns_ = 0;
    bool data_initialized_ = false;
    std::vector<int> h_bus_types_;  // Host copy of bus types for voltage updates

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
        if (d_angle_var_idx_) cudaFree(d_angle_var_idx_);
        if (d_mag_var_idx_) cudaFree(d_mag_var_idx_);
        if (d_specified_magnitudes_) cudaFree(d_specified_magnitudes_);
        if (d_load_types_) cudaFree(d_load_types_);
        if (d_currents_) cudaFree(d_currents_);
        if (d_powers_) cudaFree(d_powers_);
        if (d_jacobian_) cudaFree(d_jacobian_);
        if (d_corrections_) cudaFree(d_corrections_);

        d_mismatch_indices_ = nullptr;
        d_voltage_indices_ = nullptr;
        d_angle_var_idx_ = nullptr;
        d_mag_var_idx_ = nullptr;
        d_specified_magnitudes_ = nullptr;
        d_load_types_ = nullptr;
        d_currents_ = nullptr;
        d_powers_ = nullptr;
        d_jacobian_ = nullptr;
        d_corrections_ = nullptr;

        data_initialized_ = false;
    }

    void initialize_gpu_data(NetworkData const& network_data, SparseMatrix const& admittance_matrix,
                             PowerFlowConfig const& config) {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("GPUNewtonRaphson");

        LOG_INFO(logger, "Initializing GPU-resident data structures");

        // Clean up any existing data
        cleanup_gpu_data();

        num_buses_ = network_data.num_buses;

        // === PER-UNIT SYSTEM NORMALIZATION ===
        // Calculate base impedance: Z_base = V_base² / S_base
        Float v_base = network_data.buses[0].u_rated;               // Base voltage in Volts
        Float inv_base_power = 1.0 / config.base_power;             // 1/VA (avoid division)
        Float base_impedance = (v_base * v_base) * inv_base_power;  // Ohms

        LOG_INFO(logger, "  Base voltage:", v_base * 1e-3,
                 "kV");  // multiply by 1e-3 instead of divide by 1e3
        LOG_INFO(logger, "  Base impedance:", base_impedance, "Ohms");
        LOG_INFO(logger, "  Base power:", config.base_power / 1e6, "MVA");

        // Create per-unit admittance matrix: Y_pu = Y_siemens × Z_base
        SparseMatrix y_pu = admittance_matrix;  // Copy structure
        for (size_t i = 0; i < y_pu.values.size(); ++i) {
            y_pu.values[i] = admittance_matrix.values[i] * base_impedance;
        }

        num_buses_ = network_data.num_buses;

        // Validate inputs
        if (num_buses_ == 0 || network_data.buses.empty()) {
            throw std::runtime_error("Invalid network data: no buses");
        }

        if (y_pu.nnz == 0 || y_pu.row_ptr.empty()) {
            throw std::runtime_error("Invalid admittance matrix: empty or uninitialized");
        }

        LOG_INFO(logger, "  Admittance matrix: ", y_pu.num_rows, "x", y_pu.num_cols,
                 ", nnz=", y_pu.nnz);

        // Count unknowns and create index mappings
        std::vector<int> h_mismatch_indices(num_buses_, -1);
        std::vector<int> h_voltage_indices(num_buses_, -1);
        std::vector<int> h_angle_var_idx(num_buses_, -1);  // Angle variable index
        std::vector<int> h_mag_var_idx(num_buses_, -1);    // Magnitude variable index
        std::vector<double> h_magnitudes(num_buses_, 1.0);
        std::vector<int> h_bus_types(num_buses_);

        int mismatch_idx = 0;
        int voltage_idx = 0;
        int angle_idx = 0;  // Index for angle unknowns in Jacobian
        int mag_idx = 0;    // Index for magnitude unknowns in Jacobian (starts after all angles)

        // First pass: count angle variables
        int num_angles = 0;
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            auto const& bus = network_data.buses[i];
            if (bus.bus_type != BusType::SLACK) {
                num_angles++;  // All non-slack buses have angle unknowns
            }
        }

        // Second pass: assign variable indices
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            auto const& bus = network_data.buses[i];
            int bus_type_val = static_cast<int>(bus.bus_type);
            h_bus_types[i] = bus_type_val;
            h_magnitudes[i] = bus.u_pu;

            if (bus.bus_type == BusType::SLACK) {
                // Slack bus: no unknowns
                continue;
            } else if (bus.bus_type == BusType::PQ) {
                // PQ bus: P and Q unknowns, angle and magnitude variables
                h_mismatch_indices[i] = mismatch_idx;
                h_voltage_indices[i] = voltage_idx;
                h_angle_var_idx[i] = angle_idx;
                h_mag_var_idx[i] = num_angles + mag_idx;  // Magnitudes come after angles
                mismatch_idx += 2;                        // P and Q
                voltage_idx += 2;                         // Real and imag parts
                angle_idx++;
                mag_idx++;
            } else if (bus.bus_type == BusType::PV) {
                // PV bus: only P unknown, only angle variable (magnitude fixed)
                h_mismatch_indices[i] = mismatch_idx;
                h_voltage_indices[i] = voltage_idx;
                h_angle_var_idx[i] = angle_idx;
                // h_mag_var_idx[i] remains -1 (no magnitude variable)
                mismatch_idx += 1;  // Only P
                voltage_idx += 1;   // Only angle (magnitude fixed)
                angle_idx++;
            }
        }

        num_unknowns_ = mismatch_idx;

        LOG_INFO(logger, "  Number of buses:", num_buses_);
        LOG_INFO(logger, "  Number of unknowns:", num_unknowns_);

        // Store bus types for later use in voltage updates
        h_bus_types_ = h_bus_types;

        // Allocate GPU memory for per-unit admittance matrix
        gpu_admittance_.allocate_device(y_pu.num_rows, y_pu.num_cols, y_pu.nnz);

        // Copy per-unit admittance matrix to GPU
        cudaMemcpy(gpu_admittance_.d_row_ptr, y_pu.row_ptr.data(), (num_buses_ + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_admittance_.d_col_idx, y_pu.col_idx.data(), y_pu.nnz * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Convert Complex to cuDoubleComplex
        std::vector<cuDoubleComplex> cu_values(y_pu.nnz);
        for (int i = 0; i < y_pu.nnz; ++i) {
            cu_values[i] = make_cuDoubleComplex(y_pu.values[i].real(), y_pu.values[i].imag());
        }
        cudaMemcpy(gpu_admittance_.d_values, cu_values.data(),
                   admittance_matrix.nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        // Allocate power flow data structures
        gpu_data_.allocate_device(num_buses_, num_unknowns_);

        // Allocate index mapping arrays
        cudaMalloc(&d_mismatch_indices_, num_buses_ * sizeof(int));
        cudaMalloc(&d_voltage_indices_, num_buses_ * sizeof(int));
        cudaMalloc(&d_angle_var_idx_, num_buses_ * sizeof(int));
        cudaMalloc(&d_mag_var_idx_, num_buses_ * sizeof(int));
        cudaMalloc(&d_specified_magnitudes_, num_buses_ * sizeof(double));
        cudaMalloc(&d_load_types_, num_buses_ * sizeof(int));
        cudaMalloc(&d_currents_, num_buses_ * sizeof(cuDoubleComplex));
        cudaMalloc(&d_powers_, num_buses_ * sizeof(cuDoubleComplex));
        cudaMalloc(&d_jacobian_, num_unknowns_ * num_unknowns_ * sizeof(double));
        cudaMalloc(&d_corrections_, num_unknowns_ * sizeof(double));

        // Copy index mappings to device
        cudaMemcpy(d_mismatch_indices_, h_mismatch_indices.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_voltage_indices_, h_voltage_indices.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_angle_var_idx_, h_angle_var_idx.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_mag_var_idx_, h_mag_var_idx.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_specified_magnitudes_, h_magnitudes.data(), num_buses_ * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_data_.d_bus_types, h_bus_types.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Copy power injections to device
        // Aggregate power from buses and appliances and convert to per-unit
        std::vector<cuDoubleComplex> h_power_inj(num_buses_, make_cuDoubleComplex(0.0, 0.0));
        std::vector<int> h_load_types(num_buses_, 0);  // Default: const_pq

        // First, power specified directly on buses
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            auto const& bus = network_data.buses[i];
            // Normalize to per-unit using multiplication
            h_power_inj[i] = make_cuDoubleComplex(bus.active_power * inv_base_power,
                                                  bus.reactive_power * inv_base_power);
        }

        // Add power from appliances connected to buses and extract load types
        for (auto const& appliance : network_data.appliances) {
            if (appliance.status == 1 && (appliance.type == ApplianceType::SOURCE ||
                                          appliance.type == ApplianceType::LOADGEN)) {
                // Find bus index for this appliance
                for (size_t i = 0; i < network_data.buses.size(); ++i) {
                    if (network_data.buses[i].id == appliance.node) {
                        cuDoubleComplex existing = h_power_inj[i];
                        // Normalize appliance power to per-unit using multiplication
                        h_power_inj[i] = make_cuDoubleComplex(
                            cuCreal(existing) + appliance.p_specified * inv_base_power,
                            cuCimag(existing) + appliance.q_specified * inv_base_power);

                        // Extract load type for ZIP model (use first LOADGEN type found)
                        if (appliance.type == ApplianceType::LOADGEN) {
                            h_load_types[i] = static_cast<int>(appliance.load_gen_type);
                        }
                        break;
                    }
                }
            }
        }

        cudaMemcpy(gpu_data_.d_power_injections, h_power_inj.data(),
                   num_buses_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_load_types_, h_load_types.data(), num_buses_ * sizeof(int),
                   cudaMemcpyHostToDevice);

        data_initialized_ = true;

        LOG_INFO(logger, "GPU data initialization complete");
        LOG_INFO(logger, "  Admittance matrix: ", y_pu.nnz, " non-zeros");
    }

    /**
     * @brief Convert dense Jacobian on GPU to sparse CSR format
     */
    SparseMatrix convert_dense_to_sparse_jacobian(double* d_dense_jacobian, int n,
                                                  double threshold) {
        // Copy dense matrix from device to host
        std::vector<double> h_jacobian(n * n);
        cudaMemcpy(h_jacobian.data(), d_dense_jacobian, n * n * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // Convert to sparse CSR format
        SparseMatrix sparse;
        sparse.num_rows = n;
        sparse.num_cols = n;
        sparse.row_ptr.resize(n + 1, 0);

        int nnz = 0;
        sparse.row_ptr[0] = 0;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double val = h_jacobian[i * n + j];
                if (std::abs(val) > threshold) {
                    sparse.col_idx.push_back(j);
                    sparse.values.push_back(Complex(val, 0.0));
                    nnz++;
                }
            }
            sparse.row_ptr[i + 1] = nnz;
        }

        sparse.nnz = nnz;
        return sparse;
    }

    /**
     * @brief Apply voltage corrections using Newton method
     */
    void apply_voltage_corrections(std::vector<double> const& corrections, double damping = 0.9) {
        if (corrections.size() != static_cast<size_t>(num_unknowns_)) {
            throw std::runtime_error("Corrections size mismatch");
        }

        // Get current voltages from GPU
        gpu_data_.sync_voltages_from_device();

        int corr_idx = 0;

        // Update angles (all non-slack buses)
        for (int i = 0; i < num_buses_; ++i) {
            if (h_bus_types_[i] == 2) continue;  // Skip slack (BusType::SLACK = 2)

            if (corr_idx < num_unknowns_) {
                Complex& v = gpu_data_.h_voltages[i];
                double v_mag = std::abs(v);
                double v_angle = std::arg(v);

                double delta_theta = damping * corrections[corr_idx];
                if (std::abs(delta_theta) > 0.15) {
                    delta_theta = 0.15 * (delta_theta > 0 ? 1.0 : -1.0);
                }
                v_angle += delta_theta;

                gpu_data_.h_voltages[i] = Complex(v_mag * cos(v_angle), v_mag * sin(v_angle));

                corr_idx++;
            }
        }

        // Update magnitudes (PQ buses only)
        for (int i = 0; i < num_buses_; ++i) {
            if (h_bus_types_[i] != 0) continue;  // Only PQ (BusType::PQ = 0)

            if (corr_idx < num_unknowns_) {
                Complex& v = gpu_data_.h_voltages[i];
                double v_mag = std::abs(v);
                double v_angle = std::arg(v);

                double delta_vmag = damping * corrections[corr_idx];
                if (std::abs(delta_vmag) > 0.08) {
                    delta_vmag = 0.08 * (delta_vmag > 0 ? 1.0 : -1.0);
                }
                v_mag += delta_vmag;
                v_mag = std::max(v_mag, 0.5);
                v_mag = std::min(v_mag, 1.5);

                gpu_data_.h_voltages[i] = Complex(v_mag * cos(v_angle), v_mag * sin(v_angle));
                corr_idx++;
            }
        }

        // Copy back to GPU
        std::vector<cuDoubleComplex> cu_voltages(num_buses_);
        for (int i = 0; i < num_buses_; ++i) {
            cu_voltages[i] = make_cuDoubleComplex(gpu_data_.h_voltages[i].real(),
                                                  gpu_data_.h_voltages[i].imag());
        }
        cudaMemcpy(gpu_data_.d_voltages, cu_voltages.data(), num_buses_ * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice);
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

        // Clear previous iteration states if capturing
        if (capture_states_) {
            iteration_states_.clear();
            LOG_INFO(logger, "  State capture enabled");
        }

        PowerFlowResult result;
        result.bus_voltages.resize(network_data.num_buses);

        // Initialize or update GPU data
        if (!data_initialized_ || num_buses_ != network_data.num_buses) {
            initialize_gpu_data(network_data, admittance_matrix, config);
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
            // Calculate current injections I = Y * V
            gpu_kernels::launch_calculate_current_injections(
                gpu_admittance_.d_row_ptr, gpu_admittance_.d_col_idx, gpu_admittance_.d_values,
                gpu_data_.d_voltages, d_currents_, num_buses_);

            // Step 2: Calculate power mismatches
            // First zero out the mismatch array (important for sparse writes)
            cudaMemset(gpu_data_.d_mismatches, 0, num_unknowns_ * sizeof(double));

            gpu_kernels::launch_calculate_power_mismatches(
                gpu_data_.d_voltages, d_currents_, gpu_data_.d_power_injections,
                gpu_data_.d_bus_types, d_load_types_, gpu_data_.d_mismatches, num_buses_,
                d_mismatch_indices_);

            // Step 3: Check convergence (sync mismatches to host for now)
            gpu_data_.sync_mismatches_from_device();

            double max_mismatch = 0.0;
            for (double mismatch : gpu_data_.h_mismatches) {
                max_mismatch = std::max(max_mismatch, std::abs(mismatch));
            }

            result.final_mismatch = max_mismatch;

            // Capture iteration state if enabled
            if (capture_states_) {
                IterationState state;
                state.iteration = iter;
                state.max_mismatch = max_mismatch;
                state.mismatches = gpu_data_.h_mismatches;

                // Copy voltages from device
                gpu_data_.sync_voltages_from_device();
                state.voltages = gpu_data_.h_voltages;

                // Copy currents from device
                std::vector<cuDoubleComplex> h_currents(num_buses_);
                cudaMemcpy(h_currents.data(), d_currents_, num_buses_ * sizeof(cuDoubleComplex),
                           cudaMemcpyDeviceToHost);
                state.currents.resize(num_buses_);
                for (int i = 0; i < num_buses_; ++i) {
                    state.currents[i] = Complex(cuCreal(h_currents[i]), cuCimag(h_currents[i]));
                }

                // Copy power injections from device
                std::vector<cuDoubleComplex> h_powers(num_buses_);
                cudaMemcpy(h_powers.data(), gpu_data_.d_power_injections,
                           num_buses_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
                state.power_injections.resize(num_buses_);
                for (int i = 0; i < num_buses_; ++i) {
                    state.power_injections[i] = Complex(cuCreal(h_powers[i]), cuCimag(h_powers[i]));
                }

                iteration_states_.push_back(state);
            }

            if (config.verbose) {
                LOG_INFO(logger, "  Iteration", iter + 1, "- Max mismatch:", max_mismatch);
            }

            if (max_mismatch < config.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                LOG_INFO(logger, "Converged in", iter + 1, "iterations");
                break;
            }

            // Step 4: Build Jacobian matrix on GPU
            // First, calculate power injections S = V * conj(I)
            gpu_kernels::launch_calculate_power_injections(gpu_data_.d_voltages, d_currents_,
                                                           d_powers_, num_buses_);

            // Zero out the Jacobian matrix before building (important for sparse writes)
            cudaMemset(d_jacobian_, 0, num_unknowns_ * num_unknowns_ * sizeof(double));

            // Build full Jacobian matrix (dense format for cuDSS)
            gpu_kernels::launch_build_jacobian_dense(
                gpu_admittance_.d_row_ptr, gpu_admittance_.d_col_idx, gpu_admittance_.d_values,
                gpu_data_.d_voltages, d_powers_, gpu_data_.d_bus_types,
                d_angle_var_idx_,  // Correct: angle variable indices
                d_mag_var_idx_,    // Correct: magnitude variable indices
                d_jacobian_, num_buses_, num_unknowns_);

            // Step 5: Solve linear system J * ΔX = -F using GPU solver
            if (!lu_solver_) {
                LOG_ERROR(logger, "LU solver not set");
                break;
            }

            // Convert dense Jacobian to sparse CSR format
            // For power flow, Jacobian is sparse (similar pattern to Y-bus)
            SparseMatrix jacobian_sparse = convert_dense_to_sparse_jacobian(
                d_jacobian_, num_unknowns_, 1e-12  // threshold for sparsity
            );

            // Factorize Jacobian (only once, or when structure changes)
            bool factorized = lu_solver_->factorize(jacobian_sparse);
            if (!factorized) {
                LOG_ERROR(logger, "Failed to factorize Jacobian at iteration", iter + 1);
                break;
            }

            // Prepare RHS: -F (negative mismatches as complex vector)
            ComplexVector rhs_complex(num_unknowns_);
            for (int i = 0; i < num_unknowns_; ++i) {
                rhs_complex[i] = Complex(-gpu_data_.h_mismatches[i], 0.0);
            }

            // Solve J * ΔX = -F
            ComplexVector corrections_complex = lu_solver_->solve(rhs_complex);

            // Extract real part (Jacobian is real for power flow)
            std::vector<double> corrections(num_unknowns_);
            for (int i = 0; i < num_unknowns_; ++i) {
                corrections[i] = corrections_complex[i].real();
            }

            // Copy corrections to GPU
            cudaMemcpy(d_corrections_, corrections.data(), num_unknowns_ * sizeof(double),
                       cudaMemcpyHostToDevice);

            // Step 6: Update voltages on GPU with corrections
            apply_voltage_corrections(corrections, config.acceleration_factor);

            if (config.verbose) {
                LOG_INFO(logger, "  TODO: Complete Jacobian solve and voltage update");
            }

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

    // State capture methods for debugging
    void enable_state_capture(bool enable) override { capture_states_ = enable; }

    const std::vector<IterationState>& get_iteration_states() const override {
        return iteration_states_;
    }

    void clear_iteration_states() override { iteration_states_.clear(); }
};

}  // namespace gap::solver

// C-style interface for dynamic loading
extern "C" {
gap::solver::IPowerFlowSolver* create_gpu_powerflow_solver() {
    return new gap::solver::GPUNewtonRaphson();
}

void destroy_gpu_powerflow_solver(gap::solver::IPowerFlowSolver* instance) { delete instance; }
}
