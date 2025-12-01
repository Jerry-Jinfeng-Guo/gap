#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <ranges>

#ifdef GAP_OPENMP_AVAILABLE
#include <omp.h>
#endif

#include "gap/logging/logger.h"
#include "gap/solver/powerflow_interface.h"

namespace gap::solver {

class CPUIterativeCurrent : public IPowerFlowSolver {
  private:
    std::shared_ptr<ILUSolver> lu_solver_;
    gap::logging::Logger& logger = gap::logging::global_logger;

    // Y-bus factorization caching
    bool y_bus_factorized_ = false;
    SparseMatrix cached_y_bus_;

    // State capture for debugging
    bool capture_states_ = false;
    std::vector<IterationState> iteration_states_;

    // Profiling data
    struct ProfilingData {
        double total_injection_time = 0.0;
        double total_factorize_time = 0.0;
        double total_solve_time = 0.0;
        double total_convergence_time = 0.0;
        int iteration_count = 0;
    };

    // Convergence diagnostics
    struct ConvergenceDiagnostics {
        std::vector<Float> mismatch_history;      // Max mismatch per iteration
        std::vector<Float> avg_mismatch_history;  // Average mismatch per iteration
        std::vector<int> slow_buses;              // Buses with slow convergence
        Float convergence_rate = 0.0;             // Estimated convergence rate
        bool is_monotonic = true;                 // Whether convergence is monotonic
    };

  public:
    PowerFlowResult solve_power_flow(NetworkData const& network_data,
                                     SparseMatrix const& admittance_matrix,
                                     PowerFlowConfig const& config) override {
        logger.setComponent("CPUIterativeCurrent");
        LOG_INFO(logger, "Starting iterative current power flow solution");
        LOG_DEBUG(logger, "  Number of buses:", network_data.num_buses);
        LOG_DEBUG(logger, "  Tolerance:", config.tolerance);
        LOG_DEBUG(logger, "  Max iterations:", config.max_iterations);
        LOG_DEBUG(logger, "  Base power:", config.base_power / 1e6, "MVA");

        // === PER-UNIT SYSTEM NORMALIZATION ===
        // Calculate base impedance: Z_base = V_base² / S_base
        Float v_base = network_data.buses[0].u_rated;                  // Base voltage in Volts
        Float base_impedance = (v_base * v_base) / config.base_power;  // Ohms

        LOG_INFO(logger, "  Base voltage:", v_base / 1e3, "kV");
        LOG_INFO(logger, "  Base impedance:", base_impedance, "Ohms");
        LOG_INFO(logger, "  Base power:", config.base_power / 1e6, "MVA");

        // Create per-unit admittance matrix: Y_pu = Y_siemens × Z_base
        SparseMatrix y_pu = admittance_matrix;  // Copy structure
        for (size_t i = 0; i < y_pu.values.size(); ++i) {
            y_pu.values[i] = admittance_matrix.values[i] * base_impedance;
        }

        // Debug: Log Y-bus diagonal elements
        LOG_INFO(logger, "  Y-bus diagonal elements (per-unit):");
        for (int bus = 0; bus < network_data.num_buses; ++bus) {
            if (bus < static_cast<int>(y_pu.row_ptr.size() - 1)) {
                for (int idx = y_pu.row_ptr[bus]; idx < y_pu.row_ptr[bus + 1]; ++idx) {
                    if (y_pu.col_idx[idx] == bus) {  // Diagonal element
                        LOG_INFO(logger, "    Y[", bus, ",", bus, "] =", y_pu.values[idx].real(),
                                 "+j", y_pu.values[idx].imag());
                    }
                }
            }
        }

        PowerFlowResult result;
        result.bus_voltages.resize(network_data.num_buses);

        // Initialize voltages (flat start)
        if (config.use_flat_start) {
            LOG_DEBUG(logger, "  Using flat start initialization");
            for (size_t i = 0; i < network_data.buses.size(); ++i) {
                if (network_data.buses[i].bus_type == BusType::SLACK) {  // Slack bus
                    result.bus_voltages[i] = Complex(network_data.buses[i].u_pu, 0.0);
                } else if (network_data.buses[i].bus_type == BusType::PV) {  // PV bus
                    result.bus_voltages[i] = Complex(network_data.buses[i].u_pu, 0.0);
                } else {                                          // PQ bus
                    result.bus_voltages[i] = Complex(0.98, 0.0);  // Slightly lower for load buses
                }
            }
        }

        // Initialize profiling
        ProfilingData profiling;
        ConvergenceDiagnostics diagnostics;
        auto total_start = std::chrono::high_resolution_clock::now();

#ifdef GAP_OPENMP_AVAILABLE
        int num_threads = omp_get_max_threads();
        LOG_INFO(logger, "  OpenMP enabled with", num_threads, "threads");
#else
        LOG_INFO(logger, "  OpenMP not available - running serially");
#endif

        // Clear previous iteration states if capturing
        if (capture_states_) {
            iteration_states_.clear();
        }

        // Factorize Y-bus once (cached for subsequent solves with same topology)
        // Note: Source admittance is added to diagonal during factorization
        factorize_y_bus_if_needed(y_pu, network_data, profiling);

        // Iterative current power flow iterations
        for (int iter = 0; iter < config.max_iterations; ++iter) {
            if (config.verbose) {
                LOG_DEBUG(logger, "  Iteration", (iter + 1));
            }
            profiling.iteration_count = iter + 1;

            // Step 1: Calculate current injections I = conj(S / V) for loads, I = Y_source * U_ref
            // for sources
            auto t1 = std::chrono::high_resolution_clock::now();
            ComplexVector current_injections(network_data.num_buses, Complex(0.0, 0.0));
            calculate_current_injections(network_data, result.bus_voltages, current_injections,
                                         config.base_power);
            auto t2 = std::chrono::high_resolution_clock::now();
            profiling.total_injection_time +=
                std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Step 2: Solve linear system Y * V = I for new voltages
            // Note: Y already includes source admittance on diagonal (added during factorization)
            auto t3 = std::chrono::high_resolution_clock::now();
            ComplexVector new_voltages = lu_solver_->solve(current_injections);
            auto t4 = std::chrono::high_resolution_clock::now();
            profiling.total_solve_time +=
                std::chrono::duration<double, std::milli>(t4 - t3).count();

            // Step 3: Enforce slack bus voltage constraint
            // PGM naturally maintains slack voltage through Y_source, but we enforce it explicitly
            // for numerical stability. PV buses are NOT supported in basic IC method.
            for (size_t i = 0; i < network_data.buses.size(); ++i) {
                if (network_data.buses[i].bus_type == BusType::SLACK) {
                    // Enforce exact slack voltage (both magnitude and angle)
                    new_voltages[i] = Complex(network_data.buses[i].u_pu, 0.0);
                }
                // Note: PV buses are not properly handled by basic iterative current method
                // They require reactive power iteration which is not implemented
            }

            // Step 4: Check convergence - max voltage change
            // PGM: Compare new_voltages (rhs_u_) with old voltages (u) before updating
            auto t5 = std::chrono::high_resolution_clock::now();
            Float max_voltage_change =
                calculate_max_voltage_change(result.bus_voltages, new_voltages);

            // Calculate average voltage change for diagnostics
            Float total_change = 0.0;
            for (int i = 0; i < network_data.num_buses; ++i) {
                total_change += std::abs(new_voltages[i] - result.bus_voltages[i]);
            }
            Float avg_voltage_change = total_change / network_data.num_buses;

            auto t6 = std::chrono::high_resolution_clock::now();
            profiling.total_convergence_time +=
                std::chrono::duration<double, std::milli>(t6 - t5).count();

            // Store convergence diagnostics
            diagnostics.mismatch_history.push_back(max_voltage_change);
            diagnostics.avg_mismatch_history.push_back(avg_voltage_change);

            // Check for non-monotonic convergence
            if (iter > 0 && max_voltage_change > diagnostics.mismatch_history[iter - 1]) {
                diagnostics.is_monotonic = false;
            }

            result.final_mismatch = max_voltage_change;

            // Step 5: Update voltages for next iteration
            result.bus_voltages = new_voltages;

            // Capture iteration state if enabled
            if (capture_states_) {
                IterationState state;
                state.iteration = iter;
                state.max_mismatch = max_voltage_change;
                state.voltages = result.bus_voltages;
                state.currents = current_injections;

                // Calculate power injections: S = V * conj(I)
                std::vector<Complex> powers(network_data.num_buses);
                for (int i = 0; i < network_data.num_buses; ++i) {
                    powers[i] = result.bus_voltages[i] * std::conj(current_injections[i]);
                }
                state.power_injections = powers;

                iteration_states_.push_back(state);
            }

            if (max_voltage_change < config.tolerance) {
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

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Print convergence diagnostics
        if (config.verbose || !result.converged) {
            LOG_INFO(logger, "\n=== CONVERGENCE DIAGNOSTICS ===");
            LOG_INFO(logger, "Convergence:", (result.converged ? "YES" : "NO"));
            LOG_INFO(logger, "Iterations:", result.iterations);
            LOG_INFO(logger, "Final mismatch:", result.final_mismatch);
            LOG_INFO(logger, "Monotonic convergence:", (diagnostics.is_monotonic ? "YES" : "NO"));

            // Calculate convergence rate from last few iterations
            if (diagnostics.mismatch_history.size() >= 3) {
                size_t n = diagnostics.mismatch_history.size();
                Float rate =
                    diagnostics.mismatch_history[n - 1] / diagnostics.mismatch_history[n - 2];
                diagnostics.convergence_rate = rate;
                LOG_INFO(logger, "Convergence rate (last iteration):", rate);
            }

            // Show convergence history for last 5 iterations
            if (diagnostics.mismatch_history.size() > 0) {
                LOG_INFO(logger, "Mismatch history (last",
                         std::min(size_t(5), diagnostics.mismatch_history.size()), "iterations):");
                size_t start = diagnostics.mismatch_history.size() > 5
                                   ? diagnostics.mismatch_history.size() - 5
                                   : 0;
                for (size_t i = start; i < diagnostics.mismatch_history.size(); ++i) {
                    LOG_INFO(logger, "  Iter", (i + 1), "- Max:", diagnostics.mismatch_history[i],
                             "Avg:", diagnostics.avg_mismatch_history[i]);
                }
            }
            LOG_INFO(logger, "================================\n");
        }

        // Print profiling summary
        LOG_INFO(logger, "\n=== PERFORMANCE PROFILING ===");
        LOG_INFO(logger, "Total iterations:", profiling.iteration_count);
        LOG_INFO(logger, "Total time:", total_time, "ms");
        LOG_INFO(logger, "\nTime breakdown (total / avg per iteration):");
        LOG_INFO(logger, "  Current injection:", profiling.total_injection_time, "ms /",
                 profiling.total_injection_time / profiling.iteration_count, "ms", "(",
                 100.0 * profiling.total_injection_time / total_time, "%)");
        LOG_INFO(logger, "  LU factorization:", profiling.total_factorize_time, "ms", "(one-time)",
                 "(", 100.0 * profiling.total_factorize_time / total_time, "%)");
        LOG_INFO(logger, "  Linear solve:", profiling.total_solve_time, "ms /",
                 profiling.total_solve_time / profiling.iteration_count, "ms", "(",
                 100.0 * profiling.total_solve_time / total_time, "%)");
        LOG_INFO(logger, "  Convergence check:", profiling.total_convergence_time, "ms /",
                 profiling.total_convergence_time / profiling.iteration_count, "ms", "(",
                 100.0 * profiling.total_convergence_time / total_time, "%)");

        double accounted = profiling.total_injection_time + profiling.total_factorize_time +
                           profiling.total_solve_time + profiling.total_convergence_time;
        LOG_INFO(logger, "  Other (overhead):", total_time - accounted, "ms", "(",
                 100.0 * (total_time - accounted) / total_time, "%)");
        LOG_INFO(logger, "============================\n");

        // Debug: Log final voltage magnitudes
        if (config.verbose) {
            LOG_DEBUG(logger, "  Final voltage magnitudes (p.u.):");
            for (size_t i = 0; i < result.bus_voltages.size(); ++i) {
                Float vm = std::abs(result.bus_voltages[i]);
                LOG_DEBUG(logger, "    Bus", (i + 1), ":", vm, "p.u.");
            }
        }

        return result;
    }

    void set_lu_solver(std::shared_ptr<ILUSolver> lu_solver) override {
        lu_solver_ = lu_solver;
        // Reset factorization cache when solver changes
        y_bus_factorized_ = false;
        LOG_DEBUG(logger, "CPUIterativeCurrent: LU solver backend set");
    }

    std::vector<Float> calculate_mismatches(NetworkData const& /*network_data*/,
                                            ComplexVector const& /*bus_voltages*/,
                                            SparseMatrix const& /*admittance_matrix*/) override {
        // For iterative current, mismatch is voltage change rather than power mismatch
        // This is mainly for interface compatibility
        LOG_DEBUG(logger, "CPUIterativeCurrent: calculate_mismatches called (not typically used)");

        // Return empty vector as this method is not central to iterative linear algorithm
        return std::vector<Float>();
    }

    BackendType get_backend_type() const noexcept override { return BackendType::CPU; }

    void enable_state_capture(bool enable) override { capture_states_ = enable; }

    std::vector<IterationState> const& get_iteration_states() const override {
        return iteration_states_;
    }

    void clear_iteration_states() override { iteration_states_.clear(); }

    /**
     * @brief Batch solve with Y-bus factorization reuse
     * This is THE critical optimization for iterative current method - factorize once,
     * solve many times with different load profiles
     */
    BatchPowerFlowResult solve_power_flow_batch(std::vector<NetworkData> const& network_scenarios,
                                                SparseMatrix const& admittance_matrix,
                                                BatchPowerFlowConfig const& config) override {
        logger.setComponent("CPUIterativeCurrent::Batch");
        LOG_INFO(logger, "Starting batch power flow solution");
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
                LOG_ERROR(logger, "Scenario", i,
                          "has different number of buses:", network_scenarios[i].num_buses, "vs",
                          num_buses);
                throw std::runtime_error("All scenarios must have same network topology");
            }
        }

        BatchPowerFlowResult batch_result;
        batch_result.results.reserve(network_scenarios.size());

        auto batch_start = std::chrono::high_resolution_clock::now();

        // Pre-factorize Y-bus if caching is enabled
        if (config.reuse_y_bus_factorization) {
            // Force factorization for first scenario
            y_bus_factorized_ = false;
            LOG_INFO(logger, "Pre-factorizing Y-bus for batch (one-time cost)");

            // Do a dry-run factorization with first scenario
            ProfilingData dummy_profiling;
            factorize_y_bus_if_needed(admittance_matrix, network_scenarios[0], dummy_profiling);

            LOG_INFO(logger, "Y-bus factorization complete, will be reused for all",
                     network_scenarios.size(), "scenarios");
        }

        // Solve each scenario
        for (size_t i = 0; i < network_scenarios.size(); ++i) {
            auto const& network = network_scenarios[i];

            if (config.base_config.verbose || config.verbose_summary) {
                LOG_INFO(logger, "Solving scenario", i + 1, "of", network_scenarios.size());
            }

            // Solve this scenario (Y-bus factorization will be reused automatically)
            auto result = solve_power_flow(network, admittance_matrix, config.base_config);

            // Accumulate statistics
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

        // Print summary
        if (config.verbose_summary) {
            LOG_INFO(logger, "=== Batch Solution Summary ===");
            LOG_INFO(logger, "  Total scenarios:", network_scenarios.size());
            LOG_INFO(logger, "  Converged:", batch_result.converged_count);
            LOG_INFO(logger, "  Failed:", batch_result.failed_count);
            LOG_INFO(logger, "  Total iterations:", batch_result.total_iterations);
            LOG_INFO(logger, "  Avg iterations per scenario:",
                     batch_result.total_iterations / static_cast<double>(network_scenarios.size()));
            LOG_INFO(logger, "  Total time:", batch_result.total_solve_time_ms, "ms");
            LOG_INFO(logger, "  Avg time per scenario:", batch_result.avg_solve_time_ms, "ms");
            LOG_INFO(logger, "  Y-bus factorization reused:",
                     config.reuse_y_bus_factorization ? "Yes" : "No");
        }

        return batch_result;
    }

  private:
    /**
     * @brief Factorize Y-bus matrix if not already factorized
     * Adds source admittance to diagonal before factorization (PGM approach)
     */
    void factorize_y_bus_if_needed(SparseMatrix const& y_matrix, NetworkData const& network_data,
                                   ProfilingData& profiling) {
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
            LOG_INFO(logger, "  Factorizing Y-bus matrix (one-time cost)");

            if (!lu_solver_) {
                LOG_ERROR(logger, "LU solver not initialized");
                return;
            }

            // Create modified Y-bus with source admittance added to diagonal
            // This matches PGM's initialize_derived_solver approach
            SparseMatrix y_modified = y_matrix;

            // Add source admittance to diagonal for slack buses
            for (size_t bus_idx = 0; bus_idx < network_data.buses.size(); ++bus_idx) {
                if (network_data.buses[bus_idx].bus_type == BusType::SLACK) {
                    // Find diagonal element for this bus
                    for (int idx = y_modified.row_ptr[bus_idx];
                         idx < y_modified.row_ptr[bus_idx + 1]; ++idx) {
                        if (y_modified.col_idx[idx] == static_cast<int>(bus_idx)) {
                            // Add source admittance to diagonal
                            // Y_source represents connection to infinite bus
                            Complex y_source(1.0, -10.0);  // Strong source impedance
                            y_modified.values[idx] += y_source;
                            LOG_DEBUG(logger, "  Added Y_source to bus", bus_idx,
                                      "diagonal:", y_source.real(), "+j", y_source.imag());
                            break;
                        }
                    }
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            lu_solver_->factorize(y_modified);
            auto t2 = std::chrono::high_resolution_clock::now();

            profiling.total_factorize_time +=
                std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Cache the factorized Y-bus
            cached_y_bus_ = y_modified;
            y_bus_factorized_ = true;

            LOG_INFO(logger, "  Y-bus factorization complete");
        } else {
            LOG_DEBUG(logger, "  Reusing cached Y-bus factorization");
        }
    }

    /**
     * @brief Calculate current injections from specified powers and voltages
     * Matches PGM implementation supporting ZIP load model:
     * - const_pq: I = conj(S / V) - Constant power
     * - const_y:  I = conj(S) * V - Constant impedance
     * - const_i:  I = conj(S * |V| / V) - Constant current
     * - For sources (slack): I = Y_source * U_ref
     */
    void calculate_current_injections(NetworkData const& network_data,
                                      ComplexVector const& voltages, ComplexVector& currents,
                                      Float base_power) const {
        LOG_TRACE(logger, "CPUIterativeCurrent: Calculating current injections");

        // Initialize all currents to zero
        std::fill(currents.begin(), currents.end(), Complex(0.0, 0.0));

        // Parallelize the loop over buses using OpenMP
        // Each bus calculation is independent
#ifdef GAP_OPENMP_AVAILABLE
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            if (network_data.buses[i].bus_type == BusType::SLACK) {
                // PGM approach: I_inj = Y_source * U_ref
                // This current is added to RHS, while Y_source is added to Y-bus diagonal
                Complex u_ref = Complex(network_data.buses[i].u_pu, 0.0);
                Complex y_source(1.0, -10.0);  // Must match the value added to Y-bus
                currents[i] = y_source * u_ref;
                continue;
            }

            // Get specified power for this bus (from appliances + direct bus specification)
            Complex specified_power = get_specified_power_at_bus(network_data, i, base_power);

            // Skip if no power specified (zero injection bus)
            if (std::abs(specified_power) < 1e-14) {
                currents[i] = Complex(0.0, 0.0);
                continue;
            }

            // Determine load type from connected appliances
            // Default to constant power if no appliance specifies otherwise
            LoadGenType load_type = LoadGenType::const_pq;
            int bus_id = network_data.buses[i].id;

            for (auto const& appliance : network_data.appliances) {
                if (appliance.node == bus_id && appliance.status == 1 &&
                    appliance.type == ApplianceType::LOADGEN) {
                    load_type = appliance.load_gen_type;
                    break;  // Use first matching appliance's type
                }
            }

            Complex voltage = voltages[i];

            // Avoid division by zero
            if (std::abs(voltage) < 1e-10) {
                LOG_WARN(logger, "  Very low voltage at bus", i,
                         "- using small value for current calculation");
                voltage = Complex(1e-10, 0.0);
            }

            // Calculate current based on ZIP load model
            // PGM formulas from iterative_current_pf_solver.hpp
            switch (load_type) {
                case LoadGenType::const_pq:
                    // Constant power: I = conj(S / V)
                    currents[i] = std::conj(specified_power / voltage);
                    break;

                case LoadGenType::const_y:
                    // Constant impedance: I = conj(S * V^2 / V) = conj(S) * V
                    // At rated voltage: I = conj(S_rated / V_rated)
                    // At actual voltage: I = conj(S_rated) * V
                    currents[i] = std::conj(specified_power) * voltage;
                    break;

                case LoadGenType::const_i:
                    // Constant current: I = conj(S * |V| / V)
                    // Maintains constant current magnitude
                    currents[i] = std::conj(specified_power * std::abs(voltage) / voltage);
                    break;
            }
        }
    }

    /**
     * @brief Solve linear system Y * V = I using cached LU factorization
     */
    void solve_linear_system(SparseMatrix const& /*y_matrix*/, ComplexVector const& currents,
                             ComplexVector& voltages) const {
        LOG_TRACE(logger, "CPUIterativeCurrent: Solving linear system Y*V=I");

        if (!lu_solver_) {
            LOG_ERROR(logger, "LU solver not initialized");
            return;
        }

        // Solve using cached factorization (no refactorization needed)
        voltages = lu_solver_->solve(currents);
    }

    /**
     * @brief Enforce voltage magnitude constraints for PV buses and slack bus
     */
    void enforce_voltage_constraints(NetworkData const& network_data,
                                     ComplexVector& voltages) const {
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            if (network_data.buses[i].bus_type == BusType::SLACK) {
                // Slack bus: fix both magnitude and angle
                voltages[i] = Complex(network_data.buses[i].u_pu, 0.0);
            } else if (network_data.buses[i].bus_type == BusType::PV) {
                // PV bus: fix magnitude, keep calculated angle
                Float angle = std::arg(voltages[i]);
                Float target_magnitude = network_data.buses[i].u_pu;
                voltages[i] = std::polar(target_magnitude, angle);
            }
            // PQ buses: no constraints, use calculated values
        }
    }

    /**
     * @brief Calculate maximum voltage change between iterations
     */
    Float calculate_max_voltage_change(ComplexVector const& v_old,
                                       ComplexVector const& v_new) const {
        Float max_change = 0.0;

        for (size_t i = 0; i < v_old.size(); ++i) {
            Float change = std::abs(v_new[i] - v_old[i]);
            max_change = std::max(max_change, change);
        }

        return max_change;
    }

    /**
     * @brief Get specified power injection at a bus from connected appliances and bus data
     * @param network_data Network data with bus and appliance information
     * @param bus_idx Bus index
     * @param base_power Base power for per-unit conversion (VA)
     * @return Specified power in per-unit
     */
    Complex get_specified_power_at_bus(NetworkData const& network_data, size_t bus_idx,
                                       Float base_power) const {
        Complex specified_power(0.0, 0.0);
        int bus_id = network_data.buses[bus_idx].id;

        // First, check if power is specified directly on the bus
        if (network_data.buses[bus_idx].active_power != 0.0 ||
            network_data.buses[bus_idx].reactive_power != 0.0) {
            specified_power = Complex(network_data.buses[bus_idx].active_power,
                                      network_data.buses[bus_idx].reactive_power);
        }

        // Also sum power from all appliances connected to this bus
        for (auto const& appliance : network_data.appliances) {
            if (appliance.node == bus_id && appliance.status == 1 &&
                (appliance.type == ApplianceType::SOURCE ||
                 appliance.type == ApplianceType::LOADGEN)) {
                // Sources inject power (positive P), loads consume power (negative P)
                specified_power += Complex(appliance.p_specified, appliance.q_specified);
            }
            // SHUNT appliances are handled in admittance matrix
        }

        // Convert to per-unit
        if (base_power > 0.0) {
            specified_power /= base_power;
        }

        return specified_power;
    }
};

// Register the CPU iterative current solver implementation
// This allows the factory to create instances of this solver
namespace {
[[maybe_unused]] auto cpu_iterative_current_registration = []() {
    // Registration logic can be added here if needed
    return 0;
}();
}  // namespace

}  // namespace gap::solver
