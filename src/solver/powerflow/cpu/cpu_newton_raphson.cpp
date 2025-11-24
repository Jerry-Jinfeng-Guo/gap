#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <ranges>

#include "gap/logging/logger.h"
#include "gap/solver/powerflow_interface.h"

namespace gap::solver {

class CPUNewtonRaphson : public IPowerFlowSolver {
  private:
    std::shared_ptr<ILUSolver> lu_solver_;
    gap::logging::Logger& logger = gap::logging::global_logger;

    // State capture for debugging
    bool capture_states_ = false;
    std::vector<IterationState> iteration_states_;

    // Profiling data
    struct ProfilingData {
        double total_mismatch_time = 0.0;
        double total_jacobian_time = 0.0;
        double total_factorize_time = 0.0;
        double total_solve_time = 0.0;
        double total_update_time = 0.0;
        int iteration_count = 0;
    };

  public:
    PowerFlowResult solve_power_flow(NetworkData const& network_data,
                                     SparseMatrix const& admittance_matrix,
                                     PowerFlowConfig const& config) override {
        logger.setComponent("CPUNewtonRaphson");
        LOG_INFO(logger, "Starting power flow solution");
        LOG_DEBUG(logger, "  Number of buses:", network_data.num_buses);
        LOG_DEBUG(logger, "  Tolerance:", config.tolerance);
        LOG_DEBUG(logger, "  Max iterations:", config.max_iterations);
        LOG_DEBUG(logger, "  Base power:", config.base_power / 1e6, "MVA");

        // === PER-UNIT SYSTEM NORMALIZATION ===
        // Calculate base impedance: Z_base = V_base² / S_base
        // For now, assume all buses have the same u_rated (single voltage level)
        Float v_base = network_data.buses[0].u_rated;                  // Base voltage in Volts
        Float base_impedance = (v_base * v_base) / config.base_power;  // Ohms

        LOG_INFO(logger, "  Base voltage:", v_base / 1e3, "kV");
        LOG_INFO(logger, "  Base impedance:", base_impedance, "Ohms");
        LOG_INFO(logger, "  Base power:", config.base_power / 1e6, "MVA");

        // Create per-unit admittance matrix: Y_pu = Y_siemens × Z_base
        // The admittance matrix from the builder is in Siemens (Y = 1/Z with Z in Ohms)
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

        // Initialize voltages (flat start or previous solution)
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
        auto total_start = std::chrono::high_resolution_clock::now();

        // Clear previous iteration states if capturing
        if (capture_states_) {
            iteration_states_.clear();
        }

        // Newton-Raphson iterations
        for (int iter = 0; iter < config.max_iterations; ++iter) {
            if (config.verbose) {
                LOG_DEBUG(logger, "  Iteration", (iter + 1));
            }
            profiling.iteration_count = iter + 1;

            // Calculate mismatches (uses per-unit Y-bus and base_power)
            auto t1 = std::chrono::high_resolution_clock::now();
            auto mismatches = calculate_mismatches_impl(network_data, result.bus_voltages, y_pu,
                                                        config.base_power);
            auto t2 = std::chrono::high_resolution_clock::now();
            profiling.total_mismatch_time +=
                std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Check convergence
            Float max_mismatch = 0.0;
            if (!mismatches.empty()) {
                auto max_iter = std::ranges::max_element(
                    mismatches, [](Float a, Float b) { return std::abs(a) < std::abs(b); });
                max_mismatch = std::abs(*max_iter);
            }

            result.final_mismatch = max_mismatch;

            // Capture iteration state if enabled
            if (capture_states_) {
                IterationState state;
                state.iteration = iter;
                state.max_mismatch = max_mismatch;
                state.voltages = result.bus_voltages;
                state.mismatches = mismatches;

                // Calculate currents: I = Y * V
                std::vector<Complex> currents(network_data.num_buses, Complex(0.0, 0.0));
                for (int i = 0; i < network_data.num_buses; ++i) {
                    for (int j_idx = y_pu.row_ptr[i]; j_idx < y_pu.row_ptr[i + 1]; ++j_idx) {
                        int j = y_pu.col_idx[j_idx];
                        currents[i] += y_pu.values[j_idx] * result.bus_voltages[j];
                    }
                }
                state.currents = currents;

                // Calculate power injections: S = V * conj(I)
                std::vector<Complex> powers(network_data.num_buses);
                for (int i = 0; i < network_data.num_buses; ++i) {
                    powers[i] = result.bus_voltages[i] * std::conj(currents[i]);
                }
                state.power_injections = powers;

                iteration_states_.push_back(state);
            }

            if (max_mismatch < config.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                LOG_INFO(logger, "  Converged in", (iter + 1), "iterations");
                break;
            }

            // Build Jacobian matrix and solve Newton's correction equations
            auto corrections = solve_newton_correction(network_data, mismatches,
                                                       result.bus_voltages, y_pu, profiling);

            // Update voltage estimates with Newton's method: V = V + ΔV
            auto t_update_start = std::chrono::high_resolution_clock::now();
            update_voltage_estimates(network_data, result.bus_voltages, corrections);
            auto t_update_end = std::chrono::high_resolution_clock::now();
            profiling.total_update_time +=
                std::chrono::duration<double, std::milli>(t_update_end - t_update_start).count();
        }

        if (!result.converged) {
            LOG_WARN(logger, "  Failed to converge after", config.max_iterations, "iterations");
            result.iterations = config.max_iterations;
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Print profiling summary
        LOG_INFO(logger, "\n=== PERFORMANCE PROFILING ===");
        LOG_INFO(logger, "Total iterations:", profiling.iteration_count);
        LOG_INFO(logger, "Total time:", total_time, "ms");
        LOG_INFO(logger, "\nTime breakdown (total / avg per iteration):");
        LOG_INFO(logger, "  Mismatch calculation:", profiling.total_mismatch_time, "ms /",
                 profiling.total_mismatch_time / profiling.iteration_count, "ms", "(",
                 100.0 * profiling.total_mismatch_time / total_time, "%)");
        LOG_INFO(logger, "  Jacobian construction:", profiling.total_jacobian_time, "ms /",
                 profiling.total_jacobian_time / profiling.iteration_count, "ms", "(",
                 100.0 * profiling.total_jacobian_time / total_time, "%)");
        LOG_INFO(logger, "  LU factorization:", profiling.total_factorize_time, "ms /",
                 profiling.total_factorize_time / profiling.iteration_count, "ms", "(",
                 100.0 * profiling.total_factorize_time / total_time, "%)");
        LOG_INFO(logger, "  Linear solve:", profiling.total_solve_time, "ms /",
                 profiling.total_solve_time / profiling.iteration_count, "ms", "(",
                 100.0 * profiling.total_solve_time / total_time, "%)");
        LOG_INFO(logger, "  Voltage update:", profiling.total_update_time, "ms /",
                 profiling.total_update_time / profiling.iteration_count, "ms", "(",
                 100.0 * profiling.total_update_time / total_time, "%)");

        double accounted = profiling.total_mismatch_time + profiling.total_jacobian_time +
                           profiling.total_factorize_time + profiling.total_solve_time +
                           profiling.total_update_time;
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
        LOG_DEBUG(logger, "CPUNewtonRaphson: LU solver backend set");
    }

    std::vector<Float> calculate_mismatches(NetworkData const& network_data,
                                            ComplexVector const& bus_voltages,
                                            SparseMatrix const& admittance_matrix) override {
        // Public interface - uses default base_power of 100 MVA
        Float base_power = 100e6;
        auto mismatches_pu =
            calculate_mismatches_impl(network_data, bus_voltages, admittance_matrix, base_power);

        // Convert from per-unit back to absolute units (Watts) for backward compatibility
        std::vector<Float> mismatches_abs(mismatches_pu.size());
        for (size_t i = 0; i < mismatches_pu.size(); ++i) {
            mismatches_abs[i] = mismatches_pu[i] * base_power;
        }

        return mismatches_abs;
    }

  private:
    std::vector<Float> calculate_mismatches_impl(NetworkData const& network_data,
                                                 ComplexVector const& bus_voltages,
                                                 SparseMatrix const& admittance_matrix,
                                                 Float base_power) {
        LOG_TRACE(logger, "CPUNewtonRaphson: Calculating power mismatches");

        std::vector<Float> mismatches;

        // Calculate power injections S_i = V_i * conj(I_i) = V_i * conj(Y_bus * V)
        // If Y-bus is in per-unit, this gives S in per-unit
        auto calculated_powers = calculate_bus_power_injections(bus_voltages, admittance_matrix);

        // Debug: Log power values for first iteration
        static bool first_call = true;
        if (first_call) {
            first_call = false;
            LOG_INFO(logger, "  === Power injection debug (per-unit) ===");
            for (size_t i = 0; i < network_data.buses.size(); ++i) {
                auto specified = get_specified_power_at_bus(network_data, i, base_power);
                LOG_INFO(logger, "    Bus", (i + 1), "- Calculated:", calculated_powers[i].real(),
                         "pu,", calculated_powers[i].imag(), "pu | Specified:", specified.real(),
                         "pu,", specified.imag(), "pu");
            }
        }

        // Calculate mismatches for each bus equation
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            if (network_data.buses[i].bus_type == BusType::SLACK) {
                // Skip slack bus - no power equations
                continue;
            }

            // Get specified power from appliances connected to this bus (in per-unit)
            auto specified_power = get_specified_power_at_bus(network_data, i, base_power);

            // P mismatch: ΔP_i = P_calculated - P_specified (Newton-Raphson standard convention)
            Float p_mismatch = calculated_powers[i].real() - specified_power.real();
            mismatches.push_back(p_mismatch);

            if (network_data.buses[i].bus_type == BusType::PQ) {
                // Q mismatch: ΔQ_i = Q_calculated - Q_specified (Newton-Raphson standard
                // convention)
                Float q_mismatch = calculated_powers[i].imag() - specified_power.imag();
                mismatches.push_back(q_mismatch);
            }
            // PV buses: only P equation, Q is free variable within limits
        }

        return mismatches;
    }
    /**
     * @brief Calculate power injections at all buses: S = V * conj(Y * V)
     */
    ComplexVector calculate_bus_power_injections(ComplexVector const& voltages,
                                                 SparseMatrix const& Y_bus) const {
        ComplexVector powers(voltages.size(), Complex(0.0, 0.0));

        // Check if matrix is properly initialized
        if (Y_bus.nnz == 0 || Y_bus.row_ptr.empty() || Y_bus.col_idx.empty() ||
            Y_bus.values.empty()) {
            LOG_DEBUG(logger, "Empty admittance matrix, returning zero power injections");
            return powers;
        }

        // For each bus i: S_i = V_i * conj(Σ(Y_ik * V_k))
        for (int i = 0; i < static_cast<int>(voltages.size()); ++i) {
            // Check bounds for row pointer access
            if (i >= static_cast<int>(Y_bus.row_ptr.size() - 1)) {
                continue;
            }

            Complex current_injection(0.0, 0.0);

            // Calculate current injection: I_i = Σ(Y_ik * V_k)
            for (int idx = Y_bus.row_ptr[i]; idx < Y_bus.row_ptr[i + 1]; ++idx) {
                // Check bounds for arrays access
                if (idx >= static_cast<int>(Y_bus.col_idx.size()) ||
                    idx >= static_cast<int>(Y_bus.values.size())) {
                    break;
                }

                int k = Y_bus.col_idx[idx];
                if (k >= static_cast<int>(voltages.size())) {
                    continue;
                }

                current_injection += Y_bus.values[idx] * voltages[k];
            }

            // Power injection: S_i = V_i * conj(I_i)
            powers[i] = voltages[i] * std::conj(current_injection);
        }

        return powers;
    }

    /**
     * @brief Get specified power injection at a bus from connected appliances and bus data
     * @param network_data Network data with bus and appliance information
     * @param bus_idx Bus index
     * @param base_power Base power for per-unit conversion (VA)
     * @return Specified power in per-unit (if base_power provided) or absolute (if 0)
     */
    Complex get_specified_power_at_bus(NetworkData const& network_data, size_t bus_idx,
                                       Float base_power = 0.0) const {
        Complex specified_power(0.0, 0.0);
        int bus_id = network_data.buses[bus_idx].id;

        // First, check if power is specified directly on the bus (e.g., from Python bindings)
        // This takes priority as it's the simplified interface
        if (network_data.buses[bus_idx].active_power != 0.0 ||
            network_data.buses[bus_idx].reactive_power != 0.0) {
            specified_power = Complex(network_data.buses[bus_idx].active_power,
                                      network_data.buses[bus_idx].reactive_power);
        }

        // Also sum power from all appliances connected to this bus
        // (for compatibility with existing test cases that use appliances)
        for (auto const& appliance : network_data.appliances) {
            if (appliance.node == bus_id && appliance.status == 1 &&
                (appliance.type == ApplianceType::SOURCE ||
                 appliance.type == ApplianceType::LOADGEN)) {
                // Sources inject power (positive P), loads consume power (negative P)
                specified_power += Complex(appliance.p_specified, appliance.q_specified);
            }
            // SHUNT appliances are handled in admittance matrix, not as power injections
        }

        // Convert to per-unit if base_power is provided
        if (base_power > 0.0) {
            specified_power /= base_power;
        }

        return specified_power;
    }

    /**
     * @brief Solve Newton's correction equations: J * Δx = -F
     */
    std::vector<Float> solve_newton_correction(NetworkData const& network_data,
                                               std::vector<Float> const& mismatches,
                                               ComplexVector const& voltages,
                                               SparseMatrix const& Y_bus,
                                               ProfilingData& profiling) const {
        LOG_TRACE(logger, "CPUNewtonRaphson: Building Jacobian and solving linear system");

        // Build the Jacobian matrix for power flow equations
        auto t_jac_start = std::chrono::high_resolution_clock::now();
        auto jacobian = build_jacobian_matrix(network_data, voltages, Y_bus);
        auto t_jac_end = std::chrono::high_resolution_clock::now();
        profiling.total_jacobian_time +=
            std::chrono::duration<double, std::milli>(t_jac_end - t_jac_start).count();

        if (!lu_solver_) {
            LOG_ERROR(logger, "LU solver not initialized");
            return std::vector<Float>(mismatches.size(), 0.0);
        }

        // Convert Jacobian to sparse format for LU solver
        SparseMatrix jacobian_sparse = convert_to_sparse_matrix(jacobian);

        // Negate mismatches for Newton's method: J * Δx = -F
        ComplexVector rhs_complex(mismatches.size());
        std::ranges::transform(mismatches, rhs_complex.begin(),
                               [](Float mismatch) { return Complex(-mismatch, 0.0); });

        // Factorize and solve
        auto t_fact_start = std::chrono::high_resolution_clock::now();
        bool factorized = lu_solver_->factorize(jacobian_sparse);
        auto t_fact_end = std::chrono::high_resolution_clock::now();
        profiling.total_factorize_time +=
            std::chrono::duration<double, std::milli>(t_fact_end - t_fact_start).count();

        if (!factorized) {
            LOG_ERROR(logger, "Failed to factorize Jacobian matrix");
            return std::vector<Float>(mismatches.size(), 0.0);
        }

        auto t_solve_start = std::chrono::high_resolution_clock::now();
        ComplexVector solution_complex = lu_solver_->solve(rhs_complex);
        auto t_solve_end = std::chrono::high_resolution_clock::now();
        profiling.total_solve_time +=
            std::chrono::duration<double, std::milli>(t_solve_end - t_solve_start).count();

        // Convert back to real solution
        std::vector<Float> solution(solution_complex.size());
        std::ranges::transform(solution_complex, solution.begin(),
                               [](Complex const& c) { return c.real(); });

        return solution;
    }

    /**
     * @brief Update voltage estimates with Newton corrections
     */
    void update_voltage_estimates(NetworkData const& network_data, ComplexVector& voltages,
                                  std::vector<Float> const& corrections) const {
        if (corrections.empty()) {
            return;
        }

        int n_buses = static_cast<int>(network_data.buses.size());
        int slack_idx = -1;

        // Find slack bus
        auto slack_iter = std::ranges::find_if(
            network_data.buses, [](auto const& bus) { return bus.bus_type == BusType::SLACK; });
        if (slack_iter != network_data.buses.end()) {
            slack_idx = static_cast<int>(std::distance(network_data.buses.begin(), slack_iter));
        }

        // Map corrections back to voltage updates
        int corr_idx = 0;

        // Use adaptive damping factor to prevent overshooting
        Float const damping_factor = 0.9;  // Less conservative for faster convergence

        // Update voltage angles (excluding slack bus)
        for (int i = 0; i < n_buses; ++i) {
            if (i != slack_idx && corr_idx < static_cast<int>(corrections.size())) {
                Float V_mag = std::abs(voltages[i]);
                Float V_angle =
                    std::arg(voltages[i]) + damping_factor * corrections[corr_idx];  // Damped Δθ

                // Limit angle changes to prevent instability
                Float angle_correction = damping_factor * corrections[corr_idx];
                if (std::abs(angle_correction) > 0.15) {  // Limit to ~8.5 degrees
                    angle_correction = 0.15 * (angle_correction > 0 ? 1.0 : -1.0);
                }
                V_angle = std::arg(voltages[i]) + angle_correction;

                voltages[i] = Complex(V_mag * cos(V_angle), V_mag * sin(V_angle));
                corr_idx++;
            }
        }

        // Update voltage magnitudes (only PQ buses)
        for (int i = 0; i < n_buses; ++i) {
            if (network_data.buses[i].bus_type == BusType::PQ &&
                corr_idx < static_cast<int>(corrections.size())) {
                Float V_mag_old = std::abs(voltages[i]);
                Float V_mag_correction = damping_factor * corrections[corr_idx];  // Damped ΔV

                // Limit magnitude changes to prevent instability
                if (std::abs(V_mag_correction) > 0.08) {  // Limit to 8% change
                    V_mag_correction = 0.08 * (V_mag_correction > 0 ? 1.0 : -1.0);
                }

                Float V_mag = V_mag_old + V_mag_correction;
                Float V_angle = std::arg(voltages[i]);

                // Keep voltage magnitude in reasonable bounds (but not too restrictive)
                V_mag = std::max(
                    V_mag,
                    0.5);  // Minimum voltage magnitude (0.5 p.u.) - very permissive for convergence
                V_mag = std::min(V_mag, 1.5);  // Maximum voltage magnitude (1.5 p.u.)

                voltages[i] = Complex(V_mag * cos(V_angle), V_mag * sin(V_angle));
                corr_idx++;
            }
        }
    }

    /**
     * @brief Build the Jacobian matrix for Newton-Raphson power flow
     */
    std::vector<std::vector<Float>> build_jacobian_matrix(NetworkData const& network_data,
                                                          ComplexVector const& voltages,
                                                          SparseMatrix const& Y_bus) const {
        // Count number of variables and equations
        int n_vars = count_newton_variables(network_data);

        std::vector<std::vector<Float>> jacobian(n_vars, std::vector<Float>(n_vars, 0.0));

        if (n_vars == 0) {
            return jacobian;  // No variables to solve for
        }

        // Calculate Jacobian elements using power flow derivatives
        calculate_jacobian_elements(jacobian, network_data, voltages, Y_bus);

        return jacobian;
    }

    /**
     * @brief Count the number of Newton-Raphson variables
     */
    int count_newton_variables(NetworkData const& network_data) const {
        int n_buses = static_cast<int>(network_data.buses.size());

        int slack_buses = std::ranges::count_if(
            network_data.buses, [](auto const& bus) { return bus.bus_type == BusType::SLACK; });

        int n_pq = std::ranges::count_if(
            network_data.buses, [](auto const& bus) { return bus.bus_type == BusType::PQ; });

        // Variables: (n-1) angles + n_pq magnitudes
        return (n_buses - slack_buses) + n_pq;
    }

    /**
     * @brief Calculate Jacobian matrix elements using analytical power flow derivatives
     *
     * Power flow equations: P_i = V_i * Σ_k V_k * (G_ik cos(θ_i - θ_k) + B_ik sin(θ_i - θ_k))
     *                       Q_i = V_i * Σ_k V_k * (G_ik sin(θ_i - θ_k) - B_ik cos(θ_i - θ_k))
     *
     * Jacobian elements:
     *   ∂P_i/∂θ_i = -Q_i - B_ii * V_i²
     *   ∂P_i/∂θ_j = V_i * V_j * (G_ij sin(θ_i - θ_j) - B_ij cos(θ_i - θ_j))   for i ≠ j
     *   ∂P_i/∂V_i = (P_i/V_i) + G_ii * V_i
     *   ∂P_i/∂V_j = V_i * (G_ij cos(θ_i - θ_j) + B_ij sin(θ_i - θ_j))         for i ≠ j
     *   ∂Q_i/∂θ_i = P_i - G_ii * V_i²
     *   ∂Q_i/∂θ_j = -V_i * V_j * (G_ij cos(θ_i - θ_j) + B_ij sin(θ_i - θ_j))  for i ≠ j
     *   ∂Q_i/∂V_i = (Q_i/V_i) - B_ii * V_i
     *   ∂Q_i/∂V_j = V_i * (G_ij sin(θ_i - θ_j) - B_ij cos(θ_i - θ_j))         for i ≠ j
     */
    void calculate_jacobian_elements(std::vector<std::vector<Float>>& jacobian,
                                     NetworkData const& network_data, ComplexVector const& voltages,
                                     SparseMatrix const& Y_bus) const {
        if (Y_bus.nnz == 0 || jacobian.empty()) {
            // For empty matrix case, create a well-conditioned identity-based matrix
            for (size_t i = 0; i < jacobian.size() && i < jacobian[0].size(); ++i) {
                jacobian[i][i] = 1.0;
            }
            return;
        }

        int n_buses = static_cast<int>(network_data.buses.size());

        // Extract voltage magnitudes and angles
        std::vector<Float> V_mag(n_buses);
        std::vector<Float> V_angle(n_buses);
        for (int i = 0; i < n_buses; ++i) {
            V_mag[i] = std::abs(voltages[i]);
            V_angle[i] = std::arg(voltages[i]);
            if (V_mag[i] < 1e-6) V_mag[i] = 1.0;  // Avoid division by zero
        }

        // Build G and B matrices (conductance and susceptance) from Y_bus
        std::vector<std::vector<Float>> G(n_buses, std::vector<Float>(n_buses, 0.0));
        std::vector<std::vector<Float>> B(n_buses, std::vector<Float>(n_buses, 0.0));

        for (int i = 0; i < n_buses; ++i) {
            if (i >= static_cast<int>(Y_bus.row_ptr.size() - 1)) continue;

            for (int idx = Y_bus.row_ptr[i]; idx < Y_bus.row_ptr[i + 1]; ++idx) {
                if (idx >= static_cast<int>(Y_bus.col_idx.size())) continue;

                int j = Y_bus.col_idx[idx];
                if (j >= n_buses) continue;

                G[i][j] = Y_bus.values[idx].real();
                B[i][j] = Y_bus.values[idx].imag();
            }
        }

        // Calculate current power injections for diagonal terms
        auto calculated_powers = calculate_bus_power_injections(voltages, Y_bus);

        // Build variable mapping
        std::vector<int> angle_var_idx(n_buses, -1);
        std::vector<int> mag_var_idx(n_buses, -1);
        int slack_idx = -1;

        // Find slack bus
        for (int i = 0; i < n_buses; ++i) {
            if (network_data.buses[i].bus_type == BusType::SLACK) {
                slack_idx = i;
                break;
            }
        }

        int var_idx = 0;
        // Angle variables (excluding slack)
        for (int i = 0; i < n_buses; ++i) {
            if (i != slack_idx) {
                angle_var_idx[i] = var_idx++;
            }
        }
        // Magnitude variables (PQ buses only)
        for (int i = 0; i < n_buses; ++i) {
            if (network_data.buses[i].bus_type == BusType::PQ) {
                mag_var_idx[i] = var_idx++;
            }
        }

        // Fill Jacobian with analytical power flow derivatives
        int eq_idx = 0;

        for (int i = 0; i < n_buses; ++i) {
            if (network_data.buses[i].bus_type == BusType::SLACK) continue;

            Float Vi = V_mag[i];
            Float theta_i = V_angle[i];
            Float Pi = calculated_powers[i].real();
            Float Qi = calculated_powers[i].imag();

            // P equation row
            if (eq_idx < static_cast<int>(jacobian.size())) {
                // ∂P_i/∂θ_j terms
                for (int j = 0; j < n_buses; ++j) {
                    if (angle_var_idx[j] >= 0 &&
                        angle_var_idx[j] < static_cast<int>(jacobian[eq_idx].size())) {
                        if (i == j) {
                            // Diagonal: ∂P_i/∂θ_i = -Q_i - B_ii * V_i²
                            jacobian[eq_idx][angle_var_idx[j]] = -Qi - B[i][i] * Vi * Vi;
                        } else {
                            // Off-diagonal: ∂P_i/∂θ_j = V_i * V_j * (G_ij sin(θ_i - θ_j) - B_ij
                            // cos(θ_i - θ_j))
                            Float Vj = V_mag[j];
                            Float theta_j = V_angle[j];
                            Float theta_ij = theta_i - theta_j;
                            jacobian[eq_idx][angle_var_idx[j]] =
                                Vi * Vj *
                                (G[i][j] * std::sin(theta_ij) - B[i][j] * std::cos(theta_ij));
                        }
                    }
                }

                // ∂P_i/∂V_j terms
                for (int j = 0; j < n_buses; ++j) {
                    if (mag_var_idx[j] >= 0 &&
                        mag_var_idx[j] < static_cast<int>(jacobian[eq_idx].size())) {
                        if (i == j) {
                            // Diagonal: ∂P_i/∂V_i = (P_i/V_i) + G_ii * V_i
                            jacobian[eq_idx][mag_var_idx[j]] = (Pi / Vi) + G[i][i] * Vi;
                        } else {
                            // Off-diagonal: ∂P_i/∂V_j = V_i * (G_ij cos(θ_i - θ_j) + B_ij sin(θ_i -
                            // θ_j))
                            Float theta_j = V_angle[j];
                            Float theta_ij = theta_i - theta_j;
                            jacobian[eq_idx][mag_var_idx[j]] =
                                Vi * (G[i][j] * std::cos(theta_ij) + B[i][j] * std::sin(theta_ij));
                        }
                    }
                }

                eq_idx++;
            }

            // Q equation row (only for PQ buses)
            if (network_data.buses[i].bus_type == BusType::PQ &&
                eq_idx < static_cast<int>(jacobian.size())) {
                // ∂Q_i/∂θ_j terms
                for (int j = 0; j < n_buses; ++j) {
                    if (angle_var_idx[j] >= 0 &&
                        angle_var_idx[j] < static_cast<int>(jacobian[eq_idx].size())) {
                        if (i == j) {
                            // Diagonal: ∂Q_i/∂θ_i = P_i - G_ii * V_i²
                            jacobian[eq_idx][angle_var_idx[j]] = Pi - G[i][i] * Vi * Vi;
                        } else {
                            // Off-diagonal: ∂Q_i/∂θ_j = -V_i * V_j * (G_ij cos(θ_i - θ_j) + B_ij
                            // sin(θ_i - θ_j))
                            Float Vj = V_mag[j];
                            Float theta_j = V_angle[j];
                            Float theta_ij = theta_i - theta_j;
                            jacobian[eq_idx][angle_var_idx[j]] =
                                -Vi * Vj *
                                (G[i][j] * std::cos(theta_ij) + B[i][j] * std::sin(theta_ij));
                        }
                    }
                }

                // ∂Q_i/∂V_j terms
                for (int j = 0; j < n_buses; ++j) {
                    if (mag_var_idx[j] >= 0 &&
                        mag_var_idx[j] < static_cast<int>(jacobian[eq_idx].size())) {
                        if (i == j) {
                            // Diagonal: ∂Q_i/∂V_i = (Q_i/V_i) - B_ii * V_i
                            jacobian[eq_idx][mag_var_idx[j]] = (Qi / Vi) - B[i][i] * Vi;
                        } else {
                            // Off-diagonal: ∂Q_i/∂V_j = V_i * (G_ij sin(θ_i - θ_j) - B_ij cos(θ_i -
                            // θ_j))
                            Float theta_j = V_angle[j];
                            Float theta_ij = theta_i - theta_j;
                            jacobian[eq_idx][mag_var_idx[j]] =
                                Vi * (G[i][j] * std::sin(theta_ij) - B[i][j] * std::cos(theta_ij));
                        }
                    }
                }

                eq_idx++;
            }
        }
    }

    /**
     * @brief Convert dense matrix to sparse CSR format
     */
    SparseMatrix convert_to_sparse_matrix(
        std::vector<std::vector<Float>> const& dense_matrix) const {
        SparseMatrix sparse;

        if (dense_matrix.empty()) {
            sparse.num_rows = sparse.num_cols = sparse.nnz = 0;
            return sparse;
        }

        int rows = static_cast<int>(dense_matrix.size());
        int cols = static_cast<int>(dense_matrix[0].size());

        sparse.num_rows = rows;
        sparse.num_cols = cols;
        sparse.row_ptr.resize(rows + 1);

        // Count non-zeros and build CSR structure
        int nnz = 0;
        sparse.row_ptr[0] = 0;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (std::abs(dense_matrix[i][j]) > 1e-12) {  // Tolerance for zero
                    sparse.col_idx.push_back(j);
                    sparse.values.push_back(Complex(dense_matrix[i][j], 0.0));
                    nnz++;
                }
            }
            sparse.row_ptr[i + 1] = nnz;
        }

        sparse.nnz = nnz;
        return sparse;
    }

    BackendType get_backend_type() const noexcept override { return BackendType::CPU; }

    // State capture interface
    void enable_state_capture(bool enable) override { capture_states_ = enable; }

    const std::vector<IterationState>& get_iteration_states() const override {
        return iteration_states_;
    }

    void clear_iteration_states() override { iteration_states_.clear(); }
};

}  // namespace gap::solver
