#include <algorithm>
#include <cmath>
#include <iostream>

#include "gap/logging/logger.h"
#include "gap/solver/powerflow_interface.h"

namespace gap::solver {

class CPUNewtonRaphson : public IPowerFlowSolver {
  private:
    std::shared_ptr<ILUSolver> lu_solver_;
    gap::logging::Logger& logger = gap::logging::global_logger;

  public:
    PowerFlowResult solve_power_flow(NetworkData const& network_data,
                                     SparseMatrix const& admittance_matrix,
                                     PowerFlowConfig const& config) override {
        // TODO: Implement CPU-based Newton-Raphson power flow solver
        logger.setComponent("CPUNewtonRaphson");
        LOG_INFO(logger, "Starting power flow solution");
        LOG_DEBUG(logger, "  Number of buses:", network_data.num_buses);
        LOG_DEBUG(logger, "  Tolerance:", config.tolerance);
        LOG_DEBUG(logger, "  Max iterations:", config.max_iterations);

        PowerFlowResult result;
        result.bus_voltages.resize(network_data.num_buses);

        // Initialize voltages (flat start or previous solution)
        if (config.use_flat_start) {
            LOG_DEBUG(logger, "  Using flat start initialization");
            for (size_t i = 0; i < network_data.buses.size(); ++i) {
                if (network_data.buses[i].bus_type == BusType::SLACK) {  // Slack bus
                    result.bus_voltages[i] = Complex(network_data.buses[i].u_pu, 0.0);
                } else {
                    result.bus_voltages[i] = Complex(1.0, 0.0);  // Flat start
                }
            }
        }

        // Newton-Raphson iterations
        for (int iter = 0; iter < config.max_iterations; ++iter) {
            if (config.verbose) {
                LOG_DEBUG(logger, "  Iteration", (iter + 1));
            }

            // Calculate mismatches
            auto mismatches =
                calculate_mismatches(network_data, result.bus_voltages, admittance_matrix);

            // Check convergence
            double max_mismatch = 0.0;
            for (double mismatch : mismatches) {
                max_mismatch = std::max(max_mismatch, std::abs(mismatch));
            }

            result.final_mismatch = max_mismatch;

            if (max_mismatch < config.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                LOG_INFO(logger, "  Converged in", (iter + 1), "iterations");
                break;
            }

            // TODO: Build Jacobian matrix and solve correction equations
            // In real implementation:
            // 1. Calculate Jacobian matrix (partial derivatives)
            // 2. Solve J * Δx = -F for corrections Δx
            // 3. Update voltage estimates: x = x + α * Δx (with acceleration factor)

            // Placeholder: simple update (not physically correct)
            for (auto& voltage : result.bus_voltages) {
                voltage += Complex(0.001, 0.0);  // Dummy update
            }
        }

        if (!result.converged) {
            LOG_WARN(logger, "  Failed to converge after", config.max_iterations, "iterations");
            result.iterations = config.max_iterations;
        }

        return result;
    }

    void set_lu_solver(std::shared_ptr<ILUSolver> lu_solver) override {
        lu_solver_ = lu_solver;
        LOG_DEBUG(logger, "CPUNewtonRaphson: LU solver backend set");
    }

    std::vector<double> calculate_mismatches(NetworkData const& network_data,
                                             ComplexVector const& bus_voltages,
                                             SparseMatrix const& admittance_matrix) override {
        LOG_TRACE(logger, "CPUNewtonRaphson: Calculating power mismatches");

        std::vector<double> mismatches;

        // Calculate power injections S_i = V_i * conj(I_i) = V_i * conj(Y_bus * V)
        auto calculated_powers = calculate_bus_power_injections(bus_voltages, admittance_matrix);

        // Calculate mismatches for each bus equation
        for (size_t i = 0; i < network_data.buses.size(); ++i) {
            if (network_data.buses[i].bus_type == BusType::SLACK) {
                // Skip slack bus - no power equations
                continue;
            }

            // Get specified power from appliances connected to this bus
            auto specified_power = get_specified_power_at_bus(network_data, i);

            // P mismatch: ΔP_i = P_specified - P_calculated
            double p_mismatch = specified_power.real() - calculated_powers[i].real();
            mismatches.push_back(p_mismatch);

            if (network_data.buses[i].bus_type == BusType::PQ) {
                // Q mismatch: ΔQ_i = Q_specified - Q_calculated
                double q_mismatch = specified_power.imag() - calculated_powers[i].imag();
                mismatches.push_back(q_mismatch);
            }
            // PV buses: only P equation, Q is free variable within limits
        }

        return mismatches;
    }

  private:
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
     * @brief Get specified power injection at a bus from connected appliances
     */
    Complex get_specified_power_at_bus(NetworkData const& network_data, size_t bus_idx) const {
        Complex specified_power(0.0, 0.0);
        int bus_id = network_data.buses[bus_idx].id;

        // Sum power from all appliances connected to this bus
        for (auto const& appliance : network_data.appliances) {
            if (appliance.node == bus_id && appliance.status == 1) {
                if (appliance.type == ApplianceType::SOURCE) {
                    // Sources inject power (positive P means generation)
                    specified_power += Complex(appliance.p_specified, appliance.q_specified);
                } else if (appliance.type == ApplianceType::LOADGEN) {
                    // Loads consume power (negative P means consumption)
                    specified_power += Complex(appliance.p_specified, appliance.q_specified);
                }
                // SHUNT appliances are handled in admittance matrix, not as power injections
            }
        }

        return specified_power;
    }

    BackendType get_backend_type() const noexcept override { return BackendType::CPU; }
};

}  // namespace gap::solver
