#pragma once

#include <chrono>
#include <memory>

#include "gap/core/types.h"
#include "gap/solver/lu_solver_interface.h"

namespace gap::solver {

/**
 * @brief Power flow solver configuration
 */
struct PowerFlowConfig {
    Float tolerance = 1e-6;           // Convergence tolerance
    int max_iterations = 50;          // Maximum number of iterations
    bool use_flat_start = true;       // Use flat start for voltages
    Float acceleration_factor = 1.0;  // Acceleration factor (reduced for GPU stability)
    bool verbose = false;             // Enable verbose output
    Float base_power = 100e6;         // Base power for per-unit system (VA), default 100 MVA
};

/**
 * @brief Iteration state for debugging
 */
struct IterationState {
    int iteration;
    std::vector<Complex> voltages;
    std::vector<Complex> currents;
    std::vector<Complex> power_injections;
    std::vector<double> mismatches;
    double max_mismatch;
};

/**
 * @brief Power flow solution result
 */
struct PowerFlowResult {
    bool converged = false;        // Convergence status
    int iterations = 0;            // Number of iterations
    Float final_mismatch = 0.0;    // Final mismatch norm
    ComplexVector bus_voltages;    // Bus voltage phasors
    ComplexVector bus_injections;  // Calculated bus injections
};

/**
 * @brief Batch power flow configuration
 */
struct BatchPowerFlowConfig {
    PowerFlowConfig base_config;            // Base solver configuration
    bool reuse_y_bus_factorization = true;  // Cache Y-bus factorization across batch
    bool warm_start = false;                // Use previous solution as starting point
    bool verbose_summary = false;           // Print batch summary statistics
};

/**
 * @brief Batch power flow results
 */
struct BatchPowerFlowResult {
    std::vector<PowerFlowResult> results;  // Individual solve results
    int total_iterations = 0;              // Sum of all iterations
    double total_solve_time_ms = 0.0;      // Total time in milliseconds
    double avg_solve_time_ms = 0.0;        // Average time per scenario
    int converged_count = 0;               // Number of converged scenarios
    int failed_count = 0;                  // Number of failed scenarios
};

/**
 * @brief Abstract interface for Newton-Raphson power flow solver
 */
class IPowerFlowSolver {
  public:
    virtual ~IPowerFlowSolver() = default;

    /**
     * @brief Solve power flow using Newton-Raphson method
     * @param network_data Power system network data
     * @param admittance_matrix System admittance matrix
     * @param config Solver configuration
     * @return Power flow solution result
     */
    virtual PowerFlowResult solve_power_flow(NetworkData const& network_data,
                                             SparseMatrix const& admittance_matrix,
                                             PowerFlowConfig const& config = PowerFlowConfig{}) = 0;

    /**
     * @brief Set LU solver backend
     * @param lu_solver Pointer to LU solver implementation
     */
    virtual void set_lu_solver(std::shared_ptr<ILUSolver> lu_solver) = 0;

    /**
     * @brief Calculate power mismatches
     * @param network_data Power system network data
     * @param bus_voltages Current bus voltage estimates
     * @param admittance_matrix System admittance matrix
     * @return Vector of power mismatches
     */
    virtual std::vector<Float> calculate_mismatches(NetworkData const& network_data,
                                                    ComplexVector const& bus_voltages,
                                                    SparseMatrix const& admittance_matrix) = 0;

    /**
     * @brief Get backend type
     * @return Backend execution type
     */
    virtual BackendType get_backend_type() const noexcept = 0;

    /**
     * @brief Enable or disable iteration state capture (for debugging)
     * @param enable True to capture states, false to disable
     * @note Default implementation does nothing
     */
    virtual void enable_state_capture(bool enable) { (void)enable; }

    /**
     * @brief Get captured iteration states (for debugging)
     * @return Reference to vector of captured states
     * @note Default implementation returns empty vector
     */
    virtual std::vector<IterationState> const& get_iteration_states() const {
        static std::vector<IterationState> empty;
        return empty;
    }

    /**
     * @brief Clear captured iteration states
     * @note Default implementation does nothing
     */
    virtual void clear_iteration_states() {}

    /**
     * @brief Solve multiple power flow scenarios with shared resources
     * @param network_scenarios Vector of network data for each scenario (must have same topology)
     * @param admittance_matrix System admittance matrix (shared across all scenarios)
     * @param config Batch solver configuration
     * @return Batch results containing all individual solve results
     * @note Default implementation falls back to repeated single solves without caching
     */
    virtual BatchPowerFlowResult solve_power_flow_batch(
        std::vector<NetworkData> const& network_scenarios, SparseMatrix const& admittance_matrix,
        BatchPowerFlowConfig const& config = BatchPowerFlowConfig{}) {
        BatchPowerFlowResult batch_result;
        batch_result.results.reserve(network_scenarios.size());

        auto start = std::chrono::high_resolution_clock::now();

        for (auto const& network : network_scenarios) {
            auto result = solve_power_flow(network, admittance_matrix, config.base_config);
            batch_result.results.push_back(result);
            batch_result.total_iterations += result.iterations;
            if (result.converged) {
                batch_result.converged_count++;
            } else {
                batch_result.failed_count++;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        batch_result.total_solve_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        batch_result.avg_solve_time_ms =
            batch_result.total_solve_time_ms / network_scenarios.size();

        return batch_result;
    }
};

}  // namespace gap::solver
