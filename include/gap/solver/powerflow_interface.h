#pragma once

#include "gap/core/types.h"
#include "gap/solver/lu_solver_interface.h"
#include <memory>

namespace gap::solver {

/**
 * @brief Power flow solver configuration
 */
struct PowerFlowConfig {
    double tolerance = 1e-6;           // Convergence tolerance
    int max_iterations = 50;           // Maximum number of iterations
    bool use_flat_start = true;        // Use flat start for voltages
    double acceleration_factor = 1.4;  // Acceleration factor
    bool verbose = false;              // Enable verbose output
};

/**
 * @brief Power flow solution result
 */
struct PowerFlowResult {
    ComplexVector bus_voltages;        // Bus voltage phasors
    bool converged = false;            // Convergence status
    int iterations = 0;                // Number of iterations
    double final_mismatch = 0.0;       // Final mismatch norm
    std::vector<Complex> bus_injections; // Calculated bus injections
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
    virtual PowerFlowResult solve_power_flow(
        const NetworkData& network_data,
        const SparseMatrix& admittance_matrix,
        const PowerFlowConfig& config = PowerFlowConfig{}
    ) = 0;
    
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
    virtual std::vector<double> calculate_mismatches(
        const NetworkData& network_data,
        const ComplexVector& bus_voltages,
        const SparseMatrix& admittance_matrix
    ) = 0;
    
    /**
     * @brief Get backend type
     * @return Backend execution type
     */
    virtual BackendType get_backend_type() const = 0;
};

} // namespace gap::solver