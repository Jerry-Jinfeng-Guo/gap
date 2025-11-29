#pragma once

#include <memory>
#include <string>

#include "gap/admittance/admittance_interface.h"
#include "gap/io/io_interface.h"
#include "gap/solver/lu_solver_interface.h"
#include "gap/solver/powerflow_interface.h"

namespace gap::core {

/**
 * @brief Factory class for creating backend implementations
 */
class BackendFactory {
  public:
    /**
     * @brief Create IO module instance
     * @return Unique pointer to IO module implementation
     */
    static std::unique_ptr<io::IIOModule> create_io_module();

    /**
     * @brief Create admittance matrix backend
     * @param backend_type Type of backend (CPU or GPU)
     * @return Unique pointer to admittance matrix implementation
     */
    static std::unique_ptr<admittance::IAdmittanceMatrix> create_admittance_backend(
        BackendType backend_type);

    /**
     * @brief Create LU solver backend
     * @param backend_type Type of backend (CPU or GPU)
     * @return Unique pointer to LU solver implementation
     */
    static std::unique_ptr<solver::ILUSolver> create_lu_solver(BackendType backend_type);

    /**
     * @brief Create power flow solver backend
     * @param backend_type Type of backend (CPU or GPU)
     * @param method Power flow solution method (default: Newton-Raphson)
     * @return Unique pointer to power flow solver implementation
     */
    static std::unique_ptr<solver::IPowerFlowSolver> create_powerflow_solver(
        BackendType backend_type, PowerFlowMethod method = PowerFlowMethod::NEWTON_RAPHSON);

    /**
     * @brief Check if backend type is available
     * @param backend_type Type of backend to check
     * @return true if backend is available
     */
    static bool is_backend_available(BackendType backend_type);

    /**
     * @brief Get available backend types
     * @return Vector of available backend types
     */
    static std::vector<BackendType> get_available_backends();

  private:
    /**
     * @brief Load shared library backend
     * @param library_name Name of the shared library
     * @return Handle to the loaded library
     */
    static void* load_backend_library(std::string const& library_name);

    /**
     * @brief Check CUDA availability
     * @return true if CUDA is available
     */
    static bool check_cuda_availability();
};

}  // namespace gap::core