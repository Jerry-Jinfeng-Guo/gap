#pragma once

#include <memory>

#include "gap/core/types.h"

namespace gap::admittance {

/**
 * @brief Abstract interface for admittance matrix calculation
 */
class IAdmittanceMatrix {
  public:
    virtual ~IAdmittanceMatrix() = default;

    /**
     * @brief Build the network admittance matrix
     * @param network_data Power system network data
     * @return Sparse admittance matrix in CSR format
     */
    virtual std::unique_ptr<SparseMatrix> build_admittance_matrix(
        NetworkData const& network_data) = 0;

    /**
     * @brief Update admittance matrix for topology changes
     * @param matrix Existing admittance matrix
     * @param branch_changes Vector of branch status changes
     * @return Updated sparse admittance matrix
     */
    virtual std::unique_ptr<SparseMatrix> update_admittance_matrix(
        SparseMatrix const& matrix, std::vector<BranchData> const& branch_changes) = 0;

    /**
     * @brief Get backend type
     * @return Backend execution type
     */
    virtual BackendType get_backend_type() const noexcept = 0;
};

}  // namespace gap::admittance
