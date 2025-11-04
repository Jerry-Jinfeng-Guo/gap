#pragma once

#include <memory>

#include "gap/core/types.h"

namespace gap::solver {

/**
 * @brief Abstract interface for LU factorization and solve operations
 */
class ILUSolver {
  public:
    virtual ~ILUSolver() = default;

    /**
     * @brief Perform LU factorization of the admittance matrix
     * @param matrix Sparse admittance matrix
     * @return true if factorization was successful
     */
    virtual bool factorize(SparseMatrix const& matrix) = 0;

    /**
     * @brief Solve linear system Ax = b using precomputed LU factors
     * @param rhs Right-hand side vector
     * @return Solution vector x
     */
    virtual ComplexVector solve(ComplexVector const& rhs) = 0;

    /**
     * @brief Update factorization for modified matrix
     * @param matrix Modified admittance matrix
     * @return true if update was successful
     */
    virtual bool update_factorization(SparseMatrix const& matrix) = 0;

    /**
     * @brief Get backend type
     * @return Backend execution type
     */
    virtual BackendType get_backend_type() const noexcept = 0;

    /**
     * @brief Check if factorization is valid
     * @return true if factorization is available and valid
     */
    virtual bool is_factorized() const noexcept = 0;
};

}  // namespace gap::solver
