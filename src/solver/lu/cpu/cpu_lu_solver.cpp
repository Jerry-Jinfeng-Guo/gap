#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

#include "gap/logging/logger.h"
#include "gap/solver/lu_solver_interface.h"

namespace gap::solver {

/**
 * @brief Custom sparse LU solver implementation for power system matrices
 *
 * This implementation follows a three-phase approach:
 * 1. Symbolic Analysis: Analyze sparsity pattern and predict fill-in
 * 2. Numerical Factorization: Gaussian elimination with partial pivoting
 * 3. Solve: Forward and backward substitution
 */
class CPULUSolver : public ILUSolver {
  private:
    // Factorization state
    bool factorized_ = false;
    int matrix_size_ = 0;

    // Symbolic analysis results
    struct SymbolicFactors {
        std::vector<int> l_row_ptr;  // L matrix row pointers
        std::vector<int> l_col_idx;  // L matrix column indices
        std::vector<int> u_row_ptr;  // U matrix row pointers (stored by rows)
        std::vector<int> u_col_idx;  // U matrix column indices
        int l_nnz = 0;               // Non-zeros in L
        int u_nnz = 0;               // Non-zeros in U
        bool valid = false;          // Symbolic analysis completed
    } symbolic_;

    // Numerical factorization results
    struct NumericalFactors {
        std::vector<Complex> l_values;  // L matrix values (lower triangular)
        std::vector<Complex> u_values;  // U matrix values (upper triangular)
        std::vector<int> pivot_row;     // Row permutation P: PAQ = LU
        std::vector<int> pivot_col;     // Column permutation Q (if needed)
        bool valid = false;             // Numerical factorization completed
    } numerical_;

    // Working arrays for factorization
    std::vector<Complex> work_row_;  // Working row for elimination
    std::vector<int> work_pattern_;  // Working pattern for current row
    std::vector<bool> work_marker_;  // Marker array for sparsity detection

    // Tolerance for numerical stability
    static constexpr double pivot_tolerance_ = 1e-14;
    static constexpr double drop_tolerance_ = 1e-16;
    static constexpr double growth_factor_limit_ = 1e12;

  public:
    /**
     * @brief Perform complete LU factorization in three phases
     */
    bool factorize(SparseMatrix const& matrix) override {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPULUSolver");

        logger.logInfo("Starting three-phase factorization");
        LOG_DEBUG(logger, "Matrix size:", matrix.num_rows, "x", matrix.num_cols);
        LOG_DEBUG(logger, "Non-zeros:", matrix.nnz);

        // Validate input matrix
        if (matrix.num_rows != matrix.num_cols) {
            logger.logError("Matrix must be square for LU factorization");
            return false;
        }

        if (matrix.nnz <= 0 || matrix.values.empty()) {
            logger.logError("Matrix has no non-zero elements");
            return false;
        }

        matrix_size_ = matrix.num_rows;

        // Phase 1: Symbolic Analysis
        if (!perform_symbolic_analysis(matrix)) {
            logger.logError("Symbolic analysis failed");
            return false;
        }

        // Phase 2: Numerical Factorization
        if (!perform_numerical_factorization(matrix)) {
            logger.logError("Numerical factorization failed");
            return false;
        }

        factorized_ = true;
        LOG_INFO(logger, "  Three-phase factorization completed successfully");
        LOG_INFO(logger, "  L nnz:", symbolic_.l_nnz, ", U nnz:", symbolic_.u_nnz);

        return true;
    }

    /**
     * @brief Solve linear system using precomputed LU factors
     */
    ComplexVector solve(ComplexVector const& rhs) override {
        if (!factorized_) {
            throw std::runtime_error("Matrix not factorized. Call factorize() first.");
        }

        if (static_cast<int>(rhs.size()) != matrix_size_) {
            throw std::runtime_error("RHS size does not match matrix dimension");
        }

        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPULUSolver");
        LOG_INFO(logger, "Solving with forward/backward substitution");
        LOG_DEBUG(logger, "  RHS size:", rhs.size());

        // Phase 3: Solve system PAQ x = b
        // Step 1: Apply row permutation -> b' = P * b
        ComplexVector permuted_rhs = apply_row_permutation(rhs);

        // Step 2: Forward substitution -> solve L * y = b' for y
        ComplexVector intermediate = forward_substitution(permuted_rhs);

        // Step 3: Backward substitution -> solve U * z = y for z
        ComplexVector permuted_solution = backward_substitution(intermediate);

        // Step 4: Apply column permutation -> x = Q^T * z
        ComplexVector solution = apply_column_permutation_inverse(permuted_solution);

        return solution;
    }

    /**
     * @brief Update existing factorization (for now, performs full refactorization)
     */
    bool update_factorization(SparseMatrix const& matrix) override {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPULUSolver");
        LOG_INFO(logger, "Updating factorization");

        // For now, perform full refactorization
        // Future optimization: implement efficient update for modified matrices
        return factorize(matrix);
    }

    BackendType get_backend_type() const noexcept override { return BackendType::CPU; }

    bool is_factorized() const noexcept override { return factorized_; }

  private:
    /**
     * @brief Phase 1: Symbolic Analysis
     * Analyze sparsity pattern and predict fill-in locations
     */
    bool perform_symbolic_analysis(const SparseMatrix& matrix) {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPULUSolver");
        LOG_INFO(logger, "  Phase 1: Symbolic analysis...");

        // Validate CSR format
        if (matrix.row_ptr.size() != static_cast<size_t>(matrix_size_ + 1)) {
            logger.logError("Invalid row_ptr size");
            return false;
        }

        if (matrix.col_idx.size() != static_cast<size_t>(matrix.nnz) ||
            matrix.values.size() != static_cast<size_t>(matrix.nnz)) {
            logger.logError("Inconsistent matrix data sizes");
            return false;
        }

        // Initialize symbolic structures
        symbolic_.l_row_ptr.assign(matrix_size_ + 1, 0);
        symbolic_.u_row_ptr.assign(matrix_size_ + 1, 0);
        symbolic_.l_col_idx.clear();
        symbolic_.u_col_idx.clear();

        // Use elimination tree approach for fill-in prediction
        // For simplicity, we'll use a conservative estimate based on matrix structure

        std::vector<std::set<int>> l_pattern(matrix_size_);
        std::vector<std::set<int>> u_pattern(matrix_size_);

        // Initialize patterns with original matrix structure
        for (int row = 0; row < matrix_size_; ++row) {
            // L matrix gets lower triangular part (including diagonal)
            // U matrix gets upper triangular part (including diagonal)

            for (int idx = matrix.row_ptr[row]; idx < matrix.row_ptr[row + 1]; ++idx) {
                int col = matrix.col_idx[idx];

                if (col <= row) {
                    // Lower triangular part goes to L
                    l_pattern[row].insert(col);
                }
                if (col >= row) {
                    // Upper triangular part goes to U
                    u_pattern[row].insert(col);
                }
            }

            // Ensure diagonal elements are present in both L and U
            l_pattern[row].insert(row);
            u_pattern[row].insert(row);
        }

        // More accurate fill-in prediction using elimination tree concept
        // Only predict fill-in where it's likely to actually occur
        for (int k = 0; k < matrix_size_ - 1; ++k) {
            // Collect non-zero columns in row k (pivot row)
            std::vector<int> pivot_row_structure;
            for (int col : u_pattern[k]) {
                if (col > k) {  // Only consider columns to the right of pivot
                    pivot_row_structure.push_back(col);
                }
            }

            // For each row below pivot row
            for (int i = k + 1; i < matrix_size_; ++i) {
                if (l_pattern[i].count(k) > 0) {
                    // Row i will be modified by elimination step k
                    // Only add fill-in for structural interactions
                    for (int col : pivot_row_structure) {
                        // Conservative: only add if there's a path through existing structure
                        bool add_to_l = (col <= i) && (col > k);
                        bool add_to_u = (col >= i) && (col > k);

                        if (add_to_l) {
                            // Only add if it creates a meaningful structural connection
                            if (l_pattern[i].size() < static_cast<size_t>(matrix_size_) / 2) {
                                l_pattern[i].insert(col);
                            }
                        }
                        if (add_to_u) {
                            if (u_pattern[i].size() < static_cast<size_t>(matrix_size_) / 2) {
                                u_pattern[i].insert(col);
                            }
                        }
                    }
                }
            }
        }

        // Convert patterns to CSR format
        symbolic_.l_nnz = 0;
        symbolic_.u_nnz = 0;

        // Build L matrix structure
        for (int row = 0; row < matrix_size_; ++row) {
            symbolic_.l_row_ptr[row] = symbolic_.l_nnz;
            for (int col : l_pattern[row]) {
                symbolic_.l_col_idx.push_back(col);
                symbolic_.l_nnz++;
            }
        }
        symbolic_.l_row_ptr[matrix_size_] = symbolic_.l_nnz;

        // Build U matrix structure
        for (int row = 0; row < matrix_size_; ++row) {
            symbolic_.u_row_ptr[row] = symbolic_.u_nnz;
            for (int col : u_pattern[row]) {
                symbolic_.u_col_idx.push_back(col);
                symbolic_.u_nnz++;
            }
        }
        symbolic_.u_row_ptr[matrix_size_] = symbolic_.u_nnz;

        symbolic_.valid = true;

        double fill_ratio = static_cast<double>(symbolic_.l_nnz + symbolic_.u_nnz) / matrix.nnz;
        LOG_DEBUG(logger, "    L nnz:", symbolic_.l_nnz, ", U nnz:", symbolic_.u_nnz);
        LOG_DEBUG(logger, "    Fill ratio:", fill_ratio, "x");

        return true;
    }

    /**
     * @brief Phase 2: Numerical Factorization
     * Perform sparse Gaussian elimination with threshold pivoting
     */
    bool perform_numerical_factorization(const SparseMatrix& matrix) {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPULUSolver");
        LOG_INFO(logger, "  Phase 2: Numerical factorization...");

        if (!symbolic_.valid) {
            logger.logError("Symbolic analysis must be completed first");
            return false;
        }

        // Initialize permutation matrices (start with identity)
        numerical_.pivot_row.resize(matrix_size_);
        numerical_.pivot_col.resize(matrix_size_);
        std::iota(numerical_.pivot_row.begin(), numerical_.pivot_row.end(), 0);
        std::iota(numerical_.pivot_col.begin(), numerical_.pivot_col.end(), 0);

        // Initialize factor storage with zeros
        numerical_.l_values.assign(symbolic_.l_nnz, Complex(0.0, 0.0));
        numerical_.u_values.assign(symbolic_.u_nnz, Complex(0.0, 0.0));

        // SIMPLIFIED APPROACH: Use controlled dense factorization for accuracy
        // This ensures numerical accuracy while being mindful of memory usage

        // Only use dense approach for matrices that fit reasonably in memory
        if (matrix_size_ > 1000) {
            logger.logError("Matrix too large for current implementation (size=" +
                            std::to_string(matrix_size_) + ")");
            return false;
        }

        // Create a working dense matrix - this is temporary for accuracy
        std::vector<std::vector<Complex>> dense_work(
            matrix_size_, std::vector<Complex>(matrix_size_, Complex(0.0, 0.0)));

        // Copy original matrix to dense working format
        for (int row = 0; row < matrix_size_; ++row) {
            for (int idx = matrix.row_ptr[row]; idx < matrix.row_ptr[row + 1]; ++idx) {
                int col = matrix.col_idx[idx];
                dense_work[row][col] = matrix.values[idx];
            }
        }

        // Perform LU factorization with scaled partial pivoting for maximum accuracy
        std::vector<double> scale(matrix_size_);

        // Compute scaling factors (largest element in each row)
        for (int i = 0; i < matrix_size_; ++i) {
            double max_val = 0.0;
            for (int j = 0; j < matrix_size_; ++j) {
                max_val = std::max(max_val, std::abs(dense_work[i][j]));
            }
            scale[i] = (max_val > 0.0) ? max_val : 1.0;
        }

        // Main elimination with scaled partial pivoting
        for (int k = 0; k < matrix_size_ - 1; ++k) {
            // Find pivot with scaled partial pivoting
            int pivot_row = k;
            double max_scaled = std::abs(dense_work[k][k]) / scale[k];

            for (int i = k + 1; i < matrix_size_; ++i) {
                double scaled_val = std::abs(dense_work[i][k]) / scale[i];
                if (scaled_val > max_scaled) {
                    max_scaled = scaled_val;
                    pivot_row = i;
                }
            }

            // Check for singularity
            if (std::abs(dense_work[pivot_row][k]) < pivot_tolerance_) {
                logger.logError("Matrix is numerically singular at step " + std::to_string(k) +
                                " (pivot " + std::to_string(std::abs(dense_work[pivot_row][k])) +
                                ")");
                return false;
            }

            // Row interchange
            if (pivot_row != k) {
                std::swap(dense_work[k], dense_work[pivot_row]);
                std::swap(numerical_.pivot_row[k], numerical_.pivot_row[pivot_row]);
                std::swap(scale[k], scale[pivot_row]);
            }

            Complex pivot = dense_work[k][k];

            // Elimination
            for (int i = k + 1; i < matrix_size_; ++i) {
                if (std::abs(dense_work[i][k]) > drop_tolerance_) {
                    Complex multiplier = dense_work[i][k] / pivot;
                    dense_work[i][k] = multiplier;  // Store L factor

                    for (int j = k + 1; j < matrix_size_; ++j) {
                        dense_work[i][j] -= multiplier * dense_work[k][j];
                    }
                }
            }
        }

        // Rebuild symbolic structure based on actual factorization results
        // This ensures perfect alignment between symbolic and numerical phases

        // Clear and rebuild L and U structures
        symbolic_.l_row_ptr.assign(matrix_size_ + 1, 0);
        symbolic_.u_row_ptr.assign(matrix_size_ + 1, 0);
        symbolic_.l_col_idx.clear();
        symbolic_.u_col_idx.clear();
        numerical_.l_values.clear();
        numerical_.u_values.clear();

        // Build L factors (lower triangular with unit diagonal)
        for (int row = 0; row < matrix_size_; ++row) {
            symbolic_.l_row_ptr[row] = numerical_.l_values.size();

            for (int col = 0; col <= row; ++col) {
                Complex value = (col == row) ? Complex(1.0, 0.0) : dense_work[row][col];

                if (std::abs(value) > drop_tolerance_ || col == row) {
                    symbolic_.l_col_idx.push_back(col);
                    numerical_.l_values.push_back(value);
                }
            }
        }
        symbolic_.l_row_ptr[matrix_size_] = numerical_.l_values.size();
        symbolic_.l_nnz = numerical_.l_values.size();

        // Build U factors (upper triangular)
        for (int row = 0; row < matrix_size_; ++row) {
            symbolic_.u_row_ptr[row] = numerical_.u_values.size();

            for (int col = row; col < matrix_size_; ++col) {
                Complex value = dense_work[row][col];

                if (std::abs(value) > drop_tolerance_) {
                    symbolic_.u_col_idx.push_back(col);
                    numerical_.u_values.push_back(value);
                }
            }
        }
        symbolic_.u_row_ptr[matrix_size_] = numerical_.u_values.size();
        symbolic_.u_nnz = numerical_.u_values.size();

        numerical_.valid = true;

        // Calculate actual non-zeros after dropping small elements
        int actual_l_nnz = 0, actual_u_nnz = 0;
        for (const auto& val : numerical_.l_values) {
            if (std::abs(val) > drop_tolerance_) actual_l_nnz++;
        }
        for (const auto& val : numerical_.u_values) {
            if (std::abs(val) > drop_tolerance_) actual_u_nnz++;
        }

        LOG_DEBUG(logger, "    Pivot tolerance:", pivot_tolerance_);
        LOG_DEBUG(logger, "    Drop tolerance:", drop_tolerance_);
        LOG_DEBUG(logger, "    Growth factor limit:", growth_factor_limit_);
        LOG_DEBUG(logger, "    Actual L nnz:", actual_l_nnz, "/", symbolic_.l_nnz);
        LOG_DEBUG(logger, "    Actual U nnz:", actual_u_nnz, "/", symbolic_.u_nnz);

        return true;
    }

    /**
     * @brief Apply row permutation: result = P * vector
     */
    ComplexVector apply_row_permutation(const ComplexVector& vector) const {
        ComplexVector result(vector.size());
        for (int i = 0; i < matrix_size_; ++i) {
            result[i] = vector[numerical_.pivot_row[i]];
        }
        return result;
    }

    /**
     * @brief Apply inverse column permutation: result = Q^T * vector
     */
    ComplexVector apply_column_permutation_inverse(const ComplexVector& vector) const {
        ComplexVector result(vector.size());
        for (int i = 0; i < matrix_size_; ++i) {
            result[numerical_.pivot_col[i]] = vector[i];
        }
        return result;
    }

    /**
     * @brief Forward substitution: solve L * y = b for y
     * L is unit lower triangular (diagonal elements = 1)
     */
    ComplexVector forward_substitution(const ComplexVector& rhs) const {
        ComplexVector result(matrix_size_, Complex(0.0, 0.0));

        // Forward substitution: L * y = b
        // Since L is unit lower triangular, diagonal elements are 1
        for (int row = 0; row < matrix_size_; ++row) {
            Complex sum(0.0, 0.0);

            // Process non-zero elements in L[row][0:row-1]
            for (int idx = symbolic_.l_row_ptr[row]; idx < symbolic_.l_row_ptr[row + 1]; ++idx) {
                int col = symbolic_.l_col_idx[idx];

                if (col < row) {
                    // Off-diagonal element: L[row][col] * y[col]
                    sum += numerical_.l_values[idx] * result[col];
                }
                // Skip diagonal element (implicitly 1.0)
            }

            // y[row] = (b[row] - sum) / L[row][row]
            // Since L[row][row] = 1.0 for unit lower triangular matrix
            result[row] = rhs[row] - sum;
        }

        return result;
    }

    /**
     * @brief Backward substitution: solve U * x = y for x
     * U is upper triangular with non-unit diagonal
     */
    ComplexVector backward_substitution(const ComplexVector& rhs) const {
        ComplexVector result(matrix_size_, Complex(0.0, 0.0));

        // Backward substitution: U * x = y
        for (int row = matrix_size_ - 1; row >= 0; --row) {
            Complex sum(0.0, 0.0);
            Complex diagonal(0.0, 0.0);

            // Process non-zero elements in U[row][row:n-1]
            for (int idx = symbolic_.u_row_ptr[row]; idx < symbolic_.u_row_ptr[row + 1]; ++idx) {
                int col = symbolic_.u_col_idx[idx];

                if (col == row) {
                    // Diagonal element: U[row][row]
                    diagonal = numerical_.u_values[idx];
                } else if (col > row) {
                    // Off-diagonal element: U[row][col] * x[col]
                    sum += numerical_.u_values[idx] * result[col];
                }
            }

            // Check for zero diagonal (singular matrix)
            if (std::abs(diagonal) < pivot_tolerance_) {
                throw std::runtime_error(
                    "Singular matrix encountered in backward substitution at row " +
                    std::to_string(row));
            }

            // x[row] = (y[row] - sum) / U[row][row]
            result[row] = (rhs[row] - sum) / diagonal;
        }

        return result;
    }
};

}  // namespace gap::solver
