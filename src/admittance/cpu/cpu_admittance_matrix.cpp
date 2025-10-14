#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "gap/admittance/admittance_interface.h"

namespace gap::admittance {

class CPUAdmittanceMatrix : public IAdmittanceMatrix {
  public:
    std::unique_ptr<SparseMatrix> build_admittance_matrix(
        const NetworkData& network_data) override {
        std::cout << "CPUAdmittanceMatrix: Building admittance matrix" << std::endl;
        std::cout << "  Number of buses: " << network_data.num_buses << std::endl;
        std::cout << "  Number of branches: " << network_data.num_branches << std::endl;

        auto matrix = std::make_unique<SparseMatrix>();
        matrix->num_rows = network_data.num_buses;
        matrix->num_cols = network_data.num_buses;

        // Initialize diagonal admittance accumulator for each bus
        std::vector<Complex> diagonal_elements(network_data.num_buses, Complex(0.0, 0.0));
        std::vector<std::vector<std::pair<int, Complex>>> off_diagonal_elements(
            network_data.num_buses);

        constexpr Complex one_comp{1.0, 0.0};
        // Iterate over all branches to build admittance matrix
        for (const auto& branch : network_data.branches) {
            if (!branch.status) {
                continue;  // Skip out-of-service branches
            }

            int from_bus = branch.from_bus - 1;  // Convert to 0-based indexing
            int to_bus = branch.to_bus - 1;      // Convert to 0-based indexing

            // Calculate branch admittance (Y = 1/Z)
            Complex branch_admittance = one_comp / branch.impedance;

            // Add to diagonal elements (self-admittance)
            diagonal_elements[from_bus] +=
                branch_admittance + Complex(0.0, branch.susceptance / 2.0);
            diagonal_elements[to_bus] += branch_admittance + Complex(0.0, branch.susceptance / 2.0);

            // Add off-diagonal elements (mutual admittance is negative)
            off_diagonal_elements[from_bus].emplace_back(to_bus, -branch_admittance);
            off_diagonal_elements[to_bus].emplace_back(from_bus, -branch_admittance);
        }

        // Build CSR format sparse matrix
        matrix->row_ptr.clear();
        matrix->col_idx.clear();
        matrix->values.clear();
        matrix->row_ptr.reserve(network_data.num_buses + 1);

        int nnz = 0;
        matrix->row_ptr.push_back(0);

        for (int i = 0; i < network_data.num_buses; ++i) {
            // Add diagonal element
            matrix->col_idx.push_back(i);
            matrix->values.push_back(diagonal_elements[i]);
            nnz++;

            // Sort off-diagonal elements by column index
            std::sort(off_diagonal_elements[i].begin(), off_diagonal_elements[i].end(),
                      [](const std::pair<int, std::complex<double>>& a,
                         const std::pair<int, std::complex<double>>& b) {
                          return a.first < b.first;  // Sort by column index only
                      });

            // Add off-diagonal elements
            for (const auto& [col, value] : off_diagonal_elements[i]) {
                matrix->col_idx.push_back(col);
                matrix->values.push_back(value);
                nnz++;
            }

            matrix->row_ptr.push_back(nnz);
        }

        matrix->nnz = nnz;

        std::cout << "  Matrix constructed with " << matrix->nnz << " non-zero elements"
                  << std::endl;
        return matrix;
    }

    std::unique_ptr<SparseMatrix> update_admittance_matrix(
        const SparseMatrix& matrix, const std::vector<BranchData>& branch_changes) override {
        std::cout << "CPUAdmittanceMatrix: Updating admittance matrix" << std::endl;
        std::cout << "  Branch changes: " << branch_changes.size() << std::endl;

        auto updated_matrix = std::make_unique<SparseMatrix>(matrix);

        // Iterate over each branch change to update the matrix
        for (const auto& branch_change : branch_changes) {
            int from_bus = branch_change.from_bus - 1;  // Convert to 0-based indexing
            int to_bus = branch_change.to_bus - 1;      // Convert to 0-based indexing

            std::cout << "  Updating branch " << branch_change.from_bus << " -> "
                      << branch_change.to_bus
                      << " (status: " << (branch_change.status ? "in-service" : "out-of-service")
                      << ")" << std::endl;

            // Calculate branch admittance change
            Complex branch_admittance_change = Complex(0.0, 0.0);
            if (branch_change.status) {
                // Branch coming into service - add admittance
                branch_admittance_change = Complex(1.0, 0.0) / branch_change.impedance;
            } else {
                // Branch going out of service - subtract admittance
                branch_admittance_change = -(Complex(1.0, 0.0) / branch_change.impedance);
            }

            // Update diagonal elements (self-admittance changes)
            Complex shunt_change = Complex(0.0, branch_change.susceptance / 2.0);
            if (!branch_change.status) {
                shunt_change = -shunt_change;  // Remove shunt if going out of service
            }

            // Find and update matrix elements
            // This is a simplified approach - in practice, we'd need more efficient sparse matrix
            // updates
            for (int row = 0; row < updated_matrix->num_rows; ++row) {
                int start_idx = updated_matrix->row_ptr[row];
                int end_idx = updated_matrix->row_ptr[row + 1];

                for (int idx = start_idx; idx < end_idx; ++idx) {
                    int col = updated_matrix->col_idx[idx];

                    // Update diagonal elements
                    if (row == col && (row == from_bus || row == to_bus)) {
                        updated_matrix->values[idx] += branch_admittance_change + shunt_change;
                    }
                    // Update off-diagonal elements
                    else if ((row == from_bus && col == to_bus) ||
                             (row == to_bus && col == from_bus)) {
                        updated_matrix->values[idx] -=
                            branch_admittance_change;  // Off-diagonal is negative
                    }
                }
            }
        }

        std::cout << "  Matrix update completed" << std::endl;
        return updated_matrix;
    }

    BackendType get_backend_type() const override { return BackendType::CPU; }
};

}  // namespace gap::admittance
