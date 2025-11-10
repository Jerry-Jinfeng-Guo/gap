#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "gap/admittance/admittance_interface.h"
#include "gap/logging/logger.h"

namespace gap::admittance {

class CPUAdmittanceMatrix : public IAdmittanceMatrix {
  public:
    std::unique_ptr<SparseMatrix> build_admittance_matrix(
        NetworkData const& network_data) override {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPUAdmittanceMatrix");
        LOG_INFO(logger, "Building admittance matrix");
        LOG_INFO(logger, "  Number of buses:", network_data.num_buses);
        LOG_INFO(logger, "  Number of branches:", network_data.num_branches);
        LOG_INFO(logger, "  Number of appliances:", network_data.appliances.size());

        auto matrix = std::make_unique<SparseMatrix>();
        matrix->num_rows = network_data.num_buses;
        matrix->num_cols = network_data.num_buses;

        // Initialize diagonal admittance accumulator for each bus
        std::vector<Complex> diagonal_elements(network_data.num_buses, Complex(0.0, 0.0));
        std::vector<std::vector<std::pair<int, Complex>>> off_diagonal_elements(
            network_data.num_buses);

        // Process shunt appliances (capacitors, reactors) first
        // Use actual appliances vector size rather than potentially uninitialized num_appliances
        for (auto const& appliance : network_data.appliances) {
            if (appliance.type == ApplianceType::SHUNT) {
                int bus_idx = appliance.node - 1;  // Convert to 0-based indexing
                if (bus_idx >= 0 && bus_idx < network_data.num_buses) {
                    // Add shunt admittance to diagonal element
                    diagonal_elements[bus_idx] += Complex(appliance.g1, appliance.b1);
                }
            }
        }

        // Iterate over all branches to build admittance matrix
        for (auto const& branch : network_data.branches) {
            if (!branch.status) {
                continue;  // Skip out-of-service branches
            }

            int from_bus = branch.from_bus - 1;  // Convert to 0-based indexing
            int to_bus = branch.to_bus - 1;      // Convert to 0-based indexing

            // Calculate branch admittance from r1, x1, g1, b1 parameters
            // Series impedance: Z = r1 + j*x1
            // Series admittance: Y_series = 1/Z = g1_calc + j*b1_calc
            // where g1_calc = r1/(r1^2 + x1^2) and b1_calc = -x1/(r1^2 + x1^2)
            Float z_magnitude_sq = branch.r1 * branch.r1 + branch.x1 * branch.x1;
            Complex series_admittance;
            if (z_magnitude_sq > 1e-12) {  // Avoid division by zero
                series_admittance =
                    Complex(branch.r1 / z_magnitude_sq, -branch.x1 / z_magnitude_sq);
            } else {
                series_admittance = Complex(0.0, 0.0);  // Open circuit
            }

            // Add parallel conductance and susceptance
            Complex branch_admittance = series_admittance + Complex(branch.g1, branch.b1);

            // Add to diagonal elements (self-admittance)
            // Include half of line charging susceptance at each end
            diagonal_elements[from_bus] += branch_admittance + Complex(0.0, branch.b1 / 2.0);
            diagonal_elements[to_bus] += branch_admittance + Complex(0.0, branch.b1 / 2.0);

            // Add off-diagonal elements (mutual admittance is negative, only series part)
            off_diagonal_elements[from_bus].emplace_back(to_bus, -series_admittance);
            off_diagonal_elements[to_bus].emplace_back(from_bus, -series_admittance);
        }

        // Build CSR format sparse matrix
        matrix->row_ptr.clear();
        matrix->col_idx.clear();
        matrix->values.clear();
        matrix->row_ptr.reserve(network_data.num_buses + 1);

        int nnz = 0;
        matrix->row_ptr.push_back(0);

        for (int i = 0; i < network_data.num_buses; ++i) {
            // Sort off-diagonal elements by column index
            std::sort(off_diagonal_elements[i].begin(), off_diagonal_elements[i].end(),
                      [](const std::pair<int, std::complex<Float>>& a,
                         const std::pair<int, std::complex<Float>>& b) {
                          return a.first < b.first;  // Sort by column index only
                      });

            // Build complete row with proper column ordering
            std::vector<std::pair<int, Complex>> row_elements;

            // Add diagonal element
            row_elements.emplace_back(i, diagonal_elements[i]);

            // Add off-diagonal elements
            for (auto const& [col, value] : off_diagonal_elements[i]) {
                row_elements.emplace_back(col, value);
            }

            // Sort all elements (diagonal + off-diagonal) by column index
            std::sort(row_elements.begin(), row_elements.end(),
                      [](const std::pair<int, Complex>& a, const std::pair<int, Complex>& b) {
                          return a.first < b.first;
                      });

            // Add sorted elements to CSR arrays
            for (auto const& [col, value] : row_elements) {
                matrix->col_idx.push_back(col);
                matrix->values.push_back(value);
                nnz++;
            }

            matrix->row_ptr.push_back(nnz);
        }

        matrix->nnz = nnz;

        LOG_INFO(logger, "  Matrix constructed with", matrix->nnz, "non-zero elements");
        return matrix;
    }

    std::unique_ptr<SparseMatrix> update_admittance_matrix(
        const SparseMatrix& matrix, const std::vector<BranchData>& branch_changes) override {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPUAdmittanceMatrix");
        LOG_INFO(logger, "Updating admittance matrix");
        LOG_INFO(logger, "  Branch changes:", branch_changes.size());

        auto updated_matrix = std::make_unique<SparseMatrix>(matrix);

        // Iterate over each branch change to update the matrix
        for (auto const& branch_change : branch_changes) {
            int from_bus = branch_change.from_bus - 1;  // Convert to 0-based indexing
            int to_bus = branch_change.to_bus - 1;      // Convert to 0-based indexing

            LOG_DEBUG(logger, "  Updating branch", branch_change.from_bus, "->",
                      branch_change.to_bus,
                      "(status:", (branch_change.status ? "in-service" : "out-of-service"), ")");

            // Calculate branch admittance change using new r1, x1, g1, b1 parameters
            Complex series_admittance_change = Complex(0.0, 0.0);
            Float z_magnitude_sq =
                branch_change.r1 * branch_change.r1 + branch_change.x1 * branch_change.x1;
            if (z_magnitude_sq > 1e-12) {  // Avoid division by zero
                series_admittance_change =
                    Complex(branch_change.r1 / z_magnitude_sq, -branch_change.x1 / z_magnitude_sq);
            }

            Complex total_admittance_change =
                series_admittance_change + Complex(branch_change.g1, branch_change.b1);

            if (!branch_change.status) {
                // Branch going out of service - subtract admittance
                total_admittance_change = -total_admittance_change;
                series_admittance_change = -series_admittance_change;
            }

            // Update diagonal elements (self-admittance changes)
            Complex shunt_change = Complex(0.0, branch_change.b1 / 2.0);
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
                        updated_matrix->values[idx] += total_admittance_change + shunt_change;
                    }
                    // Update off-diagonal elements (only series admittance)
                    else if ((row == from_bus && col == to_bus) ||
                             (row == to_bus && col == from_bus)) {
                        updated_matrix->values[idx] -=
                            series_admittance_change;  // Off-diagonal is negative
                    }
                }
            }
        }

        LOG_INFO(logger, "  Matrix update completed");
        return updated_matrix;
    }

    BackendType get_backend_type() const noexcept override { return BackendType::CPU; }
};

}  // namespace gap::admittance
