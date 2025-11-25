#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <ranges>
#include <vector>

#include "gap/admittance/admittance_interface.h"
#include "gap/logging/logger.h"

namespace gap::admittance {

class CPUAdmittanceMatrix : public IAdmittanceMatrix {
  private:
    // Triplet: sparse matrix element for efficient construction
    struct Triplet {
        int row;
        int col;
        Complex value;

        // Sort by (row, col) for CSR format
        bool operator<(Triplet const& other) const {
            if (row != other.row) return row < other.row;
            return col < other.col;
        }

        bool operator==(Triplet const& other) const { return row == other.row && col == other.col; }
    };

    // Construction statistics for debugging
    struct ConstructionStats {
        double build_time_ms = 0.0;
        size_t triplets_allocated = 0;
        size_t triplets_used = 0;
        size_t final_nnz = 0;
        size_t memory_bytes = 0;
    };

    mutable ConstructionStats last_stats_;

  public:
    // Get construction statistics (for debugging/profiling)
    ConstructionStats get_construction_stats() const { return last_stats_; }
    std::unique_ptr<SparseMatrix> build_admittance_matrix(
        NetworkData const& network_data) override {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPUAdmittanceMatrix");
        LOG_INFO(logger, "Building admittance matrix");
        LOG_INFO(logger, "  Number of buses:", network_data.num_buses);
        LOG_INFO(logger, "  Number of branches:", network_data.num_branches);
        LOG_INFO(logger, "  Number of appliances:", network_data.appliances.size());

        // Auto-detect bus ID scheme and create mapping
        int min_bus_id = network_data.num_buses;
        int max_bus_id = -1;

        for (auto const& bus : network_data.buses) {
            min_bus_id = std::min(min_bus_id, bus.id);
            max_bus_id = std::max(max_bus_id, bus.id);
        }

        int id_offset = min_bus_id;

        // Create ID-to-index mapping for robust lookup
        std::vector<int> id_to_idx(max_bus_id + 1, -1);
        for (size_t idx = 0; idx < network_data.buses.size(); ++idx) {
            int bus_id = network_data.buses[idx].id;
            if (bus_id >= 0 && bus_id <= max_bus_id) {
                id_to_idx[bus_id] = static_cast<int>(idx);
            }
        }

        LOG_INFO(logger, "  Bus ID range: [", min_bus_id, ",", max_bus_id, "], offset:", id_offset);

        // === OPTIMIZATION: Triplet-based accumulation ===
        // Estimate non-zeros: diagonal (n) + branches (2 per branch, bidirectional)
        size_t estimated_nnz = network_data.num_buses + 2 * network_data.num_branches;
        std::vector<Triplet> triplets;
        triplets.reserve(estimated_nnz);

        last_stats_.triplets_allocated = estimated_nnz;

        // Accumulate diagonal elements from shunt appliances
        std::vector<Complex> diagonal_accumulator(network_data.num_buses, Complex(0.0, 0.0));

        for (auto const& appliance : network_data.appliances) {
            if (appliance.type == ApplianceType::SHUNT) {
                if (appliance.node >= 0 && appliance.node <= max_bus_id) {
                    int bus_idx = id_to_idx[appliance.node];
                    if (bus_idx >= 0 && bus_idx < network_data.num_buses) {
                        diagonal_accumulator[bus_idx] += Complex(appliance.g1, appliance.b1);
                    }
                }
            }
        }

        // Process branches and accumulate admittances as triplets
        for (auto const& branch : network_data.branches) {
            if (!branch.status) {
                continue;
            }

            int from_bus = -1, to_bus = -1;
            if (branch.from_bus >= 0 && branch.from_bus <= max_bus_id) {
                from_bus = id_to_idx[branch.from_bus];
            }
            if (branch.to_bus >= 0 && branch.to_bus <= max_bus_id) {
                to_bus = id_to_idx[branch.to_bus];
            }

            if (from_bus < 0 || from_bus >= network_data.num_buses || to_bus < 0 ||
                to_bus >= network_data.num_buses) {
                LOG_WARN(logger, "  Skipping branch", branch.id,
                         "with invalid bus IDs:", branch.from_bus, "→", branch.to_bus);
                continue;
            }

            // Calculate branch admittance
            Float z_magnitude_sq = branch.r1 * branch.r1 + branch.x1 * branch.x1;
            Complex series_admittance;
            if (z_magnitude_sq > 1e-12) {
                series_admittance =
                    Complex(branch.r1 / z_magnitude_sq, -branch.x1 / z_magnitude_sq);
            } else {
                series_admittance = Complex(0.0, 0.0);
            }

            Complex branch_admittance = series_admittance + Complex(branch.g1, 0.0);
            Complex shunt_admittance = Complex(0.0, branch.b1 / 2.0);

            // Accumulate diagonal elements (self-admittance + shunt)
            diagonal_accumulator[from_bus] += branch_admittance + shunt_admittance;
            diagonal_accumulator[to_bus] += branch_admittance + shunt_admittance;

            // Add off-diagonal triplets (mutual admittance is negative)
            triplets.push_back({from_bus, to_bus, -series_admittance});
            triplets.push_back({to_bus, from_bus, -series_admittance});
        }

        // Add diagonal elements as triplets
        for (int i = 0; i < network_data.num_buses; ++i) {
            triplets.push_back({i, i, diagonal_accumulator[i]});
        }

        last_stats_.triplets_used = triplets.size();

        // === OPTIMIZATION: Stable sort ensures deterministic ordering for duplicate (row,col)
        // entries === (needed when multiple branches connect the same bus pair)
        std::ranges::stable_sort(triplets, [](Triplet const& a, Triplet const& b) {
            if (a.row != b.row) return a.row < b.row;
            return a.col < b.col;
        });

        // Build CSR format directly from sorted triplets
        auto matrix = std::make_unique<SparseMatrix>();
        matrix->num_rows = network_data.num_buses;
        matrix->num_cols = network_data.num_buses;

        // Pre-allocate CSR arrays
        matrix->col_idx.reserve(triplets.size());
        matrix->values.reserve(triplets.size());
        matrix->row_ptr.reserve(network_data.num_buses + 1);

        int current_row = 0;
        matrix->row_ptr.push_back(0);

        for (auto const& triplet : triplets) {
            // Fill row_ptr for any empty rows
            while (current_row < triplet.row) {
                matrix->row_ptr.push_back(static_cast<int>(matrix->col_idx.size()));
                current_row++;
            }

            matrix->col_idx.push_back(triplet.col);
            matrix->values.push_back(triplet.value);
        }

        // Fill remaining row_ptr entries
        while (current_row < network_data.num_buses) {
            matrix->row_ptr.push_back(static_cast<int>(matrix->col_idx.size()));
            current_row++;
        }

        matrix->nnz = static_cast<int>(matrix->values.size());
        last_stats_.final_nnz = matrix->nnz;
        last_stats_.memory_bytes = matrix->values.size() * sizeof(Complex) +
                                   matrix->col_idx.size() * sizeof(int) +
                                   matrix->row_ptr.size() * sizeof(int);

        auto end_time = std::chrono::high_resolution_clock::now();
        last_stats_.build_time_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();

        LOG_INFO(logger, "  Matrix constructed with", matrix->nnz, "non-zero elements");
        LOG_DEBUG(logger, "  Build time:", last_stats_.build_time_ms, "ms");
        LOG_DEBUG(logger, "  Triplets: allocated", last_stats_.triplets_allocated, "used",
                  last_stats_.triplets_used);

        return matrix;
    }

    std::unique_ptr<SparseMatrix> update_admittance_matrix(
        SparseMatrix const& matrix, const std::vector<BranchData>& branch_changes) override {
        auto& logger = gap::logging::global_logger;
        logger.setComponent("CPUAdmittanceMatrix");
        LOG_INFO(logger, "Updating admittance matrix");
        LOG_INFO(logger, "  Branch changes:", branch_changes.size());

        // Create a copy of the matrix
        auto updated_matrix = std::make_unique<SparseMatrix>(matrix);

        // For each branch change, find and update the affected matrix elements
        for (auto const& branch_change : branch_changes) {
            // Note: Assuming 0-based indexing for consistency with build_admittance_matrix
            // If branch_change uses 1-based, adjust here
            int from_bus = branch_change.from_bus;
            int to_bus = branch_change.to_bus;

            // Check if we need to convert from 1-based to 0-based
            if (from_bus > 0 && to_bus > 0 &&
                (from_bus > updated_matrix->num_rows || to_bus > updated_matrix->num_rows)) {
                from_bus -= 1;
                to_bus -= 1;
            }

            if (from_bus < 0 || from_bus >= updated_matrix->num_rows || to_bus < 0 ||
                to_bus >= updated_matrix->num_rows) {
                LOG_WARN(logger, "  Skipping branch update with invalid indices:", from_bus, "→",
                         to_bus);
                continue;
            }

            LOG_DEBUG(logger, "  Updating branch", from_bus, "→", to_bus,
                      "(status:", (branch_change.status ? "in-service" : "out-of-service"), ")");

            // Calculate branch admittance change
            Float z_magnitude_sq =
                branch_change.r1 * branch_change.r1 + branch_change.x1 * branch_change.x1;
            Complex series_admittance_change = Complex(0.0, 0.0);
            if (z_magnitude_sq > 1e-12) {
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

            Complex shunt_change = Complex(0.0, branch_change.b1 / 2.0);
            if (!branch_change.status) {
                shunt_change = -shunt_change;
            }

            // === OPTIMIZATION: Direct CSR lookup instead of full scan ===
            // Update diagonal elements for from_bus
            for (int idx = updated_matrix->row_ptr[from_bus];
                 idx < updated_matrix->row_ptr[from_bus + 1]; ++idx) {
                if (updated_matrix->col_idx[idx] == from_bus) {
                    updated_matrix->values[idx] += total_admittance_change + shunt_change;
                    break;
                }
            }

            // Update diagonal elements for to_bus
            for (int idx = updated_matrix->row_ptr[to_bus];
                 idx < updated_matrix->row_ptr[to_bus + 1]; ++idx) {
                if (updated_matrix->col_idx[idx] == to_bus) {
                    updated_matrix->values[idx] += total_admittance_change + shunt_change;
                    break;
                }
            }

            // Update off-diagonal element (from_bus, to_bus)
            for (int idx = updated_matrix->row_ptr[from_bus];
                 idx < updated_matrix->row_ptr[from_bus + 1]; ++idx) {
                if (updated_matrix->col_idx[idx] == to_bus) {
                    updated_matrix->values[idx] -= series_admittance_change;
                    break;
                }
            }

            // Update off-diagonal element (to_bus, from_bus)
            for (int idx = updated_matrix->row_ptr[to_bus];
                 idx < updated_matrix->row_ptr[to_bus + 1]; ++idx) {
                if (updated_matrix->col_idx[idx] == from_bus) {
                    updated_matrix->values[idx] -= series_admittance_change;
                    break;
                }
            }
        }

        LOG_INFO(logger, "  Matrix update completed");
        return updated_matrix;
    }

    BackendType get_backend_type() const noexcept override { return BackendType::CPU; }
};

}  // namespace gap::admittance
