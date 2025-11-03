#include <cmath>

#include "gap/admittance/admittance_interface.h"
#include "gap/core/backend_factory.h"

#include "test_framework.h"

using namespace gap;

void test_cpu_admittance_creation() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);
    ASSERT_TRUE(admittance != nullptr);
    ASSERT_BACKEND_EQ(BackendType::CPU, admittance->get_backend_type());
}

void test_admittance_matrix_build() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create dummy network data
    NetworkData network;
    network.num_buses = 3;
    network.num_branches = 2;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,  // 230 kV nominal voltage
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};  // Slack bus
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,  // 230 kV nominal voltage
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,    // 100 MW in watts
                    .reactive_power = 50e6};  // 50 MVAr in VAr
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,  // 230 kV nominal voltage
                    .bus_type = BusType::PV,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 150e6,   // 150 MW in watts
                    .reactive_power = 0.0};  // PV bus

    network.buses = {bus1, bus2, bus3};

    auto matrix = admittance->build_admittance_matrix(network);
    ASSERT_TRUE(matrix != nullptr);
    ASSERT_EQ(3, matrix->num_rows);
    ASSERT_EQ(3, matrix->num_cols);
}

void test_admittance_matrix_update() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create dummy network data and matrix
    NetworkData network;
    network.num_buses = 2;
    network.num_branches = 1;

    auto matrix = admittance->build_admittance_matrix(network);

    // Create branch changes
    std::vector<BranchData> changes = {{.id = 1,
                                        .from_bus = 1,
                                        .from_status = 1,
                                        .to_bus = 2,
                                        .to_status = 1,
                                        .status = 0,    // Out of service
                                        .r1 = 0.01,     // resistance (ohm)
                                        .x1 = 0.1,      // reactance (ohm)
                                        .g1 = 0.0,      // conductance (siemens)
                                        .b1 = 0.0,      // susceptance (siemens)
                                        .k = 1.0,       // tap ratio
                                        .theta = 0.0,   // phase shift
                                        .sn = 100e6}};  // 100 MVA rating

    auto updated_matrix = admittance->update_admittance_matrix(*matrix, changes);
    ASSERT_TRUE(updated_matrix != nullptr);
}

void test_admittance_matrix_branch_iteration() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create a simple 2-bus system with one branch
    NetworkData network;
    network.num_buses = 2;
    network.num_branches = 1;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,    // 100 MW
                    .reactive_power = 50e6};  // 50 MVAr
    network.buses = {bus1, bus2};

    // Branch with r1=0.02 ohm, x1=0.06 ohm, and b1=0.05 siemens
    BranchData branch = {.id = 1,
                         .from_bus = 1,
                         .from_status = 1,
                         .to_bus = 2,
                         .to_status = 1,
                         .status = 1,   // In service
                         .r1 = 0.02,    // resistance (ohm)
                         .x1 = 0.06,    // reactance (ohm)
                         .g1 = 0.0,     // conductance (siemens)
                         .b1 = 0.05,    // susceptance (siemens)
                         .k = 1.0,      // tap ratio
                         .theta = 0.0,  // phase shift
                         .sn = 100e6};  // 100 MVA rating
    network.branches = {branch};

    auto matrix = admittance->build_admittance_matrix(network);

    // Verify matrix dimensions
    ASSERT_EQ(2, matrix->num_rows);
    ASSERT_EQ(2, matrix->num_cols);

    // For a 2x2 matrix with one branch, we should have 4 non-zero elements
    // (2 diagonal + 2 off-diagonal)
    ASSERT_EQ(4, matrix->nnz);

    // Calculate expected admittance: Y = 1 / (0.02 + j0.06)
    // Complex expected_admittance = Complex(1.0, 0.0) / Complex(0.02, 0.06);
    // double expected_shunt = 0.05 / 2.0;

    // Verify CSR structure: row_ptr should be [0, 2, 4]
    ASSERT_EQ(3, matrix->row_ptr.size());
    ASSERT_EQ(0, matrix->row_ptr[0]);
    ASSERT_EQ(2, matrix->row_ptr[1]);
    ASSERT_EQ(4, matrix->row_ptr[2]);
}

void test_admittance_matrix_three_bus_system() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create a 3-bus system with 2 branches (triangle topology)
    NetworkData network;
    network.num_buses = 3;
    network.num_branches = 2;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,    // 100 MW
                    .reactive_power = 50e6};  // 50 MVAr
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PV,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 150e6,  // 150 MW
                    .reactive_power = 0.0};
    network.buses = {bus1, bus2, bus3};

    BranchData branch1 = {.id = 1,
                          .from_bus = 1,
                          .from_status = 1,
                          .to_bus = 2,
                          .to_status = 1,
                          .status = 1,
                          .r1 = 0.01,    // resistance (ohm)
                          .x1 = 0.03,    // reactance (ohm)
                          .g1 = 0.0,     // conductance (siemens)
                          .b1 = 0.02,    // susceptance (siemens)
                          .k = 1.0,      // tap ratio
                          .theta = 0.0,  // phase shift
                          .sn = 100e6};  // 100 MVA rating
    BranchData branch2 = {.id = 2,
                          .from_bus = 2,
                          .from_status = 1,
                          .to_bus = 3,
                          .to_status = 1,
                          .status = 1,
                          .r1 = 0.02,    // resistance (ohm)
                          .x1 = 0.04,    // reactance (ohm)
                          .g1 = 0.0,     // conductance (siemens)
                          .b1 = 0.03,    // susceptance (siemens)
                          .k = 1.0,      // tap ratio
                          .theta = 0.0,  // phase shift
                          .sn = 100e6};  // 100 MVA rating
    network.branches = {branch1, branch2};

    auto matrix = admittance->build_admittance_matrix(network);

    // Verify matrix dimensions
    ASSERT_EQ(3, matrix->num_rows);
    ASSERT_EQ(3, matrix->num_cols);

    // Verify we have at least 3 diagonal elements
    ASSERT_TRUE(matrix->nnz >= 3);

    // Verify CSR structure is valid
    ASSERT_EQ(4, matrix->row_ptr.size());  // num_buses + 1
    ASSERT_EQ(0, matrix->row_ptr[0]);
    ASSERT_TRUE(matrix->row_ptr[3] == matrix->nnz);
}

void test_admittance_matrix_out_of_service_branch() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create a 2-bus system with an out-of-service branch
    NetworkData network;
    network.num_buses = 2;
    network.num_branches = 1;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,    // 100 MW
                    .reactive_power = 50e6};  // 50 MVAr
    network.buses = {bus1, bus2};

    // Out-of-service branch (status = false)
    BranchData branch = {.id = 1,
                         .from_bus = 1,
                         .from_status = 1,
                         .to_bus = 2,
                         .to_status = 1,
                         .status = 0,   // Out of service
                         .r1 = 0.02,    // resistance (ohm)
                         .x1 = 0.06,    // reactance (ohm)
                         .g1 = 0.0,     // conductance (siemens)
                         .b1 = 0.05,    // susceptance (siemens)
                         .k = 1.0,      // tap ratio
                         .theta = 0.0,  // phase shift
                         .sn = 100e6};  // 100 MVA rating
    network.branches = {branch};

    auto matrix = admittance->build_admittance_matrix(network);

    // With no in-service branches, we should only have diagonal elements (zeros)
    ASSERT_EQ(2, matrix->nnz);  // Only diagonal elements

    // Verify diagonal elements are zero (no connections)
    ASSERT_EQ(0, matrix->col_idx[0]);  // First diagonal element (bus 0)
    ASSERT_EQ(1, matrix->col_idx[1]);  // Second diagonal element (bus 1)

    // Values should be zero since no branches are in service
    ASSERT_TRUE(std::abs(matrix->values[0]) < 1e-10);  // Check magnitude is effectively zero
    ASSERT_TRUE(std::abs(matrix->values[1]) < 1e-10);  // Check magnitude is effectively zero
}

void test_admittance_matrix_update_branch_status() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create initial network with in-service branch
    NetworkData network;
    network.num_buses = 2;
    network.num_branches = 1;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,    // 100 MW
                    .reactive_power = 50e6};  // 50 MVAr
    network.buses = {bus1, bus2};

    BranchData branch = {.id = 1,
                         .from_bus = 1,
                         .from_status = 1,
                         .to_bus = 2,
                         .to_status = 1,
                         .status = 1,   // In service
                         .r1 = 0.02,    // resistance (ohm)
                         .x1 = 0.06,    // reactance (ohm)
                         .g1 = 0.0,     // conductance (siemens)
                         .b1 = 0.05,    // susceptance (siemens)
                         .k = 1.0,      // tap ratio
                         .theta = 0.0,  // phase shift
                         .sn = 100e6};  // 100 MVA rating
    network.branches = {branch};

    auto original_matrix = admittance->build_admittance_matrix(network);

    // Store original values for comparison (commented out to avoid unused variable warnings)
    // Complex original_diag_0 = original_matrix->values[0];
    // Complex original_diag_1 = original_matrix->values[1];
    // Complex original_off_diag = original_matrix->values[2];  // Assuming off-diagonal at index 2

    // Create branch change: take the branch out of service
    std::vector<BranchData> changes = {{.id = 1,
                                        .from_bus = 1,
                                        .from_status = 1,
                                        .to_bus = 2,
                                        .to_status = 1,
                                        .status = 0,    // Out of service
                                        .r1 = 0.02,     // resistance (ohm)
                                        .x1 = 0.06,     // reactance (ohm)
                                        .g1 = 0.0,      // conductance (siemens)
                                        .b1 = 0.05,     // susceptance (siemens)
                                        .k = 1.0,       // tap ratio
                                        .theta = 0.0,   // phase shift
                                        .sn = 100e6}};  // 100 MVA rating

    auto updated_matrix = admittance->update_admittance_matrix(*original_matrix, changes);

    ASSERT_TRUE(updated_matrix != nullptr);
    ASSERT_EQ(original_matrix->num_rows, updated_matrix->num_rows);
    ASSERT_EQ(original_matrix->num_cols, updated_matrix->num_cols);

    // The diagonal elements should be reduced (branch admittance removed)
    // Off-diagonal elements should also be modified
    ASSERT_TRUE(updated_matrix->values.size() == original_matrix->values.size());
}

void test_admittance_matrix_multiple_branches() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create a 3-bus system with multiple branches from one bus
    NetworkData network;
    network.num_buses = 3;
    network.num_branches = 3;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,    // 100 MW
                    .reactive_power = 50e6};  // 50 MVAr
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PV,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 150e6,  // 150 MW
                    .reactive_power = 0.0};
    network.buses = {bus1, bus2, bus3};

    // Create a star configuration: bus 1 connected to both bus 2 and bus 3
    BranchData branch1 = {.id = 1,
                          .from_bus = 1,
                          .from_status = 1,
                          .to_bus = 2,
                          .to_status = 1,
                          .status = 1,
                          .r1 = 0.01,    // resistance (ohm)
                          .x1 = 0.03,    // reactance (ohm)
                          .g1 = 0.0,     // conductance (siemens)
                          .b1 = 0.02,    // susceptance (siemens)
                          .k = 1.0,      // tap ratio
                          .theta = 0.0,  // phase shift
                          .sn = 100e6};  // 100 MVA rating
    BranchData branch2 = {.id = 2,
                          .from_bus = 1,
                          .from_status = 1,
                          .to_bus = 3,
                          .to_status = 1,
                          .status = 1,
                          .r1 = 0.015,   // resistance (ohm)
                          .x1 = 0.04,    // reactance (ohm)
                          .g1 = 0.0,     // conductance (siemens)
                          .b1 = 0.025,   // susceptance (siemens)
                          .k = 1.0,      // tap ratio
                          .theta = 0.0,  // phase shift
                          .sn = 100e6};  // 100 MVA rating
    BranchData branch3 = {.id = 3,
                          .from_bus = 2,
                          .from_status = 1,
                          .to_bus = 3,
                          .to_status = 1,
                          .status = 1,
                          .r1 = 0.02,    // resistance (ohm)
                          .x1 = 0.05,    // reactance (ohm)
                          .g1 = 0.0,     // conductance (siemens)
                          .b1 = 0.03,    // susceptance (siemens)
                          .k = 1.0,      // tap ratio
                          .theta = 0.0,  // phase shift
                          .sn = 100e6};  // 100 MVA rating
    network.branches = {branch1, branch2, branch3};

    auto matrix = admittance->build_admittance_matrix(network);

    // In a fully connected 3-bus system, we should have:
    // - 3 diagonal elements
    // - 6 off-diagonal elements (3 branches × 2 directions each)
    ASSERT_EQ(9, matrix->nnz);  // 3x3 fully populated matrix

    // Verify matrix structure
    ASSERT_EQ(3, matrix->num_rows);
    ASSERT_EQ(3, matrix->num_cols);

    // Each row should have 3 elements (1 diagonal + 2 off-diagonal for full connectivity)
    ASSERT_EQ(3, matrix->row_ptr[1] - matrix->row_ptr[0]);  // Row 0
    ASSERT_EQ(3, matrix->row_ptr[2] - matrix->row_ptr[1]);  // Row 1
    ASSERT_EQ(3, matrix->row_ptr[3] - matrix->row_ptr[2]);  // Row 2
}

void test_admittance_matrix_shunt_appliances() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create a 3-bus system with shunt appliances (capacitor banks)
    NetworkData network;
    network.num_buses = 3;
    network.num_branches = 1;

    BusData bus1 = {.id = 1,
                    .u_rated = 230000.0,
                    .bus_type = BusType::SLACK,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 0.0,
                    .reactive_power = 0.0};
    BusData bus2 = {.id = 2,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PQ,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 100e6,
                    .reactive_power = 50e6};
    BusData bus3 = {.id = 3,
                    .u_rated = 230000.0,
                    .bus_type = BusType::PV,
                    .energized = 1,
                    .u = 230000.0,
                    .u_pu = 1.0,
                    .u_angle = 0.0,
                    .active_power = 150e6,
                    .reactive_power = 0.0};
    network.buses = {bus1, bus2, bus3};

    // Single branch connecting bus 1 and 2
    BranchData branch = {.id = 1,
                         .from_bus = 1,
                         .from_status = 1,
                         .to_bus = 2,
                         .to_status = 1,
                         .status = 1,
                         .r1 = 0.01,  // resistance (ohm)
                         .x1 = 0.03,  // reactance (ohm)
                         .g1 = 0.0,   // conductance (siemens)
                         .b1 = 0.02,  // susceptance (siemens)
                         .k = 1.0,
                         .theta = 0.0,
                         .sn = 100e6};
    network.branches = {branch};

    // Add shunt appliances (capacitor banks and reactors)
    ApplianceData capacitor_bank = {.id = 10,
                                    .node = 2,  // Connected to bus 2
                                    .status = 1,
                                    .type = ApplianceType::SHUNT,
                                    .g1 = 0.001,  // Small conductance (losses)
                                    .b1 = 0.05};  // Capacitive susceptance (+ve)

    ApplianceData reactor = {.id = 11,
                             .node = 3,  // Connected to bus 3
                             .status = 1,
                             .type = ApplianceType::SHUNT,
                             .g1 = 0.0,
                             .b1 = -0.03};  // Inductive susceptance (-ve)

    // Multiple shunts on same bus
    ApplianceData second_capacitor = {.id = 12,
                                      .node = 2,  // Also connected to bus 2
                                      .status = 1,
                                      .type = ApplianceType::SHUNT,
                                      .g1 = 0.0005,
                                      .b1 = 0.02};

    network.appliances = {capacitor_bank, reactor, second_capacitor};

    // Build admittance matrix
    auto matrix = admittance->build_admittance_matrix(network);

    // Verify matrix structure
    ASSERT_EQ(3, matrix->num_rows);
    ASSERT_EQ(3, matrix->num_cols);

    // We should have at least diagonal elements and off-diagonal from the branch
    ASSERT_TRUE(matrix->nnz >= 3);

    // Calculate expected shunt contributions to verify correctness
    // Bus 1: No shunt appliances, only branch contribution
    // Bus 2: capacitor_bank (0.001 + j0.05) + second_capacitor (0.0005 + j0.02)
    //        = 0.0015 + j0.07 total shunt admittance
    // Bus 3: reactor (0.0 - j0.03) = -j0.03 total shunt admittance

    // Find diagonal elements in the sparse matrix
    Complex bus2_diagonal, bus3_diagonal;
    bool found_bus2 = false, found_bus3 = false;

    // Parse CSR format to find diagonal elements
    for (int row = 0; row < matrix->num_rows; ++row) {
        int start_idx = matrix->row_ptr[row];
        int end_idx = matrix->row_ptr[row + 1];

        for (int idx = start_idx; idx < end_idx; ++idx) {
            int col = matrix->col_idx[idx];
            if (row == col) {    // Diagonal element
                if (row == 1) {  // Bus 2 (0-indexed)
                    bus2_diagonal = matrix->values[idx];
                    found_bus2 = true;
                } else if (row == 2) {  // Bus 3 (0-indexed)
                    bus3_diagonal = matrix->values[idx];
                    found_bus3 = true;
                }
            }
        }
    }

    ASSERT_TRUE(found_bus2);
    ASSERT_TRUE(found_bus3);

    // Verify the correct CSR structure for topology: Bus1 <-> Bus2, Bus3_isolated
    // Expected: row_ptr=[0,2,4,5], col_idx=[0,1,0,1,2], nnz=5
    ASSERT_EQ(5, matrix->nnz);             // Should have exactly 5 non-zeros
    ASSERT_EQ(4, matrix->row_ptr.size());  // 3 buses + 1
    ASSERT_EQ(0, matrix->row_ptr[0]);      // First row starts at 0
    ASSERT_EQ(2, matrix->row_ptr[1]);  // Bus 0 has 2 elements (diagonal + off-diagonal to bus 1)
    ASSERT_EQ(4, matrix->row_ptr[2]);  // Bus 1 has 2 elements (diagonal + off-diagonal to bus 0)
    ASSERT_EQ(5, matrix->row_ptr[3]);  // Bus 2 has 1 element (only diagonal)

    // Verify column indices are properly sorted within each row
    std::vector<int> expected_cols = {0, 1, 0, 1, 2};
    ASSERT_EQ(expected_cols.size(), matrix->col_idx.size());
    for (size_t i = 0; i < expected_cols.size(); ++i) {
        ASSERT_EQ(expected_cols[i], matrix->col_idx[i]);
    }

    // Verify that shunt admittances are correctly included in diagonal elements
    // Expected calculation for Bus 2:
    // - Branch admittance Y ≈ 9.9 - j29.7 (from 1/(0.01+j0.03))
    // - Shunt contribution: 0.0015 + j0.07 (from capacitor banks)
    // - Total diagonal ≈ 10.0 - j29.6

    ASSERT_TRUE(bus2_diagonal.real() > 10.0);  // Should be ~10.0015 (includes shunt conductance)
    ASSERT_TRUE(bus2_diagonal.imag() >
                -30.0);  // Should be ~-29.62 (branch inductive + shunt capacitive)
    ASSERT_TRUE(bus2_diagonal.imag() < -29.5);  // Verify it's in expected range

    // Bus 3: Only reactor shunt, no branch connection
    ASSERT_NEAR(0.0, bus3_diagonal.real(), 1e-10);    // No resistance/conductance
    ASSERT_NEAR(-0.03, bus3_diagonal.imag(), 1e-10);  // Only reactor susceptance
}

void test_admittance_matrix_csr_format() {
    auto admittance = core::BackendFactory::create_admittance_backend(BackendType::CPU);

    // Create various network topologies to test CSR format correctness

    // Test 1: Isolated buses (diagonal-only matrix)
    {
        NetworkData network;
        network.num_buses = 3;
        network.num_branches = 0;  // No branches

        BusData bus1 = {.id = 1, .u_rated = 230000.0, .bus_type = BusType::SLACK, .energized = 1};
        BusData bus2 = {.id = 2, .u_rated = 230000.0, .bus_type = BusType::PQ, .energized = 1};
        BusData bus3 = {.id = 3, .u_rated = 230000.0, .bus_type = BusType::PV, .energized = 1};
        network.buses = {bus1, bus2, bus3};

        auto matrix = admittance->build_admittance_matrix(network);

        // Verify CSR structure for diagonal-only matrix
        ASSERT_EQ(3, matrix->nnz);             // Only 3 diagonal elements
        ASSERT_EQ(4, matrix->row_ptr.size());  // n_buses + 1
        ASSERT_EQ(0, matrix->row_ptr[0]);      // First row starts at 0
        ASSERT_EQ(1, matrix->row_ptr[1]);      // Bus 0: 1 element
        ASSERT_EQ(2, matrix->row_ptr[2]);      // Bus 1: 1 element
        ASSERT_EQ(3, matrix->row_ptr[3]);      // Bus 2: 1 element

        // Column indices should be [0, 1, 2] (diagonal elements)
        std::vector<int> expected_cols = {0, 1, 2};
        ASSERT_EQ(expected_cols.size(), matrix->col_idx.size());
        for (size_t i = 0; i < expected_cols.size(); i++) {
            ASSERT_EQ(expected_cols[i], matrix->col_idx[i]);
        }
    }

    // Test 2: Single branch (2x2 matrix)
    {
        NetworkData network;
        network.num_buses = 2;
        network.num_branches = 1;

        BusData bus1 = {.id = 1, .u_rated = 230000.0, .bus_type = BusType::SLACK, .energized = 1};
        BusData bus2 = {.id = 2, .u_rated = 230000.0, .bus_type = BusType::PQ, .energized = 1};
        network.buses = {bus1, bus2};

        BranchData branch = {.id = 1,
                             .from_bus = 1,
                             .from_status = 1,
                             .to_bus = 2,
                             .to_status = 1,
                             .status = 1,
                             .r1 = 0.01,
                             .x1 = 0.05,
                             .sn = 100e6};
        network.branches = {branch};

        auto matrix = admittance->build_admittance_matrix(network);

        // Verify CSR structure for full 2x2 matrix
        ASSERT_EQ(4, matrix->nnz);             // 2 diagonal + 2 off-diagonal
        ASSERT_EQ(3, matrix->row_ptr.size());  // n_buses + 1
        ASSERT_EQ(0, matrix->row_ptr[0]);      // Row 0 starts at index 0
        ASSERT_EQ(2, matrix->row_ptr[1]);      // Row 0 has 2 elements [0,1]
        ASSERT_EQ(4, matrix->row_ptr[2]);      // Row 1 has 2 elements [0,1]

        // Column indices should be [0,1,0,1] (both rows have elements in columns 0 and 1)
        std::vector<int> expected_cols = {0, 1, 0, 1};
        ASSERT_EQ(expected_cols.size(), matrix->col_idx.size());
        for (size_t i = 0; i < expected_cols.size(); i++) {
            ASSERT_EQ(expected_cols[i], matrix->col_idx[i]);
        }

        // Verify columns are sorted within each row
        for (int row = 0; row < matrix->num_rows; ++row) {
            int start = matrix->row_ptr[row];
            int end = matrix->row_ptr[row + 1];
            for (int i = start + 1; i < end; ++i) {
                ASSERT_TRUE(matrix->col_idx[i - 1] <= matrix->col_idx[i]);
            }
        }
    }

    // Test 3: Complex topology (fully connected 3-bus)
    {
        NetworkData network;
        network.num_buses = 3;
        network.num_branches = 3;

        BusData bus1 = {.id = 1, .u_rated = 230000.0, .bus_type = BusType::SLACK, .energized = 1};
        BusData bus2 = {.id = 2, .u_rated = 230000.0, .bus_type = BusType::PQ, .energized = 1};
        BusData bus3 = {.id = 3, .u_rated = 230000.0, .bus_type = BusType::PV, .energized = 1};
        network.buses = {bus1, bus2, bus3};

        // Full mesh: 1-2, 1-3, 2-3
        BranchData branch1 = {.id = 1,
                              .from_bus = 1,
                              .from_status = 1,
                              .to_bus = 2,
                              .to_status = 1,
                              .status = 1,
                              .r1 = 0.01,
                              .x1 = 0.05};
        BranchData branch2 = {.id = 2,
                              .from_bus = 1,
                              .from_status = 1,
                              .to_bus = 3,
                              .to_status = 1,
                              .status = 1,
                              .r1 = 0.02,
                              .x1 = 0.06};
        BranchData branch3 = {.id = 3,
                              .from_bus = 2,
                              .from_status = 1,
                              .to_bus = 3,
                              .to_status = 1,
                              .status = 1,
                              .r1 = 0.015,
                              .x1 = 0.055};
        network.branches = {branch1, branch2, branch3};

        auto matrix = admittance->build_admittance_matrix(network);

        // Verify CSR structure for fully connected 3x3 matrix
        ASSERT_EQ(9, matrix->nnz);             // Full 3x3 matrix
        ASSERT_EQ(4, matrix->row_ptr.size());  // n_buses + 1
        ASSERT_EQ(0, matrix->row_ptr[0]);      // Row 0 starts at 0
        ASSERT_EQ(3, matrix->row_ptr[1]);      // Row 0: 3 elements [0,1,2]
        ASSERT_EQ(6, matrix->row_ptr[2]);      // Row 1: 3 elements [0,1,2]
        ASSERT_EQ(9, matrix->row_ptr[3]);      // Row 2: 3 elements [0,1,2]

        // Column indices should be [0,1,2,0,1,2,0,1,2] (all rows fully populated)
        std::vector<int> expected_cols = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        ASSERT_EQ(expected_cols.size(), matrix->col_idx.size());
        for (size_t i = 0; i < expected_cols.size(); i++) {
            ASSERT_EQ(expected_cols[i], matrix->col_idx[i]);
        }

        // Verify each row has all columns 0,1,2 in sorted order
        for (int row = 0; row < 3; ++row) {
            int start = matrix->row_ptr[row];
            ASSERT_EQ(0, matrix->col_idx[start]);      // Column 0
            ASSERT_EQ(1, matrix->col_idx[start + 1]);  // Column 1
            ASSERT_EQ(2, matrix->col_idx[start + 2]);  // Column 2
        }
    }

    // Test 4: Partial connectivity (mixed pattern)
    {
        NetworkData network;
        network.num_buses = 4;
        network.num_branches = 2;

        BusData bus1 = {.id = 1, .u_rated = 230000.0, .bus_type = BusType::SLACK, .energized = 1};
        BusData bus2 = {.id = 2, .u_rated = 230000.0, .bus_type = BusType::PQ, .energized = 1};
        BusData bus3 = {.id = 3, .u_rated = 230000.0, .bus_type = BusType::PQ, .energized = 1};
        BusData bus4 = {.id = 4, .u_rated = 230000.0, .bus_type = BusType::PV, .energized = 1};
        network.buses = {bus1, bus2, bus3, bus4};

        // Star pattern: 1-2, 1-3 (bus 4 isolated)
        BranchData branch1 = {.id = 1,
                              .from_bus = 1,
                              .from_status = 1,
                              .to_bus = 2,
                              .to_status = 1,
                              .status = 1,
                              .r1 = 0.01,
                              .x1 = 0.05};
        BranchData branch2 = {.id = 2,
                              .from_bus = 1,
                              .from_status = 1,
                              .to_bus = 3,
                              .to_status = 1,
                              .status = 1,
                              .r1 = 0.02,
                              .x1 = 0.06};
        network.branches = {branch1, branch2};

        auto matrix = admittance->build_admittance_matrix(network);

        // Expected pattern:
        // Row 0: [0,1,2]     - Bus 1 connected to buses 2,3 + self (3 elements)
        // Row 1: [0,1]       - Bus 2 connected to bus 1 + self (2 elements)
        // Row 2: [0,2]       - Bus 3 connected to bus 1 + self (2 elements)
        // Row 3: [3]         - Bus 4 isolated (only self) (1 element)

        ASSERT_EQ(8, matrix->nnz);             // 3+2+2+1 = 8
        ASSERT_EQ(5, matrix->row_ptr.size());  // n_buses + 1
        ASSERT_EQ(0, matrix->row_ptr[0]);      // Row 0 starts at 0

        // Verify column indices are properly sorted within each row
        for (int row = 0; row < matrix->num_rows; ++row) {
            int start = matrix->row_ptr[row];
            int end = matrix->row_ptr[row + 1];
            for (int i = start + 1; i < end; ++i) {
                ASSERT_TRUE(matrix->col_idx[i - 1] < matrix->col_idx[i]);  // Strictly increasing
            }
        }
    }
}

void test_gpu_admittance_availability() {
    bool gpu_available = core::BackendFactory::is_backend_available(BackendType::GPU_CUDA);

    if (gpu_available) {
        auto admittance = core::BackendFactory::create_admittance_backend(BackendType::GPU_CUDA);
        ASSERT_TRUE(admittance != nullptr);
        ASSERT_BACKEND_EQ(BackendType::GPU_CUDA, admittance->get_backend_type());
    }

    // Test should pass regardless of GPU availability
    ASSERT_TRUE(true);
}

void register_admittance_tests(TestRunner& runner) {
    runner.add_test("CPU Admittance Creation", test_cpu_admittance_creation);
    runner.add_test("Admittance Matrix Build", test_admittance_matrix_build);
    runner.add_test("Admittance Matrix Update", test_admittance_matrix_update);
    runner.add_test("Admittance Matrix Branch Iteration", test_admittance_matrix_branch_iteration);
    runner.add_test("Admittance Matrix Three Bus System", test_admittance_matrix_three_bus_system);
    runner.add_test("Admittance Matrix Out-of-Service Branch",
                    test_admittance_matrix_out_of_service_branch);
    runner.add_test("Admittance Matrix Update Branch Status",
                    test_admittance_matrix_update_branch_status);
    runner.add_test("Admittance Matrix Multiple Branches",
                    test_admittance_matrix_multiple_branches);
    runner.add_test("Admittance Matrix Shunt Appliances", test_admittance_matrix_shunt_appliances);
    runner.add_test("Admittance Matrix CSR Format", test_admittance_matrix_csr_format);
    runner.add_test("GPU Admittance Availability", test_gpu_admittance_availability);
}
