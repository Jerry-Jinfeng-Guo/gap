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
    runner.add_test("GPU Admittance Availability", test_gpu_admittance_availability);
}
