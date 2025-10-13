#include "test_framework.h"
#include "gap/admittance/admittance_interface.h"
#include "gap/core/backend_factory.h"

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
    network.base_mva = 100.0;
    
    BusData bus1 = {1, 1.0, 0.0, 0.0, 0.0, 2}; // Slack bus
    BusData bus2 = {2, 1.0, 0.0, 100.0, 50.0, 0}; // PQ bus
    BusData bus3 = {3, 1.0, 0.0, 150.0, 0.0, 1}; // PV bus
    
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
    std::vector<BranchData> changes = {
        {1, 2, Complex(0.01, 0.1), Complex(0.0, 0.0), 0.0, false}
    };
    
    auto updated_matrix = admittance->update_admittance_matrix(*matrix, changes);
    ASSERT_TRUE(updated_matrix != nullptr);
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
    runner.add_test("GPU Admittance Availability", test_gpu_admittance_availability);
}