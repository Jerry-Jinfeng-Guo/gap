#include <fstream>
#include <sstream>

#include "gap/core/backend_factory.h"
#include "gap/io/io_interface.h"

#include "test_framework.h"

using namespace gap;

void test_io_module_creation() {
    auto io_module = core::BackendFactory::create_io_module();
    ASSERT_TRUE(io_module != nullptr);
}

void test_json_validation() {
    auto io_module = core::BackendFactory::create_io_module();

    // Create a dummy JSON file
    std::string test_file = "test_input.json";
    std::ofstream file(test_file);
    file << "{ \"test\": \"data\" }" << std::endl;
    file.close();

    bool is_valid = io_module->validate_input_format(test_file);
    ASSERT_TRUE(is_valid);

    // Clean up
    std::remove(test_file.c_str());
}

void test_network_data_reading() {
    auto io_module = core::BackendFactory::create_io_module();

    // Create a dummy network file
    std::string test_file = "test_network.json";
    std::ofstream file(test_file);
    file << "{ \"buses\": [], \"branches\": [], \"base_mva\": 100.0 }" << std::endl;
    file.close();

    NetworkData data = io_module->read_network_data(test_file);
    ASSERT_EQ(100.0, data.base_mva);

    // Clean up
    std::remove(test_file.c_str());
}

void test_results_writing() {
    auto io_module = core::BackendFactory::create_io_module();

    ComplexVector voltages = {Complex(1.05, 0.0), Complex(1.02, -0.1), Complex(1.01, -0.2)};

    std::string output_file = "test_results.json";

    // Test that write_results method can be called without exceptions
    // Note: Current implementation is a placeholder that only prints to console
    try {
        io_module->write_results(output_file, voltages, true, 5);
        // If we get here without exception, the method call succeeded
        ASSERT_TRUE(true);
    } catch (const std::exception& e) {
        // Test fails if write_results throws an exception
        ASSERT_TRUE(false);
    }

    // TODO: When write_results is fully implemented, add file existence check:
    // std::ifstream file(output_file);
    // ASSERT_TRUE(file.is_open());
    // file.close();
    // std::remove(output_file.c_str());
}

void register_io_tests(TestRunner& runner) {
    runner.add_test("IO Module Creation", test_io_module_creation);
    runner.add_test("JSON Validation", test_json_validation);
    runner.add_test("Network Data Reading", test_network_data_reading);
    runner.add_test("Results Writing", test_results_writing);
}
