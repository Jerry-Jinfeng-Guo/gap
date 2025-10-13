#include "../unit/test_main.cpp"
#include "gap/io/io_interface.h"
#include "gap/core/backend_factory.h"
#include <fstream>
#include <sstream>

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
    
    ComplexVector voltages = {
        Complex(1.05, 0.0),
        Complex(1.02, -0.1),
        Complex(1.01, -0.2)
    };
    
    std::string output_file = "test_results.json";
    io_module->write_results(output_file, voltages, true, 5);
    
    // Check if file was created
    std::ifstream file(output_file);
    ASSERT_TRUE(file.is_open());
    file.close();
    
    // Clean up
    std::remove(output_file.c_str());
}

int main() {
    TestRunner runner;
    
    runner.add_test("IO Module Creation", test_io_module_creation);
    runner.add_test("JSON Validation", test_json_validation);
    runner.add_test("Network Data Reading", test_network_data_reading);
    runner.add_test("Results Writing", test_results_writing);
    
    runner.run_all();
    
    return runner.get_failed_count();
}