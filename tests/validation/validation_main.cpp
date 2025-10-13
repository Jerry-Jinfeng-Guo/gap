#include "../unit/test_framework.h"

// Forward declarations of test registration functions
void register_validation_tests(TestRunner& runner);

int main() {
    std::cout << "GAP Power Flow Calculator - Validation Tests\n" << std::endl;
    
    TestRunner runner;
    
    // Register validation test suites
    register_validation_tests(runner);
    
    // Run all tests
    runner.run_all();
    
    return runner.get_failed_count();
}