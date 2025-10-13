#include "test_framework.h"

// Forward declarations of test registration functions
void register_io_tests(TestRunner& runner);
void register_admittance_tests(TestRunner& runner);
void register_lu_solver_tests(TestRunner& runner);
void register_powerflow_tests(TestRunner& runner);

int main() {
    TestRunner runner;
    
    // Register all test suites
    register_io_tests(runner);
    register_admittance_tests(runner);
    register_lu_solver_tests(runner);
    register_powerflow_tests(runner);
    
    // Run all tests
    runner.run_all();
    
    return runner.get_failed_count();
}