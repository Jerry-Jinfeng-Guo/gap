#include <cstdlib>
#include <iostream>

#include "test_framework.h"

// Forward declarations of test registration functions
void register_io_tests(TestRunner& runner);
void register_admittance_tests(TestRunner& runner);
void register_lu_solver_tests(TestRunner& runner);
void register_powerflow_tests(TestRunner& runner);
void register_iterative_current_tests(TestRunner& runner);
void register_newton_raphson_function_tests(TestRunner& runner);
void register_pgm_tests(TestRunner& runner);
void register_batch_tests(TestRunner& runner);

int main() {
    TestRunner runner;

    // Register all test suites
    register_io_tests(runner);
    register_admittance_tests(runner);
    register_lu_solver_tests(runner);
    register_powerflow_tests(runner);
    register_iterative_current_tests(runner);
    register_newton_raphson_function_tests(runner);
    register_pgm_tests(runner);
    register_batch_tests(runner);

    // Run all tests
    runner.run_all();

    int failed_count = runner.get_failed_count();

    // Use _Exit instead of return to skip global destructors
    // This prevents segfault from CUDA cleanup during static destructor phase
    // Note: This means we skip proper cleanup, but for a test executable that's acceptable
    std::_Exit(failed_count);
}
