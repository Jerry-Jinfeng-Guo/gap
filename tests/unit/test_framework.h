#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Include gap types for backend_type_to_string function
#include "gap/core/types.h"

/**
 * @brief Simple test framework
 */
class TestRunner {
  private:
    struct Test {
        std::string name;
        std::function<void()> test_func;
    };

    std::vector<Test> tests_;
    int passed_ = 0;
    int failed_ = 0;

  public:
    void add_test(std::string const& name, std::function<void()> test_func) {
        tests_.push_back({name, test_func});
    }

    void run_all() {
        std::cout << "Running " << tests_.size() << " tests...\n" << std::endl;

        for (auto const& test : tests_) {
            try {
                std::cout << "Running " << test.name << "... ";
                test.test_func();
                std::cout << "PASSED" << std::endl;
                passed_++;
            } catch (std::exception const& e) {
                std::cout << "FAILED: " << e.what() << std::endl;
                failed_++;
            } catch (...) {
                std::cout << "FAILED: Unknown exception" << std::endl;
                failed_++;
            }
        }

        std::cout << "\nTest Results:" << std::endl;
        std::cout << "  Passed: " << passed_ << std::endl;
        std::cout << "  Failed: " << failed_ << std::endl;
        std::cout << "  Total:  " << tests_.size() << std::endl;
    }

    constexpr int get_failed_count() const noexcept { return failed_; }
};

// Global test state for simple test functions
static int g_test_passed = 0;
static int g_test_failed = 0;

// Simple test runner function
inline void run_test(std::string const& name, std::function<void()> test_func) {
    try {
        std::cout << "Running " << name << "... ";
        test_func();
        std::cout << "PASSED" << std::endl;
        g_test_passed++;
    } catch (std::exception const& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
        g_test_failed++;
    } catch (...) {
        std::cout << "FAILED: Unknown exception" << std::endl;
        g_test_failed++;
    }
}

inline void print_test_summary() {
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "  Passed: " << g_test_passed << std::endl;
    std::cout << "  Failed: " << g_test_failed << std::endl;
    std::cout << "  Total:  " << (g_test_passed + g_test_failed) << std::endl;
}

inline int get_failed_count() { return g_test_failed; }

// Test assertion macros
#define ASSERT_TRUE(condition)                                         \
    do {                                                               \
        if (!(condition)) {                                            \
            throw std::runtime_error("Assertion failed: " #condition); \
        }                                                              \
    } while (0)

#define ASSERT_FALSE(condition)                                                           \
    do {                                                                                  \
        if (condition) {                                                                  \
            throw std::runtime_error("Assertion failed: " #condition " should be false"); \
        }                                                                                 \
    } while (0)

// Helper functions for converting values to strings for assertions
template <typename T>
inline std::string assert_to_string(T const& val) {
    return std::to_string(val);
}

inline std::string assert_to_string(std::string const& val) { return "\"" + val + "\""; }

inline std::string assert_to_string(char const* val) { return std::string("\"") + val + "\""; }

inline std::string assert_to_string(gap::BackendType const& val) {
    return gap::backend_type_to_string(val);
}

#define ASSERT_EQ(expected, actual)                                                               \
    do {                                                                                          \
        if ((expected) != (actual)) {                                                             \
            throw std::runtime_error("Assertion failed: expected " + assert_to_string(expected) + \
                                     " but got " + assert_to_string(actual));                     \
        }                                                                                         \
    } while (0)

#define ASSERT_BACKEND_EQ(expected, actual)                                                \
    do {                                                                                   \
        if ((expected) != (actual)) {                                                      \
            throw std::runtime_error("Assertion failed: expected " +                       \
                                     gap::backend_type_to_string(expected) + " but got " + \
                                     gap::backend_type_to_string(actual));                 \
        }                                                                                  \
    } while (0)

#define ASSERT_NEAR(expected, actual, tolerance)                                                \
    do {                                                                                        \
        if (std::abs((expected) - (actual)) > (tolerance)) {                                    \
            throw std::runtime_error("Assertion failed: expected " + std::to_string(expected) + \
                                     " but got " + std::to_string(actual) +                     \
                                     " (tolerance: " + std::to_string(tolerance) + ")");        \
        }                                                                                       \
    } while (0)
