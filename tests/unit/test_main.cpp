#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

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

#define ASSERT_EQ(expected, actual)                                                             \
    do {                                                                                        \
        if ((expected) != (actual)) {                                                           \
            throw std::runtime_error("Assertion failed: expected " + std::to_string(expected) + \
                                     " but got " + std::to_string(actual));                     \
        }                                                                                       \
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
