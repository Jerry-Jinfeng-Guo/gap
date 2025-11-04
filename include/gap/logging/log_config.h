#pragma once

#include <string>

#include "gap/logging/logger.h"

namespace gap {
namespace logging {

/**
 * @brief Configuration utility for GAP logging system
 *
 * Provides easy configuration for different use cases:
 * - Development: Detailed logging to console
 * - Testing: Buffered logging for analysis
 * - Production: Minimal logging for performance
 * - Benchmarking: No logging for maximum performance
 */
class LogConfig {
  public:
    /**
     * @brief Configure logging for development (detailed console output)
     */
    static void forDevelopment() {
        auto& logger = Logger::getInstance();
        logger.configure(LogLevel::DEBUG, LogOutput::CONSOLE);
    }

    /**
     * @brief Configure logging for testing (buffered for analysis)
     */
    static void forTesting() {
        auto& logger = Logger::getInstance();
        logger.configure(LogLevel::INFO, LogOutput::BUFFER);
    }

    /**
     * @brief Configure logging for production (minimal file logging)
     */
    static void forProduction(const std::string& logfile = "gap.log") {
        auto& logger = Logger::getInstance();
        logger.configure(LogLevel::WARN, LogOutput::FILE, logfile);
    }

    /**
     * @brief Configure logging for benchmarking (no logging overhead)
     */
    static void forBenchmarking() {
        auto& logger = Logger::getInstance();
        logger.configure(LogLevel::OFF, LogOutput::NONE);
    }

    /**
     * @brief Configure logging from environment variable GAP_LOG_LEVEL
     */
    static void fromEnvironment() {
        const char* level_env = std::getenv("GAP_LOG_LEVEL");
        const char* output_env = std::getenv("GAP_LOG_OUTPUT");
        const char* file_env = std::getenv("GAP_LOG_FILE");

        LogLevel level = LogLevel::INFO;  // default
        if (level_env) {
            std::string level_str(level_env);
            if (level_str == "TRACE")
                level = LogLevel::TRACE;
            else if (level_str == "DEBUG")
                level = LogLevel::DEBUG;
            else if (level_str == "INFO")
                level = LogLevel::INFO;
            else if (level_str == "WARN")
                level = LogLevel::WARN;
            else if (level_str == "ERROR")
                level = LogLevel::ERROR;
            else if (level_str == "OFF")
                level = LogLevel::OFF;
        }

        LogOutput output = LogOutput::CONSOLE;  // default
        if (output_env) {
            std::string output_str(output_env);
            if (output_str == "CONSOLE")
                output = LogOutput::CONSOLE;
            else if (output_str == "FILE")
                output = LogOutput::FILE;
            else if (output_str == "BUFFER")
                output = LogOutput::BUFFER;
            else if (output_str == "NONE")
                output = LogOutput::NONE;
        }

        std::string filename = file_env ? file_env : "gap.log";

        auto& logger = Logger::getInstance();
        logger.configure(level, output, filename);
    }

    /**
     * @brief Enable performance mode (minimal overhead)
     * This is the recommended setting for production LU solver usage
     */
    static void enablePerformanceMode() {
        auto& logger = Logger::getInstance();
        // Only log errors, and buffer them to avoid I/O during computation
        logger.configure(LogLevel::ERROR, LogOutput::BUFFER);
    }

    /**
     * @brief Enable debug mode (maximum information)
     */
    static void enableDebugMode() {
        auto& logger = Logger::getInstance();
        logger.configure(LogLevel::TRACE, LogOutput::CONSOLE);
    }
};

}  // namespace logging
}  // namespace gap