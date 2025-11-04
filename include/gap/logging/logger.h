#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace gap {
namespace logging {

/**
 * @brief Log levels for different types of messages
 */
enum class LogLevel {
    TRACE = 0,  // Detailed execution flow
    DEBUG = 1,  // Debug information
    INFO = 2,   // General information
    WARN = 3,   // Warning messages
    ERROR = 4,  // Error messages
    OFF = 5     // Disable logging
};

/**
 * @brief Log output destinations
 */
enum class LogOutput {
    CONSOLE,  // Standard output (std::cout)
    FILE,     // File output
    BUFFER,   // In-memory buffer (for performance)
    NONE      // No output (maximum performance)
};

/**
 * @brief High-performance logger with configurable output and minimal overhead
 *
 * Design principles:
 * - Zero-cost when disabled (compile-time optimization)
 * - Minimal overhead when enabled (buffered output)
 * - Thread-safe
 * - Configurable log levels and outputs
 */
class Logger {
  private:
    LogLevel current_level_;
    LogOutput output_type_;
    std::unique_ptr<std::ofstream> file_stream_;
    std::ostringstream buffer_;
    std::mutex mutex_;
    std::string component_name_;

    static Logger* instance_;
    static std::mutex instance_mutex_;

  public:
    /**
     * @brief Get the singleton logger instance
     */
    static Logger& getInstance();

    /**
     * @brief Configure the logger
     */
    void configure(LogLevel level, LogOutput output, std::string const& filename = "");

    /**
     * @brief Set component name for context
     */
    void setComponent(std::string const& name) { component_name_ = name; }

    /**
     * @brief Check if a log level is enabled (for zero-cost optimization)
     */
    constexpr bool isEnabled(LogLevel level) const noexcept {
        return level >= current_level_ && output_type_ != LogOutput::NONE;
    }

    /**
     * @brief Log a message with specific level (inline template to avoid linking issues)
     */
    template <typename... Args>
    inline void log(LogLevel level, std::string const& format, Args&&... args) {
        if (!isEnabled(level)) return;

        std::ostringstream oss;
        oss << format;

        // Simple approach: just append arguments separated by spaces
        if constexpr (sizeof...(args) > 0) {
            oss << " ";
            ((oss << args << " "), ...);
        }

        writeMessage(level, oss.str());
    }

    /**
     * @brief Simple log functions for common cases
     */
    void logInfo(std::string const& message);
    void logDebug(std::string const& message);
    void logError(std::string const& message);

    /**
     * @brief Flush any buffered output
     */
    void flush();

    /**
     * @brief Get current buffer contents (for BUFFER output)
     */
    std::string getBuffer() const noexcept;

    /**
     * @brief Clear buffer contents
     */
    void clearBuffer();

  private:
    Logger() noexcept : current_level_(LogLevel::INFO), output_type_(LogOutput::CONSOLE) {}

    void writeMessage(LogLevel level, std::string const& message);
    std::string getCurrentTime() const noexcept;
    std::string levelToString(LogLevel level) const noexcept;
};

// Convenience macros for zero-cost logging
#define LOG_TRACE(logger, ...)                                        \
    do {                                                              \
        if (logger.isEnabled(::gap::logging::LogLevel::TRACE))        \
            logger.log(::gap::logging::LogLevel::TRACE, __VA_ARGS__); \
    } while (0)

#define LOG_DEBUG(logger, ...)                                        \
    do {                                                              \
        if (logger.isEnabled(::gap::logging::LogLevel::DEBUG))        \
            logger.log(::gap::logging::LogLevel::DEBUG, __VA_ARGS__); \
    } while (0)

#define LOG_INFO(logger, ...)                                        \
    do {                                                             \
        if (logger.isEnabled(::gap::logging::LogLevel::INFO))        \
            logger.log(::gap::logging::LogLevel::INFO, __VA_ARGS__); \
    } while (0)

#define LOG_WARN(logger, ...)                                        \
    do {                                                             \
        if (logger.isEnabled(::gap::logging::LogLevel::WARN))        \
            logger.log(::gap::logging::LogLevel::WARN, __VA_ARGS__); \
    } while (0)

#define LOG_ERROR(logger, ...)                                        \
    do {                                                              \
        if (logger.isEnabled(::gap::logging::LogLevel::ERROR))        \
            logger.log(::gap::logging::LogLevel::ERROR, __VA_ARGS__); \
    } while (0)

// Global logger instance for convenience
extern Logger& global_logger;

}  // namespace logging
}  // namespace gap