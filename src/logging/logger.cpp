#include "gap/logging/logger.h"

#include <cstdio>
#include <iomanip>
#include <iostream>

namespace gap {
namespace logging {

// Static member definitions
Logger* Logger::instance_ = nullptr;
std::mutex Logger::instance_mutex_;

// Global logger instance
Logger& global_logger = Logger::getInstance();

Logger& Logger::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (instance_ == nullptr) {
        instance_ = new Logger();
    }
    return *instance_;
}

void Logger::configure(LogLevel level, LogOutput output, std::string const& filename) {
    std::lock_guard<std::mutex> lock(mutex_);

    current_level_ = level;
    output_type_ = output;

    if (output == LogOutput::FILE && !filename.empty()) {
        file_stream_ = std::make_unique<std::ofstream>(filename, std::ios::app);
        if (!file_stream_->is_open()) {
            // Fallback to console if file cannot be opened
            output_type_ = LogOutput::CONSOLE;
            std::cerr << "Warning: Cannot open log file '" << filename
                      << "', falling back to console output" << std::endl;
        }
    }
}

void Logger::writeMessage(LogLevel level, std::string const& message) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream formatted;
    formatted << getCurrentTime() << " [" << levelToString(level) << "]";

    if (!component_name_.empty()) {
        formatted << " [" << component_name_ << "]";
    }

    formatted << " " << message;

    switch (output_type_) {
        case LogOutput::CONSOLE:
            std::cout << formatted.str() << std::endl;
            break;

        case LogOutput::FILE:
            if (file_stream_ && file_stream_->is_open()) {
                *file_stream_ << formatted.str() << std::endl;
                file_stream_->flush();  // Ensure immediate write for critical info
            }
            break;

        case LogOutput::BUFFER:
            buffer_ << formatted.str() << "\n";
            break;

        case LogOutput::NONE:
            // No output - maximum performance
            break;
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (output_type_ == LogOutput::FILE && file_stream_) {
        file_stream_->flush();
    } else if (output_type_ == LogOutput::CONSOLE) {
        std::cout.flush();
    }
}

std::string Logger::getBuffer() const noexcept {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
    return buffer_.str();
}

void Logger::clearBuffer() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.str("");
    buffer_.clear();
}

std::string Logger::getCurrentTime() const noexcept {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}

std::string Logger::levelToString(LogLevel level) const noexcept {
    switch (level) {
        case LogLevel::TRACE:
            return "TRACE";
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARN:
            return "WARN";
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::OFF:
            return "OFF";
        default:
            return "UNKNOWN";
    }
}

void Logger::logInfo(std::string const& message) { log(LogLevel::INFO, message); }

void Logger::logDebug(std::string const& message) { log(LogLevel::DEBUG, message); }

void Logger::logError(std::string const& message) { log(LogLevel::ERROR, message); }

}  // namespace logging
}  // namespace gap