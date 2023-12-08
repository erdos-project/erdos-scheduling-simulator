#ifndef _TETRISCHED_TYPES_HPP_
#define _TETRISCHED_TYPES_HPP_

#include <cstdint>
#include <cstring>
#include <exception>
#include <experimental/source_location>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

// Macros for logging.
#define TETRISCHED_LOGGING_ENABLED false
#define TETRISCHED_LOGGING_DIR_ENV_NAME "TETRISCHED_LOGGING_DIR"
#define TETRISCHED_LOG_FILE_NAME "libtetrisched.log"
#define TETRISCHED_DEFAULT_LOG_LEVEL tetrisched::logging::INFO
#define TETRISCHED_DEBUG(x)                                                   \
  if (TETRISCHED_LOGGING_ENABLED &&                                           \
      TETRISCHED_DEFAULT_LOG_LEVEL >= tetrisched::logging::DEBUG) {           \
    auto sourceLocation = std::experimental::source_location::current();      \
    tetrisched::logging::Logger::debug()                                      \
        << "[DEBUG, " << sourceLocation.function_name() << "] " << x << "\n"; \
  }
#define TETRISCHED_INFO(x)                                                   \
  if (TETRISCHED_LOGGING_ENABLED &&                                          \
      TETRISCHED_DEFAULT_LOG_LEVEL >= tetrisched::logging::INFO) {           \
    auto sourceLocation = std::experimental::source_location::current();     \
    tetrisched::logging::Logger::info()                                      \
        << "[INFO, " << sourceLocation.function_name() << "] " << x << "\n"; \
  }

// Macros for timing.
// Uncomment the following line to enable timing.
#define TETRISCHED_TIMING_ENABLED
#define TETRISCHED_TIMING_FILE_NAME "libtetrisched_performance.csv"
#ifdef TETRISCHED_TIMING_ENABLED
#define TETRISCHED_SCOPE_TIMER(TIMER_NAME) \
  tetrisched::timing::ScopeTimer timer##__LINE__(TIMER_NAME);
#else
#define TETRISCHED_SCOPE_TIMER(TIMER_NAME)
#endif

// Macro for the coefficient and the permissible values for the Variables.
// (Sukrit): It is unknown if the ILP will perform better if the coefficients
// and variables are int32_t or double. This is something that we should
// experiment with. Note that both CPLEX and Gurobi do not like 32-bit floating
// points (due to documented numerical difficulties) so the only permissible
// values for this macro is supposed to be int32_t or double.
#define TETRISCHED_ILP_TYPE double

namespace tetrisched {
/// Defines the exceptions that the methods can throw.
namespace exceptions {

/// An exception that is thrown when the construction of an Expression fails.
class ExpressionConstructionException : public std::exception {
 private:
  std::string message;

 public:
  ExpressionConstructionException(std::string message) : message(message) {}
  const char* what() const noexcept override { return message.c_str(); }
};

/// An exception that is thrown when solving of an Expression fails.
class ExpressionSolutionException : public std::exception {
 private:
  std::string message;

 public:
  ExpressionSolutionException(std::string message) : message(message) {}
  const char* what() const noexcept override { return message.c_str(); }
};

/// An exception that is thrown when the Solver is used incorrectly.
class SolverException : public std::exception {
 private:
  std::string message;

 public:
  SolverException(std::string message) : message(message) {}
  const char* what() const noexcept override { return message.c_str(); }
};

/// A general Runtime exception.
class RuntimeException : public std::exception {
 private:
  std::string message;

 public:
  RuntimeException(std::string message) : message(message) {}
  const char* what() const noexcept override { return message.c_str(); }
};
}  // namespace exceptions

/// Defines the namespace for logging.
namespace logging {
enum LogLevel {
  DEBUG,
  INFO,
  WARN,
  ERROR,
};

class Logger {
 private:
  std::ostream& outputStream;
  LogLevel logLevel;

 public:
  Logger(std::ostream& outputStream = std::cout, LogLevel level = INFO)
      : logLevel(level), outputStream(outputStream) {
    outputStream << std::unitbuf;
  }

  template <typename T>
  Logger& operator<<(const T& val) {
    if (logLevel >= TETRISCHED_DEFAULT_LOG_LEVEL) {
      outputStream << val;
    }
    return *this;
  }

  ~Logger() { outputStream << std::endl << std::nounitbuf; }

  void flush() { outputStream.flush(); }

  static Logger& debug() {
    static std::filesystem::path outputDir =
        std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
            ? std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
            : "./";
    static std::filesystem::path outputFile =
        outputDir / TETRISCHED_LOG_FILE_NAME;
    static std::ofstream outputStream(outputFile,
                                      std::ios::app | std::ios::out);
    static Logger logger(outputStream.is_open() ? outputStream : std::cout,
                         LogLevel::DEBUG);
    return logger;
  }

  static Logger& info() {
    static std::filesystem::path outputDir =
        std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
            ? std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
            : "./";
    static std::filesystem::path outputFile =
        outputDir / TETRISCHED_LOG_FILE_NAME;
    static std::ofstream outputStream(outputFile,
                                      std::ios::app | std::ios::out);
    static Logger logger(outputStream.is_open() ? outputStream : std::cout,
                         LogLevel::INFO);
    return logger;
  }
};
}  // namespace logging

namespace timing {
/// A class to measure the time taken for a block of code to execute.
class ScopeTimer {
 private:
  std::string scopeTimerName;
  std::chrono::high_resolution_clock::time_point startTime;
  static std::ofstream& getOutputFileStream() {
    static std::filesystem::path outputDir =
        std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
            ? std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
            : "./";
    static std::filesystem::path outputFile =
        outputDir / TETRISCHED_TIMING_FILE_NAME;
    static std::ofstream file(outputFile, std::ios::app | std::ios::out);
    return file;
  }

 public:
  ScopeTimer(std::string scopeTimerName) : scopeTimerName(scopeTimerName) {
    startTime = std::chrono::high_resolution_clock::now();
    getOutputFileStream()
        << "BEGIN," << scopeTimerName << ","
        << std::chrono::duration_cast<std::chrono::microseconds>(
               startTime.time_since_epoch())
               .count()
        << std::endl;
  }

  ~ScopeTimer() {
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        endTime - startTime)
                        .count();

    auto startTimeMicroseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(
            startTime.time_since_epoch())
            .count();
    auto endtimeMicroseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(
            endTime.time_since_epoch())
            .count();
    getOutputFileStream() << "END," << scopeTimerName << ","
                          << startTimeMicroseconds << "," << endtimeMicroseconds
                          << "," << duration << std::endl;
  }
};
}  // namespace timing

/// A Representation of time in the system.
/// We currently use a uint32_t since it translates well from the simulator.
/// When this library is deployed for real use, this might need to change to a
/// double.
using Time = uint32_t;

/// A representation of a range of time.
/// This is intended to represent a start and finish time for either the
/// start or the finish times from each Expression.
using TimeRange = std::pair<Time, Time>;

/// General forward declarations.
class Expression;
using ExpressionPtr = std::shared_ptr<Expression>;

/// Forward declarations for Solver instantiations so that we can declare
/// them as friend classes in the model.
class CPLEXSolver;
class GurobiSolver;
class GoogleCPSolver;
}  // namespace tetrisched

#endif
