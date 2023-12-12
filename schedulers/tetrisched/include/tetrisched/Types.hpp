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
#include <mutex>
#include <string>

// Macros for logging.
#define TETRISCHED_LOGGING_ENABLED false
#define TETRISCHED_LOGGING_DIR_ENV_NAME "TETRISCHED_LOGGING_DIR"
#define TETRISCHED_LOG_FILE_NAME "libtetrisched.log"
#define TETRISCHED_DEFAULT_LOG_LEVEL tetrisched::logging::INFO
#define TETRISCHED_DEBUG(x)                                              \
  if (TETRISCHED_LOGGING_ENABLED &&                                      \
      TETRISCHED_DEFAULT_LOG_LEVEL >= tetrisched::logging::DEBUG) {      \
    auto sourceLocation = std::experimental::source_location::current(); \
    auto& logger = tetrisched::logging::Logger::info();                  \
    std::lock_guard<std::mutex> lock(logger.writeMutex);                 \
    logger << "[DEBUG, " << sourceLocation.function_name() << "] " << x  \
           << "\n";                                                      \
  }
#define TETRISCHED_INFO(x)                                               \
  if (TETRISCHED_LOGGING_ENABLED &&                                      \
      TETRISCHED_DEFAULT_LOG_LEVEL >= tetrisched::logging::INFO) {       \
    auto sourceLocation = std::experimental::source_location::current(); \
    auto& logger = tetrisched::logging::Logger::info();                  \
    std::lock_guard<std::mutex> lock(logger.writeMutex);                 \
    logger << "[INFO, " << sourceLocation.function_name() << "] " << x   \
           << "\n";                                                      \
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
  ExpressionConstructionException(std::string message);
  const char* what() const noexcept override;
};

/// An exception that is thrown when solving of an Expression fails.
class ExpressionSolutionException : public std::exception {
 private:
  std::string message;

 public:
  ExpressionSolutionException(std::string message);
  const char* what() const noexcept override;
};

/// An exception that is thrown when the Solver is used incorrectly.
class SolverException : public std::exception {
 private:
  std::string message;

 public:
  SolverException(std::string message);
  const char* what() const noexcept override;
};

/// A general Runtime exception.
class RuntimeException : public std::exception {
 private:
  std::string message;

 public:
  RuntimeException(std::string message);
  const char* what() const noexcept override;
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
  // A mutex to ensure that the output stream is not corrupted.
  std::mutex writeMutex;

  Logger(std::ostream& outputStream = std::cout, LogLevel level = INFO);

  template <typename T>
  Logger& operator<<(const T& val);

  ~Logger();

  void flush();

  static Logger& debug();

  static Logger& info();
};
}  // namespace logging

namespace timing {
/// A class to measure the time taken for a block of code to execute.
class ScopeTimer {
 private:
  std::string scopeTimerName;
  std::chrono::high_resolution_clock::time_point startTime;
  static std::ofstream& getOutputFileStream();

  static std::mutex sharedLock;  // Shared lock for all ScopeTimer instances

 public:
  ScopeTimer(std::string scopeTimerName);

  ~ScopeTimer();
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
