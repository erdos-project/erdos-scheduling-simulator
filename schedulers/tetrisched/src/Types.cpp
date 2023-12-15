#include "tetrisched/Types.hpp"

namespace tetrisched {
namespace exceptions {

/* Method definitions for ExpressionConstructionException */
ExpressionConstructionException::ExpressionConstructionException(
    std::string message)
    : message(message) {}
const char* ExpressionConstructionException::what() const noexcept {
  return message.c_str();
}

/* Method definitions for ExpressionSolutionException */
ExpressionSolutionException::ExpressionSolutionException(std::string message)
    : message(message) {}

const char* ExpressionSolutionException::what() const noexcept {
  return message.c_str();
}

/* Method definitions for SolverException */
SolverException::SolverException(std::string message) : message(message) {}
const char* SolverException::what() const noexcept { return message.c_str(); }

/* Method definitions for RuntimeException */
RuntimeException::RuntimeException(std::string message) : message(message) {}
const char* RuntimeException::what() const noexcept { return message.c_str(); }
}  // namespace exceptions

namespace logging {

/* Method definitions for Logger */
Logger::Logger(std::ostream& outputStream, LogLevel logLevel)
    : outputStream(outputStream), logLevel(logLevel) {
  std::lock_guard<std::mutex> lock(writeMutex);
  outputStream << std::unitbuf;
}

template <typename T>
Logger& Logger::operator<<(const T& value) {
  if (logLevel >= TETRISCHED_DEFAULT_LOG_LEVEL) {
    outputStream << value;
  }
  return *this;
}

Logger::~Logger() { outputStream << std::endl << std::nounitbuf; }

void Logger::flush() { outputStream.flush(); }

Logger& Logger::debug() {
  static std::filesystem::path outputDir =
      std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
          ? std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
          : "./";
  static std::filesystem::path outputFile =
      outputDir / TETRISCHED_LOG_FILE_NAME;
  static std::ofstream outputStream(outputFile, std::ios::app | std::ios::out);
  static Logger logger(outputStream.is_open() ? outputStream : std::cout,
                       LogLevel::DEBUG);
  return logger;
}

Logger& Logger::info() {
  static std::filesystem::path outputDir =
      std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
          ? std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
          : "./";
  static std::filesystem::path outputFile =
      outputDir / TETRISCHED_LOG_FILE_NAME;
  static std::ofstream outputStream(outputFile, std::ios::app | std::ios::out);
  static Logger logger(outputStream.is_open() ? outputStream : std::cout,
                       LogLevel::INFO);
  return logger;
}

}  // namespace logging

namespace timing {

/* Method definitions for ScopeTimer */
std::mutex ScopeTimer::sharedLock;

std::ofstream& ScopeTimer::getOutputFileStream() {
  static std::filesystem::path outputDir =
      std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
          ? std::getenv(TETRISCHED_LOGGING_DIR_ENV_NAME)
          : "./";
  static std::filesystem::path outputFile =
      outputDir / TETRISCHED_TIMING_FILE_NAME;
  static std::ofstream file(outputFile, std::ios::app | std::ios::out);
  return file;
}

ScopeTimer::ScopeTimer(std::string scopeTimerName)
    : scopeTimerName(scopeTimerName),
      startTime(std::chrono::high_resolution_clock::now()) {}

ScopeTimer::~ScopeTimer() {
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime)
          .count();

  auto startTimeMicroseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(
          startTime.time_since_epoch())
          .count();
  auto endtimeMicroseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(
          endTime.time_since_epoch())
          .count();
  std::lock_guard<std::mutex> lock(sharedLock);  // Acquire the shared lock
  getOutputFileStream() << "END," << scopeTimerName << ","
                        << startTimeMicroseconds << "," << endtimeMicroseconds
                        << "," << duration << "\n";
}

}  // namespace timing
}  // namespace tetrisched
