#ifndef _TETRISCHED_TYPES_HPP_
#define _TETRISCHED_TYPES_HPP_

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <iostream>

// Macros for logging.
#define TETRISCHED_DEBUG_ENABLED true
#define TETRISCHED_DEBUG(x) \
  if (TETRISCHED_DEBUG_ENABLED) { \
    std::cout << x << std::endl; \
  }

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
}  // namespace exceptions

/// A Representation of time in the system.
/// We currently use a uint32_t since it translates well from the simulator.
/// When this library is deployed for real use, this might need to change to a
/// double.
typedef uint32_t Time;

/// General forward declarations.
class Expression;
typedef std::unique_ptr<Expression> ExpressionPtr;

/// Forward declarations for Solver instantiations so that we can declare
/// them as friend classes in the model.
class CPLEXSolver;
}  // namespace tetrisched

#endif
