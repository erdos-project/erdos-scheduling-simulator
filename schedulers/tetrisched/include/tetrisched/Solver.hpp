#ifndef _TETRISCHED_SOLVER_HPP_
#define _TETRISCHED_SOLVER_HPP_

#include "tetrisched/SolverModel.hpp"

namespace tetrisched {
/// The `SolutionType` enum represents the different types of solutions
/// that we can retrieve from the solver.
enum SolutionType {
  /// The Solver returned a feasible (but not optimal) solution.
  FEASIBLE = 0,
  /// The Solver returned an optimal solution.
  OPTIMAL = 1,
  /// The Solver returned an infeasible solution.
  INFEASIBLE = 2,
  /// The Solver returned an unbounded solution.
  UNBOUNDED = 3,
  /// The Solver returned an unknown solution.
  UNKNOWN = 4
};

/// The SolverSolution structure represents the information that we
/// retrieve from either the callbacks or the Solver that we can use
/// to evaluate our approach.
struct SolverSolution {
  /// The type of the solution.
  SolutionType solutionType;
  /// The objective value of the solution.
  std::optional<double> objectiveValue;
  /// The time taken by the solver to find the solution (in microseconds).
  uint64_t solverTimeMicroseconds;

  /// Get a string representation of the Solver's type.
  std::string getSolutionTypeStr() const {
    switch (solutionType) {
      case SolutionType::FEASIBLE:
        return "FEASIBLE";
      case SolutionType::OPTIMAL:
        return "OPTIMAL";
      case SolutionType::INFEASIBLE:
        return "INFEASIBLE";
      case SolutionType::UNBOUNDED:
        return "UNBOUNDED";
      case SolutionType::UNKNOWN:
        return "UNKNOWN";
      default:
        return "NOTIMPLEMENTED";
    }
  }
};
using SolverSolution = struct SolverSolution;
using SolverSolutionPtr = std::shared_ptr<SolverSolution>;

/// The Solver class is the abstract base class for all solver
/// backends and provides a common interface for the STRL
/// expressions to communicate with the backend.
class Solver {
 public:
  /// Retrieve a pointer to the SolverModel.
  /// The SolverModel is the interface to define STRL expressions over.
  virtual SolverModelPtr getModel() = 0;

  /// Replace the SolverModel in this instance with the given model.
  /// This may be used to switch backends when a model is already constructed.
  virtual void setModel(SolverModelPtr model) = 0;

  /// Translate the SolverModel into a backend-specific model.
  virtual void translateModel() = 0;

  /// Export the constructed model to the given file.
  virtual void exportModel(const std::string& fileName) = 0;

  /// Solve the constructed model.
  virtual SolverSolutionPtr solveModel() = 0;

  /// Get the name of the Solver.
  virtual std::string getName() const = 0;
};
using SolverPtr = std::shared_ptr<tetrisched::Solver>;
}  // namespace tetrisched
#endif  // _TETRISCHED_SOLVER_HPP_
