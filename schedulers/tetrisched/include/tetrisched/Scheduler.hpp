#ifndef _TETRISCHED_SCHEDULER_HPP_
#define _TETRISCHED_SCHEDULER_HPP_

#include "tetrisched/OptimizationPasses.hpp"
#include "tetrisched/Partition.hpp"
#include "tetrisched/Solver.hpp"

namespace tetrisched {
class Scheduler {
 private:
  /// The solver instance underlying the Scheduler with the given type.
  std::shared_ptr<Solver> solver;
  SolverBackendType solverBackend;
  /// The solver model to be passed to the expressions during parsing.
  SolverModelPtr solverModel;
  /// The solution to the last solver invocation (if available).
  std::optional<SolverSolutionPtr> solverSolution;
  /// The time discretization to use for the solver.
  Time discretization;
  /// The registered STRL expression (if available).
  std::optional<ExpressionPtr> expression;
  /// The registered OptimizationPasses to run.
  OptimizationPassRunner optimizationPasses;
  /// The directory where the logs are to be output.
  std::string logDir;

 public:
  /// Initialize the scheduler with a solver backend.
  Scheduler(Time discretization, SolverBackendType solverBackend,
            std::string logDir = "./", bool enableDynamicDiscretization = false,
            Time maxDiscretization = 5, float maxOccupancyThreshold = 0.8);

  /// Registers the STRL expression for the scheduler to schedule from
  /// and parses it to populate the SolverModel.
  void registerSTRL(
      ExpressionPtr expression, Partitions availablePartitions,
      Time currentTime, bool optimize = false,
      std::vector<std::pair<TimeRange, Time>> timeRangeToGranularities = {});

  /// Invokes the solver to schedule the registered STRL expression
  /// on the given partitions at the given time.
  /// Use expression->getSolution() to retrieve the solution.
  void schedule(Time currentTime);

  /// Retrieve the solution from the last invocation of the solver.
  SolverSolutionPtr getLastSolverSolution() const;

  /// Exports the model from the last invocation of the solver.
  void exportLastSolverModel(const std::string &fileName) const;

  /// Exports the solution from the last invocation of the solver.
  void exportLastSolverSolution(const std::string &fileName) const;
};
}  // namespace tetrisched

#endif  //_TETRISCHED_SCHEDULER_HPP_
