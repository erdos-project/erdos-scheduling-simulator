#ifndef _TETRISCHED_SCHEDULER_HPP_
#define _TETRISCHED_SCHEDULER_HPP_

#include "tetrisched/Partition.hpp"
#include "tetrisched/Solver.hpp"

namespace tetrisched {
class Scheduler {
 private:
  /// The solver instance underlying the Scheduler.
  std::shared_ptr<Solver> solver;
  /// The solver model to be passed to the expressions during parsing.
  SolverModelPtr solverModel;
  /// The solution to the last solver invocation (if available).
  std::optional<SolverSolutionPtr> solverSolution;
  /// The time discretization to use for the solver.
  Time discretization;
  /// The registered STRL expression (if available).
  std::optional<ExpressionPtr> expression;

 public:
  /// Initialize the scheduler with a solver backend.
  Scheduler(Time discretization);

  /// Registers the STRL expression for the scheduler to schedule from
  /// and parses it to populate the SolverModel.
  void registerSTRL(ExpressionPtr expression, Partitions availablePartitions,
                    Time currentTime);

  /// Invokes the solver to schedule the registered STRL expression
  /// on the given partitions at the given time.
  /// Use expression->getSolution() to retrieve the solution.
  void schedule();

  /// Retrieve the solution from the last invocation of the solver.
  SolverSolutionPtr getLastSolverSolution() const;
};
}  // namespace tetrisched

#endif  //_TETRISCHED_SCHEDULER_HPP_
