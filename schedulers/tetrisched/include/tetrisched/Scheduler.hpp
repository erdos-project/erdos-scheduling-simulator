#ifndef _TETRISCHED_SCHEDULER_HPP_
#define _TETRISCHED_SCHEDULER_HPP_

#include "tetrisched/Solver.hpp"

namespace tetrisched {
class Scheduler {
 private:
  /// The solver instance underlying the Scheduler.
  std::shared_ptr<Solver> solver;
  /// The solver model to be passed to the expressions during parsing.
  SolverModelPtr solverModel;

 public:
  /// Initialize the scheduler with a solver backend.
  Scheduler();

  /// Registers the STRL expression for the scheduler to schedule from.
  void registerSTRL(ExpressionPtr expression);

  /// Invokes the solver to schedule the registered STRL expression.
  /// Returns the computed schedule.
  void schedule();
};
}  // namespace tetrisched

#endif  //_TETRISCHED_SCHEDULER_HPP_
