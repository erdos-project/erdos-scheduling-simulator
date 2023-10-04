#include "tetrisched/Scheduler.hpp"

#include "tetrisched/CPLEXSolver.hpp"

namespace tetrisched {
Scheduler::Scheduler(Time discretization) : discretization(discretization) {
  solver = std::make_shared<CPLEXSolver>();
  solverModel = solver->getModel();
}

void Scheduler::scheduleSTRL(ExpressionPtr expression,
                             Partitions availablePartitions, Time currentTime) {
  // Check if the expression is an objective function.
  if (expression->getType() != ExpressionType::EXPR_OBJECTIVE) {
    throw exceptions::ExpressionConstructionException(
        "The expression passed to the scheduler is not an objective function, "
        "but is of type: " +
        std::to_string(expression->getType()) + ".");
  }

  // Create the CapacityConstraintMap for the STRL tree to add constraints to.
  CapacityConstraintMap capacityConstraintMap(discretization);

  // Parse the ExpressionTree to populate the solver model.
  expression->parse(solverModel, availablePartitions, capacityConstraintMap,
                    currentTime);

  // Invoke the solver, and populate the results into the expression tree.
  solver->solveModel();
}
}  // namespace tetrisched
