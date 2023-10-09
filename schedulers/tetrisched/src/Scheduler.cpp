#include "tetrisched/Scheduler.hpp"

#include "tetrisched/CPLEXSolver.hpp"
#include "tetrisched/Expression.hpp"

namespace tetrisched {
Scheduler::Scheduler(Time discretization) : discretization(discretization) {
  solver = std::make_shared<CPLEXSolver>();
  solverModel = solver->getModel();
}

void Scheduler::registerSTRL(ExpressionPtr expression,
                             Partitions availablePartitions, Time currentTime) {
  // Check if the expression is an objective function.
  if (expression->getType() != ExpressionType::EXPR_OBJECTIVE) {
    throw exceptions::ExpressionConstructionException(
        "The expression passed to the scheduler is not an objective function, "
        "but is of type: " +
        std::to_string(expression->getType()) + ".");
  }
  // Save the expression.
  this->expression = expression;

  // Create the CapacityConstraintMap for the STRL tree to add constraints to.
  CapacityConstraintMap capacityConstraintMap(discretization);

  // Parse the ExpressionTree to populate the solver model.
  auto _ = expression->parse(solverModel, availablePartitions,
                             capacityConstraintMap, currentTime);
}

void Scheduler::schedule() {
  if (!this->expression.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No expression has been registered with the scheduler. "
        "Please invoke registerSTRL() first.");
  }

  // Translate the model to the solver backend.
  this->solver->translateModel();

  // Solve the model.
  solverSolution = this->solver->solveModel();

  // Populate the results from the solver into the expression tree.
  this->expression.value()->populateResults(solverModel);
}

SolverSolutionPtr Scheduler::getLastSolverSolution() const {
  if (!solverSolution.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No solution has been computed yet. Please invoke schedule() first.");
  }
  return solverSolution.value();
}
}  // namespace tetrisched
