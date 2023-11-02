#include "tetrisched/Scheduler.hpp"

#ifdef _TETRISCHED_WITH_CPLEX_
#include "tetrisched/CPLEXSolver.hpp"
#endif
#include "tetrisched/Expression.hpp"
#ifdef _TETRISCHED_WITH_GUROBI_
#include "tetrisched/GurobiSolver.hpp"
#endif

namespace tetrisched {
Scheduler::Scheduler(Time discretization, SolverBackendType solverBackend)
    : discretization(discretization), solverBackend(solverBackend) {
  // Initialize the solver backend.
  switch (solverBackend) {
#ifdef _TETRISCHED_WITH_CPLEX_
    case SolverBackendType::CPLEX:
      solver = std::make_shared<CPLEXSolver>();
      break;
#endif
#ifdef _TETRISCHED_WITH_GUROBI_
    case SolverBackendType::GUROBI:
      solver = std::make_shared<GurobiSolver>();
      break;
#endif
    default:
      throw exceptions::SolverException(
          "The solver backend type is not supported.");
  }
  solverModel = solver->getModel();
}

void Scheduler::registerSTRL(ExpressionPtr expression,
                             Partitions availablePartitions, Time currentTime) {
  // Clear the previously saved expressions in the SolverModel.
  solverModel->clear();

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

void Scheduler::schedule(Time currentTime) {
  if (!this->expression.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No expression has been registered with the scheduler. "
        "Please invoke registerSTRL() first.");
  }

  // Set the log file based on the current time.
  std::string logFileName = "tetrisched_" + std::to_string(currentTime) + ".log";
  this->solver->setLogFile(logFileName);

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

void Scheduler::exportLastSolverModel(const std::string& fileName) const {
  if (!solverSolution.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No solution has been computed yet. Please invoke schedule() first.");
  }
  solver->exportModel(fileName);
}

void Scheduler::exportLastSolverSolution(const std::string& fileName) const {
  if (!solverSolution.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No solution has been computed yet. Please invoke schedule() first.");
  }
  throw exceptions::RuntimeException("Not Implemented yet!");
}
}  // namespace tetrisched
