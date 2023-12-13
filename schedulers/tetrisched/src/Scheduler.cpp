#include "tetrisched/Scheduler.hpp"

#include <chrono>

#ifdef _TETRISCHED_WITH_CPLEX_
#include "tetrisched/CPLEXSolver.hpp"
#endif
#include "tetrisched/Expression.hpp"
#ifdef _TETRISCHED_WITH_GUROBI_
#include "tetrisched/GurobiSolver.hpp"
#endif

namespace tetrisched {
Scheduler::Scheduler(Time discretization, SolverBackendType solverBackend,
                     std::string logDir, bool enableDynamicDiscretization,
                     Time maxDiscretization, float maxOccupancyThreshold)
    : solverBackend(solverBackend),
      discretization(discretization),
      logDir(logDir) {
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
  optimizationPasses =
      OptimizationPassRunner(true, enableDynamicDiscretization, discretization,
                             maxDiscretization, maxOccupancyThreshold);
}

void Scheduler::registerSTRL(
    ExpressionPtr expression, Partitions availablePartitions, Time currentTime,
    bool optimize,
    std::vector<std::pair<TimeRange, Time>> timeRangeToGranularities) {
  TETRISCHED_SCOPE_TIMER("Scheduler::registerSTRL," +
                         std::to_string(currentTime));
  TETRISCHED_INFO("Registering the STRL expression: " << expression->getName()
                                                      << " for time "
                                                      << currentTime << ".")
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
  CapacityConstraintMapPtr capacityConstraintMap;

  // Create the CapacityConstraintMap for the STRL tree to add constraints to.
  if (timeRangeToGranularities.size() == 0) {
    capacityConstraintMap =
        std::make_shared<CapacityConstraintMap>(discretization);
  } else {
    capacityConstraintMap =
        std::make_shared<CapacityConstraintMap>(timeRangeToGranularities);
  }

  // Run the Pre-Translation OptimizationPasses on this expression.
  if (optimize) {
    TETRISCHED_SCOPE_TIMER(
        "Scheduler::registerSTRL::preTranslationOptimizationPasses," +
        std::to_string(currentTime));
    auto optimizerStartTime = std::chrono::high_resolution_clock::now();
    optimizationPasses.runPreTranslationPasses(currentTime, expression,
                                               capacityConstraintMap);
    auto optimizerEndTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        optimizerEndTime - optimizerStartTime)
                        .count();
    TETRISCHED_DEBUG("Pre Translation Optimization Passes took: "
                     << duration << " microseconds.")
  }

  {
    TETRISCHED_SCOPE_TIMER("Scheduler::registerSTRL::parse," +
                           std::to_string(currentTime));
    // Parse the ExpressionTree to populate the solver model.
    TETRISCHED_DEBUG("Beginning the parsing of the ExpressionTree rooted at "
                     << expression->getName() << ".")
    auto _ = expression->parse(solverModel, availablePartitions,
                               capacityConstraintMap, currentTime);
    TETRISCHED_DEBUG("Finished parsing the ExpressionTree rooted at "
                     << expression->getName() << ".")
  }

  // Run the Post-Translation OptimizationPasses on this expression.
  if (optimize) {
    TETRISCHED_SCOPE_TIMER(
        "Scheduler::registerSTRL::postTranslationOptimizationPasses," +
        std::to_string(currentTime));
    auto optimizerStartTime = std::chrono::high_resolution_clock::now();
    optimizationPasses.runPostTranslationPasses(currentTime, expression,
                                                capacityConstraintMap);
    auto optimizerEndTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        optimizerEndTime - optimizerStartTime)
                        .count();
    TETRISCHED_DEBUG("Post Translation Optimization Passes took: "
                     << duration << " microseconds.")
  }
}

void Scheduler::schedule(Time currentTime) {
  TETRISCHED_INFO("Invoking the Scheduler for time " << currentTime << ".")
  if (!this->expression.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No expression has been registered with the scheduler. "
        "Please invoke registerSTRL() first.");
  }

  // Set the log file based on the current time.
  std::filesystem::path logFileName =
      "tetrisched_" + std::to_string(currentTime) + ".log";
  this->solver->setLogFile(std::filesystem::path(logDir) / logFileName);

  // Translate the model to the solver backend.
  {
    TETRISCHED_SCOPE_TIMER("Scheduler::schedule::translateModel," +
                           std::to_string(currentTime));
    this->solver->translateModel();
  }

  // Solve the model.
  {
    TETRISCHED_SCOPE_TIMER("Scheduler::schedule::solveModel," +
                           std::to_string(currentTime));
    solverSolution = this->solver->solveModel();
  }

  // Populate the results from the solver into the expression tree.
  {
    TETRISCHED_SCOPE_TIMER("Scheduler::schedule::populateResults," +
                           std::to_string(currentTime));
    if (solverSolution.has_value() && solverSolution.value()->isValid()) {
      this->expression.value()->populateResults(solverModel);
    }
  }
}

SolverSolutionPtr Scheduler::getLastSolverSolution() const {
  if (!solverSolution.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No solution has been computed yet. Please invoke schedule() first.");
  }
  return solverSolution.value();
}

void Scheduler::exportLastSolverModel(const std::string &fileName) const {
  if (!solverSolution.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No solution has been computed yet. Please invoke schedule() first.");
  }
  solver->exportModel(fileName);
}

void Scheduler::exportLastSolverSolution(
    const std::string & /* fileName */) const {
  if (!solverSolution.has_value()) {
    throw exceptions::ExpressionSolutionException(
        "No solution has been computed yet. Please invoke schedule() first.");
  }
  throw exceptions::RuntimeException("Not Implemented yet!");
}
}  // namespace tetrisched
