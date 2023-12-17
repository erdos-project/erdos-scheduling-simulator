#include "tetrisched/GurobiSolver.hpp"

#include <chrono>
#include <cmath>
#include <thread>

#define TETRISCHED_SOLUTION_UPPER_BOUND_DELTA 0.1

namespace tetrisched {

/**
 * Methods for GurobiInterruptOptimizationCallback.
 */
GurobiSolver::GurobiInterruptOptimizationCallback::
    GurobiInterruptOptimizationCallback(GurobiInterruptParams params)
    : params(params), startTime(std::chrono::steady_clock::now()) {}

void GurobiSolver::GurobiInterruptOptimizationCallback::callback() {
  try {
    auto currentTime = std::chrono::steady_clock::now();
    if (where == GRB_CB_POLLING && params.timeLimitMs.has_value()) {
      auto elapsedTimeMs =
          std::chrono::duration_cast<std::chrono::milliseconds>(currentTime -
                                                                startTime)
              .count();

      if (elapsedTimeMs > params.timeLimitMs.value()) {
        abort();
      }
    } else if (where == GRB_CB_MIPSOL && params.utilityUpperBound.has_value()) {
      if (getDoubleInfo(GRB_CB_MIPSOL_OBJ) >=
          params.utilityUpperBound.value() -
              TETRISCHED_SOLUTION_UPPER_BOUND_DELTA) {
        abort();
      }
    }
  } catch (GRBException& e) {
    std::cout << "Gurobi Solver failed with error code: " << e.getErrorCode()
              << std::endl;
    std::cout << "The error message was: " << e.getMessage() << std::endl;
  } catch (...) {
    std::cout << "Error during GurobiInterruptOptimizationCallback::callback()"
              << std::endl;
  }
}

/**
 * Methods for GurobiSolver.
 */
GurobiSolver::GurobiSolver()
    : gurobiEnv(new GRBEnv()),
      gurobiModel(new GRBModel(*gurobiEnv)),
      logFileName("") {}

SolverModelPtr GurobiSolver::getModel() {
  if (!solverModel) {
    solverModel = std::shared_ptr<SolverModel>(new SolverModel());
  }
  return solverModel;
}

void GurobiSolver::setModel(SolverModelPtr solverModelPtr) {
  solverModel = solverModelPtr;
}

void GurobiSolver::setParameters(GRBModel& gurobiModel) {
  // Set the maximum numer of threads.
  const auto thread_count = std::thread::hardware_concurrency();
  gurobiModel.set(GRB_IntParam_Threads, thread_count);

  // Ask Gurobi to aggressively cut the search space.
  gurobiModel.set(GRB_IntParam_Cuts, 3);

  // Ask Gurobi to conservatively presolve the model.
  gurobiModel.set(GRB_IntParam_Presolve, 1);

  // Ask Gurobi to find new incumbent solutions rather than prove bounds.
  gurobiModel.set(GRB_IntParam_MIPFocus, 1);

  // Ask Gurobi to not output to the console, and instead direct it
  // to the specified file.
  // gurobiModel.set(GRB_IntParam_LogToConsole, 0);
  gurobiModel.set(GRB_StringParam_LogFile, logFileName);
}

GRBVar GurobiSolver::translateVariable(GRBModel& gurobiModel,
                                       const VariablePtr& variable) const {
  // Note (Sukrit): Do not use value_or here since the type coercion renders
  // GRB_INFINITY to a nonsensical value.
  double lowerBound =
      variable->lowerBound.has_value() ? variable->lowerBound.value() : 0;
  double upperBound = variable->upperBound.has_value()
                          ? variable->upperBound.value()
                          : GRB_INFINITY;
  GRBVar var;
  switch (variable->variableType) {
    case VariableType::VAR_INTEGER:
      var = gurobiModel.addVar(lowerBound, upperBound, 0.0, GRB_INTEGER,
                               variable->variableName);
      break;
    case VariableType::VAR_CONTINUOUS:
      var = gurobiModel.addVar(lowerBound, upperBound, 0.0, GRB_CONTINUOUS,
                               variable->variableName);
      break;
    case VariableType::VAR_INDICATOR:
      var = gurobiModel.addVar(lowerBound, upperBound, 0.0, GRB_BINARY,
                               variable->variableName);
      break;
    default:
      throw tetrisched::exceptions::SolverException(
          "Invalid variable type: " + std::to_string(variable->variableType));
  }
  // Give the Gurobi variable an initial solution value if it is available.
  if (variable->initialValue.has_value()) {
    var.set(GRB_DoubleAttr_Start, variable->initialValue.value());
    TETRISCHED_DEBUG("Setting start value of variable "
                     << variable->getName() << "(" << variable->getId()
                     << ") to " << variable->initialValue.value());
  }
  return var;
}

GRBConstr GurobiSolver::translateConstraint(
    GRBModel& gurobiModel, const ConstraintPtr& constraint) const {
  // Construct all the terms.
  std::vector<TETRISCHED_ILP_TYPE> coefficients;
  std::vector<GRBVar> variables;
  coefficients.reserve(constraint->terms.size());
  variables.reserve(constraint->terms.size());

  for (const auto& [coefficient, variable] : constraint->terms) {
    if (variable) {
      auto gurobiVariable = variable->gurobiVariable;
      if (!gurobiVariable.has_value()) {
        throw tetrisched::exceptions::SolverException(
            "Variable " + variable->getName() +
            " was not found in the Gurobi model.");
      }
      coefficients.push_back(coefficient);
      variables.push_back(gurobiVariable.value());
    } else {
      throw tetrisched::exceptions::SolverException(
          "Constant terms in LHS are not supported in Gurobi.");
    }
  }

  // Construct the LHS of the Constraint.
  GRBLinExpr constraintExpr = 0;
  constraintExpr.addTerms(&coefficients[0], &variables[0],
                          constraint->terms.size());

  // Translate the constraint.
  GRBConstr gurobiConstraint;
  switch (constraint->constraintType) {
    case ConstraintType::CONSTR_EQ:
      gurobiConstraint = gurobiModel.addConstr(constraintExpr, GRB_EQUAL,
                                               constraint->rightHandSide,
                                               constraint->getName());
      break;
    case ConstraintType::CONSTR_GE:
      gurobiConstraint = gurobiModel.addConstr(
          constraintExpr, GRB_GREATER_EQUAL, constraint->rightHandSide,
          constraint->getName());
      break;
    case ConstraintType::CONSTR_LE:
      gurobiConstraint = gurobiModel.addConstr(constraintExpr, GRB_LESS_EQUAL,
                                               constraint->rightHandSide,
                                               constraint->getName());
      break;
    default:
      throw tetrisched::exceptions::SolverException(
          "Invalid constraint type: " +
          std::to_string(constraint->constraintType));
  }

  // Add the attributes (if any) to the constraint.
  for (auto& attribute : constraint->attributes) {
    switch (attribute) {
      case ConstraintAttribute::LAZY_CONSTRAINT:
        gurobiConstraint.set(GRB_IntAttr_Lazy, 3);
        break;
      default:
        throw tetrisched::exceptions::SolverException(
            "Constraint attribute: " + std::to_string(attribute) +
            " not supported by Gurobi.");
    }
  }
  return gurobiConstraint;
}

GRBLinExpr GurobiSolver::translateObjectiveFunction(
    GRBModel& /* gurobiModel */,
    const ObjectiveFunctionPtr& objectiveFunction) const {
  // TODO (Sukrit): We are currently assuming that all constraints and objective
  // functions are linear. We may need to support quadratic constraints.
  GRBLinExpr objectiveExpr;

  // Construct all the terms.
  for (const auto& [coefficient, variable] : objectiveFunction->terms) {
    if (variable) {
      auto gurobiVariable = variable->gurobiVariable;
      if (!gurobiVariable.has_value()) {
        throw tetrisched::exceptions::SolverException(
            "Variable " + variable->getName() +
            " was not found in the Gurobi model.");
      }
      objectiveExpr += coefficient * gurobiVariable.value();
    } else {
      objectiveExpr += coefficient;
    }
  }

  return objectiveExpr;
}

void GurobiSolver::translateModel() {
  if (!solverModel) {
    throw tetrisched::exceptions::SolverException(
        "Empty SolverModel for GurobiSolver. Nothing to translate!");
  }

  gurobiModel = std::make_unique<GRBModel>(*gurobiEnv);
  setParameters(*gurobiModel);

  {
    TETRISCHED_SCOPE_TIMER("GurobiSolver::translateModel::translateVariables")
    // Generate all the variables and keep a cache of the variable indices
    // to the Gurobi variables.
    for (const auto& [variableId, variable] : solverModel->modelVariables) {
      TETRISCHED_DEBUG("Adding variable " << variable->getName() << "("
                                          << variableId
                                          << ") to Gurobi Model.");
      variable->gurobiVariable = translateVariable(*gurobiModel, variable);
    }
  }

  {
    TETRISCHED_SCOPE_TIMER("GurobiSolver::translateModel::translateConstraints")
    // Generate all the constraints.
    for (const auto& [constraintId, constraint] :
         solverModel->modelConstraints) {
      if (constraint->isActive()) {
        TETRISCHED_DEBUG("Adding active Constraint " << constraint->getName()
                                                     << "(" << constraintId
                                                     << ") to Gurobi Model.");
        translateConstraint(*gurobiModel, constraint);
      } else {
        TETRISCHED_DEBUG("Skipping the addition of inactive Constraint "
                         << constraint->getName() << "(" << constraintId
                         << ") to Gurobi Model.");
      }
    }
  }

  // Translate the objective function.
  auto objectiveExpr =
      translateObjectiveFunction(*gurobiModel, solverModel->objectiveFunction);
  switch (solverModel->objectiveFunction->objectiveType) {
    case ObjectiveType::OBJ_MINIMIZE:
      gurobiModel->setObjective(objectiveExpr, GRB_MINIMIZE);
      break;
    case ObjectiveType::OBJ_MAXIMIZE:
      gurobiModel->setObjective(objectiveExpr, GRB_MAXIMIZE);
      break;
    default:
      throw tetrisched::exceptions::SolverException(
          "Invalid objective type: " +
          std::to_string(solverModel->objectiveFunction->objectiveType));
  }

  // Construct the Interrupt parameters.
  if (solverModel->objectiveFunction->upperBound.has_value()) {
    interruptParams.utilityUpperBound =
        solverModel->objectiveFunction->upperBound.value();
  }
}

void GurobiSolver::exportModel(const std::string& fileName) {
  gurobiModel->write(fileName);
}

void GurobiSolver::setLogFile(const std::string& fileName) {
  logFileName = fileName;
}

SolverSolutionPtr GurobiSolver::solveModel() {
  // Create the result object.
  SolverSolutionPtr solverSolution = std::make_shared<SolverSolution>();

  // Construct the Interrupt callback, and register it with the model.
  GurobiInterruptOptimizationCallback interruptCallback(interruptParams);
  gurobiModel->setCallback(&interruptCallback);

  // Solve the model.
  auto solverStartTime = std::chrono::high_resolution_clock::now();
  gurobiModel->optimize();
  auto solverEndTime = std::chrono::high_resolution_clock::now();
  solverSolution->solverTimeMicroseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(solverEndTime -
                                                            solverStartTime)
          .count();

  // Retrieve the solution type.
  switch (gurobiModel->get(GRB_IntAttr_Status)) {
    case GRB_OPTIMAL:
      solverSolution->solutionType = SolutionType::OPTIMAL;
      solverSolution->objectiveValue = gurobiModel->get(GRB_DoubleAttr_ObjVal);
      break;
    case GRB_SUBOPTIMAL:
      solverSolution->solutionType = SolutionType::FEASIBLE;
      solverSolution->objectiveValue = gurobiModel->get(GRB_DoubleAttr_ObjVal);
      break;
    case GRB_INFEASIBLE:
      solverSolution->solutionType = SolutionType::INFEASIBLE;
      return solverSolution;
    case GRB_INTERRUPTED: {
      auto solutionCount = gurobiModel->get(GRB_IntAttr_SolCount);
      if (solutionCount > 0) {
        solverSolution->solutionType = SolutionType::FEASIBLE;
        break;
      } else {
        solverSolution->solutionType = SolutionType::NO_SOLUTION;
        return solverSolution;
      }
    }
    case GRB_INF_OR_UNBD:
    case GRB_UNBOUNDED:
      solverSolution->solutionType = SolutionType::UNBOUNDED;
      return solverSolution;
    default:
      solverSolution->solutionType = SolutionType::UNKNOWN;
      TETRISCHED_DEBUG("The Gurobi solver returned the value: "
                       << gurobiModel->get(GRB_IntAttr_Status));
      return solverSolution;
  }
  TETRISCHED_DEBUG("The Gurobi solver took "
                   << solverSolution->solverTimeMicroseconds << " microseconds "
                   << "to solve the model.");

  // Retrieve all the variables from the Gurobi model into the SolverModel.
  for (const auto& [variableId, variable] : solverModel->modelVariables) {
    if (!variable->gurobiVariable.has_value()) {
      throw tetrisched::exceptions::SolverException(
          "Variable " + variable->getName() +
          " was not found in the Gurobi model.");
    }
    auto gurobiVariable = variable->gurobiVariable.value();
    switch (variable->variableType) {
      case tetrisched::VariableType::VAR_CONTINUOUS: {
        double variableValue = gurobiVariable.get(GRB_DoubleAttr_X);
        TETRISCHED_DEBUG("Variable " << variable->getName() << "(" << variableId
                                     << ") has solved value " << variableValue);
        variable->solutionValue = variableValue;
        break;
      }
      case tetrisched::VariableType::VAR_INTEGER:
      case tetrisched::VariableType::VAR_INDICATOR: {
        // If this is an integer or an indicator variable, we round it to
        // nearest integer value.
        double variableValue = std::round(gurobiVariable.get(GRB_DoubleAttr_X));
        TETRISCHED_DEBUG("Variable " << variable->getName() << "(" << variableId
                                     << ") has solved value " << variableValue);
        variable->solutionValue = variableValue;
        break;
      }
      default:
        throw tetrisched::exceptions::SolverException(
            "Unsupported variable type: " + variable->variableType);
    }
  }
  return solverSolution;
}
}  // namespace tetrisched
