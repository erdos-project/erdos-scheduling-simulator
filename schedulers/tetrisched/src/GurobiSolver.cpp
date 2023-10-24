#include "tetrisched/GurobiSolver.hpp"

#include <chrono>

namespace tetrisched {
GurobiSolver::GurobiSolver()
    : gurobiEnv(new GRBEnv()), gurobiModel(new GRBModel(*gurobiEnv)) {}

SolverModelPtr GurobiSolver::getModel() {
  if (!solverModel) {
    solverModel = std::shared_ptr<SolverModel>(new SolverModel());
  }
  return solverModel;
}

void GurobiSolver::setModel(SolverModelPtr solverModelPtr) {
  solverModel = solverModelPtr;
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
  switch (variable->variableType) {
    case VariableType::VAR_INTEGER:
      return gurobiModel.addVar(lowerBound, upperBound, 0.0, GRB_INTEGER,
                                variable->variableName);
    case VariableType::VAR_CONTINUOUS:
      return gurobiModel.addVar(lowerBound, upperBound, 0.0, GRB_CONTINUOUS,
                                variable->variableName);
    case VariableType::VAR_INDICATOR:
      return gurobiModel.addVar(lowerBound, upperBound, 0.0, GRB_BINARY,
                                variable->variableName);
    default:
      throw tetrisched::exceptions::SolverException(
          "Invalid variable type: " + std::to_string(variable->variableType));
  }
}

GRBConstr GurobiSolver::translateConstraint(
    GRBModel& gurobiModel, const ConstraintPtr& constraint) const {
  // TODO (Sukrit): We are currently assuming that all constraints and objective
  // functions are linear. We may need to support quadratic constraints.
  GRBLinExpr constraintExpr;

  // Construct all the terms.
  for (const auto& [coefficient, variable] : constraint->terms) {
    if (variable) {
      constraintExpr += coefficient * gurobiVariables.at(variable->getId());
    } else {
      constraintExpr += coefficient;
    }
  }

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
        gurobiConstraint.set(GRB_IntAttr_Lazy, 1);
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
    GRBModel& gurobiModel,
    const ObjectiveFunctionPtr& objectiveFunction) const {
  // TODO (Sukrit): We are currently assuming that all constraints and objective
  // functions are linear. We may need to support quadratic constraints.
  GRBLinExpr objectiveExpr;

  // Construct all the terms.
  for (const auto& [coefficient, variable] : objectiveFunction->terms) {
    if (variable) {
      objectiveExpr += coefficient * gurobiVariables.at(variable->getId());
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

  // Generate all the variables and keep a cache of the variable indices
  // to the Gurobi variables.
  for (const auto& [variableId, variable] : solverModel->variables) {
    TETRISCHED_DEBUG("Adding variable " << variable->getName() << "("
                                        << variableId << ") to Gurobi Model.");
    gurobiVariables[variableId] = translateVariable(*gurobiModel, variable);
  }

  // Generate all the constraints.
  for (const auto& [constraintId, constraint] : solverModel->constraints) {
    TETRISCHED_DEBUG("Adding Constraint " << constraint->getName() << "("
                                          << constraintId
                                          << ") to Gurobi Model.");
    auto _ = translateConstraint(*gurobiModel, constraint);
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
}

void GurobiSolver::exportModel(const std::string& fileName) {
  gurobiModel->write(fileName);
}

SolverSolutionPtr GurobiSolver::solveModel() {
  // Create the result object.
  SolverSolutionPtr solverSolution = std::make_shared<SolverSolution>();

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
      break;
    case GRB_INF_OR_UNBD:
    case GRB_UNBOUNDED:
      solverSolution->solutionType = SolutionType::UNBOUNDED;
      break;
    default:
      solverSolution->solutionType = SolutionType::UNKNOWN;
      TETRISCHED_DEBUG("The Gurobi solver returned the value: "
                       << gurobiModel->get(GRB_IntAttr_Status));
      break;
  }
  TETRISCHED_DEBUG("The Gurobi solver took "
                   << solverSolution->solverTimeMicroseconds << " microseconds "
                   << "to solve the model.");

  // Retrieve all the variables from the Gurobi model into the SolverModel.
  for (const auto& [variableId, variable] : solverModel->variables) {
    if (gurobiVariables.find(variableId) == gurobiVariables.end()) {
      throw tetrisched::exceptions::SolverException(
          "Variable " + variable->getName() +
          " was not found in the Gurobi model.");
    }
    switch (variable->variableType) {
      case tetrisched::VariableType::VAR_INTEGER:
      case tetrisched::VariableType::VAR_CONTINUOUS:
      case tetrisched::VariableType::VAR_INDICATOR:
        variable->solutionValue =
            gurobiVariables.at(variableId).get(GRB_DoubleAttr_X);
        break;
      default:
        throw tetrisched::exceptions::SolverException(
            "Unsupported variable type: " + variable->variableType);
    }
  }
  return solverSolution;
}
}  // namespace tetrisched
