#include "tetrisched/GoogleCPSolver.hpp"

namespace tetrisched {
GoogleCPSolver::GoogleCPSolver() : cpModel(new CpModelBuilder()) {}

SolverModelPtr GoogleCPSolver::getModel() {
  if (!solverModel) {
    solverModel = std::shared_ptr<SolverModel>(new SolverModel());
  }
  return solverModel;
}

void GoogleCPSolver::setModel(SolverModelPtr solverModelPtr) {
  solverModel = solverModelPtr;
}

GoogleCPSolver::GoogleCPVarType GoogleCPSolver::translateVariable(
    const VariablePtr& variable) const {
  // Check that a continuous variable is not passed in.
  if (variable->variableType == VariableType::VAR_CONTINUOUS) {
    throw exceptions::SolverException(
        "Cannot construct a continuous variable in ORTools.");
  } else if (variable->variableType == VariableType::VAR_INTEGER) {
    // Check that the Variable has been given both a lower and an upper bound.
    // This is required to set a Domain in ORTools.
    if (!(variable->lowerBound.has_value() &&
          variable->upperBound.has_value())) {
      throw exceptions::SolverException(
          "Cannot construct a variable without a lower and upper bound in "
          "ORTools: " +
          variable->toString());
    }

    // Construct the domain for the variable.
    const operations_research::Domain domain(variable->lowerBound.value(),
                                             variable->upperBound.value());
    return cpModel->NewIntVar(domain).WithName(variable->variableName);
  } else if (variable->variableType == VariableType::VAR_INDICATOR) {
    // Construct the Indicator variable.
    return cpModel->NewBoolVar().WithName(variable->variableName);
  } else {
    throw exceptions::SolverException("Cannot construct a variable of type " +
                                      std::to_string(variable->variableType) +
                                      " in ORTools.");
  }
}

operations_research::sat::Constraint GoogleCPSolver::translateConstraint(
    const ConstraintPtr& constraint) {
  // TODO (Sukrit): We are currently assuming that all constraints and
  // objectives are linear. We may need to support quadratic constraints.
  operations_research::sat::LinearExpr constraintExpr;

  // Construct all the terms.
  for (const auto& [coefficient, variable] : constraint->terms) {
    if (variable) {
      switch (variable->variableType) {
        case VariableType::VAR_INTEGER:
          constraintExpr +=
              coefficient * std::get<IntVar>(cpVariables.at(variable->getId()));
          break;
        case VariableType::VAR_INDICATOR:
          constraintExpr +=
              coefficient *
              std::get<BoolVar>(cpVariables.at(variable->getId()));
          break;
        default:
          throw exceptions::SolverException(
              "Cannot construct a constraint with a variable of type " +
              std::to_string(variable->variableType) + " in ORTools.");
      }
    } else {
      constraintExpr += coefficient;
    }
  }

  // Translate the constraint.
  switch (constraint->constraintType) {
    case ConstraintType::CONSTR_EQ:
      return cpModel->AddEquality(constraintExpr, constraint->rightHandSide)
          .WithName(constraint->getName());
    case ConstraintType::CONSTR_GE:
      return cpModel
          ->AddGreaterOrEqual(constraintExpr, constraint->rightHandSide)
          .WithName(constraint->getName());
    case ConstraintType::CONSTR_LE:
      return cpModel->AddLessOrEqual(constraintExpr, constraint->rightHandSide)
          .WithName(constraint->getName());
    default:
      throw exceptions::SolverException(
          "Invalid constraint type: " +
          std::to_string(constraint->constraintType));
  }
}

LinearExpr GoogleCPSolver::translateObjectiveFunction(
    const ObjectiveFunctionPtr& objectiveFunction) const {
  LinearExpr objectiveExpr;

  // Construct all the terms.
  for (const auto& [coefficient, variable] : objectiveFunction->terms) {
    if (variable) {
      switch (variable->variableType) {
        case VariableType::VAR_INTEGER:
          objectiveExpr +=
              coefficient * std::get<IntVar>(cpVariables.at(variable->getId()));
          break;
        case VariableType::VAR_INDICATOR:
          objectiveExpr += coefficient *
                           std::get<BoolVar>(cpVariables.at(variable->getId()));
          break;
        default:
          throw exceptions::SolverException(
              "Cannot construct an objective function with a variable of "
              "type " +
              std::to_string(variable->variableType) + " in ORTools.");
      }
    } else {
      objectiveExpr += coefficient;
    }
  }
  return objectiveExpr;
}

void GoogleCPSolver::translateModel() {
  if (!solverModel) {
    throw tetrisched::exceptions::SolverException(
        "Empty SolverModel for GurobiSolver. Nothing to translate!");
  }

  // Generate all the variables and keep a cache of the variable indices
  // to the ORTools variables.
  for (const auto& [variableId, variable] : solverModel->modelVariables) {
    TETRISCHED_DEBUG("Adding variable " << variable->getName() << "("
                                        << variable->getId()
                                        << ") to ORTools model.");
    cpVariables[variableId] = translateVariable(variable);
  }

  // Generate all the constraints.
  for (const auto& [constraintId, constraint] : solverModel->modelConstraints) {
    TETRISCHED_DEBUG("Adding constraint " << constraint->getName() << "("
                                          << constraint->getId()
                                          << ") to ORTools model.");
    auto _ = translateConstraint(constraint);
  }

  // Translate the objective function.
  auto objectiveExpr =
      translateObjectiveFunction(solverModel->objectiveFunction);
  switch (solverModel->objectiveFunction->objectiveType) {
    case ObjectiveType::OBJ_MINIMIZE:
      cpModel->Minimize(objectiveExpr);
      break;
    case ObjectiveType::OBJ_MAXIMIZE:
      cpModel->Maximize(objectiveExpr);
      break;
    default:
      throw exceptions::SolverException(
          "Invalid objective type: " +
          std::to_string(solverModel->objectiveFunction->objectiveType));
  }
}

void GoogleCPSolver::exportModel(const std::string& fileName) {
  cpModel->ExportToFile(fileName);
}

void GoogleCPSolver::setLogFile(const std::string& /* fname */) {
  throw tetrisched::exceptions::SolverException(
      "setLogFile() not implemented for GoogleCPSolver.");
}

SolverSolutionPtr GoogleCPSolver::solveModel() {
  throw exceptions::SolverException("Not implemented yet!");
}

}  // namespace tetrisched
