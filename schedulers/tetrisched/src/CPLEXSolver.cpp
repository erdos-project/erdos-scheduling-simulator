#include "tetrisched/CPLEXSolver.hpp"

#include <chrono>

#include "tetrisched/Types.hpp"
namespace tetrisched {

CPLEXSolver::CPLEXSolver()
    : cplexEnv(IloEnv()),
      solverModel(nullptr),
      cplexInstance(IloCplex(cplexEnv)) {}

SolverModelPtr CPLEXSolver::getModel() {
  if (!solverModel) {
    solverModel = std::shared_ptr<SolverModel>(new SolverModel());
  }
  return solverModel;
}

void CPLEXSolver::setModel(SolverModelPtr solverModelPtr) {
  solverModel = solverModelPtr;
}

CPLEXSolver::CPLEXVarType CPLEXSolver::translateVariable(
    const VariablePtr& variable) const {
  if (variable->variableType == tetrisched::VariableType::VAR_INTEGER) {
    // NOTE (Sukrit): Do not use value_or here since the type coercion renders
    // the IloIntMin and IloIntMax fallbacks incorrect.
    IloInt lowerBound =
        variable->lowerBound.has_value() ? variable->lowerBound.value() : 0;
    IloInt upperBound = variable->upperBound.has_value()
                            ? variable->upperBound.value()
                            : IloIntMax;
    return IloIntVar(cplexEnv, lowerBound, upperBound,
                     variable->getName().c_str());
  } else if (variable->variableType ==
             tetrisched::VariableType::VAR_CONTINUOUS) {
    IloNum lowerBound =
        variable->lowerBound.has_value() ? variable->lowerBound.value() : 0;
    IloNum upperBound = variable->upperBound.has_value()
                            ? variable->upperBound.value()
                            : IloInfinity;
    return IloNumVar(cplexEnv, lowerBound, upperBound, IloNumVar::Type::Float,
                     variable->getName().c_str());
  } else if (variable->variableType ==
             tetrisched::VariableType::VAR_INDICATOR) {
    return IloBoolVar(cplexEnv, variable->getName().c_str());
  } else {
    throw tetrisched::exceptions::SolverException(
        "Unsupported variable type: " + variable->variableType);
  }
}

IloRange CPLEXSolver::translateConstraint(
    const ConstraintPtr& constraint) const {
  IloExpr constraintExpr(cplexEnv);

  // Construct all the terms.
  for (const auto& [coefficient, variable] : constraint->terms) {
    if (variable) {
      // If the term has not been translated, throw an error.
      if (cplexVariables.find(variable->getId()) == cplexVariables.end()) {
        throw tetrisched::exceptions::SolverException(
            "Variable " + variable->getName() + " not found in CPLEX model.");
      }
      // Call the relevant function to add the term to the constraint.
      switch (variable->variableType) {
        case tetrisched::VariableType::VAR_INTEGER:
          constraintExpr +=
              coefficient *
              std::get<IloIntVar>(cplexVariables.at(variable->getId()));
          break;
        case tetrisched::VariableType::VAR_CONTINUOUS:
          constraintExpr +=
              coefficient *
              std::get<IloNumVar>(cplexVariables.at(variable->getId()));
          break;
        case tetrisched::VariableType::VAR_INDICATOR:
          constraintExpr +=
              coefficient *
              std::get<IloBoolVar>(cplexVariables.at(variable->getId()));
          break;
        default:
          throw tetrisched::exceptions::SolverException(
              "Unsupported variable type: " + variable->variableType);
      }
    } else {
      constraintExpr += coefficient;
    }
  }

  // Construct the RHS of the Constraint.
  IloRange rangeConstraint;
  switch (constraint->constraintType) {
    case tetrisched::ConstraintType::CONSTR_LE:
      rangeConstraint = (constraintExpr <= constraint->rightHandSide);
      break;
    case tetrisched::ConstraintType::CONSTR_EQ:
      rangeConstraint = (constraintExpr == constraint->rightHandSide);
      break;
    case tetrisched::ConstraintType::CONSTR_GE:
      rangeConstraint = (constraintExpr >= constraint->rightHandSide);
      break;
    default:
      throw tetrisched::exceptions::SolverException(
          "Unsupported constraint type: " + constraint->constraintType);
  }
  rangeConstraint.setName(constraint->getName().c_str());

  return rangeConstraint;
}

IloObjective CPLEXSolver::translateObjectiveFunction(
    const ObjectiveFunctionPtr& objectiveFunction) const {
  IloExpr objectiveExpr(cplexEnv);

  // Construct all the terms.
  for (const auto& [coefficient, variable] : objectiveFunction->terms) {
    if (variable) {
      // If the variable has not been translated, throw an error.
      if (cplexVariables.find(variable->getId()) == cplexVariables.end()) {
        throw tetrisched::exceptions::SolverException(
            "Variable " + variable->getName() + " not found in CPLEX model.");
      }
      // Call the relevant function to add the term to the constraint.
      switch (variable->variableType) {
        case tetrisched::VariableType::VAR_INTEGER:
          objectiveExpr +=
              coefficient *
              std::get<IloIntVar>(cplexVariables.at(variable->getId()));
          break;
        case tetrisched::VariableType::VAR_CONTINUOUS:
          objectiveExpr +=
              coefficient *
              std::get<IloNumVar>(cplexVariables.at(variable->getId()));
          break;
        case tetrisched::VariableType::VAR_INDICATOR:
          objectiveExpr +=
              coefficient *
              std::get<IloBoolVar>(cplexVariables.at(variable->getId()));
          break;
        default:
          throw tetrisched::exceptions::SolverException(
              "Unsupported variable type: " + variable->variableType);
      }
    } else {
      objectiveExpr += coefficient;
    }
  }

  // Construct the Sense of the Constraint.
  IloObjective objectiveConstraint;
  switch (objectiveFunction->objectiveType) {
    case tetrisched::ObjectiveType::OBJ_MAXIMIZE:
      objectiveConstraint = IloMaximize(cplexEnv, objectiveExpr);
      break;
    case tetrisched::ObjectiveType::OBJ_MINIMIZE:
      objectiveConstraint = IloMinimize(cplexEnv, objectiveExpr);
      break;
    default:
      throw tetrisched::exceptions::SolverException(
          "Unsupported objective type: " + objectiveFunction->objectiveType);
  }

  return objectiveConstraint;
}

void CPLEXSolver::translateModel() {
  if (!solverModel) {
    throw tetrisched::exceptions::SolverException(
        "Empty SolverModel for CPLEXSolver. Nothing to translate!");
  }

  // Generate the model to add the variables and constraints to.
  IloModel cplexModel(cplexEnv);

  // Generate all the variables and keep a cache of the variable indices
  // to the CPLEX variables.
  for (const auto& [variableId, variable] : solverModel->variables) {
    TETRISCHED_DEBUG("Adding Variable " << variable->getName() << "("
                                        << variableId << ") to CPLEX Model.");
    cplexVariables[variableId] = translateVariable(variable);
  }

  // Generate all the constraints and add it to the model.
  for (const auto& [constraintId, constraint] : solverModel->constraints) {
    TETRISCHED_DEBUG("Adding Constraint " << constraint->getName() << "("
                                          << constraintId
                                          << ") to CPLEX Model.");
    cplexModel.add(translateConstraint(constraint));
  }

  // Translate the objective function.
  cplexModel.add(translateObjectiveFunction(solverModel->objectiveFunction));

  // Extract the model to the CPLEX instance.
  cplexInstance.extract(cplexModel);
}

void CPLEXSolver::exportModel(const std::string& fname) {
  cplexInstance.exportModel(fname.c_str());
}

SolverSolutionPtr CPLEXSolver::solveModel() {
  // Create the result object.
  SolverSolutionPtr solverSolution = std::make_shared<SolverSolution>();

  // Solve the model.
  auto solverStartTime = std::chrono::high_resolution_clock::now();
  cplexInstance.solve();
  auto solverEndTime = std::chrono::high_resolution_clock::now();
  solverSolution->solverTimeMicroseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(solverEndTime -
                                                            solverStartTime)
          .count();

  // Retrieve the solution type.
  switch (cplexInstance.getStatus()) {
    case IloAlgorithm::Optimal:
      solverSolution->solutionType = SolutionType::OPTIMAL;
      solverSolution->objectiveValue = cplexInstance.getObjValue();
      break;
    case IloAlgorithm::Feasible:
      solverSolution->solutionType = SolutionType::FEASIBLE;
      solverSolution->objectiveValue = cplexInstance.getObjValue();
      break;
    case IloAlgorithm::Infeasible:
      solverSolution->solutionType = SolutionType::INFEASIBLE;
      break;
    case IloAlgorithm::Unbounded:
      solverSolution->solutionType = SolutionType::UNBOUNDED;
      break;
    default:
      solverSolution->solutionType = SolutionType::UNKNOWN;
      break;
  }
  TETRISCHED_DEBUG("Finished solving the model using the CPLEX solver!")

  // Retrieve all the variables from the CPLEX model into the SolverModel.
  for (const auto& [variableId, variable] : solverModel->variables) {
    if (cplexVariables.find(variableId) == cplexVariables.end()) {
      throw tetrisched::exceptions::SolverException(
          "Variable " + variable->getName() + " not found in CPLEX model.");
    }
    switch (variable->variableType) {
      case tetrisched::VariableType::VAR_INTEGER:
        variable->solutionValue = cplexInstance.getValue(
            std::get<IloIntVar>(cplexVariables.at(variableId)));
        break;
      case tetrisched::VariableType::VAR_CONTINUOUS:
        variable->solutionValue = cplexInstance.getValue(
            std::get<IloNumVar>(cplexVariables.at(variableId)));
        break;
      case tetrisched::VariableType::VAR_INDICATOR:
        variable->solutionValue = cplexInstance.getValue(
            std::get<IloBoolVar>(cplexVariables.at(variableId)));
        break;
      default:
        throw tetrisched::exceptions::SolverException(
            "Unsupported variable type: " + variable->variableType);
    }
    TETRISCHED_DEBUG("Setting value of variable "
                     << variable->getName() << " to "
                     << variable->solutionValue.value());
  }
  TETRISCHED_DEBUG("Successfully populated the solution values for all "
                   << solverModel->variables.size()
                   << " variables from CPLEX to SolverModel.");

  return solverSolution;
}

CPLEXSolver::~CPLEXSolver() { cplexEnv.end(); }

}  // namespace tetrisched
