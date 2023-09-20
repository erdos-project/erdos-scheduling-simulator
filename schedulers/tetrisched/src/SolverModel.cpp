#include "tetrisched/SolverModel.hpp"

namespace tetrisched {

/*
 * Methods for VariableT.
 * These methods provide an implementation of the VariableT class.
 */

template <typename T>
VariableT<T>::VariableT(VariableType type, std::string name)
    : variableType(type), variableName(name) {}

template <typename T>
VariableT<T>::VariableT(VariableType type, std::string name, T lowerBound)
    : variableType(type), variableName(name), lowerBound(lowerBound) {}

template <typename T>
VariableT<T>::VariableT(VariableType type, std::string name, T lowerBound,
                        T upperBound)
    : variableType(type),
      variableName(name),
      lowerBound(lowerBound),
      upperBound(upperBound) {}

template <typename T>
VariableT<T>::VariableT(VariableType type, std::string name,
                        std::pair<T, T> range)
    : variableType(type),
      variableName(name),
      lowerBound(range.first),
      upperBound(range.second) {}

template <typename T>
void VariableT<T>::hint(T hintValue) {
  initialValue = hintValue;
}

template <typename T>
std::string VariableT<T>::toString() const {
  return variableName;
}

/*
 * Methods for Constraint.
 * These methods provide an implementation of the Constraint class.
 */

template <typename T>
ConstraintT<T>::ConstraintT(std::string constraintName, ConstraintType type,
                            T rightHandSide)
    : constraintName(constraintName),
      constraintType(type),
      rightHandSide(rightHandSide) {}

template <typename T>
void ConstraintT<T>::addTerm(std::pair<T, std::shared_ptr<VariableT<T>>> term) {
  terms.push_back(term);
}

template <typename T>
void ConstraintT<T>::addTerm(T coefficient,
                             std::shared_ptr<VariableT<T>> variable) {
  this->addTerm(std::make_pair(coefficient, variable));
}

template <typename T>
void ConstraintT<T>::addTerm(T constant) {
  this->addTerm(std::make_pair(constant, nullptr));
}

template <typename T>
std::string ConstraintT<T>::toString() const {
  std::string constraintString;
  for (auto &term : terms) {
    constraintString += "(" + std::to_string(term.first);
    if (term.second != nullptr) {
      constraintString += "*" + term.second->toString();
    }
    constraintString += ")";
    if (&term != &terms.back()) constraintString += "+";
  }
  switch (constraintType) {
    case CONSTR_EQ:
      constraintString += " = ";
      break;
    case CONSTR_LE:
      constraintString += " <= ";
      break;
    case CONSTR_GE:
      constraintString += " >= ";
      break;
  }
  constraintString += std::to_string(rightHandSide);
  return constraintString;
}

template <typename T>
size_t ConstraintT<T>::size() const {
  return terms.size();
}

/*
 * Methods for ObjectiveFunction.
 * These methods provide an implementation of the Constraint class.
 */
template <typename T>
ObjectiveFunctionT<T>::ObjectiveFunctionT(ObjectiveType type)
    : objectiveType(type) {}

template <typename T>
void ObjectiveFunctionT<T>::addTerm(T coefficient,
                                    std::shared_ptr<VariableT<T>> variable) {
  terms.push_back(std::make_pair(coefficient, variable));
}

template <typename T>
std::string ObjectiveFunctionT<T>::toString() const {
  std::string objectiveString;
  switch (objectiveType) {
    case OBJ_MAXIMIZE:
      objectiveString += "Maximize: ";
      break;
    case OBJ_MINIMIZE:
      objectiveString += "Minimize: ";
      break;
  }
  for (auto &term : terms) {
    objectiveString +=
        "(" + std::to_string(term.first) + "*" + term.second->toString() + ")";
    if (&term != &terms.back()) objectiveString += "+";
  }
  return objectiveString;
}

template <typename T>
size_t ObjectiveFunctionT<T>::size() const {
  return terms.size();
}

/*
 * Methods for SolverModel.
 * These methods provide an implementation of the Constraint class.
 */

template <typename T>
void SolverModelT<T>::addVariable(std::shared_ptr<VariableT<T>> variable) {
  variables.push_back(variable);
}

template <typename T>
void SolverModelT<T>::addConstraint(
    std::unique_ptr<ConstraintT<T>> constraint) {
  constraints.push_back(std::move(constraint));
}

template <typename T>
void SolverModelT<T>::setObjectiveFunction(
    std::unique_ptr<ObjectiveFunctionT<T>> objectiveFunction) {
  this->objectiveFunction = std::move(objectiveFunction);
}

template <typename T>
std::string SolverModelT<T>::toString() const {
  std::string modelString;
  modelString += objectiveFunction->toString();
  modelString += "\nConstraints: \n";
  for (auto &constraint : constraints) {
    modelString += "\t" + constraint->toString() + "\n";
  }
  modelString += "Variables: ";
  for (auto &variable : variables) {
    modelString += variable->toString();
    if (&variable != &variables.back()) {
      modelString += ", ";
    }
  }
  return modelString;
}

template <typename T>
size_t SolverModelT<T>::numVariables() const {
  return variables.size();
}

template <typename T>
size_t SolverModelT<T>::numConstraints() const {
  return constraints.size();
}
}  // namespace tetrisched
