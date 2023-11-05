#include "tetrisched/SolverModel.hpp"

namespace tetrisched {

/*
 * Methods for VariableT.
 * These methods provide an implementation of the VariableT class.
 */
// Initialize the static counter for variable IDs.
// This is required by compiler.
template <typename T>
uint32_t VariableT<T>::variableIdCounter = 0;

template <typename T>
VariableType VariableT<T>::isTypeValid(VariableType type) {
  if constexpr (std::is_integral_v<T>) {
    if (type == VAR_CONTINUOUS) {
      throw exceptions::SolverException(
          "Cannot construct a continuous variable with an integral type.");
    }
  }
  return type;
}

template <typename T>
VariableT<T>::VariableT(VariableType type, std::string name)
    : variableType(isTypeValid(type)),
      variableId(variableIdCounter++),
      variableName(name),
      lowerBound(0) {}

template <typename T>
VariableT<T>::VariableT(VariableType type, std::string name, T lowerBound)
    : variableType(isTypeValid(type)),
      variableId(variableIdCounter++),
      variableName(name),
      lowerBound(lowerBound) {}

template <typename T>
VariableT<T>::VariableT(VariableType type, std::string name, T lowerBound,
                        T upperBound)
    : variableType(isTypeValid(type)),
      variableId(variableIdCounter++),
      variableName(name),
      lowerBound(lowerBound),
      upperBound(upperBound) {}

template <typename T>
VariableT<T>::VariableT(VariableType type, std::string name,
                        std::pair<T, T> range)
    : variableType(isTypeValid(type)),
      variableId(variableIdCounter++),
      variableName(name),
      lowerBound(range.first),
      upperBound(range.second) {}

template <typename T>
void VariableT<T>::hint(T hintValue) {
  initialValue = hintValue;
}

template <typename T>
std::string VariableT<T>::toString() const {
  switch (variableType) {
    case VAR_CONTINUOUS:
      return "Variable<Continuous, " + getName() + ">";
    case VAR_INTEGER:
      return "Variable<Integer, " + getName() + ">";
    case VAR_INDICATOR:
      return "Variable<Indicator, " + getName() + ">";
    default:
      return "Variable<Unknown, " + getName() + ">";
  }
}

template <typename T>
std::string VariableT<T>::getName() const {
  return variableName;
}

template <typename T>
uint32_t VariableT<T>::getId() const {
  return variableId;
}

template <typename T>
std::optional<T> VariableT<T>::getValue() const {
  return solutionValue;
}

template <typename T>
std::optional<T> VariableT<T>::getLowerBound() const {
  return lowerBound;
}

template <typename T>
std::optional<T> VariableT<T>::getUpperBound() const {
  return upperBound;
}

/*
 * Methods for Constraint.
 * These methods provide an implementation of the Constraint class.
 */

// Initialize the static counter for constraint IDs.
// This is required by compiler.
template <typename T>
uint32_t ConstraintT<T>::constraintIdCounter = 0;

template <typename T>
ConstraintT<T>::ConstraintT(std::string constraintName, ConstraintType type,
                            T rightHandSide)
    : constraintId(constraintIdCounter++),
      constraintName(constraintName),
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
void ConstraintT<T>::addTerm(std::shared_ptr<VariableT<T>> variable) {
  this->addTerm(std::make_pair(1, variable));
}

template <typename T>
void ConstraintT<T>::addTerm(const XOrVariableT<T>& term) {
  if (term.isVariable()) {
    this->addTerm(term.template get<VariablePtr>());
  } else {
    this->addTerm(term.template get<T>());
  }
}

template <typename T>
void ConstraintT<T>::addTerm(T coefficient, const XOrVariableT<T>& term) {
  if (term.isVariable()) {
    this->addTerm(coefficient, term.template get<VariablePtr>());
  } else {
    this->addTerm(coefficient * term.template get<T>());
  }
}

template <typename T>
void ConstraintT<T>::addAttribute(ConstraintAttribute attribute) {
  attributes.insert(attribute);
}

template <typename T>
std::string ConstraintT<T>::toString() const {
  std::string constraintString;
  for (auto& term : terms) {
    constraintString += "(" + std::to_string(term.first);
    if (term.second != nullptr) {
      constraintString += "*" + term.second->getName();
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
std::string ConstraintT<T>::getName() const {
  return constraintName;
}

template <typename T>
uint32_t ConstraintT<T>::getId() const {
  return constraintId;
}

template <typename T>
size_t ConstraintT<T>::size() const {
  return terms.size();
}

template <typename T>
bool ConstraintT<T>::isTriviallySatisfiable() const {
  if (constraintType == CONSTR_EQ) {
    // For now, we assume that EQ constraints cannot be trivially satisfied.
    return false;
  }
  T bound = 0;
  for (const auto& [coefficient, variable] : terms) {
    if (variable) {
      auto variableBound = constraintType == CONSTR_LE
                               ? variable->getUpperBound()
                               : variable->getLowerBound();
      if (!variableBound.has_value()) {
        // If there is any variable that does not have a relevant bound, we
        // cannot guarantee trivial satisfiability.
        return false;
      } else {
        // Add the bound of this variable to the bound of the constraint.
        bound += coefficient * variableBound.value();
      }
    } else {
      // There was no Variable in this term. Just add the coefficient.
      bound += coefficient;
    }
  }

  switch (constraintType) {
    case CONSTR_LE:
      return bound <= rightHandSide;
    case CONSTR_GE:
      return bound >= rightHandSide;
    default:
      return false;
  }
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
void ObjectiveFunctionT<T>::addTerm(T constant) {
  terms.push_back(std::make_pair(constant, nullptr));
}

template <typename T>
std::shared_ptr<ConstraintT<T>> ObjectiveFunctionT<T>::toConstraint(
    std::string constraintName, ConstraintType constraintType,
    T rightHandSide) {
  auto constraint = std::make_shared<ConstraintT<T>>(
      constraintName, constraintType, rightHandSide);
  for (auto& term : terms) {
    constraint->addTerm(term);
  }
  return constraint;
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
  for (auto& term : terms) {
    if (term.second) {
      objectiveString +=
          "(" + std::to_string(term.first) + "*" + term.second->getName() + ")";
    } else {
      objectiveString += std::to_string(term.first);
    }
    if (&term != &terms.back()) objectiveString += "+";
  }
  return objectiveString;
}

template <typename T>
size_t ObjectiveFunctionT<T>::size() const {
  return terms.size();
}

template <typename T>
ObjectiveFunctionT<T>& ObjectiveFunctionT<T>::operator+=(
    const ObjectiveFunctionT<T>& other) {
  for (auto& term : other.terms) {
    terms.push_back(term);
  }
  return *this;
}

template <typename T>
ObjectiveFunctionT<T> ObjectiveFunctionT<T>::operator+(
    const ObjectiveFunctionT<T>& other) const {
  auto result = *this;
  for (auto& term : other.terms) {
    result.terms.push_back(term);
  }
  return result;
}

template <typename T>
ObjectiveFunctionT<T> ObjectiveFunctionT<T>::operator*(const T& scalar) const {
  auto result = *this;
  for (auto& term : result.terms) {
    term.first *= scalar;
  }
  return result;
}

template <typename T>
T ObjectiveFunctionT<T>::getValue() const {
  T value = 0;
  for (const auto& [coefficient, variable] : terms) {
    if (variable == nullptr) {
      value += coefficient;
    } else {
      auto variableValue = variable->getValue();
      if (variableValue) {
        value += coefficient * variableValue.value();
      } else {
        throw exceptions::ExpressionSolutionException(
            "Cannot retrieve value of variable " + variable->getName() +
            " in objective function.");
      }
    }
  }
  return value;
}

/*
 * Methods for SolverModel.
 * These methods provide an implementation of the Constraint class.
 */

template <typename T>
void SolverModelT<T>::addVariable(std::shared_ptr<VariableT<T>> variable) {
  variables[variable->getId()] = variable;
}

template <typename T>
void SolverModelT<T>::addConstraint(
    std::shared_ptr<ConstraintT<T>> constraint) {
  constraints[constraint->getId()] = constraint;
}

template <typename T>
void SolverModelT<T>::setObjectiveFunction(
    std::shared_ptr<ObjectiveFunctionT<T>> objectiveFunction) {
  this->objectiveFunction = objectiveFunction;
}

template <typename T>
std::string SolverModelT<T>::toString() const {
  std::string modelString;
  if (objectiveFunction != nullptr) {
    modelString += objectiveFunction->toString() + "\n\n";
  }
  if (constraints.size() > 0) {
    modelString += "Constraints: \n";
    for (auto& [_, constraint] : constraints) {
      modelString +=
          constraint->getName() + ": \t" + constraint->toString() + "\n";
    }
    modelString += "\n\n";
  } else {
    modelString += "No Constraints Found!\n\n";
  }
  if (variables.size() > 0) {
    modelString += "Variables: \n";
    for (auto& [_, variable] : variables) {
      modelString += "\t" + variable->toString();
    }
  } else {
    modelString += "No Variables Found!";
  }
  return modelString;
}

template <typename T>
void SolverModelT<T>::exportModel(std::string filename) const {
  std::ofstream modelFile(filename);
  modelFile << this->toString();
  modelFile.close();
}

template <typename T>
size_t SolverModelT<T>::numVariables() const {
  return variables.size();
}

template <typename T>
size_t SolverModelT<T>::numConstraints() const {
  return constraints.size();
}

template <typename T>
T SolverModelT<T>::getObjectiveValue() const {
  return objectiveFunction->getValue();
}

template <typename T>
void SolverModelT<T>::clear() {
  std::cout << "Inside SolverModelT<T>::clear" << std::endl;
  // Clear the cache first
  solutionValueCache.clear();
  // TODO: For each variable, if it has a solution value, then save it to the cache
  for (auto const& [id, variable] : variables) {
    if (variable->getValue()) {
      solutionValueCache[variable->getName()] = variable->getValue().value();
      std::cout << "Caching solution value for variable " << variable->getName() << "(" << id << ")" << std::endl;
    }
  }

  variables.clear();
  constraints.clear();
  objectiveFunction.reset();
}
}  // namespace tetrisched
