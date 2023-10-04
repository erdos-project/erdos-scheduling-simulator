#ifndef _TETRISCHED_SOLVERMODEL_HPP_
#define _TETRISCHED_SOLVERMODEL_HPP_

#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tetrisched/Types.hpp"

namespace tetrisched {

/// A `VariableType` enumeration represents the types of variable that we allow
/// the user to construct. These map to the types of variables that the
/// underlying solver supports.
enum VariableType { VAR_CONTINUOUS, VAR_INTEGER, VAR_INDICATOR };
using VariableType = enum VariableType;

/// A `VariableT` class represents a variable in the solver model. Note that
/// this is an internal representation of a VariableT and does not contain an
/// actual variable in a solver. This is used to program the STRL expression and
/// the translation to the underlying solution strategy is done at model
/// creation time.
template <typename T>
class VariableT {
 private:
  /// Used to generate unique IDs for each Variable.
  static uint32_t variableIdCounter;
  /// The type of the variable (as supported by the underlying solver).
  VariableType variableType;
  /// The ID of the variable.
  uint32_t variableId;
  /// The name for the variable.
  std::string variableName;
  /// An optional initial value for the variable.
  /// If unspecified, the solver will choose an initial value.
  std::optional<T> initialValue;
  /// An optional lower-bound for the variable.
  /// If unspecified, the solver will choose the lower bound for the given T.
  std::optional<T> lowerBound;
  /// An optional upper-bound for the variable.
  /// If unspecified, the solver will choose the upper bound for the given T.
  std::optional<T> upperBound;
  /// An optional solution value for the variable.
  /// If unspecified, the solver has not found a solution for this problem yet.
  std::optional<T> solutionValue;
  /// Checks if the VariableType is valid.
  /// Throws an exception if the VariableType is invalid.
  /// Returns the type if it is valid.
  static VariableType isTypeValid(VariableType type);

 public:
  /// Generate a new variable with the given type and name.
  /// This variable is constrained to the entire possible range for this type,
  /// and provides no hint to the solver for its initial value.
  VariableT(VariableType variableType, std::string variableName);

  /// Generate a new variable with the given type, name and a lower bound.
  /// This variable is constrained to be higher than the given lower bound.
  VariableT(VariableType variableType, std::string variableName, T lowerBound);

  /// Generate a new variable with the given type, name and a range of values.
  /// This variable can take any value between the lower and uppper bound.
  VariableT(VariableType variableType, std::string variableName, T lowerBound,
            T upperBound);
  VariableT(VariableType variableType, std::string variableName,
            std::pair<T, T> range);

  /// Hints the solver that the given value might be a good starting point for
  /// search.
  void hint(T hintValue);

  /// Retrieve a string representation of this VariableT.
  std::string toString() const;

  /// Retrieve the name of this VariableT.
  std::string getName() const;

  /// Retrieve the ID of this VariableT.
  uint32_t getId() const;

  /// Retrieve the solution value for this VariableT.
  /// If the solution value is not set, then the solver hasn't found a solution
  /// (yet).
  std::optional<T> getValue() const;

  /// Annotate friend classes for Solvers so that they have access to internals.
  friend tetrisched::CPLEXSolver;
  friend tetrisched::GurobiSolver;
  friend tetrisched::GoogleCPSolver;
};

// Specialize the VariableT class for Integer type.
template class VariableT<TETRISCHED_ILP_TYPE>;
using Variable = VariableT<TETRISCHED_ILP_TYPE>;
using VariablePtr = std::shared_ptr<Variable>;

/// A `XOrVariableT` type encapsulates either a value known at runtime
/// prior to the solver being invoked, or a VariableT that is created
/// for the solver to assign a value to.
template <typename X>
class XOrVariableT {
 private:
  std::variant<std::monostate, X, VariablePtr> value;

 public:
  /// Constructors and operators.
  XOrVariableT() : value(std::monostate()) {}
  XOrVariableT(const X newValue) : value(newValue) {}
  XOrVariableT(const VariablePtr newValue) : value(newValue) {}
  XOrVariableT(const XOrVariableT& newValue) = default;
  XOrVariableT(XOrVariableT&& newValue) = default;
  XOrVariableT& operator=(const X newValue) {
    value = newValue;
    return *this;
  }
  XOrVariableT& operator=(const VariablePtr newValue) {
    value = newValue;
    return *this;
  }
  template <typename Y>
  operator XOrVariableT<Y>() const {
    if (this->isVariable()) {
      return XOrVariableT<Y>(this->get<VariablePtr>());
    } else {
      return XOrVariableT<Y>(static_cast<Y>(this->get<X>()));
    }
  }

  /// Resolves the value inside this class.
  X resolve() const {
    // If the value is the provided type, then return it.
    if (std::holds_alternative<X>(value)) {
      return std::get<X>(value);
    } else if (std::holds_alternative<VariablePtr>(value)) {
      // If the value is a VariablePtr, then return the value of the variable.
      auto variable = std::get<VariablePtr>(value);
      auto variableValue = variable->getValue();
      if (!variableValue) {
        throw tetrisched::exceptions::ExpressionSolutionException(
            "No solution was found for the variable name: " +
            variable->getName());
      }
      return variableValue.value();
    } else {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "XOrVariableT was resolved with an invalid type.");
    }
  }

  /// Checks if the class contains a Variable.
  bool isVariable() const { return std::holds_alternative<VariablePtr>(value); }

  /// Returns the (unresolved) value in the container.
  template <typename T>
  T get() const {
    return std::get<T>(value);
  }
};

/// A `ConstraintType` enumeration represents the types of constraints that we
/// allow the user to construct. These map to the types of constraints that the
/// underlying solver supports.
enum ConstraintType {
  /// LHS <= RHS
  CONSTR_LE,
  /// LHS = RHS
  CONSTR_EQ,
  /// LHS >= RHS
  CONSTR_GE,
};
using ConstraintType = enum ConstraintType;

template <typename T>
class ConstraintT {
 private:
  /// Used to generate unique IDs for each Constraint.
  static uint32_t constraintIdCounter;
  /// The ID of this constraint.
  uint32_t constraintId;
  /// The name of this constraint.
  std::string constraintName;
  /// The terms in this constraint.
  /// Note that a nullptr Variable indicates a constant term.
  std::vector<std::pair<T, std::shared_ptr<VariableT<T>>>> terms;
  /// The right hand side of this constraint.
  T rightHandSide;
  /// The operation between the terms and the right hand side.
  ConstraintType constraintType;

 public:
  /// Generate a new constraint with the given type and right hand side.
  ConstraintT(std::string constraintName, ConstraintType constraintType,
              T rightHandSide);

  /// Adds a term to the left-hand side constraint.
  void addTerm(std::pair<T, std::shared_ptr<VariableT<T>>> term);

  /// Adds a variable term to the left-hand side constraint.
  void addTerm(T coefficient, std::shared_ptr<VariableT<T>> variable);

  /// Adds a constant term to the left-hand side constraint.
  void addTerm(T constant);

  /// Adds a variable term to the left-hand side of the constraint
  /// with the coefficient of 1.
  void addTerm(std::shared_ptr<VariableT<T>> variable);

  /// Adds a term to the left-hand side of this constraint that can be
  /// resolved to either a constant or a variable.
  void addTerm(const XOrVariableT<T>& term);

  /// Adds a term to the left-hand side of this constraint with a given
  /// coefficient and a term that can either be a constant or a variable.
  void addTerm(T coefficient, const XOrVariableT<T>& term);

  /// Retrieve a string representation of this Constraint.
  std::string toString() const;

  /// Retrieve the name of this Constraint.
  std::string getName() const;

  /// Retrieve the ID of this Constraint.
  uint32_t getId() const;

  /// Retrieve the number of terms in this Constraint.
  size_t size() const;

  /// Annotate friend classes for Solvers so that they have access to internals.
  friend tetrisched::CPLEXSolver;
  friend tetrisched::GurobiSolver;
  friend tetrisched::GoogleCPSolver;
};

// Specialize the Constraint class for Integer.
template class ConstraintT<TETRISCHED_ILP_TYPE>;
using Constraint = ConstraintT<TETRISCHED_ILP_TYPE>;
// NOTE (Sukrit): I would have liked to kept this as a unique_ptr, since only
// the SolverModel should own the constraint. But, I have to move to a
// shared_ptr to satisfy Python's referencing mechanism. Inside C++, we should
// try to use std::move(...) as much as possible to maintain single ownership.
using ConstraintPtr = std::shared_ptr<Constraint>;

/// A `ObjectiveType` enumeration represents the types of objective functions
/// that we allow the user to construct.
enum ObjectiveType {
  /// Maximize the objective function.
  OBJ_MAXIMIZE,
  /// Minimize the objective function.
  OBJ_MINIMIZE,
};
using ObjectiveType = enum ObjectiveType;

template <typename T>
class ObjectiveFunctionT {
 private:
  /// The terms in this objective function.
  std::vector<std::pair<T, std::shared_ptr<VariableT<T>>>> terms;
  /// The type of the objective function.
  ObjectiveType objectiveType;

 public:
  ObjectiveFunctionT(const ObjectiveFunctionT& other) = default;

  /// Generate a new objective function with the given type.
  ObjectiveFunctionT(ObjectiveType objectiveType);

  /// Adds a term to the left-hand side constraint.
  void addTerm(T coefficient, std::shared_ptr<VariableT<T>> variable);

  // The objective is left hand side of the constraint
  std::shared_ptr<ConstraintT<T>> toConstraint(std::string constraintName,
                                               ConstraintType constraintType,
                                               T rightHandSide);

  /// Retrieve a string representation of this ObjectiveFunction.
  std::string toString() const;

  /// Retrieve the number of terms in this ObjectiveFunction.
  size_t size() const;

  /// Merges the current utility with the utility of the other objective,
  /// and returns a reference to the current utility.
  ObjectiveFunctionT<T>& operator+=(const ObjectiveFunctionT<T>& other);

  /// Retrieves the value of the utility of this ObjectiveFunction.
  T getValue() const;

  /// Annotate friend classes for Solvers so that they have access to internals.
  friend tetrisched::CPLEXSolver;
  friend tetrisched::GurobiSolver;
  friend tetrisched::GoogleCPSolver;
};

// Specialize the ObjectiveFunction class for Integer.
template class ObjectiveFunctionT<TETRISCHED_ILP_TYPE>;
using ObjectiveFunction = ObjectiveFunctionT<TETRISCHED_ILP_TYPE>;
// NOTE(Sukrit): Similar to above, try to enforce single ownership.
using ObjectiveFunctionPtr = std::shared_ptr<ObjectiveFunction>;

template <typename T>
class SolverModelT {
 private:
  /// The variables in this model.
  std::unordered_map<uint32_t, std::shared_ptr<VariableT<T>>> variables;
  /// The constraints in this model.
  std::unordered_map<uint32_t, std::shared_ptr<ConstraintT<T>>> constraints;
  /// The objective function in this model.
  std::shared_ptr<ObjectiveFunctionT<T>> objectiveFunction;

  /// Generate a new solver model.
  /// Construct a Solver to get an instance of the Model.
  SolverModelT() = default;

 public:
  /// Add a variable to the model.
  void addVariable(std::shared_ptr<VariableT<T>> variable);

  /// Add a constraint to the model.
  /// This method consumes the Constraint.
  void addConstraint(std::shared_ptr<ConstraintT<T>> constraint);

  /// Set the objective function for the model.
  /// This method consumes the ObjectiveFunction.
  void setObjectiveFunction(
      std::shared_ptr<ObjectiveFunctionT<T>> objectiveFunction);

  /// Retrieve a string representation of this SolverModel.
  std::string toString() const;

  /// Export the string representation of this SolverModel to a file.
  void exportModel(std::string filename) const;

  /// Retrieve the number of variables in this SolverModel.
  size_t numVariables() const;

  /// Retrieve the number of constraints in this SolverModel.
  size_t numConstraints() const;

  /// Retrieves the value of the objective function of this SolverModel.
  /// Throws an error if the model was not solved first.
  T getObjectiveValue() const;

  /// All the Solver implementations should be a friend of the SolverModel.
  /// This allows Solver implementations to construct the model to pass
  /// back to the user.
  friend tetrisched::CPLEXSolver;
  friend tetrisched::GurobiSolver;
  friend tetrisched::GoogleCPSolver;
};

// Specialize the SolverModel class for Integer.
template class SolverModelT<TETRISCHED_ILP_TYPE>;
using SolverModel = SolverModelT<TETRISCHED_ILP_TYPE>;
using SolverModelPtr = std::shared_ptr<SolverModel>;

}  // namespace tetrisched
#endif
