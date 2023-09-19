#ifndef _TETRISCHED_SOLVERMODEL_HPP_
#define _TETRISCHED_SOLVERMODEL_HPP_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tetrisched/Types.hpp"

namespace tetrisched {

/// A `VariableType` enumeration represents the types of variable that we allow
/// the user to construct. These map to the types of variables that the
/// underlying solver supports.
typedef enum VariableType {
  VAR_CONTINUOUS,
  VAR_INTEGER,
  VAR_INDICATOR
} VariableType;

/// A `VariableT` class represents a variable in the solver model. Note that
/// this is an internal representation of a VariableT and does not contain an
/// actual variable in a solver. This is used to program the STRL expression and
/// the translation to the underlying solution strategy is done at model
/// creation time.
template <typename T>
class VariableT {
 private:
  /// The type of the variable (as supported by the underlying solver).
  VariableType variableType;
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
};

// Specialize the VariableT class for Integer type.
template class VariableT<int32_t>;
typedef VariableT<int32_t> Variable;
typedef std::shared_ptr<Variable> VariablePtr;

/// A `ConstraintType` enumeration represents the types of constraints that we
/// allow the user to construct. These map to the types of constraints that the
/// underlying solver supports.
typedef enum ConstraintType {
  /// LHS <= RHS
  CONSTR_LE,
  /// LHS = RHS
  CONSTR_EQ,
  /// LHS >= RHS
  CONSTR_GE,
} ConstraintType;

template <typename T>
class ConstraintT {
 private:
  /// The terms in this constraint.
  std::vector<std::pair<T, std::shared_ptr<VariableT<T>>>> terms;
  /// The right hand side of this constraint.
  T rightHandSide;
  /// The operation between the terms and the right hand side.
  ConstraintType constraintType;

 public:
  /// Generate a new constraint with the given type and right hand side.
  ConstraintT(ConstraintType constraintType, T rightHandSide);

  /// Adds a term to the left-hand side constraint.
  void addTerm(std::pair<T, std::shared_ptr<VariableT<T>>> term);

  /// Adds a term to the left-hand side constraint.
  void addTerm(T coefficient, std::shared_ptr<VariableT<T>> variable);

  /// Retrieve a string representation of this Constraint.
  std::string toString() const;

  /// Retrieve the number of terms in this Constraint.
  size_t size() const;
};

// Specialize the Constraint class for Integer.
template class ConstraintT<int32_t>;
typedef ConstraintT<int32_t> Constraint;
typedef std::unique_ptr<Constraint> ConstraintPtr;

/// A `ObjectiveType` enumeration represents the types of objective functions
/// that we allow the user to construct.
typedef enum ObjectiveType {
  /// Maximize the objective function.
  OBJ_MAXIMIZE,
  /// Minimize the objective function.
  OBJ_MINIMIZE,
} ObjectiveType;

template <typename T>
class ObjectiveFunctionT {
 private:
  /// The terms in this objective function.
  std::vector<std::pair<T, std::shared_ptr<VariableT<T>>>> terms;
  /// The type of the objective function.
  ObjectiveType objectiveType;

 public:
  /// Generate a new objective function with the given type.
  ObjectiveFunctionT(ObjectiveType objectiveType);

  /// Adds a term to the left-hand side constraint.
  void addTerm(T coefficient, std::shared_ptr<VariableT<T>> variable);

  /// Retrieve a string representation of this ObjectiveFunction.
  std::string toString() const;

  /// Retrieve the number of terms in this ObjectiveFunction.
  size_t size() const;
};

// Specialize the ObjectiveFunction class for Integer.
template class ObjectiveFunctionT<int32_t>;
typedef ObjectiveFunctionT<int32_t> ObjectiveFunction;
typedef std::unique_ptr<ObjectiveFunction> ObjectiveFunctionPtr;

template <typename T>
class SolverModelT {
 private:
  /// The variables in this model.
  std::vector<std::shared_ptr<VariableT<T>>> variables;
  /// The constraints in this model.
  std::vector<std::unique_ptr<ConstraintT<T>>> constraints;
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
  void addConstraint(std::unique_ptr<ConstraintT<T>> constraint);

  /// Set the objective function for the model.
  /// This method consumes the ObjectiveFunction.
  void setObjectiveFunction(
      std::unique_ptr<ObjectiveFunctionT<T>> objectiveFunction);

  /// Retrieve a string representation of this SolverModel.
  std::string toString() const;

  /// Retrieve the number of variables in this SolverModel.
  size_t numVariables() const;

  /// Retrieve the number of constraints in this SolverModel.
  size_t numConstraints() const;

  /// All the Solver implementations should be a friend of the SolverModel.
  /// This allows Solver implementations to construct the model to pass
  /// back to the user.
  friend tetrisched::CPLEXSolver;
};

// Specialize the SolverModel class for Integer.
template class SolverModelT<int32_t>;
typedef SolverModelT<int32_t> SolverModel;
typedef std::shared_ptr<SolverModel> SolverModelPtr;

}  // namespace tetrisched

#endif
