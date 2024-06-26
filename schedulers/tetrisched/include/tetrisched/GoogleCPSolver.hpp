#ifndef _TETRISCHED_GOOGLE_CP_SOLVER_HPP_
#define _TETRISCHED_GOOGLE_CP_SOLVER_HPP_

#include <variant>

#include "ortools/sat/cp_model.h"
#include "tetrisched/Solver.hpp"

namespace tetrisched {
// Import the relevant names from the ORTools namespace.
using operations_research::sat::BoolVar;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;

class GoogleCPSolver : public Solver {
  using GoogleCPVarType = std::variant<IntVar, BoolVar>;

 private:
  /// The SolverModelPtr associated with this GoogleCPSolver.
  SolverModelPtr solverModel;
  /// The ORTools model associated with this GoogleCPSolver.
  std::unique_ptr<CpModelBuilder> cpModel;
  /// A map from the Variable ID to the ORTools variable.
  std::unordered_map<uint32_t, GoogleCPVarType> cpVariables;

  /// Translates the VariablePtr into an IntVar / BoolVar.
  GoogleCPVarType translateVariable(const VariablePtr& variable) const;
  /// Translates the ConstraintPtr into a Constraint and adds it to the model.
  operations_research::sat::Constraint translateConstraint(
      const ConstraintPtr& constraint);
  /// Translates the ObjectiveFunctionPtr into an Expression in Google OR-Tools.
  LinearExpr translateObjectiveFunction(
      const ObjectiveFunctionPtr& objectiveFunction) const;

 public:
  /// Create a new CP-SAT solver.
  GoogleCPSolver();

  /// Retrieve a pointer to the SolverModel.
  SolverModelPtr getModel() override;

  /// Replace the SolverModel in this instance with the given model.
  /// This may be used to switch backends when a model is already constructed.
  void setModel(SolverModelPtr model) override;

  /// Translates the SolverModel into a CP-SAT model.
  void translateModel(SolverConfigPtr solverConfig) override;

  /// Export the constructed model to the given file.
  void exportModel(const std::string& fileName) override;

  /// Set the log file for the solver to output its log to.
  void setLogFile(const std::string& fileName) override;

  /// Solve the constructed model.
  SolverSolutionPtr solveModel() override;

  /// Get the name of the Solver.
  std::string getName() const override { return "GoogleCPSolver"; }

  /// Get the backend type of the Solver.
  SolverBackendType getBackendType() const override {
    return SolverBackendType::GOOGLE_CP;
  }
};
}  // namespace tetrisched
#endif  // _TETRISCHED_GOOGLE_CP_SOLVER_HPP_
