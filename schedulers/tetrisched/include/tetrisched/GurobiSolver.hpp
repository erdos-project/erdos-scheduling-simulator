#ifndef _TETRISCHED_GUROBI_SOLVER_HPP_
#define _TETRISCHED_GUROBI_SOLVER_HPP_

#include "gurobi_c++.h"
#include "tetrisched/Solver.hpp"

namespace tetrisched {
class GurobiSolver : public Solver {
 private:
  /// The environment variable for this instance of Gurobi.
  std::unique_ptr<GRBEnv> gurobiEnv;
  /// The model being used by this solver.
  std::unique_ptr<GRBModel> gurobiModel;
  /// The SolverModel instance associated with this GurobiSolver.
  SolverModelPtr solverModel;
  /// A map from the ID of the SolverModel variables to internal Gurobi
  /// variables.
  std::unordered_map<uint32_t, GRBVar> gurobiVariables;

  /// Translate the variable to a Gurobi variable.
  GRBVar translateVariable(GRBModel& gurobiModel,
                           const VariablePtr& variable) const;

  /// Translate the Constraint into a Gurobi expression.
  GRBConstr translateConstraint(GRBModel& gurobiModel,
                                const ConstraintPtr& constraint) const;

  /// Translate teh ObjectiveFunction into a Gurobi expression.
  GRBLinExpr translateObjectiveFunction(
      GRBModel& gurobiModel,
      const ObjectiveFunctionPtr& objectiveFunction) const;

 public:
  /// Create a new GurobiSolver.
  GurobiSolver();

  /// Retrieve a pointer to the SolverModel.
  /// The SolverModel is the interface to define STRL expressions over.
  SolverModelPtr getModel() override;

  /// Translates the SolverModel into a Gurobi model.
  void translateModel() override;

  /// Export the constructed model to the given file.
  void exportModel(const std::string& fileName) override;

  /// Solve the constructed model.
  void solveModel() override;
};
}  // namespace tetrisched
#endif  // _TETRISCHED_GUROBI_SOLVER_HPP_
