#ifndef _TETRISCHED_CPLEX_SOLVER_HPP_
#define _TETRISCHED_CPLEX_SOLVER_HPP_

#include <ilcplex/ilocplex.h>

#include <variant>

#include "tetrisched/Solver.hpp"

namespace tetrisched {
class CPLEXSolver : public Solver {
 protected:
  using CPLEXVarType = std::variant<IloNumVar, IloBoolVar, IloIntVar>;

 private:
  /// The environment variable for this instance of CPLEX.
  IloEnv cplexEnv;
  /// The SolverModel instance associated with this CPLEXSolver.
  std::shared_ptr<SolverModel> solverModel;
  /// The CPLEX model associated with this CPLEXSolver.
  IloCplex cplexInstance;

  /// A map from the ID of the SolverModel variables to internal CPLEX
  /// variables.
  std::unordered_map<uint32_t, CPLEXVarType> cplexVariables;

  /// Translates the Variable into a CPLEX variable.
  CPLEXVarType translateVariable(const VariablePtr& variable) const;

  /// Translates the Constraint into a CPLEX expression.
  IloRange translateConstraint(const ConstraintPtr& constraint) const;

  /// Translates the ObjectiveFunction into a CPLEX expression.
  IloObjective translateObjectiveFunction(
      const ObjectiveFunctionPtr& objectiveFunction) const;

 public:
  /// Create a new CPLEXSolver.
  CPLEXSolver();

  /// Retrieve a pointer to the SolverModel.
  /// The SolverModel is the interface to define STRL expressions over.
  SolverModelPtr getModel() override;

  /// Translates the SolverModel into a CPLEX model.
  void translateModel() override;

  /// Export the constructed model to the given file.
  void exportModel(const std::string& fileName) override;
};
}  // namespace tetrisched
#endif  // _TETRISCHED_CPLEX_SOLVER_HPP_
