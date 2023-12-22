#ifndef _TETRISCHED_GUROBI_SOLVER_HPP_
#define _TETRISCHED_GUROBI_SOLVER_HPP_

#include "gurobi_c++.h"
#include "tetrisched/Solver.hpp"

namespace tetrisched {

class GurobiSolver : public Solver {
 private:
  /// A structure representing the parameters that the interrupt callback
  /// can use to determine when to interrupt the computation.
  struct GurobiInterruptParams {
    /// The time limit for the optimization (in milliseconds).
    std::optional<Time> timeLimitMs;
    /// The upper bound of the utility (if available).
    std::optional<TETRISCHED_ILP_TYPE> utilityUpperBound;
  };

  /// Callback class for Gurobi to use to interrupt the optimization.
  class GurobiInterruptOptimizationCallback : public GRBCallback {
   private:
    /// The interrupt parameters for the callback.
    GurobiInterruptParams params;
    /// The start time of the optimization.
    std::chrono::steady_clock::time_point startTime;

   public:
    /// Create a new GurobiInterruptOptimizationCallback.
    GurobiInterruptOptimizationCallback(GurobiInterruptParams params);

   protected:
    /// Callback function for Gurobi to call when it wants to interrupt the
    /// optimization.
    void callback();
  };

  /// The environment variable for this instance of Gurobi.
  std::unique_ptr<GRBEnv> gurobiEnv;
  /// The model being used by this solver.
  std::unique_ptr<GRBModel> gurobiModel;
  /// The SolverModel instance associated with this GurobiSolver.
  SolverModelPtr solverModel;
  /// The name of the log file for the Gurobi solving.
  std::string logFileName;
  /// The interrupt parameters for the current iteration of the optimization.
  GurobiInterruptParams interruptParams;
  /// The number of cached variables for the current model.
  mutable uint64_t numCachedVariables;
  /// The number of uncached variables for the current model.
  mutable uint64_t numUncachedVariables;

  /// Set the defaults for parameters on the model.
  void setParameters(GRBModel& gurobiModel);

  /// Translate the variable to a Gurobi variable.
  GRBVar translateVariable(GRBModel& gurobiModel,
                           const VariablePtr& variable) const;

  /// Translate the Constraint into a Gurobi expression.
  GRBConstr translateConstraint(GRBModel& gurobiModel,
                                const ConstraintPtr& constraint) const;

  /// Translate the ObjectiveFunction into a Gurobi expression.
  GRBLinExpr translateObjectiveFunction(
      GRBModel& gurobiModel,
      const ObjectiveFunctionPtr& objectiveFunction) const;

 public:
  /// Create a new GurobiSolver.
  GurobiSolver();

  /// Retrieve a pointer to the SolverModel.
  /// The SolverModel is the interface to define STRL expressions over.
  SolverModelPtr getModel() override;

  /// Replace the SolverModel in this instance with the given model.
  /// This may be used to switch backends when a model is already constructed.
  void setModel(SolverModelPtr model) override;

  /// Translates the SolverModel into a Gurobi model.
  void translateModel() override;

  /// Export the constructed model to the given file.
  void exportModel(const std::string& fileName) override;

  /// Set the log file for the solver to output its log to.
  void setLogFile(const std::string& fileName) override;

  /// Solve the constructed model.
  SolverSolutionPtr solveModel() override;

  /// Get the name of the Solver.
  std::string getName() const override { return "GurobiSolver"; }

  /// Get the backend type of the Solver.
  SolverBackendType getBackendType() const override {
    return SolverBackendType::GUROBI;
  }
};
}  // namespace tetrisched
#endif  // _TETRISCHED_GUROBI_SOLVER_HPP_
