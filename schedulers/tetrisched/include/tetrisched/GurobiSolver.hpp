#ifndef _TETRISCHED_GUROBI_SOLVER_HPP_
#define _TETRISCHED_GUROBI_SOLVER_HPP_

#include "gurobi_c++.h"
#include "tetrisched/Solver.hpp"

namespace tetrisched {
class GurobiSolver : public Solver {
 private:
  std::unique_ptr<GRBEnv> gurobiEnv;
  SolverModelPtr solverModel;

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
};
}  // namespace tetrisched
#endif  // _TETRISCHED_GUROBI_SOLVER_HPP_

