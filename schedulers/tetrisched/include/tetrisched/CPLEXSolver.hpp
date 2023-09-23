#ifndef _TETRISCHED_CPLEX_SOLVER_HPP_
#define _TETRISCHED_CPLEX_SOLVER_HPP_

#include <ilcplex/ilocplex.h>

#include "tetrisched/Solver.hpp"

namespace tetrisched {
class CPLEXSolver : public Solver {
 private:
  /// The environment variable for this instance of CPLEX.
  IloEnv cplexEnv;

 public:
  /// Create a new CPLEXSolver.
  CPLEXSolver() = default;

  /// Retrieve a pointer to the SolverModel.
  /// The SolverModel is the interface to define STRL expressions over.
  SolverModelPtr getModel() override;

  /// Export the constructed model to the given file.
  void exportModel(const std::string& fileName) override;
};
}  // namespace tetrisched
#endif  // _TETRISCHED_CPLEX_SOLVER_HPP_
