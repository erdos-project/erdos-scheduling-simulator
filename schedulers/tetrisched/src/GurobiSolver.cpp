#include "tetrisched/GurobiSolver.hpp"

namespace tetrisched {
GurobiSolver::GurobiSolver() : gurobiEnv(new GRBEnv()) {}

SolverModelPtr GurobiSolver::getModel() {
  if (!solverModel) {
    solverModel = std::shared_ptr<SolverModel>(new SolverModel());
  }
  return solverModel;
}

void GurobiSolver::translateModel() {}

void GurobiSolver::exportModel(const std::string& fileName) {}
}  // namespace tetrisched

