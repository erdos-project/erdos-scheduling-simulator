#include <pybind11/pybind11.h>

#include "Backends.cpp"
#include "SolverModel.cpp"

namespace py = pybind11;

PYBIND11_MODULE(tetrisched_py, tetrisched_m) {
  // Define the top-level module for the TetriSched Python API.
  tetrisched_m.doc() = "Python API for TetriSched.";

  // Implement the modelling basics.
  auto tetrisched_m_solver_model = tetrisched_m.def_submodule(
      "model", "Modelling primitives for the TetriSched Python API.");
  defineModelVariable(tetrisched_m_solver_model);
  defineModelConstraint(tetrisched_m_solver_model);
  defineModelObjective(tetrisched_m_solver_model);
  defineSolverModel(tetrisched_m_solver_model);

  // Implement the Solver backends.
  auto tetrisched_m_solver_backend = tetrisched_m.def_submodule(
      "backends", "Solver backends for the TetriSched Python API.");
  defineCPLEXBackend(tetrisched_m_solver_backend);
}
