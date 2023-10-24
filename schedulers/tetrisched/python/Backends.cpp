#include <pybind11/pybind11.h>

#include "tetrisched/Solver.hpp"

namespace py = pybind11;

void defineSolverSolution(py::module_& tetrisched_m) {
  // Define the solution type enum.
  py::enum_<tetrisched::SolutionType>(tetrisched_m, "SolutionType")
      .value("FEASIBLE", tetrisched::SolutionType::FEASIBLE)
      .value("OPTIMAL", tetrisched::SolutionType::OPTIMAL)
      .value("INFEASIBLE", tetrisched::SolutionType::INFEASIBLE)
      .value("UNBOUNDED", tetrisched::SolutionType::UNBOUNDED)
      .value("UNKNOWN", tetrisched::SolutionType::UNKNOWN);

  // Define the backend type enum.
  py::enum_<tetrisched::SolverBackendType>(tetrisched_m, "SolverBackendType")
#ifdef _TETRISCHED_WITH_CPLEX_
      .value("CPLEX", tetrisched::SolverBackendType::CPLEX)
#endif
#ifdef _TETRISCHED_WITH_GUROBI_
      .value("GUROBI", tetrisched::SolverBackendType::GUROBI)
#endif
#ifdef _TETRISCHED_WITH_OR_TOOLS_
      .value("GOOGLE_CP", tetrisched::SolverBackendType::GOOGLE_CP)
#endif
      ;

  // Define the SolverSolution structure.
  py::class_<tetrisched::SolverSolution, tetrisched::SolverSolutionPtr>(
      tetrisched_m, "SolverSolution")
      .def_property_readonly("solutionType",
                             [](const tetrisched::SolverSolution& solution) {
                               return solution.solutionType;
                             })
      .def_property_readonly("objectiveValue",
                             [](const tetrisched::SolverSolution& solution) {
                               return solution.objectiveValue;
                             })
      .def_property_readonly("solverTimeMicroseconds",
                             [](const tetrisched::SolverSolution& solution) {
                               return solution.solverTimeMicroseconds;
                             })
      .def("__str__", [](const tetrisched::SolverSolution& solution) {
        return "SolverSolution<type=" + solution.getSolutionTypeStr() +
               ", objectiveValue=" +
               (solution.objectiveValue.has_value()
                    ? std::to_string(solution.objectiveValue.value())
                    : "None") +
               ", solverTimeMicroseconds=" +
               std::to_string(solution.solverTimeMicroseconds) + ">";
      });
}

#ifdef _TETRISCHED_WITH_CPLEX_
#include "tetrisched/CPLEXSolver.hpp"

void defineCPLEXBackend(py::module_& tetrisched_m) {
  py::class_<tetrisched::CPLEXSolver>(tetrisched_m, "CPLEXSolver")
      .def(py::init<>())
      .def("getModel", &tetrisched::CPLEXSolver::getModel)
      .def("translateModel", &tetrisched::CPLEXSolver::translateModel)
      .def("exportModel", &tetrisched::CPLEXSolver::exportModel)
      .def("solveModel", &tetrisched::CPLEXSolver::solveModel);
}
#endif  //_TETRISCHED_WITH_CPLEX_

#ifdef _TETRISCHED_WITH_GUROBI_
#include "tetrisched/GurobiSolver.hpp"

void defineGurobiBackend(py::module_& tetrisched_m) {
  py::class_<tetrisched::GurobiSolver>(tetrisched_m, "GurobiSolver")
      .def(py::init<>())
      .def("getModel", &tetrisched::GurobiSolver::getModel)
      .def("translateModel", &tetrisched::GurobiSolver::translateModel)
      .def("exportModel", &tetrisched::GurobiSolver::exportModel)
      .def("solveModel", &tetrisched::GurobiSolver::solveModel);
}
#endif  //_TETRISCHED_WITH_GUROBI_
