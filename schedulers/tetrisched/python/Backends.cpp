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
      .value("UNKNOWN", tetrisched::SolutionType::UNKNOWN)
      .value("NO_SOLUTION", tetrisched::SolutionType::NO_SOLUTION);

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
      .def_property_readonly(
          "solutionType",
          [](const tetrisched::SolverSolution& solution) {
            return solution.solutionType;
          },
          "The type of solution returned by the solver.")
      .def_property_readonly(
          "objectiveValue",
          [](const tetrisched::SolverSolution& solution) {
            return solution.objectiveValue;
          },
          "The objective value of the solution (if available).")
      .def_property_readonly(
          "solverTimeMicroseconds",
          [](const tetrisched::SolverSolution& solution) {
            return solution.solverTimeMicroseconds;
          },
          "The time taken by the solver to solve the model (in microseconds).")
      .def("isValid", &tetrisched::SolverSolution::isValid,
           "Check if the solution was valid.")
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
      .def("getModel", &tetrisched::CPLEXSolver::getModel,
           "Returns the underlying SolverModel abstraction used by this "
           "instance of CPLEXSolver.")
      .def("translateModel", &tetrisched::CPLEXSolver::translateModel,
           "Translates the underlying SolverModel to a CPLEX model instance.")
      .def("exportModel", &tetrisched::CPLEXSolver::exportModel,
           "Exports the converted CPLEX model to an LP file. \n"
           "\nArgs:\n"
           "  filename (str): The filename to export the model to.",
           py::arg("filename"))
      .def("solveModel", &tetrisched::CPLEXSolver::solveModel,
           "Solves the CPLEX model.\n"
           "\nReturns:\n"
           "  SolverSolution: The characteristics of the solution.");
}
#endif  //_TETRISCHED_WITH_CPLEX_

#ifdef _TETRISCHED_WITH_GUROBI_
#include "tetrisched/GurobiSolver.hpp"

void defineGurobiBackend(py::module_& tetrisched_m) {
  py::class_<tetrisched::GurobiSolver>(tetrisched_m, "GurobiSolver")
      .def(py::init<>())
      .def("getModel", &tetrisched::GurobiSolver::getModel,
           "Returns the underlying SolverModel abstraction used by this "
           "instance of GurobiSolver.")
      .def("translateModel", &tetrisched::GurobiSolver::translateModel,
           "Translates the underlying SolverModel to a Gurobi model instance.")
      .def("exportModel", &tetrisched::GurobiSolver::exportModel,
           "Exports the converted Gurobi model to an LP file. \n"
           "\nArgs: \n"
           "  filename (str): The filename to export the model to.",
           py::arg("filename"))
      .def("solveModel", &tetrisched::GurobiSolver::solveModel,
           "Solves the Gurobi model.\n"
           "\nReturns:\n"
           "  SolverSolution: The characteristics of the solution.");
}
#endif  //_TETRISCHED_WITH_GUROBI_
