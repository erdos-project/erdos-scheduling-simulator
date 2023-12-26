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
          "numVariables",
          [](const tetrisched::SolverSolution& solution) {
            return solution.numVariables;
          },
          "The number of variables in the model.")
      .def_property_readonly(
          "numCachedVariables",
          [](const tetrisched::SolverSolution& solution) {
            return solution.numCachedVariables;
          },
          "The number of variables that were cached before.")
      .def_property_readonly(
          "numUncachedVariables",
          [](const tetrisched::SolverSolution& solution) {
            return solution.numUncachedVariables;
          },
          "The number of variables that were not cached before.")
      .def_property_readonly(
          "numConstraints",
          [](const tetrisched::SolverSolution& solution) {
            return solution.numConstraints;
          },
          "The number of constraints in the model.")
      .def_property_readonly(
          "numDeactivatedConstraints",
          [](const tetrisched::SolverSolution& solution) {
            return solution.numDeactivatedConstraints;
          },
          "The number of constraints that were generated but deactivated.")
      .def_property_readonly(
          "numNonZeroCoefficients",
          [](const tetrisched::SolverSolution& solution) {
            return solution.numNonZeroCoefficients;
          },
          "The number of non-zero coefficients in the model.")
      .def_property_readonly(
          "numSolutions",
          [](const tetrisched::SolverSolution& solution) {
            return solution.numSolutions;
          },
          "The number of solutions found by the solver.")
      .def_property_readonly(
          "objectiveValue",
          [](const tetrisched::SolverSolution& solution) {
            return solution.objectiveValue;
          },
          "The objective value of the solution (if available).")
      .def_property_readonly(
          "objectiveValueBound",
          [](const tetrisched::SolverSolution& solution) {
            return solution.objectiveValueBound;
          },
          "The objective value of the bound retrieved from STRL (if "
          "available).")
      .def_property_readonly(
          "solverTimeMicroseconds",
          [](const tetrisched::SolverSolution& solution) {
            return solution.solverTimeMicroseconds;
          },
          "The time taken by the solver to solve the model (in microseconds).")
      .def("isValid", &tetrisched::SolverSolution::isValid,
           "Check if the solution was valid.")
      .def("__str__", [](const tetrisched::SolverSolution& solution) {
        std::string solutionTypeStr =
            "SolverSolution<type=" + solution.getSolutionTypeStr();
        if (solution.numVariables.has_value()) {
          solutionTypeStr +=
              ", numVariables=" + std::to_string(solution.numVariables.value());
        }
        if (solution.numCachedVariables.has_value()) {
          solutionTypeStr +=
              ", numCachedVariables=" +
              std::to_string(solution.numCachedVariables.value());
        }
        if (solution.numUncachedVariables.has_value()) {
          solutionTypeStr +=
              ", numUncachedVariables=" +
              std::to_string(solution.numUncachedVariables.value());
        }
        if (solution.numConstraints.has_value()) {
          solutionTypeStr += ", numConstraints=" +
                             std::to_string(solution.numConstraints.value());
        }
        if (solution.numDeactivatedConstraints.has_value()) {
          solutionTypeStr +=
              ", numDeactivatedConstraints=" +
              std::to_string(solution.numDeactivatedConstraints.value());
        }
        if (solution.numNonZeroCoefficients.has_value()) {
          solutionTypeStr +=
              ", numNonZeroCoefficients=" +
              std::to_string(solution.numNonZeroCoefficients.value());
        }
        if (solution.numSolutions.has_value()) {
          solutionTypeStr +=
              ", numSolutions=" + std::to_string(solution.numSolutions.value());
        }
        if (solution.objectiveValue.has_value()) {
          solutionTypeStr += ", objectiveValue=" +
                             std::to_string(solution.objectiveValue.value());
        }
        if (solution.objectiveValueBound.has_value()) {
          solutionTypeStr +=
              ", objectiveValueBound=" +
              std::to_string(solution.objectiveValueBound.value());
        }
        solutionTypeStr += ", solverTimeMicroseconds=" +
                           std::to_string(solution.solverTimeMicroseconds) +
                           ">";
        return solutionTypeStr;
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
