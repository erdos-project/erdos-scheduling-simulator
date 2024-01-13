#include <pybind11/pybind11.h>

#include "Backends.cpp"
#include "Expressions.cpp"
#include "SolverModel.cpp"
#include "tetrisched/Scheduler.hpp"

namespace py = pybind11;

void defineBasicTypes(py::module_& tetrisched_m) {
  // Define the Partition type.
  py::class_<tetrisched::Partition, tetrisched::PartitionPtr>(
      tetrisched_m, "Partition", py::dynamic_attr())
      .def(py::init<uint32_t, std::string>(),
           "Initializes an empty Partition.\n"
           "\nArgs:\n"
           "  partitionId (int): The ID of this Partition.\n"
           "  partitionName (str): The name of this Partition.",
           py::arg("partitionId"), py::arg("partitionName"))
      .def(py::init<uint32_t, std::string, size_t>(),
           "Initializes a Partition with the given quantity.\n"
           "\nArgs:\n"
           "  partitionId (int): The ID of this Partition.\n"
           "  partitionName (str): The name of this Partition.\n"
           "  quantity (int): The quantity of this Partition.",
           py::arg("partitionId"), py::arg("partitionName"),
           py::arg("quantity"))
      .def("__add__", &tetrisched::Partition::operator+=,
           "Adds the given quantity to this Partition.", py::arg("quantity"))
      .def("__len__", &tetrisched::Partition::getQuantity,
           "Returns the number of Workers in this Partition.")
      .def_property_readonly("id", &tetrisched::Partition::getPartitionId,
                             "The ID of this Partition.")
      .def_property_readonly("name", &tetrisched::Partition::getPartitionName,
                             "The name of this Partition.");

  // Define the Partitions.
  py::class_<tetrisched::Partitions>(tetrisched_m, "Partitions",
                                     py::dynamic_attr())
      .def(py::init<>(), "Initializes an empty Partitions.")
      .def("addPartition", &tetrisched::Partitions::addPartition,
           "Adds a Partition to this Partitions.\n"
           "\nArgs:\n"
           "  partition (Partition): The Partition to add to this Partitions.",
           py::arg("partition"))
      .def("getPartitions", &tetrisched::Partitions::getPartitions,
           "Returns the Partitions in this Partitions.")
      .def("__getitem__", &tetrisched::Partitions::getPartition,
           "Returns the Partition with the given ID (if exists).",
           py::arg("id"))
      .def("__len__", &tetrisched::Partitions::size,
           "Returns the number of Partitions in this Partitions.");
}

/// Define the Scheduler interface.
void defineScheduler(py::module_& tetrisched_m) {
  py::class_<tetrisched::Scheduler>(tetrisched_m, "Scheduler")
      .def(py::init<tetrisched::Time, tetrisched::SolverBackendType,
                    std::string, bool, tetrisched::Time, float, bool, tetrisched::Time>(),
           "Initializes the Scheduler with the given backend.\n"
           "\nArgs:\n"
           "  discretization (int): The time discretization to use for the "
           "scheduler.\n"
           "  solverBackend (SolverBackendType): The solver backend to use for "
           "the scheduler.\n"
           "  logDir (str): The directory where the logs are to be output.\n"
           "  enableDynamicDiscretization (bool): Whether to enable dynamic "
           "discretization.\n"
           "  maxDiscretization (int): The maximum discretization to use for "
           "dynamic discretization.\n"
           "  maxOccupancyThreshold (float): The maximum occupancy threshold "
           "to use for dynamic discretization.\n",
            "  finerDiscretizationAtPrevSolution (bool): Enables finer discretization "
           "At previous solution.\n",
           "  finerDiscretizationWindow (int): The discretization around prev solution until which the discretization would be 1 ",
           py::arg("discretization"), py::arg("solverBackend"),
           py::arg("logDir") = "./",
           py::arg("enableDynamicDiscretization") = false,
           py::arg("maxDiscretization") = 5,
           py::arg("maxOccupancyThreshold") = 0.8,
           py::arg("finerDiscretizationAtPrevSolution") = false,
           py::arg("finerDiscretizationWindow") = 5)
      .def(
          "registerSTRL", &tetrisched::Scheduler::registerSTRL,
          "Registers the STRL expression for the scheduler to schedule from.\n"
          "\nArgs:\n"
          "  expression (Expression): The STRL expression to register.\n"
          "  availablePartitions (Partitions): The available Partitions to "
          "schedule on.\n"
          "  currentTime (int): The current time.\n"
          "  optimize (bool): Whether to optimize the schedule.\n"
          "  timeRangeToGranularities (list): The time ranges to granularities "
          "to use for dynamic discretization.",
          py::arg("expression"), py::arg("availablePartitions"),
          py::arg("currentTime"), py::arg("optimize") = false,
          py::arg("timeRangeToGranularities") =
              std::vector<std::pair<tetrisched::TimeRange, tetrisched::Time>>())
      .def("schedule", &tetrisched::Scheduler::schedule,
           "Invokes the solver to schedule the registered STRL expression.\n"
           "\nArgs:\n"
           "  currentTime (int): The current time.",
           py::arg("currentTime"))
      .def("getLastSolverSolution",
           &tetrisched::Scheduler::getLastSolverSolution,
           "Retrieve the solution from the last invocation of the solver.")
      .def("exportLastSolverModel",
           &tetrisched::Scheduler::exportLastSolverModel,
           "Exports the model from the last invocation of the solver.\n"
           "\nArgs:\n"
           "  fileName (str): The filename to export the model to.",
           py::arg("fileName"));
}

/// Define the Solver backends.
void defineSolverBackends(py::module_& tetrisched_m) {
  defineSolverSolution(tetrisched_m);
#ifdef _TETRISCHED_WITH_CPLEX_
  defineCPLEXBackend(tetrisched_m);
#endif  //_TETRISCHED_WITH_CPLEX_
#ifdef _TETRISCHED_WITH_GUROBI_
  defineGurobiBackend(tetrisched_m);
#endif  //_TETRISCHED_WITH_GUROBI_
}

PYBIND11_MODULE(tetrisched_py, tetrisched_m) {
  // Define the top-level module for the TetriSched Python API.
  tetrisched_m.doc() = "Python API for TetriSched.";

  // Define the top-level basic types.
  defineBasicTypes(tetrisched_m);

  // Implement bindings for STRL.
  auto tetrisched_m_strl = tetrisched_m.def_submodule(
      "strl", "STRL primitives for the TetriSched Python API.");
  defineSTRLExpressions(tetrisched_m_strl);

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
  defineSolverBackends(tetrisched_m_solver_backend);

  // Finally, define the scheduler.
  defineScheduler(tetrisched_m);
}
