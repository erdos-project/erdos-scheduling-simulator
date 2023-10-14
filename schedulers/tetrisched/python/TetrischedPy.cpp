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
      .def(py::init<uint32_t, std::string>(), "Initializes an empty Partition.")
      .def(py::init<uint32_t, std::string, size_t>(),
           "Initializes a Partition with the given quantity.")
      .def("__add__", &tetrisched::Partition::operator+=,
           "Adds the given quantity to this Partition.")
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
           "Adds a Partition to this Partitions.")
      .def("getPartitions", &tetrisched::Partitions::getPartitions,
           "Returns the Partitions in this Partitions.")
      .def("__getitem__", &tetrisched::Partitions::getPartition,
           "Returns the Partition with the given ID (if exists).")
      .def("__len__", &tetrisched::Partitions::size,
           "Returns the number of Partitions in this Partitions.");
}

/// Define the Scheduler interface.
void defineScheduler(py::module_& tetrisched_m) {
  py::class_<tetrisched::Scheduler>(tetrisched_m, "Scheduler")
      .def(py::init<tetrisched::Time, tetrisched::SolverBackendType>(),
           "Initializes the Scheduler with the given backend.")
      .def("registerSTRL", &tetrisched::Scheduler::registerSTRL,
           "Registers the STRL expression for the scheduler to schedule from.")
      .def("schedule", &tetrisched::Scheduler::schedule,
           "Invokes the solver to schedule the registered STRL expression.")
      .def("getLastSolverSolution",
           &tetrisched::Scheduler::getLastSolverSolution,
           "Retrieve the solution from the last invocation of the solver.");
}

/// Define the Solver backends.
void defineSolverBackends(py::module_& tetrisched_m) {
  auto tetrisched_m_solver_backend = tetrisched_m.def_submodule(
      "backends", "Solver backends for the TetriSched Python API.");
  defineSolverSolution(tetrisched_m_solver_backend);
#ifdef _TETRISCHED_WITH_CPLEX_
  defineCPLEXBackend(tetrisched_m_solver_backend);
#endif  //_TETRISCHED_WITH_CPLEX_
#ifdef _TETRISCHED_WITH_GUROBI_
  defineGurobiBackend(tetrisched_m_solver_backend);
#endif  //_TETRISCHED_WITH_GUROBI_
}

PYBIND11_MODULE(tetrisched_py, tetrisched_m) {
  // Define the top-level module for the TetriSched Python API.
  tetrisched_m.doc() = "Python API for TetriSched.";

  // Define the top-level basic types and the Scheduler itself.
  defineBasicTypes(tetrisched_m);
  defineScheduler(tetrisched_m);

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
  defineSolverBackends(tetrisched_m);
}
