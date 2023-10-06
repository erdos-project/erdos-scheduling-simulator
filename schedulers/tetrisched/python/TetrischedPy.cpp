#include <pybind11/pybind11.h>

#include "Backends.cpp"
#include "Expressions.cpp"
#include "SolverModel.cpp"
#include "tetrisched/Scheduler.hpp"
#include "tetrisched/Worker.hpp"

namespace py = pybind11;

void defineBasicTypes(py::module_& tetrisched_m) {
  // Define the Worker type.
  py::class_<tetrisched::Worker, tetrisched::WorkerPtr>(tetrisched_m, "Worker")
      .def(py::init([](uint32_t workerId, std::string workerName) {
             return std::make_shared<tetrisched::Worker>(workerId, workerName);
           }),
           "Initializes the Worker with the given ID and name.")
      .def_property_readonly("id", &tetrisched::Worker::getWorkerId,
                             "The ID of this Worker.")
      .def_property_readonly("name", &tetrisched::Worker::getWorkerName,
                             "The name of this Worker.");

  // Define the Partition type.
  py::class_<tetrisched::Partition, tetrisched::PartitionPtr>(tetrisched_m,
                                                              "Partition")
      .def(py::init([]() { return std::make_shared<tetrisched::Partition>(); }),
           "Initializes an empty Partition.")
      .def("addWorker", &tetrisched::Partition::addWorker,
           "Adds a Worker to this Partition.")
      .def("__len__", &tetrisched::Partition::size,
           "Returns the number of Workers in this Partition.")
      .def_property_readonly("id", &tetrisched::Partition::getPartitionId,
                             "The ID of this Partition.");

  // Define the Partitions.
  py::class_<tetrisched::Partitions>(tetrisched_m, "Partitions")
      .def(py::init<>(), "Initializes an empty Partitions.")
      .def("addPartition", &tetrisched::Partitions::addPartition,
           "Adds a Partition to this Partitions.")
      .def("getPartitions", &tetrisched::Partitions::getPartitions,
           "Returns the Partitions in this Partitions.")
      .def("__len__", &tetrisched::Partitions::size,
           "Returns the number of Partitions in this Partitions.");
}

/// Define the Scheduler interface.
void defineScheduler(py::module_& tetrisched_m) {
  py::class_<tetrisched::Scheduler>(tetrisched_m, "Scheduler")
      .def(py::init<tetrisched::Time>(), "Initializes the Scheduler.")
      .def("registerSTRL", &tetrisched::Scheduler::registerSTRL,
           "Registers the STRL expression for the scheduler to schedule from.")
      .def("schedule", &tetrisched::Scheduler::schedule,
           "Invokes the solver to schedule the registered STRL expression.");
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
  auto tetrisched_m_solver_backend = tetrisched_m.def_submodule(
      "backends", "Solver backends for the TetriSched Python API.");
#ifdef _TETRISCHED_WITH_CPLEX_
  defineCPLEXBackend(tetrisched_m_solver_backend);
#endif //_TETRISCHED_WITH_CPLEX_
}
