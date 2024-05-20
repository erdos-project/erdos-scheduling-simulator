#include <pybind11/pybind11.h>

#include "Backends.cpp"
#include "Expressions.cpp"
#include "SolverModel.cpp"
#include "tetrisched/Scheduler.hpp"

namespace py = pybind11;

void defineBasicTypes(py::module_& tetrisched_m) {
  // define OPT pass enum
  py::enum_<tetrisched::OptimizationPassCategory>(tetrisched_m, "OptimizationPassCategory")
      .value("CRITICAL_PATH_PASS", tetrisched::OptimizationPassCategory::CRITICAL_PATH_PASS)
      .value("DYNAMIC_DISCRETIZATION_PASS", tetrisched::OptimizationPassCategory::DYNAMIC_DISCRETIZATION_PASS)
      .value("CAPACITY_CONSTRAINT_PURGE_PASS", tetrisched::OptimizationPassCategory::CAPACITY_CONSTRAINT_PURGE_PASS);

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
  // Define the Config for Optimization Passes
  py::class_<tetrisched::OptimizationPassConfig, tetrisched::OptimizationPassConfigPtr>(tetrisched_m, "OptimizationPassConfig")
      .def(py::init<>(), "Initializes an empty OptimizationPassConfig.")
      .def_readwrite("minDiscretization", &tetrisched::OptimizationPassConfig::minDiscretization,
                     "The minimum discretization for Dynamic Discretiazation OPT pass.")
      .def_readwrite("maxDiscretization", &tetrisched::OptimizationPassConfig::maxDiscretization,
                     "The maximum discretization for Dynamic Discretiazation OPT pass.")
      .def_readwrite("maxOccupancyThreshold", &tetrisched::OptimizationPassConfig::maxOccupancyThreshold,
                     "The max occupancy threshold beyond which dynamic discretization is always min.")
      .def_readwrite("finerDiscretizationAtPrevSolution", &tetrisched::OptimizationPassConfig::finerDiscretizationAtPrevSolution,
                     "Whether to enabled finer discretization at solved solutions.")
      .def_readwrite("finerDiscretizationWindow", &tetrisched::OptimizationPassConfig::finerDiscretizationWindow,
                     "The window upto which finer discretization should be enabled around previously solved solutions.")
      .def("toString", &tetrisched::OptimizationPassConfig::toString, "Print String Representation");

  // Define the Config for the Scheduler.
  py::class_<tetrisched::SchedulerConfig, tetrisched::SchedulerConfigPtr>(
      tetrisched_m, "SchedulerConfig")
      .def(py::init<>(), "Initializes an empty SchedulerConfig.")
      .def_readwrite("numThreads", &tetrisched::SchedulerConfig::numThreads,
                     "The number of threads to use for the solver.")
      .def_readwrite("totalSolverTimeMs",
                     &tetrisched::SchedulerConfig::totalSolverTimeMs,
                     "The total solver time to use for the solver.")
      .def_readwrite("newSolutionTimeMs",
                     &tetrisched::SchedulerConfig::newSolutionTimeMs,
                     "The new solution time to use for the solver.");

  // Define the interface to the Scheduler.
  py::class_<tetrisched::Scheduler>(tetrisched_m, "Scheduler")
      .def(
          py::init<tetrisched::Time, tetrisched::SolverBackendType, std::string, tetrisched::OptimizationPassConfigPtr>(),
          "Initializes the Scheduler with the given backend.\n"
          "\nArgs:\n"
          "  discretization (int): The time discretization to use for the "
          "scheduler.\n"
          "  solverBackend (SolverBackendType): The solver backend to use for "
          "the scheduler.\n"
          "  logDir (str): The directory where the logs are to be output.\n"
         "OptimizationPass Config for OPT passes",
          py::arg("discretization"), py::arg("solverBackend"),
          py::arg("logDir") = "./",
          py::arg("optConfig") = nullptr)
      .def(
          "registerSTRL", &tetrisched::Scheduler::registerSTRL,
          "Registers the STRL expression for the scheduler to schedule from.\n"
          "\nArgs:\n"
          "  expression (Expression): The STRL expression to register.\n"
          "  availablePartitions (Partitions): The available Partitions to "
          "schedule on.\n"
          "  currentTime (int): The current time.\n"
          "  schedulerConfig (SchedulerConfig): The configuration for the "
          "scheduler.\n"
          "  timeRangeToGranularities (list): The time ranges to granularities "
          "to use for dynamic discretization.",
          py::arg("expression"), py::arg("availablePartitions"),
          py::arg("currentTime"), py::arg("schedulerConfig"),
          py::arg("timeRangeToGranularities") =
              std::vector<std::pair<tetrisched::TimeRange, tetrisched::Time>>())
      .def(
          "addOptimizationPass", &tetrisched::Scheduler::addOptimizationPass,
          "Add Optimization passes to Tetrisched Based Declarative Scheduler. \n"
          "\nArgs:\n"
          "  optPass (OptimizationPassCategory): The Optimization Pass that will get enabled.\n",
          py::arg("optPass"))
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
  defineSolverConfig(tetrisched_m);
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
