#include <pybind11/pybind11.h>

#include "tetrisched/Expression.hpp"

namespace py = pybind11;

void defineSTRLExpressions(py::module_& tetrisched_m) {
  // Define the ExpressionType enum.
  py::enum_<tetrisched::ExpressionType>(tetrisched_m, "ExpressionType")
      .value("EXPR_CHOOSE", tetrisched::ExpressionType::EXPR_CHOOSE)
      .value("EXPR_OBJECTIVE", tetrisched::ExpressionType::EXPR_OBJECTIVE)
      .value("EXPR_MIN", tetrisched::ExpressionType::EXPR_MIN)
      .value("EXPR_MAX", tetrisched::ExpressionType::EXPR_MAX)
      .value("EXPR_SCALE", tetrisched::ExpressionType::EXPR_SCALE)
      .value("EXPR_LESSTHAN", tetrisched::ExpressionType::EXPR_LESSTHAN)
      .value("EXPR_ALLOCATION", tetrisched::ExpressionType::EXPR_ALLOCATION)
      .value("EXPR_MALLEABLE_CHOOSE",
             tetrisched::ExpressionType::EXPR_MALLEABLE_CHOOSE)
      .value("EXPR_WINDOWED_CHOOSE",
             tetrisched::ExpressionType::EXPR_WINDOWED_CHOOSE)
      .export_values();

  // Define the Placement object.
  py::class_<tetrisched::Placement, tetrisched::PlacementPtr>(tetrisched_m,
                                                              "Placement")
      .def_property_readonly("name", &tetrisched::Placement::getName,
                             "The name of the Placement.")
      .def_property_readonly("startTime", &tetrisched::Placement::getStartTime)
      .def("isPlaced", &tetrisched::Placement::isPlaced,
           "Returns true if the Placement was placed, false otherwise.")
      .def("getPartitionAllocations",
           &tetrisched::Placement::getPartitionAllocations,
           "Returns the Partition assignments for this Placement.");

  // Define the SolutionResult.
  py::class_<tetrisched::SolutionResult, tetrisched::SolutionResultPtr>(
      tetrisched_m, "SolutionResult")
      .def_property_readonly(
          "startTime",
          [](const tetrisched::SolutionResult& result) {
            return result.startTime;
          },
          "The start time of the expression.")
      .def_property_readonly(
          "endTime",
          [](const tetrisched::SolutionResult& result) {
            return result.endTime;
          },
          "The end time of the expression.")
      .def_property_readonly(
          "utility",
          [](const tetrisched::SolutionResult& result) {
            return result.utility;
          },
          "The utility of the expression.")
      .def(
          "getPlacement",
          [](const tetrisched::SolutionResult& result,
             std::string taskName) -> std::optional<tetrisched::PlacementPtr> {
            if (result.placements.find(taskName) == result.placements.end()) {
              return std::nullopt;
            }
            return result.placements.at(taskName);
          },
          "Returns the Placement for the given task.")
      .def("__str__", [](const tetrisched::SolutionResult& result) {
        return "Placement<start=" +
               (result.startTime.has_value()
                    ? std::to_string(result.startTime.value())
                    : "None") +
               ", end=" +
               (result.endTime.has_value()
                    ? std::to_string(result.endTime.value())
                    : "None") +
               ", utility=" +
               (result.utility.has_value()
                    ? std::to_string(result.utility.value())
                    : "None") +
               ">";
      });

  // Define the base Expression.
  py::class_<tetrisched::Expression, tetrisched::ExpressionPtr>(tetrisched_m,
                                                                "Expression")
      .def("getNumChildren", &tetrisched::Expression::getNumChildren,
           "Returns the number of children of this Expression.")
      .def("getNumParents", &tetrisched::Expression::getNumParents,
           "Returns the number of parents of this Expression.")
      .def("getChildren", &tetrisched::Expression::getChildren,
           "Returns the children of this Expression.")
      .def("getType", &tetrisched::Expression::getType,
           "Returns the type of this Expression.")
      .def("addChild", &tetrisched::Expression::addChild,
           "Adds a child to this Expression.")
      .def("getSolution", &tetrisched::Expression::getSolution,
           "Returns the solution for this Expression.")
      .def("exportToDot", &tetrisched::Expression::exportToDot,
           "Exports the Expression to a dot file.")
      .def("__str__",
           [](const tetrisched::Expression& expr) {
             return "Expression<name=" + expr.getName() +
                    ", type=" + expr.getTypeString() + ">";
           })
      .def_property_readonly("name", &tetrisched::Expression::getName)
      .def_property_readonly("id", &tetrisched::Expression::getId);

  // Define the ChooseExpression.
  py::class_<tetrisched::ChooseExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::ChooseExpression>>(tetrisched_m,
                                                            "ChooseExpression")
      .def(py::init([](std::string taskName, tetrisched::Partitions partitions,
                       uint32_t numRequiredMachines, tetrisched::Time startTime,
                       tetrisched::Time duration, TETRISCHED_ILP_TYPE utility) {
             return std::make_shared<tetrisched::ChooseExpression>(
                 taskName, partitions, numRequiredMachines, startTime, duration,
                 utility);
           }),
           "Initializes a ChooseExpression for the given task to be placed on "
           "`numRequiredMachines` from the given partition at the given "
           "startTime, running for the given duration.");

  // Define the WindowedChooseExpression.
  py::class_<tetrisched::WindowedChooseExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::WindowedChooseExpression>>(
      tetrisched_m, "WindowedChooseExpression")
      .def(py::init([](std::string taskName, tetrisched::Partitions partitions,
                       uint32_t numRequiredMachines, tetrisched::Time startTime,
                       tetrisched::Time duration, tetrisched::Time endTime,
                       tetrisched::Time granularity,
                       TETRISCHED_ILP_TYPE utility) {
             return std::make_shared<tetrisched::WindowedChooseExpression>(
                 taskName, partitions, numRequiredMachines, startTime, duration,
                 endTime, granularity, utility);
           }),
           "Initializes a WindowedChooseExpression for the given task to be "
           "placed on `numRequiredMachines` from the given partition "
           "between the given startTime and endTime, and "
           "running for the given duration.");

  // Define the MalleableChooseExpression.
  py::class_<tetrisched::MalleableChooseExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::MalleableChooseExpression>>(
      tetrisched_m, "MalleableChooseExpression")
      .def(py::init([](std::string taskName, tetrisched::Partitions partitions,
                       uint32_t resourceTimeSlots, tetrisched::Time startTime,
                       tetrisched::Time endTime, tetrisched::Time granularity,
                       TETRISCHED_ILP_TYPE utility) {
             return std::make_shared<tetrisched::MalleableChooseExpression>(
                 taskName, partitions, resourceTimeSlots, startTime, endTime,
                 granularity, utility);
           }),
           "Initializes a MalleableChooseExpression for the given task to be "
           "placed on the given partitions at the given startTime, "
           "ending at the given end time and taking up the given "
           "resourceTimeSlots");

  // Define the AllocationExpression.
  py::class_<tetrisched::AllocationExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::AllocationExpression>>(
      tetrisched_m, "AllocationExpression")
      .def(
          py::init([](std::string taskName,
                      std::vector<std::pair<tetrisched::PartitionPtr, uint32_t>>
                          partitionAssignments,
                      tetrisched::Time startTime, tetrisched::Time duration) {
            return std::make_shared<tetrisched::AllocationExpression>(
                taskName, partitionAssignments, startTime, duration);
          }),
          "Initializes an AllocationExpression for the given task to be "
          "placed on the given partitions at the given startTime, "
          "running for the given duration.");

  // Define the ObjectiveExpression.
  py::class_<tetrisched::ObjectiveExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::ObjectiveExpression>>(
      tetrisched_m, "ObjectiveExpression")
      .def(py::init<std::string>(),
           "Initializes an empty ObjectiveExpression.");

  // Define the MinExpression.
  py::class_<tetrisched::MinExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::MinExpression>>(tetrisched_m,
                                                         "MinExpression")
      .def(py::init<std::string>(),
           "Initializes a MinExpression with the given helper name.");

  // Define the MaxExpression.
  py::class_<tetrisched::MaxExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::MaxExpression>>(tetrisched_m,
                                                         "MaxExpression")
      .def(py::init<std::string>(),
           "Initializes a MaxExpression with the given helper name.");

  // Define the ScaleExpression.
  py::class_<tetrisched::ScaleExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::ScaleExpression>>(tetrisched_m,
                                                           "ScaleExpression")
      .def(py::init<std::string, TETRISCHED_ILP_TYPE>(),
           "Initializes a ScaleExpression with the given helper name and "
           "scaling factor.");

  // Define the LessThanExpression.
  py::class_<tetrisched::LessThanExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::LessThanExpression>>(
      tetrisched_m, "LessThanExpression")
      .def(py::init<std::string>(),
           "Initializes a LessThanExpression with the given helper name.");
}
