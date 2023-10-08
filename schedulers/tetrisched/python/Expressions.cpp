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
      .export_values();

  // Define the Placement object.
  py::class_<tetrisched::Placement, tetrisched::PlacementPtr>(tetrisched_m,
                                                              "Placement")
      .def_property_readonly("name", &tetrisched::Placement::getName,
                             "The name of the Placement.")
      .def_property_readonly("startTime", &tetrisched::Placement::getStartTime)
      .def("isPlaced", &tetrisched::Placement::isPlaced,
           "Returns true if the Placement was placed, false otherwise.")
      .def("getPartitionAssignments",
           &tetrisched::Placement::getPartitionAssignments,
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
      .def("__str__",
           [](const tetrisched::Expression& expr) {
             return "Expression<name=" + expr.getName() +
                    ", type=" + expr.getTypeString() + ">";
           })
      .def_property_readonly("name", &tetrisched::Expression::getName);

  // Define the ChooseExpression.
  py::class_<tetrisched::ChooseExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::ChooseExpression>>(tetrisched_m,
                                                            "ChooseExpression")
      .def(py::init([](std::string taskName, tetrisched::Partitions partitions,
                       uint32_t numRequiredMachines, tetrisched::Time startTime,
                       tetrisched::Time duration) {
             return std::make_shared<tetrisched::ChooseExpression>(
                 taskName, partitions, numRequiredMachines, startTime,
                 duration);
           }),
           "Initializes a ChooseExpression for the given task to be placed on "
           "`numRequiredMachines` from the given partition at the given "
           "startTime, running for the given duration.");

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
