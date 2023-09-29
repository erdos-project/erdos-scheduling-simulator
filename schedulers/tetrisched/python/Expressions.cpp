#include <pybind11/pybind11.h>

#include "tetrisched/Expression.hpp"

namespace py = pybind11;

void defineSTRLExpressions(py::module_& tetrisched_m) {
  // Define the base Expression.
  py::class_<tetrisched::Expression, tetrisched::ExpressionPtr>(tetrisched_m,
                                                                "Expression")
      .def("getNumChildren", &tetrisched::Expression::getNumChildren,
           "Returns the number of children of this Expression.")
      .def("addChild", &tetrisched::Expression::addChild,
           "Adds a child to this Expression.");

  // Define the ChooseExpression.
  py::class_<tetrisched::ChooseExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::ChooseExpression>>(tetrisched_m,
                                                            "ChooseExpression")
      .def(py::init([](tetrisched::TaskPtr associatedTask,
                       tetrisched::Partitions partitions,
                       uint32_t numRequiredMachines, tetrisched::Time startTime,
                       tetrisched::Time duration) {
             return std::make_shared<tetrisched::ChooseExpression>(
                 associatedTask, partitions, numRequiredMachines, startTime,
                 duration);
           }),
           "Initializes a ChooseExpression for the given task to be placed on "
           "`numRequiredMachines` from the given partition at the given "
           "startTime, running for the given duration.");

  // Define the ObjectiveExpression.
  py::class_<tetrisched::ObjectiveExpression, tetrisched::Expression,
             std::shared_ptr<tetrisched::ObjectiveExpression>>(
      tetrisched_m, "ObjectiveExpression")
      .def(py::init<>(), "Initializes an empty ObjectiveExpression.");

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
