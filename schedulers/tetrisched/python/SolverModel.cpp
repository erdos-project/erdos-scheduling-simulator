#include "tetrisched/SolverModel.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/// Implements the Python bindings for tetrisched::Variable.
/// Assumes that the type and the pointer type are defined by libtetrisched.so.
void defineModelVariable(py::module_& tetrisched_m) {
  // Implement the enumeration for the variable type.
  py::enum_<tetrisched::VariableType>(tetrisched_m, "VariableType")
      .value("VAR_CONTINUOUS", tetrisched::VariableType::VAR_CONTINUOUS)
      .value("VAR_INTEGER", tetrisched::VariableType::VAR_INTEGER)
      .value("VAR_INDICATOR", tetrisched::VariableType::VAR_INDICATOR)
      .export_values();

  // Implement the bindings for the model variable.
  py::class_<tetrisched::Variable, tetrisched::VariablePtr>(tetrisched_m,
                                                            "Variable")
      .def(py::init([](tetrisched::VariableType type, std::string name) {
             return std::make_shared<tetrisched::Variable>(type, name);
           }),
           "Initializes the Variable with the given type and name.")
      .def(py::init([](tetrisched::VariableType type, std::string name,
                       TETRISCHED_ILP_TYPE lowerBound) {
             return std::make_shared<tetrisched::Variable>(type, name,
                                                           lowerBound);
           }),
           "Initializes the Variable with the given type, name and a lower "
           "bound.")
      .def(py::init([](tetrisched::VariableType type, std::string name,
                       TETRISCHED_ILP_TYPE lowerBound,
                       TETRISCHED_ILP_TYPE upperBound) {
             return std::make_shared<tetrisched::Variable>(
                 type, name, lowerBound, upperBound);
           }),
           "Initializes the Variable with the given type, name and a lower and "
           "upper bound.")
      .def("hint", &tetrisched::Variable::hint,
           "Provides a hint to the solver for the initial value of this "
           "Variable.")
      .def("__str__", &tetrisched::Variable::toString)
      .def_property_readonly("id", &tetrisched::Variable::getId,
                             "The ID of this Variable.")
      .def_property_readonly("name", &tetrisched::Variable::getName,
                             "The name of this Variable.")
      .def_property_readonly("value", &tetrisched::Variable::getValue,
                             "The value of this Variable.");
}

/// Implments the Python bindings for tetrisched::Constraint.
/// Assumes that the type and the pointer type are defined by libtetrisched.so.
void defineModelConstraint(py::module_& tetrisched_m) {
  // Implement the enumeration for the constraint type.
  py::enum_<tetrisched::ConstraintType>(tetrisched_m, "ConstraintType")
      .value("CONSTR_LE", tetrisched::ConstraintType::CONSTR_LE)
      .value("CONSTR_EQ", tetrisched::ConstraintType::CONSTR_EQ)
      .value("CONSTR_GE", tetrisched::ConstraintType::CONSTR_GE)
      .export_values();

  // Implement the bindings for the model Constraint.
  py::class_<tetrisched::Constraint, tetrisched::ConstraintPtr>(tetrisched_m,
                                                                "Constraint")
      .def(py::init([](std::string name, tetrisched::ConstraintType type,
                       TETRISCHED_ILP_TYPE rhs) {
             return std::make_shared<tetrisched::Constraint>(name, type, rhs);
           }),
           "Initializes the Constraint with the given name, type and RHS.")
      .def("addTerm",
           py::overload_cast<TETRISCHED_ILP_TYPE, tetrisched::VariablePtr>(
               &tetrisched::Constraint::addTerm),
           "Adds a new term to the LHS of this Constraint.")
      .def("addTerm",
           py::overload_cast<TETRISCHED_ILP_TYPE>(
               &tetrisched::Constraint::addTerm),
           "Adds a new constant to the LHS of this Constraint.")
      .def("__str__", &tetrisched::Constraint::toString)
      .def("__len__", &tetrisched::Constraint::size)
      .def_property_readonly("name", &tetrisched::Constraint::getName,
                             "The name of this Constraint.")
      .def_property_readonly("id", &tetrisched::Constraint::getId,
                             "The ID of this Constraint.");
}

/// Implements the Python bindings for tetrisched::ObjectiveFunction.
/// Assumes that the type and the pointer type are defined by libtetrisched.so.
void defineModelObjective(py::module_& tetrisched_m) {
  // Implement the enumeration for the objective type.
  py::enum_<tetrisched::ObjectiveType>(tetrisched_m, "ObjectiveType")
      .value("OBJ_MAXIMIZE", tetrisched::ObjectiveType::OBJ_MAXIMIZE)
      .value("OBJ_MINIMIZE", tetrisched::ObjectiveType::OBJ_MINIMIZE)
      .export_values();

  // Implement the bindings for the model ObjectiveFunction.
  py::class_<tetrisched::ObjectiveFunction, tetrisched::ObjectiveFunctionPtr>(
      tetrisched_m, "ObjectiveFunction")
      .def(py::init([](tetrisched::ObjectiveType type) {
             return std::make_shared<tetrisched::ObjectiveFunction>(type);
           }),
           "Initializes the ObjectiveFunction with the given type.")
      .def("addTerm", &tetrisched::ObjectiveFunction::addTerm,
           "Adds a new term to the ObjectiveFunction.")
      .def("toConstraint", &tetrisched::ObjectiveFunction::toConstraint,
           "Converts this ObjectiveFunction into a Constraint.")
      .def("__str__", &tetrisched::ObjectiveFunction::toString)
      .def("__len__", &tetrisched::ObjectiveFunction::size)
      .def_property_readonly("value", &tetrisched::ObjectiveFunction::getValue,
                             "The value of this ObjectiveFunction.");
}

/// Implements the Python bindings for tetrisched::SolverModel.
/// Assumes that the type and the pointer type are defined by libtetrisched.so.
void defineSolverModel(py::module_& tetrisched_m) {
  // Implement the bindings for the solver model.
  py::class_<tetrisched::SolverModel, tetrisched::SolverModelPtr>(tetrisched_m,
                                                                  "SolverModel")
      .def("addVariable", &tetrisched::SolverModel::addVariable,
           "Adds a new Variable to the model.")
      .def("addConstraint", &tetrisched::SolverModel::addConstraint,
           "Adds a new Constraint to the model.")
      .def("setObjectiveFunction",
           &tetrisched::SolverModel::setObjectiveFunction,
           "Sets the ObjectiveFunction of the model.")
      .def("exportModel", &tetrisched::SolverModel::exportModel,
           "Exports the model to the given file.")
      .def("__str__", &tetrisched::SolverModel::toString)
      .def_property_readonly("num_variables",
                             &tetrisched::SolverModel::numVariables,
                             "The number of variables in the model.")
      .def_property_readonly("num_constraints",
                             &tetrisched::SolverModel::numConstraints,
                             "The number of constraints in the model.")
      .def_property_readonly("objective_value",
                             &tetrisched::SolverModel::getObjectiveValue,
                             "The value of the model's objective function.");
}
