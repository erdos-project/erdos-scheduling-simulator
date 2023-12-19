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
      .value("VAR_INDICATOR", tetrisched::VariableType::VAR_INDICATOR);

  // Implement the bindings for the model variable.
  py::class_<tetrisched::Variable, tetrisched::VariablePtr>(tetrisched_m,
                                                            "Variable")
      .def(py::init([](tetrisched::VariableType type, std::string name) {
             return std::make_shared<tetrisched::Variable>(type, name);
           }),
           "Initializes the Variable with the given type and name.\n"
           "\nArgs:\n"
           "  type (VariableType): The type of this Variable.\n"
           "  name (str): The name of this Variable.",
           py::arg("type"), py::arg("name"))
      .def(py::init([](tetrisched::VariableType type, std::string name,
                       TETRISCHED_ILP_TYPE lowerBound) {
             return std::make_shared<tetrisched::Variable>(type, name,
                                                           lowerBound);
           }),
           "Initializes the Variable with the given type, name and a lower "
           "bound.\n"
           "\nArgs:\n"
           "  type (VariableType): The type of this Variable.\n"
           "  name (str): The name of this Variable.\n"
           "  lowerBound (int): The lower bound of this Variable.",
           py::arg("type"), py::arg("name"), py::arg("lowerBound"))
      .def(py::init([](tetrisched::VariableType type, std::string name,
                       TETRISCHED_ILP_TYPE lowerBound,
                       TETRISCHED_ILP_TYPE upperBound) {
             return std::make_shared<tetrisched::Variable>(
                 type, name, lowerBound, upperBound);
           }),
           "Initializes the Variable with the given type, name and a lower and "
           "upper bound.\n"
           "\nArgs:\n"
           "  type (VariableType): The type of this Variable.\n"
           "  name (str): The name of this Variable.\n"
           "  lowerBound (int): The lower bound of this Variable.\n"
           "  upperBound (int): The upper bound of this Variable.",
           py::arg("type"), py::arg("name"), py::arg("lowerBound"),
           py::arg("upperBound"))
      .def("hint", &tetrisched::Variable::hint,
           "Provides a hint to the solver for the initial value of this "
           "Variable.\n"
           "\nArgs:\n"
           "  value (int): The value to set for this Variable.",
           py::arg("value"))
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
      .value("CONSTR_GE", tetrisched::ConstraintType::CONSTR_GE);

  // Implement the bindings for the model Constraint.
  py::class_<tetrisched::Constraint, tetrisched::ConstraintPtr>(tetrisched_m,
                                                                "Constraint")
      .def(py::init([](std::string name, tetrisched::ConstraintType type,
                       TETRISCHED_ILP_TYPE rhs) {
             return std::make_shared<tetrisched::Constraint>(name, type, rhs);
           }),
           "Initializes the Constraint with the given name, type and RHS.\n"
           "\nArgs:\n"
           "  name (str): The name of this Constraint.\n"
           "  type (ConstraintType): The type of this Constraint.\n"
           "  rhs (int): The RHS of this Constraint.",
           py::arg("name"), py::arg("type"), py::arg("rhs"))
      .def("addTerm",
           py::overload_cast<TETRISCHED_ILP_TYPE, tetrisched::VariablePtr>(
               &tetrisched::Constraint::addTerm),
           "Adds a new term to the LHS of this Constraint.\n"
           "\nArgs:\n"
           "  coefficient (int): The coefficient of the new term.\n"
           "  variable (Variable): The Variable of the new term.",
           py::arg("coefficient"), py::arg("variable"))
      .def("addTerm",
           py::overload_cast<TETRISCHED_ILP_TYPE>(
               &tetrisched::Constraint::addTerm),
           "Adds a new constant to the LHS of this Constraint.\n"
           "\nArgs:\n"
           "  constant (int): The constant of the new term.",
           py::arg("constant"))
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
      .value("OBJ_MINIMIZE", tetrisched::ObjectiveType::OBJ_MINIMIZE);

  // Implement the bindings for the model ObjectiveFunction.
  py::class_<tetrisched::ObjectiveFunction, tetrisched::ObjectiveFunctionPtr>(
      tetrisched_m, "ObjectiveFunction")
      .def(py::init([](tetrisched::ObjectiveType type) {
             return std::make_shared<tetrisched::ObjectiveFunction>(type);
           }),
           "Initializes the ObjectiveFunction with the given type.\n"
           "\nArgs:\n"
           "  type (ObjectiveType): The type of this ObjectiveFunction.",
           py::arg("type"))
      .def("addTerm",
           py::overload_cast<TETRISCHED_ILP_TYPE, tetrisched::VariablePtr>(
               &tetrisched::ObjectiveFunction::addTerm),
           "Adds a new term to the ObjectiveFunction.\n"
           "\nArgs:\n"
           "  coefficient (int): The coefficient of the new term.\n"
           "  variable (Variable): The Variable of the new term.",
           py::arg("coefficient"), py::arg("variable"))
      .def("addTerm",
           py::overload_cast<TETRISCHED_ILP_TYPE>(
               &tetrisched::ObjectiveFunction::addTerm),
           "Adds a new constant to the ObjectiveFunction.\n"
           "\nArgs:\n"
           "  constant (int): The constant of the new term.",
           py::arg("constant"))
      .def("toConstraint", &tetrisched::ObjectiveFunction::toConstraint,
           "Converts this ObjectiveFunction into a Constraint.\n"
           "\nArgs:\n"
           "  name (str): The name of the Constraint to be returned.\n"
           "  type (ConstraintType): The type of the Constraint to be "
           "returned.\n"
           "  rhs (int): The RHS of the Constraint to be returned.",
           py::arg("name"), py::arg("type"), py::arg("rhs"))
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
           "Adds a new Variable to the model.\n"
           "\nArgs:\n"
           "  variable (Variable): The Variable to add to the model.",
           py::arg("variable"))
      .def("addConstraint", &tetrisched::SolverModel::addConstraint,
           "Adds a new Constraint to the model.\n"
           "\nArgs:\n"
           "  constraint (Constraint): The Constraint to add to the model.",
           py::arg("constraint"))
      .def("setObjectiveFunction",
           &tetrisched::SolverModel::setObjectiveFunction,
           "Sets the ObjectiveFunction of the model.\n"
           "\nArgs:\n"
           "  objective (ObjectiveFunction): The ObjectiveFunction to set.",
           py::arg("objective"))
      .def("exportModel", &tetrisched::SolverModel::exportModel,
           "Exports the model to the given file.\n"
           "\nArgs:\n"
           "  fileName (str): The name of the file to export the model to.",
           py::arg("fileName"))
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
