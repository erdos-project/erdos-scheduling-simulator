#include <gtest/gtest.h>

#include "tetrisched/Solver.hpp"
#include "tetrisched/SolverModel.hpp"

TEST(SolverModelTypes, TestVariableConstruction) {
  std::string varName = "intVar";
  tetrisched::VariablePtr intVar =
      std::make_shared<tetrisched::Variable>(tetrisched::VAR_INTEGER, varName);
  EXPECT_EQ(intVar->toString(), varName);
}

TEST(SolverModelTypes, TestConstraintConstruction) {
  tetrisched::ConstraintPtr constraint =
      std::make_unique<tetrisched::Constraint>(tetrisched::CONSTR_LE, 10);
  tetrisched::VariablePtr intVar =
      std::make_shared<tetrisched::Variable>(tetrisched::VAR_INTEGER, "intVar");

  constraint->addTerm(1, intVar);
  EXPECT_EQ(constraint->toString(), "(1*intVar) <= 10");
  EXPECT_EQ(constraint->size(), 1);
}

TEST(SolverModelTypes, TestObjectiveFnConstruction) {
  tetrisched::ObjectiveFunction objectiveFn(tetrisched::OBJ_MAXIMIZE);
  tetrisched::VariablePtr intVar =
      std::make_shared<tetrisched::Variable>(tetrisched::VAR_INTEGER, "intVar");
  objectiveFn.addTerm(1, intVar);
  EXPECT_EQ(objectiveFn.toString(), "Maximize: (1*intVar)");
  EXPECT_EQ(objectiveFn.size(), 1);
}

TEST(SolverModelTypes, TestSolverModel) {
  tetrisched::CPLEXSolver cplexSolver;
  tetrisched::SolverModelPtr solverModel = cplexSolver.getModel();
  tetrisched::VariablePtr intVar =
      std::make_shared<tetrisched::Variable>(tetrisched::VAR_INTEGER, "intVar");
  tetrisched::ConstraintPtr constraint =
      std::make_unique<tetrisched::Constraint>(tetrisched::CONSTR_LE, 10);
  constraint->addTerm(1, intVar);
  tetrisched::ObjectiveFunctionPtr objectiveFn =
      std::make_unique<tetrisched::ObjectiveFunction>(tetrisched::OBJ_MAXIMIZE);
  objectiveFn->addTerm(1, intVar);

  solverModel->addVariable(intVar);
  solverModel->addConstraint(std::move(constraint));
  solverModel->setObjectiveFunction(std::move(objectiveFn));

  EXPECT_EQ(solverModel->numVariables(), 1);
  EXPECT_EQ(solverModel->numConstraints(), 1);
  EXPECT_EQ(solverModel->toString(),
            "Maximize: (1*intVar)\n"
            "Constraints: \n\t(1*intVar) <= 10\n"
            "Variables: intVar");
}

TEST(SolverModel, TestCPLEXSolverInitialization) {
  tetrisched::CPLEXSolver cplexSolver;
}

TEST(SolverModel, TestCPLEXModelExport) {}
