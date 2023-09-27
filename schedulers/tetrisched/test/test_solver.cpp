#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#ifdef _TETRISCHED_WITH_CPLEX_
#include "tetrisched/CPLEXSolver.hpp"
#endif  //_TETRISCHED_WITH_CPLEX_
#ifdef _TETRISCHED_WITH_GUROBI_
#include "tetrisched/GurobiSolver.hpp"
#endif  //_TETRISCHED_WITH_GUROBI_
#ifdef _TETRISCHED_WITH_OR_TOOLS_
#include "tetrisched/GoogleCPSolver.hpp"
#endif  //_TETRISCHED_WITH_OR_TOOLS_
#include "tetrisched/Solver.hpp"
#include "tetrisched/SolverModel.hpp"

TEST(SolverModelTypes, TestVariableConstruction) {
  std::string varName = "intVar";
  tetrisched::VariablePtr intVar =
      std::make_shared<tetrisched::Variable>(tetrisched::VAR_INTEGER, varName);
  EXPECT_EQ(intVar->toString(), varName);
}

/// Throw an error if we try to make an Integer variable continuous.
TEST(SolverModelTypes, TestIncorrectVariableConstruction) {
  std::string varName = "intVar";
  EXPECT_THROW(std::make_shared<tetrisched::Variable>(
                   tetrisched::VAR_CONTINUOUS, varName),
               tetrisched::exceptions::SolverException);
}

TEST(SolverModelTypes, TestConstraintConstruction) {
  tetrisched::ConstraintPtr constraint =
      std::make_unique<tetrisched::Constraint>("TestConstraint",
                                               tetrisched::CONSTR_LE, 10);
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

tetrisched::VariablePtr constructModel(
    tetrisched::SolverModelPtr& solverModelPtr) {
  auto intVar = std::make_shared<tetrisched::Variable>(tetrisched::VAR_INTEGER,
                                                       "intVar", 0, 100);
  solverModelPtr->addVariable(intVar);
  auto constraint = std::make_unique<tetrisched::Constraint>(
      "TestConstraint", tetrisched::ConstraintType::CONSTR_LE, 10);
  constraint->addTerm(2, intVar);
  constraint->addTerm(5);
  solverModelPtr->addConstraint(std::move(constraint));
  auto objectiveFunction = std::make_unique<tetrisched::ObjectiveFunction>(
      tetrisched::ObjectiveType::OBJ_MAXIMIZE);
  objectiveFunction->addTerm(1, intVar);
  solverModelPtr->setObjectiveFunction(std::move(objectiveFunction));
  return intVar;
}

#ifdef _TETRISCHED_WITH_CPLEX_
TEST(SolverModelTypes, TestSolverModel) {
  tetrisched::CPLEXSolver cplexSolver;
  tetrisched::SolverModelPtr solverModel = cplexSolver.getModel();
  tetrisched::VariablePtr intVar =
      std::make_shared<tetrisched::Variable>(tetrisched::VAR_INTEGER, "intVar");
  tetrisched::ConstraintPtr constraint =
      std::make_unique<tetrisched::Constraint>("TestConstraint",
                                               tetrisched::CONSTR_LE, 10);
  constraint->addTerm(1, intVar);
  tetrisched::ObjectiveFunctionPtr objectiveFn =
      std::make_unique<tetrisched::ObjectiveFunction>(tetrisched::OBJ_MAXIMIZE);
  objectiveFn->addTerm(1, intVar);

  solverModel->addVariable(intVar);
  solverModel->addConstraint(std::move(constraint));
  solverModel->setObjectiveFunction(std::move(objectiveFn));

  EXPECT_EQ(solverModel->numVariables(), 1);
  EXPECT_EQ(solverModel->numConstraints(), 1);
  solverModel->exportModel("test.lp");
  EXPECT_TRUE(std::filesystem::exists("test.lp"))
      << "The file test.lp was not created.";
  std::filesystem::remove("test.lp");
}

TEST(SolverModel, TestCPLEXSolverTranslation) {
  tetrisched::CPLEXSolver cplexSolver;
  auto solverModelPtr = cplexSolver.getModel();
  auto intVar = constructModel(solverModelPtr);
  solverModelPtr->exportModel("test_solvermodel.lp");
  EXPECT_TRUE(std::filesystem::exists("test_solvermodel.lp"))
      << "The file test_solvermodel.lp was not created.";
  std::filesystem::remove("test_solvermodel.lp");
  cplexSolver.translateModel();
  cplexSolver.exportModel("test_cplexmodel.lp");
  EXPECT_TRUE(std::filesystem::exists("test_cplexmodel.lp"))
      << "The file test_cplexmodel.lp was not created.";
  std::filesystem::remove("test_cplexmodel.lp");

  // Solve the model.
  cplexSolver.solveModel();

  // Check if solution is correct.
  auto solutionValue = intVar->getValue();
  EXPECT_TRUE(solutionValue.has_value()) << "No solution found.";
  EXPECT_EQ(solutionValue.value(), 2) << "Solution is not correct.";
}
#endif  //_TETRISCHED_WITH_CPLEX_

#ifdef _TETRISCHED_WITH_GUROBI_
TEST(SolverModel, TestGurobiSolverTranslation) {
  tetrisched::GurobiSolver gurobiSolver;
  auto solverModelPtr = gurobiSolver.getModel();
  constructModel(solverModelPtr);
  solverModelPtr->exportModel("test_solvermodel.lp");
  EXPECT_TRUE(std::filesystem::exists("test_solvermodel.lp"))
      << "The file test_solvermodel.lp was not created.";
  std::filesystem::remove("test_solvermodel.lp");
  gurobiSolver.translateModel();
  gurobiSolver.exportModel("test_gurobimodel.lp");
  EXPECT_TRUE(std::filesystem::exists("test_gurobimodel.lp"))
      << "The file test_gurobimodel.lp was not created.";
  std::filesystem::remove("test_gurobimodel.lp");
}
#endif  //_TETRISCHED_WITH_GUROBI_

#ifdef _TETRISCHED_WITH_ORTOOLS_
TEST(SolverModel, TestOrToolsSolverTranslation) {
  tetrisched::GoogleCPSolver googleCPSolver;
  auto solverModelPtr = googleCPSolver.getModel();
  constructModel(solverModelPtr);
  solverModelPtr->exportModel("test_solvermodel.lp");
  EXPECT_TRUE(std::filesystem::exists("test_solvermodel.lp"))
      << "The file test_solvermodel.lp was not created.";
  std::filesystem::remove("test_solvermodel.lp");
  googleCPSolver.translateModel();
  googleCPSolver.exportModel("test_ortoolsmodel.lp");
  EXPECT_TRUE(std::filesystem::exists("test_ortoolsmodel.lp"))
      << "The file test_ortoolsmodel.lp was not created.";
  std::filesystem::remove("test_ortoolsmodel.lp");
}
#endif  //_TETRISCHED_WITH_ORTOOLS_
