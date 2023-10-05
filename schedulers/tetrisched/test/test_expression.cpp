#include <gtest/gtest.h>

#include "tetrisched/Expression.hpp"
#include "tetrisched/Partition.hpp"
#include "tetrisched/Solver.hpp"
#ifdef _TETRISCHED_WITH_CPLEX_
#include "tetrisched/CPLEXSolver.hpp"
#endif  //_TETRISCHED_WITH_CPLEX_

/// Checks that no children can be added to a ChooseExpression.
/// i.e., a ChooseExpression is a leaf node in the expression tree.
TEST(Expression, TestChooseExpressionIsLeaf) {
  tetrisched::Partitions partitions = tetrisched::Partitions();
  auto chooseExpression = std::make_unique<tetrisched::ChooseExpression>(
      "task1", partitions, 0, 0, 10);
  auto chooseExpression2 = std::make_unique<tetrisched::ChooseExpression>(
      "task1", partitions, 0, 0, 10);
  EXPECT_THROW(chooseExpression->addChild(std::move(chooseExpression2)),
               tetrisched::exceptions::ExpressionConstructionException);
}

TEST(Expression, TestMinExpressionIsNOTLeaf) {
  tetrisched::Partitions partitions = tetrisched::Partitions();
  tetrisched::ExpressionPtr chooseExpression =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 0, 0,
                                                     10);
  tetrisched::ExpressionPtr chooseExpression2 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 0, 0,
                                                     10);

  tetrisched::ExpressionPtr minExpression =
      std::make_shared<tetrisched::MinExpression>("TEST_MIN");
  minExpression->addChild(chooseExpression2);
  minExpression->addChild(chooseExpression);
  EXPECT_TRUE(minExpression->getNumChildren() == 2);
  EXPECT_TRUE(minExpression->getNumParents() == 0);
  EXPECT_TRUE(chooseExpression2->getNumParents() == 1);
  EXPECT_TRUE(chooseExpression->getNumParents() == 1);
}

#ifdef _TETRISCHED_WITH_CPLEX_
/// Checks that for two trivially satisfiable expressions, the less than
/// expression is only satisfied if their times are ordered.
TEST(Expression, TestLessThanEnforcesOrdering) {
  // Construct the Workers and a Partition.
  tetrisched::WorkerPtr worker1 =
      std::make_shared<tetrisched::Worker>(1, "worker1");
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>();
  partition->addWorker(worker1, 2);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition});

  // Construct the choice for the two tasks.
  tetrisched::ExpressionPtr chooseTask1 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 1, 0,
                                                     100);
  tetrisched::ExpressionPtr chooseTask2 =
      std::make_shared<tetrisched::ChooseExpression>("task2", partitions, 1,
                                                     200, 100);

  // Construct the LessThan expression.
  tetrisched::ExpressionPtr lessThanExpression =
      std::make_shared<tetrisched::LessThanExpression>(
          "task1_less_than_task_2");
  lessThanExpression->addChild(std::move(chooseTask1));
  lessThanExpression->addChild(std::move(chooseTask2));

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>();
  objectiveExpression->addChild(std::move(lessThanExpression));

  // Construct a Solver.
  tetrisched::CPLEXSolver cplexSolver = tetrisched::CPLEXSolver();
  auto solverModelPtr = cplexSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap;

  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);
  solverModelPtr->exportModel("testLessThanEnforcesOrdering.lp");

  // Translate and solve the model.
  cplexSolver.translateModel();
  EXPECT_EQ(0, capacityConstraintMap.size())
      << "Capacity map should be drained after translation.";
  cplexSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);
  EXPECT_TRUE(result->utility);
  EXPECT_EQ(1, result->utility.value());
}

// Check that the STRL parsing for the MaxExpression enforces only
// one of the children to be true.
TEST(Expression, TestMaxExpressionEnforcesSingleChoice) {
  // Construct the Workers and a Partition.
  tetrisched::WorkerPtr worker1 =
      std::make_shared<tetrisched::Worker>(1, "worker1");
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>();
  partition->addWorker(worker1, 2);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition});

  // Construct two choices for a task.
  tetrisched::ExpressionPtr chooseTask1_1 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 1, 0,
                                                     100);
  tetrisched::ExpressionPtr chooseTask1_2 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 1,
                                                     100, 100);

  // Constrain only one choice to actually happen.
  tetrisched::ExpressionPtr maxChooseExpr =
      std::make_shared<tetrisched::MaxExpression>("maxChooseTask1");
  maxChooseExpr->addChild(std::move(chooseTask1_1));
  maxChooseExpr->addChild(std::move(chooseTask1_2));

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>();
  objectiveExpression->addChild(std::move(maxChooseExpr));

  // Construct a Solver.
  tetrisched::CPLEXSolver cplexSolver = tetrisched::CPLEXSolver();
  auto solverModelPtr = cplexSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap;
  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);
  solverModelPtr->exportModel("testMaxExpressionEnforcesSingleChoice.lp");

  // Translate and solve the model.
  cplexSolver.translateModel();
  cplexSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);
  EXPECT_TRUE(result->utility);
  EXPECT_EQ(1, result->utility.value()) << "Only one choice should be made.";
}

// Check that the STRL parsing for the MinExpression enforces all
// children to be true.
TEST(Expression, TestMinExpressionEnforcesAllChildrenSatisfied) {
  // Construct the Workers and a Partition.
  tetrisched::WorkerPtr worker1 =
      std::make_shared<tetrisched::Worker>(1, "worker1");
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>();
  partition->addWorker(worker1, 1);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition});

  // Construct two choices for a task.
  tetrisched::ExpressionPtr chooseTask1_1 =
      std::make_shared<tetrisched::ChooseExpression>("task1_1", partitions, 1,
                                                     0, 100);
  tetrisched::ExpressionPtr chooseTask1_2 =
      std::make_shared<tetrisched::ChooseExpression>("task1_2", partitions, 1,
                                                     100, 100);

  // Constrain both choices to actually happen.
  tetrisched::ExpressionPtr minChooseExpr =
      std::make_shared<tetrisched::MinExpression>("minChooseTaskBoth");
  minChooseExpr->addChild(std::move(chooseTask1_1));
  minChooseExpr->addChild(std::move(chooseTask1_2));

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>();
  objectiveExpression->addChild(std::move(minChooseExpr));

  // Construct a Solver.
  tetrisched::CPLEXSolver cplexSolver = tetrisched::CPLEXSolver();
  auto solverModelPtr = cplexSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap;
  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);
  solverModelPtr->exportModel(
      "testMinExpressionEnforcesAllChildrenSatisfied.lp");

  // Translate and solve the model.
  cplexSolver.translateModel();
  cplexSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);
  EXPECT_TRUE(result->utility);
  EXPECT_EQ(1, result->utility.value()) << "Both choices should be satisfied.";
}

// Check that the STRL parsing for the MinExpression enforces no expression to
// be true if all children can't be satisfied.
TEST(Expression, TestMinExpressionEnforcesNoneSatisfied) {
  // Construct the Workers and a Partition.
  tetrisched::WorkerPtr worker1 =
      std::make_shared<tetrisched::Worker>(1, "worker1");
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>();
  partition->addWorker(worker1, 1);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition});

  // Construct two choices for a task.
  tetrisched::ExpressionPtr chooseTask1_1 =
      std::make_shared<tetrisched::ChooseExpression>("task1_1", partitions, 1,
                                                     0, 100);
  tetrisched::ExpressionPtr chooseTask1_2 =
      std::make_shared<tetrisched::ChooseExpression>("task1_2", partitions, 1,
                                                     50, 100);

  // Constrain both choices to actually happen.
  tetrisched::ExpressionPtr minChooseExpr =
      std::make_shared<tetrisched::MinExpression>("minChooseTaskNone");
  minChooseExpr->addChild(std::move(chooseTask1_1));
  minChooseExpr->addChild(std::move(chooseTask1_2));

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>();
  objectiveExpression->addChild(std::move(minChooseExpr));

  // Construct a Solver.
  tetrisched::CPLEXSolver cplexSolver = tetrisched::CPLEXSolver();
  auto solverModelPtr = cplexSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap;
  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);
  solverModelPtr->exportModel("testMinExpressionEnforcesNoneSatisfied.lp");

  // Translate and solve the model.
  cplexSolver.translateModel();
  cplexSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);
  EXPECT_TRUE(result->utility);
  EXPECT_EQ(0, result->utility.value()) << "No choices should be satisfied.";
}
#endif
