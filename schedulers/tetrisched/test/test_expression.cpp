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

/// Checks that the MaxExpression can only have ChooseExpression as children.
TEST(Expression, TestMaxExpressionOnlyAllowsChooseExpressionChildren) {
  tetrisched::Partitions partitions = tetrisched::Partitions();
  // Ensure ChooseExpression can be added to MaxExpression.
  tetrisched::ExpressionPtr chooseExpression =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 0, 0,
                                                     10);
  tetrisched::ExpressionPtr maxExpression =
      std::make_shared<tetrisched::MaxExpression>("TestMax");
  EXPECT_NO_THROW(maxExpression->addChild(chooseExpression))
      << "MaxExpression should allow ChooseExpression children.";

  // Ensure MinExpression cannot be added to MaxExpression.
  tetrisched::ExpressionPtr minExpression =
      std::make_shared<tetrisched::MinExpression>("TestMin");
  EXPECT_THROW(maxExpression->addChild(minExpression),
               tetrisched::exceptions::ExpressionConstructionException)
      << "MaxExpression should not allow MinExpression children.";
}

#ifdef _TETRISCHED_WITH_CPLEX_
/// Checks that for two trivially satisfiable expressions, the less than
/// expression is only satisfied if their times are ordered.
TEST(Expression, TestLessThanEnforcesOrdering) {
  // Construct the Partition.
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>(1, "partition1", 1);
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
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
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

  // Ensure that the LessThanExpression's variables are correct.
  EXPECT_TRUE(result->utility.has_value()) << "Utility should be set.";
  EXPECT_EQ(1, result->utility.value()) << "Utility should be 1.";

  EXPECT_TRUE(result->startTime.has_value()) << "Start time should be set.";
  EXPECT_EQ(0, result->startTime.value()) << "Start time should be 0.";

  EXPECT_TRUE(result->endTime.has_value()) << "End time should be set.";
  EXPECT_EQ(300, result->endTime.value()) << "End time should be 100.";

  // Ensure that the Placements are correct.
  EXPECT_EQ(result->placements.size(), 2) << "There should be 2 placements.";
  EXPECT_TRUE(result->placements["task1"]->isPlaced())
      << "task1 should be placed";
  EXPECT_EQ(result->placements["task1"]->getPartitionAssignments()[0].first,
            partition->getPartitionId())
      << "task1 should be placed on partition.";
  EXPECT_EQ(result->placements["task1"]->getStartTime().value(), 0)
      << "task1 should start at 0.";

  EXPECT_TRUE(result->placements["task2"]->isPlaced())
      << "task2 should be placed";
  EXPECT_EQ(result->placements["task2"]->getPartitionAssignments()[0].first,
            partition->getPartitionId())
      << "task2 should be placed on partition.";
  EXPECT_EQ(result->placements["task2"]->getStartTime().value(), 200)
      << "task2 should start at 200.";
}

// Check that the STRL parsing for the MaxExpression enforces only
// one of the children to be true.
TEST(Expression, TestMaxExpressionEnforcesSingleChoice) {
  // Construct the Partition.
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>(1, "partition1", 2);
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
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
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
  // Construct the Partition.
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>(1, "partition1", 1);
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
  minChooseExpr->addChild(chooseTask1_1);
  minChooseExpr->addChild(chooseTask1_2);

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
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
  EXPECT_EQ(chooseTask1_1->getSolution().value()->utility.value(), 1)
      << "First Choose expression must be satisfied.";
  EXPECT_EQ(chooseTask1_2->getSolution().value()->utility.value(), 1)
      << "Second Choose expression must be satisfied.";
}

// Check that the STRL parsing for the MinExpression enforces no expression to
// be true if all children can't be satisfied.
TEST(Expression, TestMinExpressionEnforcesNoneSatisfied) {
  // Construct the Partition.
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>(1, "partition1", 1);
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
  minChooseExpr->addChild(chooseTask1_1);
  minChooseExpr->addChild(chooseTask1_2);

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
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
  EXPECT_EQ(chooseTask1_1->getSolution().value()->utility.value(), 0)
      << "First Choose expression must not be satisfied.";
  EXPECT_EQ(chooseTask1_2->getSolution().value()->utility.value(), 0)
      << "Second Choose expression must not be satisfied.";
}

TEST(Expression, TestScaleExpressionDoublesUtility) {
  // Construct the Partition.
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>(1, "partition1", 1);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition});

  // Construct the choice for a task.
  tetrisched::ExpressionPtr chooseExpression =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 1, 0,
                                                     10);
  // Scale the utility by 2.
  tetrisched::ExpressionPtr scaleExpression =
      std::make_shared<tetrisched::ScaleExpression>("TEST_SCALE", 4);
  scaleExpression->addChild(chooseExpression);

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
  objectiveExpression->addChild(std::move(scaleExpression));

  // Construct a Solver.
  tetrisched::CPLEXSolver cplexSolver = tetrisched::CPLEXSolver();
  auto solverModelPtr = cplexSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap;
  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);

  // Translate and solve the model.
  cplexSolver.translateModel();
  cplexSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);
  EXPECT_TRUE(result->utility) << "Result should have some utility.";
  EXPECT_EQ(4, result->utility.value())
      << "The utility after Scale should be 4.";
  EXPECT_EQ(1, chooseExpression->getSolution().value()->utility.value())
      << "The utility for the individual Choose should be 1.";
}
#endif
