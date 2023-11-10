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
      "task1", partitions, 0, 0, 10, 1);
  auto chooseExpression2 = std::make_unique<tetrisched::ChooseExpression>(
      "task1", partitions, 0, 0, 10, 1);
  EXPECT_THROW(chooseExpression->addChild(std::move(chooseExpression2)),
               tetrisched::exceptions::ExpressionConstructionException);
}

TEST(Expression, TestMinExpressionIsNOTLeaf) {
  tetrisched::Partitions partitions = tetrisched::Partitions();
  tetrisched::ExpressionPtr chooseExpression =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 0, 0,
                                                     10, 1);
  tetrisched::ExpressionPtr chooseExpression2 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 0, 0,
                                                     10, 1);

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
                                                     10, 1);
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

/// Checks that the time bound ranges are calculated correctly.
TEST(Expression, TestExpressionTimeBoundRanges) {
  // Test the time bound ranges for a ChooseExpression.
  tetrisched::Partitions partitions = tetrisched::Partitions();
  tetrisched::ExpressionPtr chooseExpression_1 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 0, 10,
                                                     20, 1);
  auto timeBounds = chooseExpression_1->getTimeBounds();

  EXPECT_EQ(timeBounds.startTimeRange.first, 10)
      << "Start time for choose should be 10.";
  EXPECT_EQ(timeBounds.startTimeRange.second, 10)
      << "Start time for choose should be 10.";
  EXPECT_EQ(timeBounds.endTimeRange.first, 30)
      << "End time for choose should be 30.";
  EXPECT_EQ(timeBounds.endTimeRange.second, 30)
      << "End time for choose should be 30.";

  // Test the time bound ranges for a MaxExpression.
  tetrisched::ExpressionPtr chooseExpression_2 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 0, 20,
                                                     20, 1);
  tetrisched::ExpressionPtr chooseExpression_3 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 0, 30,
                                                     20, 1);
  tetrisched::ExpressionPtr maxExpression_1 =
      std::make_shared<tetrisched::MaxExpression>("MaxTask1");
  maxExpression_1->addChild(chooseExpression_1);
  maxExpression_1->addChild(chooseExpression_2);
  maxExpression_1->addChild(chooseExpression_3);
  timeBounds = maxExpression_1->getTimeBounds();

  EXPECT_EQ(timeBounds.startTimeRange.first, 10)
      << "Start time for max should be 10.";
  EXPECT_EQ(timeBounds.startTimeRange.second, 30)
      << "Start time for max should be 30.";
  EXPECT_EQ(timeBounds.endTimeRange.first, 30)
      << "Start time for max should be 30.";
  EXPECT_EQ(timeBounds.endTimeRange.second, 50)
      << "Start time for max should be 50.";

  // Test the time bounds for a LessThanExpression.
  tetrisched::ExpressionPtr chooseExpression_4 =
      std::make_shared<tetrisched::ChooseExpression>("task2", partitions, 0, 30,
                                                     30, 1);
  tetrisched::ExpressionPtr chooseExpression_5 =
      std::make_shared<tetrisched::ChooseExpression>("task2", partitions, 0, 40,
                                                     30, 1);
  tetrisched::ExpressionPtr maxExpression_2 =
      std::make_shared<tetrisched::MaxExpression>("MaxTask2");
  maxExpression_2->addChild(chooseExpression_4);
  maxExpression_2->addChild(chooseExpression_5);
  tetrisched::ExpressionPtr lessThanExpression =
      std::make_shared<tetrisched::LessThanExpression>("LessThanTask1_2");
  lessThanExpression->addChild(maxExpression_1);
  lessThanExpression->addChild(maxExpression_2);
  timeBounds = lessThanExpression->getTimeBounds();

  EXPECT_EQ(timeBounds.startTimeRange.first, 10)
      << "Start time for less than should be 10.";
  EXPECT_EQ(timeBounds.startTimeRange.second, 30)
      << "Start time for less than should be 30.";
  EXPECT_EQ(timeBounds.endTimeRange.first, 60)
      << "End time for less than should be 60.";
  EXPECT_EQ(timeBounds.endTimeRange.second, 70)
      << "End time for less than should be 70.";
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
                                                     100, 1);
  tetrisched::ExpressionPtr chooseTask2 =
      std::make_shared<tetrisched::ChooseExpression>("task2", partitions, 1,
                                                     200, 100, 1);

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
  cplexSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);

  // Ensure that the LessThanExpression's variables are correct.
  EXPECT_TRUE(result->utility.has_value()) << "Utility should be set.";
  EXPECT_EQ(2, result->utility.value()) << "Utility should be 2.";

  EXPECT_TRUE(result->startTime.has_value()) << "Start time should be set.";
  EXPECT_EQ(0, result->startTime.value()) << "Start time should be 0.";

  EXPECT_TRUE(result->endTime.has_value()) << "End time should be set.";
  EXPECT_EQ(300, result->endTime.value()) << "End time should be 100.";

  // Ensure that the Placements are correct.
  EXPECT_EQ(result->placements.size(), 2) << "There should be 2 placements.";
  EXPECT_TRUE(result->placements["task1"]->isPlaced())
      << "task1 should be placed";

  auto allocationsForTask1 =
      result->placements["task1"]->getPartitionAllocations();
  EXPECT_TRUE(allocationsForTask1.find(partition->getPartitionId()) !=
              allocationsForTask1.end())
      << "task1 should be placed on partition 1.";
  EXPECT_EQ(result->placements["task1"]->getStartTime().value(), 0)
      << "task1 should start at 0.";

  EXPECT_TRUE(result->placements["task2"]->isPlaced())
      << "task2 should be placed";
  auto allocationsForTask2 =
      result->placements["task2"]->getPartitionAllocations();
  EXPECT_TRUE(allocationsForTask2.find(partition->getPartitionId()) !=
              allocationsForTask2.end())
      << "task2 should be placed on partition 1.";
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
                                                     100, 1);
  tetrisched::ExpressionPtr chooseTask1_2 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 1,
                                                     100, 100, 1);

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
  cplexSolver.exportModel("testMaxExpressionEnforcesSingleChoice_2.lp");
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
                                                     0, 100, 1);
  tetrisched::ExpressionPtr chooseTask1_2 =
      std::make_shared<tetrisched::ChooseExpression>("task1_2", partitions, 1,
                                                     100, 100, 1);

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
  EXPECT_EQ(2, result->utility.value()) << "Both choices should be satisfied.";
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
                                                     0, 100, 1);
  tetrisched::ExpressionPtr chooseTask1_2 =
      std::make_shared<tetrisched::ChooseExpression>("task1_2", partitions, 1,
                                                     50, 100, 1);

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
                                                     10, 1);
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

TEST(Expression, TestAllocationExpressionFailsChoice) {
  // Construct the Partition.
  tetrisched::PartitionPtr partition =
      std::make_shared<tetrisched::Partition>(1, "partition1", 1);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition});

  // Allocate one Task on the Partition.
  std::vector<std::pair<tetrisched::PartitionPtr, uint32_t>>
      partitionAssignments;
  partitionAssignments.push_back({partition, 1});
  tetrisched::ExpressionPtr allocationExpression =
      std::make_shared<tetrisched::AllocationExpression>(
          "task1", partitionAssignments, 0, 50);

  // Construct the Choice for a second task.
  tetrisched::ExpressionPtr chooseExpression =
      std::make_shared<tetrisched::ChooseExpression>("task2", partitions, 1, 10,
                                                     20, 1);

  // Try to meet both of the Expressions.
  tetrisched::ExpressionPtr minExpression =
      std::make_shared<tetrisched::MinExpression>("MinExpression");
  minExpression->addChild(allocationExpression);
  minExpression->addChild(chooseExpression);

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
  objectiveExpression->addChild(minExpression);

  // Construct a Solver.
  tetrisched::CPLEXSolver cplexSolver = tetrisched::CPLEXSolver();
  auto solverModelPtr = cplexSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap(10);
  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);

  // Translate and solve the model.
  cplexSolver.translateModel();
  solverModelPtr->exportModel("testAllocationExpressionFailsChoice.lp");
  cplexSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);
  EXPECT_TRUE(result->utility) << "Result should have some utility.";
  EXPECT_EQ(0, result->utility.value())
      << "The utility for the Expressions should be 0";
}
#endif

#ifdef _TETRISCHED_WITH_GUROBI_
#include "tetrisched/GurobiSolver.hpp"

/// Test that a MalleableChooseExpression correctly generates the ILP.
TEST(Expression, TestMalleableChooseExpressionConstruction) {
  // Construct the Partition.
  tetrisched::PartitionPtr partition1 =
      std::make_shared<tetrisched::Partition>(1, "partition1", 5);
  //   tetrisched::PartitionPtr partition2 =
  //       std::make_shared<tetrisched::Partition>(2, "partition2", 5);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition1});

  // Construct the MalleableChooseExpression.
  tetrisched::ExpressionPtr malleableChooseExpression =
      std::make_shared<tetrisched::MalleableChooseExpression>(
          "task1", partitions, 15, 5, 10, 1, 1);

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
  objectiveExpression->addChild(malleableChooseExpression);

  // Construct a Solver.
  tetrisched::GurobiSolver gurobiSolver = tetrisched::GurobiSolver();
  auto solverModelPtr = gurobiSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap;
  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);
  solverModelPtr->exportModel("testMalleableChooseExpression.lp");
}

/// Test that a MalleableChooseExpression constructs variable space-time
/// rectangles.
TEST(Expression, TestMalleableChooseExpressionConstructsVariableRectangles) {
  // Construct the Partition.
  tetrisched::PartitionPtr partition1 =
      std::make_shared<tetrisched::Partition>(1, "partition1", 5);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition1});

  // Construct an AllocationExpression to allocate the task to the partition.
  std::vector<std::pair<tetrisched::PartitionPtr, uint32_t>>
      partitionAssignments;
  partitionAssignments.push_back(std::make_pair(partition1, 3));
  tetrisched::ExpressionPtr allocationExpression =
      std::make_shared<tetrisched::AllocationExpression>(
          "task1", partitionAssignments, 5, 2);

  // Construct a MalleableChooseExpression to allocate around the previous
  // usage.
  tetrisched::ExpressionPtr malleableChooseExpression =
      std::make_shared<tetrisched::MalleableChooseExpression>(
          "task2", partitions, 10, 4, 10, 1, 1);

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
  objectiveExpression->addChild(allocationExpression);
  objectiveExpression->addChild(malleableChooseExpression);

  // Construct a Solver.
  tetrisched::GurobiSolver gurobiSolver = tetrisched::GurobiSolver();
  auto solverModelPtr = gurobiSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap(2);
  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);
  solverModelPtr->exportModel("testMalleableChooseVariableRectangle.lp");

  // Translate and Solve the model.
  gurobiSolver.translateModel();
  gurobiSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);
  EXPECT_TRUE(result->utility) << "Result should have some utility.";
  EXPECT_EQ(2, result->utility.value()) << "The utility should be 2.";
  auto task2Placement = result->placements["task2"]->getPartitionAllocations();
  EXPECT_TRUE(task2Placement.find(partition1->getPartitionId()) !=
              task2Placement.end())
      << "task2 should be placed on partition 1.";
  auto partition1Allocation = task2Placement[partition1->getPartitionId()];
  uint32_t totalAllocationSum = 0;
  for (auto& [time, allocation] : partition1Allocation) {
    totalAllocationSum += allocation;
  }
  EXPECT_EQ(totalAllocationSum, 10)
      << "The total allocation on partition 1 should be 10.";
}

TEST(Expression, TestNonOverlappingChooseIsAllowed) {
  // Construct the Partition.
  tetrisched::PartitionPtr partition1 =
      std::make_shared<tetrisched::Partition>(1, "partition1", 5);
  tetrisched::Partitions partitions = tetrisched::Partitions({partition1});

  // Construct choices for two task.
  tetrisched::ExpressionPtr chooseTask1 =
      std::make_shared<tetrisched::ChooseExpression>("task1", partitions, 5, 0,
                                                     40, 1);
  tetrisched::ExpressionPtr chooseTask2 =
      std::make_shared<tetrisched::ChooseExpression>("task2", partitions, 5, 0,
                                                     40, 1);

  // Construct a MinExpression.
  tetrisched::ExpressionPtr minExpression =
      std::make_shared<tetrisched::MinExpression>("minChooseBothTasks");
  minExpression->addChild(chooseTask1);
  minExpression->addChild(chooseTask2);

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_shared<tetrisched::ObjectiveExpression>("TestObjective");
  objectiveExpression->addChild(minExpression);

  // Construct a Solver.
  tetrisched::GurobiSolver gurobiSolver = tetrisched::GurobiSolver();
  auto solverModelPtr = gurobiSolver.getModel();

  // Construct a CapacityConstraintMap and parse the expression tree.
  tetrisched::CapacityConstraintMap capacityConstraintMap(100, true);
  auto _ = objectiveExpression->parse(solverModelPtr, partitions,
                                      capacityConstraintMap, 0);
  solverModelPtr->exportModel("testNonOverlappingChooseIsAllowed.lp");

  // Translate and Solve the model.
  gurobiSolver.translateModel();
  gurobiSolver.solveModel();

  auto result = objectiveExpression->populateResults(solverModelPtr);
  EXPECT_TRUE(result->utility) << "Result should have some utility.";
  EXPECT_EQ(2, result->utility.value()) << "The utility should be 2.";
  auto task1Placement = result->placements["task1"];
  EXPECT_TRUE(task1Placement->isPlaced()) << "task1 should be placed.";
  auto task2Placement = result->placements["task2"];
  EXPECT_TRUE(task2Placement->isPlaced()) << "task2 should be placed.";
  EXPECT_EQ(task1Placement->getStartTime().value(), 0)
      << "task1 should start at 0.";
  EXPECT_EQ(task2Placement->getStartTime().value(), 0)
      << "task2 should start at 0.";
}
#endif
