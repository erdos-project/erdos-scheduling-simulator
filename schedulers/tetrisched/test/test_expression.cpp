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
  tetrisched::TaskPtr task = std::make_shared<tetrisched::Task>(1, "task1");
  tetrisched::Partitions partitions = tetrisched::Partitions();
  tetrisched::ExpressionPtr chooseExpression =
      std::make_unique<tetrisched::ChooseExpression>(task, partitions, 0, 0,
                                                     10);
  tetrisched::ExpressionPtr chooseExpression2 =
      std::make_unique<tetrisched::ChooseExpression>(task, partitions, 0, 0,
                                                     10);
  EXPECT_THROW(chooseExpression->addChild(std::move(chooseExpression2)),
               tetrisched::exceptions::ExpressionConstructionException);
}

#ifdef _TETRISCHED_WITH_CPLEX_
/// Checks that for two trivially satisfiable expressions, the less than
/// expression is only satisfied if their times are ordered.
TEST(Expression, TestLessThanEnforcesOrdering) {
  // Construct the tasks.
  tetrisched::TaskPtr task1 = std::make_shared<tetrisched::Task>(1, "task1");
  tetrisched::TaskPtr task2 = std::make_shared<tetrisched::Task>(2, "task2");

  // Construct the Workers and a Partition.
  tetrisched::WorkerPtr worker1 =
      std::make_shared<tetrisched::Worker>(1, "worker1");
  tetrisched::WorkerPtr worker2 =
      std::make_shared<tetrisched::Worker>(2, "worker2");
  tetrisched::PartitionPtr partition = std::make_shared<tetrisched::Partition>(
      std::initializer_list<tetrisched::WorkerPtr>{worker1, worker2});
  tetrisched::Partitions partitions = tetrisched::Partitions({partition});

  // Construct the choice for the two tasks.
  tetrisched::ExpressionPtr chooseTask1 =
      std::make_unique<tetrisched::ChooseExpression>(task1, partitions, 1, 0,
                                                     100);
  tetrisched::ExpressionPtr chooseTask2 =
      std::make_unique<tetrisched::ChooseExpression>(task2, partitions, 1, 200,
                                                     100);

  // Construct the LessThan expression.
  tetrisched::ExpressionPtr lessThanExpression =
      std::make_unique<tetrisched::LessThanExpression>(
          "task1_less_than_task_2");
  lessThanExpression->addChild(std::move(chooseTask1));
  lessThanExpression->addChild(std::move(chooseTask2));

  // Construct an ObjectiveExpression.
  tetrisched::ExpressionPtr objectiveExpression =
      std::make_unique<tetrisched::ObjectiveExpression>(
          tetrisched::ObjectiveType::OBJ_MAXIMIZE);
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
  EXPECT_EQ(0, capacityConstraintMap.size()) << "Capacity map should be drained after translation.";
  cplexSolver.solveModel();

  auto result = objectiveExpression->solve(solverModelPtr);
  EXPECT_EQ(1, result->utility.value());
}
#endif
