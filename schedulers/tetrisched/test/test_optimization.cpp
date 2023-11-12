#include <gtest/gtest.h>

#include "tetrisched/OptimizationPasses.hpp"

TEST(OptimizationTest, TestBasicCriticalPathOptimizationPass) {
  // Create an OptimizationPass object.
  tetrisched::CriticalPathOptimizationPass optimizationPass;

  // Run the OptimizationPass on a simple STRL expression.
  tetrisched::Partitions partitions = tetrisched::Partitions();
  tetrisched::ExpressionPtr chooseExpression_1 =
      std::make_shared<tetrisched::ChooseExpression>("task1_1", partitions, 0, 10,
                                                     20, 1);
  tetrisched::ExpressionPtr chooseExpression_2 =
      std::make_shared<tetrisched::ChooseExpression>("task1_2", partitions, 0, 20,
                                                     20, 1);
  tetrisched::ExpressionPtr chooseExpression_3 =
      std::make_shared<tetrisched::ChooseExpression>("task1_3", partitions, 0, 30,
                                                     20, 1);
  tetrisched::ExpressionPtr maxExpression_1 =
      std::make_shared<tetrisched::MaxExpression>("MaxTask1");
  maxExpression_1->addChild(chooseExpression_1);
  maxExpression_1->addChild(chooseExpression_2);
  maxExpression_1->addChild(chooseExpression_3);

  tetrisched::ExpressionPtr chooseExpression_4 =
      std::make_shared<tetrisched::ChooseExpression>("task2_1", partitions, 0, 10,
                                                     30, 1);
  tetrisched::ExpressionPtr chooseExpression_5 =
      std::make_shared<tetrisched::ChooseExpression>("task2_2", partitions, 0, 20,
                                                     30, 1);
  tetrisched::ExpressionPtr chooseExpression_6 =
      std::make_shared<tetrisched::ChooseExpression>("task2_3", partitions, 0, 30,
                                                     30, 1);
  tetrisched::ExpressionPtr maxExpression_2 =
      std::make_shared<tetrisched::MaxExpression>("MaxTask2");
  maxExpression_2->addChild(chooseExpression_4);
  maxExpression_2->addChild(chooseExpression_5);
  maxExpression_2->addChild(chooseExpression_6);

  tetrisched::ExpressionPtr lessThanExpression =
      std::make_shared<tetrisched::LessThanExpression>("LessThan");
  lessThanExpression->addChild(maxExpression_1);
  lessThanExpression->addChild(maxExpression_2);
  lessThanExpression->exportToDot("PreOptimizationPass.dot");

  tetrisched::CapacityConstraintMap capacityConstraintMap(1);
  optimizationPass.runPass(lessThanExpression, capacityConstraintMap, std::nullopt);
  lessThanExpression->exportToDot("PostOptimizationPass.dot");
}
