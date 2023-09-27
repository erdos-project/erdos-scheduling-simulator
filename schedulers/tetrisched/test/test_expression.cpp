#include <gtest/gtest.h>

#include "tetrisched/Expression.hpp"
#include "tetrisched/Partition.hpp"

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


TEST(Expression, TestMinExpressionIsNOTLeaf) {
  tetrisched::TaskPtr task = std::make_shared<tetrisched::Task>(1, "task1");
  tetrisched::Partitions partitions = tetrisched::Partitions();
  tetrisched::ExpressionPtr chooseExpression =
      std::make_unique<tetrisched::ChooseExpression>(task, partitions, 0, 0,
                                                     10);
  tetrisched::ExpressionPtr chooseExpression2 =
      std::make_unique<tetrisched::ChooseExpression>(task, partitions, 0, 0,
                                                     10);

  std::unique_ptr<tetrisched::MinExpression> minExpression =
      std::make_unique<tetrisched::MinExpression>();
  minExpression->addChild(std::move(chooseExpression2));
  minExpression->addChild(std::move(chooseExpression));
  EXPECT_TRUE(minExpression->GetNumChildren() == 2);
}
