#include "tetrisched/OptimizationPasses.hpp"

#include <stack>

namespace tetrisched {

/* Methods for OptimizationPass */
OptimizationPass::OptimizationPass(std::string name) : name(name) {}

/* Methods for CriticalPathOptimizationPass */
CriticalPathOptimizationPass::CriticalPathOptimizationPass()
    : OptimizationPass("CriticalPathOptimizationPass") {}

void CriticalPathOptimizationPass::runPass(ExpressionPtr strlExpression) {
  if (strlExpression->getType() == ExpressionType::EXPR_LESSTHAN) {
    auto children = strlExpression->getChildren();
    if (children[0]->getType() == ExpressionType::EXPR_MAX &&
        children[1]->getType() == ExpressionType::EXPR_MAX) {
      // Get the children's start and end time range.
      auto leftChildTimeBounds = children[0]->getTimeBounds();
      TETRISCHED_DEBUG("Left Child's Time Bounds: ("
                       << leftChildTimeBounds.startTimeRange.first << ", "
                       << leftChildTimeBounds.startTimeRange.second << ") ("
                       << leftChildTimeBounds.endTimeRange.first << ", "
                       << leftChildTimeBounds.endTimeRange.second << ")")
      auto rightChildTimeBounds = children[1]->getTimeBounds();
      TETRISCHED_DEBUG("Right Child's Time Bounds: ("
                       << rightChildTimeBounds.startTimeRange.first << ", "
                       << rightChildTimeBounds.startTimeRange.second << ") ("
                       << rightChildTimeBounds.endTimeRange.first << ", "
                       << rightChildTimeBounds.endTimeRange.second << ")")

      // The right child cannot start any earlier than the earliest
      // time that the left child can finish.
      rightChildTimeBounds.startTimeRange.first =
          std::max(rightChildTimeBounds.startTimeRange.first,
                   leftChildTimeBounds.endTimeRange.first);

      // The left child cannot end any earlier than the latest time
      // that the right child needs to start to finish its deadline.
      leftChildTimeBounds.endTimeRange.second =
          std::min(leftChildTimeBounds.endTimeRange.second,
                   rightChildTimeBounds.startTimeRange.second);

      TETRISCHED_DEBUG("Updated Left Child's Time Bounds: ("
                       << leftChildTimeBounds.startTimeRange.first << ", "
                       << leftChildTimeBounds.startTimeRange.second << ") ("
                       << leftChildTimeBounds.endTimeRange.first << ", "
                       << leftChildTimeBounds.endTimeRange.second << ")")
      TETRISCHED_DEBUG("Updated Right Child's Time Bounds: ("
                       << rightChildTimeBounds.startTimeRange.first << ", "
                       << rightChildTimeBounds.startTimeRange.second << ") ("
                       << rightChildTimeBounds.endTimeRange.first << ", "
                       << rightChildTimeBounds.endTimeRange.second << ")")

      // Save the updated bounds.
      expressionTimeBoundMap[children[0]->getId()] = leftChildTimeBounds;
      expressionTimeBoundMap[children[1]->getId()] = rightChildTimeBounds;

      // Purge the children that do not fit within the time bounds.
      for (auto& child : children) {
        for (auto& grandChild : child->getChildren()) {
          auto grandChildBounds = grandChild->getTimeBounds();
          auto childBounds = expressionTimeBoundMap[child->getId()];
          if (grandChildBounds.startTimeRange.first <
                  childBounds.startTimeRange.first ||
              grandChildBounds.startTimeRange.second >
                  childBounds.startTimeRange.second ||
              grandChildBounds.endTimeRange.first <
                  childBounds.endTimeRange.first ||
              grandChildBounds.endTimeRange.second >
                  childBounds.endTimeRange.second) {
            TETRISCHED_DEBUG("Purging grandchild "
                             << grandChild->getId() << "("
                             << grandChild->getName() << ") from "
                             << child->getId() << "(" << child->getName()
                             << ")")
            child->removeChild(grandChild);
          }
        }
      }
    }
  }
}

/* Methods for OptimizationPassRunner */
OptimizationPassRunner::OptimizationPassRunner() {
  // Register the Critical Path optimization pass.
  registeredPasses.push_back(std::make_shared<CriticalPathOptimizationPass>());
}

void OptimizationPassRunner::runPasses(ExpressionPtr strlExpression) {
  // Go through each node in the STRL tree and run the pass.
  std::stack<ExpressionPtr> expressionStack;
  expressionStack.push(strlExpression);

  while (!expressionStack.empty()) {
    auto expression = expressionStack.top();
    expressionStack.pop();

    // Run the optimization pass on the current node.
    for (auto& pass : registeredPasses) {
      pass->runPass(expression);
    }

    // Add the children to the stack.
    for (auto& child : expression->getChildren()) {
      expressionStack.push(child);
    }
  }
}
}  // namespace tetrisched
