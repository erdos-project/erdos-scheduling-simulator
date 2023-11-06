#include "tetrisched/OptimizationPasses.hpp"

#include <queue>
#include <stack>

namespace tetrisched {

/* Methods for OptimizationPass */
OptimizationPass::OptimizationPass(std::string name, OptimizationPassType type)
    : name(name), type(type) {}

OptimizationPassType OptimizationPass::getType() const { return type; }

/* Methods for CriticalPathOptimizationPass */
CriticalPathOptimizationPass::CriticalPathOptimizationPass()
    : OptimizationPass("CriticalPathOptimizationPass",
                       OptimizationPassType::PRE_TRANSLATION_PASS) {}

void CriticalPathOptimizationPass::computeTimeBounds(ExpressionPtr expression) {
  /* Do a Post-Order Traversal of the DAG. */
  std::stack<ExpressionPtr> firstStack;
  std::stack<ExpressionPtr> secondStack;
  firstStack.push(expression);

  while (!firstStack.empty()) {
    // Move the expression to the second stack.
    auto currentExpression = firstStack.top();
    firstStack.pop();
    secondStack.push(currentExpression);

    // Add the children to the first stack.
    auto expressionChildren = currentExpression->getChildren();
    for (auto child = expressionChildren.rbegin();
         child != expressionChildren.rend(); ++child) {
      firstStack.push(*child);
    }
  }

  // A Post-Order Traversal will now be the order in which
  // the expressions are popped from the second stack.
  std::set<std::string> visitedExpressions;
  while (!secondStack.empty()) {
    auto currentExpression = secondStack.top();
    secondStack.pop();
    if (visitedExpressions.find(currentExpression->getId()) ==
        visitedExpressions.end()) {
      visitedExpressions.insert(currentExpression->getId());
    } else {
      continue;
    }
    TETRISCHED_DEBUG("[POSTORDER] Computing time bounds for "
                     << currentExpression->getId() << "("
                     << currentExpression->getName() << ")")

    // If this is an ordering expression, we can cut some time bounds here.
    if (currentExpression->getType() == ExpressionType::EXPR_LESSTHAN) {
      auto leftChild = currentExpression->getChildren()[0];
      if (expressionTimeBoundMap.find(leftChild->getId()) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "Left child " + leftChild->getId() + "(" + leftChild->getName() +
            ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto& leftChildTimeBounds = expressionTimeBoundMap[leftChild->getId()];
      TETRISCHED_DEBUG("Left child " << leftChild->getId() << "("
                                     << leftChild->getName()
                                     << ") has time bounds: "
                                     << leftChildTimeBounds.toString())

      auto rightChild = currentExpression->getChildren()[1];
      if (expressionTimeBoundMap.find(rightChild->getId()) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "Right child " + rightChild->getId() + "(" + rightChild->getName() +
            ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto& rightChildTimeBounds = expressionTimeBoundMap[rightChild->getId()];
      TETRISCHED_DEBUG("Right child " << rightChild->getId() << "("
                                      << rightChild->getName()
                                      << ") has time bounds: "
                                      << rightChildTimeBounds.toString())

      // The right child cannot start any earlier than the earliest
      // time that the left child can finish.
      rightChildTimeBounds.startTimeRange.first =
          std::max(rightChildTimeBounds.startTimeRange.first,
                   leftChildTimeBounds.endTimeRange.first);
      TETRISCHED_DEBUG("Right child " << rightChild->getId() << "("
                                      << rightChild->getName()
                                      << ") has updated time bounds: "
                                      << rightChildTimeBounds.toString())

      // The left child cannot end any later than the earliest time
      // that the right child needs to start to finish its deadline.
      leftChildTimeBounds.endTimeRange.second =
          std::min(leftChildTimeBounds.endTimeRange.second,
                   rightChildTimeBounds.startTimeRange.second);
      TETRISCHED_DEBUG("Left child " << leftChild->getId() << "("
                                     << leftChild->getName()
                                     << ") has updated time bounds: "
                                     << leftChildTimeBounds.toString())
    }
    // Assign the bounds of the Expression.
    expressionTimeBoundMap[currentExpression->getId()] =
        currentExpression->getTimeBounds();
    TETRISCHED_DEBUG(
        "The time bound for "
        << currentExpression->getId() << "(" << currentExpression->getName()
        << ") is "
        << expressionTimeBoundMap[currentExpression->getId()].toString())
  }
}

void CriticalPathOptimizationPass::pushDownTimeBounds(
    ExpressionPtr expression) {
  /* Do a Reverse Post-Order Traversal of the DAG. */
  std::stack<ExpressionPtr> traversalStack;
  std::queue<ExpressionPtr> traversalQueue;
  traversalStack.push(expression);

  while (!traversalStack.empty()) {
    // Move the expression to the second stack.
    auto currentExpression = traversalStack.top();
    traversalStack.pop();
    traversalQueue.push(currentExpression);

    // Add the children to the first stack.
    auto expressionChildren = currentExpression->getChildren();
    for (auto child = expressionChildren.rbegin();
         child != expressionChildren.rend(); ++child) {
      traversalStack.push(*child);
    }
  }

  // A Reverse Post-Order Traversal will now be the order in which
  // the expressions are popped from the queue.
  while (!traversalQueue.empty()) {
    auto currentExpression = traversalQueue.front();
    traversalQueue.pop();

    if (expressionTimeBoundMap.find(currentExpression->getId()) ==
        expressionTimeBoundMap.end()) {
      throw exceptions::RuntimeException(
          "Expression " + currentExpression->getId() + "(" +
          currentExpression->getName() + ") does not have time bounds.");
    }

    auto expressionBounds = expressionTimeBoundMap[currentExpression->getId()];
    TETRISCHED_DEBUG("[REVERSE_POSTORDER] Pushing down bounds for "
                     << currentExpression->getId() << "("
                     << currentExpression->getName()
                     << "): " << expressionBounds.toString())

    if (currentExpression->getType() == ExpressionType::EXPR_LESSTHAN) {
      auto leftChild = currentExpression->getChildren()[0];
      if (expressionTimeBoundMap.find(leftChild->getId()) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "Left child " + leftChild->getId() + "(" + leftChild->getName() +
            ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto& leftChildTimeBounds = expressionTimeBoundMap[leftChild->getId()];
      TETRISCHED_DEBUG("Left child " << leftChild->getId() << "("
                                     << leftChild->getName()
                                     << ") has time bounds: "
                                     << leftChildTimeBounds.toString())

      auto rightChild = currentExpression->getChildren()[1];
      if (expressionTimeBoundMap.find(rightChild->getId()) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "Right child " + rightChild->getId() + "(" + rightChild->getName() +
            ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto& rightChildTimeBounds = expressionTimeBoundMap[rightChild->getId()];
      TETRISCHED_DEBUG("Right child " << rightChild->getId() << "("
                                      << rightChild->getName()
                                      << ") has time bounds: "
                                      << rightChildTimeBounds.toString())

      // Update the time bounds.
      if (leftChildTimeBounds.startTimeRange.first <
          expressionBounds.startTimeRange.first) {
        leftChildTimeBounds.startTimeRange.first =
            expressionBounds.startTimeRange.first;
      }
      if (leftChildTimeBounds.startTimeRange.second >
          expressionBounds.startTimeRange.second) {
        leftChildTimeBounds.startTimeRange.second =
            expressionBounds.startTimeRange.second;
      }
      if (rightChildTimeBounds.endTimeRange.first <
          expressionBounds.endTimeRange.first) {
        rightChildTimeBounds.endTimeRange.first =
            expressionBounds.endTimeRange.first;
      }
      if (rightChildTimeBounds.endTimeRange.second >
          expressionBounds.endTimeRange.second) {
        rightChildTimeBounds.endTimeRange.second =
            expressionBounds.endTimeRange.second;
      }

      TETRISCHED_DEBUG("Left child " << leftChild->getId() << "("
                                     << leftChild->getName()
                                     << ") has updated time bounds: "
                                     << leftChildTimeBounds.toString())

      TETRISCHED_DEBUG("Right child " << rightChild->getId() << "("
                                      << rightChild->getName()
                                      << ") has updated time bounds: "
                                      << rightChildTimeBounds.toString())
    } else {
      // Iterate through the children and tighten the bounds, if possible.
      for (auto& child : currentExpression->getChildren()) {
        if (expressionTimeBoundMap.find(child->getId()) ==
            expressionTimeBoundMap.end()) {
          throw exceptions::RuntimeException(
              "Child " + child->getId() + "(" + child->getName() + ") of " +
              currentExpression->getId() + "(" + currentExpression->getName() +
              ") does not have time bounds.");
        }
        auto& childTimeBounds = expressionTimeBoundMap[child->getId()];
        TETRISCHED_DEBUG("Child "
                         << child->getId() << "(" << child->getName()
                         << ") has time bounds: " << childTimeBounds.toString())
        if (childTimeBounds.startTimeRange.first <
            expressionBounds.startTimeRange.first) {
          childTimeBounds.startTimeRange.first =
              expressionBounds.startTimeRange.first;
        }
        if (childTimeBounds.startTimeRange.second >
            expressionBounds.startTimeRange.second) {
          childTimeBounds.startTimeRange.second =
              expressionBounds.startTimeRange.second;
        }
        if (childTimeBounds.endTimeRange.first <
            expressionBounds.endTimeRange.first) {
          childTimeBounds.endTimeRange.first =
              expressionBounds.endTimeRange.first;
        }
        if (childTimeBounds.endTimeRange.second >
            expressionBounds.endTimeRange.second) {
          childTimeBounds.endTimeRange.second =
              expressionBounds.endTimeRange.second;
        }
        TETRISCHED_DEBUG("Child " << child->getId() << "(" << child->getName()
                                  << ") has updated time bounds: "
                                  << childTimeBounds.toString())
      }
    }
  }
}

void CriticalPathOptimizationPass::purgeNodes(ExpressionPtr expression) {
  /* Do a Post-Order Traversal of the DAG. */
  std::stack<ExpressionPtr> firstStack;
  std::stack<ExpressionPtr> secondStack;
  firstStack.push(expression);

  while (!firstStack.empty()) {
    // Move the expression to the second stack.
    auto currentExpression = firstStack.top();
    firstStack.pop();
    secondStack.push(currentExpression);

    // Add the children to the first stack.
    auto expressionChildren = currentExpression->getChildren();
    for (auto child = expressionChildren.rbegin();
         child != expressionChildren.rend(); ++child) {
      firstStack.push(*child);
    }
  }

  // A Post-Order Traversal will now be the order in which
  // the expressions are popped from the second stack.
  std::set<std::string> visitedExpressions;
  std::set<std::string> purgedExpressions;
  while (!secondStack.empty()) {
    auto currentExpression = secondStack.top();
    secondStack.pop();
    if (visitedExpressions.find(currentExpression->getId()) ==
        visitedExpressions.end()) {
      visitedExpressions.insert(currentExpression->getId());
    } else {
      continue;
    }
    TETRISCHED_DEBUG("[POSTORDER] Attending to nodes for "
                     << currentExpression->getId() << "("
                     << currentExpression->getName() << ")")

    if (expressionTimeBoundMap.find(currentExpression->getId()) ==
        expressionTimeBoundMap.end()) {
      throw exceptions::RuntimeException(
          "Expression " + currentExpression->getId() + "(" +
          currentExpression->getName() + ") does not have time bounds.");
    }

    if (currentExpression->getNumChildren() == 0) {
      // If this is a leaf node, we check if it can be purged.
      auto newTimeBounds = expressionTimeBoundMap[currentExpression->getId()];
      auto originalTimeBounds = currentExpression->getTimeBounds();
      TETRISCHED_DEBUG("Original Time bounds for "
                       << currentExpression->getId() << "("
                       << currentExpression->getName()
                       << "): " << originalTimeBounds.toString())

      TETRISCHED_DEBUG("New Time bounds for "
                       << currentExpression->getId() << "("
                       << currentExpression->getName()
                       << "): " << newTimeBounds.toString())
      if (originalTimeBounds.startTimeRange.first <
              newTimeBounds.startTimeRange.first ||
          originalTimeBounds.startTimeRange.second >
              newTimeBounds.startTimeRange.second ||
          originalTimeBounds.endTimeRange.first <
              newTimeBounds.endTimeRange.first ||
          originalTimeBounds.endTimeRange.second >
              newTimeBounds.endTimeRange.second) {
        purgedExpressions.insert(currentExpression->getId());
        TETRISCHED_DEBUG("Purging node " << currentExpression->getId() << "("
                                         << currentExpression->getName() << ")")
      }
    } else {
      // This is not a leaf node, so we need to remove the children that
      // need to be purged.
      std::vector<ExpressionPtr> purgedChildrens;
      for (auto& child : currentExpression->getChildren()) {
        if (purgedExpressions.find(child->getId()) != purgedExpressions.end()) {
          purgedChildrens.push_back(child);
        }
      }
      for (auto& purgedChild : purgedChildrens) {
        currentExpression->removeChild(purgedChild);
      }

      if (currentExpression->getNumChildren() == 0 ||
          (currentExpression->getNumChildren() == 1 &&
           currentExpression->getType() == ExpressionType::EXPR_LESSTHAN)) {
        purgedExpressions.insert(currentExpression->getId());
        TETRISCHED_DEBUG("Purging node " << currentExpression->getId() << "("
                                         << currentExpression->getName() << ")")
      }
    }
  }
}

void CriticalPathOptimizationPass::runPass(
    ExpressionPtr strlExpression, CapacityConstraintMap& capacityConstraints) {
  /* Phase 1: We first do a bottom-up traversal of the tree to compute
  a tight bound for each node in the STRL tree. */
  computeTimeBounds(strlExpression);

  /* Phase 2: The previous phase computes the tight bounds but does not
  push them down necessarily. In this phase, we push the bounds down. */
  pushDownTimeBounds(strlExpression);

  /* Phase 3: The bounds have been pushed down now, we can do a bottom-up
  traversal and start purging nodes that cannot be satisfied. */
  purgeNodes(strlExpression);
}

/* Methods for CapacityConstraintMapPurgingOptimizationPass */
CapacityConstraintMapPurgingOptimizationPass::
    CapacityConstraintMapPurgingOptimizationPass()
    : OptimizationPass("CapacityConstraintMapPurgingOptimizationPass",
                       OptimizationPassType::POST_TRANSLATION_PASS) {}

void CapacityConstraintMapPurgingOptimizationPass::runPass(
    ExpressionPtr strlExpression, CapacityConstraintMap& capacityConstraints) {
  throw tetrisched::exceptions::RuntimeException("Not implemented yet!");
}

/* Methods for OptimizationPassRunner */
OptimizationPassRunner::OptimizationPassRunner() {
  // Register the Critical Path optimization pass.
  registeredPasses.push_back(std::make_shared<CriticalPathOptimizationPass>());
  // Register the CapacityConstraintMapPurging optimization pass.
  registeredPasses.push_back(
      std::make_shared<CapacityConstraintMapPurgingOptimizationPass>());
}

void OptimizationPassRunner::runPreTranslationPasses(
    ExpressionPtr strlExpression, CapacityConstraintMap& capacityConstraints) {
  // Run the registered optimization passes on the given STRL expression.
  for (auto& pass : registeredPasses) {
    if (pass->getType() == OptimizationPassType::PRE_TRANSLATION_PASS) {
      pass->runPass(strlExpression, capacityConstraints);
    }
  }
}

void OptimizationPassRunner::runPostTranslationPasses(
    ExpressionPtr strlExpression, CapacityConstraintMap& capacityConstraints) {
  // Run the registered optimization passes on the given STRL expression.
  for (auto& pass : registeredPasses) {
    if (pass->getType() == OptimizationPassType::POST_TRANSLATION_PASS) {
      pass->runPass(strlExpression, capacityConstraints);
    }
  }
}
}  // namespace tetrisched
