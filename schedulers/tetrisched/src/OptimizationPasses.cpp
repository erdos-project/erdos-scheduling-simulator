#include "tetrisched/OptimizationPasses.hpp"

#include <chrono>
#include <queue>
#include <stack>

namespace tetrisched {

/* Methods for OptimizationPass */
OptimizationPass::OptimizationPass(std::string name, OptimizationPassType type)
    : name(name), type(type) {}

OptimizationPassType OptimizationPass::getType() const { return type; }

std::string OptimizationPass::getName() const { return name; }

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
    TETRISCHED_DEBUG("[" << name << "] "
                         << "Computing time bounds for "
                         << currentExpression->getId() << " ("
                         << currentExpression->getName() << ")")

    // If this is an ordering expression, we can cut some time bounds here.
    if (currentExpression->getType() == ExpressionType::EXPR_LESSTHAN) {
      auto leftChild = currentExpression->getChildren()[0];
      if (expressionTimeBoundMap.find(leftChild->getId()) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "[" + name + "] Left child " + leftChild->getId() + "(" +
            leftChild->getName() + ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto& leftChildTimeBounds = expressionTimeBoundMap[leftChild->getId()];
      TETRISCHED_DEBUG("[" << name << "] Left child " << leftChild->getId()
                           << " (" << leftChild->getName()
                           << ") has time bounds: "
                           << leftChildTimeBounds.toString())

      auto rightChild = currentExpression->getChildren()[1];
      if (expressionTimeBoundMap.find(rightChild->getId()) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "[" + name + "] Right child " + rightChild->getId() + " (" +
            rightChild->getName() + ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto& rightChildTimeBounds = expressionTimeBoundMap[rightChild->getId()];
      TETRISCHED_DEBUG("[" << name << "] Right child " << rightChild->getId()
                           << " (" << rightChild->getName()
                           << ") has time bounds: "
                           << rightChildTimeBounds.toString())

      // The right child cannot start any earlier than the earliest
      // time that the left child can finish.
      if (rightChildTimeBounds.startTimeRange.first <
          leftChildTimeBounds.endTimeRange.first) {
        rightChildTimeBounds.startTimeRange.first =
            leftChildTimeBounds.endTimeRange.first;
        rightChildTimeBounds.endTimeRange.first =
            rightChildTimeBounds.startTimeRange.first +
            rightChildTimeBounds.duration;
        rightChild->setTimeBounds(rightChildTimeBounds);
        TETRISCHED_DEBUG("[" << name << "] Right child " << rightChild->getId()
                             << " (" << rightChild->getName()
                             << ") has updated time bounds: "
                             << rightChildTimeBounds.toString())
      }

      // The left child cannot end any later than the earliest time
      // that the right child needs to start to finish its deadline.
      if (leftChildTimeBounds.endTimeRange.second >
          rightChildTimeBounds.startTimeRange.second) {
        leftChildTimeBounds.endTimeRange.second =
            rightChildTimeBounds.startTimeRange.second;
        leftChildTimeBounds.startTimeRange.second =
            leftChildTimeBounds.endTimeRange.second -
            leftChildTimeBounds.duration;
        leftChild->setTimeBounds(leftChildTimeBounds);
        TETRISCHED_DEBUG("[" << name << "] Left child " << leftChild->getId()
                             << " (" << leftChild->getName()
                             << ") has updated time bounds: "
                             << leftChildTimeBounds.toString())
      }
    }
    // Assign the bounds of the Expression.
    expressionTimeBoundMap[currentExpression->getId()] =
        currentExpression->getTimeBounds();
    TETRISCHED_DEBUG(
        "[" << name << "] The time bound for " << currentExpression->getId()
            << " (" << currentExpression->getName() << ") is "
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
          "[" + name + "] Expression " + currentExpression->getId() + "(" +
          currentExpression->getName() + ") does not have time bounds.");
    }

    auto expressionBounds = expressionTimeBoundMap[currentExpression->getId()];
    TETRISCHED_DEBUG("[" << name << "] Pushing down bounds for "
                         << currentExpression->getId() << " ("
                         << currentExpression->getName()
                         << "): " << expressionBounds.toString())

    if (currentExpression->getType() == ExpressionType::EXPR_LESSTHAN) {
      auto leftChild = currentExpression->getChildren()[0];
      if (expressionTimeBoundMap.find(leftChild->getId()) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "[" + name + "] Left child " + leftChild->getId() + " (" +
            leftChild->getName() + ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto& leftChildTimeBounds = expressionTimeBoundMap[leftChild->getId()];
      TETRISCHED_DEBUG("[" << name << "] Left child " << leftChild->getId()
                           << " (" << leftChild->getName()
                           << ") has time bounds: "
                           << leftChildTimeBounds.toString())

      auto rightChild = currentExpression->getChildren()[1];
      if (expressionTimeBoundMap.find(rightChild->getId()) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "[" + name + "] Right child " + rightChild->getId() + "(" +
            rightChild->getName() + ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto& rightChildTimeBounds = expressionTimeBoundMap[rightChild->getId()];
      TETRISCHED_DEBUG("[" << name << "] Right child " << rightChild->getId()
                           << "(" << rightChild->getName()
                           << ") has time bounds: "
                           << rightChildTimeBounds.toString())

      // Update the time bounds.
      bool leftChildUpdated = false;
      bool rightChildUpdated = false;
      if (leftChildTimeBounds.startTimeRange.first <
          expressionBounds.startTimeRange.first) {
        leftChildTimeBounds.startTimeRange.first =
            expressionBounds.startTimeRange.first;
        leftChildTimeBounds.endTimeRange.first =
            leftChildTimeBounds.startTimeRange.first +
            leftChildTimeBounds.duration;
        leftChildUpdated = true;
      }
      if (leftChildTimeBounds.startTimeRange.second >
          expressionBounds.startTimeRange.second) {
        leftChildTimeBounds.startTimeRange.second =
            expressionBounds.startTimeRange.second;
        leftChildTimeBounds.endTimeRange.second =
            leftChildTimeBounds.startTimeRange.second +
            leftChildTimeBounds.duration;
        leftChildUpdated = true;
      }
      if (rightChildTimeBounds.endTimeRange.first <
          expressionBounds.endTimeRange.first) {
        rightChildTimeBounds.endTimeRange.first =
            expressionBounds.endTimeRange.first;
        rightChildTimeBounds.startTimeRange.first =
            rightChildTimeBounds.endTimeRange.first -
            rightChildTimeBounds.duration;
        rightChildUpdated = true;
      }
      if (rightChildTimeBounds.endTimeRange.second >
          expressionBounds.endTimeRange.second) {
        rightChildTimeBounds.endTimeRange.second =
            expressionBounds.endTimeRange.second;
        rightChildTimeBounds.startTimeRange.second =
            rightChildTimeBounds.endTimeRange.second -
            rightChildTimeBounds.duration;
        rightChildUpdated = true;
      }

      if (leftChildUpdated) {
        TETRISCHED_DEBUG("[" << name << "] Left child " << leftChild->getId()
                             << " (" << leftChild->getName()
                             << ") has updated time bounds: "
                             << leftChildTimeBounds.toString())
      }

      if (rightChildUpdated) {
        TETRISCHED_DEBUG("[" << name << "] Right child " << rightChild->getId()
                             << " (" << rightChild->getName()
                             << ") has updated time bounds: "
                             << rightChildTimeBounds.toString())
      }
    } else {
      // Iterate through the children and tighten the bounds, if possible.
      for (auto& child : currentExpression->getChildren()) {
        if (expressionTimeBoundMap.find(child->getId()) ==
            expressionTimeBoundMap.end()) {
          throw exceptions::RuntimeException(
              "[" + name + "] Child " + child->getId() + "(" +
              child->getName() + ") of " + currentExpression->getId() + "(" +
              currentExpression->getName() + ") does not have time bounds.");
        }
        auto& childTimeBounds = expressionTimeBoundMap[child->getId()];
        TETRISCHED_DEBUG("[" << name << "] Child " << child->getId() << " ("
                             << child->getName() << ") has time bounds: "
                             << childTimeBounds.toString())

        bool childUpdated = false;
        if (childTimeBounds.startTimeRange.first <
            expressionBounds.startTimeRange.first) {
          childTimeBounds.startTimeRange.first =
              expressionBounds.startTimeRange.first;
          childTimeBounds.endTimeRange.first =
              childTimeBounds.startTimeRange.first + childTimeBounds.duration;
          childUpdated = true;
        }
        if (childTimeBounds.startTimeRange.second >
            expressionBounds.startTimeRange.second) {
          childTimeBounds.startTimeRange.second =
              expressionBounds.startTimeRange.second;
          childTimeBounds.endTimeRange.second =
              childTimeBounds.startTimeRange.second + childTimeBounds.duration;
          childUpdated = true;
        }
        if (childTimeBounds.endTimeRange.first <
            expressionBounds.endTimeRange.first) {
          childTimeBounds.endTimeRange.first =
              expressionBounds.endTimeRange.first;
          childTimeBounds.startTimeRange.first =
              childTimeBounds.endTimeRange.first - childTimeBounds.duration;
          childUpdated = true;
        }
        if (childTimeBounds.endTimeRange.second >
            expressionBounds.endTimeRange.second) {
          childTimeBounds.endTimeRange.second =
              expressionBounds.endTimeRange.second;
          childTimeBounds.startTimeRange.second =
              childTimeBounds.endTimeRange.second - childTimeBounds.duration;
          childUpdated = true;
        }

        if (childUpdated) {
          TETRISCHED_DEBUG("[" << name << "] Child " << child->getId() << " ("
                               << child->getName()
                               << ") has updated time bounds: "
                               << childTimeBounds.toString())
        }
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
    TETRISCHED_DEBUG("[" << name << "] Attending to nodes for "
                         << currentExpression->getId() << " ("
                         << currentExpression->getName() << ")")

    if (expressionTimeBoundMap.find(currentExpression->getId()) ==
        expressionTimeBoundMap.end()) {
      throw exceptions::RuntimeException(
          "[" + name + "] Expression " + currentExpression->getId() + "(" +
          currentExpression->getName() + ") does not have time bounds.");
    }

    if (currentExpression->getNumChildren() == 0) {
      // If this is a leaf node, we check if it can be purged.
      auto newTimeBounds = expressionTimeBoundMap[currentExpression->getId()];
      auto originalTimeBounds = currentExpression->getTimeBounds();
      TETRISCHED_DEBUG("[" << name << "] Original Time bounds for "
                           << currentExpression->getId() << "("
                           << currentExpression->getName()
                           << "): " << originalTimeBounds.toString())

      TETRISCHED_DEBUG("[" << name << "] New Time bounds for "
                           << currentExpression->getId() << "("
                           << currentExpression->getName()
                           << "): " << newTimeBounds.toString())

      if (currentExpression->getType() == ExpressionType::EXPR_CHOOSE) {
        // Choose expression leads to one choice and if that choice becomes
        // invalid, we should purge the Choose expression.
        if (originalTimeBounds.startTimeRange.first <
                newTimeBounds.startTimeRange.first ||
            originalTimeBounds.startTimeRange.second >
                newTimeBounds.startTimeRange.second ||
            originalTimeBounds.endTimeRange.first <
                newTimeBounds.endTimeRange.first ||
            originalTimeBounds.endTimeRange.second >
                newTimeBounds.endTimeRange.second) {
          purgedExpressions.insert(currentExpression->getId());
          TETRISCHED_DEBUG("[" << name << "] Purging node "
                               << currentExpression->getId() << " ("
                               << currentExpression->getName() << ")")
        }
      } else if (currentExpression->getType() ==
                     ExpressionType::EXPR_WINDOWED_CHOOSE ||
                 currentExpression->getType() ==
                     ExpressionType::EXPR_MALLEABLE_CHOOSE) {
        // Both WindowedChoose and MalleableChoose can generate various options.
        // We only purge them if all the options are invalid. Otherwise, we just
        // tighten the bounds.
        if (newTimeBounds.startTimeRange.first >
            newTimeBounds.endTimeRange.second) {
          // The expression is being asked to start after it can finish at the
          // earliest. This can definitely be purged.
          purgedExpressions.insert(currentExpression->getId());
          TETRISCHED_DEBUG("[" << name << "] Purging node "
                               << currentExpression->getId() << "("
                               << currentExpression->getName() << ")")
        } else {
          currentExpression->setTimeBounds(newTimeBounds);
        }
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
           currentExpression->getType() == ExpressionType::EXPR_LESSTHAN) ||
          (purgedChildrens.size() > 0 &&
           currentExpression->getType() == ExpressionType::EXPR_MIN)) {
        purgedExpressions.insert(currentExpression->getId());
        TETRISCHED_DEBUG("[" << name << "] Purging node "
                             << currentExpression->getId() << "("
                             << currentExpression->getName() << ")")
      }
    }
  }
}

void CriticalPathOptimizationPass::runPass(
    ExpressionPtr strlExpression, CapacityConstraintMap& capacityConstraints,
    std::optional<std::string> debugFile) {
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

void CriticalPathOptimizationPass::clean() { expressionTimeBoundMap.clear(); }

/* Methods for DiscretizationSelectorOptimizationPass */
DiscretizationSelectorOptimizationPass::DiscretizationSelectorOptimizationPass()
    : OptimizationPass("DiscretizationSelectorOptimizationPass",
                       OptimizationPassType::PRE_TRANSLATION_PASS) {}

void DiscretizationSelectorOptimizationPass::runPass(
    ExpressionPtr strlExpression, CapacityConstraintMap& capacityConstraints,
    std::optional<std::string> debugFile) {
  /* Do a Post-Order Traversal of the DAG. */
  std::stack<ExpressionPtr> firstStack;
  std::stack<ExpressionPtr> secondStack;
  firstStack.push(strlExpression);

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

  std::vector<std::pair<TimeRange, uint32_t>> occupancyRequestRanges;
  Time minOccupancyTime = std::numeric_limits<Time>::max();
  Time maxOccupancyTime = 0;

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
    TETRISCHED_DEBUG(
        "[DiscretizationSelectorOptimizationPass] Attending to nodes for "
        << currentExpression->getId() << "(" << currentExpression->getName()
        << ")")

    if (currentExpression->getNumChildren() != 0) {
      TETRISCHED_DEBUG("[DiscretizationSelectorOptimizationPass] Skipping "
                       << currentExpression->getId() << "("
                       << currentExpression->getName()
                       << ") since it is not a leaf node.")
      continue;
    }

    if (currentExpression->getType() == ExpressionType::EXPR_CHOOSE ||
        currentExpression->getType() == ExpressionType::EXPR_WINDOWED_CHOOSE ||
        currentExpression->getType() == ExpressionType::EXPR_MALLEABLE_CHOOSE) {
      // Get the time bounds of the choose expression.
      auto timeBounds = currentExpression->getTimeBounds();
      TimeRange occupancyRequestRange = std::make_pair(
          timeBounds.startTimeRange.first, timeBounds.endTimeRange.second);

      // Update min and max occupancy times, if necessary.
      if (occupancyRequestRange.first < minOccupancyTime) {
        minOccupancyTime = occupancyRequestRange.first;
      }
      if (occupancyRequestRange.second > maxOccupancyTime) {
        maxOccupancyTime = occupancyRequestRange.second;
      }

      // Push the occupancy request range into the vector.
      occupancyRequestRanges.push_back(
          {occupancyRequestRange, currentExpression->getResourceQuantity()});
      TETRISCHED_DEBUG(
          "[DiscretizationSelectorOptimizationPass] Adding "
          << currentExpression->getId() << "(" << currentExpression->getName()
          << ") to the occupancy request ranges with range: "
          << "[" << occupancyRequestRange.first << ", "
          << occupancyRequestRange.second << "]"
          << " and quantity: " << currentExpression->getResourceQuantity())
    }
  }

  // We now have the occupancy request ranges, we can now discretize them.
  std::vector<uint32_t> occupancyRequests(
      maxOccupancyTime - minOccupancyTime + 1, 0);
  for (auto& [occupancyRequestTimeRange, occupancyRequest] :
       occupancyRequestRanges) {
    // Add the occupancy request throughout this time range.
    for (Time time = occupancyRequestTimeRange.first;
         time <= occupancyRequestTimeRange.second; time++) {
      occupancyRequests[time - minOccupancyTime] += occupancyRequest;
    }
  }

  // Log the occupancy requests and times.
  TETRISCHED_DEBUG(
      "[DiscretizationSelectorOptimizationPass] Occupancy requests between "
      << minOccupancyTime << " and " << maxOccupancyTime << ": ");
  for (size_t i = 0; i < occupancyRequests.size(); i++) {
    TETRISCHED_DEBUG("[DiscretizationSelectorOptimizationPass] "
                     << i + minOccupancyTime << ": " << occupancyRequests[i]);
  }

  // finding the right discretization
  // TODO(Alind): all these values should be made input from python for this optimization pass
  uint32_t minDiscretization = 1;
  uint32_t maxDiscretization = 5;
  uint32_t maxOccupancyThreshold = 50;

  // decay function for finding the right discretization
  // TODO(Alind): Type of Decay fn should also be made input from python
  auto linearDecayFunction = [minDiscretization, maxDiscretization, maxOccupancyThreshold](double curentOccupancy){ 
      double fraction = curentOccupancy / maxOccupancyThreshold; 
      double value = (maxDiscretization - minDiscretization)*fraction;
      auto predictedDiscretization = std::max(maxDiscretization - value, (double) minDiscretization);
      return (uint32_t)std::round(predictedDiscretization);
  };
  // algorithm for deciding the discretization based on occupancy map
  std::vector<std::pair<TimeRange, Time>> timeRangeToGranularities;
  int planAheadIndex = -1;
  for (size_t i = 0; i < occupancyRequests.size(); i++) {
      // if discretization is decided for more plan ahead continue
      if (planAheadIndex > -1){
        Time endTime = timeRangeToGranularities[planAheadIndex].first.second;
        Time currentTime = minOccupancyTime + i;
        if (currentTime < endTime){
          continue;
        }
      }

      // find the right discretization for current occupancy
      uint32_t currentOccupancy = occupancyRequests[i];
      uint32_t predictedDiscretization = linearDecayFunction(currentOccupancy);
      auto nextPlanAhead = std::min(i + predictedDiscretization, occupancyRequests.size());

      // find the right discretization such that average occupancy for that period predicts same discretization
      while (nextPlanAhead >= (i+1)) {
        double averageOccupancy = 0;
        int count = 0;
        for(size_t j = i; j < nextPlanAhead; j++){
          averageOccupancy += occupancyRequests[j];
          count++;
        }
        averageOccupancy /= count;
        uint32_t avgPredictedDiscretization = linearDecayFunction(averageOccupancy);
        if (avgPredictedDiscretization == predictedDiscretization){
          TimeRange discretizedtimeRange = std::make_pair(minOccupancyTime + i, minOccupancyTime + nextPlanAhead);
          Time granularity = (nextPlanAhead - i);
          auto value = std::make_pair(discretizedtimeRange, granularity);
          timeRangeToGranularities.push_back(value);
          planAheadIndex++; 
          break;
        }
        nextPlanAhead--;
      }
  }

  // set capacity constraint map to dynamic and update it
  // TODO(Alind): check if the usage was registered for different dynamic discretization?
  // capacityConstraints.setDynamicDiscretization(timeRangeToGranularities);

  // Log the found dynamic discretizations and times.
  TETRISCHED_DEBUG(
      "[DiscretizationSelectorOptimizationPass] Dynamic Discretization between "
      << minOccupancyTime << " and " << maxOccupancyTime << ": ");

  for (auto &[discretizationTimeRange, granularity] :
       timeRangeToGranularities){
    TETRISCHED_DEBUG("[DiscretizationSelectorOptimizationPassDiscreteTime] "
                     << discretizationTimeRange.first << " - " << discretizationTimeRange.second << " : " << granularity);
  }
}

void DiscretizationSelectorOptimizationPass::clean() {}

/* Methods for CapacityConstraintMapPurgingOptimizationPass */
CapacityConstraintMapPurgingOptimizationPass::
    CapacityConstraintMapPurgingOptimizationPass()
    : OptimizationPass("CapacityConstraintMapPurgingOptimizationPass",
                       OptimizationPassType::POST_TRANSLATION_PASS) {}

void CapacityConstraintMapPurgingOptimizationPass::computeCliques(
    ExpressionPtr expression) {
  /* Do a Depth-First Search of the DAG and match Max -> {Leaves} */
  std::stack<ExpressionPtr> expressionStack;
  std::set<std::string> visitedExpressions;
  expressionStack.push(expression);

  while (!expressionStack.empty()) {
    // Pop the expression from the stack, and check if it needs to be visited.
    auto currentExpression = expressionStack.top();
    expressionStack.pop();
    if (visitedExpressions.find(currentExpression->getId()) !=
        visitedExpressions.end()) {
      continue;
    }
    visitedExpressions.insert(currentExpression->getId());

    TETRISCHED_DEBUG("Visiting " << currentExpression->getId() << " ("
                                 << currentExpression->getName() << ") of type "
                                 << currentExpression->getTypeString())

    // If the Expression is a MaxExpression, and all of its children are
    // ChooseExpressions, then we use this clique.
    if (currentExpression->getType() == ExpressionType::EXPR_MAX) {
      bool allChildrenAreChooseExpressions = true;
      for (auto& child : currentExpression->getChildren()) {
        if (child->getType() != ExpressionType::EXPR_CHOOSE &&
            child->getType() != ExpressionType::EXPR_MALLEABLE_CHOOSE) {
          allChildrenAreChooseExpressions = false;
          TETRISCHED_DEBUG("Child "
                           << child->getId() << " (" << child->getName()
                           << ") is not a ChooseExpression. It is of type "
                           << child->getTypeString())
          break;
        }
      }
      if (allChildrenAreChooseExpressions) {
        TETRISCHED_DEBUG("Creating a clique for Expression "
                         << currentExpression->getId() << " ("
                         << currentExpression->getName() << ")")
        std::unordered_set<ExpressionPtr> clique;
        for (auto& child : currentExpression->getChildren()) {
          clique.insert(child);
          TETRISCHED_DEBUG("Inserting " << child->getId() << " ("
                                        << child->getName()
                                        << ") to the clique for "
                                        << currentExpression->getId())
        }
        cliques[currentExpression] = clique;
      }
    }

    // Visit all the children.
    for (auto& child : currentExpression->getChildren()) {
      if (child->getNumChildren() > 0) {
        expressionStack.push(child);
      }
    }
  }
  TETRISCHED_DEBUG("Computed " << cliques.size() << " cliques.")
}

void CapacityConstraintMapPurgingOptimizationPass::
    deactivateCapacityConstraints(CapacityConstraintMap& capacityConstraints,
                                  std::optional<std::string> debugFile) {
  std::ofstream debugFileStream;
  if (debugFile.has_value()) {
    debugFileStream.open(debugFile.value());
  }
  // Construct a vector of the size of the number of cliques.
  // This vector will keep track of if the clique was used in a constraint,
  // and if so, its maximum usage.
  TETRISCHED_DEBUG("Running deactivation of constraints from a map of size "
                   << capacityConstraints.size())
  size_t deactivatedConstraints = 0;

  std::unordered_map<ExpressionPtr, uint32_t> expressionUsageMap;

  // Iterate over each of the individual CapacityConstraints in the map.
  for (auto& [key, capacityConstraint] :
       capacityConstraints.capacityConstraints) {
    // If the capacity check is trivially satisfiable, don't even bother
    // checking the cliques.
    if (capacityConstraint->capacityConstraint->isTriviallySatisfiable()) {
      TETRISCHED_DEBUG("Deactivating " << capacityConstraint->getName()
                                       << " since it is trivially satisfied.")
      deactivatedConstraints++;
      capacityConstraint->deactivate();
      continue;
    }

    auto maxCliqueStartTime = std::chrono::high_resolution_clock::now();
    // Reset the clique usage map.
    expressionUsageMap.clear();

    // Iterate over all the Expressions that contribute to a usage in
    // this CapacityConstraint, and turn on their clique usage.
    for (auto& [expression, usage] : capacityConstraint->usageVector) {
      if (expression->getNumParents() != 1) {
        throw tetrisched::exceptions::RuntimeException(
            "Expression " + expression->getId() + " (" + expression->getName() +
            ") of type " + expression->getTypeString() +
            " has more than one parent. This is not supported.");
      }
      auto expressionKey = expression->getParents()[0];
      if (expressionKey->getType() != ExpressionType::EXPR_MAX) {
        expressionKey = expression;
      }
      // if (cliques.find(expressionKey) == cliques.end()) {
      //   expressionKey = expression;
      // }

      // We make note of the maximum usage that this clique can
      // contribute to the CapacityConstraint.
      uint32_t constraintUsage = std::numeric_limits<uint32_t>::max();
      if (usage.isVariable()) {
        auto usageUpperBound = usage.get<VariablePtr>()->getUpperBound();
        if (usageUpperBound.has_value()) {
          constraintUsage = static_cast<uint32_t>(usageUpperBound.value());
        }
      } else {
        constraintUsage = usage.get<uint32_t>();
      }

      // We now insert the usage of this expression into the map.
      if (expressionUsageMap.find(expressionKey) == expressionUsageMap.end()) {
        expressionUsageMap[expressionKey] = constraintUsage;
      } else {
        expressionUsageMap[expressionKey] =
            std::max(expressionUsageMap[expressionKey], constraintUsage);
      }
    }
    auto maxCliqueEndTime = std::chrono::high_resolution_clock::now();

    // All the MAX cliques have been identified, if they are immediately
    // ordered by a < expression, then we can keep bubbling up the checks
    // until they reach a min.
    std::unordered_set<ExpressionPtr> keysToDelete;
    do {
      // Clear up the keys to delete.
      keysToDelete.clear();

      // Bubble up cliques.
      for (auto& [clique, usage] : expressionUsageMap) {
        for (auto& parent : clique->getParents()) {
          if (parent->getType() == ExpressionType::EXPR_LESSTHAN) {
            keysToDelete.insert(clique);
            if (expressionUsageMap.find(parent) == expressionUsageMap.end()) {
              expressionUsageMap[parent] = usage;
            } else {
              expressionUsageMap[parent] =
                  std::max(expressionUsageMap[parent], usage);
            }
          }
        }
      }

      // Delete redundant keys.
      for (auto& key : keysToDelete) {
        expressionUsageMap.erase(key);
      }
    } while (keysToDelete.size() > 0);
    auto lessThanCliqueEndTime = std::chrono::high_resolution_clock::now();
    auto maxCliqueDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            maxCliqueEndTime - maxCliqueStartTime)
            .count();
    auto lessThanCliqueDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            lessThanCliqueEndTime - maxCliqueEndTime)
            .count();

    // If the clique usage is <= RHS, then we can deactivate this constraint.
    uint32_t totalUsage = 0;
    for (auto& [clique, usage] : expressionUsageMap) {
      totalUsage += usage;
    }
    if (totalUsage <= capacityConstraint->getQuantity()) {
      deactivatedConstraints++;
      capacityConstraint->deactivate();
      TETRISCHED_DEBUG("Deactivating " << capacityConstraint->getName()
                                       << " since its total quantity is "
                                       << capacityConstraint->getQuantity()
                                       << " and its maximum resource usage is "
                                       << totalUsage)
    } else {
      TETRISCHED_DEBUG("Cannot deactivate "
                       << capacityConstraint->getName()
                       << " since its total quantity is "
                       << capacityConstraint->getQuantity()
                       << " and its maximum resource usage is " << totalUsage)
    }

    if (debugFile.has_value()) {
      debugFileStream << capacityConstraint->getName() << ": " << totalUsage
                      << ", " << capacityConstraint->getQuantity() << std::endl;
    }
  }
  TETRISCHED_DEBUG("Deactivated " << deactivatedConstraints << " out of "
                                  << capacityConstraints.size()
                                  << " constraints.")
}

void CapacityConstraintMapPurgingOptimizationPass::runPass(
    ExpressionPtr strlExpression, CapacityConstraintMap& capacityConstraints,
    std::optional<std::string> debugFile) {
  /* Phase 1: We compute the cliques from  the Expressions in the DAG. */
  // auto cliqueStartTime = std::chrono::high_resolution_clock::now();
  // computeCliques(strlExpression);
  // auto cliqueEndTime = std::chrono::high_resolution_clock::now();
  // auto cliqueDuration =
  // std::chrono::duration_cast<std::chrono::microseconds>(
  //                           cliqueEndTime - cliqueStartTime)
  //                           .count();
  // TETRISCHED_DEBUG("Computing cliques took: " << cliqueDuration
  //                                             << " microseconds.")
  // std::cout << "Computing cliques took: " << cliqueDuration << "
  // microseconds."
  //           << std::endl;

  /* Phase 2: We go over each of the CapacityConstraint in the map, and
  deactivate the constraint that is trivially satisfied. */
  auto deactivationStartTime = std::chrono::high_resolution_clock::now();
  deactivateCapacityConstraints(capacityConstraints, debugFile);
  auto deactivationEndTime = std::chrono::high_resolution_clock::now();
  auto deactivationDuration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          deactivationEndTime - deactivationStartTime)
          .count();
  TETRISCHED_DEBUG("Deactivating constraints took: " << deactivationDuration
                                                     << " microseconds.")
}

void CapacityConstraintMapPurgingOptimizationPass::clean() { cliques.clear(); }

/* Methods for OptimizationPassRunner */
OptimizationPassRunner::OptimizationPassRunner(bool debug) : debug(debug) {
  // Register the Critical Path optimization pass.
  registeredPasses.push_back(std::make_shared<CriticalPathOptimizationPass>());
  // Register the DiscretizationGenerator pass.
  registeredPasses.push_back(
      std::make_shared<DiscretizationSelectorOptimizationPass>());
  // Register the CapacityConstraintMapPurging optimization pass.
  // registeredPasses.push_back(
  //     std::make_shared<CapacityConstraintMapPurgingOptimizationPass>());
}

void OptimizationPassRunner::runPreTranslationPasses(
    Time currentTime, ExpressionPtr strlExpression,
    CapacityConstraintMap& capacityConstraints) {
  // Run the registered optimization passes on the given STRL expression.
  for (auto& pass : registeredPasses) {
    if (pass->getType() == OptimizationPassType::PRE_TRANSLATION_PASS) {
      auto debugFile =
          debug
              ? std::optional<std::string>(pass->getName() + "_" +
                                           std::to_string(currentTime) + ".log")
              : std::nullopt;
      pass->runPass(strlExpression, capacityConstraints, debugFile);
      pass->clean();
    }
  }
}

void OptimizationPassRunner::runPostTranslationPasses(
    Time currentTime, ExpressionPtr strlExpression,
    CapacityConstraintMap& capacityConstraints) {
  // Run the registered optimization passes on the given STRL expression.
  for (auto& pass : registeredPasses) {
    if (pass->getType() == OptimizationPassType::POST_TRANSLATION_PASS) {
      auto debugFile =
          debug
              ? std::optional<std::string>(pass->getName() + "_" +
                                           std::to_string(currentTime) + ".log")
              : std::nullopt;
      pass->runPass(strlExpression, capacityConstraints, debugFile);
      pass->clean();
    }
  }
}
}  // namespace tetrisched
