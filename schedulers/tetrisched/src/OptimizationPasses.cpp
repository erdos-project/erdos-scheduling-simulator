#include "tetrisched/OptimizationPasses.hpp"

#include <chrono>
#include <stack>

namespace tetrisched {

/* Methods for OptimizationPass */
OptimizationPass::OptimizationPass(std::string name, OptimizationPassType type)
    : name(name), type(type) {}

OptimizationPass::ExpressionPostOrderTraversal
OptimizationPass::computePostOrderTraversal(ExpressionPtr expression) {
  TETRISCHED_SCOPE_TIMER(
      "CriticalPathOptimizationPass::computePostOrderTraversal");
  /* Do a Post-Order Traversal of the DAG. */
  std::stack<ExpressionPtr> firstStack;
  std::deque<ExpressionPtr> postOrderTraversal;
  firstStack.push(expression);

  while (!firstStack.empty()) {
    // Move the expression to the second stack.
    auto currentExpression = firstStack.top();
    firstStack.pop();
    postOrderTraversal.push_back(currentExpression);

    // Add the children of the expression to the first stack.
    auto expressionChildren = currentExpression->getChildren();
    for (auto child = expressionChildren.rbegin();
         child != expressionChildren.rend(); ++child) {
      firstStack.push(*child);
    }
  }

  return postOrderTraversal;
}

OptimizationPassType OptimizationPass::getType() const { return type; }

std::string OptimizationPass::getName() const { return name; }

/* Methods for CriticalPathOptimizationPass */
CriticalPathOptimizationPass::CriticalPathOptimizationPass()
    : OptimizationPass("CriticalPathOptimizationPass",
                       OptimizationPassType::PRE_TRANSLATION_PASS) {}

void CriticalPathOptimizationPass::computeTimeBounds(
    const ExpressionPostOrderTraversal &postOrderTraversal) {
  // Iterate over the post-order traversal and compute the time bounds.
  std::unordered_set<std::string> visitedExpressions;
  visitedExpressions.reserve(postOrderTraversal.size());
  for (auto currentExpressionIt = postOrderTraversal.rbegin();
       currentExpressionIt != postOrderTraversal.rend();
       ++currentExpressionIt) {
    auto currentExpression = *currentExpressionIt;
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
      if (expressionTimeBoundMap.find(leftChild) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "[" + name + "] Left child " + leftChild->getId() + "(" +
            leftChild->getName() + ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto &leftChildTimeBounds = expressionTimeBoundMap[leftChild];
      TETRISCHED_DEBUG("[" << name << "] Left child " << leftChild->getId()
                           << " (" << leftChild->getName()
                           << ") has time bounds: "
                           << leftChildTimeBounds.toString())

      auto rightChild = currentExpression->getChildren()[1];
      if (expressionTimeBoundMap.find(rightChild) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "[" + name + "] Right child " + rightChild->getId() + " (" +
            rightChild->getName() + ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto &rightChildTimeBounds = expressionTimeBoundMap[rightChild];
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
    expressionTimeBoundMap[currentExpression] =
        currentExpression->getTimeBounds();
    TETRISCHED_DEBUG(
        "[" << name << "] The time bound for " << currentExpression->getId()
            << " (" << currentExpression->getName() << ") is "
            << expressionTimeBoundMap[currentExpression].toString())
  }
}

void CriticalPathOptimizationPass::pushDownTimeBounds(
    const ExpressionPostOrderTraversal &postOrderTraversal) {
  // Iterate over the post-order traversal in the reverse order and push down
  // the time bounds.
  for (auto currentExpressionIt = postOrderTraversal.begin();
       currentExpressionIt != postOrderTraversal.end(); ++currentExpressionIt) {
    auto currentExpression = *currentExpressionIt;
    if (expressionTimeBoundMap.find(currentExpression) ==
        expressionTimeBoundMap.end()) {
      throw exceptions::RuntimeException(
          "[" + name + "] Expression " + currentExpression->getId() + "(" +
          currentExpression->getName() + ") does not have time bounds.");
    }

    auto expressionBounds = expressionTimeBoundMap[currentExpression];
    TETRISCHED_DEBUG("[" << name << "] Pushing down bounds for "
                         << currentExpression->getId() << " ("
                         << currentExpression->getName()
                         << "): " << expressionBounds.toString())

    if (currentExpression->getType() == ExpressionType::EXPR_LESSTHAN) {
      auto leftChild = currentExpression->getChildren()[0];
      if (expressionTimeBoundMap.find(leftChild) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "[" + name + "] Left child " + leftChild->getId() + " (" +
            leftChild->getName() + ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto &leftChildTimeBounds = expressionTimeBoundMap[leftChild];
      TETRISCHED_DEBUG("[" << name << "] Left child " << leftChild->getId()
                           << " (" << leftChild->getName()
                           << ") has time bounds: "
                           << leftChildTimeBounds.toString())

      auto rightChild = currentExpression->getChildren()[1];
      if (expressionTimeBoundMap.find(rightChild) ==
          expressionTimeBoundMap.end()) {
        throw exceptions::RuntimeException(
            "[" + name + "] Right child " + rightChild->getId() + "(" +
            rightChild->getName() + ") of " + currentExpression->getId() + "(" +
            currentExpression->getName() + ") does not have time bounds.");
      }
      auto &rightChildTimeBounds = expressionTimeBoundMap[rightChild];
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
      for (auto &child : currentExpression->getChildren()) {
        if (expressionTimeBoundMap.find(child) ==
            expressionTimeBoundMap.end()) {
          throw exceptions::RuntimeException(
              "[" + name + "] Child " + child->getId() + "(" +
              child->getName() + ") of " + currentExpression->getId() + "(" +
              currentExpression->getName() + ") does not have time bounds.");
        }
        auto &childTimeBounds = expressionTimeBoundMap[child];
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

void CriticalPathOptimizationPass::purgeNodes(
    const ExpressionPostOrderTraversal &postOrderTraversal) {
  // Iterate over the post-order traversal and purge the nodes that cannot be
  // satisfied.
  std::unordered_set<ExpressionPtr> visitedExpressions;
  visitedExpressions.reserve(postOrderTraversal.size());
  std::unordered_set<ExpressionPtr> purgedExpressions;
  purgedExpressions.reserve(postOrderTraversal.size());
  for (auto currentExpressionIt = postOrderTraversal.rbegin();
       currentExpressionIt != postOrderTraversal.rend();
       ++currentExpressionIt) {
    auto currentExpression = *currentExpressionIt;
    if (visitedExpressions.find(currentExpression) ==
        visitedExpressions.end()) {
      visitedExpressions.insert(currentExpression);
    } else {
      continue;
    }
    TETRISCHED_DEBUG("[" << name << "] Attending to nodes for "
                         << currentExpression->getId() << " ("
                         << currentExpression->getName() << ")")

    auto newTimeBoundsLocation = expressionTimeBoundMap.find(currentExpression);
    if (newTimeBoundsLocation == expressionTimeBoundMap.end()) {
      throw exceptions::RuntimeException(
          "[" + name + "] Expression " + currentExpression->getId() + "(" +
          currentExpression->getName() + ") does not have time bounds.");
    }

    if (currentExpression->getNumChildren() == 0) {
      // If this is a leaf node, we check if it can be purged.
      auto newTimeBounds = newTimeBoundsLocation->second;
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
          purgedExpressions.insert(currentExpression);
          TETRISCHED_DEBUG("[" << name << "] Purging node "
                               << currentExpression->getId() << " ("
                               << currentExpression->getName() << ")")
        }
      } else if (currentExpression->getType() ==
                     ExpressionType::EXPR_WINDOWED_CHOOSE ||
                 currentExpression->getType() ==
                     ExpressionType::EXPR_MALLEABLE_CHOOSE) {
        // Both WindowedChoose and MalleableChoose can generate various
        // options. We only purge them if all the options are invalid.
        // Otherwise, we just tighten the bounds.
        if (newTimeBounds.startTimeRange.first >
            newTimeBounds.endTimeRange.second) {
          // The expression is being asked to start after it can finish at the
          // earliest. This can definitely be purged.
          purgedExpressions.insert(currentExpression);
          TETRISCHED_DEBUG("[" << name << "] Purging node "
                               << currentExpression->getId() << "("
                               << currentExpression->getName() << ")")
        } else {
          currentExpression->setTimeBounds(newTimeBounds);
        }
      }
    } else {
      // This is not a leaf node, we purge it if its children have been
      // affected.
      std::vector<ExpressionPtr> savedChildren;
      savedChildren.reserve(currentExpression->getNumChildren());
      for (auto &child : currentExpression->getChildren()) {
        if (purgedExpressions.find(child) == purgedExpressions.end()) {
          savedChildren.push_back(child);
        }
      }

      if (currentExpression->getType() == ExpressionType::EXPR_MAX) {
        if (savedChildren.size() == 0) {
          purgedExpressions.insert(currentExpression);
          TETRISCHED_DEBUG("[" << name << "] Purging node "
                               << currentExpression->getId() << "("
                               << currentExpression->getName() << ")")
        } else {
          currentExpression->replaceChildren(std::move(savedChildren));
        }
      } else if (currentExpression->getType() ==
                     ExpressionType::EXPR_LESSTHAN ||
                 currentExpression->getType() == ExpressionType::EXPR_MIN) {
        if (savedChildren.size() != currentExpression->getNumChildren()) {
          // Some children have been purged, we purge this node as well.
          purgedExpressions.insert(currentExpression);
          TETRISCHED_DEBUG("[" << name << "] Purging node "
                               << currentExpression->getId() << "("
                               << currentExpression->getName() << ")")
        }
      }
    }
  }
}

void CriticalPathOptimizationPass::runPass(
    ExpressionPtr strlExpression,
    CapacityConstraintMapPtr /* capacityConstraints */,
    std::optional<std::string> /* debugFile */) {
  /* Preprocessing: We first compute the post-order traversal of the
  Expression graph since all subsequent steps use it. */
  auto postOrderTraversal = computePostOrderTraversal(strlExpression);

  // We reserve enough space in the map to avoid rehashing.
  expressionTimeBoundMap.reserve(postOrderTraversal.size());

  /* Phase 1: We first do a bottom-up traversal of the tree to compute
  a tight bound for each node in the STRL tree. */
  {
    TETRISCHED_SCOPE_TIMER(
        "CriticalPathOptimizationPass::runPass::computeTimeBounds");
    computeTimeBounds(postOrderTraversal);
  }

  /* Phase 2: The previous phase computes the tight bounds but does not
  push them down necessarily. In this phase, we push the bounds down. */
  {
    TETRISCHED_SCOPE_TIMER(
        "CriticalPathOptimizationPass::runPass::pushDownTimeBounds");
    pushDownTimeBounds(postOrderTraversal);
  }

  /* Phase 3: The bounds have been pushed down now, we can do a bottom-up
  traversal and start purging nodes that cannot be satisfied. */
  {
    TETRISCHED_SCOPE_TIMER("CriticalPathOptimizationPass::runPass::purgeNodes");
    purgeNodes(postOrderTraversal);
  }
}

void CriticalPathOptimizationPass::clean() {
  TETRISCHED_SCOPE_TIMER("CriticalPathOptimizationPass::clean");
  expressionTimeBoundMap.clear();
}

/* Methods for DiscretizationSelectorOptimizationPass */
DiscretizationSelectorOptimizationPass::DiscretizationSelectorOptimizationPass()
    : OptimizationPass("DiscretizationSelectorOptimizationPass",
                       OptimizationPassType::PRE_TRANSLATION_PASS),
      minDiscretization(1),
      maxDiscretization(5),
      maxOccupancyThreshold(0.8) {}

DiscretizationSelectorOptimizationPass::DiscretizationSelectorOptimizationPass(
    Time minDiscretization, Time maxDiscretization, float maxOccupancyThreshold)
    : OptimizationPass("DiscretizationSelectorOptimizationPass",
                       OptimizationPassType::PRE_TRANSLATION_PASS),
      minDiscretization(minDiscretization),
      maxDiscretization(maxDiscretization),
      maxOccupancyThreshold(maxOccupancyThreshold) {}

void DiscretizationSelectorOptimizationPass::runPass(
    ExpressionPtr strlExpression, CapacityConstraintMapPtr capacityConstraints,
    std::optional<std::string> /* debugFile */) {
  /* Do a Post-Order Traversal of the DAG. */
  std::stack<ExpressionPtr> firstStack;
  firstStack.push(strlExpression);

  std::vector<ExpressionPtr> maxOverNckExprs;
  std::vector<ExpressionPtr> independentNcks;

  while (!firstStack.empty()) {
    // Move the expression to the second stack.
    auto currentExpression = firstStack.top();
    firstStack.pop();
    bool isMaxExpr = false;
    if (currentExpression->getType() == ExpressionType::EXPR_MAX) {
      isMaxExpr = true;
    } else if ((currentExpression->getType() == ExpressionType::EXPR_CHOOSE ||
                currentExpression->getType() ==
                    ExpressionType::EXPR_WINDOWED_CHOOSE) &&
               currentExpression->getParents()[0]->getType() !=
                   ExpressionType::EXPR_MAX) {
      // TODO(Alind): Actual check should be that MAX has all children NCK for
      // EXPR_CHOOSE
      independentNcks.push_back(currentExpression);
    }
    bool allChildrenNck = false;
    size_t numChildNcks = 0;

    // Add the children to the first stack.
    auto expressionChildren = currentExpression->getChildren();
    for (auto child = expressionChildren.rbegin();
         child != expressionChildren.rend(); ++child) {
      if ((*child)->getType() == ExpressionType::EXPR_CHOOSE) {
        numChildNcks++;
      }
      firstStack.push(*child);
    }
    if (numChildNcks == expressionChildren.size()) {
      allChildrenNck = true;
    }
    if (isMaxExpr && allChildrenNck) {
      maxOverNckExprs.push_back(currentExpression);
    }
  }
  if (maxOverNckExprs.size() == 0 && independentNcks.size() == 0) {
    throw tetrisched::exceptions::RuntimeException(
        "Inside Discretization Optimization pass: but no max over nck or "
        "independent ncks found!");
  }

  std::vector<std::pair<TimeRange, uint32_t>> occupancyRequestRanges;
  Time minOccupancyTime = std::numeric_limits<Time>::max();
  Time maxOccupancyTime = 0;
  for (auto maxNckExpr : maxOverNckExprs) {
    auto timeBounds = maxNckExpr->getTimeBounds();
    TimeRange occupancyRequestRange = std::make_pair(
        timeBounds.startTimeRange.first, timeBounds.endTimeRange.second);

    // Update min and max occupancy times, if necessary.
    if (occupancyRequestRange.first < minOccupancyTime) {
      minOccupancyTime = occupancyRequestRange.first;
    }
    if (occupancyRequestRange.second > maxOccupancyTime) {
      maxOccupancyTime = occupancyRequestRange.second;
    }
    auto numResources = maxNckExpr->getChildren()[0]->getResourceQuantity();
    // Push the occupancy request range into the vector.
    occupancyRequestRanges.push_back({occupancyRequestRange, numResources});
  }
  for (auto iNck : independentNcks) {
    auto timeBounds = iNck->getTimeBounds();
    TimeRange occupancyRequestRange = std::make_pair(
        timeBounds.startTimeRange.first, timeBounds.endTimeRange.second);

    // Update min and max occupancy times, if necessary.
    if (occupancyRequestRange.first < minOccupancyTime) {
      minOccupancyTime = occupancyRequestRange.first;
    }
    if (occupancyRequestRange.second > maxOccupancyTime) {
      maxOccupancyTime = occupancyRequestRange.second;
    }
    auto numResources = iNck->getResourceQuantity();
    // Push the occupancy request range into the vector.
    occupancyRequestRanges.push_back({occupancyRequestRange, numResources});
  }

  // We now have the occupancy request ranges, we can now discretize them.
  std::vector<uint32_t> occupancyRequests(
      maxOccupancyTime - minOccupancyTime + 1, 0);
  for (auto &[occupancyRequestTimeRange, occupancyRequest] :
       occupancyRequestRanges) {
    // Add the occupancy request throughout this time range.
    for (Time time = occupancyRequestTimeRange.first;
         time <= occupancyRequestTimeRange.second; time++) {
      occupancyRequests[time - minOccupancyTime] += occupancyRequest;
    }
  }

  // Log the occupancy requests and times.
  // std::cout << "[" << minOccupancyTime << "]"
  //           << "[DiscretizationSelectorOptimizationPass] Occupancy requests
  //           between "
  //           << minOccupancyTime << " and " << maxOccupancyTime << ": " <<
  //           std::endl;
  uint32_t maxOccupancyVal = 0;
  for (size_t i = 0; i < occupancyRequests.size(); i++) {
    auto currentOccupancy = occupancyRequests[i];
    if (currentOccupancy > maxOccupancyVal) {
      maxOccupancyVal = currentOccupancy;
    }
    // std::cout << "\t" << "[" << minOccupancyTime << "]" <<
    // "[DiscretizationSelectorOptimizationPass] "
    //                  << i + minOccupancyTime << ": " <<
    //                  occupancyRequests[i]
    //                  << std::endl;
  }

  double autoMaxOccupancyThreshold = maxOccupancyThreshold * maxOccupancyVal;
  // std::cout << "** [DiscretizationSelectorOptimizationPass] Max
  // Discretization Value" << maxOccupancyVal << " Threshold Decided is: " <<
  // autoMaxOccupancyThreshold << " threshold val: " << maxOccupancyThreshold
  // << std::endl;

  // finding the right discretization

  // decay function for finding the right discretization
  // TODO(Alind): Type of Decay fn should also be made input from python
  auto linearDecayFunction =
      [this, autoMaxOccupancyThreshold](double curentOccupancy) {
        double fraction = curentOccupancy / autoMaxOccupancyThreshold;
        double value = (maxDiscretization - minDiscretization) * fraction;
        auto predictedDiscretization =
            std::max(maxDiscretization - value, (double)minDiscretization);
        return (uint32_t)std::round(predictedDiscretization);
      };
  // algorithm for deciding the discretization based on occupancy map
  std::vector<std::pair<TimeRange, Time>> timeRangeToGranularities;
  int planAheadIndex = -1;
  for (size_t i = 0; i < occupancyRequests.size(); i++) {
    // if discretization is decided for more plan ahead continue
    if (planAheadIndex > -1) {
      Time endTime = timeRangeToGranularities[planAheadIndex].first.second;
      Time currentTime = minOccupancyTime + i;
      if (currentTime < endTime) {
        continue;
      }
    }

    // find the right discretization for current occupancy
    uint32_t currentOccupancy = occupancyRequests[i];
    uint32_t predictedDiscretization = linearDecayFunction(currentOccupancy);
    auto nextPlanAhead =
        std::min(i + predictedDiscretization, occupancyRequests.size());

    // find the right discretization such that average occupancy for that
    // period predicts same discretization
    while (nextPlanAhead >= (i + 1)) {
      double averageOccupancy = 0;
      int count = 0;
      for (size_t j = i; j < nextPlanAhead; j++) {
        averageOccupancy += occupancyRequests[j];
        count++;
      }
      averageOccupancy /= count;
      uint32_t avgPredictedDiscretization =
          linearDecayFunction(averageOccupancy);
      if (avgPredictedDiscretization == predictedDiscretization) {
        TimeRange discretizedtimeRange = std::make_pair(
            minOccupancyTime + i, minOccupancyTime + nextPlanAhead);
        Time granularity = (nextPlanAhead - i);
        auto value = std::make_pair(discretizedtimeRange, granularity);
        timeRangeToGranularities.push_back(value);
        // newStartTimes.push_back(minOccupancyTime + i);
        planAheadIndex++;
        break;
      }
      nextPlanAhead--;
    }
  }

  // set capacity constraint map to dynamic and update it
  capacityConstraints->setDynamicDiscretization(timeRangeToGranularities);

  // // Log the found dynamic discretizations and times.
  // std::cout <<
  //     "[DiscretizationSelectorOptimizationPass] Dynamic Discretization
  //     between "
  //     << minOccupancyTime << " and " << maxOccupancyTime << ": "<<
  //     std::endl;

  // for (auto &[discretizationTimeRange, granularity] :
  //      timeRangeToGranularities) {
  //   std::cout << "\t"
  //             << "[" << minOccupancyTime << "]"
  //             << "[DiscretizationSelectorOptimizationPassDiscreteTime] "
  //             << discretizationTimeRange.first << " - " <<
  //             discretizationTimeRange.second << " : " << granularity << "
  //             Occuapncy: " <<
  //             occupancyRequests[discretizationTimeRange.first
  //             - minOccupancyTime] << std::endl;
  // }

  // changing the STRL expressions
  // Find Max expressions over NCK and remove NCK expressions with redundant
  // start times
  for (auto &[discreteTimeRange, discreteGranularity] :
       timeRangeToGranularities) {
    auto startTime = discreteTimeRange.first;
    auto endTime = discreteTimeRange.second;
    for (auto maxNckExpr : maxOverNckExprs) {
      // find ncks within startTime and endTime for this Max expr
      std::vector<ExpressionPtr> ncksWithinTimeRange;
      auto expressionChildren = maxNckExpr->getChildren();
      ExpressionPtr minStartTimeNckExpr = nullptr;
      for (auto child = expressionChildren.rbegin();
           child != expressionChildren.rend(); ++child) {
        auto startTimeNck = (*child)->getTimeBounds().startTimeRange.first;
        if (startTimeNck >= startTime && startTimeNck < endTime) {
          ncksWithinTimeRange.push_back(*child);
          if (minStartTimeNckExpr != nullptr) {
            if (minStartTimeNckExpr->getTimeBounds().startTimeRange.first >
                startTimeNck) {
              minStartTimeNckExpr = *child;
            }
          } else {
            minStartTimeNckExpr = *child;
          }
        }
      }
      if (ncksWithinTimeRange.size() > 1) {
        // if more than one nck found within the time range, remove it as only
        // one nck within granularity is sufficient. Nck with minimum start
        // time is kept within this time range
        for (auto redundantNckExpr : ncksWithinTimeRange) {
          if (redundantNckExpr->getId() != minStartTimeNckExpr->getId()) {
            maxNckExpr->removeChild(redundantNckExpr);
            // std::cout << "[DiscretizationSelectorOptimizationPassRemoveNck]
            // Removing NCK: " + redundantNckExpr->getName() + " From Max: " +
            // maxNckExpr->getName() << "Time Range: [" << startTime << ", "
            // << endTime << "]";
          }
        }
      }
    }
  }
}

void DiscretizationSelectorOptimizationPass::clean() {}

/* Methods for CapacityConstraintMapPurgingOptimizationPass */
CapacityConstraintMapPurgingOptimizationPass::
    CapacityConstraintMapPurgingOptimizationPass()
    : OptimizationPass("CapacityConstraintMapPurgingOptimizationPass",
                       OptimizationPassType::POST_TRANSLATION_PASS) {}

void CapacityConstraintMapPurgingOptimizationPass::computeCliques(
    const ExpressionPostOrderTraversal &postOrderTraversal) {
  // Iterate over the post-order traversal and compute the cliques.
  std::unordered_set<ExpressionPtr> visitedExpressions;
  visitedExpressions.reserve(postOrderTraversal.size());

  for (auto currentExpressionIt = postOrderTraversal.rbegin();
       currentExpressionIt != postOrderTraversal.rend();
       ++currentExpressionIt) {
    auto currentExpression = *currentExpressionIt;

    if (visitedExpressions.find(currentExpression) ==
        visitedExpressions.end()) {
      visitedExpressions.insert(currentExpression);
    } else {
      continue;
    }

    if (currentExpression->getNumChildren() == 0) {
      if (currentExpression->getType() == ExpressionType::EXPR_CHOOSE &&
          currentExpression->getNumParents() == 1 &&
          currentExpression->getParents()[0]->getType() ==
              ExpressionType::EXPR_MAX) {
        // PERF: We generate a lot of ChooseExpressions, which make this
        // pass extremely slow to run. When possible, we use the MaxExpression
        // to generate the clique.
        continue;
      } else {
        // This is a leaf node, we just add it to its own clique.
        cliques[currentExpression] =
            std::unordered_set<ExpressionPtr>({currentExpression});
        childLeafExpressions[currentExpression] =
            std::unordered_set<ExpressionPtr>({currentExpression});
        TETRISCHED_DEBUG("[" << name << "] " << currentExpression->getId()
                             << " (" << currentExpression->getName() << ") has "
                             << currentExpression->getNumChildren()
                             << " children. "
                             << "Constructed the clique with "
                             << currentExpression->getName());
      }
    } else if (currentExpression->getType() == ExpressionType::EXPR_MAX) {
      TETRISCHED_SCOPE_TIMER(
          "CriticalPathOptimizationPass::computeCliques::MaxExpression");
      // NOTE (Sukrit): We make some optimizations here since we know that
      // in our current STRL generation, a MaxExpression is never more than a
      // single level away from the leaves. We can both construct the cliques
      // and the child leaf expressions in one go.
      childLeafExpressions[currentExpression] =
          std::unordered_set<ExpressionPtr>({currentExpression});
      cliques[currentExpression] =
          std::unordered_set<ExpressionPtr>({currentExpression});
      // {
      //   TETRISCHED_SCOPE_TIMER(
      //       "CriticalPathOptimizationPass::computeCliques::"
      //       "MaxExpression::constructChildCliques");
      //   for (auto &child : maxChildren) {
      //     auto childClique = cliques.find(child);
      //     if (childClique == cliques.end()) {
      //       throw exceptions::RuntimeException(
      //           "[" + name + "] Child " + child->getId() + "(" +
      //           child->getName() + ") of " + currentExpression->getId() + "("
      //           + currentExpression->getName() + ") does not have a
      //           clique.");
      //     }
      //     childClique->second.insert(maxChildren.begin(), maxChildren.end());
      //   }
      // }
      TETRISCHED_DEBUG("[" << name << "] " << currentExpression->getId() << " ("
                           << currentExpression->getName() << ") has "
                           << currentExpression->getNumChildren()
                           << " children. "
                           << "Constructed the MaxExpression clique for "
                           << currentExpression->getName());
    } else {
      TETRISCHED_SCOPE_TIMER("CriticalPathOptimizationPass::computeCliques::" +
                             currentExpression->getTypeString());
      // Construct the leaf expressions under the current expression as
      // a merger of all the leaf expressions under its children.
      childLeafExpressions[currentExpression] =
          std::unordered_set<ExpressionPtr>();

      for (auto &child : currentExpression->getChildren()) {
        // Add the child expression set to the current expression.
        auto childLeafExpressionSet = childLeafExpressions.find(child);
        if (childLeafExpressionSet == childLeafExpressions.end()) {
          throw exceptions::RuntimeException(
              "[" + name + "] Child " + child->getId() + "(" +
              child->getName() + ") of " + currentExpression->getId() + "(" +
              currentExpression->getName() + ") does not have a clique.");
        }
        childLeafExpressions[currentExpression].insert(
            childLeafExpressionSet->second.begin(),
            childLeafExpressionSet->second.end());
        TETRISCHED_DEBUG("[" << name << "] Adding child expressions from "
                             << child->getName() << " to "
                             << currentExpression->getName());
      }

      TETRISCHED_DEBUG(
          "[" << name << "] " << currentExpression->getId() << " ("
              << currentExpression->getName() << ") has "
              << currentExpression->getNumChildren() << " children. "
              << "Constructed the clique with " << currentExpression->getName()
              << " for " << currentExpression->getTypeString());

      if (currentExpression->getType() == ExpressionType::EXPR_LESSTHAN) {
        // For a LESSTHAN expression, we go over each of the two child
        // expressions, and add the leave expressions from the other child to
        // its clique.
        auto currentExpressionChildren = currentExpression->getChildren();

        auto leftChild = currentExpressionChildren[0];
        auto leftChildLeafExpressions = childLeafExpressions.find(leftChild);
        if (leftChildLeafExpressions == childLeafExpressions.end()) {
          throw exceptions::RuntimeException(
              "[" + name + "] Left child " + leftChild->getId() + "(" +
              leftChild->getName() + ") of " + currentExpression->getId() +
              "(" + currentExpression->getName() + ") does not have a clique.");
        }

        auto rightChild = currentExpressionChildren[1];
        auto rightChildLeafExpressions = childLeafExpressions.find(rightChild);
        if (rightChildLeafExpressions == childLeafExpressions.end()) {
          throw exceptions::RuntimeException(
              "[" + name + "] Right child " + rightChild->getId() + "(" +
              rightChild->getName() + ") of " + currentExpression->getId() +
              "(" + currentExpression->getName() + ") does not have a clique.");
        }

        for (auto &leftChildLeafExpression : leftChildLeafExpressions->second) {
          auto leftChildClique = cliques.find(leftChildLeafExpression);
          if (leftChildClique == cliques.end()) {
            throw exceptions::RuntimeException(
                "[" + name + "] Left child leaf " +
                leftChildLeafExpression->getId() + "(" +
                leftChildLeafExpression->getName() + ") of " +
                currentExpression->getId() + "(" +
                currentExpression->getName() + ") does not have a clique.");
          }
          leftChildClique->second.insert(
              rightChildLeafExpressions->second.begin(),
              rightChildLeafExpressions->second.end());
        }

        for (auto &rightChildLeafExpression :
             rightChildLeafExpressions->second) {
          auto rightChildClique = cliques.find(rightChildLeafExpression);
          if (rightChildClique == cliques.end()) {
            throw exceptions::RuntimeException(
                "[" + name + "] Right child leaf " +
                rightChildLeafExpression->getId() + "(" +
                rightChildLeafExpression->getName() + ") of " +
                currentExpression->getId() + "(" +
                currentExpression->getName() + ") does not have a clique.");
          }
          rightChildClique->second.insert(
              leftChildLeafExpressions->second.begin(),
              leftChildLeafExpressions->second.end());
        }

        TETRISCHED_DEBUG("[" << name << "] Adding child expressions from "
                             << leftChild->getName() << " and "
                             << rightChild->getName()
                             << " to each other's clique as part of "
                             << currentExpression->getName() << " ("
                             << currentExpression->getTypeString() << ").");
      }
    }
  }
}

void CapacityConstraintMapPurgingOptimizationPass::
    deactivateCapacityConstraints(CapacityConstraintMapPtr capacityConstraints,
                                  std::optional<std::string> debugFile) {
  std::ofstream debugFileStream;
  if (debugFile.has_value()) {
    debugFileStream.open(debugFile.value());
  }
  // Construct a vector of the size of the number of cliques.
  // This vector will keep track of if the clique was used in a constraint,
  // and if so, its maximum usage.
  TETRISCHED_DEBUG("Running deactivation of constraints from a map of size "
                   << capacityConstraints->size())
  size_t deactivatedConstraints = 0;

  std::unordered_map<ExpressionPtr, uint32_t> expressionUsageMap;

  // Iterate over each of the individual CapacityConstraints in the map.
  for (auto &[key, capacityConstraint] :
       capacityConstraints->capacityConstraints) {
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
    for (auto &[expression, usage] : capacityConstraint->usageVector) {
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
      for (auto &[clique, usage] : expressionUsageMap) {
        for (auto &parent : clique->getParents()) {
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
      for (auto &key : keysToDelete) {
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
    for (auto &[clique, usage] : expressionUsageMap) {
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
                                  << capacityConstraints->size()
                                  << " constraints.")
}

void CapacityConstraintMapPurgingOptimizationPass::runPass(
    ExpressionPtr strlExpression, CapacityConstraintMapPtr capacityConstraints,
    std::optional<std::string> debugFile) {
  // Open the debug file if one was provided.
  std::ofstream debugFileStream;
  if (debugFile.has_value()) {
    debugFileStream.open(debugFile.value());
  }

  /* Preprocessing: Compute the post-order traversal to compute the cliques. */
  auto postOrderTraversal = computePostOrderTraversal(strlExpression);
  cliques.reserve(postOrderTraversal.size());
  childLeafExpressions.reserve(postOrderTraversal.size());

  /* Phase 1: Compute the cliques from the Expressions in the DAG. */
  {
    TETRISCHED_SCOPE_TIMER(
        "CapacityConstraintMapPurgingOptimizationPass::runPass::"
        "computeCliques");
    computeCliques(postOrderTraversal);
  }

  // Output the cliques, if a debug file was provided.
  if (debugFile.has_value()) {
    for (auto &[key, clique] : cliques) {
      debugFileStream << key->getName() << ": ";
      for (auto &expression : clique) {
        debugFileStream << expression->getName() << ", ";
      }
      debugFileStream << std::endl;
    }
  }

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
  // auto deactivationStartTime = std::chrono::high_resolution_clock::now();
  // deactivateCapacityConstraints(capacityConstraints, debugFile);
  // auto deactivationEndTime = std::chrono::high_resolution_clock::now();
  // auto deactivationDuration =
  //     std::chrono::duration_cast<std::chrono::microseconds>(
  //         deactivationEndTime - deactivationStartTime)
  //         .count();
  // TETRISCHED_DEBUG("Deactivating constraints took: " << deactivationDuration
  //                                                    << " microseconds.")
}

void CapacityConstraintMapPurgingOptimizationPass::clean() { cliques.clear(); }

/* Methods for OptimizationPassRunner */
OptimizationPassRunner::OptimizationPassRunner(bool debug,
                                               bool enableDynamicDiscretization,
                                               Time minDiscretization,
                                               Time maxDiscretization,
                                               float maxOccupancyThreshold)
    : debug(debug), enableDynamicDiscretization(enableDynamicDiscretization) {
  // Register the Critical Path optimization pass.
  registeredPasses.push_back(std::make_shared<CriticalPathOptimizationPass>());

  if (enableDynamicDiscretization) {
    // Register the DiscretizationGenerator pass.
    registeredPasses.push_back(
        std::make_shared<DiscretizationSelectorOptimizationPass>(
            minDiscretization, maxDiscretization, maxOccupancyThreshold));
  }

  // Register the CapacityConstraintMapPurging optimization pass.
  registeredPasses.push_back(
      std::make_shared<CapacityConstraintMapPurgingOptimizationPass>());
}

void OptimizationPassRunner::runPreTranslationPasses(
    Time currentTime, ExpressionPtr strlExpression,
    CapacityConstraintMapPtr capacityConstraints) {
  // Run the registered optimization passes on the given STRL expression.
  for (auto &pass : registeredPasses) {
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
    CapacityConstraintMapPtr capacityConstraints) {
  // Run the registered optimization passes on the given STRL expression.
  for (auto &pass : registeredPasses) {
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
