#include "tetrisched/Expression.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <stack>

#include "tbb/task_group.h"

namespace tetrisched {

// UUID Generation. Copied from
// https://stackoverflow.com/questions/24365331/how-can-i-generate-uuid-in-c-without-using-boost-library
namespace uuid {
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_int_distribution<> dis(0, 15);
static std::uniform_int_distribution<> dis2(8, 11);

std::string generate_uuid() {
  std::stringstream ss;
  int i;
  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-4";
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  };
  return ss.str();
}
}  // namespace uuid

/* Method definitions for Placement */
Placement::Placement(std::string taskName, Time startTime, Time endTime)
    : taskName(taskName),
      placed(true),
      startTime(startTime),
      endTime(endTime) {}

Placement::Placement(std::string taskName)
    : taskName(taskName),
      placed(false),
      startTime(std::nullopt),
      endTime(std::nullopt) {}

bool Placement::isPlaced() const {
  if (placed && (!startTime.has_value() || !endTime.has_value())) {
    throw tetrisched::exceptions::ExpressionSolutionException("Placement " +
                                                              taskName +
                                                              " is placed, "
                                                              "but does not "
                                                              "have a start "
                                                              "or end time.");
  }
  return placed;
}

void Placement::addPartitionAllocation(uint32_t partitionId, Time time,
                                       uint32_t allocation) {
  if (partitionToResourceAllocations.find(partitionId) ==
      partitionToResourceAllocations.end()) {
    partitionToResourceAllocations[partitionId] =
        std::set<std::pair<Time, uint32_t>>();
  }
  partitionToResourceAllocations[partitionId].insert(
      std::make_pair(time, allocation));
}

std::string Placement::getName() const { return taskName; }

void Placement::setStartTime(Time startTime) {
  this->startTime = startTime;
  this->placed = true;
}

void Placement::setEndTime(Time endTime) {
  this->endTime = endTime;
  this->placed = true;
}

std::optional<Time> Placement::getStartTime() const { return startTime; }

std::optional<Time> Placement::getEndTime() const { return endTime; }

const std::unordered_map<uint32_t, std::set<std::pair<Time, uint32_t>>>&
Placement::getPartitionAllocations() const {
  return partitionToResourceAllocations;
}

/* Method definitions for ExpressionTimeBounds */
ExpressionTimeBounds::ExpressionTimeBounds()
    : startTimeRange(
          {std::numeric_limits<Time>::min(), std::numeric_limits<Time>::max()}),
      endTimeRange(
          {std::numeric_limits<Time>::min(), std::numeric_limits<Time>::max()}),
      duration(0),
      specified(false) {}

ExpressionTimeBounds::ExpressionTimeBounds(TimeRange startTimeRange,
                                           TimeRange endTimeRange,
                                           Time duration)
    : startTimeRange(startTimeRange),
      endTimeRange(endTimeRange),
      duration(duration),
      specified(true) {}

bool ExpressionTimeBounds::isSpecified() const { return specified; }

std::string ExpressionTimeBounds::toString() const {
  return "S: [" + std::to_string(startTimeRange.first) + ", " +
         std::to_string(startTimeRange.second) + "], E: [" +
         std::to_string(endTimeRange.first) + ", " +
         std::to_string(endTimeRange.second) + "]";
}

/* Method definitions for Expression */

Expression::Expression(std::string name, ExpressionType type)
    : name(name),
      id(tetrisched::uuid::generate_uuid()),
      type(type),
      timeBounds(ExpressionTimeBounds()) {}

std::string Expression::getName() const { return name; }

std::string Expression::getId() const { return id; }

size_t Expression::getNumChildren() const { return children.size(); }

std::vector<ExpressionPtr> Expression::getChildren() const { return children; }

void Expression::addChild(ExpressionPtr child) {
  if (child == nullptr) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "Cannot add a null child to the Expression " + name + ".");
  }
  child->addParent(shared_from_this());
  children.push_back(child);
}

void Expression::removeChild(ExpressionPtr child) {
  if (child == nullptr) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "Cannot remove a null child from the Expression " + name + ".");
  }
  std::erase_if(children, [&child](ExpressionPtr& childPtr) {
    return childPtr->getId() == child->getId();
  });
}

ExpressionType Expression::getType() const { return type; }

std::string Expression::getTypeString() const {
  switch (type) {
    case ExpressionType::EXPR_CHOOSE:
      return "ChooseExpression";
    case ExpressionType::EXPR_OBJECTIVE:
      return "ObjectiveExpression";
    case ExpressionType::EXPR_MIN:
      return "MinExpression";
    case ExpressionType::EXPR_MAX:
      return "MaxExpression";
    case ExpressionType::EXPR_SCALE:
      return "ScaleExpression";
    case ExpressionType::EXPR_LESSTHAN:
      return "LessThanExpression";
    case ExpressionType::EXPR_ALLOCATION:
      return "AllocationExpression";
    case ExpressionType::EXPR_MALLEABLE_CHOOSE:
      return "MalleableChooseExpression";
    case ExpressionType::EXPR_WINDOWED_CHOOSE:
      return "WindowedChooseExpression";
    default:
      return "UnknownExpression";
  }
}

ExpressionTimeBounds Expression::getTimeBounds() const {
  if (getNumChildren() == 0) {
    return timeBounds;
  } else if (getNumChildren() == 1) {
    return children[0]->getTimeBounds();
  } else {
    if (type == ExpressionType::EXPR_LESSTHAN) {
      // If this is a LessThanExpression, we take the time bounds of the
      // first and second child.

      if (getNumChildren() != 2) {
        throw tetrisched::exceptions::ExpressionConstructionException(
            "LessThanExpression must have two children.");
      }

      // The start and end time ranges of a LessThanExpression are defined
      // by the start of the first child and the end of the second.
      return {
          .startTimeRange = children[0]->getTimeBounds().startTimeRange,
          .endTimeRange = children[1]->getTimeBounds().endTimeRange,
          .duration = children[0]->getTimeBounds().duration +
                      children[1]->getTimeBounds().duration,
      };
    } else {
      // For multiple children, we merge their time bounds, and find the
      // minimum duration.
      TimeRange startTimeRange =
          std::make_pair(std::numeric_limits<Time>::max(), 0);
      TimeRange endTimeRange =
          std::make_pair(std::numeric_limits<Time>::max(), 0);
      Time minDuration = std::numeric_limits<Time>::max();

      for (auto& child : children) {
        auto childTimeBounds = child->getTimeBounds();
        // If this child starts earlier than any other previous start time
        // bound, or starts later than any other previous child, use its bound.
        if (childTimeBounds.startTimeRange.first < startTimeRange.first) {
          startTimeRange.first = childTimeBounds.startTimeRange.first;
        }
        if (childTimeBounds.startTimeRange.second > startTimeRange.second) {
          startTimeRange.second = childTimeBounds.startTimeRange.second;
        }

        // If this child ends later than any other previous end time bound, use
        // it.
        if (childTimeBounds.endTimeRange.first < endTimeRange.first) {
          endTimeRange.first = childTimeBounds.endTimeRange.first;
        }
        if (childTimeBounds.endTimeRange.second > endTimeRange.second) {
          endTimeRange.second = childTimeBounds.endTimeRange.second;
        }

        // If this child has a smaller duration than any other previous
        // duration, use it.
        if (childTimeBounds.duration < minDuration) {
          minDuration = childTimeBounds.duration;
        }
      }

      return {
          .startTimeRange = startTimeRange,
          .endTimeRange = endTimeRange,
          .duration = minDuration,
      };
    }
  }
}

void Expression::setTimeBounds(ExpressionTimeBounds timeBounds) {
  this->timeBounds = timeBounds;
}

void Expression::addParent(ExpressionPtr parent) { parents.push_back(parent); }

size_t Expression::getNumParents() const { return parents.size(); }

std::vector<ExpressionPtr> Expression::getParents() const {
  std::vector<ExpressionPtr> returnParents;
  returnParents.reserve(parents.size());
  for (decltype(parents)::size_type i = 0; i < parents.size(); i++) {
    returnParents.push_back(parents[i].lock());
  }
  return returnParents;
}

SolutionResultPtr Expression::populateResults(SolverModelPtr solverModel) {
  TETRISCHED_DEBUG("Populating results for " << name << " of type "
                                             << getTypeString() << ".")
  // Check that the Expression was parsed before.
  if (!parsedResult) {
    solution = std::make_shared<SolutionResult>();
    solution->type = SolutionResultType::EXPRESSION_NO_UTILITY;
    return solution;
  }

  if (solution) {
    // Solution was already available, just return the same instance.
    return solution;
  }

  // Populate results for all children first.
  for (auto& childExpression : children) {
    auto _ = childExpression->populateResults(solverModel);
  }

  // Construct the SolutionResult.
  solution = std::make_shared<SolutionResult>();
  switch (parsedResult->type) {
    case ParseResultType::EXPRESSION_NO_UTILITY:
      TETRISCHED_DEBUG("Expression " << name << " of type " << getTypeString()
                                     << " had no utility to maximize.")
      solution->type = SolutionResultType::EXPRESSION_NO_UTILITY;
      return solution;
    case ParseResultType::EXPRESSION_UTILITY:
      solution->type = SolutionResultType::EXPRESSION_UTILITY;
      break;
    default:
      throw tetrisched::exceptions::ExpressionSolutionException(
          "Expression " + name +
          " was parsed with an invalid ParseResultType: " +
          std::to_string(static_cast<int>(parsedResult->type)));
  }

  // Retrieve the start, end times and the indicator from the SolverModel.
  if (!parsedResult->startTime) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression " + name +
        " with a utility was parsed without a start time.");
  }
  solution->startTime = parsedResult->startTime->resolve();
  TETRISCHED_DEBUG("Set start time to "
                   << solution->startTime.value() << " for expression " << name
                   << " of type " << getTypeString() << ".");

  if (!parsedResult->endTime) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression " + name +
        " with a utility was parsed without an end time.");
  }
  solution->endTime = parsedResult->endTime->resolve();
  TETRISCHED_DEBUG("Set end time to " << solution->endTime.value()
                                      << " for expression " << name
                                      << " of type " << getTypeString() << ".");

  if (!parsedResult->utility) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression " + name + " with a utility was parsed without a utility.");
  }
  solution->utility = parsedResult->utility.value()->getValue();
  TETRISCHED_DEBUG("Set utility to " << solution->utility.value()
                                     << " for expression " << name
                                     << " of type " << getTypeString() << ".");

  // Our default way of populating the placements is to retrieve the
  // children's placements and coalesce them into a single Placements map.
  for (auto& childExpression : children) {
    auto childExpressionSolution = childExpression->getSolution();
    if (!childExpressionSolution.has_value()) {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "Expression " + name +
          " with a utility was parsed without a solution for child " +
          childExpression->getName() + ".");
    }
    if (childExpressionSolution.value()->utility == 0) {
      // This child was not satisfied. Skip it.
      continue;
    }

    // The child was satisfied, merge its Placement objects with our own.
    for (auto& [taskName, placement] :
         childExpressionSolution.value()->placements) {
      solution->placements[taskName] = placement;
    }
  }

  TETRISCHED_DEBUG("Finished populating results for " << name << " of type "
                                                      << getTypeString() << ".")
  return solution;
}

std::optional<SolutionResultPtr> Expression::getSolution() const {
  if (!solution) {
    return std::nullopt;
  }
  return solution;
}

void Expression::exportToDot(std::string fileName) const {
  // Open the file for writing.
  std::ofstream dotFile;
  dotFile.open(fileName);

  // Output the Graph.
  dotFile << "digraph " << name << " {" << std::endl;

  // Go through all the nodes in the graph and output a mapping from
  // nodes to other nodes.
  std::stack<const Expression*> nodesToVisit;
  nodesToVisit.push(this);
  std::unordered_set<std::string> visitedNodes;

  while (!nodesToVisit.empty()) {
    const Expression* node = nodesToVisit.top();
    nodesToVisit.pop();

    // Check if we have already visited this node. If so, skip.
    if (visitedNodes.find(node->getId()) != visitedNodes.end()) {
      continue;
    }
    visitedNodes.insert(node->getId());

    // Output the label for the node.
    dotFile << "\"" << node->getId() << "\" [label=\""
            << node->getDescriptiveName() << "\"]" << std::endl;

    // Output the edges.
    for (auto& child : node->getChildren()) {
      dotFile << "\"" << node->getId() << "\" -> \"" << child->getId() << "\""
              << std::endl;
      nodesToVisit.push(child.get());
    }
  }

  // their ID to their name.
  dotFile << "}";
}

std::string Expression::getDescriptiveName() const {
  return this->getTypeString() + "(" + name + ")";
}

uint32_t Expression::getResourceQuantity() const { return 0; }

std::optional<ParseResultPtr> Expression::getParsedResult() const {
  // Check if the ParsedResultPtr is null, and return std::nullopt if so.
  if (!parsedResult) {
    return std::nullopt;
  } else {
    return parsedResult;
  }
}

/* Method definitions for ChooseExpression */

ChooseExpression::ChooseExpression(std::string taskName,
                                   Partitions resourcePartitions,
                                   uint32_t numRequiredMachines, Time startTime,
                                   Time duration, TETRISCHED_ILP_TYPE utility)
    : Expression(taskName, ExpressionType::EXPR_CHOOSE),
      resourcePartitions(resourcePartitions),
      numRequiredMachines(numRequiredMachines),
      startTime(startTime),
      duration(duration),
      endTime(startTime + duration),
      utility(utility) {
  // Set the time bounds for this Choice.
  timeBounds = ExpressionTimeBounds({startTime, startTime}, {endTime, endTime},
                                    duration);
}

void ChooseExpression::addChild(ExpressionPtr /* child */) {
  throw tetrisched::exceptions::ExpressionConstructionException(
      "ChooseExpression cannot have a child.");
}

uint32_t ChooseExpression::getResourceQuantity() const {
  return numRequiredMachines;
}

ParseResultPtr ChooseExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  // TETRISCHED_SCOPE_TIMER("ChooseExpression::parse," +
  //                        std::to_string(currentTime) + "," + name + "," + id)
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // return the already parsed sub-tree from another parent
    // this assumes a sub-tree can have > 1 parent and enables
    // STRL DAG structures
    return parsedResult;
  }
  // Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  if (currentTime > startTime) {
    TETRISCHED_DEBUG("Choose expression for "
                     << name << " to be placed starting at time " << startTime
                     << " and ending at " << endTime
                     << " does not provide utility because it is in the past.");
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
    return parsedResult;
  }
  TETRISCHED_DEBUG("Parsing Choose expression for "
                   << name << " to be placed starting at time " << startTime
                   << " and ending at " << endTime << ".");

  // Find the partitions that this Choose expression can be placed in.
  // This is the intersection of the Partitions that the Choose expression
  // was instantiated with and the Partitions that are available at the
  // time of the parsing.
  Partitions schedulablePartitions = resourcePartitions | availablePartitions;
  TETRISCHED_DEBUG("The Choose Expression for "
                   << name << " will be limited to "
                   << schedulablePartitions.size() << " partitions.");
  if (schedulablePartitions.size() == 0) {
    // There are no schedulable partitions, this expression cannot be
    // satisfied. and should provide 0 utility.
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
    return parsedResult;
  }

  // This Choose expression needs to be passed to the Solver.
  // We generate an Indicator variable for the Choose expression signifying
  // if this expression was satisfied.
  VariablePtr isSatisfiedVar = std::make_shared<Variable>(
      VariableType::VAR_INDICATOR,
      name + "_placed_at_" + std::to_string(startTime) + "_for_" + id);
  solverModel->addVariable(isSatisfiedVar);

  ConstraintPtr fulfillsDemandConstraint = std::make_shared<Constraint>(
      name + "_fulfills_demand_at_" + std::to_string(startTime) + "_for_" + id,
      ConstraintType::CONSTR_EQ, 0, schedulablePartitions.size() + 1);
  for (PartitionPtr& partition : schedulablePartitions.getPartitions()) {
    // For each partition, generate an integer that represents how many
    // resources were taken from this partition.
    VariablePtr allocationVar = std::make_shared<Variable>(
        VariableType::VAR_INTEGER,
        name + "_using_partition_" +
            std::to_string(partition->getPartitionId()) + "_at_" +
            std::to_string(startTime),
        0,
        std::min(static_cast<uint32_t>(partition->getQuantity()),
                 numRequiredMachines));
    solverModel->addVariable(allocationVar);

    // Save a reference to this Variable for this particular Partition.
    // We use this later to retrieve the placement.
    partitionVariables[partition->getPartitionId()] = allocationVar;

    // Add the variable to the demand constraint.
    fulfillsDemandConstraint->addTerm(allocationVar);

    // Register this indicator with the capacity constraints that
    // are being bubbled up.
    capacityConstraints->registerUsageForDuration(
        shared_from_this(), *partition, startTime, duration, isSatisfiedVar,
        allocationVar, std::nullopt);
  }
  // Ensure that if the Choose expression is satisfied, it fulfills the
  // demand for this expression. Pass the constraint to the model.
  fulfillsDemandConstraint->addTerm(
      -1 * static_cast<TETRISCHED_ILP_TYPE>(numRequiredMachines),
      isSatisfiedVar);
  solverModel->addConstraint(std::move(fulfillsDemandConstraint));

  // Construct the Utility function for this Choose expression.
  auto utilityFunction =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  utilityFunction->addTerm(utility, isSatisfiedVar);

  // Construct the return value.
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = startTime;
  parsedResult->endTime = endTime;
  parsedResult->indicator = isSatisfiedVar;
  parsedResult->utility = std::move(utilityFunction);
  return parsedResult;
}

SolutionResultPtr ChooseExpression::populateResults(
    SolverModelPtr solverModel) {
  // Populate the results for the SolverModel's variables (i.e, this
  // Expression's utility, start time and end time) from the Base Expression
  // class.
  Expression::populateResults(solverModel);

  // Populate the Placements from the SolverModel.
  if (!solution->utility || solution->utility.value() == 0) {
    // This Choose expression was not satisfied.
    // No placements to populate.
    return solution;
  }

  // Find the ID of the Partition that was chosen.
  PlacementPtr placement = std::make_shared<Placement>(
      name, solution->startTime.value(), solution->endTime.value());
  for (const auto& [partitionId, variable] : partitionVariables) {
    auto variableValue = variable->getValue();
    if (!variableValue.has_value()) {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "ChooseExpression " + name +
          " was parsed with a variable that does not have a value.");
    }

    if (std::abs(variableValue.value()) < 0.1) {
      // This partition was not used.
      continue;
    }
    // This partition was used. Add it to the Placement.
    placement->addPartitionAllocation(partitionId, solution->startTime.value(),
                                      variableValue.value());
  }
  solution->placements[name] = std::move(placement);
  return solution;
}

std::string ChooseExpression::getDescriptiveName() const {
  return "Choose(" + name + ", S=" + std::to_string(startTime) +
         ", F=" + std::to_string(endTime) + ")";
}

/* Method definitions for WindowedChooseExpression */

WindowedChooseExpression::WindowedChooseExpression(
    std::string taskName, Partitions resourcePartitions,
    uint32_t numRequiredMachines, Time startTime, Time duration, Time endTime,
    Time granularity, TETRISCHED_ILP_TYPE utility)
    : Expression(taskName, ExpressionType::EXPR_WINDOWED_CHOOSE),
      resourcePartitions(resourcePartitions),
      numRequiredMachines(numRequiredMachines),
      startTime(startTime),
      duration(duration),
      endTime(endTime),
      granularity(granularity),
      utility(utility) {
  // Set the time bounds for this Choice.
  Time endTimeLowerBound = static_cast<Time>(
      granularity *
      ceil(static_cast<double>(startTime + duration) / granularity));
  Time endTimeUpperBound = static_cast<Time>(
      granularity *
      ceil(static_cast<double>(endTime + duration) / granularity));
  timeBounds = ExpressionTimeBounds(
      {startTime, endTime}, {endTimeLowerBound, endTimeUpperBound}, duration);
}

void WindowedChooseExpression::addChild(ExpressionPtr /* child */) {
  throw tetrisched::exceptions::ExpressionConstructionException(
      "WindowedChooseExpression " + name + " cannot have a child.");
}

uint32_t WindowedChooseExpression::getResourceQuantity() const {
  return numRequiredMachines;
}

ParseResultPtr WindowedChooseExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  std::lock_guard<std::mutex> lockGuard(expressionMutex);
  TETRISCHED_SCOPE_TIMER("WindowedChooseExpression::parse," +
                         std::to_string(currentTime) + "," + name + "," + id)
  // Check that the Expression was parsed before.
  if (parsedResult != nullptr) {
    // Return the already parsed STRL-tree from another parent.
    return parsedResult;
  }

  // Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  if (currentTime > endTime) {
    TETRISCHED_DEBUG("WindowedChooseExpression for "
                     << name << " to be finished by " << endTime
                     << " does not provide utility because the current time is "
                     << currentTime)
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
    return parsedResult;
  }
  TETRISCHED_DEBUG("Parsing WindowedChooseExpression for "
                   << name << " to be placed after start time " << startTime
                   << ", running for " << duration << " and finishing by "
                   << endTime << ".")

  // Find the partitions that this Choose expression can be placed in.
  // This is the intersection of the Partitions that the Choose expression
  // was instantiated with and the Partitions that are available at the
  // time of the parsing.
  Partitions schedulablePartitions = resourcePartitions | availablePartitions;
  TETRISCHED_DEBUG("The Choose Expression for "
                   << name << " will be limited to "
                   << schedulablePartitions.size() << " partitions.");
  if (schedulablePartitions.size() == 0) {
    // There are no schedulable partitions, this expression cannot be
    // satisfied. and should provide 0 utility.
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
    return parsedResult;
  }

  // We now emit choices for each of the possible start times that finish
  // before the end time of the expression, at the provided granularity.
  size_t numChoices = 0;
  Time chooseTimeLowerBound = static_cast<Time>(
      granularity *
      ceil(static_cast<double>(timeBounds.startTimeRange.first) / granularity));
  Time chooseTimeUpperBound = static_cast<Time>(
      granularity * ceil(static_cast<double>(timeBounds.startTimeRange.second) /
                         granularity));
  TETRISCHED_DEBUG("Generating choices for " << name << " between "
                                             << chooseTimeLowerBound << " and "
                                             << chooseTimeUpperBound << ".")
  for (Time chooseTime = chooseTimeLowerBound;
       chooseTime <= chooseTimeUpperBound; chooseTime += granularity) {
    if (chooseTime + duration > timeBounds.endTimeRange.second) {
      // This choice is not valid, skip it.
      TETRISCHED_DEBUG("Skipping choice for "
                       << name << " at time " << chooseTime
                       << " as it does not fit within the finish time bounds: ("
                       << timeBounds.endTimeRange.first << ", "
                       << timeBounds.endTimeRange.second << ").")
      continue;
    }

    // Keep track of the number of choices being made.
    numChoices++;

    TETRISCHED_DEBUG("Generating a Choice for "
                     << name << " at time " << chooseTime
                     << " as part of the WindowedChooseExpression.")
    // Generate an indicator variable to signify if this expression was
    // placed at this particular chooseTime. Add the Variable to the
    // SolverModel, and maintain a reference to it for later population
    // of results.
    VariablePtr placedAtChooseTime = std::make_shared<Variable>(
        VariableType::VAR_INDICATOR,
        name + "_placed_at_" + std::to_string(chooseTime) + "_for_" + id);
    solverModel->addVariable(placedAtChooseTime);
    placementTimeVariables[chooseTime] = placedAtChooseTime;

    // Keep track of the allocation variables for this time.
    std::vector<std::pair<uint32_t, VariablePtr>> allocationVariables;

    ConstraintPtr fulfillsDemandConstraint = std::make_shared<Constraint>(
        name + "_fulfills_demand_at_" + std::to_string(chooseTime) + "_for_" +
            id,
        ConstraintType::CONSTR_EQ, 0, schedulablePartitions.size() + 1);
    for (PartitionPtr& partition : schedulablePartitions.getPartitions()) {
      // For each partition, we generate an integer that represents how many
      // resources were taken from this partition at this particular time.
      VariablePtr allocationVar = std::make_shared<Variable>(
          VariableType::VAR_INTEGER,
          name + "_using_partition_" +
              std::to_string(partition->getPartitionId()) + "_at_" +
              std::to_string(chooseTime),
          0,
          std::min(static_cast<uint32_t>(partition->getQuantity()),
                   numRequiredMachines));
      solverModel->addVariable(allocationVar);

      // Save the variable for this particular partition.
      allocationVariables.push_back(
          std::make_pair(partition->getPartitionId(), allocationVar));

      // Add the variable to the demand constraint for this time.
      fulfillsDemandConstraint->addTerm(allocationVar);

      // Register the allocation variable with the capacity constraints.
      capacityConstraints->registerUsageForDuration(
          shared_from_this(), *partition, chooseTime, duration,
          placedAtChooseTime, allocationVar, std::nullopt);
    }

    // Keep track of the allocation variables for this time.
    placementPartitionVariables[chooseTime] = allocationVariables;

    // Ensure that if the Choose expression is satisfied, it fulfills the
    // demand for this expression. Pass the constraint to the model.
    fulfillsDemandConstraint->addTerm(
        -1 * static_cast<TETRISCHED_ILP_TYPE>(numRequiredMachines),
        placedAtChooseTime);
    solverModel->addConstraint(std::move(fulfillsDemandConstraint));
  }

  if (numChoices == 0) {
    TETRISCHED_DEBUG(
        "WindowedChooseExpression "
        << name
        << " was instantiated with parameters that do not lead to any choices.")
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
    return parsedResult;
  }
  TETRISCHED_DEBUG("Generated " << numChoices << " choices for " << name
                                << " as part of the "
                                << "WindowedChooseExpression.")

  // All the choices have been enumerated now, we need to
  // set the start and end times accordingly.
  VariablePtr windowStartTime = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, name + "_start_time");
  solverModel->addVariable(windowStartTime);
  ConstraintPtr windowStartTimeConstraint = std::make_shared<Constraint>(
      name + "_start_time_constraint", ConstraintType::CONSTR_GE, 0,
      placementTimeVariables.size() + 3);

  VariablePtr windowEndTime =
      std::make_shared<Variable>(VariableType::VAR_INTEGER, name + "_end_time");
  solverModel->addVariable(windowEndTime);
  ConstraintPtr windowEndTimeConstraint = std::make_shared<Constraint>(
      name + "_end_time_constraint", ConstraintType::CONSTR_LE, 0,
      placementTimeVariables.size() + 1);

  // Add a constraint that forces only one choice to be allowed.
  VariablePtr windowIndicator = std::make_shared<Variable>(
      VariableType::VAR_INDICATOR, name + "_window_indicator");
  solverModel->addVariable(windowIndicator);
  ConstraintPtr chooseOneConstraint = std::make_shared<Constraint>(
      name + "_choose_one_constraint", ConstraintType::CONSTR_EQ, 0,
      placementTimeVariables.size() + 1);

  // Construct the Utility function for this ChooseExpression.
  auto utilityFunction =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);

  // Iterate over all the children and set the start, end times and
  // indicator accordingly.
  TimeRange startTimeBounds =
      std::make_pair(std::numeric_limits<Time>::max(), 0);
  TimeRange endTimeBounds = std::make_pair(std::numeric_limits<Time>::max(), 0);
  for (auto& [placementTime, placementVariable] : placementTimeVariables) {
    windowStartTimeConstraint->addTerm(placementTime, placementVariable);
    windowEndTimeConstraint->addTerm(placementTime + duration,
                                     placementVariable);
    chooseOneConstraint->addTerm(placementVariable);
    utilityFunction->addTerm(utility, placementVariable);

    if (placementTime < startTimeBounds.first) {
      startTimeBounds.first = placementTime;
    }
    if (placementTime > startTimeBounds.second) {
      startTimeBounds.second = placementTime;
    }
    if (placementTime + duration < endTimeBounds.first) {
      endTimeBounds.first = placementTime + duration;
    }
    if (placementTime + duration > endTimeBounds.second) {
      endTimeBounds.second = placementTime + duration;
    }
  }

  // This Expression is satisfied only if one of its children
  // are satisfied.
  chooseOneConstraint->addTerm(-1, windowIndicator);
  solverModel->addConstraint(std::move(chooseOneConstraint));

  // Constrain the start time to be less than the or equal to
  // start time of the placement choice that is satisfied.
  // If none are satisfied, we set the start time to the lower bound.
  windowStartTime->setLowerBound(
      -1 * static_cast<TETRISCHED_ILP_TYPE>(startTimeBounds.first));
  windowStartTime->setUpperBound(
      static_cast<TETRISCHED_ILP_TYPE>(startTimeBounds.second));
  windowStartTimeConstraint->addTerm(startTimeBounds.first);
  windowStartTimeConstraint->addTerm(
      -1 * static_cast<TETRISCHED_ILP_TYPE>(startTimeBounds.first),
      windowIndicator);
  windowStartTimeConstraint->addTerm(-1, windowStartTime);
  solverModel->addConstraint(std::move(windowStartTimeConstraint));
  TETRISCHED_DEBUG(
      "Setting bounds for start time of WindowedChooseExpression "
      << name << " to ["
      << -1 * static_cast<TETRISCHED_ILP_TYPE>(startTimeBounds.first) << ", "
      << startTimeBounds.second << "].")

  // Constrain the end time to be greater than or equal to the
  // end time of the placement choice that is satisfied.
  windowEndTime->setLowerBound(0);
  windowEndTime->setUpperBound(
      static_cast<TETRISCHED_ILP_TYPE>(endTimeBounds.second));
  windowEndTimeConstraint->addTerm(-1, windowEndTime);
  solverModel->addConstraint(std::move(windowEndTimeConstraint));
  TETRISCHED_DEBUG("Setting bounds for end time "
                   << name << " to [0, " << endTimeBounds.second << "].")

  // Construct and return the ParseResult.
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = windowStartTime;
  parsedResult->endTime = windowEndTime;
  parsedResult->indicator = windowIndicator;
  parsedResult->utility = std::move(utilityFunction);
  return parsedResult;
}

SolutionResultPtr WindowedChooseExpression::populateResults(
    SolverModelPtr solverModel) {
  // Populate the results for the SolverModel's variables (i.e, this
  // Expression's utility, start time and end time) from the Base Expression
  // class.
  Expression::populateResults(solverModel);

  // Populate the Placements from the SolverModel.
  if (!solution->utility || solution->utility.value() == 0) {
    // This Choose expression was not satisfied.
    // No placements to populate.
    return solution;
  }

  // Find the ID of the Partition that was chosen.
  PlacementPtr placement = std::make_shared<Placement>(
      name, solution->startTime.value(), solution->endTime.value());
  for (auto& [chooseTime, placementVariable] : placementTimeVariables) {
    auto placementVariableValue = placementVariable->getValue();
    if (!placementVariableValue.has_value()) {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "WindowedChooseExpression " + name +
          " was parsed with a variable that does not have a value.");
    }

    if (std::abs(placementVariableValue.value()) < 0.1) {
      // This placement was not used.
      TETRISCHED_DEBUG("[WindowedChoose] The placement for time "
                       << chooseTime << " and task " << name << " was 0.")
      continue;
    }
    TETRISCHED_DEBUG("[WindowedChoose] The placement for time "
                     << chooseTime << " and task " << name
                     << " was 1 because the placementValueVariable with name "
                     << placementVariable->getName() << " was "
                     << *placementVariableValue << ".")

    if (placementPartitionVariables.find(chooseTime) ==
        placementPartitionVariables.end()) {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "WindowedChooseExpression " + name +
          " was parsed without any allocation variables for choose time " +
          std::to_string(chooseTime));
    }

    // This placement was used. Add it to the Placement.
    for (auto& [partitionId, allocationVariable] :
         placementPartitionVariables[chooseTime]) {
      auto allocationVariableValue = allocationVariable->getValue();
      if (!allocationVariableValue.has_value()) {
        throw tetrisched::exceptions::ExpressionSolutionException(
            "WindowedChooseExpression " + name +
            " was parsed with a variable that does not have a value.");
      }
      if (std::abs(allocationVariableValue.value()) < 0.1) {
        // This partition was not used.
        continue;
      }
      // This partition was used. Add it to the Placement.
      placement->addPartitionAllocation(partitionId, chooseTime,
                                        allocationVariableValue.value());
    }
    placement->setStartTime(chooseTime);
    placement->setEndTime(chooseTime + duration);
  }
  solution->placements[name] = std::move(placement);
  return solution;
}

std::string WindowedChooseExpression::getDescriptiveName() const {
  return "WindowedChoose(" + name + ", " + timeBounds.toString() +
         ", D=" + std::to_string(duration) + ")";
}

/* Method definitions for MalleableChooseExpression */

MalleableChooseExpression::MalleableChooseExpression(
    std::string taskName, Partitions resourcePartitions,
    uint32_t resourceTimeSlots, Time startTime, Time endTime, Time granularity,
    TETRISCHED_ILP_TYPE utility)
    : Expression(taskName, ExpressionType::EXPR_MALLEABLE_CHOOSE),
      resourcePartitions(resourcePartitions),
      resourceTimeSlots(resourceTimeSlots),
      startTime(startTime),
      endTime(endTime),
      granularity(granularity),
      utility(utility),
      partitionVariables() {
  if (endTime < startTime) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "MalleableChooseExpression " + name +
        " was instantiated with an end time " + std::to_string(endTime) +
        " that is before its start time " + std::to_string(startTime) + ".");
  }
  // Set the time bounds for this Choice.
  timeBounds = ExpressionTimeBounds({startTime, startTime}, {endTime, endTime},
                                    endTime - startTime);
}

void MalleableChooseExpression::addChild(ExpressionPtr /* child */) {
  throw tetrisched::exceptions::ExpressionConstructionException(
      "MalleableChooseExpression cannot have a child.");
}

uint32_t MalleableChooseExpression::getResourceQuantity() const {
  return resourceTimeSlots;
}

ParseResultPtr MalleableChooseExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  std::lock_guard<std::mutex> lockGuard(expressionMutex);
  TETRISCHED_SCOPE_TIMER("MalleableChooseExpression::parse," +
                         std::to_string(currentTime) + "," + name + "," + id)
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // Return the alread parsed sub-tree.
    return parsedResult;
  }

  // Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  if (currentTime > startTime) {
    TETRISCHED_DEBUG("MalleableChooseExpression "
                     << name << " to be placed starting at time " << startTime
                     << " and ending at " << endTime
                     << " does not provide utility because it is in the past.");
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
    return parsedResult;
  }
  TETRISCHED_DEBUG("Parsing MalleableChooseExpression "
                   << name << " to be placed starting at time " << startTime
                   << " and ending at " << endTime << ".")

  // Find the partitions that this Choose expression can be placed in.
  // This is the intersection of the Partitions that the Choose expression
  // was instantiated with and the Partitions that are available at the
  // time of the parsing.
  Partitions schedulablePartitions = resourcePartitions | availablePartitions;
  TETRISCHED_DEBUG("The MalleableChooseExpression "
                   << name << " will be limited to "
                   << schedulablePartitions.size() << " partitions.");

  // We generate an Indicator variable for the Choose expression signifying
  // if this expression was satisfied.
  std::string satisfiedVarName = name + "_placed_from_" +
                                 std::to_string(startTime) + "_to_" +
                                 std::to_string(endTime);
  VariablePtr isSatisfiedVar =
      std::make_shared<Variable>(VariableType::VAR_INDICATOR, satisfiedVarName);
  solverModel->addVariable(isSatisfiedVar);
  TETRISCHED_DEBUG(
      "The MalleableChooseExpression's satisfaction will be indicated by "
      << satisfiedVarName << ".");

  ConstraintPtr fulfillsDemandConstraint = std::make_shared<Constraint>(
      name + "_fulfills_demand_from_" + std::to_string(startTime) + "_to_" +
          std::to_string(endTime),
      ConstraintType::CONSTR_EQ, 0, schedulablePartitions.size() + 1);

  // For each partition, and each time unit, generate an integer that
  // represents how many resources were taken from this partition at
  // this particular time.
  for (PartitionPtr& partition : schedulablePartitions.getPartitions()) {
    for (auto time = startTime; time < endTime; time += granularity) {
      auto mapKey = std::make_pair(partition->getPartitionId(), time);
      if (partitionVariables.find(mapKey) == partitionVariables.end()) {
        // Create a new Integer variable specifying how many resources we
        // have from this Partition.
        VariablePtr allocationAtTime = std::make_shared<Variable>(
            VariableType::VAR_INTEGER,
            name + "_using_partition_" +
                std::to_string(partition->getPartitionId()) + "_at_" +
                std::to_string(time),
            0,
            std::min(static_cast<uint32_t>(partition->getQuantity()),
                     resourceTimeSlots));
        solverModel->addVariable(allocationAtTime);
        partitionVariables[mapKey] = allocationAtTime;
        fulfillsDemandConstraint->addTerm(allocationAtTime);

        // Register this Integer variable with the CapacityConstraintMap
        // that is being bubbled up.
        capacityConstraints->registerUsageForDuration(
            shared_from_this(), *partition, time, granularity, isSatisfiedVar,
            allocationAtTime, std::nullopt);
      } else {
        throw tetrisched::exceptions::ExpressionConstructionException(
            "Multiple variables detected for the Partition " +
            partition->getPartitionName() + " at time " + std::to_string(time));
      }
    }
  }

  // Ensure that if the Choose expression is satisfied, it fulfills the
  // demand for this expression. Pass the constraint to the model.
  fulfillsDemandConstraint->addTerm(
      -1 * static_cast<TETRISCHED_ILP_TYPE>(resourceTimeSlots), isSatisfiedVar);
  solverModel->addConstraint(std::move(fulfillsDemandConstraint));

  // We now need to ensure that for each time, there is an indicator variable
  // that signifies if there was *any* allocation to this Expression at that
  // time.
  std::unordered_map<Time, VariablePtr> timeToOccupationIndicator;
  // For each time instance, we have an Indicator variable that specifies if
  // this Expression is using any partitions at that time. To set the correct
  // values for the Indicator, we add a constraint such that:
  // Indicator <= Sum(Allocation). Thus, if there is no allocation (i.e., the
  // sum of Partition variables is 0, then Indicator has to be 0).
  std::unordered_map<Time, ConstraintPtr> timeToLowerBoundConstraint;
  // In the above example, we correctly set the Indicator if there is no
  // allocation to the Expression. However, if there is an allocation, we
  // need to ensure that the Indicator is set to 1. We do this by adding a
  // constraint such that: Sum(Allocation) <= Indicator * resourceTimeSlots.
  // Thus, there was any allocation, then the Indicator has to be 1.
  std::unordered_map<Time, ConstraintPtr> timeToUpperBoundConstraint;
  for (auto& [key, variable] : partitionVariables) {
    auto& [partition, time] = key;

    // Generate the Indicator variable for this time.
    if (timeToOccupationIndicator.find(time) ==
        timeToOccupationIndicator.end()) {
      VariablePtr occupationAtTime = std::make_shared<Variable>(
          VariableType::VAR_INDICATOR,
          name + "_occupied_at_" + std::to_string(time));
      solverModel->addVariable(occupationAtTime);
      timeToOccupationIndicator[time] = occupationAtTime;
    }

    // Lower bound the Indicator variable to allow a value of 0.
    if (timeToLowerBoundConstraint.find(time) ==
        timeToLowerBoundConstraint.end()) {
      ConstraintPtr lowerBoundConstraint = std::make_shared<Constraint>(
          name + "_lower_bound_occupation_at_" + std::to_string(time),
          ConstraintType::CONSTR_LE, 0, 1);
      solverModel->addConstraint(lowerBoundConstraint);
      lowerBoundConstraint->addTerm(1, timeToOccupationIndicator[time]);
      timeToLowerBoundConstraint[time] = lowerBoundConstraint;
    }
    timeToLowerBoundConstraint[time]->addTerm(-1, variable);

    // Upper bound the Indicator variable to allow a value of 1.
    if (timeToUpperBoundConstraint.find(time) ==
        timeToUpperBoundConstraint.end()) {
      ConstraintPtr upperBoundConstraint = std::make_shared<Constraint>(
          name + "_upper_bound_occupation_at_" + std::to_string(time),
          ConstraintType::CONSTR_LE, 0, 1);
      solverModel->addConstraint(upperBoundConstraint);
      upperBoundConstraint->addTerm(
          -1 * static_cast<TETRISCHED_ILP_TYPE>(resourceTimeSlots),
          timeToOccupationIndicator[time]);
      timeToUpperBoundConstraint[time] = upperBoundConstraint;
    }
    timeToUpperBoundConstraint[time]->addTerm(1, variable);
  }

  // Now that we have the Indicator variables specifying if there is
  // an allocation to the Task for each time, we need to find the first
  // time when the Task is allocated any resources. To do this, we add
  // a new set of Indicator variables such that only one of them is
  // set to 1, and the one that is set to 1 indicates the first phase-shift
  // of the resource assignments (i.e., 0...0, 1)
  std::unordered_map<Time, VariablePtr> timeToPhaseShiftIndicatorForStartTime;
  for (auto time = startTime; time < endTime; time += granularity) {
    VariablePtr phaseShiftIndicator = std::make_shared<Variable>(
        VariableType::VAR_INDICATOR,
        name + "_phase_shift_start_time_at_" + std::to_string(time));
    solverModel->addVariable(phaseShiftIndicator);
    timeToPhaseShiftIndicatorForStartTime[time] = phaseShiftIndicator;
  }

  // Add a constraint that forces only one phase shift to be allowed.
  ConstraintPtr startTimePhaseShiftGUBConstraint = std::make_shared<Constraint>(
      name + "_phase_shift_start_time_gub_constraint",
      ConstraintType::CONSTR_LE, 1,
      timeToPhaseShiftIndicatorForStartTime.size());
  for (auto& [time, variable] : timeToPhaseShiftIndicatorForStartTime) {
    startTimePhaseShiftGUBConstraint->addTerm(variable);
  }
  solverModel->addConstraint(std::move(startTimePhaseShiftGUBConstraint));

  // Add constraints that ensures that the phase shift does not happen
  // when the resource allocation indicator is 0.
  for (auto& [time, variable] : timeToPhaseShiftIndicatorForStartTime) {
    ConstraintPtr phaseShiftLowerBoundConstraint = std::make_shared<Constraint>(
        name + "_phase_shift_start_time_constraint_lower_bounded_at_" +
            std::to_string(time),
        ConstraintType::CONSTR_LE, 0, 2);
    phaseShiftLowerBoundConstraint->addTerm(variable);
    phaseShiftLowerBoundConstraint->addTerm(-1,
                                            timeToOccupationIndicator[time]);
    solverModel->addConstraint(std::move(phaseShiftLowerBoundConstraint));
  }

  // Add constraints that ensure that each allocation indicator is
  // less than or equal to the sum of its past phase-shift indicators.
  // This critical constraint ensures that the first time the allocation
  // turns to 1, the phase shift indicator is set to 1.
  for (auto& [allocationTime, occupationIndicator] :
       timeToOccupationIndicator) {
    ConstraintPtr phaseShiftConstraint = std::make_shared<Constraint>(
        name + "_phase_shift_start_time_constraint_at_" +
            std::to_string(allocationTime),
        ConstraintType::CONSTR_LE, 0,
        timeToPhaseShiftIndicatorForStartTime.size() + 1);
    phaseShiftConstraint->addTerm(occupationIndicator);
    for (auto& [phaseShiftTime, phaseShiftIndicator] :
         timeToPhaseShiftIndicatorForStartTime) {
      if (phaseShiftTime > allocationTime) {
        // We only care about the phase shift indicators up-till this point.
        continue;
      }
      phaseShiftConstraint->addTerm(-1, phaseShiftIndicator);
    }
    solverModel->addConstraint(std::move(phaseShiftConstraint));
  }

  // Emit the start-time of the Expression using the phase-shift indicators.
  VariablePtr startTimeVariable = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, name + "_start_time", 0, endTime);
  solverModel->addVariable(startTimeVariable);
  ConstraintPtr startTimeConstraint = std::make_shared<Constraint>(
      name + "_start_time_constraint", ConstraintType::CONSTR_EQ, 0,
      timeToPhaseShiftIndicatorForStartTime.size() + 1);
  for (auto& [time, phaseShiftIndicator] :
       timeToPhaseShiftIndicatorForStartTime) {
    startTimeConstraint->addTerm(time, phaseShiftIndicator);
  }
  startTimeConstraint->addTerm(-1, startTimeVariable);
  solverModel->addConstraint(std::move(startTimeConstraint));

  // Similar to the start time, we generate indicator variables for
  // the end time of the Expression by reversing the phase-shift assignment.
  std::unordered_map<Time, VariablePtr> timeToPhaseShiftIndicatorForEndTime;
  for (auto time = startTime; time < endTime; time += granularity) {
    VariablePtr phaseShiftIndicator = std::make_shared<Variable>(
        VariableType::VAR_INDICATOR,
        name + "_phase_shift_end_time_at_" + std::to_string(time));
    solverModel->addVariable(phaseShiftIndicator);
    timeToPhaseShiftIndicatorForEndTime[time] = phaseShiftIndicator;
  }

  // Only one of the end time phase shifts is allowed.
  ConstraintPtr endTimePhaseShiftGUBConstraint = std::make_shared<Constraint>(
      name + "_phase_shift_end_time_gub_constraint", ConstraintType::CONSTR_LE,
      1, timeToPhaseShiftIndicatorForEndTime.size());
  for (auto& [time, variable] : timeToPhaseShiftIndicatorForEndTime) {
    endTimePhaseShiftGUBConstraint->addTerm(variable);
  }
  solverModel->addConstraint(std::move(endTimePhaseShiftGUBConstraint));

  // Add constraints that ensure that the phase shift does not happen
  // when the resource allocation indicator is 0.
  for (auto& [time, variable] : timeToPhaseShiftIndicatorForEndTime) {
    ConstraintPtr phaseShiftLowerBoundConstraint = std::make_shared<Constraint>(
        name + "_phase_shift_end_time_constraint_lower_bounded_at_" +
            std::to_string(time),
        ConstraintType::CONSTR_LE, 0, 2);
    phaseShiftLowerBoundConstraint->addTerm(variable);
    phaseShiftLowerBoundConstraint->addTerm(-1,
                                            timeToOccupationIndicator[time]);
    solverModel->addConstraint(std::move(phaseShiftLowerBoundConstraint));
  }

  // Add constraints that ensure that each allocation indicator is
  // less than or equal to the sum of its future phase-shift indicators.
  for (auto& [allocationTime, occupationIndicator] :
       timeToOccupationIndicator) {
    ConstraintPtr phaseShiftConstraint = std::make_shared<Constraint>(
        name + "_phase_shift_end_time_constraint_at_" +
            std::to_string(allocationTime),
        ConstraintType::CONSTR_LE, 0,
        timeToPhaseShiftIndicatorForEndTime.size() + 1);
    phaseShiftConstraint->addTerm(occupationIndicator);
    for (auto& [phaseShiftTime, phaseShiftIndicator] :
         timeToPhaseShiftIndicatorForEndTime) {
      if (phaseShiftTime < allocationTime) {
        // We only care about the phase shift indicators after this point.
        continue;
      }
      phaseShiftConstraint->addTerm(-1, phaseShiftIndicator);
    }
    solverModel->addConstraint(std::move(phaseShiftConstraint));
  }

  // Emit the end-time of the Expression using the phase-shift indicators.
  VariablePtr endTimeVariable = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, name + "_end_time", 0, endTime);
  solverModel->addVariable(endTimeVariable);
  ConstraintPtr endTimeConstraint = std::make_shared<Constraint>(
      name + "_end_time_constraint", ConstraintType::CONSTR_EQ, 0,
      timeToPhaseShiftIndicatorForEndTime.size() + 1);
  for (auto& [time, phaseShiftIndicator] :
       timeToPhaseShiftIndicatorForEndTime) {
    endTimeConstraint->addTerm(time, phaseShiftIndicator);
  }
  endTimeConstraint->addTerm(-1, endTimeVariable);
  solverModel->addConstraint(std::move(endTimeConstraint));

  // Construct the Utility function for this Choose expression.
  auto utilityFunction =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  utilityFunction->addTerm(utility, isSatisfiedVar);

  // Construct the return value.
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = startTimeVariable;
  parsedResult->endTime = endTimeVariable;
  parsedResult->indicator = isSatisfiedVar;
  parsedResult->utility = std::move(utilityFunction);
  return parsedResult;
}

SolutionResultPtr MalleableChooseExpression::populateResults(
    SolverModelPtr solverModel) {
  // Populate the results for the SolverModel's variables (i.e, this
  // Expression's utility, start time and end time) from the Base Expression
  // class.
  Expression::populateResults(solverModel);

  // Populate the Placements from the SolverModel.
  if (!solution->utility || solution->utility.value() == 0) {
    // This Choose expression was not satisfied.
    // No placements to populate.
    return solution;
  }

  // Find the IDs of the Partitions that were chosen.
  PlacementPtr placement = std::make_shared<Placement>(
      name, solution->startTime.value(), solution->endTime.value());
  for (const auto& [partitionPair, variable] : partitionVariables) {
    auto variableValue = variable->getValue();
    if (!variableValue.has_value()) {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "MalleableChooseExpression " + name +
          " was parsed with a variable that does not have a value.");
    }
    if (std::abs(variableValue.value()) < 0.1) {
      // This partition was not used.
      continue;
    }
    auto& [partitionId, time] = partitionPair;
    placement->addPartitionAllocation(partitionId, time, variableValue.value());
  }
  solution->placements[name] = std::move(placement);
  return solution;
}

std::string MalleableChooseExpression::getDescriptiveName() const {
  return "MalleableChoose(" + name + ", S=" + std::to_string(startTime) +
         ", F=" + std::to_string(endTime) + ")";
}

/* Method definitions for AllocationExpression */

AllocationExpression::AllocationExpression(
    std::string taskName,
    std::vector<std::pair<PartitionPtr, uint32_t>> allocatedResources,
    Time startTime, Time duration)
    : Expression(taskName, ExpressionType::EXPR_ALLOCATION),
      allocatedResources(allocatedResources),
      startTime(startTime),
      duration(duration),
      endTime(startTime + duration) {
  // Set the time bounds for this Choice.
  timeBounds = ExpressionTimeBounds({startTime, startTime}, {endTime, endTime},
                                    duration);
}

void AllocationExpression::addChild(ExpressionPtr /* child */) {
  throw tetrisched::exceptions::ExpressionConstructionException(
      "AllocationExpression cannot have a child.");
}

ParseResultPtr AllocationExpression::parse(
    SolverModelPtr /* solverModel */, Partitions /* availablePartitions */,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  std::lock_guard<std::mutex> lockGuard(expressionMutex);
  TETRISCHED_SCOPE_TIMER("AllocationExpression::parse," +
                         std::to_string(currentTime) + "," + name + "," + id)
  // Check that the Expression was parsed before.
  if (parsedResult != nullptr) {
    // Return the already parsed sub-tree.
    return parsedResult;
  }

  // Create and save the ParseResult.
  TETRISCHED_DEBUG("Parsing AllocationExpression for "
                   << name << " to be placed starting at time " << startTime
                   << " and ending at " << endTime << ".")
  parsedResult = std::make_shared<ParseResult>();
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = startTime;
  parsedResult->endTime = endTime;
  parsedResult->indicator = 1;
  parsedResult->utility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  (parsedResult->utility).value()->addTerm(1);
  for (const auto& [partition, allocation] : allocatedResources) {
    capacityConstraints->registerUsageForDuration(
        shared_from_this(), *partition, startTime, duration, 1, allocation,
        std::nullopt);
  }
  TETRISCHED_DEBUG("Finished parsing AllocationExpression for "
                   << name << " to be placed starting at time " << startTime
                   << " and ending at " << endTime << ".")
  return parsedResult;
}

SolutionResultPtr AllocationExpression::populateResults(
    SolverModelPtr solverModel) {
  // Populate the results for the SolverModel's variables (i.e, this
  // Expression's utility, start time and end time) from the Base Expression
  // class.
  Expression::populateResults(solverModel);
  return solution;
}

std::string AllocationExpression::getDescriptiveName() const {
  return "Allocation(" + name + ", S=" + std::to_string(startTime) +
         ", F=" + std::to_string(endTime) + ")";
}

/* Method definitions for ObjectiveExpression */

ObjectiveExpression::ObjectiveExpression(std::string name)
    : Expression(name, ExpressionType::EXPR_OBJECTIVE) {}

ParseResultPtr ObjectiveExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  std::lock_guard<std::mutex> lockGuard(expressionMutex);
  TETRISCHED_SCOPE_TIMER("ObjectiveExpression::parse," +
                         std::to_string(currentTime) + "," + name + "," + id)
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // return the already parsed sub-tree from another parent
    // this assumes a sub-tree can have > 1 parent and enables
    // STRL DAG structures
    return parsedResult;
  }
  TETRISCHED_DEBUG("Parsing ObjectiveExpression with name " << name << ".")

  // Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;

  // Construct the overall utility of this expression.
  auto utility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);

  // Parse the children and collect the utiltiies.
  for (auto& child : children) {
    auto result = child->parse(solverModel, availablePartitions,
                               capacityConstraints, currentTime);
    if (result->type == ParseResultType::EXPRESSION_UTILITY) {
      if (!result->utility.has_value()) {
        throw tetrisched::exceptions::ExpressionConstructionException(
            "ObjectiveExpression " + name + "'s child " + child->getName() +
            " was supposed to provide utility, but doesn't.");
      }
      (*utility) += *(result->utility.value());
    }
  }
  TETRISCHED_DEBUG("Finished parsing the children for ObjectiveExpression with "
                   << name << ".")

  // All the children have been parsed. Finalize the CapacityConstraintMap.
  TETRISCHED_DEBUG(
      "Finalizing the CapacityConstraintMap for ObjectiveExpression " << name
                                                                      << ".")
  capacityConstraints->translate(solverModel);
  TETRISCHED_DEBUG("Finished finalizing the CapacityConstraintMap for " << name
                                                                        << ".")

  // Construct the parsed result.
  parsedResult->utility = std::make_shared<ObjectiveFunction>(*utility);
  parsedResult->startTime = std::numeric_limits<Time>::min();
  parsedResult->endTime = std::numeric_limits<Time>::max();

  // Add the utility to the SolverModel.
  solverModel->setObjectiveFunction(std::move(utility));

  TETRISCHED_DEBUG("Finished parsing ObjectiveExpression with name " << name
                                                                     << ".")

  return parsedResult;
}

SolutionResultPtr ObjectiveExpression::populateResults(
    SolverModelPtr solverModel) {
  // Use the Base definition for populating everything.
  Expression::populateResults(solverModel);

  if (solution->utility == 0) {
    // This ObjectiveExpression was not satisfied.
    // No start and end time to fix.
    return solution;
  }

  // We don't specify the start time and end times in ObjectiveExpression's
  // model. We can, however, retrieve them now that all the children have been
  // evaluated.
  Time minStartTime = std::numeric_limits<Time>::max();
  Time maxEndTime = std::numeric_limits<Time>::min();
  for (auto& childExpression : children) {
    auto childExpressionSolution = childExpression->getSolution();
    if (!childExpressionSolution.has_value()) {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "ObjectiveExpression " + name + "'s child " +
          childExpression->getName() + " does not have a solution.");
    }

    auto childExpressionSolutionValue = childExpressionSolution.value();

    // If the child expression was not satisfied, skip it.
    if (childExpressionSolutionValue->type !=
            ParseResultType::EXPRESSION_UTILITY ||
        childExpressionSolutionValue->utility == 0) {
      continue;
    }

    if (!childExpressionSolutionValue->startTime.has_value() ||
        !childExpressionSolutionValue->endTime.has_value()) {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "ObjectiveExpression " + name + "'s child " +
          childExpression->getName() + " does not have a start or end time.");
    }

    // If the child has a smaller start time, use it.
    if (childExpressionSolutionValue->startTime.value() < minStartTime) {
      minStartTime = childExpressionSolution.value()->startTime.value();
    }

    // If the child has a larger end time, use it.
    if (childExpressionSolutionValue->endTime.value() > maxEndTime) {
      maxEndTime = childExpressionSolution.value()->endTime.value();
    }
  }

  // Set up the start and end times correctly.
  solution->startTime = minStartTime;
  solution->endTime = maxEndTime;
  return solution;
}

/* Method definitions for LessThanExpression */

LessThanExpression::LessThanExpression(std::string name)
    : Expression(name, ExpressionType::EXPR_LESSTHAN) {}

void LessThanExpression::addChild(ExpressionPtr child) {
  if (children.size() == 2) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "LessThanExpression cannot have more than two children.");
  }
  Expression::addChild(child);
}

ParseResultPtr LessThanExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  std::lock_guard<std::mutex> lockGuard(expressionMutex);
  TETRISCHED_SCOPE_TIMER("LessThanExpression::parse," +
                         std::to_string(currentTime) + "," + name + "," + id)
  // Sanity check the children.
  if (children.size() != 2) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "LessThanExpression must have two children.");
  }

  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // return the already parsed sub-tree from another parent
    // this assumes a sub-tree can have > 1 parent and enables
    // STRL DAG structures
    return parsedResult;
  }

  TETRISCHED_DEBUG("Parsing LessThanExpression with name " << name << ".")

  // Parse both the children.
  auto firstChildResult = children[0]->parse(solverModel, availablePartitions,
                                             capacityConstraints, currentTime);
  auto secondChildResult = children[1]->parse(solverModel, availablePartitions,
                                              capacityConstraints, currentTime);
  TETRISCHED_DEBUG(
      "Finished parsing the children for LessThanExpression with name " << name
                                                                        << ".")

  // Generate the result of parsing the expression.
  parsedResult = std::make_shared<ParseResult>();

  // If either of the children cannot be satisfied, then this expression cannot
  // be satisfied.
  if (firstChildResult->type != ParseResultType::EXPRESSION_UTILITY ||
      secondChildResult->type != ParseResultType::EXPRESSION_UTILITY) {
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
    TETRISCHED_DEBUG("LessThanExpression with name "
                     << name
                     << " has no utility because one of its children has no "
                        "utility.")
    return parsedResult;
  }

  // If we have known values for the children, then we can just do a simple
  // check instead of emitting constraints into the model.
  if (!firstChildResult->endTime->isVariable() &&
      !secondChildResult->startTime->isVariable()) {
    auto firstChildEndTime =
        static_cast<int64_t>(firstChildResult->endTime->get<Time>());
    auto secondChildStartTime =
        static_cast<int64_t>(secondChildResult->startTime->get<Time>());

    if (firstChildEndTime <= secondChildStartTime) {
      // This clause is trivially satisfied, populate the results.
      parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
      parsedResult->startTime.emplace(firstChildResult->startTime.value());
      parsedResult->endTime.emplace(secondChildResult->endTime.value());
      parsedResult->indicator = 1;
      TETRISCHED_DEBUG(
          "LessThanExpression with name "
          << name
          << " has been trivially satisfied since the left child's end time is "
          << firstChildEndTime << " and the right child's start time is "
          << secondChildStartTime << ".")
    } else {
      // The first child does not end by the time the second child has to start,
      // this expression cannot be satisfied.
      parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
      TETRISCHED_DEBUG(
          "LessThanExpression with name "
          << name << " has no utility because the first child ends ("
          << firstChildEndTime << ") after the second child starts ("
          << secondChildStartTime << ").");
      return parsedResult;
    }
  } else {
    // We do not have known values for the children, so we need to emit
    // constraints into the model.
    parsedResult->type = ParseResultType::EXPRESSION_UTILITY;

    // Bubble up the start time of the first expression and the end time of
    // the second expression as a bound on the
    if (!firstChildResult->endTime || !secondChildResult->startTime ||
        !firstChildResult->startTime || !secondChildResult->endTime) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          "LessThanExpression must have children with start and end times.");
    }
    parsedResult->startTime.emplace(firstChildResult->startTime.value());
    parsedResult->endTime.emplace(secondChildResult->endTime.value());

    // Add a constraint that the Expression is only satisfied if both the
    // children are satisfied.
    VariablePtr isSatisfiedVar = std::make_shared<Variable>(
        VariableType::VAR_INDICATOR, name + "_is_satisfied");
    solverModel->addVariable(isSatisfiedVar);

    // Ensure that both of the children of the LessThanExpression can be
    // enforced.
    ConstraintPtr lessThanIndicatorConstraint =
        std::make_shared<Constraint>(name + "_less_than_indicator_constraint",
                                     ConstraintType::CONSTR_EQ, 0, 3);
    double numEnforceableChildren = 0;

    if (!firstChildResult->indicator || !secondChildResult->indicator) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          "LessThanExpression must have children with indicators.");
    }
    if (firstChildResult->indicator->isVariable()) {
      lessThanIndicatorConstraint->addTerm(
          firstChildResult->indicator->get<VariablePtr>());
      numEnforceableChildren += 1;
    } else if (firstChildResult->indicator->get<uint32_t>() == 0) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          "LessThanExpression must have children with non-0 indicators. "
          "Child " +
          children[0]->getName() + " has indicator 0.");
    }
    if (secondChildResult->indicator->isVariable()) {
      lessThanIndicatorConstraint->addTerm(
          secondChildResult->indicator->get<VariablePtr>());
      numEnforceableChildren += 1;
    } else if (secondChildResult->indicator->get<uint32_t>() == 0) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          "LessThanExpression must have children with non-0 indicators. "
          "Child " +
          children[1]->getName() + " has indicator 0.");
    }
    lessThanIndicatorConstraint->addTerm(-1 * numEnforceableChildren,
                                         isSatisfiedVar);
    solverModel->addConstraint(lessThanIndicatorConstraint);
    parsedResult->indicator = isSatisfiedVar;

    // Add a constraint that the first child must occur before the second.
    auto happensBeforeConstraintName = name + "_happens_before_constraint";
    ConstraintPtr happensBeforeConstraint = std::make_shared<Constraint>(
        happensBeforeConstraintName, ConstraintType::CONSTR_LE, 0, 2);
    happensBeforeConstraint->addTerm(firstChildResult->endTime.value());
    happensBeforeConstraint->addTerm(-1, secondChildResult->startTime.value());
    solverModel->addConstraint(std::move(happensBeforeConstraint));
    TETRISCHED_DEBUG("Finished adding constraint "
                     << happensBeforeConstraintName
                     << " to enforce ordering in LessThanExpression with name "
                     << name << ".")
  }

  // Construct a utility function that is the sum of the two utilities.
  if (!firstChildResult->utility || !secondChildResult->utility) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "LessThanExpression must have children with utilities.");
  }

  auto utility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  (*utility) += *(firstChildResult->utility.value());
  (*utility) += *(secondChildResult->utility.value());

  parsedResult->utility = std::move(utility);

  // Return the result.
  return parsedResult;
}

/* Method definitions for MinExpression */

MinExpression::MinExpression(std::string name)
    : Expression(name, ExpressionType::EXPR_MIN) {}

ParseResultPtr MinExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  std::lock_guard<std::mutex> lockGuard(expressionMutex);
  TETRISCHED_SCOPE_TIMER("MinExpression::parse," + std::to_string(currentTime) +
                         "," + name + "," + id)
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // Return the already parsed sub-tree from another parent
    // this assumes a sub-tree can have > 1 parent and enables
    // STRL DAG structures
    return parsedResult;
  }

  /// Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  // Check that there are more than 0 children.
  auto numChildren = this->getNumChildren();
  if (numChildren == 0) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Number of children should be >=1 for MIN");
  }

  // Indicator for MIN.
  VariablePtr minIndicator = std::make_shared<Variable>(
      VariableType::VAR_INDICATOR, name + "_min_indicator");
  solverModel->addVariable(minIndicator);

  // Start Time of MIN
  VariablePtr minStartTime = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, name + "_min_start_time");
  solverModel->addVariable(minStartTime);

  // End Time of MIN
  VariablePtr minEndTime = std::make_shared<Variable>(VariableType::VAR_INTEGER,
                                                      name + "_min_end_time");
  solverModel->addVariable(minEndTime);

  // Constraint to ensure that all children of the MIN operator are satisfied.
  ConstraintPtr minIndicatorConstraint = std::make_shared<Constraint>(
      name + "_min_enforce_all_children", ConstraintType::CONSTR_EQ, 0,
      numChildren + 1);
  std::atomic_uint32_t numEnforceableChildren = 0;

  // Utility of MIN.
  auto minUtility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);

  // Run parsing of all the children in parallel.
  tbb::task_group minChildrenParsing;

  // Keep track of the bounds of start and end time.
  // std::pair<double, double> startTimeRange =
  //     std::make_pair(std::numeric_limits<Time>::max(), 0);
  // std::pair<double, double> endTimeRange =
  //     std::make_pair(std::numeric_limits<Time>::max(), 0);

  for (auto& child : children) {
    minChildrenParsing.run([&] {
      // Parse the Child.
      auto childParsedResult = child->parse(solverModel, availablePartitions,
                                            capacityConstraints, currentTime);

      if (childParsedResult->type != ParseResultType::EXPRESSION_UTILITY) {
        // If any of the children cannot provide a utility, the MIN expression
        // cannot be satisfied.
        return;
      }

      // Ensure that the MIN is satisfied only if this child is satisfied.
      if (childParsedResult->indicator.has_value()) {
        auto childIndicator = childParsedResult->indicator.value();
        if (childIndicator.isVariable()) {
          minIndicatorConstraint->addTerm(childIndicator);
          ++numEnforceableChildren;
        } else {
          auto indicatorValue = childIndicator.get<uint32_t>();
          if (indicatorValue == 1) {
            // This child is trivially satisifed, we shouldn't add its utility.
            TETRISCHED_DEBUG("Child " << child->getName() << " of MIN " << name
                                      << " is trivially satisfied.")
            // continue;
          } else {
            throw tetrisched::exceptions::ExpressionConstructionException(
                "Indicator needed from " + child->getName() +
                " for MIN, but was 0.");
          }
        }
      } else {
        throw tetrisched::exceptions::ExpressionConstructionException(
            "Indicator needed from " + child->getName() +
            " for MIN, but was not present!");
      }

      // Ensure that the MIN's start time is <= the start time of this child.
      ConstraintPtr minStartTimeConstraint = std::make_shared<Constraint>(
          name + "_min_start_time_constr_child_" + child->getName(),
          ConstraintType::CONSTR_GE, 0, 2);  // minStartTime < childStartTime
      if (childParsedResult->startTime.has_value()) {
        auto childStartTime = childParsedResult->startTime.value();
        if (childStartTime.isVariable()) {
          auto childStartTimeVariable = childStartTime.get<VariablePtr>();
          minStartTimeConstraint->addTerm(1, childStartTimeVariable);
          // if (auto lowerBound = childStartTimeVariable->getLowerBound();
          //     lowerBound.has_value()) {
          // auto lowerBoundValue = lowerBound.value();
          // std::cout << "Lower bound for " <<
          // childStartTimeVariable->getName()
          //           << " is " << lowerBoundValue << std::endl;
          // if (lowerBoundValue < startTimeRange.first) {
          //   startTimeRange.first = lowerBoundValue;
          // }
          // if (lowerBoundValue > startTimeRange.second) {
          //   startTimeRange.second = lowerBoundValue;
          // }
          // }
        } else {
          auto childStartTimeValue = childStartTime.get<Time>();
          minStartTimeConstraint->addTerm(childStartTimeValue);
          // if (childStartTimeValue < startTimeRange.first) {
          //   startTimeRange.first = childStartTimeValue;
          // }
          // if (childStartTimeValue > startTimeRange.second) {
          //   startTimeRange.second = childStartTimeValue;
          // }
        }
        minStartTimeConstraint->addTerm(-1, minStartTime);

        // Add the constraint to solver
        solverModel->addConstraint(std::move(minStartTimeConstraint));
      } else {
        throw tetrisched::exceptions::ExpressionConstructionException(
            "Start Time needed from " + child->getName() +
            " for MIN, but was not present!");
      }

      // Ensure that the MIN's end time is >= the end time of this child.
      ConstraintPtr minEndTimeConstraint = std::make_shared<Constraint>(
          name + "_min_end_time_constr_child_" + child->getName(),
          ConstraintType::CONSTR_LE, 0, 2);
      if (childParsedResult->endTime.has_value()) {
        auto childEndTime = childParsedResult->endTime.value();
        if (childEndTime.isVariable()) {
          auto childEndTimeVariable = childEndTime.get<VariablePtr>();
          minEndTimeConstraint->addTerm(1, childEndTimeVariable);
          // if (auto lowerBound = childEndTimeVariable->getLowerBound();
          //     lowerBound.has_value()) {
          //   std::cout << "Lower bound for " <<
          //   childEndTimeVariable->getName()
          //             << " is " << lowerBound.value() << std::endl;
          //   auto lowerBoundValue = lowerBound.value();
          //   if (lowerBoundValue < endTimeRange.first) {
          //     endTimeRange.first = lowerBoundValue;
          //   }
          //   if (lowerBoundValue > endTimeRange.second) {
          //     endTimeRange.second = lowerBoundValue;
          //   }
          // }
        } else {
          auto childEndTimeValue = childEndTime.get<Time>();
          minEndTimeConstraint->addTerm(childEndTimeValue);
          // if (childEndTimeValue < endTimeRange.first) {
          //   endTimeRange.first = childEndTimeValue;
          // }
          // if (childEndTimeValue > endTimeRange.second) {
          //   endTimeRange.second = childEndTimeValue;
          // }
        }
        minEndTimeConstraint->addTerm(-1, minEndTime);
        // Add the constraint to solver
        solverModel->addConstraint(std::move(minEndTimeConstraint));
      } else {
        throw tetrisched::exceptions::ExpressionConstructionException(
            "End Time needed from " + child->getName() +
            " for MIN, but was not present!");
      }

      if (childParsedResult->utility.has_value()) {
        (*minUtility) += *(childParsedResult->utility.value());
      } else {
        throw tetrisched::exceptions::ExpressionConstructionException(
            "Utility needed from " + child->getName() +
            " for MIN, but was not present!");
      }
    });
  }
  minChildrenParsing.wait();

  // Ensure that MIN satisfies all of its children.
  if (numEnforceableChildren == 0) {
    // All the children have been trivially satisfied.
    parsedResult->indicator = 1;
    minUtility->addTerm(1);
  } else {
    minIndicatorConstraint->addTerm(
        -1 * static_cast<TETRISCHED_ILP_TYPE>(numEnforceableChildren),
        minIndicator);
    solverModel->addConstraint(std::move(minIndicatorConstraint));
    parsedResult->indicator = minIndicator;

    // Set the lower and upper bounds for times.
    // std::cout << "Setting bounds for " << name << " to " <<
    // startTimeRange.first
    //           << " " << startTimeRange.second << " " << endTimeRange.first
    //           << " " << endTimeRange.second << std::endl;
    // if (startTimeRange.first != std::numeric_limits<Time>::max() &&
    //     startTimeRange.second != 0) {
    //   minStartTime->setLowerBound(startTimeRange.first);
    //   minStartTime->setUpperBound(startTimeRange.second);
    // }

    // if (endTimeRange.first != std::numeric_limits<Time>::max() &&
    //     endTimeRange.second != 0) {
    //   minEndTime->setLowerBound(endTimeRange.first);
    //   minEndTime->setUpperBound(endTimeRange.second);
    // }
  }

  // Check that all the children provided a utility.
  bool allChildrenProvidedUtilities = true;
  for (auto& child : children) {
    auto childParsedResult = child->getParsedResult();
    if (!childParsedResult.has_value()) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          "Child " + child->getName() + " of MIN " + name +
          " does not have a parsed result.");
    }
    auto childParsedResultValue = childParsedResult.value();
    if (childParsedResultValue->type != ParseResultType::EXPRESSION_UTILITY) {
      allChildrenProvidedUtilities = false;
      break;
    }
  }

  // Construct and return the ParsedResult.
  if (allChildrenProvidedUtilities) {
    parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
    parsedResult->utility = std::move(minUtility);
    parsedResult->startTime = minStartTime;
    parsedResult->endTime = minEndTime;
  } else {
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
  }
  return parsedResult;
}

/* Method definitions for MaxExpression */

MaxExpression::MaxExpression(std::string name)
    : Expression(name, ExpressionType::EXPR_MAX) {}

void MaxExpression::addChild(ExpressionPtr child) {
  if (child->getType() != ExpressionType::EXPR_CHOOSE &&
      child->getType() != ExpressionType::EXPR_MALLEABLE_CHOOSE &&
      child->getType() != ExpressionType::EXPR_WINDOWED_CHOOSE) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "MaxExpression can only have ChooseExpression children. Received "
        "expression of type " +
        child->getTypeString() + " instead.");
  }
  Expression::addChild(child);
}

ParseResultPtr MaxExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  std::lock_guard<std::mutex> lockGuard(expressionMutex);
  TETRISCHED_SCOPE_TIMER("MaxExpression::parse," + std::to_string(currentTime) +
                         "," + name + "," + id)
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // return the already parsed sub-tree from another parent
    // this assumes a sub-tree can have > 1 parent and enables
    // STRL DAG structures
    return parsedResult;
  }
  TETRISCHED_DEBUG("Parsing MaxExpression with name " << name << ".")
  // Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  auto numChildren = this->getNumChildren();
  if (numChildren == 0) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Number of children should be >=1 for MAX");
  }

  // Define the start time, end time and the utility bubbled up
  // by the MaxExpression.
  VariablePtr maxStartTime = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, name + "_max_start_time");
  solverModel->addVariable(maxStartTime);

  VariablePtr maxEndTime = std::make_shared<Variable>(VariableType::VAR_INTEGER,
                                                      name + "_max_end_time");
  solverModel->addVariable(maxEndTime);

  ObjectiveFunctionPtr maxObjectiveFunction =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);

  // Indicator of MAX operator
  VariablePtr maxIndicator = std::make_shared<Variable>(
      VariableType::VAR_INDICATOR, name + "_max_indicator");
  solverModel->addVariable(maxIndicator);

  // Constraint to allow only one sub-expression to have indicator = 1
  // Sum(child_indicator) - max_indicator <= 0
  ConstraintPtr maxChildSubexprConstraint = std::make_shared<Constraint>(
      name + "_max_child_subexpr_constr", ConstraintType::CONSTR_EQ, 0,
      numChildren + 1);

  // Constraint to set startTime of MAX
  // Sum(Indicator * child_start) >= maxStartTime
  ConstraintPtr maxStartTimeConstraint = std::make_shared<Constraint>(
      name + "_max_start_time_constr", ConstraintType::CONSTR_GE, 0,
      numChildren + 3);

  // Constraint to set endTime of MAX
  // Sum(Indicator * child_end) <= maxEndTime
  ConstraintPtr maxEndTimeConstraint = std::make_shared<Constraint>(
      name + "_max_end_time_constr", ConstraintType::CONSTR_LE, 0,
      numChildren + 1);

  // Parse each of the children and constrain the MaxExpression's start time,
  // end time and utility as a function of the children's start time, end time
  // and utility.
  TimeRange startTimeBounds =
      std::make_pair(std::numeric_limits<Time>::max(), 0);
  TimeRange endTimeBounds = std::make_pair(std::numeric_limits<Time>::max(), 0);
  bool anyChildrenWithUtilities = false;
  for (size_t i = 0; i < numChildren; i++) {
    auto childParsedResult = children[i]->parse(
        solverModel, availablePartitions, capacityConstraints, currentTime);

    if (childParsedResult->type != ParseResultType::EXPRESSION_UTILITY) {
      TETRISCHED_DEBUG(name + " child-" + std::to_string(i) +
                       " is not an Expression with utility. Skipping.");
      continue;
    }
    anyChildrenWithUtilities = true;

    // Check that the MaxExpression's childrens were specified correctly.
    if (!childParsedResult->startTime ||
        (childParsedResult->startTime.value().isVariable() &&
         children[i]->getType() != ExpressionType::EXPR_WINDOWED_CHOOSE)) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          name + " child-" + std::to_string(i) + " (" + children[i]->getName() +
          ") of type " + children[i]->getTypeString() +
          " must have a non-variable start time.");
    }
    if (!childParsedResult->endTime ||
        (childParsedResult->endTime.value().isVariable() &&
         children[i]->getType() != ExpressionType::EXPR_WINDOWED_CHOOSE)) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          name + " child-" + std::to_string(i) + " (" + children[i]->getName() +
          ") of type " + children[i]->getTypeString() +
          " must have a non-variable end time.");
    }
    if (!childParsedResult->indicator) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          name + " child-" + std::to_string(i) + " (" + children[i]->getName() +
          ") must have an indicator.");
    }
    if (!childParsedResult->utility) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          name + " child-" + std::to_string(i) + " (" + children[i]->getName() +
          ") must have a utility.");
    }

    // Retrieve the indicator for the child.
    auto childIndicator = childParsedResult->indicator.value();

    // Enforce that only one of the children is satisfied.
    maxChildSubexprConstraint->addTerm(childIndicator);

    // Set the start time of the Max expression.
    if (childParsedResult->startTime.value().isVariable()) {
      // This variable comes from subexpressions that must already have their
      // start times modulated by indicators. We can just merge them instead
      // of using the child indicator again.
      auto childStartTime =
          childParsedResult->startTime.value().get<VariablePtr>();
      maxStartTimeConstraint->addTerm(childStartTime);

      // The bounds are set by the child's bounds.
      auto childBounds = children[i]->getTimeBounds().startTimeRange;
      if (childBounds.first < startTimeBounds.first) {
        startTimeBounds.first = childBounds.first;
      }
      if (childBounds.second > startTimeBounds.second) {
        startTimeBounds.second = childBounds.second;
      }
    } else {
      // Add the start time of the child to the MaxExpression's start time.
      auto childStartTime = childParsedResult->startTime.value().get<Time>();
      maxStartTimeConstraint->addTerm(childStartTime, childIndicator);

      // Set the bounds of the start time of the MaxExpression.
      if (childStartTime < startTimeBounds.first) {
        startTimeBounds.first = childStartTime;
      }
      if (childStartTime > startTimeBounds.second) {
        startTimeBounds.second = childStartTime;
      }
    }

    // Set the end time of the Max expression.
    if (childParsedResult->endTime.value().isVariable()) {
      // This variable comes from subexpressions that must already have their
      // end times modulated by indicators. We can just merge them instead
      // of using the child indicator again.
      auto childEndTime = childParsedResult->endTime.value().get<VariablePtr>();
      maxEndTimeConstraint->addTerm(childEndTime);

      // The bounds are set by the child's bounds.
      auto childBounds = children[i]->getTimeBounds().endTimeRange;
      if (childBounds.first < endTimeBounds.first) {
        endTimeBounds.first = childBounds.first;
      }
      if (childBounds.second > endTimeBounds.second) {
        endTimeBounds.second = childBounds.second;
      }
    } else {
      // Add the end time of the child to the MaxExpression's end time.
      auto childEndTime = childParsedResult->endTime.value().get<Time>();
      maxEndTimeConstraint->addTerm(childEndTime, childIndicator);

      // Set the bounds of the end time of the MaxExpression.
      if (childEndTime < endTimeBounds.first) {
        endTimeBounds.first = childEndTime;
      }
      if (childEndTime > endTimeBounds.second) {
        endTimeBounds.second = childEndTime;
      }
    }

    // Add the utility of the child to the MaxExpression's utility.
    auto childUtility = childParsedResult->utility.value();
    (*maxObjectiveFunction) += (*childUtility);
  }

  if (!anyChildrenWithUtilities) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        name + " must have at least one child with utility.");
  }

  // Constrain the MaxExpression's start time to be less than or equal to the
  // start time of the chosen child. However, if the MaxExpression is not
  // placed i.e., the sum of its childrens indicator is 0, then we have to let
  // the start time be free. We do this by adding the maximum start time of the
  // children as a constant that is activated if the children cannot be
  // placed.
  maxStartTimeConstraint->addTerm(startTimeBounds.first);
  maxStartTimeConstraint->addTerm(
      -1 * static_cast<TETRISCHED_ILP_TYPE>(startTimeBounds.first),
      maxIndicator);
  maxStartTimeConstraint->addTerm(-1, maxStartTime);

  // Set the bounds on the start and end time.
  maxStartTime->setLowerBound(
      -1 * static_cast<TETRISCHED_ILP_TYPE>(startTimeBounds.first));
  maxStartTime->setUpperBound(
      static_cast<TETRISCHED_ILP_TYPE>(startTimeBounds.second));
  maxEndTime->setLowerBound(0);
  maxEndTime->setUpperBound(
      static_cast<TETRISCHED_ILP_TYPE>(endTimeBounds.second));

  // Constrain the MaxExpression's end time to be greater than or equal to the
  // end time of the chosen child.
  maxEndTimeConstraint->addTerm(-1, maxEndTime);

  // Set the indicator for the MaxExpression to be equal to the sum of the
  // indicators for the children.
  maxChildSubexprConstraint->addTerm(-1, maxIndicator);

  // Add the constraints for the start time, end time and the indicator.
  solverModel->addConstraint(std::move(maxStartTimeConstraint));
  solverModel->addConstraint(std::move(maxEndTimeConstraint));
  solverModel->addConstraint(std::move(maxChildSubexprConstraint));

  // Construct the ParsedResult for the MaxExpression.
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = std::move(maxStartTime);
  parsedResult->endTime = std::move(maxEndTime);
  parsedResult->utility = std::move(maxObjectiveFunction);
  parsedResult->indicator = std::move(maxIndicator);
  TETRISCHED_DEBUG("Finished parsing MaxExpression with name " << name << ".")
  return parsedResult;
}

/* Method definitions for ScaleExpression */

ScaleExpression::ScaleExpression(std::string name,
                                 TETRISCHED_ILP_TYPE scaleFactor)
    : Expression(name, ExpressionType::EXPR_SCALE), scaleFactor(scaleFactor) {}

void ScaleExpression::addChild(ExpressionPtr child) {
  if (children.size() == 1) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "ScaleExpression can only have one child.");
  }
  Expression::addChild(child);
}

ParseResultPtr ScaleExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMapPtr capacityConstraints, Time currentTime) {
  std::lock_guard<std::mutex> lockGuard(expressionMutex);
  TETRISCHED_SCOPE_TIMER("ScaleExpression::parse," +
                         std::to_string(currentTime) + "," + name + "," + id)
  // Sanity check the children.
  if (children.size() != 1) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "ScaleExpression must have one child.");
  }

  // Parse the child expression.
  auto childParseResult = children[0]->parse(solverModel, availablePartitions,
                                             capacityConstraints, currentTime);
  if (childParseResult->type == ParseResultType::EXPRESSION_UTILITY) {
    parsedResult = std::make_shared<ParseResult>();
    parsedResult->type = ParseResultType::EXPRESSION_UTILITY;

    if (!childParseResult->utility) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          "ScaleExpression applied to a child that does not have any "
          "utility.");
    }
    TETRISCHED_DEBUG("The child utility is "
                     << childParseResult->utility.value()->toString());
    parsedResult->utility = std::make_shared<ObjectiveFunction>(
        (*childParseResult->utility.value()) * scaleFactor);
    TETRISCHED_DEBUG("The scale utility is "
                     << parsedResult->utility.value()->toString());

    if (childParseResult->startTime) {
      parsedResult->startTime.emplace(childParseResult->startTime.value());
    }
    if (childParseResult->endTime) {
      parsedResult->endTime.emplace(childParseResult->endTime.value());
    }
    if (childParseResult->indicator) {
      parsedResult->indicator.emplace(childParseResult->indicator.value());
    }
    return parsedResult;
  } else {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "ScaleExpression applied to a child that does not have any utility.");
  }
}
}  // namespace tetrisched
