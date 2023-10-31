#include "tetrisched/Expression.hpp"

#include <algorithm>
#include <limits>

namespace tetrisched {

/* Method definitions for Placement */
Placement::Placement(std::string taskName, Time startTime, Time endTime)
    : taskName(taskName),
      startTime(startTime),
      endTime(endTime),
      placed(true) {}

Placement::Placement(std::string taskName)
    : taskName(taskName), startTime(std::nullopt), placed(false) {}

bool Placement::isPlaced() const { return placed; }

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

std::optional<Time> Placement::getStartTime() const { return startTime; }

std::optional<Time> Placement::getEndTime() const { return endTime; }

const std::unordered_map<uint32_t, std::set<std::pair<Time, uint32_t>>>&
Placement::getPartitionAllocations() const {
  return partitionToResourceAllocations;
}

/* Method definitions for CapacityConstraintMap */

CapacityConstraintMap::CapacityConstraintMap(Time granularity)
    : granularity(granularity) {}

CapacityConstraintMap::CapacityConstraintMap() : granularity(1) {}

void CapacityConstraintMap::registerUsageAtTime(const Partition& partition,
                                                Time time,
                                                VariablePtr variable) {
  // Get or insert the Constraint corresponding to this partition and time.
  auto mapKey = std::make_pair(partition.getPartitionId(), time);
  if (capacityConstraints.find(mapKey) == capacityConstraints.end()) {
    capacityConstraints[mapKey] = std::make_shared<Constraint>(
        "CapacityConstraint_" + partition.getPartitionName() + "_at_" +
            std::to_string(time),
        ConstraintType::CONSTR_LE, partition.getQuantity());
  }

  // Add the variable to the Constraint.
  capacityConstraints[mapKey]->addTerm(variable);
}

void CapacityConstraintMap::registerUsageAtTime(const Partition& partition,
                                                Time time, uint32_t usage) {
  if (usage == 0) {
    // No usage was registered. We don't need to add anything.
    return;
  }
  // Get or insert the Constraint corresponding to this partition and time.
  auto mapKey = std::make_pair(partition.getPartitionId(), time);
  if (capacityConstraints.find(mapKey) == capacityConstraints.end()) {
    capacityConstraints[mapKey] = std::make_shared<Constraint>(
        "CapacityConstraint_" + partition.getPartitionName() + "_at_" +
            std::to_string(time),
        ConstraintType::CONSTR_LE, partition.getQuantity());
  }

  // Add the variable to the Constraint.
  capacityConstraints[mapKey]->addTerm(usage);
}

void CapacityConstraintMap::registerUsageForDuration(
    const Partition& partition, Time startTime, Time duration,
    VariablePtr variable, std::optional<Time> granularity) {
  Time _granularity = granularity.value_or(this->granularity);
  for (Time time = startTime; time < startTime + duration;
       time += _granularity) {
    registerUsageAtTime(partition, time, variable);
  }
}

void CapacityConstraintMap::registerUsageForDuration(
    const Partition& partition, Time startTime, Time duration, uint32_t usage,
    std::optional<Time> granularity) {
  Time _granularity = granularity.value_or(this->granularity);
  for (Time time = startTime; time < startTime + duration;
       time += _granularity) {
    registerUsageAtTime(partition, time, usage);
  }
}

void CapacityConstraintMap::translate(SolverModelPtr solverModel) {
  // Add the constraints to the SolverModel.
  for (auto& [mapKey, constraint] : capacityConstraints) {
    if (!constraint->isTriviallySatisfiable()) {
      // COMMENT (Sukrit): We can try to see if adding Lazy constraints
      // helps ever. In my initial analysis, this makes the presolve and
      // root relaxation less efficient making the overall solver time
      // higher. Maybe for too many of the CapacityConstraintMap constraints,
      // this will help.
      // constraint->addAttribute(ConstraintAttribute::LAZY_CONSTRAINT);
      solverModel->addConstraint(std::move(constraint));
    }
  }

  // Clear the map now that the constraints have been drained.
  capacityConstraints.clear();
}

size_t CapacityConstraintMap::size() const {
  return capacityConstraints.size();
}

/* Method definitions for Expression */

Expression::Expression(std::string name, ExpressionType type)
    : name(name), type(type) {}

std::string Expression::getName() const { return name; }

size_t Expression::getNumChildren() const { return children.size(); }

std::vector<ExpressionPtr> Expression::getChildren() const { return children; }

void Expression::addChild(ExpressionPtr child) {
  child->addParent(shared_from_this());
  children.push_back(child);
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
    default:
      return "UnknownExpression";
  }
}

void Expression::addParent(ExpressionPtr parent) { parents.push_back(parent); }

size_t Expression::getNumParents() const { return parents.size(); }

std::vector<ExpressionPtr> Expression::getParents() const {
  std::vector<ExpressionPtr> return_parents;
  for (int i = 0; i < parents.size(); i++) {
    return_parents.push_back(parents[i].lock());
  }
  return return_parents;
}

SolutionResultPtr Expression::populateResults(SolverModelPtr solverModel) {
  // Check that the Expression was parsed before.
  if (!parsedResult) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression " + name + " was not parsed before solve.");
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
    case ParseResultType::EXPRESSION_PRUNE:
      solution->type = SolutionResultType::EXPRESSION_PRUNE;
      return solution;
    case ParseResultType::EXPRESSION_NO_UTILITY:
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
    auto childExpressionSolution = childExpression->getSolution().value();
    if (childExpressionSolution->utility == 0) {
      // This child was not satisfied. Skip it.
      continue;
    }

    // The child was satisfied, merge its Placement objects with our own.
    for (auto& [taskName, placement] : childExpressionSolution->placements) {
      solution->placements[taskName] = placement;
    }
  }

  return solution;
}

std::optional<SolutionResultPtr> Expression::getSolution() const {
  if (!solution) {
    return std::nullopt;
  }
  return solution;
}

/* Method definitions for ChooseExpression */

ChooseExpression::ChooseExpression(std::string taskName,
                                   Partitions resourcePartitions,
                                   uint32_t numRequiredMachines, Time startTime,
                                   Time duration)
    : Expression(taskName, ExpressionType::EXPR_CHOOSE),
      resourcePartitions(resourcePartitions),
      numRequiredMachines(numRequiredMachines),
      startTime(startTime),
      duration(duration),
      endTime(startTime + duration) {}

void ChooseExpression::addChild(ExpressionPtr child) {
  throw tetrisched::exceptions::ExpressionConstructionException(
      "ChooseExpression cannot have a child.");
}

ParseResultPtr ChooseExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMap& capacityConstraints, Time currentTime) {
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
    TETRISCHED_DEBUG("Pruning Choose expression for "
                     << name << " to be placed starting at time " << startTime
                     << " and ending at " << endTime
                     << " because it is in the past.");
    parsedResult->type = ParseResultType::EXPRESSION_PRUNE;
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
      name + "_placed_at_" + std::to_string(startTime));
  solverModel->addVariable(isSatisfiedVar);

  ConstraintPtr fulfillsDemandConstraint = std::make_shared<Constraint>(
      name + "_fulfills_demand_at_" + std::to_string(startTime),
      ConstraintType::CONSTR_EQ, 0);
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
    capacityConstraints.registerUsageForDuration(
        *partition, startTime, duration, allocationVar, std::nullopt);
  }
  // Ensure that if the Choose expression is satisfied, it fulfills the
  // demand for this expression. Pass the constraint to the model.
  fulfillsDemandConstraint->addTerm(
      -1 * static_cast<TETRISCHED_ILP_TYPE>(numRequiredMachines),
      isSatisfiedVar);
  solverModel->addConstraint(std::move(fulfillsDemandConstraint));

  // Construct the Utility function for this Choose expression.
  auto utility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  utility->addTerm(1, isSatisfiedVar);

  // Construct the return value.
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = startTime;
  parsedResult->endTime = endTime;
  parsedResult->indicator = isSatisfiedVar;
  parsedResult->utility = std::move(utility);
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
    if (variableValue == 0) {
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

/* Method definitions for GeneralizedChoose */
MalleableChooseExpression::MalleableChooseExpression(
    std::string taskName, Partitions resourcePartitions,
    uint32_t resourceTimeSlots, Time startTime, Time endTime, Time granularity)
    : Expression(taskName, ExpressionType::EXPR_MALLEABLE_CHOOSE),
      resourcePartitions(resourcePartitions),
      resourceTimeSlots(resourceTimeSlots),
      startTime(startTime),
      endTime(endTime),
      granularity(granularity),
      partitionVariables() {}

void MalleableChooseExpression::addChild(ExpressionPtr child) {
  throw tetrisched::exceptions::ExpressionConstructionException(
      "MalleableChooseExpression cannot have a child.");
}

ParseResultPtr MalleableChooseExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMap& capacityConstraints, Time currentTime) {
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // Return the alread parsed sub-tree.
    return parsedResult;
  }

  // Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  if (currentTime > startTime) {
    TETRISCHED_DEBUG("Pruning MalleableChooseExpression "
                     << name << " to be placed starting at time " << startTime
                     << " and ending at " << endTime
                     << " because it is in the past.");
    parsedResult->type = ParseResultType::EXPRESSION_PRUNE;
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
      ConstraintType::CONSTR_EQ, 0);

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
        capacityConstraints.registerUsageForDuration(
            *partition, time, granularity, allocationAtTime, std::nullopt);
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
          ConstraintType::CONSTR_LE, 0);
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
          ConstraintType::CONSTR_LE, 0);
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
      ConstraintType::CONSTR_LE, 1);
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
        ConstraintType::CONSTR_LE, 0);
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
        ConstraintType::CONSTR_LE, 0);
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
      name + "_start_time_constraint", ConstraintType::CONSTR_EQ, 0);
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
      1);
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
        ConstraintType::CONSTR_LE, 0);
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
        ConstraintType::CONSTR_LE, 0);
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
      name + "_end_time_constraint", ConstraintType::CONSTR_EQ, 0);
  for (auto& [time, phaseShiftIndicator] :
       timeToPhaseShiftIndicatorForEndTime) {
    endTimeConstraint->addTerm(time, phaseShiftIndicator);
  }
  endTimeConstraint->addTerm(-1, endTimeVariable);
  solverModel->addConstraint(std::move(endTimeConstraint));

  // Construct the Utility function for this Choose expression.
  auto utility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  utility->addTerm(1, isSatisfiedVar);

  // Construct the return value.
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = startTimeVariable;
  parsedResult->endTime = endTimeVariable;
  parsedResult->indicator = isSatisfiedVar;
  parsedResult->utility = std::move(utility);
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
    if (variableValue == 0) {
      // This partition was not used.
      continue;
    }
    auto& [partitionId, time] = partitionPair;
    placement->addPartitionAllocation(partitionId, time, variableValue.value());
  }
  solution->placements[name] = std::move(placement);
  return solution;
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
      endTime(startTime + duration) {}

void AllocationExpression::addChild(ExpressionPtr child) {
  throw tetrisched::exceptions::ExpressionConstructionException(
      "AllocationExpression cannot have a child.");
}

ParseResultPtr AllocationExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMap& capacityConstraints, Time currentTime) {
  // Check that the Expression was parsed before.
  if (parsedResult != nullptr) {
    // Return the already parsed sub-tree.
    return parsedResult;
  }

  // Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = startTime;
  parsedResult->endTime = endTime;
  parsedResult->indicator = 1;
  parsedResult->utility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  (parsedResult->utility).value()->addTerm(1);
  for (const auto& [partition, allocation] : allocatedResources) {
    capacityConstraints.registerUsageForDuration(
        *partition, startTime, duration, static_cast<uint32_t>(allocation),
        std::nullopt);
  }
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

/* Method definitions for ObjectiveExpression */

ObjectiveExpression::ObjectiveExpression(std::string name)
    : Expression(name, ExpressionType::EXPR_OBJECTIVE) {}

ParseResultPtr ObjectiveExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMap& capacityConstraints, Time currentTime) {
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // return the already parsed sub-tree from another parent
    // this assumes a sub-tree can have > 1 parent and enables
    // STRL DAG structures
    return parsedResult;
  }
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
      (*utility) += *(result->utility.value());
    }
  }

  // All the children have been parsed. Finalize the CapacityConstraintMap.
  capacityConstraints.translate(solverModel);

  // Construct the parsed result.
  parsedResult->utility = std::make_shared<ObjectiveFunction>(*utility);
  parsedResult->startTime = std::numeric_limits<Time>::min();
  parsedResult->endTime = std::numeric_limits<Time>::max();

  // Add the utility to the SolverModel.
  solverModel->setObjectiveFunction(std::move(utility));

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
    auto childExpressionSolution = childExpression->getSolution().value();

    // If the child expression was not satisfied, skip it.
    if (childExpressionSolution->utility == 0) {
      continue;
    }

    // If the child has a smaller start time, use it.
    if (childExpressionSolution->startTime.value() < minStartTime) {
      minStartTime = childExpressionSolution->startTime.value();
    }

    // If the child has a larger end time, use it.
    if (childExpressionSolution->endTime.value() > maxEndTime) {
      maxEndTime = childExpressionSolution->endTime.value();
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
    CapacityConstraintMap& capacityConstraints, Time currentTime) {
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

  if (firstChildResult->type != ParseResultType::EXPRESSION_UTILITY ||
      secondChildResult->type != ParseResultType::EXPRESSION_UTILITY) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "LessThanExpression must have two children that are being "
        "evaluated.");
  }

  // Generate the result of parsing the expression.
  parsedResult = std::make_shared<ParseResult>();
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

  // Add a constraint that the first child must occur before the second.
  auto happensBeforeConstraintName = name + "_happens_before_constraint";
  ConstraintPtr happensBeforeConstraint = std::make_shared<Constraint>(
      happensBeforeConstraintName, ConstraintType::CONSTR_LE, -1);
  happensBeforeConstraint->addTerm(firstChildResult->endTime.value());
  happensBeforeConstraint->addTerm(-1, secondChildResult->startTime.value());
  solverModel->addConstraint(std::move(happensBeforeConstraint));
  TETRISCHED_DEBUG("Finished adding constraint "
                   << happensBeforeConstraintName
                   << " to enforce ordering in LessThanExpression with name "
                   << name << ".")

  // Construct a utility function that is the minimum of the two utilities.
  // Maximizing this utility will force the solver to place both of the
  // subexpressions.
  VariablePtr utilityVar =
      std::make_shared<Variable>(VariableType::VAR_INTEGER, name + "_utility");
  solverModel->addVariable(utilityVar);
  if (!firstChildResult->utility || !secondChildResult->utility) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "LessThanExpression must have children with utilities.");
  }

  ConstraintPtr constrainUtilityLessThanFirstChild =
      firstChildResult->utility.value()->toConstraint(
          name + "_utility_less_than_first_child", ConstraintType::CONSTR_GE,
          0);
  constrainUtilityLessThanFirstChild->addTerm(-1, utilityVar);
  solverModel->addConstraint(std::move(constrainUtilityLessThanFirstChild));

  ConstraintPtr constrainUtilityLessThanSecondChild =
      secondChildResult->utility.value()->toConstraint(
          name + "_utility_less_than_second_child", ConstraintType::CONSTR_GE,
          0);
  constrainUtilityLessThanSecondChild->addTerm(-1, utilityVar);
  solverModel->addConstraint(std::move(constrainUtilityLessThanSecondChild));

  // Convert the utility variable to a utility function.
  parsedResult->utility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  parsedResult->utility.value()->addTerm(1, utilityVar);
  TETRISCHED_DEBUG("LessThanExpression with name "
                   << name << " has utility " << utilityVar->getName() << ".");

  // Return the result.
  return parsedResult;
}

/* Method definitions for MinExpression */

MinExpression::MinExpression(std::string name)
    : Expression(name, ExpressionType::EXPR_MIN) {}

ParseResultPtr MinExpression::parse(SolverModelPtr solverModel,
                                    Partitions availablePartitions,
                                    CapacityConstraintMap& capacityConstraints,
                                    Time currentTime) {
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // return the already parsed sub-tree from another parent
    // this assumes a sub-tree can have > 1 parent and enables
    // STRL DAG structures
    return parsedResult;
  }
  /// Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  auto numChildren = this->getNumChildren();
  if (numChildren == 0) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Number of children should be >=1 for MIN");
  }
  // start time of MIN
  VariablePtr minStartTime = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, name + "_min_start_time");
  solverModel->addVariable(minStartTime);

  // end time of MIN
  VariablePtr minEndTime = std::make_shared<Variable>(VariableType::VAR_INTEGER,
                                                      name + "_min_end_time");
  solverModel->addVariable(minEndTime);

  // Utility of MIN operator
  auto minUtility =
      std::make_shared<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  VariablePtr minUtilityVariable = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, name + "_min_utility_variable");
  solverModel->addVariable(minUtilityVariable);

  for (int i = 0; i < numChildren; i++) {
    auto childParsedResult = children[i]->parse(
        solverModel, availablePartitions, capacityConstraints, currentTime);
    ConstraintPtr minStartTimeConstraint = std::make_shared<Constraint>(
        name + "_min_start_time_constr_child_" + std::to_string(i),
        ConstraintType::CONSTR_GE, 0);  // minStartTime < childStartTime
    if (childParsedResult->startTime.has_value()) {
      auto childStartTime = childParsedResult->startTime.value();
      if (childStartTime.isVariable()) {
        minStartTimeConstraint->addTerm(1, childStartTime.get<VariablePtr>());
      } else {
        minStartTimeConstraint->addTerm(childStartTime.get<Time>());
      }
      minStartTimeConstraint->addTerm(-1, minStartTime);

      // Add the constraint to solver
      solverModel->addConstraint(std::move(minStartTimeConstraint));
    } else {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "Start Time needed from child-" + std::to_string(i) +
          " for MIN. But not present!");
    }
    // constraint of end time: childEndTime <= minEndTime
    ConstraintPtr minEndTimeConstraint = std::make_shared<Constraint>(
        name + "_min_end_time_constr_child_" + std::to_string(i),
        ConstraintType::CONSTR_LE, 0);
    if (childParsedResult->endTime.has_value()) {
      auto childEndTime = childParsedResult->endTime.value();
      if (childEndTime.isVariable()) {
        minEndTimeConstraint->addTerm(1, childEndTime.get<VariablePtr>());
      } else {
        minEndTimeConstraint->addTerm(childEndTime.get<Time>());
      }
      minEndTimeConstraint->addTerm(-1, minEndTime);
      // Add the constraint to solver
      solverModel->addConstraint(std::move(minEndTimeConstraint));
    } else {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "End Time needed from child-" + std::to_string(i) +
          " for MIN. But not present!");
    }

    if (childParsedResult->utility.has_value()) {
      // child_utility - minUVar >= 0
      auto childUtilityConstr =
          childParsedResult->utility.value()->toConstraint(
              name + "_min_utility_constraint_child_" + std::to_string(i),
              ConstraintType::CONSTR_GE, 0);
      childUtilityConstr->addTerm(-1, minUtilityVariable);
      solverModel->addConstraint(std::move(childUtilityConstr));
    } else {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "Utility needed from child-" + std::to_string(i) +
          " for MIN. But not present!");
    }
  }
  // MinU = Max(MinUVar)
  minUtility->addTerm(1, minUtilityVariable);

  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = minStartTime;
  parsedResult->endTime = minEndTime;
  parsedResult->utility = std::move(minUtility);
  return parsedResult;
}

/* Method definitions for MaxExpression */

MaxExpression::MaxExpression(std::string name)
    : Expression(name, ExpressionType::EXPR_MAX) {}

void MaxExpression::addChild(ExpressionPtr child) {
  if (child->getType() != ExpressionType::EXPR_CHOOSE) {
    throw tetrisched::exceptions::ExpressionConstructionException(
        "MaxExpression can only have ChooseExpression children.");
  }
  Expression::addChild(child);
}

ParseResultPtr MaxExpression::parse(SolverModelPtr solverModel,
                                    Partitions availablePartitions,
                                    CapacityConstraintMap& capacityConstraints,
                                    Time currentTime) {
  // Check that the Expression was parsed before
  if (parsedResult != nullptr) {
    // return the already parsed sub-tree from another parent
    // this assumes a sub-tree can have > 1 parent and enables
    // STRL DAG structures
    return parsedResult;
  }
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
      name + "_max_child_subexpr_constr", ConstraintType::CONSTR_LE, 0);

  // Constraint to set startTime of MAX
  // Sum(Indicator * child_start) >= maxStartTime
  ConstraintPtr maxStartTimeConstraint = std::make_shared<Constraint>(
      name + "_max_start_time_constr", ConstraintType::CONSTR_GE, 0);

  // Constraint to set endTime of MAX
  // Sum(Indicator * child_end) <= maxEndTime
  ConstraintPtr maxEndTimeConstraint = std::make_shared<Constraint>(
      name + "_max_end_time_constr", ConstraintType::CONSTR_LE, 0);

  // Parse each of the children and constrain the MaxExpression's start time,
  // end time and utility as a function of the children's start time, end time
  // and utility.
  for (int i = 0; i < numChildren; i++) {
    auto childParsedResult = children[i]->parse(
        solverModel, availablePartitions, capacityConstraints, currentTime);

    if (childParsedResult->type != ParseResultType::EXPRESSION_UTILITY) {
      TETRISCHED_DEBUG(name + " child-" + std::to_string(i) +
                       " is not an Expression with utility. Skipping.");
      continue;
    }

    // Check that the MaxExpression's childrens were specified correctly.
    if (!childParsedResult->startTime ||
        childParsedResult->startTime.value().isVariable()) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          name + " child-" + std::to_string(i) + " (" + children[i]->getName() +
          ") must have a non-variable start time.");
    }
    if (!childParsedResult->endTime ||
        childParsedResult->endTime.value().isVariable()) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          name + " child-" + std::to_string(i) + " (" + children[i]->getName() +
          ") must have a non-variable end time.");
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

    auto childStartTime = childParsedResult->startTime.value().get<Time>();
    auto childEndTime = childParsedResult->endTime.value().get<Time>();
    auto childIndicator = childParsedResult->indicator.value();
    auto childUtility = childParsedResult->utility.value();

    // Enforce that only one of the children is satisfied.
    maxChildSubexprConstraint->addTerm(childIndicator);

    // Add the start time of the child to the MaxExpression's start time.
    maxStartTimeConstraint->addTerm(childStartTime, childIndicator);

    // Add the end time of the child to the MaxExpression's end time.
    maxEndTimeConstraint->addTerm(childEndTime, childIndicator);

    // Add the utility of the child to the MaxExpression's utility.
    (*maxObjectiveFunction) += (*childUtility);
  }

  // Constrain the MaxExpression's start time to be less than or equal to the
  // start time of the chosen child.
  maxStartTimeConstraint->addTerm(-1, maxStartTime);

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
    CapacityConstraintMap& capacityConstraints, Time currentTime) {
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
