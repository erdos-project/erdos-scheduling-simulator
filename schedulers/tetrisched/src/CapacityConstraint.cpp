#include "tetrisched/CapacityConstraint.hpp"

namespace tetrisched {

/* Method definitions for PartitionTimePairHasher */

size_t PartitionTimePairHasher::operator()(
    const std::pair<uint32_t, Time>& pair) const {
  auto partitionIdHash = std::hash<uint32_t>()(pair.first);
  auto timeHash = std::hash<Time>()(pair.second);
  if (partitionIdHash != timeHash) {
    return partitionIdHash ^ timeHash;
  }
  return partitionIdHash;
}

/* Method definitions for CapacityConstraint */
CapacityConstraint::CapacityConstraint(const Partition& partition,
                                       Time constraintTime, Time granularity)
    : name("CapacityConstraint_" + partition.getPartitionName() + "_at_" +
           std::to_string(constraintTime)),
      quantity(partition.getQuantity()),
      granularity(granularity),
      usageUpperBound(0),
      durationUpperBound(0),
      overlapVariable(std::make_shared<Variable>(VariableType::VAR_INDICATOR,
                                                 name + "_overlap")),
      upperBoundConstraint(std::make_shared<Constraint>(
          name + "_upper_bound", ConstraintType::CONSTR_LE, quantity)),
      lowerBoundConstraint(std::make_shared<Constraint>(
          name + "_lower_bound", ConstraintType::CONSTR_GE, quantity + 0.5)),
      capacityConstraint(std::make_shared<Constraint>(
          name, ConstraintType::CONSTR_LE, quantity)) {}

void CapacityConstraint::registerUsage(const ExpressionPtr expression,
                                       const IndicatorT usageIndicator,
                                       const PartitionUsageT usageVariable) {
  if (!usageVariable.isVariable() && usageVariable.get<uint32_t>() == 0) {
    // No usage was registered. We don't need to add anything.
    return;
  }
  capacityConstraint->addTerm(usageVariable);
  usageVector.emplace_back(expression, usageVariable);
}

void CapacityConstraint::translate(SolverModelPtr solverModel) {
  solverModel->addConstraint(capacityConstraint);
  // if (!capacityConstraint->isTriviallySatisfiable()) {
  //   // COMMENT (Sukrit): We can try to see if adding Lazy constraints
  //   // helps ever. In my initial analysis, this makes the presolve and
  //   // root relaxation less efficient making the overall solver time
  //   // higher. Maybe for too many of the CapacityConstraintMap constraints,
  //   // this will help.
  //   //
  //   capacityConstraint->addAttribute(ConstraintAttribute::LAZY_CONSTRAINT);
  //   solverModel->addConstraint(capacityConstraint);
  // }
}

void CapacityConstraint::deactivate() { capacityConstraint->deactivate(); }

uint32_t CapacityConstraint::getQuantity() const { return quantity; }

std::string CapacityConstraint::getName() const { return name; }

/* Method definitions for CapacityConstraintMap */

CapacityConstraintMap::CapacityConstraintMap(Time granularity)
    : granularity(granularity) {}

CapacityConstraintMap::CapacityConstraintMap() : granularity(1) {}

void CapacityConstraintMap::registerUsageAtTime(
    const ExpressionPtr expression, const Partition& partition, Time time,
    const IndicatorT usageIndicator, const PartitionUsageT usageVariable) {
  if (!usageIndicator.isVariable() && usageIndicator.get<uint32_t>() == 0) {
    // No usage was registered. We don't need to add anything.
    return;
  }
  // Get or insert the Constraint corresponding to this partition and time.
  auto mapKey = std::make_pair(partition.getPartitionId(), time);
  if (capacityConstraints.find(mapKey) == capacityConstraints.end()) {
    capacityConstraints[mapKey] =
        std::make_shared<CapacityConstraint>(partition, time, granularity);
  }

  // Add the variable to the Constraint.
  capacityConstraints[mapKey]->registerUsage(expression, usageIndicator,
                                             usageVariable);
}

void CapacityConstraintMap::registerUsageForDuration(
    const ExpressionPtr expression, const Partition& partition, Time startTime,
    Time duration, const IndicatorT usageIndicator,
    const PartitionUsageT variable, std::optional<Time> granularity) {
  Time _granularity = granularity.value_or(this->granularity);
  for (Time time = startTime; time < startTime + duration;
       time += _granularity) {
    registerUsageAtTime(expression, partition, time, usageIndicator, variable);
  }
}

void CapacityConstraintMap::translate(SolverModelPtr solverModel) {
  // Add the constraints to the SolverModel.
  for (auto& [mapKey, constraint] : capacityConstraints) {
    constraint->translate(solverModel);
  }

  // Clear the map now that the constraints have been drained.
  // capacityConstraints.clear();
}

size_t CapacityConstraintMap::size() const {
  return capacityConstraints.size();
}
}  // namespace tetrisched
