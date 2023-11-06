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
CapacityConstraint::CapacityConstraint(const Partition& partition, Time time)
    : name("CapacityConstraint_" + partition.getPartitionName() + "_at_" +
           std::to_string(time)),
      quantity(partition.getQuantity()),
      capacityConstraint(std::make_shared<Constraint>(
          name, ConstraintType::CONSTR_LE, quantity)) {}

void CapacityConstraint::registerUsage(const ExpressionPtr expression,
                                       uint32_t usage) {
  if (usage == 0) {
    // No usage was registered. We don't need to add anything.
    return;
  }
  capacityConstraint->addTerm(usage);
  usageVector.emplace_back(expression, usage);
}

void CapacityConstraint::registerUsage(const ExpressionPtr expression,
                                       VariablePtr variable) {
  capacityConstraint->addTerm(variable);
  usageVector.emplace_back(expression, variable);
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

void CapacityConstraintMap::registerUsageAtTime(const ExpressionPtr expression,
                                                const Partition& partition,
                                                Time time,
                                                VariablePtr variable) {
  // Get or insert the Constraint corresponding to this partition and time.
  auto mapKey = std::make_pair(partition.getPartitionId(), time);
  if (capacityConstraints.find(mapKey) == capacityConstraints.end()) {
    capacityConstraints[mapKey] =
        std::make_shared<CapacityConstraint>(partition, time);
  }

  // Add the variable to the Constraint.
  capacityConstraints[mapKey]->registerUsage(expression, variable);
}

void CapacityConstraintMap::registerUsageAtTime(const ExpressionPtr expression,
                                                const Partition& partition,
                                                Time time, uint32_t usage) {
  if (usage == 0) {
    // No usage was registered. We don't need to add anything.
    return;
  }
  // Get or insert the Constraint corresponding to this partition and time.
  auto mapKey = std::make_pair(partition.getPartitionId(), time);
  if (capacityConstraints.find(mapKey) == capacityConstraints.end()) {
    capacityConstraints[mapKey] =
        std::make_shared<CapacityConstraint>(partition, time);
  }

  // Add the variable to the Constraint.
  capacityConstraints[mapKey]->registerUsage(expression, usage);
}

void CapacityConstraintMap::registerUsageForDuration(
    const ExpressionPtr expression, const Partition& partition, Time startTime,
    Time duration, VariablePtr variable, std::optional<Time> granularity) {
  Time _granularity = granularity.value_or(this->granularity);
  for (Time time = startTime; time < startTime + duration;
       time += _granularity) {
    registerUsageAtTime(expression, partition, time, variable);
  }
}

void CapacityConstraintMap::registerUsageForDuration(
    const ExpressionPtr expression, const Partition& partition, Time startTime,
    Time duration, uint32_t usage, std::optional<Time> granularity) {
  Time _granularity = granularity.value_or(this->granularity);
  for (Time time = startTime; time < startTime + duration;
       time += _granularity) {
    registerUsageAtTime(expression, partition, time, usage);
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
