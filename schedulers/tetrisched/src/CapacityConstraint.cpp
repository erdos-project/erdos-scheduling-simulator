#include "tetrisched/CapacityConstraint.hpp"

namespace tetrisched {

/* Method definitions for PartitionTimePairHasher */

size_t PartitionTimePairHasher::operator()(const std::pair<uint32_t, Time>& pair) const {
    auto partitionIdHash = std::hash<uint32_t>()(pair.first);
    auto timeHash = std::hash<Time>()(pair.second);
    if (partitionIdHash != timeHash) {
      return partitionIdHash ^ timeHash;
    }
    return partitionIdHash;
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

}  // namespace tetrisched
