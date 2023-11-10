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
                                       Time constraintTime, Time granularity,
                                       bool useOverlapConstraints)
    : useOverlapConstraints(useOverlapConstraints),
      name("CapacityConstraint_" + partition.getPartitionName() + "_at_" +
           std::to_string(constraintTime)),
      quantity(partition.getQuantity()),
      granularity(granularity),
      usageUpperBound(0),
      durationUpperBound(0),
      overlapVariable(std::make_shared<Variable>(VariableType::VAR_INDICATOR,
                                                 name + "_overlap")),
      upperBoundConstraint(std::make_shared<Constraint>(
          name + "_upper_bound", ConstraintType::CONSTR_LE, granularity)),
      lowerBoundConstraint(std::make_shared<Constraint>(
          name + "_lower_bound", ConstraintType::CONSTR_GE, granularity + 0.5)),
      capacityConstraint(std::make_shared<Constraint>(
          name, ConstraintType::CONSTR_LE, quantity)) {}

void CapacityConstraint::registerUsage(const ExpressionPtr expression,
                                       const IndicatorT usageIndicator,
                                       const PartitionUsageT usageVariable,
                                       Time duration) {
  if (!usageVariable.isVariable() && usageVariable.get<uint32_t>() == 0) {
    // No usage was registered. We don't need to add anything.
    return;
  }
  if (!usageIndicator.isVariable() && usageIndicator.get<uint32_t>() == 0) {
    // This usage will never be used. We don't need to add anything.
    return;
  }

  if (useOverlapConstraints) {
    // Bookeep the maximum duration that the tasks can run for, so we can
    // set the tightest Big-M for our constraints.
    durationUpperBound += duration;

    // Bookeep the maximum usage of this Partition at this time, so we can
    // set the tightest Big-M for our constraints.
    if (usageVariable.isVariable()) {
      auto variableUpperBound =
          usageVariable.get<VariablePtr>()->getUpperBound();
      if (!variableUpperBound.has_value()) {
        throw tetrisched::exceptions::ExpressionConstructionException(
            "Usage variable " + usageVariable.get<VariablePtr>()->getName() +
            " does not have an upper bound.");
      }
      usageUpperBound += variableUpperBound.value();
    } else {
      usageUpperBound += usageVariable.get<uint32_t>();
    }

    // Add the usage indicator to both the upper and lower bound constraints.
    // This is used so that we can correctly set the overlap variable to 1,
    // if the sum of the durations exceed the granularity.
    upperBoundConstraint->addTerm(duration, usageIndicator);
    lowerBoundConstraint->addTerm(duration, usageIndicator);
  }

  // Add the usage variable to the capacity constraint so that if there
  // is an overlap, we can ensure that it is never violated.
  capacityConstraint->addTerm(usageVariable);
  usageVector.emplace_back(expression, usageVariable);
}

void CapacityConstraint::translate(SolverModelPtr solverModel) {
  // We have added all the usage variables and indicators to the
  // constraints, now we can add the indicators to the duration
  // and capacity constraints.
  if (useOverlapConstraints) {
    auto bigM =
        std::max(durationUpperBound, static_cast<double>(granularity + 1));
    solverModel->addVariable(overlapVariable);
    upperBoundConstraint->addTerm(-1 * bigM, overlapVariable);
    solverModel->addConstraint(upperBoundConstraint);

    lowerBoundConstraint->addTerm(bigM);
    lowerBoundConstraint->addTerm(-1 * bigM, overlapVariable);
    solverModel->addConstraint(lowerBoundConstraint);

    capacityConstraint->addTerm(-1 * usageUpperBound);
    capacityConstraint->addTerm(usageUpperBound, overlapVariable);
  }
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

CapacityConstraintMap::CapacityConstraintMap(Time granularity,
                                             bool useOverlapConstraints)
    : granularity(granularity), useOverlapConstraints(useOverlapConstraints) {}

CapacityConstraintMap::CapacityConstraintMap()
    : granularity(1), useOverlapConstraints(false) {}

void CapacityConstraintMap::registerUsageAtTime(
    const ExpressionPtr expression, const Partition& partition, Time time,
    const IndicatorT usageIndicator, const PartitionUsageT usageVariable,
    Time duration) {
  if (!usageIndicator.isVariable() && usageIndicator.get<uint32_t>() == 0) {
    // No usage was registered. We don't need to add anything.
    return;
  }
  if (!usageIndicator.isVariable() && usageIndicator.get<uint32_t>() == 0) {
    // This usage will never be used. We don't need to add anything.
    return;
  }

  // Get or insert the Constraint corresponding to this partition and time.
  auto mapKey = std::make_pair(partition.getPartitionId(), time);
  if (capacityConstraints.find(mapKey) == capacityConstraints.end()) {
    capacityConstraints[mapKey] = std::make_shared<CapacityConstraint>(
        partition, time, granularity, useOverlapConstraints);
  }

  // Add the variable to the Constraint.
  capacityConstraints[mapKey]->registerUsage(expression, usageIndicator,
                                             usageVariable, duration);
}

void CapacityConstraintMap::registerUsageForDuration(
    const ExpressionPtr expression, const Partition& partition, Time startTime,
    Time duration, const IndicatorT usageIndicator,
    const PartitionUsageT variable, std::optional<Time> granularity) {
  Time _granularity = granularity.value_or(this->granularity);
  Time remainderTime = duration;
  for (Time time = startTime; time < startTime + duration;
       time += _granularity) {
    if (remainderTime > _granularity) {
      registerUsageAtTime(expression, partition, time, usageIndicator, variable,
                          _granularity);
      remainderTime -= _granularity;
    } else {
      registerUsageAtTime(expression, partition, time, usageIndicator, variable,
                          remainderTime);
      remainderTime = 0;
    }
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
