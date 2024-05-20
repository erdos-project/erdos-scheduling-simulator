#include "tetrisched/CapacityConstraint.hpp"

#include "tbb/task_group.h"

namespace tetrisched {

/* Method definitions for PartitionTimePairHasher */

size_t PartitionTimePairHasher::operator()(
    const std::pair<uint32_t, Time>& pair) const {
  return hash(pair);
}

size_t PartitionTimePairHasher::hash(const std::pair<uint32_t, Time>& pair) {
  auto partitionIdHash = std::hash<uint32_t>()(pair.first);
  auto timeHash = std::hash<Time>()(pair.second);
  if (partitionIdHash != timeHash) {
    return partitionIdHash ^ timeHash;
  }
  return partitionIdHash;
}

bool PartitionTimePairHasher::equal(const std::pair<uint32_t, Time>& pair1,
                                    const std::pair<uint32_t, Time>& pair2) {
  return pair1.first == pair2.first && pair1.second == pair2.second;
}

/* Method definitions for CapacityConstraint */
CapacityConstraint::CapacityConstraint(const Partition& partition,
                                       Time constraintTime, Time granularity,
                                       bool useOverlapConstraints)
    : useOverlapConstraints(useOverlapConstraints),
      name("CapacityConstraint_" + partition.getPartitionName() + "_at_" +
           std::to_string(constraintTime)),
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
          name, ConstraintType::CONSTR_LE, partition.getQuantity())),
      quantity(partition.getQuantity()) {}

void CapacityConstraint::registerUsage(const Expression* expression,
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
  if (usageVariable.isVariable()) {
    auto variableUpperBound = usageVariable.get<VariablePtr>()->getUpperBound();
    if (!variableUpperBound.has_value()) {
      throw tetrisched::exceptions::ExpressionConstructionException(
          "Usage variable " + usageVariable.get<VariablePtr>()->getName() +
          " does not have an upper bound.");
    }
    if (usageMap.find(expression) == usageMap.end()) {
      usageMap[expression] = variableUpperBound.value();
    } else {
      usageMap[expression] =
          std::max(usageMap[expression], variableUpperBound.value());
    }
  } else {
    auto usageVariableValue = usageVariable.get<uint32_t>();
    quantity -= usageVariableValue;
  }
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
  capacityConstraint->addAttribute(ConstraintAttribute::LAZY_CONSTRAINT);
  solverModel->addConstraint(capacityConstraint);
}

void CapacityConstraint::deactivate() { capacityConstraint->deactivate(); }

std::string CapacityConstraint::getName() const { return name; }

/* Method definitions for CapacityConstraintMap */
CapacityConstraintMap::CapacityConstraintMap()
    : CapacityConstraintMap(1, false) {}

CapacityConstraintMap::CapacityConstraintMap(Time granularity,
                                             bool useOverlapConstraints)
    : granularity(granularity),
      useOverlapConstraints(useOverlapConstraints),
      useDynamicDiscretization(false) {}

CapacityConstraintMap::CapacityConstraintMap(
    std::vector<std::pair<TimeRange, Time>> timeRangeToGranularities,
    bool useOverlapConstraints)
    : useOverlapConstraints(useOverlapConstraints),
      useDynamicDiscretization(true),
      timeRangeToGranularities(timeRangeToGranularities) {
  // Check that the time ranges provided are non-overlapping and
  // monotonically increasing.
  for (decltype(timeRangeToGranularities)::size_type i = 0;
       i < timeRangeToGranularities.size(); i++) {
    if (i != timeRangeToGranularities.size() - 1) {
      if (timeRangeToGranularities[i + 1].first.first <
          timeRangeToGranularities[i].first.second) {
        throw tetrisched::exceptions::RuntimeException(
            "Time ranges are not in increasing, non-overlapping order.");
      }
    }
  }
}

void CapacityConstraintMap::registerUsageAtTime(
    const Expression* expression, const Partition& partition, const Time time,
    const Time granularity, const IndicatorT usageIndicator,
    const PartitionUsageT usageVariable, const Time duration) {
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
  {
    decltype(capacityConstraints)::accessor capacityConstraintAccessor;

    const auto isFound =
        capacityConstraints.find(capacityConstraintAccessor, mapKey);

    if (!isFound) {
      auto insertKeyVal = std::make_pair(
          mapKey, std::make_shared<CapacityConstraint>(
                      partition, time, granularity, useOverlapConstraints));
      capacityConstraints.insert(capacityConstraintAccessor, insertKeyVal);
    }

    // Add the variable to the Constraint.
    capacityConstraintAccessor->second->registerUsage(
        expression, usageIndicator, usageVariable, duration);
  }
}

void CapacityConstraintMap::setDynamicDiscretization(
    std::vector<std::pair<TimeRange, Time>> passedtimeRangeToGranularities) {
  timeRangeToGranularities = passedtimeRangeToGranularities;
  useDynamicDiscretization = true;
}

void CapacityConstraintMap::registerUsageForDuration(
    const Expression* expression, const Partition& partition,
    const Time startTime, const Time duration, const IndicatorT usageIndicator,
    const PartitionUsageT variable, std::optional<Time> granularity, bool print,
    std::string exprName) {
  if (!useDynamicDiscretization) {
    // If we are not using dynamic discretization, then we can just
    // register the usage at the provided granularity.
    Time _granularity = granularity.value_or(this->granularity);
    Time remainderTime = duration;
    tbb::task_group registerUsageTaskGroup;
    for (Time time = startTime; time < startTime + duration;
         time += _granularity) {
      if (remainderTime > _granularity) {
        registerUsageTaskGroup.run([this, &expression, &partition, time,
                                    &_granularity, &usageIndicator, &variable] {
          registerUsageAtTime(expression, partition, time, _granularity,
                              usageIndicator, variable, _granularity);
        });
        remainderTime -= _granularity;
      } else {
        registerUsageTaskGroup.run([this, &expression, &partition, time,
                                    &_granularity, &usageIndicator, &variable,
                                    remainderTime] {
          registerUsageAtTime(expression, partition, time, _granularity,
                              usageIndicator, variable, remainderTime);
        });
        remainderTime = 0;
      }
    }
    registerUsageTaskGroup.wait();
  } else {
    // Find the first interval where the start time is in the range.
    decltype(timeRangeToGranularities)::size_type granularityIndex = 0;
    for (; granularityIndex < timeRangeToGranularities.size();
         ++granularityIndex) {
      auto& timeRange = timeRangeToGranularities[granularityIndex].first;
      if (startTime < timeRange.second) {
        break;
      }
    }

    if (granularityIndex == timeRangeToGranularities.size()) {
      throw tetrisched::exceptions::RuntimeException(
          "Start time is out of range of the discretization.");
    }

    // Time currentTime = startTime;
    Time currentTime = timeRangeToGranularities[granularityIndex].first.first;
    Time remainderTime = duration + (startTime - currentTime);
    if (print) {
      std::cout <<
        "[CapConstraint] Registering usage for expression: " <<
        exprName << " starting at " << startTime << " with duration " <<
        duration << " at time: " << currentTime << " and remainder time: " <<
        remainderTime << "." << std::endl;
    }
    while (remainderTime > 0) {
      auto& timeRange = timeRangeToGranularities[granularityIndex].first;
      auto& granularity = timeRangeToGranularities[granularityIndex].second;
      TETRISCHED_DEBUG("Registering usage for expression starting at "
                       << startTime << " with duration " << duration
                       << " at time: " << currentTime
                       << " within the time range [" << timeRange.first << ", "
                       << timeRange.second
                       << "] with granularity: " << granularity
                       << " and remainder time: " << remainderTime << ".")

      // Check if either we're in the interval, or if there's any remainder
      // time if we're in the upmost interval.
      while (currentTime < std::min(startTime + duration, timeRange.second) ||
             (remainderTime > 0 &&
              granularityIndex == timeRangeToGranularities.size() - 1)) {
        if (remainderTime > granularity) {
          registerUsageAtTime(expression, partition, currentTime, granularity,
                              usageIndicator, variable, granularity);
          currentTime += granularity;
          remainderTime -= granularity;
        } else {
          registerUsageAtTime(expression, partition, currentTime, granularity,
                              usageIndicator, variable, remainderTime);
          currentTime += granularity;
          remainderTime = 0;
        }
        if (print){
          std::cout <<
            "\t[CapConstraintInside] Registering usage for expression starting at "
            << startTime << " with duration " << duration
            << " at time: " << currentTime
            << " within the time range [" << timeRange.first
            << ", "
            << timeRange.second
            << "] with granularity: " << granularity
            << " and remainder time: " << remainderTime << "."
            << std::endl;
        }

        if (remainderTime != 0 &&
            currentTime >= std::min(startTime + duration, timeRange.second) &&
            granularityIndex == timeRangeToGranularities.size() - 1) {
          remainderTime = 0;
          break;
        }
      }
      if (currentTime >=
          timeRangeToGranularities[granularityIndex].first.second) {
        // We have passed the interval. We need to skip to the next interval,
        // if possible.
        if (granularityIndex != timeRangeToGranularities.size() - 1) {
          granularityIndex++;
        }
      }
    }
  }
}

void CapacityConstraintMap::translate(SolverModelPtr solverModel) {
  // Add the constraints to the SolverModel.
  for (auto& [mapKey, constraint] : capacityConstraints) {
    constraint->translate(solverModel);
  }
}

size_t CapacityConstraintMap::size() const {
  return capacityConstraints.size();
}

/// Destructs the CapacityConstraintMap.
/// Drops the underlying map since this is the fastest.
/// tbb::concurrent_hash_map clear() which is called by the default
/// constructor is unbelievably slow (sometimes adding ~300ms to the
/// critical path runtime).
// CapacityConstraintMap::~CapacityConstraintMap() {
//   capacityConstraints = decltype(capacityConstraints)();
// }

}  // namespace tetrisched
