#ifndef _TETRISCHED_CAPACITYCONSTRAINT_HPP_
#define _TETRISCHED_CAPACITYCONSTRAINT_HPP_

#include "tetrisched/Partition.hpp"
#include "tetrisched/SolverModel.hpp"
#include "tetrisched/Types.hpp"

namespace tetrisched {

/// A `PartitionTimePairHasher` is a hash function for a pair of Partition ID
/// and Time. This is used to hash the key for the CapacityConstraintMap.
struct PartitionTimePairHasher {
  size_t operator()(const std::pair<uint32_t, Time>& pair) const;
};

/// A `CapacityConstraint` is a constraint that enforces the resource usage
/// for a Partition at a particular time.
class CapacityConstraint {
 private:
  /// The SolverModel constraint that enforces the resource usage.
  ConstraintPtr capacityConstraint;

  /// The CapacityConstraintMap is allowed to translate this CapacityConstraint.
  void translate(SolverModelPtr solverModel);
  friend class CapacityConstraintMap;

 public:
  /// Constructs a new CapacityConstraint for the given Partition
  /// at the given Time.
  CapacityConstraint(const Partition& partition, Time time);

  /// Registers the given usage in this CapacityConstraint.
  void registerUsage(uint32_t usage);
  void registerUsage(VariablePtr variable);
};
using CapacityConstraintPtr = std::shared_ptr<CapacityConstraint>;

/// A `CapacityConstraintMap` aggregates the terms that may potentially
/// affect the capacity of a Partition at a particular time, and provides
/// the ability for Expressions to register a variable that represents their
/// potential intent to use the Partition at a particular time.
class CapacityConstraintMap {
 private:
  /// A map from the Partition ID and the time to the
  /// CapacityConstraint that enforces the resource usage for that time.
  std::unordered_map<std::pair<uint32_t, Time>, CapacityConstraintPtr,
                     PartitionTimePairHasher>
      capacityConstraints;
  /// The default granularity for the capacity constraints.
  Time granularity;

  /// The ObjectiveExpression is allowed to translate this map.
  void translate(SolverModelPtr solverModel);
  friend class ObjectiveExpression;

 public:
  /// Initialize a CapacityConstraintMap with the given granularity.
  CapacityConstraintMap(Time granularity);

  /// Initialize a CapacityConstraintMap with the granularity of 1.
  CapacityConstraintMap();

  /// Registers the usage for the given Partition at the given time
  /// as specified by the value of the variable, which is to be
  /// decided by the solver.
  void registerUsageAtTime(const Partition& partition, Time time,
                           VariablePtr variable);

  /// Registers the usage for the given Partition at the given time
  /// as specified by the known usage.
  void registerUsageAtTime(const Partition& partition, Time time,
                           uint32_t usage);

  /// Registers the usage for the given Partition in the time range
  /// starting from startTime and lasting for duration as specified
  /// by the value of the variable, which is to be decided by the solver.
  /// Optionally, a step granularity can be provided. The default granularity
  /// is the one that the CapacityConstraintMap was initialized with.
  void registerUsageForDuration(const Partition& partition, Time startTime,
                                Time duration, VariablePtr variable,
                                std::optional<Time> granularity);

  /// Registers the usage for the given Partition in the time range
  /// starting from startTime and lasting for duration as specified
  /// by the value of the variable known at runtime. Optionally, a step
  /// granularity can be provided. The default granularity is the one
  /// that the CapacityConstraintMap was initialized with.
  void registerUsageForDuration(const Partition& partition, Time startTime,
                                Time duration, uint32_t usage,
                                std::optional<Time> granularity);

  /// The number of constraints in this map.
  size_t size() const;
};
}  // namespace tetrisched

#endif  // _TETRISCHED_CAPACITYCONSTRAINT_HPP_
