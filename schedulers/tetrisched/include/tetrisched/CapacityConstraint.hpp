#ifndef _TETRISCHED_CAPACITYCONSTRAINT_HPP_
#define _TETRISCHED_CAPACITYCONSTRAINT_HPP_

#include "tbb/concurrent_hash_map.h"
#include "tetrisched/Partition.hpp"
#include "tetrisched/SolverModel.hpp"
#include "tetrisched/Types.hpp"

namespace tetrisched {

/// A `PartitionTimePairHasher` is a hash function for a pair of Partition ID
/// and Time. This is used to hash the key for the CapacityConstraintMap.
struct PartitionTimePairHasher {
  size_t operator()(const std::pair<uint32_t, Time>& pair) const;
  static size_t hash(const std::pair<uint32_t, Time>& pair);
  static bool equal(const std::pair<uint32_t, Time>& pair1,
                    const std::pair<uint32_t, Time>& pair2);
};

/// A `CapacityConstraint` is a constraint that enforces the resource usage
/// for a Partition at a particular time.
class CapacityConstraint {
 private:
  /// If True, we generate overlap constraints.
  /// If False, we fall back to using general capacity constraints.
  bool useOverlapConstraints;
  /// The name of this CapacityConstraint.
  std::string name;
  /// The RHS of this constraint i.e., the quantity of this Partition
  /// at this time.
  uint32_t quantity;
  /// The granularity of this CapacityConstraint.
  /// This is typically the duration of the Constraint.
  Time granularity;
  /// The upper bound for the usage of this Partition at this time.
  TETRISCHED_ILP_TYPE usageUpperBound;
  /// The upper bound for the duration of this Partition at this time.
  TETRISCHED_ILP_TYPE durationUpperBound;
  /// The VariablePtr that checks if there is any overlapping usage.
  VariablePtr overlapVariable;
  /// The SolverModel constraints that ensures the correct setting of
  /// the overlap variable.
  ConstraintPtr upperBoundConstraint;
  ConstraintPtr lowerBoundConstraint;
  /// The SolverModel constraint that enforces the resource usage.
  ConstraintPtr capacityConstraint;

  /// A vector of the Expression contributing the given usage to Constraint.
  std::vector<std::pair<ExpressionPtr, XOrVariableT<uint32_t>>> usageVector;

  /// The CapacityConstraintMap is allowed to translate this CapacityConstraint.
  void translate(SolverModelPtr solverModel);
  friend class CapacityConstraintMap;
  friend class CapacityConstraintMapPurgingOptimizationPass;

 public:
  /// Constructs a new CapacityConstraint for the given Partition
  /// at the given Time.
  CapacityConstraint(const Partition& partition, Time constraintTime,
                     Time granularity, bool useOverlapConstraints);

  /// Registers the given usage in this CapacityConstraint.
  void registerUsage(const ExpressionPtr expression,
                     const IndicatorT usageIndicator,
                     const PartitionUsageT usageVariable, Time duration);

  /// Retrieves the maximum quantity of this CapacityConstraint.
  uint32_t getQuantity() const;

  /// Retrieves the name for this CapacityConstraint.
  std::string getName() const;

  /// Deactivates this CapacityConstraint.
  void deactivate();
};
using CapacityConstraintPtr = std::shared_ptr<CapacityConstraint>;

/// A `CapacityConstraintMap` aggregates the terms that may potentially
/// affect the capacity of a Partition at a particular time, and provides
/// the ability for Expressions to register a variable that represents their
/// potential intent to use the Partition at a particular time.
class CapacityConstraintMap {
 private:
  /// The default granularity for the capacity constraints.
  Time granularity;
  /// If True, the constraintMap will expend extra effort in ensuring
  /// overlaps are efficiently used. While this may increase scheduling
  /// performance, it may also increase the time it takes to solve the model.
  bool useOverlapConstraints;
  /// if True, dynamic discretization is enabled. In this mode, the
  /// CapacityConstraintMap has different CapacityConstraints that cover
  /// different times.
  bool useDynamicDiscretization;
  /// A sorted list of TimeRange to the granularity at which that time
  /// range needs to be discretized.
  std::vector<std::pair<TimeRange, Time>> timeRangeToGranularities;
  /// A map from the Partition ID and the time to the
  /// CapacityConstraint that enforces the resource usage for that time.
  tbb::concurrent_hash_map<std::pair<uint32_t, Time>, CapacityConstraintPtr,
                           PartitionTimePairHasher>
      capacityConstraints;

  friend class CapacityConstraintMapPurgingOptimizationPass;

 public:
  /// Initialize a CapacityConstraintMap with the granularity of 1.
  CapacityConstraintMap();

  /// Initialize a CapacityConstraintMap with the given granularitqy.
  CapacityConstraintMap(Time granularity, bool useOverlapConstraints = false);

  /// Initialize a CapacityConstraintMap with the given range-based granularity.
  CapacityConstraintMap(
      std::vector<std::pair<TimeRange, Time>> timeRangeToGranularities,
      bool useOverlapConstraints = false);

  /// Registers the usage by the Expression for the Partition at the
  /// specified time for the specified duration, which is to be
  /// decided by the solver. The variable is expected to take on a
  /// value >= 0 only if the indicator is 1.
  void registerUsageAtTime(const ExpressionPtr expression,
                           const Partition& partition, const Time time,
                           const Time granularity,
                           const IndicatorT usageIndicator,
                           const PartitionUsageT usageVariable,
                           const Time duration);
  /// Sets dynamic discretization for capacity contraint map
  void setDynamicDiscretization(
      std::vector<std::pair<TimeRange, Time>> passedtimeRangeToGranularities);

  /// Registers the usage by the Expression for the Partition in the
  /// time range starting from startTime and lasting for duration as
  /// specified by the variable, which is to be decided by the solver.
  /// Optionally, a step granularity can be provided. The default granularity
  /// is the one that the CapacityConstraintMap was initialized with.
  void registerUsageForDuration(const ExpressionPtr expression,
                                const Partition& partition,
                                const Time startTime, const Time duration,
                                const IndicatorT usageIndicator,
                                const PartitionUsageT usageVariable,
                                std::optional<Time> granularity);

  /// Translate the CapacityConstraintMap by moving its constraints
  /// to the given model.
  void translate(SolverModelPtr solverModel);

  /// The number of constraints in this map.
  size_t size() const;
};
typedef std::shared_ptr<CapacityConstraintMap> CapacityConstraintMapPtr;
}  // namespace tetrisched

#endif  // _TETRISCHED_CAPACITYCONSTRAINT_HPP_
