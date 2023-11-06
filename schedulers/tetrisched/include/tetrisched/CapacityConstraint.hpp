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
  /// The name of this CapacityConstraint.
  std::string name;
  /// The RHS of this constraint i.e., the quantity of this Partition
  /// at this time.
  uint32_t quantity;
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
  CapacityConstraint(const Partition& partition, Time time);

  /// Registers the given usage in this CapacityConstraint.
  void registerUsage(const ExpressionPtr expression, uint32_t usage);
  void registerUsage(const ExpressionPtr expression, VariablePtr variable);

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
  /// A map from the Partition ID and the time to the
  /// CapacityConstraint that enforces the resource usage for that time.
  std::unordered_map<std::pair<uint32_t, Time>, CapacityConstraintPtr,
                     PartitionTimePairHasher>
      capacityConstraints;
  /// The default granularity for the capacity constraints.
  Time granularity;

  friend class CapacityConstraintMapPurgingOptimizationPass;

 public:
  /// Initialize a CapacityConstraintMap with the given granularity.
  CapacityConstraintMap(Time granularity);

  /// Initialize a CapacityConstraintMap with the granularity of 1.
  CapacityConstraintMap();

  /// Registers the usage by the Expression for the Partition at the
  /// time specified by the value of the variable, which is to be
  /// decided by the solver.
  void registerUsageAtTime(const ExpressionPtr expression,
                           const Partition& partition, Time time,
                           VariablePtr variable);

  /// Registers the usage by the Expression for the Partition at the
  /// time specified by the known usage.
  void registerUsageAtTime(const ExpressionPtr expression,
                           const Partition& partition, Time time,
                           uint32_t usage);

  /// Registers the usage by the Expression for the Partition in the
  /// time range starting from startTime and lasting for duration as
  /// specified by the variable, which is to be decided by the solver.
  /// Optionally, a step granularity can be provided. The default granularity
  /// is the one that the CapacityConstraintMap was initialized with.
  void registerUsageForDuration(const ExpressionPtr expression,
                                const Partition& partition, Time startTime,
                                Time duration, VariablePtr variable,
                                std::optional<Time> granularity);

  /// Registers the usage by the Expression for the Partition in the
  /// time range starting from startTime and lasting for duration as
  /// specified by the variable known at runtime. Optionally, a step
  /// granularity can be provided. The default granularity is the one
  /// that the CapacityConstraintMap was initialized with.
  void registerUsageForDuration(const ExpressionPtr expression,
                                const Partition& partition, Time startTime,
                                Time duration, uint32_t usage,
                                std::optional<Time> granularity);

  /// Translate the CapacityConstraintMap by moving its constraints
  /// to the given model.
  void translate(SolverModelPtr solverModel);

  /// The number of constraints in this map.
  size_t size() const;
};
}  // namespace tetrisched

#endif  // _TETRISCHED_CAPACITYCONSTRAINT_HPP_
