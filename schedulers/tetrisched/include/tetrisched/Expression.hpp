#ifndef _TETRISCHED_EXPRESSION_HPP_
#define _TETRISCHED_EXPRESSION_HPP_

#include <functional>
#include <optional>
#include <set>
#include <unordered_map>

#include "tetrisched/CapacityConstraint.hpp"
#include "tetrisched/Partition.hpp"
#include "tetrisched/SolverModel.hpp"
#include "tetrisched/Types.hpp"

namespace tetrisched {
/// A `ParseResultType` enumeration represents the types of results that
/// parsing an expression can return.
enum ParseResultType {
  /// The expression is known to provide no utility.
  /// Parent expressions can safely ignore this subtree.
  EXPRESSION_NO_UTILITY = 1,
  /// The expression has been parsed successfully.
  /// The utility is attached with the return along with
  /// the relevant start and finish times.
  EXPRESSION_UTILITY = 2,
};
using ParseResultType = enum ParseResultType;
using SolutionResultType = enum ParseResultType;

/// A `ParseResult` class represents the result of parsing an expression.
struct ParseResult {
  /// The type of the result.
  ParseResultType type;
  /// The start time associated with the parsed result.
  /// Can be either a Time known at runtime or a pointer to a Solver variable.
  std::optional<TimeOrVariableT> startTime;
  /// The end time associated with the parsed result.
  /// Can be either a Time known at runtime or a pointer to a Solver variable.
  std::optional<TimeOrVariableT> endTime;
  /// The utility associated with the parsed result.
  /// The utility is positive if the expression was satisfied, and 0 otherwise.
  std::optional<ObjectiveFunctionPtr> utility;
  /// A bound on the utility of the Expression (if available).
  std::optional<TETRISCHED_ILP_TYPE> utilityBound;
  /// The indicator associated with the parsed result.
  /// Can be either 1 or 0 based on wether the expression was satisfied or not.
  std::optional<IndicatorT> indicator;
};
using ParseResultPtr = std::shared_ptr<ParseResult>;

/// A `Placement` class represents the placement of a Task.
class Placement {
 private:
  /// The name or identifer for the Task being represented by this Placement.
  std::string taskName;
  /// A boolean indicating if the Task was actually placed.
  bool placed;
  /// The start time of the Placement.
  std::optional<Time> startTime;
  /// The end time of the Placement.
  std::optional<Time> endTime;
  /// A <PartitionID, <Time, Allocation>> vector that represents the
  /// allocation of resources from each Partition to this Placement at a
  /// particular time. Note that the last Partition assignments are valid
  /// until the end of the Placement.
  std::unordered_map<uint32_t, std::set<std::pair<Time, uint32_t>>>
      partitionToResourceAllocations;

 public:
  /// Initialize a Placement with the given Task name, start time and Partition
  /// ID.
  Placement(std::string taskName, Time startTime, Time endTime);

  /// Initialize a Placement with the given Task name signifying that the Task
  /// was not actually placed.
  Placement(std::string taskName);

  /// Check if the Task was actually placed.
  bool isPlaced() const;

  /// Add an allocation for the Partition at the given time.
  void addPartitionAllocation(uint32_t partitionId, Time time,
                              uint32_t allocation);

  /// Retrieve the name of the Task.
  std::string getName() const;

  /// Set the start time of the Placement.
  void setStartTime(Time startTime);

  /// Set the end time of the Placement.
  void setEndTime(Time endTime);

  /// Retrieve the start time of the Placement, if available.
  std::optional<Time> getStartTime() const;

  /// Retrieve the end time of the Placement, if available.
  std::optional<Time> getEndTime() const;

  /// Retrieve the allocations.
  const std::unordered_map<uint32_t, std::set<std::pair<Time, uint32_t>>>&
  getPartitionAllocations() const;
};
using PlacementPtr = std::shared_ptr<Placement>;

/// A `SolutionResult` class represents the solution attributed to an
/// expression.
struct SolutionResult {
  /// The type of the result.
  SolutionResultType type;
  /// The start time associated with the result.
  std::optional<Time> startTime;
  /// The end time associated with the result.
  std::optional<Time> endTime;
  /// The utility associated with the result.
  std::optional<TETRISCHED_ILP_TYPE> utility;
  /// The placement objects being bubbled up in the solution.
  std::unordered_map<std::string, PlacementPtr> placements;
};
using SolutionResultPtr = std::shared_ptr<SolutionResult>;

/// A `ExpressionType` enumeration represents the types of expressions that
/// are supported by the STRL language.
enum ExpressionType {
  /// A Choose expression represents a choice of a required number of machines
  /// from the set of resource partitions for the given duration starting at the
  /// provided start_time.
  EXPR_CHOOSE = 0,
  /// An Objective expression collates the objectives from its children and
  /// informs the SolverModel of the objective function.
  EXPR_OBJECTIVE = 1,
  /// A Min expression inserts a utility variable that is constrained by the
  /// minimum utility of its children. Under an overall maximization objective,
  /// this ensures that the expression is only satisfied if all of its children
  /// are satisfied.
  EXPR_MIN = 2,
  /// A Max expression enforces a choice of only one of its children to be
  /// satisfied.
  EXPR_MAX = 3,
  /// A Scale expression amplifies the utility of its child by a scalar factor.
  EXPR_SCALE = 4,
  /// A LessThan expression orders the two children of its expression in an
  /// ordered relationship such that the second child occurs after the first
  /// child.
  EXPR_LESSTHAN = 5,
  /// An Allocation expression represents the allocation of the given number of
  /// machines from the given Partition for the given duration starting at the
  /// provided start_time.
  EXPR_ALLOCATION = 6,
  /// A `MalleableChoose` expression represents a choice of a flexible set of
  /// requirements of resources at each time that sums up to the total required
  /// space-time allocations from the given start to the given end time.
  /// Note that a Choose expression is a specialization of this Expression that
  /// places a rectangle of length d (duration), and height r (resources)
  /// at the given start time from the space-time allocation. However, this
  /// specialization is extremely effective to lower, and whenever possible
  /// should be used insteado of the generalized choose expression.
  EXPR_MALLEABLE_CHOOSE = 7,
  /// A `WindowedChoose` expression represents a choice of a required number of
  /// machines from the set of resource partitions for the given duration
  /// starting after the provided start time, and finishing before the required
  /// endTime. Note that, a WindowedChooseExpression is equivalent to a set of
  /// ChooseExpressions modulated by a MaxExpression. We define this
  /// higher-level abstraction since it makes it easier for us to dynamically
  /// discretize the start times of the ChooseExpressions.
  EXPR_WINDOWED_CHOOSE = 8,
};
using ExpressionType = enum ExpressionType;

/// A representation of the ranges on start and finish time.
struct ExpressionTimeBounds {
  /// The range of the start time of the Expression.
  TimeRange startTimeRange;
  /// The range of the end time of the Expression.
  TimeRange endTimeRange;
  /// The duration for this Expression.
  Time duration;
  /// A boolean indicating if the time bounds are specified.
  bool specified;
  /// Constructor for unspecified time bounds.
  ExpressionTimeBounds();
  /// Constructor for specified time bounds.
  ExpressionTimeBounds(TimeRange startTimeRange, TimeRange endTimeRange,
                       Time duration);
  /// Checks if the time bounds are specified.
  bool isSpecified() const;
  /// A string representation of the bounds.
  std::string toString() const;
};

/// A Base Class for all expressions in the STRL language.
class Expression : public std::enable_shared_from_this<Expression> {
 protected:
  /// The name of the Expression.
  std::string name;
  /// A unique ID for the Expression.
  std::string id;
  /// The parsed result from the Expression.
  /// Used for retrieving the solution from the solver.
  ParseResultPtr parsedResult;
  /// The children of this Expression.
  std::vector<ExpressionPtr> children;
  /// The parents of this Expression.
  std::vector<std::weak_ptr<Expression>> parents;
  /// The type of this Expression.
  ExpressionType type;
  /// The Solution result from this Expression.
  SolutionResultPtr solution;
  /// The time bounds for this Expression.
  ExpressionTimeBounds timeBounds;
  /// A mutex representing a lock on the parsing and population of results
  /// from this Expression.
  std::mutex expressionMutex;

  /// Adds a parent to this expression.
  void addParent(ExpressionPtr parent);

 public:
  /// Construct the Expression class of the given type.
  Expression(std::string name, ExpressionType type);

  /// Prevent copies of the Expression class.
  Expression(const Expression&) = delete;
  Expression& operator=(const Expression&) = delete;

  /// Parses the expression into a set of variables and constraints for the
  /// Solver. Returns a ParseResult that contains the utility of the expression,
  /// an indicator specifying if the expression was satisfied and variables that
  /// provide a start and end time bound on this Expression.
  virtual ParseResultPtr parse(SolverModelPtr solverModel,
                               Partitions availablePartitions,
                               CapacityConstraintMapPtr capacityConstraints,
                               Time currentTime) = 0;

  /// Adds a child to this epxression.
  /// May throw tetrisched::excpetions::ExpressionConstructionException
  /// if an incorrect number of children are registered.
  virtual void addChild(ExpressionPtr child);

  /// Removes a child from this Expression.
  void removeChild(ExpressionPtr child);

  /// Returns the name of this Expression.
  std::string getName() const;

  /// Returns the ID of this Expression.
  std::string getId() const;

  /// Returns the number of parents of this Expression.
  size_t getNumParents() const;

  /// Returns the parents of this Expression.
  std::vector<ExpressionPtr> getParents() const;

  /// Returns the number of children of this Expression.
  size_t getNumChildren() const;

  /// Returns the children of this Expression.
  std::vector<ExpressionPtr> getChildren() const;

  /// Returns the type of this Expression.
  ExpressionType getType() const;

  /// Returns the type of this Expression as a string.
  std::string getTypeString() const;

  /// Returns the time range that this Expression will be subject to.
  ExpressionTimeBounds getTimeBounds() const;

  /// Sets the time range that this Expression will be subject to.
  /// Note that only certain Expressions may choose to use this information.
  void setTimeBounds(ExpressionTimeBounds timeBounds);

  /// Populates the solution of the subtree rooted at this Expression and
  /// returns the Solution for this Expression. It assumes that the
  /// SolverModelPtr has been populated with values for unknown variables and
  /// throws a tetrisched::exceptions::ExpressionSolutionException if the
  /// SolverModelPtr is not populated.
  virtual SolutionResultPtr populateResults(SolverModelPtr solverModel);

  /// Retrieve the solution for this Expression.
  /// The Solution is only available if `populateResults` has been called on
  /// this Expression.
  std::optional<SolutionResultPtr> getSolution() const;

  /// Exports the STRL rooted at this Expression into a DOT file.
  void exportToDot(std::string filename) const;

  /// Retrieves the descriptive name for the Expression.
  virtual std::string getDescriptiveName() const;

  /// Returns the resource quantity required by the Expression.
  virtual uint32_t getResourceQuantity() const;

  /// Retrieves the ParsedResultPtr for this Expression.
  /// The ParsedResultPtr is only available if `parse` has been called on
  /// this Expression.
  std::optional<ParseResultPtr> getParsedResult() const;
};

/// A `ChooseExpression` represents a choice of a required number of machines
/// from the set of resource partitions for the given duration starting at the
/// provided start_time.
class ChooseExpression : public Expression {
 private:
  /// The Resource partitions that the ChooseExpression is being asked to
  /// choose resources from.
  Partitions resourcePartitions;
  /// The number of partitions that this ChooseExpression needs to choose.
  uint32_t numRequiredMachines;
  /// The start time of the choice represented by this Expression.
  Time startTime;
  /// The duration of the choice represented by this Expression.
  Time duration;
  /// The end time of the choice represented by this Expression.
  Time endTime;
  // The utility of the choice represented by this Expression.
  TETRISCHED_ILP_TYPE utility;
  /// The variables that represent the choice of each Partition for this
  /// Expression.
  std::unordered_map<uint32_t, VariablePtr> partitionVariables;

 public:
  ChooseExpression(std::string taskName, Partitions resourcePartitions,
                   uint32_t numRequiredMachines, Time startTime, Time duration,
                   TETRISCHED_ILP_TYPE utility);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
  SolutionResultPtr populateResults(SolverModelPtr solverModel) override;
  std::string getDescriptiveName() const override;
  uint32_t getResourceQuantity() const override;
};

/// A `WindowedChooseExpression` represents a choice of a required number of
/// machines from the set of resource partitions for the given duration starting
/// after the provided start time, and finishing before the required endTime.
class WindowedChooseExpression : public Expression {
  /// The Resource partitions that the WindowedChooseExpression is being asked
  /// to choose resources from.
  Partitions resourcePartitions;
  /// The number of partitions that this WindowedChooseExpression needs to
  /// choose.
  uint32_t numRequiredMachines;
  /// The start time after which the choice needs to be placed.
  Time startTime;
  /// The duration for which the choice needs to be placed.
  Time duration;
  /// The end time before which the choice needs to finish placement.
  Time endTime;
  /// The granularity at which the choices need to be generated.
  Time granularity;
  // The utility of the choice represented by this Expression.
  TETRISCHED_ILP_TYPE utility;
  // A map from the placement time to a Variable that represents if
  // the placement is chosen or not.
  std::unordered_map<Time, VariablePtr> placementTimeVariables;
  // A map from the placement time to a vector of <PartitionId, Variable>
  // signifying how many Placement variables are chosen from each Partition.
  std::unordered_map<Time, std::vector<std::pair<uint32_t, VariablePtr>>>
      placementPartitionVariables;

 public:
  WindowedChooseExpression(std::string taskName, Partitions resourcePartitions,
                           uint32_t numRequiredMachines, Time startTime,
                           Time duration, Time endTime, Time granularity,
                           TETRISCHED_ILP_TYPE utility);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
  SolutionResultPtr populateResults(SolverModelPtr solverModel) override;
  std::string getDescriptiveName() const override;
  uint32_t getResourceQuantity() const override;
};

/// A MalleableChooseExpression represents a choice of a flexible set of
/// requirements of resources at each time that sums up to the total required
/// space-time allocations from the given start to the given end time.
class MalleableChooseExpression : public Expression {
 private:
  /// The Resource partitions that the Expression is being asked to
  /// choose resources from.
  Partitions resourcePartitions;
  /// The total resource-time slots that this Expression needs to choose.
  /// Note that the resource-time slots are defined by the
  /// discretization of the CapacityConstraintMap.
  uint32_t resourceTimeSlots;
  /// The start time of the choice represented by this Expression.
  Time startTime;
  /// The end time of the choice represented by this Expression.
  Time endTime;
  /// The granularity at which the rectangle choices are to be made.
  Time granularity;
  // The utility of the choice represented by this Expression.
  TETRISCHED_ILP_TYPE utility;
  /// The variables that represent the choice of machines from each
  /// Partition at each time corresponding to this Expression.
  std::unordered_map<std::pair<uint32_t, Time>, VariablePtr,
                     PartitionTimePairHasher>
      partitionVariables;

 public:
  MalleableChooseExpression(std::string taskName, Partitions resourcePartitions,
                            uint32_t resourceTimeSlots, Time startTime,
                            Time endTime, Time granularity,
                            TETRISCHED_ILP_TYPE utility);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
  SolutionResultPtr populateResults(SolverModelPtr solverModel) override;
  std::string getDescriptiveName() const override;
  uint32_t getResourceQuantity() const override;
};

/// An `AllocationExpression` represents the allocation of the given number of
/// machines from the given Partition for the given duration starting at the
/// provided start_time.
class AllocationExpression : public Expression {
 private:
  /// The allocation from each Partition that is part of this Placement.
  std::vector<std::pair<PartitionPtr, uint32_t>> allocatedResources;
  /// The start time of the allocation represented by this Expression.
  Time startTime;
  /// The duration of the allocation represented by this Expression.
  Time duration;
  /// The end time of the allocation represented by this Expression.
  Time endTime;

 public:
  AllocationExpression(
      std::string taskName,
      std::vector<std::pair<PartitionPtr, uint32_t>> partitionAssignments,
      Time startTime, Time duration);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
  SolutionResultPtr populateResults(SolverModelPtr solverModel) override;
  std::string getDescriptiveName() const override;
};

/// An `ObjectiveExpression` collates the objectives from its children and
/// informs the SolverModel of the objective function.
class ObjectiveExpression : public Expression {
 public:
  ObjectiveExpression(std::string name);
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
  SolutionResultPtr populateResults(SolverModelPtr solverModel) override;
};

/// A `MinExpression` inserts a utility variable that is constrained by the
/// minimum utility of its children. Under an overall maximization objective,
/// this ensures that the expression is only satisfied if all of its children
/// are satisfied.
class MinExpression : public Expression {
 public:
  MinExpression(std::string name);
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
};

/// A `MaxExpression` enforces a choice of only one of its children to be
/// satisfied.
class MaxExpression : public Expression {
 public:
  MaxExpression(std::string name);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
};

/// A `ScaleExpression` amplifies the utility of its child by a scalar factor.
class ScaleExpression : public Expression {
 private:
  /// The scalar factor to amplify the utility of the child by.
  TETRISCHED_ILP_TYPE scaleFactor;

 public:
  ScaleExpression(std::string name, TETRISCHED_ILP_TYPE scaleFactor);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
};

/// A `LessThanExpression` orders the two children of its expression in an
/// ordered relationship such that the second child occurs after the first
/// child.
class LessThanExpression : public Expression {
 public:
  LessThanExpression(std::string name);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMapPtr capacityConstraints,
                       Time currentTime) override;
};
}  // namespace tetrisched
#endif  // _TETRISCHED_EXPRESSION_HPP_
