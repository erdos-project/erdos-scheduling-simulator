#ifndef _TETRISCHED_EXPRESSION_HPP_
#define _TETRISCHED_EXPRESSION_HPP_

#include <functional>
#include <optional>
#include <unordered_map>

#include "tetrisched/Partition.hpp"
#include "tetrisched/SolverModel.hpp"
#include "tetrisched/Types.hpp"

namespace tetrisched {

/// A `UtilityFn` represents the function that is used to calculate the utility
/// of a particular expression.
template <typename T>
using UtilityFnT = std::function<T(Time, Time)>;
using UtilityFn = UtilityFnT<TETRISCHED_ILP_TYPE>;

/// A `ParseResultType` enumeration represents the types of results that
/// parsing an expression can return.
enum ParseResultType {
  /// The expression can be pruned from the subtree.
  /// This occurs if the time bounds for the choices
  /// have evolved past the current time.
  EXPRESSION_PRUNE = 0,
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
  using TimeOrVariableT = XOrVariableT<Time>;
  using IndicatorT = XOrVariableT<uint32_t>;
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
  /// The indicator associated with the parsed result.
  /// Can be either 1 or 0 based on wether the expression was satisfied or not.
  std::optional<IndicatorT> indicator;
};
using ParseResultPtr = std::shared_ptr<ParseResult>;

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
};
using SolutionResultPtr = std::shared_ptr<SolutionResult>;

struct PartitionTimePairHasher {
  size_t operator()(const std::pair<uint32_t, Time>& pair) const {
    auto partitionIdHash = std::hash<uint32_t>()(pair.first);
    auto timeHash = std::hash<Time>()(pair.second);
    if (partitionIdHash != timeHash) {
      return partitionIdHash ^ timeHash;
    }
    return partitionIdHash;
  }
};

/// A `CapacityConstraintMap` aggregates the terms that may potentially
/// affect the capacity of a Partition at a particular time, and provides
/// the ability for Expressions to register a variable that represents their
/// potential intent to use the Partition at a particular time.
class CapacityConstraintMap {
 private:
  /// A map from the Partition ID and the time to the ConstraintPtr that
  /// enforces the resource usage for that time.
  std::unordered_map<std::pair<uint32_t, Time>, ConstraintPtr,
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
};
using ExpressionType = enum ExpressionType;

/// A Base Class for all expressions in the STRL language.
class Expression : public std::enable_shared_from_this<Expression>
{
protected:
  /// The parsed result from the Expression.
  /// Used for retrieving the solution from the solver.
  ParseResultPtr parsedResult;
  /// The children of this Expression.
  std::vector<ExpressionPtr> children;
  /// The parents of this Expression.
  std::vector<std::weak_ptr<Expression>> parents;
  /// The type of this Expression.
  ExpressionType type;

 public:
  /// Construct the Expression class of the given type.
  Expression(ExpressionType type);

  /// Adds a child to this epxression.
  /// May throw tetrisched::excpetions::ExpressionConstructionException
  /// if an incorrect number of children are registered.
  virtual void addChild(ExpressionPtr child) = 0;

  /// Parses the expression into a set of variables and constraints for the
  /// Solver. Returns a ParseResult that contains the utility of the expression,
  /// an indicator specifying if the expression was satisfied and variables that
  /// provide a start and end time bound on this Expression.
  virtual ParseResultPtr parse(SolverModelPtr solverModel,
                               Partitions availablePartitions,
                               CapacityConstraintMap& capacityConstraints,
                               Time currentTime) = 0;

  /// Adds a parent to this epxression.
  void addParent(std::weak_ptr<Expression> parent);

  /// Returns the number of children of this Expression.
  size_t getNumChildren() const;

  /// Returns the number of parents of this Expression.
  size_t getNumParents() const;

  /// Returns the children of this Expression.
  std::vector<ExpressionPtr> getChildren() const;

  /// Returns the children of this Expression.
  std::vector<std::weak_ptr<Expression>> getParents() const;

  /// Returns the type of this Expression.
  ExpressionType getType() const;

  /// Solves the subtree rooted at this Expression and returns the solution.
  /// It assumes that the SolverModelPtr has been populated with values for
  /// unknown variables and throws a
  /// tetrisched::exceptions::ExpressionSolutionException if the SolverModelPtr
  /// is not populated. This method returns the actual values for the variables
  /// specified in the ParseResult.
  SolutionResultPtr solve(SolverModelPtr solverModel);
};

/// A `ChooseExpression` represents a choice of a required number of machines
/// from the set of resource partitions for the given duration starting at the
/// provided start_time.
class ChooseExpression : public Expression {
 private:
  /// The name of the Task that this ChooseExpression is being inserted
  /// into the AST in reference to.
  std::string taskName;
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

 public:
  ChooseExpression(std::string taskName, Partitions resourcePartitions,
                   uint32_t numRequiredMachines, Time startTime, Time duration);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMap& capacityConstraints,
                       Time currentTime) override;
};

/// An `ObjectiveExpression` collates the objectives from its children and
/// informs the SolverModel of the objective function.
class ObjectiveExpression : public Expression {
 public:
  ObjectiveExpression();
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMap& capacityConstraints,
                       Time currentTime) override;
};

/// A `MinExpression` inserts a utility variable that is constrained by the
/// minimum utility of its children. Under an overall maximization objective,
/// this ensures that the expression is only satisfied if all of its children
/// are satisfied.
class MinExpression : public Expression {
 private:
  /// The name of the expression.
  std::string expressionName;

 public:
  MinExpression(std::string name);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMap& capacityConstraints,
                       Time currentTime) override;
};

/// A `MaxExpression` enforces a choice of only one of its children to be
/// satisfied.
class MaxExpression : public Expression {
 private:
  /// The name of the expression.
  std::string expressionName;

 public:
  MaxExpression(std::string name);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMap& capacityConstraints,
                       Time currentTime) override;
};

/// A `ScaleExpression` amplifies the utility of its child by a scalar factor.
class ScaleExpression : public Expression {
 private:
  /// The name of the expression.
  std::string expressionName;
  /// The scalar factor to amplify the utility of the child by.
  TETRISCHED_ILP_TYPE scaleFactor;

 public:
  ScaleExpression(std::string name, TETRISCHED_ILP_TYPE scaleFactor);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMap& capacityConstraints,
                       Time currentTime) override;
};

/// A `LessThanExpression` orders the two children of its expression in an
/// ordered relationship such that the second child occurs after the first
/// child.
class LessThanExpression : public Expression {
 private:
  /// The name for this Expression.
  std::string name;

 public:
  LessThanExpression(std::string name);
  void addChild(ExpressionPtr child) override;
  ParseResultPtr parse(SolverModelPtr solverModel,
                       Partitions availablePartitions,
                       CapacityConstraintMap& capacityConstraints,
                       Time currentTime) override;
};
}  // namespace tetrisched
#endif  // _TETRISCHED_EXPRESSION_HPP_

// #ifndef _EXPRESSION_HPP_
// #define _EXPRESSION_HPP_
// // standard C/C++ libraries
// #include <iostream>
// #include <map>
// #include <tuple>
// #include <unordered_map>
// #include <vector>
// // boost libraries
// #include <boost/shared_ptr.hpp>
// // 3rd party libraries
// // alsched libraries
// #include "Partition.hpp"
// #include "Solver.hpp"
// #include "SolverModel.hpp"

// using namespace std;

// namespace alsched {

// class Expression {
//  protected:
//   bool marker = false;

//  public:
//   static const int ALSCHED_EXPR_EXCEPTION = 0xff;

//   virtual void clearMarkers() = 0;
//   virtual tuple<double, double> eval(Allocation alloc) = 0;
//   virtual void addChild(
//       ExpressionPtr newchld) = 0;  // appends new chld to chldarr
//   virtual vector<pair<double, int> > generate(
//       SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//       vector<map<double, Constraint> > &capconmap, JobPtr jobptr) = 0;
//   // return the set of all eq.classes referenced by this expression
//   (non-const) virtual void getEquivClasses(EquivClassSet &equivClasses) = 0;
//   virtual void populatePartitions(const vector<int> &node2part, int curtime,
//                                   int sched_horizon) = 0;
//   virtual string toString() = 0;
//   virtual Allocation getResults(const Solver &solver,
//                                 vector<map<double, vector<int> > > &partcap,
//                                 int start_time) = 0;
//   virtual void cacheNodeResults(const Solver &solver,
//                                 vector<map<double, vector<int> > > &partcap,
//                                 int start_time) = 0;

//   // returns interval [t_1, t_2) where this expression is meaningful
//   (non-const
//   // due to cache)
//   virtual pair<int, int> startTimeRange() = 0;
// };

// class SchedulingExpression : public Expression {
//  protected:
//   /// The Task associated with this SchedulingExpression.
//   TaskPtr associatedTask;
//   /// Set of candidate machines that this expression can be scheduled on.
//   Partitions candidateMachines;
//   /// The number of nodes that this expression requires.
//   /// Note that this number cannot be more than the number of candidate
//   nodes. uint32_t requiredMachines;
//   /// The start time at which the expression can be scheduled.
//   uint32_t startTime;
//   /// The duration for which this scheduling needs to occur.
//   uint32_t duration;
//   /// The utility to assign to this expression.
//   uint32_t utility;

//   NodesPtr _nodes;
//   int _k;
//   double _utility;
//   int _start_time;
//   double _duration;

//   // resolve() will dynamically resolve eq.classes into a set of partitions
//   vector<int> partitions;  // partition number of each partition
//   // array of par.var. indices that match partitions: requires a reset in
//   gen() vector<int>
//       partvaridx;  // partition variable for each partition. owned by
//       generate()

//   vector<int>
//       cached_nodes;  // all machines allocated to this nck leaf previously
//   // owns cache : purges it and populates it with new results from the solver
//   // resolves solver results to nodes for *this leaf and caches them
//   // mutates partcap, removing from it nodes allocated to *this leaf
//   virtual void cacheNodeResults(const Solver &solver,
//                                 vector<map<double, vector<int> > > &partcap,
//                                 int start_time);

//  public:
//   virtual ~SchedulingExpression() {}
//   SchedulingExpression(TaskPtr associatedTask, Partitions candidateMachines,
//                        uint32_t requiredMachines, uint32_t startTime,
//                        uint32_t duration, uint32_t utility)
//       : associatedTask(associatedTask),
//         candidateMachines(candidateMachines),
//         requiredMachines(requiredMachines),
//         startTime(startTime),
//         duration(duration),
//         utility(utility) {}
//   SchedulingExpression(NodesPtr nodes, int k, double utility, int start_time,
//                        double duration)
//       : _nodes(nodes),
//         _start_time(start_time),
//         _duration(duration),
//         _k(k),
//         _utility(utility) {}
//   void addChild(ExpressionPtr newchld) {
//     throw Expression::ALSCHED_EXPR_EXCEPTION;
//   }
//   virtual void getEquivClasses(EquivClassSet &equivClasses);
//   virtual void populatePartitions(const vector<int> &node2part, int curtime,
//                                   int sched_horizon);
//   virtual Allocation getResults(const Solver &solver,
//                                 vector<map<double, vector<int> > > &partcap,
//                                 int start_time);
//   void setPartitions(vector<int> parts) { partitions = parts; }

//   void clearMarkers() { marker = false; }
//   pair<int, int> startTimeRange() {
//     return make_pair(_start_time, _start_time + 1);
//   };
//   virtual string debugString() const;
// };

// class Choose : public SchedulingExpression {
//  public:
//   // choose <k> machines from <equivclass> with utility value <utilval>
//   Choose(NodesPtr nodes, int k, UtilVal utilval, int start_time,
//          double duration)
//       : SchedulingExpression(nodes, k, utilval(start_time, duration),
//                              start_time, duration){};

//   /// A Choose expression is a leaf node in the expression tree that
//   represents
//   /// a choice of a required number of machines from the set of candidate
//   /// machines for the given duration starting at the provided start_time.
//   /// Args:
//   ///   associatedTask: The Task associated with this Expression.
//   ///   candidateMachines: The set of candidate machines that this expression
//   ///         can be scheduled on.
//   ///   requiredMachines: The number of machines that this expression
//   requires.
//   ///      Note that this number cannot be more than the number of candidate
//   ///      machines.
//   ///   startTime: The start time at which the expression can be scheduled.
//   ///   duration: The duration for which this scheduling needs to occur.
//   ///   utilityFn: A function that returns the utility of scheduling this
//   ///       expression expressed through the inputs of start time and
//   duration. Choose(TaskPtr associatedTask, Partitions candidateMachines,
//          uint32_t requiredMachines, uint32_t startTime, uint32_t duration,
//          UtilityFn utilityFn)
//       : SchedulingExpression(associatedTask, candidateMachines,
//                              requiredMachines, startTime, duration,
//                              utilityFn(startTime, duration)) {}

//   /// Automatically reduces the reference counts to the candidateMachines
//   /// passed to the constructor.
//   ~Choose() {}

//   /// Parses this Expression into a set of variables and constraints for
//   /// the Solver.
//   std::unique_ptr<ParseResult> parse(SolverModelPtr solverModel,
//                                      Partitions resourcePartitions,
//                                      uint32_t currentTime,
//                                      uint32_t schedulingHorizon);

//   tuple<double, double> eval(Allocation alloc);
//   // recursively generate a model for a solver; aggregate capacity
//   constraints vector<pair<double, int> > generate(
//       SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//       vector<map<double, Constraint> > &capconmap, JobPtr jobptr);
//   virtual string toString();
// };

// // class LinearChoose: public SchedulingExpression {
// // public:
// //     typedef boost::shared_ptr<LinearChoose> lnckExprPtr;
// //     LinearChoose(NodesPtr nodes, int k, UtilVal utilval, int start_time,
// //     double duration)
// //         : SchedulingExpression(nodes, k, utilval(start_time, duration),
// //         start_time, duration) {};
// //     virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
// // vector<map<double,vector<int>
// //                                                > > &partcap,
// // vector<map<double,Constraint>
// //                                                >& capconmap, JobPtr
// jobptr);

// //     virtual ~LinearChoose() {}
// //     virtual tuple<double, double> eval(Allocation alloc);
// //     virtual string toString();
// // };

// // class PreemptingExpression: public SchedulingExpression {
// // protected:
// //     unordered_map<int, vector<int> > partition_nodes;
// //     void cacheNodeResults(const Solver &solver,
// //                                   vector<map<double, vector<int> > >
// //                                   &partcap, int start_time);

// // public:
// //     PreemptingExpression(NodesPtr nodes, double utility, int start_time,
// //     double duration)
// //             : SchedulingExpression(nodes, static_cast<int>(nodes->size()),
// //             utility, start_time, duration) {};
// //     virtual ~PreemptingExpression() {}
// //     void getEquivClasses(EquivClassSet& equivClasses) { } // No-op
// //     void populatePartitions(const vector<int> &node2part, int curtime, int
// //     sched_horizon);
// // };

// // class KillChoose: public PreemptingExpression {
// // protected:
// // public:
// //     typedef boost::shared_ptr<KillChoose> KillChooseExprPtr;
// //     KillChoose(NodesPtr nodes, UtilVal utilval, double start_time, double
// //     duration)
// //             : PreemptingExpression(nodes, utilval(start_time, duration),
// //             start_time, duration) {}
// //     virtual ~KillChoose() {}
// //     virtual tuple<double, double> eval(Allocation alloc);
// //     virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
// // vector<map<double,vector<int>
// //                                                > > &partcap,
// // vector<map<double,Constraint>
// //                                                >& capconmap, JobPtr
// jobptr);
// //     virtual string toString();

// // };

// // class KillLinearChoose: public PreemptingExpression {
// // public:
// //     typedef boost::shared_ptr<KillLinearChoose> KillLinearChooseExprPtr;
// //     KillLinearChoose(NodesPtr nodes, UtilVal utilval, double start_time,
// //     double duration)
// //             : PreemptingExpression(nodes, utilval(start_time, duration),
// //             start_time, duration) {}
// //     virtual ~KillLinearChoose() {}
// //     virtual tuple<double, double> eval(Allocation alloc);
// //     virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
// // vector<map<double,vector<int>
// //                                                > > &partcap,
// // vector<map<double,Constraint>
// //                                                >& capconmap, JobPtr
// jobptr);
// //     virtual string toString();

// // };

// // class UnaryOperator: public Expression {

// // protected:
// //     ExpressionPtr child;
// // public:
// //     UnaryOperator() {}
// //     UnaryOperator(ExpressionPtr chld) : child(chld) {}
// //     virtual ~UnaryOperator() {}
// //     void clearMarkers();
// //     void addChild(ExpressionPtr newchld);
// //     void getEquivClasses(EquivClassSet& equivClasses);
// //     void populatePartitions(const vector<int> &node2part, int curtime, int
// //     sched_horizon); Allocation getResults(const Solver &solver,
// //                           vector<map<double, vector<int> > > &partcap,
// //                           int start_time);
// //     void cacheNodeResults(const Solver &solver,
// //                           vector<map<double, vector<int> > > &partcap,
// //                           int st);
// //     pair<int, int> startTimeRange() { return child->startTimeRange(); }
// // };

// // class BarrierExpr: public UnaryOperator {
// // private:
// //     double barrier;
// //     double start_time; // start time
// //     double duration; // duration

// // public:
// //     typedef boost::shared_ptr<BarrierExpr> BarrierExprPtr;
// //     BarrierExpr(double barrier, double start_time, double duration);
// //     BarrierExpr(double barrier, double start_time, double duration,
// //     ExpressionPtr newchld); virtual ~BarrierExpr() {} virtual
// //     vector<pair<double,int> > generate(SolverModelPtr m, int I,
// //             vector<map<double,vector<int> > > &partcap,
// //             vector<map<double,Constraint> >& capconmap,
// //             JobPtr jobptr);

// //     virtual tuple<double, double> eval(Allocation alloc);
// //     virtual string toString();
// // };

// // class ScaleExpr: public UnaryOperator {
// // private:
// //     double factor;

// // public:
// //     typedef boost::shared_ptr<ScaleExpr> ScaleExprPtr;
// //     ScaleExpr(double factor);
// //     ScaleExpr(double factor, ExpressionPtr newchld);
// //     virtual ~ScaleExpr() {}
// //     virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
// //             vector<map<double,vector<int> > > &partcap,
// //             vector<map<double,Constraint> >& capconmap,
// //             JobPtr jobptr);
// //     virtual tuple<double, double> eval(Allocation alloc);
// //     virtual string toString();
// // };

// // // Job expression : sole purpose is to pass the jobptr down to children
// // class JobExpr: public UnaryOperator {
// // private:
// //     JobPtr jptr;

// // public:
// //     const JobPtr getJobPtr() const {return jptr; }
// //     typedef boost::shared_ptr<JobExpr> JobExprPtr;
// //     JobExpr() {} // default ctor
// //     JobExpr(JobPtr _jptr): jptr(_jptr) {}
// //     JobExpr(JobPtr _jptr, ExpressionPtr _chld): jptr(_jptr),
// //     UnaryOperator(_chld) { } virtual ~JobExpr() {} virtual
// //     vector<pair<double,int> > generate(SolverModelPtr m, int I,
// //             vector<map<double,vector<int> > > &partcap,
// //             vector<map<double,Constraint> >& capconmap,
// //             JobPtr jobptr);
// //     virtual tuple<double, double> eval(Allocation alloc)
// //         {return child->eval(alloc); }
// //     virtual string toString() {return child->toString();}
// // };

// // class MinExpression: public NnaryOperator {
// // private:
// //     double cached_minU; // decision variable cache -- owned by
// getResults()
// //     int minUvaridx;  // decision variable index -- owned by generate()
// //     // helper functions
// //     double upper_bound(vector<map<double,vector<int> > > &partcap);

// // public:
// //     typedef boost::shared_ptr<MinExpression> MinExprPtr;
// //     MinExpression(bool homogeneous_children_nodes = false)
// //             : NnaryOperator(homogeneous_children_nodes){ cached_minU = 0;
// }
// //     virtual ~MinExpression() {}
// //     virtual tuple<double, double> eval(Allocation alloc);
// //     virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
// // vector<map<double,vector<int>
// //                                                > > &partcap,
// // vector<map<double,Constraint>
// //                                                >& capconmap, JobPtr
// jobptr);
// //     virtual string toString();
// //     virtual Allocation getResults(const Solver &solver,
// //                                   vector<map<double, vector<int> > >
// //                                   &partcap, int start_time);
// //     virtual void cacheNodeResults(const Solver &solver,
// //                                   vector<map<double, vector<int> > >
// //                                   &partcap, int start_time);
// // };

// // class MaxExpression: public NnaryOperator {
// // private:
// //     map<ExpressionPtr, int> indvarmap;
// //     map<ExpressionPtr, int> cached_indvarmap;

// // public:
// //     typedef boost::shared_ptr<MaxExpression> MaxExprPtr;
// //     MaxExpression(bool homogeneous_children_nodes = false)
// //             : NnaryOperator(homogeneous_children_nodes){}
// //     virtual ~MaxExpression() {}

// //     ExpressionPtr removeChild(const ExpressionPtr &chld);
// //     virtual tuple<double, double> eval(Allocation alloc);
// //     //recursively generate a model for a solver; aggregate capacity
// //     constraints virtual vector<pair<double,int> > generate(SolverModelPtr
// m,
// //     int I,
// // vector<map<double,vector<int>
// //                                                > > &partcap,
// // vector<map<double,Constraint>
// //                                                >& capconmap, JobPtr
// jobptr);
// //     //friend ostream & operator<< (ostream &out, MaxExpression *objp);
// //     virtual string toString();
// //     virtual Allocation getResults(const Solver &solver,
// //                                   vector<map<double, vector<int> > >
// //                                   &partcap, int start_time);
// //     virtual void cacheNodeResults(const Solver &solver,
// //                                   vector<map<double, vector<int> > >
// //                                   &partcap, int start_time);
// // };

// // class SumExpression: public NnaryOperator {
// // private:
// //     //vector<int> indvaridx;
// //     map<ExpressionPtr, int> indvarmap;
// //     map<ExpressionPtr, int> cached_indvarmap;
// // public:
// //     typedef boost::shared_ptr<SumExpression> SumExprPtr;
// //     SumExpression(bool homogeneous_children_nodes = false)
// //             : NnaryOperator(homogeneous_children_nodes){}
// //     virtual ~SumExpression() {}

// //     ExpressionPtr removeChild(const ExpressionPtr &chld); // remove
// matching
// //     child ExpressionPtr removeChild(JobPtr jptr); // remove matching child
// //     void addChildIfNew(ExpressionPtr newchld); // adds chld only if not
// //     already present void addChildIfNew(JobPtr jptr, const
// //     std::function<ExpressionPtr(JobPtr)>& func); //adds a JobExpr child if
// //     not already present virtual tuple<double, double> eval(Allocation
// alloc);
// //     virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
// //             vector<map<double,vector<int> > > &partcap,
// //             vector<map<double,Constraint> >& capconmap,
// //             JobPtr jobptr);
// //     //friend ostream & operator<< (ostream &out, SumExpression *objp);
// //     virtual string toString();
// //     virtual Allocation getResults(const Solver &solver,
// //                                   vector<map<double, vector<int> > >
// //                                   &partcap, int start_time);
// //     virtual void cacheNodeResults(const Solver &solver,
// //                                   vector<map<double, vector<int> > >
// //                                   &partcap, int start_time);
// // };
// }  // namespace alsched

// #endif
