#ifndef _COMMON_HPP_
#define _COMMON_HPP_
#include <limits.h>

#include <boost/functional/hash.hpp>
#include <boost/shared_ptr.hpp>
#include <functional>
#include <unordered_set>
#include <vector>

#define TETRISCHED_DEBUG_ENABLED true

#define TETRISCHED_DEBUG(message)                   \
  if (TETRISCHED_DEBUG_ENABLED) {                   \
    std::cout << "DEBUG: " << message << std::endl; \
  };

using namespace std;

namespace alsched {

typedef enum { SIM_SOFT = 0, SIM_HARD = 1, SIM_NONE = 2, SIM_MAX } sim_t;

typedef enum {
  SOLVER_SEQ = 0,
  SOLVER_PBB = 1,
  SOLVER_UB = 2,
  SOLVER_GREEDY = 3,
  SOLVER_MAX
} solver_t;

typedef enum {
  EST_RUNTIME_ADJUST_INCREMENT = 0,
  EST_RUNTIME_ADJUST_MOR = 1,
  EST_RUNTIME_ADJUST_ORACLE = 2,
  EST_RUNTIME_ADJUST_MAX
} underestimate_adjust_policy_t;

typedef enum { BUDGET_JOB = 0, BUDGET_SPACETIME = 1, BUDGET_MAX } budget_t;

typedef enum {
  /// An `EXPRESSION_PRUNE` result type indicates that the parsed
  /// `Expression` will not provide any utility to the overall objective
  /// and as a result, should be pruned from the ExpressionTree.
  EXPRESSION_PRUNE = 0,
} ParseResultType;

/// A `ParseResult` is a struct that contains the relevant information
/// that an `Expression` bubbles up to help its parents execute their
/// parsing logic.
struct ParseResult {
  ParseResultType type;
  ParseResult(ParseResultType type) : type(type) {}
};

// struct config {
//     sim_t simtype;
//     string simtypestr;
//     solver_t solvertype;
//     budget_t budgettype;
//     double WebBudgetFactor;
//     double sched_horizon; // in seconds
//     bool enable_preemption;
//     underestimate_adjust_policy_t underestimate_adjust_policy;
//     double preemption_delay; // in seconds
//     vector<int> rack_cap;
//     int numMachinesWithGPU ;
//     int numMachinesWithHDFS ;
//     double secsPerAlschedTimeUnit; // numseconds/sched_decision_frequency
//     double MPIDesiredCompletionFactor;
//     double GPUDesiredCompletionFactor;
//     double HadoopDesiredCompletionFactor;
//     double NoneDesiredCompletionFactor;
//     const double HaddopMinKFactor = 0.5;
//     double WebDesiredQueueingFactor;
//     double WebDeadlineDurationFraction;
//     const double WebET_Desired = 0.25;
//     const double WebET_SLA = 0.5;
//     const double WebMu = 5.0;
//     double AvailDesiredQueueingFactor;
//     double AvailDeadlineDurationFraction;
//     vector<double> availFactor;
//     double availCalcMachineFailure ;
//     double availCalcRackFailure ;
//     double simduration; // in seconds
//     double duration_perturb_mean; // in seconds
//     double duration_perturb_std; // in seconds
//     json_spirit::mObject traces;
//     string expoutfile; // file for experimental output
//     double penalty_factor;
//     bool cache_enable;
//     bool oracle_enable;
//     string YARNaddr;
//     int YARNport;
//     int tetrischedPort;
// };

// extern struct config simconfig;

template <class T>
struct HashRange {
  size_t operator()(T const& c) const {
    return boost::hash_range(c.begin(), c.end());
  }
};

// Allocation struct encapsulates all allocation information per Job
struct Allocation {
  // a resource allocation is fundamentally defined by the space-time rectangle:
  // (a) space: set of nodes, (b) time: start and duration
  vector<int> nodes;
  map<int, int> p2c;  // partition to count mapping
  double start_time;  // start_time chosen by the solver
  double duration;    // duration chosen by the solver
  Allocation() {}     // default ctor
  Allocation(map<int, int> _p2c, double _st, double _d)
      : p2c(_p2c), start_time(_st), duration(_d) {}
  Allocation(const vector<int>& _nodes, double _st, double _d)
      : start_time(_st), duration(_d) {
    for (int i = 0; i < _nodes.size(); i++) nodes.push_back(_nodes[i]);
  }
};

const double sched_period = 1;

typedef function<double()> Distribution;
typedef function<int()> DistributionInt;

typedef function<double(double, double)> UtilVal;
typedef std::function<uint32_t(uint32_t, uint32_t)> UtilityFn;

class Timer;

class Expression;
typedef boost::shared_ptr<Expression> ExpressionPtr;

class Job;
typedef boost::shared_ptr<Job> JobPtr;
typedef function<void(JobPtr)> JobCompletionCallback;

class Event;
typedef boost::shared_ptr<Event> EventPtr;

class OpenEvent;
typedef boost::shared_ptr<OpenEvent> OpenEventPtr;
typedef function<JobPtr(OpenEventPtr)> OpenArrivalCallback;

class ClosedEvent;
typedef boost::shared_ptr<ClosedEvent> ClosedEventPtr;
typedef function<JobPtr(ClosedEventPtr)> ClosedArrivalCallback;

class TraceEvent;
typedef boost::shared_ptr<TraceEvent> TraceEventPtr;
typedef function<void(TraceEventPtr)> TraceArrivalCallback;

typedef function<void(JobPtr)> ArrivalCallback;

class ClusterManager;
class Scheduler;
class Solver;

/// A Task represents a single unit of work that can be scheduled.
class Task;
typedef std::shared_ptr<Task> TaskPtr;

/// A Worker represents a machine or a general collection of resources
/// that is allowed to be allocated to Jobs.
class Worker;  // Forward declaration for Worker in Worker.hpp
typedef std::shared_ptr<Worker> WorkerPtr;  // A shared pointer to a Worker.

/// A Partition represents a collection of Workers that are equivalent
/// to each other. A Task can specify that the scheduler can assign any
/// of the Workers in a Partition to it.
class Partition;
typedef std::shared_ptr<Partition> PartitionPtr;

class Partitions;

typedef boost::shared_ptr<vector<int>> NodesPtr;

typedef unordered_set<vector<int>, HashRange<vector<int>>> EquivClassSet;

}  // namespace alsched
#endif  // _COMMON_HPP_
