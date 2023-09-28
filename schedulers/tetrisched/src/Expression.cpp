#include "tetrisched/Expression.hpp"

#include <algorithm>

namespace tetrisched {

/* Method definitions for XOrVariableT. */
template <typename X>
XOrVariableT<X>::XOrVariableT(const X& value) : value(value) {}

template <typename X>
XOrVariableT<X>::XOrVariableT(const VariablePtr& variable) : value(variable) {}

template <typename X>
XOrVariableT<X>& XOrVariableT<X>::operator=(const X& newValue) {
  value = newValue;
  return *this;
}

template <typename X>
XOrVariableT<X>& XOrVariableT<X>::operator=(const VariablePtr& newValue) {
  value = newValue;
  return *this;
}

template <typename X>
X XOrVariableT<X>::resolve() const {
  // If the value is the provided type, then return it.
  if (std::holds_alternative<X>(value)) {
    return std::get<X>(value);
  } else if (std::holds_alternative<VariablePtr>(value)) {
    // If the value is a VariablePtr, then return the value of the variable.
    auto variable = std::get<VariablePtr>(value);
    auto variableValue = variable->getValue();
    if (!variableValue) {
      throw tetrisched::exceptions::ExpressionSolutionException(
          "No solution was found for the variable name: " +
          variable->getName());
    }
    return variableValue.value();
  } else {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "XOrVariableT was resolved with an invalid type.");
  }
}

template <typename X>
bool XOrVariableT<X>::isVariable() const {
  return std::holds_alternative<VariablePtr>(value);
}

template <typename X>
template <typename T>
T XOrVariableT<X>::get() const {
  return std::get<T>(value);
}

void CapacityConstraintMap::registerUsageAtTime(const Partition& partition,
                                                Time time,
                                                VariablePtr variable) {
  // Get or insert the Constraint corresponding to this partition and time.
  auto mapKey = std::make_pair(partition.getPartitionId(), time);
  if (capacityConstraints.find(mapKey) == capacityConstraints.end()) {
    capacityConstraints[mapKey] = std::make_unique<Constraint>(
        "CapacityConstraint_" + std::to_string(partition.getPartitionId()) +
            "_at_" + std::to_string(time),
        ConstraintType::CONSTR_LE, partition.size());
  }

  // Add the variable to the Constraint.
  capacityConstraints[mapKey]->addTerm(1, variable);
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
    capacityConstraints[mapKey] = std::make_unique<Constraint>(
        "CapacityConstraint_" + std::to_string(partition.getPartitionId()) +
            "_at_" + std::to_string(time),
        ConstraintType::CONSTR_LE, partition.size());
  }

  // Add the variable to the Constraint.
  capacityConstraints[mapKey]->addTerm(usage);
}

void CapacityConstraintMap::registerUsageForDuration(const Partition& partition,
                                                     Time startTime,
                                                     Time duration,
                                                     VariablePtr variable,
                                                     Time granularity) {
  for (Time time = startTime; time < startTime + duration;
       time += granularity) {
    registerUsageAtTime(partition, time, variable);
  }
}

void CapacityConstraintMap::registerUsageForDuration(const Partition& partition,
                                                     Time startTime,
                                                     Time duration,
                                                     uint32_t usage,
                                                     Time granularity) {
  for (Time time = startTime; time < startTime + duration;
       time += granularity) {
    registerUsageAtTime(partition, time, usage);
  }
}

SolutionResultPtr Expression::solve(SolverModelPtr solverModel) {
  // Check that the Expression was parsed before.
  if (!parsedResult) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression was not parsed before solve.");
  }

  // Construct the SolutionResult.
  SolutionResultPtr solutionResult = std::make_shared<SolutionResult>();
  switch (parsedResult->type) {
    case ParseResultType::EXPRESSION_PRUNE:
      solutionResult->type = SolutionResultType::EXPRESSION_PRUNE;
      return solutionResult;
    case ParseResultType::EXPRESSION_NO_UTILITY:
      solutionResult->type = SolutionResultType::EXPRESSION_NO_UTILITY;
      return solutionResult;
    case ParseResultType::EXPRESSION_UTILITY:
      solutionResult->type = SolutionResultType::EXPRESSION_UTILITY;
      break;
    default:
      throw tetrisched::exceptions::ExpressionSolutionException(
          "Expression was parsed with an invalid ParseResultType: " +
          std::to_string(static_cast<int>(parsedResult->type)));
  }

  // Retrieve the start, end times and the indicator from the SolverModel.
  if (!parsedResult->startTime) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression with a utility was parsed without a start time.");
  }
  solutionResult->startTime = parsedResult->startTime->resolve();

  if (!parsedResult->endTime) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression with a utility was parsed without an end time.");
  }
  solutionResult->endTime = parsedResult->endTime->resolve();

  if (!parsedResult->indicator) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression with a utility was parsed without an indicator.");
  }
  solutionResult->indicator = parsedResult->indicator->resolve();

  if (!parsedResult->utility) {
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Expression with a utility was parsed without a utility.");
  }
  solutionResult->utility = parsedResult->utility.value()->getValue();

  return solutionResult;
}

ChooseExpression::ChooseExpression(TaskPtr associatedTask,
                                   Partitions resourcePartitions,
                                   uint32_t numRequiredMachines, Time startTime,
                                   Time duration)
    : associatedTask(associatedTask),
      resourcePartitions(resourcePartitions),
      numRequiredMachines(numRequiredMachines),
      startTime(startTime),
      duration(duration),
      endTime(startTime + duration) {}

void ChooseExpression::addChild(ExpressionPtr child) {
  throw tetrisched::exceptions::ExpressionConstructionException(
      "ChooseExpression cannot have a child.");
}

size_t ChooseExpression::getNumChildren(){
  return 0;
}

ParseResultPtr ChooseExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMap& capacityConstraints, Time currentTime) {
  // Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  if (currentTime > startTime) {
    TETRISCHED_DEBUG("Pruning Choose expression for "
                     << associatedTask->getTaskName()
                     << " to be placed starting at time " << startTime
                     << " and ending at " << endTime
                     << " because it is in the past.");
    parsedResult->type = ParseResultType::EXPRESSION_PRUNE;
    return parsedResult;
  }
  TETRISCHED_DEBUG("Parsing Choose expression for "
                   << associatedTask->getTaskName()
                   << " to be placed starting at time " << startTime
                   << " and ending at " << endTime << ".");

  // Find the partitions that this Choose expression can be placed in.
  // This is the intersection of the Partitions that the Choose expression
  // was instantiated with and the Partitions that are available at the
  // time of the parsing.
  Partitions schedulablePartitions = resourcePartitions | availablePartitions;
  TETRISCHED_DEBUG("The Choose Expression for "
                   << associatedTask->getTaskName() << " will be limited to "
                   << schedulablePartitions.size() << " partitions.");
  if (schedulablePartitions.size() == 0) {
    // There are no schedulable partitions, this expression cannot be satisfied.
    // and should provide 0 utility.
    parsedResult->type = ParseResultType::EXPRESSION_NO_UTILITY;
    return parsedResult;
  }

  // This Choose expression needs to be passed to the Solver.
  // We generate an Indicator variable for the Choose expression signifying
  // if this expression was satisfied.
  VariablePtr isSatisfiedVar =
      std::make_shared<Variable>(VariableType::VAR_INDICATOR,
                                 associatedTask->getTaskName() + "_placed_at_" +
                                     std::to_string(startTime));
  ConstraintPtr fulfillsDemandConstraint = std::make_unique<Constraint>(
      associatedTask->getTaskName() + "_fulfills_demand_at_" +
          std::to_string(startTime),
      ConstraintType::CONSTR_EQ, 0);
  for (PartitionPtr& partition : schedulablePartitions.getPartitions()) {
    // For each partition, generate an integer that represents how many
    // resources were taken from this partition.
    VariablePtr allocationVar = std::make_shared<Variable>(
        VariableType::VAR_INTEGER,
        associatedTask->getTaskName() + "_using_partition_" +
            std::to_string(partition->getPartitionId()) + "_at_" +
            std::to_string(startTime),
        0,
        std::min(static_cast<uint32_t>(partition->size()),
                 numRequiredMachines));

    // Add the variable to the demand constraint.
    fulfillsDemandConstraint->addTerm(1, allocationVar);

    // Register this indicator with the capacity constraints that
    // are being bubbled up.
    capacityConstraints.registerUsageForDuration(*partition, startTime,
                                                 duration, allocationVar, 1);
  }
  // Ensure that if the Choose expression is satisfied, it fulfills the
  // demand for this expression. Pass the constraint to the model.
  fulfillsDemandConstraint->addTerm(-1 * numRequiredMachines, isSatisfiedVar);
  solverModel->addConstraint(std::move(fulfillsDemandConstraint));

  // Construct the Utility function for this Choose expression.
  auto utility =
      std::make_unique<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  utility->addTerm(1, isSatisfiedVar);

  // Construct the return value.
  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = startTime;
  parsedResult->endTime = endTime;
  parsedResult->indicator = isSatisfiedVar;
  parsedResult->utility = std::move(utility);
  return parsedResult;
}

ObjectiveExpression::ObjectiveExpression() {}

void ObjectiveExpression::addChild(ExpressionPtr child) {
  children.push_back(std::move(child));
}
size_t ObjectiveExpression::getNumChildren(){
  return children.size();
}

ParseResultPtr ObjectiveExpression::parse(
    SolverModelPtr solverModel, Partitions availablePartitions,
    CapacityConstraintMap& capacityConstraints, Time currentTime) {
  // Construct the overall utility of this expression.
  auto utility = std::make_unique<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);

  // Parse the children and collect the utiltiies.
  for (auto& child : children) {
    auto result = child->parse(solverModel, availablePartitions,
                               capacityConstraints, currentTime);
    if (result->type == ParseResultType::EXPRESSION_UTILITY) {
      utility->merge(*(result->utility.value()));
    }
  }

  // Add the utility to the SolverModel.
  solverModel->setObjectiveFunction(std::move(utility));

  return std::make_shared<ParseResult>(
      ParseResult{.type = ParseResultType::EXPRESSION_NO_UTILITY});
}

MinExpression::MinExpression(std::string name): expressionName(name) {}

size_t MinExpression::getNumChildren() { return children.size(); }

void MinExpression::addChild(ExpressionPtr child) {
  children.push_back(std::move(child));
}

ParseResultPtr MinExpression::parse(SolverModelPtr solverModel,
                                    Partitions availablePartitions,
                                    CapacityConstraintMap& capacityConstraints,
                                    Time currentTime) {
  /// Create and save the ParseResult.
  parsedResult = std::make_shared<ParseResult>();

  auto numChildren = this->getNumChildren();
  if (numChildren == 0){
    throw tetrisched::exceptions::ExpressionSolutionException(
        "Number of children should be >=1 for MIN");
  }
  // start time of MIN
  VariablePtr minStartTime = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, expressionName + "_MIN_START_TIME");
  solverModel->addVariable(minStartTime);

  // end time of MIN
  VariablePtr minEndTime = std::make_shared<Variable>(VariableType::VAR_INTEGER, expressionName + "_MIN_END_TIME");
  solverModel->addVariable(minEndTime);

  // Utility of MIN operator
  auto minUtility =
      std::make_unique<ObjectiveFunction>(ObjectiveType::OBJ_MAXIMIZE);
  VariablePtr minUtilityVariable = std::make_shared<Variable>(
      VariableType::VAR_INTEGER, expressionName + "_MIN_UTILITY_VARIABLE");
  solverModel->addVariable(minUtilityVariable);

  for (int i = 0; i < numChildren; i++) {
    auto childParsedResult = children[i]->parse(
        solverModel, availablePartitions, capacityConstraints, currentTime);
    ConstraintPtr minStartTimeConstraint = std::make_unique<Constraint>(
        expressionName + "_MIN_START_TIME_CONSTR_CHILD_" + std::to_string(i),
        ConstraintType::CONSTR_GE, 0); // minStartTime < childStartTime
    if (childParsedResult->startTime.has_value()) {
      auto childStartTime = childParsedResult->startTime.value();
      if (childStartTime.isVariable()) {
        minStartTimeConstraint->addTerm(1, childStartTime.get<VariablePtr>());
      }else {
        minStartTimeConstraint->addTerm(childStartTime.get<Time>());
      }
      minStartTimeConstraint->addTerm(-1, minStartTime);
      // Add the constraint to solver
      solverModel->addConstraint(std::move(minStartTimeConstraint));
    } else{
      throw tetrisched::exceptions::ExpressionSolutionException(
          "Start Time needed from child-" + std::to_string(i) + " for MIN. But not present!");
    }
    // constraint of end time: childEndTime <= minEndTime
    ConstraintPtr minEndTimeConstraint = std::make_unique<Constraint>(
        expressionName + "_MIN_END_TIME_CONSTR_CHILD_" + std::to_string(i),
        ConstraintType::CONSTR_LE, 0);
    if (childParsedResult->endTime.has_value()) {
      auto childEndTime = childParsedResult->endTime.value();
      if (childEndTime.isVariable()) {
        minEndTimeConstraint->addTerm(1, childEndTime.get<VariablePtr>());
      } else {
        minEndTimeConstraint->addTerm(childEndTime.get<Time>());
      }
      minEndTimeConstraint->addTerm(-1, minEndTime);
      // Add the constraint to solver
      solverModel->addConstraint(std::move(minEndTimeConstraint));
    } else{
      throw tetrisched::exceptions::ExpressionSolutionException(
          "End Time needed from child-" + std::to_string(i) + " for MIN. But not present!");
    }

    if (childParsedResult->utility.has_value()) {
      // child_utility - minUVar >= 0
      auto childUtilityConstr = childParsedResult->utility.value()->toConstraint(
          expressionName + "_MIN_UTILITY_CONSTRAINT_CHILD_" + std::to_string(i),
          ConstraintType::CONSTR_GE, 0);
      childUtilityConstr->addTerm(-1, minUtilityVariable);
      solverModel->addConstraint(std::move(childUtilityConstr));
    } else{
      throw tetrisched::exceptions::ExpressionSolutionException(
          "Utility needed from child-" + std::to_string(i) + " for MIN. But not present!");
    }
  }
  // MinU = Max(MinUVar)
  minUtility->addTerm(1, minUtilityVariable);

  parsedResult->type = ParseResultType::EXPRESSION_UTILITY;
  parsedResult->startTime = minStartTime;
  parsedResult->endTime = minEndTime;
  parsedResult->utility = std::move(minUtility);
  return parsedResult;
}

}  // namespace tetrisched

// // standard C/C++ libraries
// #include <algorithm>
// #include <cfloat>
// #include <climits>
// #include <iostream>
// #include <string>
// #include <tuple>
// #include <utility>
// #include <vector>
// // boost libraries
// #include <boost/make_shared.hpp>
// #include <boost/shared_array.hpp>
// // 3rd party libraries
// // alsched libraries
// #include "Expression.hpp"
// #include "Job.hpp"
// #include "Task.hpp"
// #include "Util.hpp"
// #include "common.hpp"

// namespace alsched {

// // return results for given start_time, caching ALL results and resolving
// nodes
// // Returns: Allocation object (a rectangle in space time) for the matching
// // start_time
// Allocation SchedulingExpression::getResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap,
//     int start_time) {
//   // we now have cached node results
//   if (start_time == this->_start_time) {
//     return Allocation(this->cached_nodes, start_time, this->_duration);
//   } else {
//     return Allocation(vector<int>(), start_time, this->_duration);
//   }
// }

// // owns cache : purges it and populates it with new results from the solver
// // resolves solver results to nodes for *this leaf and caches them
// // mutates partcap, removing from it nodes allocated to *this leaf
// void SchedulingExpression::cacheNodeResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap, int
//     st) {
//   // only cache for our start_time
//   if (this->_start_time != st) {
//     return;
//   }
//   if (marker) {
//     return;
//   }
//   const double sched_horizon = partcap[0].rbegin()->first + 1;
//   const double end_time =
//       std::min(this->_start_time + this->_duration, sched_horizon);

//   // purge any cached results
//   this->cached_nodes.clear();
//   // iterate over partitions; for each partition, query result for its
//   part.var. for (int pi = 0; pi < this->partitions.size(); pi++) {
//     int p = this->partitions[pi];
//     int vi = this->partvaridx[pi];
//     int cnt = static_cast<int>(solver.getResult(vi));
//     // grab a mutable ref to the time slice for nodes avail. for given p and
//     t vector<int> &nodesavail = partcap[p][this->_start_time]; if (cnt >
//     nodesavail.size()) {
//       cout << "[DEBUG][Choose::getResults]: "
//            << "cnt = " << cnt << "\t"
//            << "nodesavail size = " << nodesavail.size() << endl;
//       cout << this->debugString() << endl;
//     }
//     assert(cnt <= nodesavail.size());
//     for (int i = 0; i < cnt; i++) {
//       int node = nodesavail.back();
//       // remove this node from available nodes for the duration of this leaf
//       for (double t = this->_start_time; t < end_time; t += 1) {
//         vector<int> &nodesavail_future = partcap[p][t];
//         // from this vector ref, delete node <node>
//         vector<int>::iterator it =
//             find(nodesavail_future.begin(), nodesavail_future.end(), node);
//         assert(it != nodesavail_future.end());
//         nodesavail_future.erase(it);
//       }
//       this->cached_nodes.push_back(node);  // resolve
//     }
//   }
//   marker = true;
// }

// std::unique_ptr<ParseResult> Choose::parse(SolverModelPtr solverModel,
//                                            Partitions resourcePartitions,
//                                            uint32_t currentTime,
//                                            uint32_t schedulingHorizon) {
//   uint32_t endTime =
//       std::min(this->startTime + this->duration, schedulingHorizon);
//   TETRISCHED_DEBUG("Generating Choose expression for "
//                    << associatedTask->getTaskName()
//                    << " to be placed starting at time " << startTime
//                    << " and ending at " << endTime << ".");

//   // If the current time has moved past this expression's start time, prune
//   it. if (this->startTime < currentTime) {
//     TETRISCHED_DEBUG("Choose expression for " <<
//     associatedTask->getTaskName()
//                                               << " is in the past.
//                                               Pruning.");
//     return std::make_unique<ParseResult>(
//         ParseResult(ParseResultType::EXPRESSION_PRUNE));
//   }
// }

// vector<pair<double, int> > Choose::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   std::string jobName("UnnamedJob");
//   if (jobptr != nullptr) {
//     jobName = jobptr->getJobName();
//   }
//   TETRISCHED_DEBUG("Generating Choose expression for " << jobName);
//   vector<pair<double, int> > cterms;
//   vector<pair<double, int> > objf;
//   this->partvaridx.clear();  // purge partition variable indices

//   // sched_horizon is one past the last valid time in partcap
//   // note that map's are sorted
//   double curtime = partcap[0].begin()->first;
//   double sched_horizon = partcap[0].rbegin()->first + 1;
//   TETRISCHED_DEBUG("The Choose Expression for "
//                    << jobName << "will be limited from time " << curtime
//                    << " to " << sched_horizon << ".");
//   TETRISCHED_DEBUG("The Choose Expression's start time is " <<
//   this->_start_time
//                                                             << ".")
//   // if this branch's start_time is in the past, prune it
//   if ((this->_start_time < curtime) || (this->_start_time >= sched_horizon))
//     return vector<pair<double, int> >(1, pair<double, int>(0, I));

//   // end at min(start_time + duration, sched_horizon) to avoid out-of-bounds
//   in
//   // partcap
//   double end_time = min(this->_start_time + this->_duration, sched_horizon);
//   TETRISCHED_DEBUG("The Choose Expression's end time is " << end_time <<
//   ".");

//   // Calculate a factor to scale the utility value so that
//   // the space x time area beyond the scheduling horizon
//   // do not unfairly bias very long jobs
//   double utilFactor = (end_time - this->_start_time) / this->_duration;
//   int sumpc0 = 0;

//   // for each partition generate a partition variable P_p
//   TETRISCHED_DEBUG("The size of the partitions: " << this->partitions.size()
//                                                   << ".");
//   for (int pi = 0; pi < this->partitions.size(); pi++) {
//     TETRISCHED_DEBUG("Generating partition variable " << pi << ".");
//     int p = this->partitions[pi];
//     // 1. calculate P_p's initial value
//     const vector<int> &nodesavail = partcap[p][this->_start_time];

//     // TODO(atumanov): consider returning identity if no nodes are available
//     //  Intersect available nodes with cachednodes to get count. If no nodes
//     //  were previously cached => isection is empty => initval for P_p is 0
//     int pc0 = vec_isection_size(nodesavail, this->cached_nodes);
//     sumpc0 += pc0;

//     string vname = "P_" + jobName + "_time_" +
//                    std::to_string(this->_start_time) + "_" +
//                    std::to_string(p);
//     Variable partvar(VAR_INT, pair<double, double>(0, this->_k), p, vname,
//                      jobptr, this->_start_time, this->_duration,
//                      (double)pc0);
//     int vi = m->addVariable(partvar);
//     // save this vi
//     this->partvaridx.push_back(vi);  // partvar idx matches part. idx :
//     // note partition variable index order -- must match partition order

//     // aggregate demand constraint : sum(cterms) = k*I
//     cterms.push_back(pair<double, int>(1, vi));

//     // aggregate supply constraint: sum(terms) <= cap(p,t)
//     for (double t = this->_start_time; t < end_time; t += 1) {
//       if (capconmap[p].find(t) == capconmap[p].end()) {
//         capconmap[p][t] = Constraint(partcap[p][t].size(), OP_LE);
//       }
//       TETRISCHED_DEBUG("Adding constraint term for partition "
//                        << p << " at time " << t << ".");
//       capconmap[p][t].addConstraintTerm(pair<double, int>(1, vi));
//     }
//   }
//   cterms.push_back(
//       pair<double, int>(-this->_k, I));            // complete demand
//       constraint
//   m->addConstraint(Constraint(cterms, 0, OP_EQ));  // add demand constraint
//   // prepare objective function: u*I
//   objf.push_back(pair<double, int>(this->_utility * utilFactor, I));

//   return objf;
// }

// vector<pair<double, int> > LinearChoose::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   vector<pair<double, int> > cterms;
//   vector<pair<double, int> > objf;
//   this->partvaridx.clear();  // purge partition variable indices (owned by
//                              // gen())

//   // sched_horizon is one past the last valid time in partcap
//   // note that map's are sorted
//   double curtime = partcap[0].begin()->first;
//   double sched_horizon = partcap[0].rbegin()->first + 1;
//   // if this branch's start_time is in the future, prune it
//   if ((this->_start_time < curtime) || (this->_start_time >= sched_horizon))
//     return vector<pair<double, int> >(1, pair<double, int>(0, I));

//   // end at the min(start_time + duration, sched_horizon) so that we do not
//   go
//   // beyond range in partcap
//   double end_time = _start_time + _duration;
//   if (sched_horizon < end_time) {
//     end_time = sched_horizon;
//   }

//   // Calculate a factor to scale the utility value so that
//   // the space x time area beyond the scheduling horizon
//   // do not unfairly bias very long jobs
//   double utilFactor = (end_time - _start_time) / _duration;

//   // for each partition generate a partition variable P_p
//   for (int pi = 0; pi < this->partitions.size(); pi++) {
//     int p = this->partitions[pi];
//     const vector<int> &nodesavail = partcap[p][_start_time];
//     int pc0 = vec_isection_size(nodesavail, this->cached_nodes);
//     string vname = "P" + to_string(p);
//     Variable partvar(VAR_INT,
//                      pair<double, double>(
//                          0, std::min((int)partcap[p][_start_time].size(),
//                          _k)),
//                      p, vname, jobptr, _start_time, _duration, (double)pc0);
//     int vi = m->addVariable(partvar);
//     // save this vi
//     this->partvaridx.push_back(vi);  // match partition index

//     // aggregate objective function: Pi * u/k
//     objf.push_back(pair<double, int>(_utility / (1.0 * _k) * utilFactor,
//     vi));

//     // aggregate demand constraint : sum(cterms) <= k*I
//     cterms.push_back(pair<double, int>(1, vi));

//     // aggregate supply constraint: sum(terms) <= cap(p,t)
//     for (double t = _start_time; t < end_time; t += 1) {
//       if (capconmap[p].find(t) == capconmap[p].end()) {
//         // instantiate supply constraint for p,t
//         capconmap[p][t] = Constraint(partcap[p][t].size(), OP_LE);
//       }
//       capconmap[p][t].addConstraintTerm(pair<double, int>(1, vi));
//     }
//   }
//   cterms.push_back(pair<double, int>(-_k, I));     // complete demand
//   constraint m->addConstraint(Constraint(cterms, 0, OP_LE));  // add demand
//   constraint

//   return objf;
// }

// // owns cache : purges it and populates it with new results from the solver
// // resolves solver results to nodes for *this leaf and caches them
// // mutates partcap, adding it nodes preempted for *this leaf
// void PreemptingExpression::cacheNodeResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap, int
//     st) {
//   // only cache for our start_time
//   if (this->_start_time != st) {
//     return;
//   }
//   if (this->marker) {
//     return;
//   }
//   const double nodes_available_time = this->_start_time + this->_duration;
//   const double sched_horizon = partcap[0].rbegin()->first + 1;

//   // no-op if nodes will be available after sched_horizon
//   if (nodes_available_time > sched_horizon) {
//     return;
//   }

//   // purge any cached results
//   this->cached_nodes.clear();
//   // iterate over partitions; for each partition, query result for its
//   part.var.

//   // for each partition generate a partition variable P_p
//   for (int pi = 0; pi < this->partitions.size(); pi++) {
//     int p = this->partitions[pi];
//     int vi = this->partvaridx[pi];
//     auto &nodesavail = this->partition_nodes[p];
//     int cnt = static_cast<int>(solver.getResult(vi));
//     assert(cnt >= 0);
//     assert(cnt <= nodesavail.size());

//     for (int i = 0; i < cnt; i++) {
//       int m = nodesavail.back();
//       nodesavail.pop_back();
//       this->_nodes->erase(std::remove(_nodes->begin(), _nodes->end(), m),
//                           _nodes->end());
//       assert(m >= 0);
//       assert(-(m + 1) < 0);
//       this->cached_nodes.push_back(-(m + 1));
//     }
//   }
//   marker = true;
// }

// vector<pair<double, int> > KillChoose::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   vector<pair<double, int> > cterms;
//   vector<pair<double, int> > objf;
//   this->partvaridx.clear();  // purge partition variable indices

//   // sched_horizon is one past the last valid time in partcap
//   // note that map's are sorted
//   double curtime = partcap[0].begin()->first;
//   double sched_horizon = partcap[0].rbegin()->first + 1;
//   // if this branch's start_time is in the future, prune it
//   if (this->_start_time != curtime)
//     return vector<pair<double, int> >(1, pair<double, int>(0, I));

//   // end at min(start_time + duration, sched_horizon) to avoid out-of-bounds
//   in
//   // partcap
//   double end_time = _start_time + _duration;
//   if (sched_horizon < end_time) {
//     end_time = sched_horizon;
//   }

//   double preempt_start_time = _start_time + _duration;
//   double preempt_end_time =
//       min(jobptr->GetAlloc().start_time + jobptr->GetAlloc().duration,
//           sched_horizon);

//   // Calculate a factor to scale the utility value so that
//   // the space x time area beyond the scheduling horizon
//   // do not unfairly bias very long jobs
//   double utilFactor = (end_time - _start_time) / _duration;
//   int sumpc0 = 0;

//   // for each partition generate a partition variable P_p
//   for (auto &kv : this->partition_nodes) {
//     int p = kv.first;
//     const auto &nodesavail = kv.second;
//     /*
//         cout << "[DEBUG][Choose::generate]" << nodesavail.size() << " "
//              << this->cached_nodes.size() << " " << pc0 << endl;
//     */

//     string vname = "P" + to_string(p);
//     Variable partvar(VAR_INT, pair<double, double>(0, nodesavail.size()), p,
//                      vname, jobptr, _start_time, _duration, (double)0);
//     int vi = m->addVariable(partvar);
//     // save this vi
//     this->partvaridx.push_back(vi);  // partvar idx matches part. idx :
//     // note partition variable index order -- must match partition order

//     // aggregate demand constraint : sum(cterms) = k*I
//     cterms.push_back(pair<double, int>(1, vi));

//     // aggregate supply constraint: sum(terms) <= cap(p,t)
//     for (double t = preempt_start_time; t < preempt_end_time; t += 1) {
//       if (capconmap[p].find(t) == capconmap[p].end()) {
//         capconmap[p][t] = Constraint(partcap[p][t].size(), OP_LE);
//       }
//       capconmap[p][t].addConstraintTerm(pair<double, int>(-1, vi));
//     }
//   }
//   cterms.push_back(pair<double, int>(-_k, I));     // complete demand
//   constraint m->addConstraint(Constraint(cterms, 0, OP_EQ));  // add demand
//   constraint
//   // prepare objective function: u*I
//   objf.push_back(pair<double, int>(_utility * utilFactor, I));

//   return objf;
// }

// vector<pair<double, int> > KillLinearChoose::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   vector<pair<double, int> > cterms;
//   vector<pair<double, int> > objf;
//   this->partvaridx.clear();  // purge partition variable indices

//   // sched_horizon is one past the last valid time in partcap
//   // note that map's are sorted
//   double curtime = partcap[0].begin()->first;
//   double sched_horizon = partcap[0].rbegin()->first + 1;
//   // if this branch's start_time is in the future, prune it
//   if (_start_time != curtime)
//     return vector<pair<double, int> >(1, pair<double, int>(0, I));

//   // end at min(start_time + duration, sched_horizon) to avoid out-of-bounds
//   in
//   // partcap
//   double end_time = _start_time + _duration;
//   if (sched_horizon < end_time) {
//     end_time = sched_horizon;
//   }

//   double preempt_start_time = _start_time + _duration;
//   double preempt_end_time =
//       min(jobptr->GetAlloc().start_time + jobptr->GetAlloc().duration,
//           sched_horizon);

//   // Calculate a factor to scale the utility value so that
//   // the space x time area beyond the scheduling horizon
//   // do not unfairly bias very long jobs
//   double utilFactor = (end_time - _start_time) / _duration;
//   int sumpc0 = 0;

//   // for each partition generate a partition variable P_p
//   for (auto &kv : this->partition_nodes) {
//     int p = kv.first;
//     const auto &nodesavail = kv.second;
//     /*
//     cout << "[DEBUG][Choose::generate]" << nodesavail.size() << " "
//          << this->cached_nodes.size() << " " << pc0 << endl;
//     */

//     string vname = "P" + to_string(p);
//     Variable partvar(VAR_INT, pair<double, double>(0, nodesavail.size()), p,
//                      vname, jobptr, _start_time, _duration, (double)0);
//     int vi = m->addVariable(partvar);
//     // save this vi
//     this->partvaridx.push_back(vi);  // partvar idx matches part. idx :
//     // note partition variable index order -- must match partition order

//     // aggregate demand constraint : sum(cterms) = k*I
//     objf.push_back(pair<double, int>(_utility / (1.0 * _k) * utilFactor,
//     vi)); cterms.push_back(pair<double, int>(1, vi));

//     // aggregate supply constraint: sum(terms) <= cap(p,t)
//     for (double t = preempt_start_time; t < preempt_end_time; t += 1) {
//       if (capconmap[p].find(t) == capconmap[p].end()) {
//         capconmap[p][t] = Constraint(partcap[p][t].size(), OP_LE);
//       }
//       capconmap[p][t].addConstraintTerm(pair<double, int>(-1, vi));
//     }
//   }
//   cterms.push_back(pair<double, int>(-_k, I));     // complete demand
//   constraint m->addConstraint(Constraint(cterms, 0, OP_LE));  // add demand
//   constraint return objf;
// }

// vector<pair<double, int> > MaxExpression::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   vector<pair<double, int> > cterms;  // constraint: sum(Ii)<=I
//   vector<pair<double, int> > objf;
//   this->indvarmap.clear();  // reset decision variable indices

//   const int curtime = partcap[0].begin()->first;
//   const int sched_horizon = partcap[0].rbegin()->first + 1;

//   for (int i = 0; i < chldarr.size(); i++) {
//     const ExpressionPtr &c = chldarr[i];
//     const auto &startTimes = c->startTimeRange();

//     // if there is no intersection between scheduling range [curtime,
//     // sched_horizon) and startTimes, ignore children
//     if (min(sched_horizon, startTimes.second) <=
//         max(curtime, startTimes.first)) {
//       continue;
//     }

//     string vname = "maxI" + to_string(i);
//     Variable maxIvar(
//         VAR_BOOL, pair<double, double>(0, 1), -1, vname, jobptr, 0, 0,
//         (cached_indvarmap.count(c)) ? (double)(cached_indvarmap[c]) : 0);
//     int vi = m->addVariable(maxIvar);
//     indvarmap[c] = vi;

//     // aggregate indicator limit constraint: sum(I_i) <= I
//     cterms.push_back(pair<double, int>(1, vi));
//     // no supply constraint aggregation here --> only in leafs
//     vector<pair<double, int> > objfi =
//         c->generate(m, vi, partcap, capconmap, jobptr);
//     objf.insert(objf.end(), objfi.begin(), objfi.end());  // objf += objfi
//   }
//   // finalize the indicator constraint : sum(Ii) <= I
//   cterms.push_back(pair<double, int>(-1, I));
//   m->addConstraint(Constraint(cterms, 0, OP_LE));

//   return objf;
// }

// // merges allocations from children, returns allocation matching start_time
// Allocation MaxExpression::getResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap,
//     int start_time) {
//   // iterate over cached indicator variables, return alloc for one enabled
//   child for (const auto &p : this->cached_indvarmap) {
//     ExpressionPtr expr = p.first;
//     int val = p.second;
//     if (val) return expr->getResults(solver, partcap, start_time);
//   }

//   // no enabled children
//   return Allocation(vector<int>(), 0, 0);
// }

// void MaxExpression::cacheNodeResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap, int
//     st) {
//   for (const auto &p : this->indvarmap) {
//     const ExpressionPtr &chld = p.first;
//     chld->cacheNodeResults(solver, partcap, st);
//   }

//   if (marker) return;

//   bool childselected = false;
//   this->cached_indvarmap.clear();

//   for (const auto &p : this->indvarmap) {
//     const ExpressionPtr &chld = p.first;
//     int vi = p.second;
//     bool val = solver.getResult(vi);
//     this->cached_indvarmap[chld] = val;
//     // assert that at most one child is selected
//     assert(!(childselected && val));
//     if (val) childselected = true;
//   }
//   marker = true;
// }

// // Given partition capacities across all partitions and time slices,
// // return the upper bound on utility at subtree rooted in *this
// double MinExpression::upper_bound(vector<map<double, vector<int> > >
// &partcap) {
//   double max_ub = 0;  // maximum upper bound across all time slices (r.v.)

//   double curtime = partcap[0].begin()->first;
//   double sched_horizon = partcap[0].rbegin()->first + 1;
//   for (double t = curtime; t < sched_horizon; t += 1) {
//     double ub4t = DBL_MAX;  // minimal upper bound for time t
//     // for each time slice get all available nodes and eval
//     // Iterate over all partitions to get all the nodes for a given time t
//     vector<int> nodesavail;
//     for (int p = 0; p < partcap.size(); p++) {
//       const vector<int> &nodesavailpt = partcap[p][t];
//       nodesavail.insert(nodesavail.end(), nodesavailpt.begin(),
//                         nodesavailpt.end());
//     }
//     const Allocation fullcapalloc(nodesavail, t, DBL_MAX);
//     for (auto c : chldarr) {
//       double child_ub = get<0>(c->eval(fullcapalloc));
//       ub4t = std::min(child_ub, ub4t);  // we will pick the smallest child
//       util
//     }
//     max_ub = std::max(max_ub, ub4t);  // need the maximum smallest child util
//   }
//   return max_ub;
// }

// vector<pair<double, int> > MinExpression::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   // ub: what's the maximum possible utility of the subtree rooted in *this
//   // for given start_time what's the full capacity of the cluster
//   // TODO(atumanov): for each time slice, eval *this on union of nodes from
//   all
//   // p
//   //                 and pick max util
//   this->minUvaridx = -1;  // reset decision variable index
//   double ub = DBL_MAX;
//   // double ub = upper_bound(partcap);
//   assert(ub >= 0);
//   // U \in range [0; ub]
//   Variable minUvar(VAR_FLOAT, pair<double, double>(0, ub), -1, "minU",
//   jobptr,
//                    0, 0,
//                    cached_minU);  // use cached value as initial value
//   int vU = m->addVariable(minUvar);
//   this->minUvaridx = vU;  // save this variable
//   for (const auto &c : this->chldarr) {
//     vector<pair<double, int> > objfi =
//         c->generate(m, I, partcap, capconmap, jobptr);
//     objfi.push_back(pair<double, int>(-1, vU));  // objf - 1* U >= 0
//     m->addConstraint(Constraint(objfi, 0, OP_GE));
//   }
//   return vector<pair<double, int> >(1, pair<double, int>(1, vU));  // objf =
//   U
// }

// Allocation MinExpression::getResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap,
//     int start_time) {
//   // recursively aggregate allocations from all children
//   // if the list of nodes is non-empty, assert that the start_times are the
//   same Allocation allocres(vector<int>(), start_time, 0);  // r.v. for (auto
//   c : this->chldarr) {
//     Allocation childalloc = c->getResults(solver, partcap, start_time);
//     if (childalloc.nodes.empty()) continue;
//     // got a non-empty allocation from child c for start_time
//     assert(allocres.start_time == childalloc.start_time);
//     // now merge the list of nodes from child into result alloc
//     allocres.nodes.insert(allocres.nodes.end(), childalloc.nodes.begin(),
//                           childalloc.nodes.end());
//     // duration -- if already set -- match it
//     if (allocres.duration > 0)
//       assert(childalloc.duration == allocres.duration);
//     else
//       allocres.duration = childalloc.duration;
//   }
//   return allocres;
// }

// void MinExpression::cacheNodeResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap, int
//     st) {
//   for (const auto &chld : this->chldarr) {
//     chld->cacheNodeResults(solver, partcap, st);
//   }
//   if (marker) return;

//   this->cached_minU = solver.getResult(this->minUvaridx);
//   marker = true;
// }

// vector<pair<double, int> > SumExpression::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   vector<pair<double, int> > cterms;
//   vector<pair<double, int> > objf;
//   indvarmap.clear();

//   const int curtime = partcap[0].begin()->first;
//   const int sched_horizon = partcap[0].rbegin()->first + 1;

//   for (int i = 0; i < chldarr.size(); i++) {
//     const auto &c = chldarr[i];
//     const auto &startTimes = c->startTimeRange();

//     // if there is no intersection between scheduling range [curtime,
//     // sched_horizon) and startTimes, ignore children
//     if (min(sched_horizon, startTimes.second) <=
//         max(curtime, startTimes.first)) {
//       continue;
//     }

//     string vname = "sumI" + to_string(i);
//     Variable sumIvar(
//         VAR_BOOL, pair<double, double>(0, 1), -1, vname, jobptr, 0, 0,
//         (cached_indvarmap.count(c)) ? (double)(cached_indvarmap[c]) : 0);
//     int vi = m->addVariable(sumIvar);
//     indvarmap[c] = vi;
//     // aggregate indicator constraint
//     cterms.push_back(pair<double, int>(1, vi));
//     vector<pair<double, int> > objfi =
//         c->generate(m, vi, partcap, capconmap, jobptr);
//     objf.insert(objf.end(), objfi.begin(), objfi.end());  // objf += objfi
//   }
//   // complete the indicator constraint
//   cterms.push_back(pair<double, int>(-1.0 * this->chldarr.size(), I));
//   Constraint c(cterms, 0, OP_LE);
//   // m->addConstraint(Constraint(cterms, 0, OP_LE));
//   // cout << "[sum ] adding constraint :" << c.toString() << endl;
//   m->addConstraint(c);

//   return objf;
// }

// // Should be similar to MinExpr -- aggregate allocation results from all
// // children for the given start_time. Cache results for all child indicator
// // variables.
// Allocation SumExpression::getResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap,
//     int start_time) {
//   Allocation allocres(vector<int>(), start_time, 0);  // empty result alloc
//   // iterate over cache, recursively aggregate results for all enabled
//   children for (const auto &p : cached_indvarmap) {
//     const ExpressionPtr &expr = p.first;
//     auto val = p.second;
//     if (val) {
//       // if the child was selected -- get its allocation
//       Allocation childalloc = expr->getResults(solver, partcap, start_time);
//       if (childalloc.nodes.empty()) continue;
//       // got a non-empty alloc from child i for start_time -> merge it
//       allocres.nodes.insert(allocres.nodes.end(), childalloc.nodes.begin(),
//                             childalloc.nodes.end());
//       // check the start_time match
//       assert(allocres.start_time == childalloc.start_time);
//       // duration -- if already set -- match it
//       if (allocres.duration > 0) {
//         assert(childalloc.duration == allocres.duration);
//       } else {
//         allocres.duration = childalloc.duration;
//       }
//     }
//   }
//   return allocres;
// }

// void SumExpression::cacheNodeResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap, int
//     st) {
//   for (const auto &p : this->indvarmap) {
//     const ExpressionPtr &chld = p.first;
//     chld->cacheNodeResults(solver, partcap, st);
//   }

//   if (marker) return;

//   this->cached_indvarmap.clear();  // purge cache: cacheNodeResults owns it
//   // cache results for all indicator variables
//   for (const auto &p : this->indvarmap) {
//     const ExpressionPtr &chld = p.first;
//     int vi = p.second;                // get indicator var. idx for child i
//     bool val = solver.getResult(vi);  // extract result for this decision
//     var. this->cached_indvarmap[chld] = val;
//   }
//   marker = true;
// }

// Allocation UnaryOperator::getResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap,
//     int start_time) {
//   return child->getResults(solver, partcap, start_time);
// }

// void UnaryOperator::cacheNodeResults(
//     const Solver &solver, vector<map<double, vector<int> > > &partcap, int
//     st) {
//   this->child->cacheNodeResults(solver, partcap, st);
// }

// vector<pair<double, int> > ScaleExpr::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   // scale objf of the child by this->factor and return
//   vector<pair<double, int> > objf =
//       this->child->generate(m, I, partcap, capconmap, jobptr);
//   for (auto &term : objf) {
//     // scale the term by factor
//     term.first *= factor;
//   }
//   return objf;
// }

// vector<pair<double, int> > BarrierExpr::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   vector<pair<double, int> > objf =
//       this->child->generate(m, I, partcap, capconmap, jobptr);

//   // sched_horizon is one past the last valid time in partcap
//   // note that map's are sorted
//   double curtime = partcap[0].begin()->first;
//   double sched_horizon = partcap[0].rbegin()->first + 1;
//   // if this branch's start_time is in the past, prune it
//   if ((start_time < curtime) || (start_time >= sched_horizon))
//     return vector<pair<double, int> >(1, pair<double, int>(0, I));

//   // end at min(start_time + duration, sched_horizon) to avoid out-of-bounds
//   in
//   // partcap
//   double end_time = start_time + duration;
//   if (sched_horizon < end_time) {
//     end_time = sched_horizon;
//   }

//   // Calculate a factor to scale the barrier so that
//   // the space x time area beyond the scheduling horizon
//   // do not unfairly bias very long jobs
//   double utilFactor = (end_time - start_time) / duration;

//   // add constraint: f >= barrier*I (if selected, utility must exceed barrier
//   // f - barrier*I >= 0
//   objf.push_back(pair<double, int>(-1.0 * barrier * utilFactor,
//                                    I));  // add constraint term
//   m->addConstraint(Constraint(objf, 0, OP_GE));
//   return vector<pair<double, int> >(1,
//                                     pair<double, int>(barrier * utilFactor,
//                                     I));
// }

// // sole purpose of JobExpr is to pass the job pointer further down the expr
// tree
// // TODO(atumanov): do we still need jobptr?
// vector<pair<double, int> > JobExpr::generate(
//     SolverModelPtr m, int I, vector<map<double, vector<int> > > &partcap,
//     vector<map<double, Constraint> > &capconmap, JobPtr jobptr) {
//   // pass through the jobptr
//   return child->generate(m, I, partcap, capconmap, this->jptr);
// }

// given a particular assignment of nodes from partitions, what's my utility
// INPUT: assigned[i] == # of nodes assigned from partition i
// aggregate node counts from all partitions in our eq.class only
// return utilval if the aggregate node count is >=k
// duration of 0 indicates branch wasn't chosen
// tuple<double, double> Choose::eval(Allocation alloc) {
//   // Ensure we have the right time range
//   if (alloc.start_time != _start_time) return make_tuple(0.0, 0.0);

//   // Calcualte number of nodes allocated
//   int nodecnt = 0;
//   for (int nodeAlloc : alloc.nodes) {
//     for (int nodeDesired : *(this->_nodes)) {
//       if (nodeAlloc == nodeDesired) {
//         nodecnt++;
//         break;
//       }
//     }
//   }
//   if (nodecnt >= _k) return make_tuple(_utility, _duration);

//   return make_tuple(0.0, 0.0);
// }

// string Choose::toString() {
//   string result;
//   result += "nCk({";
//   for (int i = 0; i < partitions.size(); i++) {
//     result += to_string(partitions[i]) + " ";
//   }
//   result += "}," + to_string(_k) + "," + to_string(_utility) + ")";
//   return result;
// }
// string SchedulingExpression::debugString() const {
//   string result;
//   result += "nCk(m:" + to_string(marker) + ",";
//   // print nodes
//   result += "{n:";
//   for (int i = 0; i < _nodes->size(); i++) {
//     result += to_string((*_nodes)[i]) + " ";
//   }
//   result += "},{p:";
//   // print partitions
//   for (int i = 0; i < partitions.size(); i++) {
//     result += to_string(partitions[i]) + " ";
//   }
//   result += "},{c:";
//   // print cached nodes, if any
//   for (int i = 0; i < cached_nodes.size(); i++) {
//     result += to_string(cached_nodes[i]) + " ";
//   }
//   result += "}," + to_string(_k) + "," + to_string(_utility);
//   // add time bits : start_time and duration
//   result += ", s:" + to_string(_start_time) + ", d:" + to_string(_duration);
//   result += ")";
//   return result;
// }

// // given a particular assignment of nodes from partitions, what's my utility
// // INPUT: assigned[i] == # of nodes assigned from partition i
// // aggregate node counts from all partitions in our eq.class only
// // return utilval if the aggregate node count is =k
// // duration of 0 indicates branch wasn't chosen
// tuple<double, double> KillChoose::eval(Allocation alloc) {
//   // Calcualte number of nodes allocated
//   int nodecnt = 0;
//   for (int nodeAlloc : alloc.nodes) {
//     for (int nodeDesired : *(this->_nodes)) {
//       if (nodeAlloc == -(nodeDesired + 1)) {
//         nodecnt++;
//         break;
//       }
//     }
//   }
//   if (nodecnt = _k) return make_tuple(_utility, _duration);

//   return make_tuple(0.0, 0.0);
// }

// string KillChoose::toString() {
//   string result;
//   result += "KnCk({";
//   for (int i = 0; i < partitions.size(); i++) {
//     result += to_string(partitions[i]) + " ";
//   }
//   result += "}," + to_string(_k) + "," + to_string(_utility) + ")";
//   return result;
// }

// // evaluates LnCk given assignment of nodes from partitions
// // INPUT: assigned[i] == # of nodes assigned from partition i
// // iterate over assigned, aggregating node count from partitions of interest
// // duration of 0 indicates branch wasn't chosen
// tuple<double, double> LinearChoose::eval(Allocation alloc) {
//   // Ensure we have the right time range
//   if (alloc.start_time != _start_time) return make_tuple(0.0, 0.0);

//   // Calcualte number of nodes allocated
//   int nodecnt = 0;
//   for (int nodeAlloc : alloc.nodes) {
//     for (int nodeDesired : *(this->_nodes)) {
//       if (nodeAlloc == nodeDesired) {
//         nodecnt++;
//         break;
//       }
//     }
//   }
//   assert(nodecnt <= _k);
//   double ratio = static_cast<double>(nodecnt) / _k;
//   return make_tuple(ratio * _utility, _duration);
// }

// string LinearChoose::toString() {
//   string result;
//   result += "LnCk({";
//   for (int i = 0; i < partitions.size(); i++) {
//     result += to_string(partitions[i]) + " ";
//   }
//   result += "}," + to_string(_k) + "," + to_string(_utility) + ")";
//   return result;
// }

// // given a particular assignment of nodes from partitions, what's my utility
// // INPUT: assigned[i] == # of nodes assigned from partition i
// // aggregate node counts from all partitions in our eq.class only
// // return utilval if the aggregate node count is >=k
// // duration of 0 indicates branch wasn't chosen
// tuple<double, double> KillLinearChoose::eval(Allocation alloc) {
//   // Ensure we have the right time range
//   if (alloc.start_time != _start_time) return make_tuple(0.0, 0.0);

//   // Calcualte number of nodes allocated
//   int nodecnt = 0;
//   for (int nodeAlloc : alloc.nodes) {
//     for (int nodeDesired : *(this->_nodes)) {
//       if (nodeAlloc == -(nodeDesired + 1)) {
//         nodecnt++;
//         break;
//       }
//     }
//   }
//   assert(nodecnt <= _k);
//   double ratio = static_cast<double>(nodecnt) / _k;
//   return make_tuple(ratio * _utility, _duration);
// }

// string KillLinearChoose::toString() {
//   string result;
//   result += "KLnCk({";
//   for (int i = 0; i < partitions.size(); i++) {
//     result += to_string(partitions[i]) + " ";
//   }
//   result += "}," + to_string(_k) + "," + to_string(_utility) + ")";
//   return result;
// }

// // duration of 0 indicates branch wasn't chosen
// tuple<double, double> MinExpression::eval(Allocation alloc) {
//   double minutilrv = DBL_MAX;
//   double maxduration = 0;
//   if (chldarr.empty()) return make_tuple(0.0, 0.0);

//   for (auto &chld : chldarr) {
//     double uv;
//     double duration;
//     tie(uv, duration) = chld->eval(alloc);
//     if (duration <= 0) return make_tuple(0.0, 0.0);
//     if (uv < minutilrv) minutilrv = uv;
//     if (duration > maxduration) maxduration = duration;
//   }
//   return make_tuple(minutilrv, maxduration);
// }

// string MinExpression::toString() {
//   string out;
//   out += "min(";
//   for (const auto &chld : chldarr) {
//     out += chld->toString() + ",";
//   }
//   out += ")";
//   return out;
// }

// // duration of 0 indicates branch wasn't chosen
// tuple<double, double> MaxExpression::eval(Allocation alloc) {
//   double maxutilrv = -1;
//   double maxutilduration = 0;
//   for (auto &chld : chldarr) {
//     double uv;
//     double duration;
//     tie(uv, duration) = chld->eval(alloc);
//     if (duration > 0) {
//       if (uv > maxutilrv) {
//         maxutilrv = uv;
//         maxutilduration = duration;
//       } else if ((uv == maxutilrv) && (duration < maxutilduration)) {
//         maxutilduration = duration;
//       }
//     }
//   }
//   if (maxutilrv < 0)  // Check to see if maxutilrv was set
//     return make_tuple(0.0, 0.0);
//   else
//     return make_tuple(maxutilrv, maxutilduration);
// }

// string MaxExpression::toString() {
//   string result;
//   result += "max(";
//   for (const auto &chld : this->chldarr) {
//     result += chld->toString() + ",";
//   }
//   result += ")";
//   return result;
// }

// void NnaryOperator::clearMarkers() {
//   marker = false;
//   for (auto &chld : chldarr) {
//     chld->clearMarkers();
//   }
// }

// pair<int, int> NnaryOperator::startTimeRange() {
//   if (cache_dirty) {
//     int start = INT_MAX, end = INT_MIN;
//     for (auto &chld : chldarr) {
//       const pair<int, int> &&chldValidRange = chld->startTimeRange();
//       start = min(start, chldValidRange.first);
//       end = max(end, chldValidRange.second);
//     }
//     this->startTimeRangeCache.first = start;
//     this->startTimeRangeCache.second = end;
//     cache_dirty = false;
//   }
//   return this->startTimeRangeCache;  // valid cache
// }

// // INPUT: assigned[i] == # nodes assigned from partition i
// // duration of 0 indicates branch wasn't chosen
// tuple<double, double> SumExpression::eval(Allocation alloc) {
//   double sum = 0;
//   double maxduration = 0;
//   for (auto &chld : chldarr) {
//     double uv;
//     double duration;
//     tie(uv, duration) = chld->eval(alloc);
//     if (duration > 0) {
//       sum += uv;
//       if (duration > maxduration) {
//         maxduration = duration;
//       }
//     }
//   }
//   return make_tuple(sum, maxduration);
// }

// string SumExpression::toString() {
//   string out;
//   out += "sum(";
//   for (const auto &chld : this->chldarr) {
//     out += chld->toString() + ",";
//   }
//   out += ")";
//   return out;
// }

// void NnaryOperator::addChild(ExpressionPtr newchld) {
//   chldarr.push_back(newchld);
//   if (!cache_dirty) {
//     // update startTimeRangeCache
//     pair<int, int> startTimeRange = newchld->startTimeRange();
//     this->startTimeRangeCache.first =
//         min(this->startTimeRangeCache.first, startTimeRange.first);
//     this->startTimeRangeCache.second =
//         max(this->startTimeRangeCache.second, startTimeRange.second);
//   }
// }
// // specialized add new child method for JobExpr children
// void SumExpression::addChildIfNew(
//     JobPtr newjptr, const std::function<ExpressionPtr(JobPtr)> &func) {
//   for (auto chld : this->chldarr) {
//     if (JobExpr::JobExprPtr jexpr =
//             boost::dynamic_pointer_cast<JobExpr>(chld)) {
//       const JobPtr jptr = jexpr->getJobPtr();
//       if (jptr == newjptr) {
//         assert(jptr->GetSchedExpression() == newjptr->GetSchedExpression());
//         return;  // match found, do not add
//       }
//     }
//   }
//   // matching child not found --> add
//   JobExpr::JobExprPtr jexpr = boost::make_shared<JobExpr>(newjptr);
//   jexpr->addChild(func(newjptr));
//   this->addChild(jexpr);
// }

// void SumExpression::addChildIfNew(ExpressionPtr newexprptr) {
//   for (auto chld : this->chldarr)
//     if (chld == newexprptr) {
//       return;
//     }

//   this->addChild(newexprptr);
// }
// ExpressionPtr NnaryOperator::removeChild(const ExpressionPtr &chld) {
//   ExpressionPtr rv = nullptr;

//   for (int i = 0; i < this->chldarr.size(); i++) {
//     if (this->chldarr[i] == chld) {
//       cout << "[removeChild]: removing child at position " << i << endl;
//       // found a matching child
//       rv = chldarr[i];
//       chldarr.erase(chldarr.begin() + i);
//       cache_dirty = true;
//       break;  // assuming no duplicates
//     }
//   }
//   return rv;
// }
// // Removes an element in child array, indicator variable array, and cache
// array
// // that corresponds to specified expression <chld>
// ExpressionPtr SumExpression::removeChild(const ExpressionPtr &chld) {
//   ExpressionPtr rv = NnaryOperator::removeChild(chld);
//   if (rv) {
//     indvarmap.erase(rv);
//     cached_indvarmap.erase(rv);
//   }
//   return rv;
// }

// // Removes an element in child array, indicator variable array, and cache
// array
// // that corresponds to specified expression <chld>
// ExpressionPtr MaxExpression::removeChild(const ExpressionPtr &chld) {
//   ExpressionPtr rv = NnaryOperator::removeChild(chld);
//   if (rv) {
//     indvarmap.erase(rv);
//     cached_indvarmap.erase(rv);
//   }
//   return rv;
// }

// ExpressionPtr SumExpression::removeChild(JobPtr newjptr) {
//   for (int i = 0; i < this->chldarr.size(); i++) {
//     if (JobExpr::JobExprPtr jexpr =
//             boost::dynamic_pointer_cast<JobExpr>(chldarr[i])) {
//       const JobPtr jptr = jexpr->getJobPtr();
//       if (jptr == newjptr) {
//         NnaryOperator::removeChild(jexpr);
//         cout << "[removeChild]: removing child at position " << i << endl;
//         // found a matching child;
//         indvarmap.erase(jexpr);
//         cached_indvarmap.erase(jexpr);
//         return jexpr;  // assuming no duplicates
//       }
//     }
//   }
//   return nullptr;
// }

// // Barrier class implementation
// BarrierExpr::BarrierExpr(double _barrier, double _start_time,
//                          double _duration) {
//   barrier = _barrier;
//   start_time = _start_time;
//   duration = _duration;
// }

// BarrierExpr::BarrierExpr(double _barrier, double _start_time, double
// _duration,
//                          ExpressionPtr _chld) {
//   barrier = _barrier;
//   start_time = _start_time;
//   duration = _duration;
//   child = _chld;
// }
// string BarrierExpr::toString() {
//   string out;
//   out = "bar(" + to_string(this->barrier) + "," + this->child->toString() +
//   ")"; return out;
// }

// void UnaryOperator::addChild(ExpressionPtr _chld) {
//   // this implementation only accepts one child, hence replace
//   child = _chld;
// }

// void UnaryOperator::clearMarkers() {
//   marker = false;
//   child->clearMarkers();
// }

// // duration of 0 indicates branch wasn't chosen
// tuple<double, double> BarrierExpr::eval(Allocation alloc) {
//   // barrier function is a step function that evaluates to u = barrier
//   double uv;
//   double duration;
//   tie(uv, duration) = child->eval(alloc);
//   if (uv >= barrier) return make_tuple(barrier, duration);
//   return make_tuple(0.0, 0.0);
// }

// ScaleExpr::ScaleExpr(double _factor) { factor = _factor; }

// ScaleExpr::ScaleExpr(double _factor, ExpressionPtr _chld) {
//   factor = _factor;
//   child = _chld;
// }
// string ScaleExpr::toString() {
//   string out;
//   out =
//       "scale(" + to_string(this->factor) + "," + this->child->toString() +
//       ")";
//   return out;
// }

// // duration of 0 indicates branch wasn't chosen
// tuple<double, double> ScaleExpr::eval(Allocation alloc) {
//   double uv;
//   double duration;
//   tie(uv, duration) = child->eval(alloc);
//   return make_tuple(factor * uv, duration);
// }

// typedef vector<bool> BitVector;
// typedef boost::shared_array<BitVector> BitVectors;

// Get all equivalence classes in expression tree
// void SchedulingExpression::getEquivClasses(EquivClassSet &equivClasses) {
//   if (marker) {
//     return;
//   }
//   marker = true;
//   equivClasses.insert(*_nodes);
// }

// void UnaryOperator::getEquivClasses(EquivClassSet &equivClasses) {
//   if (marker) {
//     return;
//   }
//   marker = true;
//   child->getEquivClasses(equivClasses);
// }

// void NnaryOperator::getEquivClasses(EquivClassSet &equivClasses) {
//   if (marker) {
//     return;
//   }
//   marker = true;
//   if (homogeneous_children_nodes) {
//     // optimize: we know children for each time t is identical.
//     const auto &startTime = chldarr.front()->startTimeRange();
//     for (const auto &child : chldarr) {
//       if (child->startTimeRange() == startTime) {
//         child->getEquivClasses(equivClasses);
//       }
//     }
//   } else {
//     for (const auto &child : chldarr) {
//       child->getEquivClasses(equivClasses);
//     }
//   }
// }

// Fills in partitions
// void SchedulingExpression::populatePartitions(const vector<int> &node2part,
//                                               int curtime, int sched_horizon)
//                                               {
//   if (marker) {
//     return;
//   }
//   marker = true;
//   // Calcualte set of partitions
//   unordered_set<int> partitionSet;
//   for (int id : *_nodes) {
//     partitionSet.insert(node2part[id]);
//   }
//   // Return partitions
//   partitions.assign(partitionSet.begin(), partitionSet.end());
// }

// void PreemptingExpression::populatePartitions(const vector<int> &node2part,
//                                               int curtime, int sched_horizon)
//                                               {
//   if (marker) {
//     return;
//   }
//   SchedulingExpression::populatePartitions(node2part, curtime,
//   sched_horizon); this->partition_nodes.clear();
//   // Calcualte set of partitions
//   for (int id : *_nodes) {
//     int partition = node2part[id];
//     this->partition_nodes[partition].push_back(id);
//   }
//   assert(this->partition_nodes.size() == this->partitions.size());
// }

// void UnaryOperator::populatePartitions(const vector<int> &node2part,
//                                        int curtime, int sched_horizon) {
//   if (marker) {
//     return;
//   }
//   marker = true;
//   child->populatePartitions(node2part, curtime, sched_horizon);
// }

// void NnaryOperator::populatePartitions(const vector<int> &node2part,
//                                        int curtime, int sched_horizon) {
//   if (marker) {
//     return;
//   }
//   marker = true;
//   for (const auto &child : chldarr) {
//     const auto &startTimes = child->startTimeRange();

//     // if there is no intersection between scheduling range [curtime,
//     // curtime+sched_horizon) and startTimes, ignore children
//     if (min(curtime + sched_horizon, startTimes.second) <=
//         max(curtime, startTimes.first)) {
//       continue;
//     }
//     child->populatePartitions(node2part, curtime, sched_horizon);
//   }
// }

// }  // namespace alsched
