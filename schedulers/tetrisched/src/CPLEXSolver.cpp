#include "tetrisched/CPLEXSolver.hpp"

#include "tetrisched/Types.hpp"

namespace tetrisched {

CPLEXSolver::CPLEXSolver()
    : cplexEnv(IloEnv()),
      solverModel(nullptr),
      cplexInstance(IloCplex(cplexEnv)) {}

SolverModelPtr CPLEXSolver::getModel() {
  if (!solverModel) {
    solverModel = std::shared_ptr<SolverModel>(new SolverModel());
  }
  return solverModel;
}

CPLEXSolver::CPLEXVarType CPLEXSolver::translateVariable(
    const VariablePtr& variable) const {
  if (variable->variableType == tetrisched::VariableType::VAR_INTEGER) {
    // NOTE (Sukrit): Do not use value_or here since the type coercion renders
    // the IloIntMin and IloIntMax fallbacks incorrect.
    IloInt lowerBound =
        variable->lowerBound.has_value() ? variable->lowerBound.value() : 0;
    IloInt upperBound = variable->upperBound.has_value()
                            ? variable->upperBound.value()
                            : IloIntMax;
    return IloIntVar(cplexEnv, lowerBound, upperBound,
                     variable->getName().c_str());
  } else if (variable->variableType ==
             tetrisched::VariableType::VAR_CONTINUOUS) {
    IloNum lowerBound =
        variable->lowerBound.has_value() ? variable->lowerBound.value() : 0;
    IloNum upperBound = variable->upperBound.has_value()
                            ? variable->upperBound.value()
                            : IloInfinity;
    return IloNumVar(cplexEnv, lowerBound, upperBound, IloNumVar::Type::Float,
                     variable->getName().c_str());
  } else if (variable->variableType ==
             tetrisched::VariableType::VAR_INDICATOR) {
    return IloBoolVar(cplexEnv, variable->getName().c_str());
  } else {
    throw tetrisched::exceptions::SolverException(
        "Unsupported variable type: " + variable->variableType);
  }
}

IloRange CPLEXSolver::translateConstraint(
    const ConstraintPtr& constraint) const {
  IloExpr constraintExpr(cplexEnv);

  // Construct all the terms.
  for (const auto& [coefficient, variable] : constraint->terms) {
    if (variable) {
      // If the term has not been translated, throw an error.
      if (cplexVariables.find(variable->getId()) == cplexVariables.end()) {
        throw tetrisched::exceptions::SolverException(
            "Variable " + variable->getName() + " not found in CPLEX model.");
      }
      // Call the relevant function to add the term to the constraint.
      switch (variable->variableType) {
        case tetrisched::VariableType::VAR_INTEGER:
          constraintExpr +=
              coefficient *
              std::get<IloIntVar>(cplexVariables.at(variable->getId()));
          break;
        case tetrisched::VariableType::VAR_CONTINUOUS:
          constraintExpr +=
              coefficient *
              std::get<IloNumVar>(cplexVariables.at(variable->getId()));
          break;
        case tetrisched::VariableType::VAR_INDICATOR:
          constraintExpr +=
              coefficient *
              std::get<IloBoolVar>(cplexVariables.at(variable->getId()));
          break;
        default:
          throw tetrisched::exceptions::SolverException(
              "Unsupported variable type: " + variable->variableType);
      }
    } else {
      constraintExpr += coefficient;
    }
  }

  // Construct the RHS of the Constraint.
  IloRange rangeConstraint;
  switch (constraint->constraintType) {
    case tetrisched::ConstraintType::CONSTR_LE:
      rangeConstraint = (constraintExpr <= constraint->rightHandSide);
      break;
    case tetrisched::ConstraintType::CONSTR_EQ:
      rangeConstraint = (constraintExpr == constraint->rightHandSide);
      break;
    case tetrisched::ConstraintType::CONSTR_GE:
      rangeConstraint = (constraintExpr >= constraint->rightHandSide);
      break;
    default:
      throw tetrisched::exceptions::SolverException(
          "Unsupported constraint type: " + constraint->constraintType);
  }
  rangeConstraint.setName(constraint->getName().c_str());

  return rangeConstraint;
}

IloObjective CPLEXSolver::translateObjectiveFunction(
    const ObjectiveFunctionPtr& objectiveFunction) const {
  IloExpr objectiveExpr(cplexEnv);

  // Construct all the terms.
  for (const auto& [coefficient, variable] : objectiveFunction->terms) {
    if (variable) {
      // If the variable has not been translated, throw an error.
      if (cplexVariables.find(variable->getId()) == cplexVariables.end()) {
        throw tetrisched::exceptions::SolverException(
            "Variable " + variable->getName() + " not found in CPLEX model.");
      }
      // Call the relevant function to add the term to the constraint.
      switch (variable->variableType) {
        case tetrisched::VariableType::VAR_INTEGER:
          objectiveExpr +=
              coefficient *
              std::get<IloIntVar>(cplexVariables.at(variable->getId()));
          break;
        case tetrisched::VariableType::VAR_CONTINUOUS:
          objectiveExpr +=
              coefficient *
              std::get<IloNumVar>(cplexVariables.at(variable->getId()));
          break;
        case tetrisched::VariableType::VAR_INDICATOR:
          objectiveExpr +=
              coefficient *
              std::get<IloBoolVar>(cplexVariables.at(variable->getId()));
          break;
        default:
          throw tetrisched::exceptions::SolverException(
              "Unsupported variable type: " + variable->variableType);
      }
    } else {
      objectiveExpr += coefficient;
    }
  }

  // Construct the Sense of the Constraint.
  IloObjective objectiveConstraint;
  switch (objectiveFunction->objectiveType) {
    case tetrisched::ObjectiveType::OBJ_MAXIMIZE:
      objectiveConstraint = IloMaximize(cplexEnv, objectiveExpr);
      break;
    case tetrisched::ObjectiveType::OBJ_MINIMIZE:
      objectiveConstraint = IloMinimize(cplexEnv, objectiveExpr);
      break;
    default:
      throw tetrisched::exceptions::SolverException(
          "Unsupported objective type: " + objectiveFunction->objectiveType);
  }

  return objectiveConstraint;
}

void CPLEXSolver::translateModel() {
  if (!solverModel) {
    throw tetrisched::exceptions::SolverException(
        "Empty SolverModel for CPLEXSolver. Nothing to translate!");
  }

  // Generate the model to add the variables and constraints to.
  IloModel cplexModel(cplexEnv);

  // Generate all the variables and keep a cache of the variable indices
  // to the CPLEX variables.
  for (const auto& [variableId, variable] : solverModel->variables) {
    TETRISCHED_DEBUG("Adding Variable " << variable->getName() << "("
                                        << variableId << ") to CPLEX Model.");
    cplexVariables[variableId] = translateVariable(variable);
  }

  // Generate all the constraints and add it to the model.
  for (const auto& [constraintId, constraint] : solverModel->constraints) {
    TETRISCHED_DEBUG("Adding Constraint " << constraint->getName() << "("
                                          << constraintId
                                          << ") to CPLEX Model.");
    cplexModel.add(translateConstraint(constraint));
  }

  // Translate the objective function.
  cplexModel.add(translateObjectiveFunction(solverModel->objectiveFunction));

  // Extract the model to the CPLEX instance.
  cplexInstance.extract(cplexModel);
}

void CPLEXSolver::exportModel(const std::string& fname) {
  cplexInstance.exportModel(fname.c_str());
}
}  // namespace tetrisched
// // Spend at least timeLimit sec. on optimization, but once
// // this limit is reached, quit as soon as the solution is acceptable

// ILOMIPINFOCALLBACK5(timeLimitCallback,
//                     IloCplex, cplex,
//                     IloBool,  aborted,
//                     IloNum,   timeStart,
//                     IloNum,   timeLimit,
//                     IloNum,   acceptableGap)
// {
//     if ( !aborted  &&  hasIncumbent() ) {
//         IloNum gap = 100.0 * abs(getMIPRelativeGap());
//         IloNum timeUsed = cplex.getCplexTime() - timeStart;
//         //if ( timeUsed > 1 )  printf ("time used = %g\n", timeUsed);
//         if ( timeUsed > timeLimit) {
//             getEnv().out() << endl
//                            << "Good enough solution at "
//                            << timeUsed << " sec., gap = "
//                            << gap << "%, quitting." << endl;
//             aborted = IloTrue;
//             abort();
//         }
//     }
// }

// void CPLEXSolver::solve(double timeLimit)
// {

//     cplex.use(timeLimitCallback(env, cplex, IloFalse, cplex.getCplexTime(),
//     timeLimit, 5.0));

//     if (true) {
//         IloNumArray initvals(env);
//         cout << "[SOLVER][DEBUG] initval: ";
//         for (int i=0; i<this->vars.getSize(); i++) {
//             Variable v = mptr->getVariables()[i];
//             initvals.add(v.initval);
//             //cout << v.initval << " ";
//             if (v.initval>0) {
//                 cout << v.initval << " ";
//             }
//         }
//         cout << endl;
//         //cplex.addMIPStart(this->vars, initvals,
//         IloCplex::MIPStartCheckFeas); cplex.addMIPStart(this->vars, initvals,
//         IloCplex::MIPStartAuto);
//     }

//     if (!cplex.solve()) {
//         env.error() << "FAILed to optimize LP" <<endl;
//         throw "CPLEXSolver: failed to optmize LP";
//     }

// //    IloNumArray vals(env);
//     env.out() << "Solution status = " << cplex.getStatus() << endl;
//     env.out() << "Solution value  = " << cplex.getObjValue() << endl;
// /*
//     cplex.getValues(vals, this->vars);
//     env.out() << "Values        = " << endl << vals << endl;
// */
// }

// // get raw results for all decision variables
// vector<double> CPLEXSolver::getRawResults()
// {
//     // extract partition variable results only
//     vector<double> result;
//     IloNumArray vals(env);
//     assert(vals.getSize() == mptr->getVariables().size());

//     cplex.getValues(vals, this->vars);
//     for (int i=0; i < vals.getSize(); i++) {
//         result.push_back(vals[i]);
//     }
//     return result;
// }

// // get specific results of interest: which jobs consume how much from which
// partitions
// // generate results for start_time of interest
// map<JobPtr, Allocation> CPLEXSolver::getResults()
// {
//     // extract partition variable results only
//     IloNumArray vals(env);
//     // debugging
//     //cout << "[solver]: variable array sizes: " << this->vars.getSize() << "
//     " << mptr->getVariables().size(); assert(this->vars.getSize() ==
//     mptr->getVariables().size());
//     //map<JobPtr, map<int, int> > j2alloc; // jobptr -> partition -> count
//     map<JobPtr, Allocation> j2alloc;

//     cplex.getValues(vals, this->vars);
//     for (int i=0; i<vals.getSize(); i++) {
//         Variable v = mptr->getVariables()[i];
//         if (v.partidx >= 0) {
//             // partition variable --> add it, account for it
//             //const JobPtr &jptr = v.jptr;
//             JobPtr jptr = v.jptr;
//             assert (jptr);
//             if (j2alloc.find(jptr) == j2alloc.end()) {
//                 // initialize
//                 j2alloc[jptr] = Allocation();
//                 cout << "[j2alloc] " << vals[i] << endl;
//                 j2alloc[jptr].p2c[v.partidx] = vals[i];
//                 j2alloc[jptr].duration = v.duration;
//                 j2alloc[jptr].start_time = v.start_time;
//             } else {
//                 assert(j2alloc[jptr].duration == v.duration);
//                 assert(j2alloc[jptr].start_time == v.start_time);
//                 j2alloc[jptr].p2c[v.partidx] += vals[i];
//             }
//         }
//     }
//     return j2alloc;
// }

// map<JobPtr, Allocation> CPLEXSolver::getResultsForTime(double start_time)
// {
//     // extract partition variable results only
//     assert(this->vars.getSize() == mptr->getVariables().size());
//     map<JobPtr, Allocation> j2alloc;
//     const vector<Variable> &internal_vars = mptr->getVariables();

//     // iterate over our vars; pull only partition vars for start_time
//     for (int i = 0; i < internal_vars.size(); i++) {
//         const Variable &v = internal_vars[i];
//         if ((v.partidx < 0) || (v.start_time != start_time)) {
//             continue; // skip all but part.vars for given start_time
//         }
//         // get the value from CPLEX
//         IloInt val;
//         val = cplex.getIntValue(this->vars[i]);
//         if (val <= 0 )
//             continue; // don't get part.vars with zero assigned

//         JobPtr jptr = v.jptr;
//         assert (jptr);
//         if (j2alloc.find(jptr) == j2alloc.end()) {
//             // initialize
//             j2alloc[jptr] = Allocation();
//             j2alloc[jptr].p2c[v.partidx] = val;
//             j2alloc[jptr].duration = v.duration; // copy duration tag
//             j2alloc[jptr].start_time = v.start_time; // copy start_time tag
//         } else {
//             // note we should not see multiple durations and start_times for
//             // the same job in results --> assert this
//             assert(j2alloc[jptr].duration == v.duration);
//             assert(j2alloc[jptr].start_time == v.start_time);
//             j2alloc[jptr].p2c[v.partidx] += val;
//         }
//     }
//     return j2alloc;
// }

// double CPLEXSolver::getResult(int vidx) const {
//     const vector<Variable> &internal_vars = mptr->getVariables(); // internal
//     model assert(this->vars.getSize() == internal_vars.size());   // solver
//     model vars assert(0 <= vidx && vidx < internal_vars.size()); const
//     Variable &v = internal_vars[vidx]; switch(v.vartype) { case VAR_INT:
//     {
//         IloInt val = cplex.getIntValue(this->vars[vidx]);
//         return val;
//     }
//     case VAR_FLOAT:
//     {
//         IloNum val = cplex.getValue(this->vars[vidx]);
//         return val;
//     }
//     case VAR_BOOL:
//     {
//         IloBool val = cplex.getIntValue(this->vars[vidx]);
//         return val;
//     }
//     default:
//     {
//         IloNum val = cplex.getValue(this->vars[vidx]);
//         return val;
//     }
//     }
// }

// // export the stored external model
// void CPLEXSolver::exportModel(const char *fname)
// {
//     cplex.exportModel(fname);
// }

// // Export the solution to the given file.
// void CPLEXSolver::exportSolution(const char *fileName) {
//     cplex.writeSolution(fileName);
// }

// } //namespace alsched

// #endif
