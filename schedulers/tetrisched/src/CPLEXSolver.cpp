#include "tetrisched/CPLEXSolver.hpp"

namespace tetrisched {
SolverModelPtr CPLEXSolver::getModel() {
  auto a = std::shared_ptr<SolverModel>(new SolverModel());
  return a;
}

void CPLEXSolver::exportModel(const std::string& fname) {
  // cplex.exportModel(fname);
}
}  // namespace tetrisched
// #ifndef _SOLVER_CPP_
// #define _SOLVER_CPP_
// #include <boost/make_shared.hpp>
// #include <cassert>
// #include <cmath>
// #include "Solver.hpp"

// namespace alsched {

// SolverModelPtr CPLEXSolver::initModel(double curtime)
// {
//     // create new SolverModel pointer, discarding previous
//     this->env.end();
//     this->env = IloEnv();
//     mptr = boost::make_shared<SolverModel>();
//     // update current model time
//     modelstarttime = curtime;
//     return mptr;
// }

// // given expression rooted at t, generate the SolverModel
// void CPLEXSolver::genModel(ExpressionPtr t,
//                            vector<map<double,vector<int> > > partcaps)
// {
//     vector<map<double, Constraint> > capconmap(partcaps.size());
//     //JobPtr nulljobptr = boost::shared_ptr<Job>();
//     // note that indicator variables don't care about start/duration tags
//     int root_var_idx = mptr->addVariable(
//             Variable(VAR_BOOL, pair<double,double>(0, 1), -1, "rootI",
//             nullptr,
//                     0, 0, 1)); // initial value == 1 (enabled root)

//     vector<pair<double, int> > objf =
//         t->generate(mptr, 0, partcaps, capconmap, nullptr);

//     mptr->setObjective(objf, true);

//     // adding supply (aka capacity) constraints
//     for (const auto &t2c: capconmap) {
//         // each element of capconmap is a time->Constraint map
//         for (const auto &item: t2c) {
//             mptr->addConstraint(item.second);
//         }
//     }
//     // debug print
//     cout << "[solver]: printing final internal model:" << endl;
//     mptr->printModel(cout);
// }

// // helper function to translate one constraint with >=1 terms
// // NOTE: terms[i].second is the IloNumVar index
// IloExpr CPLEXSolver::translateExpression(const vector<pair<double,int> >
// &terms,
//         const IloNumVarArray &vars)
// {
//     IloExpr conexpr(env);
//     for (auto &term: terms) {
//         conexpr += term.first * vars[term.second];
//     }
//     return conexpr;
// }

// IloRange CPLEXSolver::translateConstraint(const Constraint &mycon, const
// IloNumVarArray &vars)
// {
//     IloRange range;
//     IloExpr conexpr = translateExpression(mycon.terms, vars);

//     switch(mycon.op) {
//     case OP_EQ:
//         range = (conexpr == mycon.rhs); break;
//     case OP_LE:
//         range = (conexpr <= mycon.rhs); break;
//     case OP_GE:
//         range = (conexpr >= mycon.rhs); break;
//     default:
//         throw "CPLEXSolver: unknown constraint op encountered!";
//     }
//     return range;
// }

// // translate stored internal model into external
// bool CPLEXSolver::translateModel()
// {
//     map<var_enum_t, IloNumVar::Type> int2exttype;
//     // helper map to lookup corresponding variable types
//     int2exttype[VAR_BOOL] = IloNumVar::Bool;
//     int2exttype[VAR_FLOAT] = IloNumVar::Float;
//     int2exttype[VAR_INT] = IloNumVar::Int;

//     // IloModel model(env, "debug-alsched-milp");

//     extmodel = IloModel(env, "alsched-milp");
//     this->cplex = IloCplex(this->extmodel);
//     //IloNumVarArray vars(env);
//     vars = IloNumVarArray(env);
//     IloRangeArray cons(env);
//     // translate variables
//     cout << mptr->getVariables().size() << endl;
//     for (const auto &var: mptr->getVariables()) {
//         cout << "Adding " << var.name.c_str() << endl;
//         switch(var.vartype) {
//         case VAR_BOOL:
//         {
//             IloBoolVar cplexv(env, var.name.c_str());
//             vars.add(cplexv);
//             break;
//         }
//         case VAR_INT:
//         {
//             IloIntVar cplexv(env, var.range.first, var.range.second,
//             var.name.c_str()); vars.add(cplexv); break;
//         }
//         case VAR_FLOAT:
//         {
//             IloNumVar cplexv(env, var.range.first, var.range.second,
//             ILOFLOAT, var.name.c_str()); vars.add(cplexv); break;
//         }
//         }
// //        IloNumVar::Type t = int2exttype[var.vartype];
// //        IloNumVar cplexv(env, var.range.first, var.range.second, t,
// var.name.c_str());
// //        env.out() << "VAR: " <<cplexv<<endl;
//     }

//     // translate constraints
//     for (const auto &con: mptr->getConstraints()) {
//         IloRange range = translateConstraint(con, vars);
//        env.out()<<"RANGE: "<<range<<endl;
//         cons.add(range);
//     }
//     extmodel.add(cons); // add constraints to the model

//     // translate objective function
//     IloExpr objfexpr = translateExpression(mptr->getObjective().objfexpr,
//     vars); if (mptr->getObjective().maximize) {
//         extmodel.add(IloMaximize(env, objfexpr));
//     } else {
//         extmodel.add(IloMinimize(env, objfexpr));
//     }
//     return true;
// }

// // translate given internal model into external
// void CPLEXSolver::translateModel(SolverModelPtr)
// {
//     throw "not implemented";
// }

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
