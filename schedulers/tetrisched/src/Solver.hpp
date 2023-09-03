#ifndef _SOLVER_HPP
#define _SOLVER_HPP
#include <ilcplex/ilocplex.h>
#include <string>
#include "Expression.hpp"
#include "SolverModel.hpp"

namespace alsched {
// Solver class is the interface for all MILP-type solvers to implement
class Solver
{
public:
    virtual SolverModelPtr initModel(double) = 0;
    virtual bool translateModel() = 0;
    virtual void exportModel(const char *fname) = 0;
    virtual void solve(double timeLimit) = 0;
    // TODO(atumanov): add required result methods to this class
    virtual double getResult(int vi) const = 0;
/*    virtual double getFloatResult(int vi) const = 0;
    virtual double getIntResult(int vi) const = 0;
    virtual double getBoolResult(int vi) const = 0;*/
};

// CPLEXSolver : understands how to talk to CPLEX and how to interface with
// our internal model representation.
// It essentially serves as the bridge between internal and external model
class CPLEXSolver : public Solver
{
private:
    SolverModelPtr mptr; //internal model
    double modelstarttime; // current model start time
    IloEnv env;
    IloModel extmodel;  //external model
    IloCplex cplex;
    IloNumVarArray vars;

    //helper functions
    IloExpr translateExpression(const vector<pair<double,int> > &terms, const IloNumVarArray &);
    IloRange translateConstraint(const Constraint &mycon, const IloNumVarArray &);
    //IloExpr translateObjFunction(const SolverModelPtr &model);

public:
    CPLEXSolver() {}
    ~CPLEXSolver() {}
    // creates, initializes and returns a reference to the new model
    SolverModelPtr initModel(double);
    // genModel: populate the stored internal model pointed to by mptr
    // NOTE: an alternative is to generate it externally and pass it for translation
    void genModel(ExpressionPtr t, vector<map<double,vector<int> > > partcaps);
    bool translateModel();
    // translate given internal model into external
    void translateModel(SolverModelPtr m);
    // solve the stored external model
    void solve(double timeLimit);
    // export the stored external model
    void exportModel(const char *fname);
    // Export the solution to the given file.
    void exportSolution(const char *fileName);
    vector<double> getRawResults();
    map<JobPtr, Allocation> getResults();
    map<JobPtr, Allocation> getResultsForTime(double start_time);
    // get a solution for a single variable, given internal variable index vidx
    virtual double getResult(int vidx) const;
};

} //namespace alsched

#endif
