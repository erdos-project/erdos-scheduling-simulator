#ifndef _SOLVER_MODEL_HPP_
#define _SOLVER_MODEL_HPP_

#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "common.hpp"

using namespace std;

namespace alsched {

typedef enum {
    OP_LE,
    OP_EQ,
    OP_GE,
    OP_MAX
} op_enum_t;

typedef enum {
    VAR_FLOAT,
    VAR_INT,
    VAR_BOOL,
    VAR_MAX
} var_enum_t;

struct ConstraintTerm {
    double coeff;
    int varidx;
};

//TODO(atumanov): create parent Constraint class; have IntConstraint and DoubleConstraint
// derive from it. IntConstraint will have all variables and all coefficients as ints.
// I'm concerned that the MILP solver will be unhappy about f.p. coefficients

// each Constraint is: sum(terms) op rhs; each term is coeff*variable
class Variable;
struct Constraint {
    vector<pair<double,int> > terms;
    double rhs;
    op_enum_t op;
    Constraint() {} // compiler insists its needed for capconmap[p] = Constraint
    Constraint(double _rhs, op_enum_t _op) : rhs(_rhs), op(_op) {}
    Constraint(vector<pair<double,int> > _terms, double _rhs, op_enum_t _op):
        terms(_terms), rhs(_rhs), op(_op) {}
    int addConstraintTerm(ConstraintTerm term);
    int addConstraintTerm(pair<double,int> term);
    string toString();
    string toString(const vector<Variable>& varmap);
};

//decision variable struct
struct Variable {
    //int idx; // variable index
    var_enum_t vartype; // decision variable data type
    string name; //variable name
    pair<double,double> range; //variable range [min; max]
    double initval; // initial value/guess for this var (must be feasible)

    // markers
    int partidx; //what partition this variable belongs to
    JobPtr jptr;
    double duration;
    double start_time;

    Variable(var_enum_t _type, pair<double,double> _range, int _pi, string _name,
            JobPtr _jptr, double _st, double _d, double _initval)
        : vartype(_type), partidx(_pi), range(_range), name(_name), jptr(_jptr),
          start_time(_st), duration(_d), initval(_initval) {}
    string toString();
};

struct ObjFunction {
    vector<pair<double,int> > objfexpr; // (c1*x1 + ... + cn*xn)
    bool maximize; // max or min the objective function
    ObjFunction(vector<pair<double,int> > _expr, bool _max):
        objfexpr(_expr), maximize(_max) {}
    ObjFunction() {}
    ObjFunction& operator+=(const ObjFunction& rhs);
};

class SolverModel {
protected:
    vector<Constraint> cons;
    vector<Variable> vars;
    ObjFunction objf;
    //vector<pair<double,int> > objf; // (c1*x1 + ... + cn*xn)
    //bool maximize; // max or min the objective function

public:
    SolverModel() {}
    ~SolverModel() {}
    int addVariable(const Variable v);
    void addConstraint(const Constraint c);
    void addConstraints(const vector<Constraint>& c); // add c to cons
    void setObjective(vector<pair<double,int> > objf, bool maximize);

    // const getters
    const ObjFunction& getObjective() {return objf;}
    const vector<Constraint>& getConstraints() {return cons;}
    const vector<Variable>& getVariables() {return vars;}

    // debugging
    string toString();
    string debugString();
    void printObjective(ostream &out);
    void printConstraints(ostream &out);
    void printModel(ostream &out);
};
typedef boost::shared_ptr<SolverModel> SolverModelPtr;
} //namespace alsched

#endif
