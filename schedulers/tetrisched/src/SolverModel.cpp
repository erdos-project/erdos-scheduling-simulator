#ifndef _SOLVERMODEL_CPP_
#define _SOLVERMODEL_CPP_

#include <cassert>
#include <iostream>
#include <string>
#include "SolverModel.hpp"

namespace alsched {
string Variable::toString()
{
    string out = this->name;
    out += "[" + to_string(range.first) + ".." + to_string(range.second) + "]";
    return out;
}

int Constraint::addConstraintTerm(ConstraintTerm term) {
    this->terms.push_back(pair<double, int>(term.coeff, term.varidx));
    return this->terms.size() -1;
}

// returns the index of the term added
int Constraint::addConstraintTerm(pair<double,int> term)
{
    this->terms.push_back(term);
    return this->terms.size() -1;
}

string Constraint::toString()
{
    string outstr;
    for (auto &p: this->terms) {
        string termstr = "(" +to_string(p.first) + "," + to_string(p.second)
                       + ")";
        outstr += termstr;
    }
    // op
    switch (this->op) {
    case OP_EQ:
        outstr += "="; break;
    case OP_LE:
        outstr += "<="; break;
    case OP_GE:
        outstr += ">="; break;
    default:
        throw "bad stuff happened";
    }
    // rhs
    outstr += to_string(this->rhs);
    return outstr;
}

string Constraint::toString(const vector<Variable>& varmap)
{
    string outstr;
    for (auto &p: this->terms) {
        string termstr = "(" +to_string(p.first) + "," + const_cast<Variable *>(&varmap[p.second])->toString()
                         + ")";
        outstr += termstr;
    }
    // op
    switch (this->op) {
        case OP_EQ:
            outstr += "="; break;
        case OP_LE:
            outstr += "<="; break;
        case OP_GE:
            outstr += ">="; break;
        default:
            throw "bad stuff happened";
    }
    // rhs
    outstr += to_string(this->rhs);
    return outstr;
}

int SolverModel::addVariable(const Variable v)
{
    vars.push_back(v);
    return vars.size()-1; // return variable index
}

// add a single constraint
void SolverModel::addConstraint(const Constraint c)
{
    cons.push_back(c);
}
// add multiple constraints at once
void SolverModel::addConstraints(const vector<Constraint>& cvec)
{
    for (auto c: cvec )
        addConstraint(c);
}

void SolverModel::setObjective(vector<pair<double,int> > objf, bool max)
{
    this->objf = ObjFunction(objf, max);
}

void SolverModel::printModel(ostream &out)
{
    out << "Variables: ";
    // just print indices for now
    for (int i=0; i<vars.size(); i++) {
        out << vars[i].toString() <<",";
    }
    out <<endl;

    out << "Constraints: "; printConstraints(out);
    out << "Objf: "; printObjective(out);
}

void SolverModel::printConstraints(ostream &out)
{
    for (auto &c: this->cons) {
        out << c.toString(vars) <<endl;
    }

}
void SolverModel::printObjective(ostream &out)
{
    out << ((this->objf.maximize) ? "max: " : "min: ");
    for (auto &p: this->objf.objfexpr) {
        out <<"(" << p.first <<"," <<vars[p.second].toString()<<"), ";
    }
    out <<endl;
}

} //namespace alsched

#endif
