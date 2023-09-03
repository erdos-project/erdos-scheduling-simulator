#ifndef _EXPRESSION_HPP_
#define _EXPRESSION_HPP_
// standard C/C++ libraries
#include <vector>
#include <iostream>
#include <map>
#include <tuple>
#include <unordered_map>
// boost libraries
#include <boost/shared_ptr.hpp>
// 3rd party libraries
// alsched libraries
#include "SolverModel.hpp"
#include "Solver.hpp"

using namespace std;

namespace alsched {

class Expression {
protected:
    bool marker = false;

public:
    static const int ALSCHED_EXPR_EXCEPTION = 0xff;


    virtual void clearMarkers() = 0;
    virtual tuple<double, double> eval(Allocation alloc) = 0;
    virtual void addChild(ExpressionPtr newchld) = 0; //appends new chld to chldarr
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
            vector<map<double,vector<int> > > &partcap,
            vector<map<double,Constraint> >& capconmap,
            JobPtr jobptr) = 0;
    // return the set of all eq.classes referenced by this expression (non-const)
    virtual void getEquivClasses(EquivClassSet& equivClasses) = 0;
    virtual void populatePartitions(const vector<int> &node2part, int curtime, int sched_horizon) = 0;
    virtual string toString() = 0;
    virtual Allocation getResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time) = 0;
    virtual void cacheNodeResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time) = 0;

    // returns interval [t_1, t_2) where this expression is meaningful (non-const due to cache)
    virtual pair<int, int> startTimeRange() = 0;
};


class SchedulingExpression: public Expression {
protected:
    NodesPtr _nodes; // Set of candidate nodes
    int _k;
    double _utility;
    int _start_time;
    double _duration;

    // resolve() will dynamically resolve eq.classes into a set of partitions
    vector<int> partitions; // partition number of each partition
    // array of par.var. indices that match partitions: requires a reset in gen()
    vector<int> partvaridx; // partition variable for each partition. owned by generate()


    vector<int> cached_nodes; // all machines allocated to this nck leaf previously
    //owns cache : purges it and populates it with new results from the solver
    //resolves solver results to nodes for *this leaf and caches them
    //mutates partcap, removing from it nodes allocated to *this leaf
    virtual void cacheNodeResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);

public:
    virtual ~SchedulingExpression() {}
    SchedulingExpression(NodesPtr nodes, int k, double utility, int start_time, double duration) :
        _nodes(nodes), _start_time(start_time), _duration(duration), _k(k), _utility(utility) { }
    void addChild(ExpressionPtr newchld) { throw Expression::ALSCHED_EXPR_EXCEPTION; }
    virtual void getEquivClasses(EquivClassSet& equivClasses);
    virtual void populatePartitions(const vector<int> &node2part, int curtime, int sched_horizon);
    virtual Allocation getResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);
    void setPartitions(vector<int> parts) { partitions = parts; }

    void clearMarkers() { marker = false; }
    pair<int, int> startTimeRange() { return make_pair(_start_time, _start_time + 1); };
    virtual string debugString() const;
};

class Choose: public SchedulingExpression {
public:
    typedef boost::shared_ptr<Choose> ChooseExprPtr;
    //choose <k> machines from <equivclass> with utility value <utilval>
    Choose(NodesPtr nodes, int k, UtilVal utilval, int start_time, double duration)
            : SchedulingExpression(nodes, k, utilval(start_time, duration), start_time, duration) {};
    virtual ~Choose() {}
    tuple<double, double> eval(Allocation alloc);
    //recursively generate a model for a solver; aggregate capacity constraints
    vector<pair<double,int> > generate(SolverModelPtr m, int I,
                                       vector<map<double,vector<int> > > &partcap,
                                       vector<map<double,Constraint> >& capconmap,
                                       JobPtr jobptr);
    virtual string toString();
};

class LinearChoose: public SchedulingExpression {
public:
    typedef boost::shared_ptr<LinearChoose> lnckExprPtr;
    LinearChoose(NodesPtr nodes, int k, UtilVal utilval, int start_time, double duration)
        : SchedulingExpression(nodes, k, utilval(start_time, duration), start_time, duration) {};
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
                                               vector<map<double,vector<int> > > &partcap,
                                               vector<map<double,Constraint> >& capconmap,
                                               JobPtr jobptr);

    virtual ~LinearChoose() {}
    virtual tuple<double, double> eval(Allocation alloc);
    virtual string toString();
};

class PreemptingExpression: public SchedulingExpression {
protected:
    unordered_map<int, vector<int> > partition_nodes;
    void cacheNodeResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);

public:
    PreemptingExpression(NodesPtr nodes, double utility, int start_time, double duration)
            : SchedulingExpression(nodes, static_cast<int>(nodes->size()), utility, start_time, duration) {};
    virtual ~PreemptingExpression() {}
    void getEquivClasses(EquivClassSet& equivClasses) { } // No-op
    void populatePartitions(const vector<int> &node2part, int curtime, int sched_horizon);
};

class KillChoose: public PreemptingExpression {
protected:
public:
    typedef boost::shared_ptr<KillChoose> KillChooseExprPtr;
    KillChoose(NodesPtr nodes, UtilVal utilval, double start_time, double duration)
            : PreemptingExpression(nodes, utilval(start_time, duration), start_time, duration) {}
    virtual ~KillChoose() {}
    virtual tuple<double, double> eval(Allocation alloc);
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
                                               vector<map<double,vector<int> > > &partcap,
                                               vector<map<double,Constraint> >& capconmap,
                                               JobPtr jobptr);
    virtual string toString();

};


class KillLinearChoose: public PreemptingExpression {
public:
    typedef boost::shared_ptr<KillLinearChoose> KillLinearChooseExprPtr;
    KillLinearChoose(NodesPtr nodes, UtilVal utilval, double start_time, double duration)
            : PreemptingExpression(nodes, utilval(start_time, duration), start_time, duration) {}
    virtual ~KillLinearChoose() {}
    virtual tuple<double, double> eval(Allocation alloc);
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
                                               vector<map<double,vector<int> > > &partcap,
                                               vector<map<double,Constraint> >& capconmap,
                                               JobPtr jobptr);
    virtual string toString();

};

class UnaryOperator: public Expression {

protected:
    ExpressionPtr child;
public:
    UnaryOperator() {}
    UnaryOperator(ExpressionPtr chld) : child(chld) {}
    virtual ~UnaryOperator() {}
    void clearMarkers();
    void addChild(ExpressionPtr newchld);
    void getEquivClasses(EquivClassSet& equivClasses);
    void populatePartitions(const vector<int> &node2part, int curtime, int sched_horizon);
    Allocation getResults(const Solver &solver,
                          vector<map<double, vector<int> > > &partcap,
                          int start_time);
    void cacheNodeResults(const Solver &solver,
                          vector<map<double, vector<int> > > &partcap,
                          int st);
    pair<int, int> startTimeRange() { return child->startTimeRange(); }
};

class BarrierExpr: public UnaryOperator {
private:
    double barrier;
    double start_time; // start time
    double duration; // duration

public:
    typedef boost::shared_ptr<BarrierExpr> BarrierExprPtr;
    BarrierExpr(double barrier, double start_time, double duration);
    BarrierExpr(double barrier, double start_time, double duration, ExpressionPtr newchld);
    virtual ~BarrierExpr() {}
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
            vector<map<double,vector<int> > > &partcap,
            vector<map<double,Constraint> >& capconmap,
            JobPtr jobptr);

    virtual tuple<double, double> eval(Allocation alloc);
    virtual string toString();
};

class ScaleExpr: public UnaryOperator {
private:
    double factor;

public:
    typedef boost::shared_ptr<ScaleExpr> ScaleExprPtr;
    ScaleExpr(double factor);
    ScaleExpr(double factor, ExpressionPtr newchld);
    virtual ~ScaleExpr() {}
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
            vector<map<double,vector<int> > > &partcap,
            vector<map<double,Constraint> >& capconmap,
            JobPtr jobptr);
    virtual tuple<double, double> eval(Allocation alloc);
    virtual string toString();
};

// Job expression : sole purpose is to pass the jobptr down to children
class JobExpr: public UnaryOperator {
private:
    JobPtr jptr;

public:
    const JobPtr getJobPtr() const {return jptr; }
    typedef boost::shared_ptr<JobExpr> JobExprPtr;
    JobExpr() {} // default ctor
    JobExpr(JobPtr _jptr): jptr(_jptr) {}
    JobExpr(JobPtr _jptr, ExpressionPtr _chld): jptr(_jptr), UnaryOperator(_chld) { }
    virtual ~JobExpr() {}
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
            vector<map<double,vector<int> > > &partcap,
            vector<map<double,Constraint> >& capconmap,
            JobPtr jobptr);
    virtual tuple<double, double> eval(Allocation alloc)
        {return child->eval(alloc); }
    virtual string toString() {return child->toString();}
};

class NnaryOperator: public Expression {
protected:
    vector<ExpressionPtr> chldarr;
    bool homogeneous_children_nodes;

    bool cache_dirty = true; // This is not related to cacheNodeResults
private:
    // cache fields
    pair<int, int> startTimeRangeCache = make_pair(INT_MAX, INT_MIN);

public:
    NnaryOperator(bool _homogeneous_children_nodes = false)
            : homogeneous_children_nodes(_homogeneous_children_nodes) {}
    virtual ~NnaryOperator() {}
    virtual void clearMarkers();
    virtual void addChild(ExpressionPtr newchld);    //appends new chld to chldarr
    virtual ExpressionPtr removeChild(const ExpressionPtr &chld);
    virtual void getEquivClasses(EquivClassSet& equivClasses);
    virtual void populatePartitions(const vector<int> &node2part, int curtime, int sched_horizon);

    pair<int, int> startTimeRange();
};
class MinExpression: public NnaryOperator {
private:
    double cached_minU; // decision variable cache -- owned by getResults()
    int minUvaridx;  // decision variable index -- owned by generate()
    // helper functions
    double upper_bound(vector<map<double,vector<int> > > &partcap);

public:
    typedef boost::shared_ptr<MinExpression> MinExprPtr;
    MinExpression(bool homogeneous_children_nodes = false)
            : NnaryOperator(homogeneous_children_nodes){ cached_minU = 0; }
    virtual ~MinExpression() {}
    virtual tuple<double, double> eval(Allocation alloc);
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
                                               vector<map<double,vector<int> > > &partcap,
                                               vector<map<double,Constraint> >& capconmap,
                                               JobPtr jobptr);
    virtual string toString();
    virtual Allocation getResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);
    virtual void cacheNodeResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);
};

class MaxExpression: public NnaryOperator {
private:
    map<ExpressionPtr, int> indvarmap;
    map<ExpressionPtr, int> cached_indvarmap;

public:
    typedef boost::shared_ptr<MaxExpression> MaxExprPtr;
    MaxExpression(bool homogeneous_children_nodes = false)
            : NnaryOperator(homogeneous_children_nodes){}
    virtual ~MaxExpression() {}

    ExpressionPtr removeChild(const ExpressionPtr &chld);
    virtual tuple<double, double> eval(Allocation alloc);
    //recursively generate a model for a solver; aggregate capacity constraints
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
                                               vector<map<double,vector<int> > > &partcap,
                                               vector<map<double,Constraint> >& capconmap,
                                               JobPtr jobptr);
    //friend ostream & operator<< (ostream &out, MaxExpression *objp);
    virtual string toString();
    virtual Allocation getResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);
    virtual void cacheNodeResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);
};

class SumExpression: public NnaryOperator {
private:
    //vector<int> indvaridx;
    map<ExpressionPtr, int> indvarmap;
    map<ExpressionPtr, int> cached_indvarmap;
public:
    typedef boost::shared_ptr<SumExpression> SumExprPtr;
    SumExpression(bool homogeneous_children_nodes = false)
            : NnaryOperator(homogeneous_children_nodes){}
    virtual ~SumExpression() {}

    ExpressionPtr removeChild(const ExpressionPtr &chld); // remove matching child
    ExpressionPtr removeChild(JobPtr jptr); // remove matching child
    void addChildIfNew(ExpressionPtr newchld); // adds chld only if not already present
    void addChildIfNew(JobPtr jptr, const std::function<ExpressionPtr(JobPtr)>& func); //adds a JobExpr child if not already present
    virtual tuple<double, double> eval(Allocation alloc);
    virtual vector<pair<double,int> > generate(SolverModelPtr m, int I,
            vector<map<double,vector<int> > > &partcap,
            vector<map<double,Constraint> >& capconmap,
            JobPtr jobptr);
    //friend ostream & operator<< (ostream &out, SumExpression *objp);
    virtual string toString();
    virtual Allocation getResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);
    virtual void cacheNodeResults(const Solver &solver,
                                  vector<map<double, vector<int> > > &partcap,
                                  int start_time);
};
} //namespace alsched

#endif
