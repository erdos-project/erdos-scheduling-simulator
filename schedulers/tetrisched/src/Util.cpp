#include "Util.hpp"
#include <iostream>
#include <algorithm>

using namespace std;

namespace alsched {
vector<int> & add1d(const vector<int> &v1, const vector<int> &v2, vector<int> &v3)
{
    //assert that we are adding vectors of the same dimension
    v3.resize(v1.size());
    for (int i=0; i < v1.size(); i++)
        v3[i] = v1[i] + v2[i];

    return v3;
}

vector<int> & sub1d(const vector<int> &v1, const vector<int> &v2, vector<int> &v3)
{
    //assert same size v1 & v2
    v3.resize(v1.size());
    for (int i=0; i< v1.size(); i++)
        v3[i] = v1[i] - v2[i];

    return v3;
}

int int_sum_op(int a, int b) {return a+b;} //_almost_ lambda functions...

vector<int> & incbyvec(vector<int> &v1, const vector<int> &v2)
{
    //just assume same sized vectors for now
    transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), int_sum_op);
    return v1;
}

vector<int> & scalevec(vector<int> &v, int factor)
{
    //the following will have to wait till lambda support in g++...
    //auto int_scale_op = [](int x) {return x*factor;} //nice try ...
    //transform(v.begin(), v.end(), v.begin(), int_scale_op);
    //for_each (v.begin(), v.end(), int_scale_op);
    for (int i=0; i<v.size(); i++)
        v[i] *= factor;
    return v;
}

void print1d(const vector<int> &v)
{
    for (int i=0; i<v.size(); i++) {
        cout<<v[i] <<" ";
    }
}


void print2d(const vector<vector<int> > &m)
{
    for (int i=0; i<m.size(); i++) {
        for (int j=0; j<m[0].size(); j++)
            cout<<m[i][j]<<" ";
        cout<<endl;
    }
}

// input: vectors a and b passed by value;
// output: vector res is the intersection of values in input vectors
// r.v.: size of the intersection
int vec_isection(vector<int> a, vector<int> b, vector<int> &res)
{
    if (a.empty() || b.empty()) {
        res = vector<int>(); // empty vec
        return res.size();
    }

    // mutates copies on the stack
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    set_intersection(a.begin(), a.end(), b.begin(), b.end(), back_inserter(res));
    return res.size();
}

// guarantees immutability of input vectors a and b
// returns size of intersection of input vectors a and b
int vec_isection_size(const vector<int> &a, const vector<int> &b)
{
    if (a.empty() || b.empty())
        return 0;
    vector<int> res;
    return vec_isection(a, b, res);
}

} //namespace alsched
