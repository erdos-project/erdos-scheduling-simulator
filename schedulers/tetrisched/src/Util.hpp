#ifndef _UTIL_HPP_
#define _UTIL_HPP_
#include <vector>
using namespace std;

namespace alsched {
vector<int> & add1d(const vector<int> &v1, const vector<int> &v2, vector<int> &v3);
vector<int> & sub1d(const vector<int> &v1, const vector<int> &v2, vector<int> &v3);
vector<int> & incbyvec(vector<int> &v1, const vector<int> &v2);
vector<int> & scalevec(vector<int> &v, int factor);
//intersect two vectors preserving input; return intersection size
int vec_isection(vector<int> a, vector<int> b, vector<int> &res);
int vec_isection_size(const vector<int> &a, const vector<int> &b);
//take the union of two vectors preserving input; return union size
//int vector_union(vector<int> a, vector<int> b, vector<int> &res);


void print1d(const vector<int> &v);
void print2d(const vector<vector<int> > &m);

} //namespace alsched
#endif
