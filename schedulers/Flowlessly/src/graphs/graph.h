/*
 * Flowlessly
 * Copyright (c) Ionel Gog <ionel.gog@cl.cam.ac.uk>
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT
 * LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR
 * A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
 *
 * See the Apache Version 2.0 License for specific language governing
 * permissions and limitations under the License.
 */

#ifndef FLOWLESSLY_GRAPH_H
#define FLOWLESSLY_GRAPH_H

#include <gflags/gflags.h>
#include <stdint.h>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "graphs/node.h"
#include "misc/statistics.h"

namespace flowlessly {

using namespace std;  // NOLINT

// Forward declarations.
class AdjacencyMapGraph;

struct Arc {
  Arc() {}

  Arc(uint32_t src_id, uint32_t dst_id, bool fwd, bool running,
      int32_t capacity, int32_t cap_lower_bound, int64_t cst, Arc* rev_arc)
  : src_node_id(src_id), dst_node_id(dst_id), is_alive(true), is_fwd(fwd),
    is_running(running), residual_cap(capacity), min_flow(cap_lower_bound),
    cost(cst), reverse_arc(rev_arc) {
  }

  Arc(const Arc& arc) : src_node_id(arc.src_node_id),
    dst_node_id(arc.dst_node_id), is_alive(arc.is_alive), is_fwd(arc.is_fwd),
    is_running(arc.is_running), residual_cap(arc.residual_cap),
    min_flow(arc.min_flow), cost(arc.cost), reverse_arc(arc.reverse_arc) {
  }

  explicit Arc(Arc* old_arc)
  : src_node_id(old_arc->src_node_id),
    dst_node_id(old_arc->dst_node_id),
    is_alive(old_arc->is_alive),
    is_fwd(old_arc->is_fwd),
    is_running(old_arc->is_running),
    residual_cap(old_arc->residual_cap),
    min_flow(old_arc->min_flow),
    cost(old_arc->cost) {
  }

  void CopyArc(const Arc& src_arc) {
    src_node_id = src_arc.src_node_id;
    dst_node_id = src_arc.dst_node_id;
    is_alive = src_arc.is_alive;
    is_fwd = src_arc.is_fwd;
    is_running = src_arc.is_running;
    residual_cap = src_arc.residual_cap;
    min_flow = src_arc.min_flow;
    cost = src_arc.cost;
    // NOTE: It doesn't copy the pointer to the reverse_arc
    // because the new arc will likely point to a different address.
  }

  // NOTE: There can be multiple arcs between a pair of nodes. However,
  // there can't be multiple arcs with the same cost. We use src_node_id,
  // dst_node_id and cost to uniquely identify arcs.
  uint32_t src_node_id;
  uint32_t dst_node_id;
  bool is_alive;
  bool is_fwd;
  bool is_running;
  int32_t residual_cap;
  uint32_t min_flow; // lower flow bound.
  int64_t cost;
  // Reduced cost in all algorithms is:
  //   cost - potential[src_node_id] + potential[dst_node_id].
  Arc *reverse_arc;
};

class Graph {
 public:
  Graph() : max_node_id_(0), num_arcs_(0), sink_node_(0),
    potentials_(10000000, 0) {}
  Graph(uint32_t max_node_id, uint32_t num_arcs, uint32_t sink_node,
        const set<uint32_t>& source_nodes) : max_node_id_(max_node_id),
    num_arcs_(num_arcs), sink_node_(sink_node), source_nodes_(source_nodes),
    potentials_(10000000, 0) {
  }
  virtual ~Graph() {}

  virtual Arc* AddArc(uint32_t src_node_id, uint32_t dst_node_id,
                      uint32_t min_flow, int32_t capacity,
                      int64_t cost, int32_t type) = 0;

  /**
   * Adds a new node to the graph.
   * @param supply the supply of the new node
   * @param price the price of the new node
   * @param type the type of the ndode
   * @return the id of the new node
   **/
  virtual uint32_t AddNode(int32_t supply, int64_t potential,
                           NodeType type, bool first_scheduling_iteration) = 0;
  virtual void AddNode(uint32_t node_id, int32_t supply, int64_t potential,
                       NodeType type, bool first_scheduling_iteration) = 0;

  /**
   * Change a given arc. The change can increase or decrease the arc's
   * capacity or cost.
   * @param arc the arc that is going to be changed
   * @param new_min_flow the new minimum flow bound of the arc
   * @param new_capacity the new arc capacity
   * @param new_cost the new arc cost
   * @param new_type the new arc type
   */
  virtual void ChangeArc(Arc* arc, uint32_t new_min_flow, int32_t new_capacity,
                         int64_t new_cost, int32_t new_type) = 0;

  /**
   * Change one of the arcs between two src_node_id and dst_node_id.
   * @param src_node_id the source node of the arc
   * @param dst_node_id the destination node of the arc
   * @param new_min_flow the new minimum flow bound of the arc
   * @param new_capacity the new arc capacity
   * @param new_cost the new arc cost
   * @param new_type the new arc type
   * @param is_multi_arc true if the arc is one of the multiple arcs
   * @param old_cost if the arc is one of the multiple arcs between the two
   * nodes then we use the old_cost to identify the arc
   */
  virtual void ChangeArc(uint32_t src_node_id, uint32_t dst_node_id,
                         uint32_t new_min_flow, int32_t new_capacity,
                         int64_t new_cost, int32_t new_type, bool is_multi_arc,
                         int64_t old_cost) = 0;

  virtual void GetMachinePUs(uint32_t machine_node_id,
                             set<uint32_t>* pu_ids) = 0;

  /**
   * Get a random outgoing arc.
   */
  virtual Arc* GetRandomArc(uint32_t node_id) = 0;

  /**
   * Get the cost of the flow that has been sent in the graph.
   */
  virtual int64_t GetTotalCost() = 0;

  /**
   * Checks epsilon optimality of the flow graph. A flow graph is
   * eps-optimal if there is no arc with capacity > 0 and
   * cost + potential[src_vertex_id] - potential[dst_vertex_id] < -eps.
   * @param eps Epsilon
   **/
  virtual bool IsEpsOptimal(int64_t eps) = 0;

  /**
   * Checks if the flow in the graph is feasible. A flow is feasible if
   * all the supply is satisfied, there's no excess supply left and
   * the flow on every arc is within bounds.
   * NOTE: It does not check if the flow is the min cost flow.
   **/
  virtual bool IsFeasible() = 0;

  /**
   * Checks if the nodes in the vector are in topological order.
   */
  virtual bool IsInTopologicalOrder(const vector<uint32_t>& node_ids) = 0;

  virtual void InitializeGraph() = 0;

  void ReadGraph(FILE* graph_file, bool first_scheduling_iteration,
                 bool* end_of_scheduling);

  /**
   * Removes the given arc from the graph. It updates the supply at the source
   * and destination nodes.
   * Removing an arc may change the graph s.t. there's no feasible flow.
   * However, the method does not check feasibility.
   * @param arc the arc to remove
   */
  virtual void RemoveArc(Arc* arc) = 0;

  /**
   * Remove the graph node corresponding to the given id.
   * NOTE: The id is going to be re-used when new nodes will be added to the
   * graph.
   * @param node_id the id of the node to remove
   */
  virtual void RemoveNode(uint32_t node_id) = 0;

  /**
   * Scales down all arc costs.
   * @param scale_down scale down factor
   **/
  virtual void ScaleDownCosts(int64_t scale_down) = 0;

  /**
   * Scales up all arc costs.
   * @param scale_up scale up factor.
   * @return the value from where eps should start
   **/
  virtual void ScaleUpCosts(int64_t scale_up) = 0;

  /**
   * Print the assignments of tasks to PUs.
   * NOTE: the file is not going to be closed.
   */
  virtual void WriteAssignments(FILE* out_file) = 0;

  /**
   * Print the assignments of tasks to PUs.
   */
  void WriteAssignments(const string& out_file_name);


  /**
   * NOTE: the file is not going to be closed.
   */
  virtual void WriteFlowGraph(FILE* out_graph_file) = 0;

  /**
   * Writes the flow of the graph.
   * @param out_graph_file the name of the file to which to write the flow
   */
  void WriteFlowGraph(const string& out_graph_file);

  /**
   * Writes the graph to a file using the DIMACS format.
   * @param out_graph_file the name of the file to which to write the graph
   **/
  void WriteGraph(const string& out_graph_file);

  /**
   * Writes the graph to a file using the DIMACS format.
   * @param out_graph_file the file to which to write the graph
   * NOTE: the file is not going to be closed.
   */
  virtual void WriteGraph(FILE* out_graph_file) = 0;

  inline uint32_t get_max_node_id() const {
    return max_node_id_;
  }

  inline uint32_t get_sink_node() {
    return sink_node_;
  }

  inline set<uint32_t>& get_source_nodes() {
    return source_nodes_;
  }
  inline vector<int64_t>& get_potentials() {
    return potentials_;
  }

 protected:
  uint32_t max_node_id_;
  uint32_t num_arcs_;
  uint32_t sink_node_;
  set<uint32_t> source_nodes_;
  vector<int64_t> potentials_;

 private:
  void ReadAddNode(char* new_node_line, FILE* graph_file,
                   bool first_scheduling_iteration);
};

} // namespace flowlessly

#endif // FLOWLESSLY_GRAPH_H
