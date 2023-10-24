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

#ifndef FLOWLESSLY_ADJACENCY_MAP_GRAPH_H
#define FLOWLESSLY_ADJACENCY_MAP_GRAPH_H

#include "graphs/graph.h"

#include <stdint.h>
#include <limits>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "graphs/node.h"
#include "misc/statistics.h"

namespace flowlessly {

class AdjacencyMapGraph : public Graph {
 public:
  explicit AdjacencyMapGraph(Statistics* stats);
  AdjacencyMapGraph(uint32_t max_node_id, uint32_t num_arcs, uint32_t sink_node,
                    const set<uint32_t>& source_nodes,
                    const vector<unordered_map<uint32_t, Arc*> >& arcs,
                    const vector<Node>& nodes,
                    const set<uint32_t>& unused_node_ids,
                    Statistics* stats);
  ~AdjacencyMapGraph();

  Arc* AddArc(uint32_t src_node_id, uint32_t dst_node_id,
              uint32_t min_flow, int32_t capacity,
              int64_t cost, int32_t type);
  void AddNode(uint32_t node_id, int32_t supply, int64_t potential,
               NodeType type, bool first_scheduling_iteration);
  uint32_t AddNode(int32_t supply, int64_t price, NodeType type,
                   bool first_scheduling_iteration);

  void ChangeArc(Arc* arc, uint32_t new_min_flow, int32_t new_capacity,
                 int64_t new_cost, int32_t new_type);
  void ChangeArc(uint32_t src_node_id, uint32_t dst_node_id,
                 uint32_t new_min_flow, int32_t new_capacity, int64_t new_cost,
                 int32_t new_type, bool is_multi_arc, int64_t old_cost);
  void GetMachinePUs(uint32_t machine_node_id, set<uint32_t>* pu_ids);
  Arc* GetRandomArc(uint32_t node_id);
  unordered_multimap<uint32_t, uint32_t>* GetTaskAssignments();
  int64_t GetTotalCost();
  void InitializeGraph();
  bool IsEpsOptimal(int64_t eps);
  bool IsFeasible();
  bool IsInTopologicalOrder(const vector<uint32_t>& node_ids);

  /**
   * Computes the maximum refine potential that can be used in a relabel
   * operation.
   * @param node_id the node id for which to compute the refine potential
   * @param eps current epsilon optimality value
   * @return the maximum refine potential value
   **/
  int64_t MaxRefinePotential(uint64_t node_id, int64_t eps);

  /**
   * Construct a topological order of the admissible graph.
   * @param ordered_nodes vector populated with the ordered nodes
   * @return true if there is a valid topological order
   * (i.e. the admissible graph is a DAG)
   **/
  bool TopologicalSort(vector<uint32_t>* ordered);

  void RemoveArc(Arc* arc);
  void RemoveNode(uint32_t node_id);
  void ScaleDownCosts(int64_t scale_down);
  void ScaleUpCosts(int64_t scale_up);
  void UpdateAdmissibleGraph(const vector<uint32_t>& updated_nodes);
  void WriteAssignments(FILE* out_file);
  void WriteFlowGraph(FILE* out_graph_file);
  void WriteGraph(FILE* out_graph_file);

  inline set<uint32_t>& get_active_node_ids() {
    return active_node_ids_;
  }

  inline vector<unordered_map<uint32_t, Arc*> >& get_admissible_arcs() {
    return admissible_arcs_;
  }

  inline vector<unordered_map<uint32_t, Arc*> >& get_arcs() {
    return arcs_;
  }

  inline vector<Node>& get_nodes() {
    return nodes_;
  }

 private:
  int32_t GetNumAssignedTasksToPU(uint32_t node_id);
  void RemoveArcs(uint32_t node_id);
  bool TopologicalSort(uint32_t node_id, list<uint32_t>* ordered);
  void UpdateNodeTypeOnArcRemoval(uint32_t src_node_id, uint32_t dst_node_id);

  vector<unordered_map<uint32_t, Arc*> > arcs_;
  vector<unordered_map<uint32_t, Arc*> > admissible_arcs_;
  vector<Node> nodes_;
  set<uint32_t> active_node_ids_;
  set<uint32_t> unused_node_ids_;
  uint32_t seed_;
  Statistics* statistics_;
};

}  // namespace flowlessly
#endif  // FLOWLESSLY_ADJACENCY_MAP_GRAPH_H
