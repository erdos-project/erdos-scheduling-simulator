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

#include "solvers/cycle_cancelling.h"

#include <algorithm>
#include <limits>
#include <set>
#include <vector>

#include "misc/utils.h"

namespace flowlessly {

CycleCancelling::CycleCancelling(AdjacencyMapGraph* graph, Statistics* stats)
  : Solver(stats), graph_(graph) {
}

CycleCancelling::~CycleCancelling() {
  // The solver is not the owner of the graph or stat object.
}

void CycleCancelling::AugmentFlow(const vector<uint32_t>& predecessor,
                                  uint32_t src_node, uint32_t dst_node) {
  vector<unordered_map<uint32_t, Arc*> >& arcs = graph_->get_arcs();
  auto& admissible_arcs = graph_->get_admissible_arcs();
  vector<bool> seen(graph_->get_max_node_id() + 1, false);
  uint32_t cur_node;
  // Detect the node where the cycle ends. The cycle doesn't have to start
  // and end at src_node and dst_node.
  for (cur_node = dst_node;
       !seen[cur_node];
       seen[cur_node] = true, cur_node = predecessor[cur_node]) {}
  CHECK_NE(cur_node, 0) << "We cannot augment a non-existing cycle";
  // Setting the source of the cycle.
  uint32_t cycle_start_node = cur_node;
  // Compute the minimum residual in the cycle.
  int32_t min_flow =
    arcs[predecessor[cycle_start_node]][cycle_start_node]->residual_cap;
  for (cur_node = predecessor[cycle_start_node];
       cur_node != cycle_start_node;
       cur_node = predecessor[cur_node]) {
    min_flow =
      min(min_flow, arcs[predecessor[cur_node]][cur_node]->residual_cap);
  }
  CHECK_GE(min_flow, 1) << "The minimum augmenting flow must be greater than 0";
  // Update the arc closing the cycle.
  Arc* cycle_arc = arcs[predecessor[cycle_start_node]][cycle_start_node];
  cycle_arc->residual_cap -= min_flow;
  cycle_arc->reverse_arc->residual_cap += min_flow;
  if (cycle_arc->residual_cap == 0) {
    // Remove arc from the admissible graph.
    admissible_arcs[cycle_arc->src_node_id].erase(cycle_arc->dst_node_id);
  }
  // Update the other arcs from the cycle.
  for (cur_node = predecessor[cycle_start_node];
       cur_node != cycle_start_node;
       cur_node = predecessor[cur_node]) {
    Arc* arc = arcs[predecessor[cur_node]][cur_node];
    arc->residual_cap -= min_flow;
    arc->reverse_arc->residual_cap += min_flow;
    // We don't have to update the supplies because they'll end
    // up being the same when we finish propagating the changes
    // in the cycle.
    if (arc->residual_cap == 0) {
      // Remove arc from the admissible graph.
      admissible_arcs[arc->src_node_id].erase(arc->dst_node_id);
    }
  }
}

void CycleCancelling::PrepareState() {
}

bool CycleCancelling::RemoveNegativeCycles(
    const vector<uint32_t>& predecessor) {
  const uint32_t max_node_id = graph_->get_max_node_id();
  const auto& nodes = graph_->get_nodes();
  const vector<unordered_map<uint32_t, Arc*> >& arcs = graph_->get_arcs();
  for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
    // If the node is not reachable from an active node then its distance is
    // max int64. We check if the distance is smaller because otherwise we
    // would overflow when we add the reduced_cost.
    if (nodes[node_id].distance < numeric_limits<int64_t>::max()) {
      for (auto& id_arc : arcs[node_id]) {
        if (id_arc.second->residual_cap > 0) {
          int64_t reduced_cost = id_arc.second->cost -
            nodes[node_id].potential + nodes[id_arc.first].potential;
          if (nodes[node_id].distance + reduced_cost <
              nodes[id_arc.first].distance) {
            // Found negative cycle.
            AugmentFlow(predecessor, node_id, id_arc.first);
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Establish a feasible flow x in the network
// while ( Gx contains a negative cycle ) do
//   identify a negative cycle W
//   mr = min(r(i,j)) where (i,j) is part of W
//   augment mr units of flow along the cycle W
//   update Gx
bool CycleCancelling::Run() {
  // TODO(ionel): Think if we can avoid calling ReverseMaxFlow. After we apply
  // the incremental changes we may end up with nodes with negative supply.
  // However, CycleCancelling currently assumes a single sink. We do a reverse
  // max flow in which the sources are nodes with demand in order to send
  // flow from these nodes to the sink. Alternatively, we could support multiple
  // sink so that we would not have to do this.
  ReverseMaxFlow(graph_);
  MaxFlow(graph_);
  CHECK(graph_->IsFeasible()) << "The graph is not feasible";
  const uint32_t max_node_id = graph_->get_max_node_id();
  auto& nodes = graph_->get_nodes();
  vector<uint32_t> predecessor(max_node_id + 1, 0);
  set<uint32_t> sink_nodes;
  sink_nodes.insert(graph_->get_sink_node());
  for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
    nodes[node_id].distance = numeric_limits<int64_t>::max();
  }
  // As a result of MaxFlow the sources might not be connected to the graph
  // anymore. We use the sink as the source of Bellman-Ford. The sink
  // is guaranteed to be connected.
  BellmanFordWithoutPotentials(graph_, sink_nodes, &predecessor);
  bool removed_cycle = RemoveNegativeCycles(predecessor);
  while (removed_cycle) {
    for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
      nodes[node_id].distance = numeric_limits<int64_t>::max();
      predecessor[node_id] = 0;
    }
    // CycleCancelling doesn't make use of node's potentials. Hence, we can use
    // a variant of BellmanFord that is faster.
    BellmanFordWithoutPotentials(graph_, sink_nodes, &predecessor);
    removed_cycle = RemoveNegativeCycles(predecessor);
  }
  // There are no active nodes left. We can now update the set of active nodes.
  graph_->get_active_node_ids().clear();
  return true;
}

} // namespace flowlessly
