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

#include "solvers/successive_shortest.h"

#include <algorithm>
#include <limits>
#include <set>

#include "misc/utils.h"

DEFINE_string(successive_shortest, "dijkstra",
              "The algorithm to use for shortet path. "
              "Options: bellman | dijkstra");

namespace flowlessly {

SuccessiveShortest::SuccessiveShortest(AdjacencyMapGraph* graph,
                                       Statistics* stats)
  : Solver(stats), graph_(graph) {
}

SuccessiveShortest::~SuccessiveShortest() {
  // The solver is not the owner of the graph or stat object.
}

void SuccessiveShortest::PrepareState() {
}

bool SuccessiveShortest::Run() {
  if (!FLAGS_successive_shortest.compare("bellman")) {
    SuccessiveShortestBellman();
  } else if (!FLAGS_successive_shortest.compare("dijkstra")) {
    SuccessiveShortestDijkstra();
  } else {
    LOG(FATAL) << "Unexpected successive shortest algorithm: "
               << FLAGS_successive_shortest;
  }
  return true;
}

void SuccessiveShortest::SuccessiveShortestBellman() {
  // Initial flow x is zero
  // while ( Gx contains a path from s to t ) do
  //   Find any shortest path P from s to t
  //   Augment current flow x along P
  //   update Gx
  VLOG(2) << "Using successive shortest path with Bellman-Ford";
  const uint32_t max_node_id = graph_->get_max_node_id();
  vector<unordered_map<uint32_t, Arc*> >& arcs = graph_->get_arcs();
  auto& admissible_arcs = graph_->get_admissible_arcs();
  vector<Node>& nodes = graph_->get_nodes();
  set<uint32_t> active_node_ids = graph_->get_active_node_ids();
  vector<uint32_t> predecessor(max_node_id + 1, 0);

  while (active_node_ids.size() > 0) {
    for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
      nodes[node_id].distance = numeric_limits<int64_t>::max();
      predecessor[node_id] = 0;
    }
    BellmanFord(graph_, active_node_ids, &predecessor);

    uint32_t demand_node_id = 0;
    for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
      if (nodes[node_id].distance < numeric_limits<int64_t>::max()) {
        nodes[node_id].potential -= nodes[node_id].distance;
        if (demand_node_id == 0 && nodes[node_id].supply < 0) {
          demand_node_id = node_id;
        }
      }
    }
    CHECK_NE(demand_node_id, 0) << "There's no reachable demand node";

    int32_t min_flow = numeric_limits<int32_t>::max();
    uint32_t cur_node = demand_node_id;
    for (; active_node_ids.find(cur_node) == active_node_ids.end();
         cur_node = predecessor[cur_node]) {
      Arc* arc = arcs[predecessor[cur_node]][cur_node];
      min_flow = min(min_flow, arc->residual_cap);
    }
    min_flow = min(min_flow, nodes[cur_node].supply);
    // Update the supply at the demand node.
    nodes[demand_node_id].supply += min_flow;
    for (cur_node = demand_node_id;
         active_node_ids.find(cur_node) == active_node_ids.end();
         cur_node = predecessor[cur_node]) {
      Arc* arc = arcs[predecessor[cur_node]][cur_node];
      // We don't have to update the supplies along the path. Only
      // the supplies of the source and destionation nodes change.
      arc->residual_cap -= min_flow;
      arc->reverse_arc->residual_cap += min_flow;
      // After the potential update the reduced cost of all the arcs
      // on the shortest path to a demand node is 0. We can remove the arc
      // from the admissible_arcs.
      admissible_arcs[arc->src_node_id].erase(arc->dst_node_id);
      // According to the algorithm's properties the reverse arc will have
      // a 0 reduced cost. We don't have to add it to the addmissble arcs set.
    }
    nodes[cur_node].supply -= min_flow;
    if (nodes[cur_node].supply == 0) {
      active_node_ids.erase(cur_node);
    }
  }
}

// ASSUMPTION: The graph respects reduced cost optimality
// (i.e., reduced_cost >= 0 for every arc).
void SuccessiveShortest::SuccessiveShortestDijkstra() {
  VLOG(2) << "Using successive shortest path with Dijkstra";
  const uint32_t max_node_id = graph_->get_max_node_id();
  vector<unordered_map<uint32_t, Arc*> >& arcs = graph_->get_arcs();
  auto& admissible_arcs = graph_->get_admissible_arcs();
  vector<Node>& nodes = graph_->get_nodes();
  set<uint32_t>& active_node_ids = graph_->get_active_node_ids();
  vector<uint32_t> predecessor(max_node_id + 1, 0);

  while (active_node_ids.size() > 0) {
    for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
      nodes[node_id].distance = numeric_limits<int64_t>::max();
      nodes[node_id].status = NOT_VISITED;
      predecessor[node_id] = 0;
    }
    uint32_t demand_node_id =
      DijkstraOptimized(graph_, active_node_ids, &predecessor);
    for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
      if (nodes[node_id].status == VISITED) {
        nodes[node_id].potential -= nodes[node_id].distance;
      } else {
        nodes[node_id].potential -= nodes[demand_node_id].distance;
      }
    }
    CHECK_NE(demand_node_id, 0) << "There's no reachable demand node";

    int32_t min_flow = numeric_limits<int32_t>::max();
    uint32_t cur_node = demand_node_id;
    for (; active_node_ids.find(cur_node) == active_node_ids.end();
         cur_node = predecessor[cur_node]) {
      Arc* arc = arcs[predecessor[cur_node]][cur_node];
      min_flow = min(min_flow, arc->residual_cap);
    }
    min_flow = min(min_flow, nodes[cur_node].supply);
    // Update the supply at the demand node.
    nodes[demand_node_id].supply += min_flow;
    for (cur_node = demand_node_id;
         active_node_ids.find(cur_node) == active_node_ids.end();
         cur_node = predecessor[cur_node]) {
      Arc* arc = arcs[predecessor[cur_node]][cur_node];
      // We don't have to update the supplies along the path. Only
      // the supplies of the source and destionation nodes change.
      arc->residual_cap -= min_flow;
      arc->reverse_arc->residual_cap += min_flow;
      // After the potential update the reduced cost of all the arcs
      // on the shortest path to a demand node is 0. We can remove the arc
      // from the admissible_arcs.
      admissible_arcs[arc->src_node_id].erase(arc->dst_node_id);
      // According to the algorithm's properties the reverse arc will have
      // a 0 reduced cost. We don't have to add it to the addmissble arcs set.
    }
    // Update the supply at the source active node.
    nodes[cur_node].supply -= min_flow;
    if (nodes[cur_node].supply == 0) {
      active_node_ids.erase(cur_node);
    }
  }
}

} // namespace flowlessly
