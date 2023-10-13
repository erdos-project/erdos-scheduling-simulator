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

#include "misc/utils.h"

#include <boost/heap/binomial_heap.hpp>
#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <utility>

namespace flowlessly {

void BellmanFord(AdjacencyMapGraph* graph, const set<uint32_t>& active_node_ids,
                 vector<uint32_t>* predecessor) {
  const uint32_t max_node_id = graph->get_max_node_id();
  const auto& arcs = graph->get_arcs();
  auto& nodes = graph->get_nodes();
  vector<uint32_t> old_active_ids;
  for (auto& active_node : active_node_ids) {
    nodes[active_node].distance = 0;
    old_active_ids.push_back(active_node);
  }
  for (uint32_t iter = 1;
       iter <= max_node_id && old_active_ids.size();
       ++iter) {
    vector<uint32_t> new_active_ids;
    for (auto& node_id : old_active_ids) {
      if (nodes[node_id].distance < numeric_limits<int32_t>::max()) {
        int64_t src_distance = nodes[node_id].distance;
        for (auto& arc : arcs[node_id]) {
          int64_t arc_cost = arc.second->cost -
            nodes[node_id].potential + nodes[arc.first].potential;
          if (arc.second->residual_cap > 0 &&
              src_distance + arc_cost < nodes[arc.first].distance) {
            new_active_ids.push_back(arc.first);
            nodes[arc.first].distance = src_distance + arc_cost;
            (*predecessor)[arc.first] = node_id;
          }
        }
      }
    }
    old_active_ids = new_active_ids;
  }
}

void BellmanFordWithoutPotentials(AdjacencyMapGraph* graph,
                                  const set<uint32_t>& active_node_ids,
                                  vector<uint32_t>* predecessor) {
  const uint32_t max_node_id = graph->get_max_node_id();
  const auto& arcs = graph->get_arcs();
  auto& nodes = graph->get_nodes();
  vector<uint32_t> old_active_ids;
  for (auto& active_node : active_node_ids) {
    nodes[active_node].distance = 0;
    old_active_ids.push_back(active_node);
  }
  for (uint32_t iter = 1;
       iter <= max_node_id && old_active_ids.size();
       ++iter) {
    vector<uint32_t> new_active_ids;
    for (auto& node_id : old_active_ids) {
      if (nodes[node_id].distance < numeric_limits<int32_t>::max()) {
        int64_t src_distance = nodes[node_id].distance;
        for (auto& arc : arcs[node_id]) {
          if (arc.second->residual_cap > 0 &&
              src_distance + arc.second->cost < nodes[arc.first].distance) {
            new_active_ids.push_back(arc.first);
            nodes[arc.first].distance = src_distance + arc.second->cost;
            (*predecessor)[arc.first] = node_id;
          }
        }
      }
    }
    old_active_ids = new_active_ids;
  }
}

uint32_t DijkstraOptimized(AdjacencyMapGraph* graph,
                           const set<uint32_t>& active_node_ids,
                           vector<uint32_t>* predecessor) {
  const uint32_t max_node_id = graph->get_max_node_id();
  const auto& arcs = graph->get_arcs();
  auto& nodes = graph->get_nodes();
  boost::heap::binomial_heap<pair<int64_t, uint32_t>,
                boost::heap::compare<greater<
                  pair<int64_t, uint32_t> > > > dist_heap;
  // Handles to the heap elements.
  boost::heap::binomial_heap<pair<int64_t, uint32_t>,
                boost::heap::compare<greater<
                  pair<int64_t, uint32_t> > > >::handle_type
    *handles =
    new boost::heap::binomial_heap<pair<int64_t, uint32_t>,
             boost::heap::compare<greater<
               pair<int64_t, uint32_t> > > >::handle_type[max_node_id + 1];
  // ASSUMPTION: Works with the assumption that all the elements of distance are
  // already set to INF.
  for (auto& active_node : active_node_ids) {
    nodes[active_node].distance = 0;
    nodes[active_node].status = VISITING;
    handles[active_node] = dist_heap.push(make_pair(0, active_node));
  }
  while (dist_heap.size() > 0) {
    uint32_t min_node_id = dist_heap.top().second;
    //    CHECK_LE(min_node_id, max_node_id);
    dist_heap.pop();
    nodes[min_node_id].status = VISITED;
    // We've finished visiting a node with negative supply. We can stop the
    // shortest path algorithm because we want to route flow to this node.
    if (nodes[min_node_id].supply < 0) {
      delete [] handles;
      return min_node_id;
    }
    int64_t src_distance = nodes[min_node_id].distance;
    for (auto& arc : arcs[min_node_id]) {
      int64_t arc_cost = arc.second->cost - nodes[min_node_id].potential +
        nodes[arc.first].potential;
      if (arc.second->residual_cap > 0 &&
          src_distance + arc_cost < nodes[arc.first].distance) {
        nodes[arc.first].distance = src_distance + arc_cost;
        if (nodes[arc.first].status == NOT_VISITED) {
          // The node's status is set to VISITING the first time we've computed
          // its distance. On this ocassion we also get a handle to the node's
          // entry in the binomial heap.
          nodes[arc.first].status = VISITING;
          (*predecessor)[arc.first] = min_node_id;
          handles[arc.first] =
            dist_heap.push(make_pair(nodes[arc.first].distance, arc.first));
        } else {
          (*predecessor)[arc.first] = min_node_id;
          dist_heap.update(handles[arc.first],
                           make_pair(nodes[arc.first].distance, arc.first));
        }
      }
    }
  }
  delete [] handles;
  // 0 means that no node has been found.
  return 0;
}

void DijkstraSimple(AdjacencyMapGraph* graph,
                    const set<uint32_t>& active_node_ids,
                    vector<uint32_t>* predecessor) {
  const uint32_t max_node_id = graph->get_max_node_id();
  const auto& arcs = graph->get_arcs();
  auto& nodes = graph->get_nodes();
  vector<bool> node_used(max_node_id + 1, false);
  // ASSUMPTION: Works with the assumption that all the elements of distance are
  // already set to INF.
  for (auto& active_node : active_node_ids) {
    nodes[active_node].distance = 0;
  }
  for (uint32_t iter = 1; iter <= max_node_id; ++iter) {
    int64_t min_node_distance = numeric_limits<int32_t>::max();
    uint32_t min_node_id = 0;
    // Get the closest unused vertex.
    for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
      if (!node_used[node_id] && nodes[node_id].distance < min_node_distance) {
        min_node_distance = nodes[node_id].distance;
        min_node_id = node_id;
      }
    }
    node_used[min_node_id] = true;
    for (auto& arc : arcs[min_node_id]) {
      int64_t arc_cost = arc.second->cost -
        nodes[min_node_id].potential + nodes[arc.first].potential;
      if (arc.second->residual_cap > 0 && nodes[min_node_id].distance +
          arc_cost < nodes[arc.first].distance) {
        nodes[arc.first].distance = nodes[min_node_id].distance + arc_cost;
        (*predecessor)[arc.first] = min_node_id;
      }
    }
  }
}

// Computes max flow over the graph using the Ford-Fulkerson algorithm.
// The Complexity of the algorithm is O(E * F). Where F is the max flow value.
// NOTE: This method changes the graph.
void MaxFlow(AdjacencyMapGraph* graph) {
  const uint32_t max_node_id = graph->get_max_node_id();
  vector<unordered_map<uint32_t, Arc*> >& arcs = graph->get_arcs();
  vector<Node>& nodes = graph->get_nodes();
  vector<int32_t> visited(max_node_id + 1, 0);
  vector<uint32_t> predecessor(max_node_id + 1, 0);
  set<uint32_t>& active_node_ids = graph->get_active_node_ids();
  uint32_t sink_node = graph->get_sink_node();
  bool has_path = true;
  while (has_path) {
    has_path = false;
    queue<uint32_t> to_visit;
    fill(visited.begin(), visited.end(), 0);
    fill(predecessor.begin(), predecessor.end(), 0);
    for (const auto& active_node : active_node_ids) {
      if (nodes[active_node].supply > 0) {
        to_visit.push(active_node);
        visited[active_node] = nodes[active_node].supply;
        break;
      }
    }
    while (!to_visit.empty() && !has_path) {
      uint32_t cur_node = to_visit.front();
      to_visit.pop();
      for (auto& arc : arcs[cur_node]) {
        if (!visited[arc.first] && arc.second->residual_cap > 0) {
          visited[arc.first] = min(arc.second->residual_cap, visited[cur_node]);
          to_visit.push(arc.first);
          predecessor[arc.first] = cur_node;
          if (arc.first == sink_node) {
            has_path = true;
            int32_t min_aux_flow = visited[arc.first];
            for (uint32_t cur_node = arc.first; predecessor[cur_node] > 0;
                 cur_node = predecessor[cur_node]) {
              Arc* cur_arc = arcs[predecessor[cur_node]][cur_node];
              cur_arc->residual_cap -= min_aux_flow;
              cur_arc->reverse_arc->residual_cap += min_aux_flow;
              nodes[predecessor[cur_node]].supply -= min_aux_flow;
              nodes[cur_node].supply += min_aux_flow;
            }
            break;
          }
        }
      }
    }
  }
  // Check that all the supply has been drained.
  for (auto& active_node : active_node_ids) {
    CHECK_EQ(nodes[active_node].supply, 0);
  }
  // NOTE: The method does not update the set of active nodes because
  // the nodes are used in the CycleCancelling algorithm as starting
  // nodes for the BellmanFord algorithm. However, the active_node_ids
  // are cleared at the end of the of CycleCancelling.
}

void ReverseMaxFlow(AdjacencyMapGraph* graph) {
  const uint32_t max_node_id = graph->get_max_node_id();
  auto& admissible_arcs = graph->get_admissible_arcs();
  auto& arcs = graph->get_arcs();
  vector<Node>& nodes = graph->get_nodes();
  vector<int32_t> visited(max_node_id + 1, 0);
  vector<uint32_t> predecessor(max_node_id + 1, 0);
  uint32_t sink_node = graph->get_sink_node();
  set<uint32_t> demand_node_ids;
  for (uint32_t node_id = 1; node_id <= max_node_id; ++node_id) {
    if (nodes[node_id].supply < 0 && node_id != sink_node) {
      demand_node_ids.insert(node_id);
    }
  }
  bool has_path = true;
  while (has_path) {
    has_path = false;
    queue<uint32_t> to_visit;
    fill(visited.begin(), visited.end(), 0);
    fill(predecessor.begin(), predecessor.end(), 0);
    for (const auto& demand_node : demand_node_ids) {
      if (nodes[demand_node].supply < 0 && demand_node != sink_node) {
        to_visit.push(demand_node);
        visited[demand_node] = -nodes[demand_node].supply;
        break;
      }
    }
    while (!to_visit.empty() && !has_path) {
      uint32_t cur_node = to_visit.front();
      to_visit.pop();
      for (auto& arc : arcs[cur_node]) {
        if (!visited[arc.first] && arc.second->reverse_arc->residual_cap > 0) {
          visited[arc.first] = min(arc.second->reverse_arc->residual_cap,
                                   visited[cur_node]);
          to_visit.push(arc.first);
          predecessor[arc.first] = cur_node;
          if (arc.first == sink_node) {
            has_path = true;
            int32_t min_aux_flow = visited[arc.first];
            for (uint32_t cur_node = arc.first; predecessor[cur_node] > 0;
                 cur_node = predecessor[cur_node]) {
              Arc* cur_arc = arcs[predecessor[cur_node]][cur_node];
              cur_arc->residual_cap += min_aux_flow;
              // Add the arc back to the admissible graph.
              admissible_arcs[cur_arc->src_node_id][cur_arc->dst_node_id] =
                cur_arc;
              cur_arc->reverse_arc->residual_cap -= min_aux_flow;
              nodes[predecessor[cur_node]].supply += min_aux_flow;
              nodes[cur_node].supply -= min_aux_flow;
            }
            break;
          }
        }
      }
    }
  }
}

} // namespace flowlessly
