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

#include "graphs/adjacency_map_graph.h"

#include <boost/algorithm/string.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
#include <queue>
#include <stack>
#include <string>

#include "misc/utils.h"

DECLARE_bool(graph_has_node_types);

namespace flowlessly {

AdjacencyMapGraph::AdjacencyMapGraph(Statistics* stats) : seed_(42),
                                                          statistics_(stats) {
  // Node ids start from 1. We have to push an empty node at index 0 so that
  // push_back adds a node with id 1.
  nodes_.push_back(Node());
  arcs_.push_back(unordered_map<uint32_t, Arc*>());
  admissible_arcs_.push_back(unordered_map<uint32_t, Arc*>());
}

AdjacencyMapGraph::AdjacencyMapGraph(
    uint32_t max_node_id, uint32_t num_arcs, uint32_t sink_node,
    const set<uint32_t>& source_nodes,
    const vector<unordered_map<uint32_t, Arc*> >& arcs,
    const vector<Node>& nodes,
    const set<uint32_t>& unused_node_ids,
    Statistics* stats)
  : Graph(max_node_id, num_arcs, sink_node, source_nodes), seed_(42),
    statistics_(stats) {
  for (uint32_t node_id = 0; node_id < nodes.size(); node_id++) {
    nodes_.push_back(nodes[node_id]);
    if (nodes_[node_id].supply > 0) {
      active_node_ids_.insert(node_id);
    }
    arcs_.push_back(unordered_map<uint32_t, Arc*>());
    admissible_arcs_.push_back(unordered_map<uint32_t, Arc*>());
  }
  for (const auto& node_arcs : arcs) {
    for (const auto& dst_arc : node_arcs) {
      if (dst_arc.second->is_fwd) {
        Arc* new_arc = new Arc(dst_arc.second);
        Arc* rev_new_arc = new Arc(dst_arc.second->reverse_arc);
        new_arc->reverse_arc = rev_new_arc;
        rev_new_arc->reverse_arc = new_arc;
        arcs_[new_arc->src_node_id][new_arc->dst_node_id] = new_arc;
        arcs_[new_arc->dst_node_id][new_arc->src_node_id] = rev_new_arc;
        int64_t reduced_cost = new_arc->cost -
          nodes_[new_arc->src_node_id].potential +
          nodes_[new_arc->dst_node_id].potential;
        if (reduced_cost < 0 && new_arc->residual_cap > 0) {
          admissible_arcs_[new_arc->src_node_id][new_arc->dst_node_id] =
            new_arc;
        }
        if (-reduced_cost < 0 && rev_new_arc->residual_cap > 0) {
          admissible_arcs_[new_arc->dst_node_id][new_arc->src_node_id] =
            rev_new_arc;
        }
      }
    }
  }
  unused_node_ids_ = unused_node_ids;
}

AdjacencyMapGraph::~AdjacencyMapGraph() {
  // Delete every arc.
  for (uint32_t node_id = 1; node_id <= max_node_id_; node_id++) {
    for (auto& id_arc : arcs_[node_id]) {
      delete id_arc.second;
    }
  }
  // AdjacencyMapGraph does not own the stats_ object.
}

Arc* AdjacencyMapGraph::AddArc(uint32_t src_node_id, uint32_t dst_node_id,
                               uint32_t min_flow, int32_t capacity,
                               int64_t cost, int32_t type) {
  VLOG(2) << "Adding arc (" << src_node_id << ", " << dst_node_id << ") with "
          << "min flow " << min_flow << ", capacity " << capacity << ", cost "
          << cost;
  CHECK_LE(min_flow, capacity);
  CHECK_LE(src_node_id, max_node_id_);
  CHECK_LE(dst_node_id, max_node_id_);
  // Check the arc doesn't already exist.
  if (arcs_[src_node_id].find(dst_node_id) !=
      arcs_[src_node_id].end()) {
    LOG(FATAL) << "The arc (" << src_node_id << ", " << dst_node_id
               << ") already exists";
  }
  num_arcs_++;
  // Send min_flow across the arc.
  int32_t flow = static_cast<int32_t>(min_flow);
  nodes_[src_node_id].supply -= flow;
  nodes_[dst_node_id].supply += flow;
  Arc* arc = new Arc(src_node_id, dst_node_id, true, false, capacity - min_flow,
                     min_flow, cost, NULL);
  // We don't have to increase the capacity on the reverse arc because
  // we do not want to be able to push back the minimum flow.
  Arc* reverse_arc =
    new Arc(dst_node_id, src_node_id, false, false, 0, 0, -cost, arc);
  arc->reverse_arc = reverse_arc;
  arcs_[src_node_id][dst_node_id] = arc;
  arcs_[dst_node_id][src_node_id] = reverse_arc;

  int64_t reduced_cost = arc->cost - nodes_[arc->src_node_id].potential +
    nodes_[arc->dst_node_id].potential;
  // Push flow if reduced cost is < 0.
  if (reduced_cost < 0 && arc->residual_cap > 0) {
    nodes_[src_node_id].supply -= arc->residual_cap;
    nodes_[dst_node_id].supply += arc->residual_cap;
    arc->reverse_arc->residual_cap += arc->residual_cap;
    arc->residual_cap = 0;
  }
  if (-reduced_cost < 0 && arc->reverse_arc->residual_cap > 0) {
    nodes_[src_node_id].supply += arc->reverse_arc->residual_cap;
    nodes_[dst_node_id].supply -= arc->reverse_arc->residual_cap;
    arc->residual_cap += arc->reverse_arc->residual_cap;
    arc->reverse_arc->residual_cap = 0;
  }
  // Update active nodes set.
  if (nodes_[src_node_id].supply > 0) {
    active_node_ids_.insert(src_node_id);
  } else {
    active_node_ids_.erase(src_node_id);
  }
  if (nodes_[dst_node_id].supply > 0) {
    active_node_ids_.insert(dst_node_id);
  } else {
    active_node_ids_.erase(dst_node_id);
  }
  if (type == 1) {
    arc->is_running = true;
    arc->reverse_arc->is_running = true;
  } else {
    arc->is_running = false;
    arc->reverse_arc->is_running = false;
  }
  return arc;
}

void AdjacencyMapGraph::AddNode(uint32_t node_id,
                                int32_t supply,
                                int64_t potential,
                                NodeType type,
                                bool first_scheduling_iteration) {
  VLOG(2) << "Adding node " << node_id << " with supply " << supply
          << ", potential " << potential << ", type " << type;
  while (max_node_id_ < node_id) {
    ++max_node_id_;
    Node new_node;
    new_node.potential = 0;
    new_node.supply = 0;
    new_node.type = OTHER;
    new_node.potential = 0;
    nodes_.push_back(new_node);
    arcs_.push_back(unordered_map<uint32_t, Arc*>());
    admissible_arcs_.push_back(unordered_map<uint32_t, Arc*>());
    unused_node_ids_.insert(max_node_id_);
  }
  if (unused_node_ids_.find(node_id) == unused_node_ids_.end()) {
    VLOG(2) << "Node id " << node_id << " already used. Updating it";
  } else {
    unused_node_ids_.erase(node_id);
  }
  nodes_[node_id].potential = potential;
  nodes_[node_id].supply = supply;
  nodes_[node_id].type = type;
  nodes_[node_id].status = NOT_VISITED;
  if (type == SINK) {
    sink_node_ = node_id;
  }
  if (supply < 0 && !FLAGS_graph_has_node_types) {
    CHECK_EQ(sink_node_, 0)
      << "Found mode than one sink while processing node: " << node_id;
    sink_node_ = node_id;
    nodes_[node_id].type = SINK;
  } else if (supply > 0) {
    CHECK_NE(node_id, sink_node_)
      << "Cannot update the sink node to a positive supply";
    active_node_ids_.insert(node_id);
    source_nodes_.insert(node_id);
    if (!first_scheduling_iteration) {
      CHECK_NE(sink_node_, 0) << "Sink node is not initialized";
      nodes_[sink_node_].supply -= supply;
      if (nodes_[sink_node_].supply <= 0) {
        active_node_ids_.erase(sink_node_);
      }
    }
  }
}

uint32_t AdjacencyMapGraph::AddNode(int32_t supply,
                                    int64_t potential,
                                    NodeType type,
                                    bool first_scheduling_iteration) {
  VLOG(2) << "Adding node with supply " << supply
          << ", potential " << potential << ", type " << type;
  uint32_t new_node_id;
  if (unused_node_ids_.empty()) {
    ++max_node_id_;
    new_node_id = max_node_id_;
    Node new_node;
    new_node.potential = potential;
    new_node.supply = supply;
    new_node.type = type;
    new_node.status = NOT_VISITED;
    nodes_.push_back(new_node);
    arcs_.push_back(unordered_map<uint32_t, Arc*>());
    admissible_arcs_.push_back(unordered_map<uint32_t, Arc*>());
  } else {
    set<uint32_t>::iterator it = unused_node_ids_.begin();
    new_node_id = *it;
    unused_node_ids_.erase(it);
    nodes_[new_node_id].potential = potential;
    nodes_[new_node_id].supply = supply;
    nodes_[new_node_id].type = type;
    nodes_[new_node_id].status = NOT_VISITED;
  }
  VLOG(2) << "New node allocated id " << new_node_id;
  if (supply < 0 && !FLAGS_graph_has_node_types) {
    CHECK_EQ(sink_node_, 0)
      << "Found mode than one sink while processing node: " << new_node_id;
    sink_node_ = new_node_id;
    nodes_[sink_node_].type = SINK;
  } else if (supply > 0) {
    active_node_ids_.insert(new_node_id);
    source_nodes_.insert(new_node_id);
    if (!first_scheduling_iteration) {
      CHECK_NE(sink_node_, 0) << "Sink node is not initialized";
      nodes_[sink_node_].supply -= supply;
    }
  }
  if (type == SINK) {
    sink_node_ = new_node_id;
  }
  return new_node_id;
}

void AdjacencyMapGraph::ChangeArc(Arc* arc, uint32_t new_min_flow,
                                  int32_t new_capacity, int64_t new_cost,
                                  int32_t new_type) {
  VLOG(2) << "Changing arc (" << arc->src_node_id << ", " << arc->dst_node_id
          << ") to an arc with min flow " << new_min_flow << ", capacity "
          << new_capacity << ", cost " << new_cost;
  CHECK(arc->is_fwd) << "Cannot change a reverse arc "
                     << arc->src_node_id << " " << arc->dst_node_id;

  if (new_capacity == 0) {
    RemoveArc(arc);
    return;
  }

  int32_t previous_cap =
    arc->residual_cap + arc->reverse_arc->residual_cap + arc->min_flow;

  // Update the min flow on the arc.
  if (new_min_flow <= arc->min_flow) {
    uint32_t diff_flow = arc->min_flow - new_min_flow;
    nodes_[arc->src_node_id].supply += diff_flow;
    nodes_[arc->dst_node_id].supply -= diff_flow;
    arc->residual_cap += diff_flow;
  } else {
    uint32_t diff_flow = new_min_flow - arc->min_flow;
    nodes_[arc->src_node_id].supply -= diff_flow;
    nodes_[arc->dst_node_id].supply += diff_flow;
    arc->residual_cap -= diff_flow;
  }
  arc->min_flow = new_min_flow;

  // Update the capacity on the arc. Send back some flow if required
  // to keep it within the capacity limit.
  if (previous_cap < new_capacity) {
    arc->residual_cap += new_capacity - previous_cap;
  } else {
    int32_t diff_cap = previous_cap - new_capacity;
    if (arc->residual_cap < diff_cap) {
      // Drain the excess flow from the arc.
      int32_t removed_flow = diff_cap - arc->residual_cap;
      nodes_[arc->src_node_id].supply += removed_flow;
      nodes_[arc->dst_node_id].supply -= removed_flow;
      arc->reverse_arc->residual_cap -= removed_flow;
      arc->residual_cap = 0;
    } else {
      arc->residual_cap -= diff_cap;
    }
  }
  arc->cost = new_cost;
  arc->reverse_arc->cost = -new_cost;

  int64_t reduced_cost = arc->cost - nodes_[arc->src_node_id].potential +
    nodes_[arc->dst_node_id].potential;
  // Push flow if reduced cost is < 0.
  if (reduced_cost < 0 && arc->residual_cap > 0) {
    nodes_[arc->src_node_id].supply -= arc->residual_cap;
    nodes_[arc->dst_node_id].supply += arc->residual_cap;
    arc->reverse_arc->residual_cap += arc->residual_cap;
    arc->residual_cap = 0;
  }
  if (-reduced_cost < 0 && arc->reverse_arc->residual_cap > 0) {
    nodes_[arc->src_node_id].supply += arc->reverse_arc->residual_cap;
    nodes_[arc->dst_node_id].supply -= arc->reverse_arc->residual_cap;
    arc->residual_cap += arc->reverse_arc->residual_cap;
    arc->reverse_arc->residual_cap = 0;
  }
  // Update the active node ids set.
  if (nodes_[arc->src_node_id].supply > 0) {
    active_node_ids_.insert(arc->src_node_id);
  } else {
    active_node_ids_.erase(arc->src_node_id);
  }
  if (nodes_[arc->dst_node_id].supply > 0) {
    active_node_ids_.insert(arc->dst_node_id);
  } else {
    active_node_ids_.erase(arc->dst_node_id);
  }
  if (new_type == 1) {
    arc->is_running = true;
    arc->reverse_arc->is_running = true;
  } else {
    arc->is_running = false;
    arc->reverse_arc->is_running = false;
  }
  CHECK_GE(arc->reverse_arc->residual_cap, 0);
  CHECK_EQ(arc->residual_cap + arc->reverse_arc->residual_cap + arc->min_flow,
           new_capacity);
}

void AdjacencyMapGraph::ChangeArc(uint32_t src_node_id, uint32_t dst_node_id,
                                  uint32_t new_min_flow, int32_t new_capacity,
                                  int64_t new_cost, int32_t new_type,
                                  bool is_multi_arc, int64_t old_cost) {
  if (arcs_[src_node_id].find(dst_node_id) != arcs_[src_node_id].end()) {
    ChangeArc(arcs_[src_node_id][dst_node_id], new_min_flow, new_capacity,
              new_cost, new_type);
  } else {
    AddArc(src_node_id, dst_node_id, new_min_flow, new_capacity, new_cost,
           new_type);
  }
}

void AdjacencyMapGraph::GetMachinePUs(uint32_t machine_node_id,
                                      set<uint32_t>* pu_ids) {
  for (auto& id_arc : arcs_[machine_node_id]) {
    if (nodes_[id_arc.first].type == PU) {
      pu_ids->insert(id_arc.first);
    }
  }
}

int32_t AdjacencyMapGraph::GetNumAssignedTasksToPU(uint32_t node_id) {
  int32_t num_assigned_tasks = 0;
  for (auto& id_arc : arcs_[node_id]) {
    Arc* arc = id_arc.second;
    // Increase the num_assigned_tasks by the residual capacity of the reverse
    // arc of the arc going to the sink.
    if (arc->is_fwd && nodes_[arc->dst_node_id].type == SINK) {
      num_assigned_tasks += arc->reverse_arc->residual_cap + arc->min_flow;
    }
  }
  return num_assigned_tasks;
}

Arc* AdjacencyMapGraph::GetRandomArc(uint32_t node_id) {
  auto arc_it = arcs_[node_id].begin();
  advance(arc_it, rand_r(&seed_) % arcs_[node_id].size());
  Arc* arc = arc_it->second;
  if (!arc->is_fwd) {
    return NULL;
  }
  return arc;
}

unordered_multimap<uint32_t, uint32_t>*
    AdjacencyMapGraph::GetTaskAssignments() {
  vector<queue<uint32_t> > pu_ids(max_node_id_ + 1);
  queue<uint32_t> to_visit;
  unordered_multimap<uint32_t, uint32_t>* task_mappings =
    new unordered_multimap<uint32_t, uint32_t>();
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    nodes_[node_id].status = NOT_VISITED;
    if (nodes_[node_id].type == PU) {
      // Add the not visited PU node.
      to_visit.push(node_id);
      nodes_[node_id].status = VISITING;
      // Add the node_id num_assigned_tasks to the pu_ids queue.
      for (int32_t num_assigned_tasks = GetNumAssignedTasksToPU(node_id);
           num_assigned_tasks > 0;
           --num_assigned_tasks) {
        pu_ids[node_id].push(node_id);
      }
    }
  }

  while (!to_visit.empty()) {
    uint32_t node_id = to_visit.front();
    to_visit.pop();
    nodes_[node_id].status = VISITED;
    if (nodes_[node_id].type != TASK) {
      // Iterate over arcs and add to the queue the nodes we can reach via a
      // reverse arc with positive residual capacity.
      for (auto& id_arc : arcs_[node_id]) {
        if (pu_ids[node_id].empty()) {
          break;
        }
        Arc* arc = id_arc.second;
        if (!arc->is_fwd) {
          uint32_t arc_flow = arc->residual_cap + arc->reverse_arc->min_flow;
          if (arc_flow > 0) {
            while (arc_flow > 0 && !pu_ids[node_id].empty()) {
              // Move the first id from node_id's queue to arc->dst_node_id's
              // queue.
              pu_ids[arc->dst_node_id].push(pu_ids[node_id].front());
              pu_ids[node_id].pop();
              arc_flow--;
            }
            if (nodes_[arc->dst_node_id].status != VISITING) {
              to_visit.push(arc->dst_node_id);
              nodes_[arc->dst_node_id].status = VISITING;
            }
          }
        }
      }
    } else {
      while (!pu_ids[node_id].empty()) {
        uint32_t pu_node_id = pu_ids[node_id].front();
        pu_ids[node_id].pop();
        task_mappings->insert(make_pair(node_id, pu_node_id));
      }
    }
  }
  return task_mappings;
}

int64_t AdjacencyMapGraph::GetTotalCost() {
  int64_t total_cost = 0;
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    for (auto& id_arc : arcs_[node_id]) {
      if (id_arc.second->is_fwd) {
        total_cost += id_arc.second->cost *
          (id_arc.second->reverse_arc->residual_cap + id_arc.second->min_flow);
      }
    }
  }
  VLOG(2) << "Total cost: " << total_cost;
  return total_cost;
}

void AdjacencyMapGraph::InitializeGraph() {
  arcs_.resize(max_node_id_ + 1);
  admissible_arcs_.resize(max_node_id_ + 1);
  nodes_.resize(max_node_id_ + 1);
  // All the nodes up to max_node_id_ are part of the graph. The
  // DIMACS format only required non-zero supply nodes to be
  // explicitly passed.
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    nodes_[node_id].supply = 0;
    nodes_[node_id].potential = 0;
    nodes_[node_id].type = OTHER;
    nodes_[node_id].status = NOT_VISITED;
  }
}

bool AdjacencyMapGraph::IsEpsOptimal(int64_t eps) {
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    if (nodes_[node_id].supply != 0) {
      LOG(ERROR) << "Node " << node_id << " has supply "
                 << nodes_[node_id].supply;
      return false;
    }
    for (auto& id_arc : arcs_[node_id]) {
      if (id_arc.second->residual_cap > 0 && id_arc.second->cost -
          nodes_[node_id].potential + nodes_[id_arc.first].potential < -eps) {
        LOG(ERROR) << "Arc " << node_id << " " << id_arc.first
                   << " has a cost of "
                   << id_arc.second->cost - nodes_[node_id].potential +
          nodes_[id_arc.first].potential;
        return false;
      }
    }
  }
  return true;
}

bool AdjacencyMapGraph::IsFeasible() {
  int64_t total_supply = 0;
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    if (unused_node_ids_.find(node_id) == unused_node_ids_.end()) {
      total_supply += nodes_[node_id].supply;
    }
    for (auto& id_arc : arcs_[node_id]) {
      if (id_arc.second->is_fwd &&
          (id_arc.second->residual_cap < 0 ||
           id_arc.second->residual_cap +
           id_arc.second->reverse_arc->residual_cap < 0)) {
        LOG(ERROR) << "Graph is not feasible. The flow is not within the bounds"
                   << " on arc (" << id_arc.second->src_node_id << ", "
                   << id_arc.second->dst_node_id << ")";
        return false;
      }
    }
  }
  if (total_supply != 0) {
    LOG(ERROR) << "Total supply is non-zero: " << total_supply;
    return false;
  }
  return true;
}

bool AdjacencyMapGraph::IsInTopologicalOrder(const vector<uint32_t>& node_ids) {
  vector<uint32_t> position(max_node_id_ + 1);
  for (uint32_t index = 0; index < node_ids.size(); ++index) {
    position[node_ids[index]] = index;
  }
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    for (auto& id_arc : admissible_arcs_[node_id]) {
      if (position[node_id] >= position[id_arc.first]) {
        LOG(WARNING) << "Node " << node_id << " should be before "
                     << id_arc.first;;
        return false;
      }
    }
  }
  return true;
}

int64_t AdjacencyMapGraph::MaxRefinePotential(uint64_t node_id, int64_t eps) {
  int64_t new_pot = numeric_limits<int64_t>::max();
  // Find the maximum value to adjust the potential of node_id such that
  // all the reduced arc costs are >= -eps.
  for (auto& id_arc : arcs_[node_id]) {
    if (id_arc.second->residual_cap > 0) {
      new_pot = min(new_pot,
                    id_arc.second->cost + nodes_[id_arc.first].potential);
    }
  }
  if (new_pot == numeric_limits<int64_t>::max()) {
    // Ensure that the potential will increase by eps if the node
    // has no outgoing arcs with positive residual capacity.
    new_pot = nodes_[node_id].potential;
  }
  return -nodes_[node_id].potential + new_pot + eps;
}

void AdjacencyMapGraph::RemoveArc(Arc* arc) {
  VLOG(2) << "Removing arc (" << arc->src_node_id << ", "
          << arc->dst_node_id << ")";
  CHECK(arc->is_fwd) << "Trying to remove a non-foward arc";
  if (nodes_[arc->src_node_id].type == SINK ||
      nodes_[arc->dst_node_id].type == SINK) {
    UpdateNodeTypeOnArcRemoval(arc->src_node_id, arc->dst_node_id);
  }
  int32_t flow = arc->reverse_arc->residual_cap + arc->min_flow;
  nodes_[arc->src_node_id].supply += flow;
  nodes_[arc->dst_node_id].supply -= flow;
  if (nodes_[arc->src_node_id].supply > 0 &&
      nodes_[arc->src_node_id].supply <= flow) {
    active_node_ids_.insert(arc->src_node_id);
  }
  if (nodes_[arc->dst_node_id].supply <= 0 &&
      nodes_[arc->dst_node_id].supply > -flow) {
    active_node_ids_.erase(arc->dst_node_id);
  }
  num_arcs_--;
  arcs_[arc->dst_node_id].erase(arc->src_node_id);
  arcs_[arc->src_node_id].erase(arc->dst_node_id);
  admissible_arcs_[arc->src_node_id].erase(arc->dst_node_id);
  admissible_arcs_[arc->dst_node_id].erase(arc->src_node_id);

  delete arc->reverse_arc;
  delete arc;
}

void AdjacencyMapGraph::RemoveArcs(uint32_t node_id) {
  // Delete arcs for the arcs and admissible arcs.
  for (auto it = arcs_[node_id].begin(); it != arcs_[node_id].end(); ) {
    auto to_delete_it = it;
    ++it;
    if (to_delete_it->second->is_fwd) {
      RemoveArc(to_delete_it->second);
    } else {
      RemoveArc(to_delete_it->second->reverse_arc);
    }
  }
  arcs_[node_id].clear();
  admissible_arcs_[node_id].clear();
}

void AdjacencyMapGraph::RemoveNode(uint32_t node_id) {
  VLOG(2) << "Removing node " << node_id;
  CHECK_LE(node_id, max_node_id_);
  RemoveArcs(node_id);
  CHECK_NE(sink_node_, 0) << "Sink node is not set";
  if (nodes_[node_id].supply > 0) {
    active_node_ids_.erase(node_id);
    source_nodes_.erase(node_id);
  }
  if (sink_node_ == node_id) {
    LOG(INFO) << "Removing sink node: " << sink_node_;
    sink_node_ = 0;
  } else {
    nodes_[sink_node_].supply += nodes_[node_id].supply;
    if (nodes_[sink_node_].supply > 0) {
      active_node_ids_.insert(sink_node_);
    } else {
      active_node_ids_.erase(sink_node_);
    }
  }
  nodes_[node_id].supply = 0;
  nodes_[node_id].potential = 0;
  nodes_[node_id].type = OTHER;
  nodes_[node_id].status = NOT_VISITED;
  unused_node_ids_.insert(node_id);
}

void AdjacencyMapGraph::ScaleDownCosts(int64_t scale_down) {
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    for (auto& id_arc : arcs_[node_id]) {
      id_arc.second->cost /= scale_down;
    }
  }
}

void AdjacencyMapGraph::ScaleUpCosts(int64_t scale_up) {
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    for (auto& id_arc : arcs_[node_id]) {
      id_arc.second->cost *= scale_up;
    }
  }
}

// Implements Tarjan's algorithm.
bool AdjacencyMapGraph::TopologicalSort(vector<uint32_t>* ordered) {
  stack<uint32_t> to_visit;
  list<uint32_t> ordered_list;
  // Reset the nodes to the NOT_VISITED state.
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    nodes_[node_id].status = NOT_VISITED;
  }
  // It does not matter the order in which we visit the nodes.
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    if (nodes_[node_id].status == NOT_VISITED) {
      bool no_cycle = TopologicalSort(node_id, &ordered_list);
      if (!no_cycle) {
        return false;
      }
    }
  }
  for (auto& arc_id : ordered_list) {
    ordered->push_back(arc_id);
  }
  return true;
}

bool AdjacencyMapGraph::TopologicalSort(uint32_t node_id,
                                        list<uint32_t>* ordered) {
  if (nodes_[node_id].status == VISITING) {
    // We're closing a cycle => there's no topological order.
    return false;
  } else {
    CHECK_NE(nodes_[node_id].status, VISITED);
    nodes_[node_id].status = VISITING;
    for (auto& id_arc : admissible_arcs_[node_id]) {
      if (nodes_[id_arc.first].status != VISITED) {
        bool no_cycle = TopologicalSort(id_arc.first, ordered);
        if (!no_cycle) {
          return false;
        }
      }
    }
    // The node comes before all the nodes we have finished visiting up to now.
    nodes_[node_id].status = VISITED;
    (*ordered).push_front(node_id);
  }
  return true;
}

void AdjacencyMapGraph::UpdateAdmissibleGraph(
    const vector<uint32_t>& updated_nodes) {
  for (const auto& node_id : updated_nodes) {
    int64_t src_node_potential = nodes_[node_id].potential;
    for (auto& id_arc : arcs_[node_id]) {
      int64_t reduced_cost = id_arc.second->cost - src_node_potential +
        nodes_[id_arc.first].potential;
      if (id_arc.second->residual_cap > 0 && reduced_cost < 0) {
        admissible_arcs_[node_id][id_arc.first] = id_arc.second;
      } else {
        admissible_arcs_[node_id].erase(id_arc.first);
      }
      Arc* rev_arc = id_arc.second->reverse_arc;
      if (rev_arc->residual_cap > 0 && -reduced_cost < 0) {
        admissible_arcs_[id_arc.first][node_id] = rev_arc;
      } else {
        admissible_arcs_[id_arc.first].erase(node_id);
      }
    }
  }
}

void AdjacencyMapGraph::UpdateNodeTypeOnArcRemoval(uint32_t src_node_id,
                                                   uint32_t dst_node_id) {
  // Swap the nodes such that dst_node_id is the SINK node.
  if (nodes_[src_node_id].type == SINK) {
    uint32_t tmp_node_id = dst_node_id;
    dst_node_id = src_node_id;
    src_node_id = tmp_node_id;
  }
  uint32_t num_arcs_to_sinks = 0;
  for (auto& id_arc : arcs_[src_node_id]) {
    if (nodes_[id_arc.first].type == SINK) {
      num_arcs_to_sinks++;
    }
  }
  if (num_arcs_to_sinks == 1) {
    // The node can't be a PU node anymore because it will not have any arcs
    // to sinks after we remove the arc between src_node_id and dst_node_id.
    nodes_[src_node_id].type = OTHER;
  }
}

void AdjacencyMapGraph::WriteAssignments(FILE* out_file) {
  unordered_multimap<uint32_t, uint32_t>* task_mappings = GetTaskAssignments();
  for (auto& task_pu : *task_mappings) {
    fprintf(out_file, "m %u %u\n", task_pu.first, task_pu.second);
  }
  delete task_mappings;
}

void AdjacencyMapGraph::WriteFlowGraph(FILE* graph_file) {
  int64_t min_cost = 0;
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    for (auto& id_arc : arcs_[node_id]) {
      if (id_arc.second->is_fwd) {
        uint32_t flow = id_arc.second->reverse_arc->residual_cap +
          id_arc.second->min_flow;
        if (flow > 0) {
          fprintf(graph_file, "f %u %u %d\n", node_id, id_arc.first, flow);
          min_cost += flow * id_arc.second->cost;
        }
      }
    }
  }
  fprintf(graph_file, "s %jd\n", min_cost);
}

void AdjacencyMapGraph::WriteGraph(FILE* graph_file) {
  fprintf(graph_file, "p min %u %u\n", max_node_id_, num_arcs_);
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    fprintf(graph_file, "n %u %d\n", node_id, nodes_[node_id].supply);
  }
  for (uint32_t node_id = 1; node_id <= max_node_id_; ++node_id) {
    for (auto& id_arc : arcs_[node_id]) {
      Arc* arc = id_arc.second;
      if (arc->is_fwd) {
        uint32_t flow = arc->reverse_arc->residual_cap + arc->min_flow;
        fprintf(graph_file, "a %u %u %u %d %jd\n", node_id, id_arc.first,
                flow, arc->residual_cap + flow, arc->cost);
      }
    }
  }
}

} // namespace flowlessly
