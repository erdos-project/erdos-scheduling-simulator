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

#include "graphs/graph.h"

#include <cstring>
#include <glog/logging.h>

namespace flowlessly {

void Graph::ReadAddNode(char* new_node_line, FILE* graph_file,
                        bool first_scheduling_iteration) {
  int32_t supply;
  int64_t potential;
  uint32_t num_node_arcs;
  uint32_t type = 0;
  sscanf(new_node_line, "%*c %d %" SCNd64 "%u %u",
         &supply, &potential, &num_node_arcs, &type);
  uint32_t node_id = AddNode(supply, potential, static_cast<NodeType>(type),
                             first_scheduling_iteration);
  char line[200];
  // Read the arcs of the new node.
  for (; num_node_arcs > 0; ) {
    if (fgets(line, 200, graph_file) != NULL) {
      if (line[0] == 'a') {
        uint32_t src_node, dst_node, min_flow;
        int32_t capacity;
        int64_t cost;
        uint32_t type = 0;
        sscanf(line, "%*c %u %u %u %d %" SCNd64 "%u", &src_node, &dst_node,
               &min_flow, &capacity, &cost, &type);
        // Assumption: in the input every arc has exactly a node with id 0.
        // This must be replaced with the id of the new node.
        if (src_node == 0) {
          src_node = node_id;
        } else {
          dst_node = node_id;
        }
        AddArc(src_node, dst_node, min_flow, capacity, cost, type);
        // Decrease the number of arcs left to be read.
        --num_node_arcs;
      } else if (line[0] == 'c') {
        // Ignore comment.
      } else {
        LOG(FATAL) << "Invalid operation after new node";
      }
    } else {
      LOG(FATAL) << "Invalid number of arcs for new node";
    }
  }
}

void Graph::ReadGraph(FILE* graph_file, bool first_scheduling_iteration,
                      bool* end_of_scheduling) {
  char line[200];
  bool end_of_iteration = false;
  uint32_t line_num = 0;
  while (!end_of_iteration && fgets(line, 200, graph_file) != NULL) {
    line_num++;
    switch (line[0]) {
    case 'c': {
        if (!strcmp(line, "c EOI\n")) {
          end_of_iteration = true;
        }
        if (!strcmp(line, "c EOS\n")) {
          // The scheduler has finished the work.
          *end_of_scheduling = true;
          return;
        }
        // Comment line. Ignore it.
        break;
      }
      case 'p': {
        char pr_type[4];
        uint32_t num_nodes;
        sscanf(line, "%*c %3s %u %u", pr_type, &num_nodes, &num_arcs_);
        // The maximum node id is equal to the number of nodes.
        // NOTE: We assume that all the ids between 1 and num_nodes are used.
        max_node_id_ = num_nodes;
        InitializeGraph();
        break;
      }
      case 'n': {
        // Adding a node.
        uint32_t node_id;
        int32_t supply;
        uint32_t type = 0;
        sscanf(line, "%*c %u %d %u", &node_id, &supply, &type);
        // TODO(ionel): HACK! Reusing old potential.
        CHECK_LE(node_id, potentials_.size() + 1);
        AddNode(node_id, supply, potentials_[node_id],
                static_cast<NodeType>(type), first_scheduling_iteration);
        break;
      }
      case 'a': {
        // Adding an arc.
        uint32_t src_node, dst_node, min_flow;
        int32_t capacity;
        int64_t cost;
        int32_t type = 0;
        sscanf(line, "%*c %u %u %u %d %" SCNd64 "%d", &src_node, &dst_node,
               &min_flow, &capacity, &cost, &type);
        AddArc(src_node, dst_node, min_flow, capacity, cost, type);
        break;
      }
      case 'r': {
        // Removing a node.
        uint32_t node_id;
        sscanf(line, "%*c %u", &node_id);
        RemoveNode(node_id);
        break;
      }
      case 'x': {
        uint32_t src_node, dst_node;
        int32_t capacity;
        uint32_t min_flow;
        int64_t cost;
        int32_t type = 0;
        int64_t old_cost = 0;
        int32_t num_vals_read =
          sscanf(line, "%*c %u %u %u %d %" SCNd64 "%d %" SCNd64,
                 &src_node, &dst_node, &min_flow, &capacity, &cost, &type,
                 &old_cost);
        if (num_vals_read == 7) {
          // If we have read old_cost as well then we need to pass it to
          // ChangeArc because we're changing one of the multiple arcs
          // between src_node and dst_node. The old_costs helps us to
          // identify the exact arc that must be changed.
          ChangeArc(src_node, dst_node, min_flow, capacity, cost, type, true,
                    old_cost);
        } else {
          ChangeArc(src_node, dst_node, min_flow, capacity, cost, type, false,
                    0);
        }
        break;
      }
      case 'd': {
        ReadAddNode(line, graph_file, first_scheduling_iteration);
        break;
      }
      default: {
        LOG(FATAL) << "The file doesn't respect the DIMACS format on line: "
                   << line_num;
      }
    }
  }
  // if (!first_run_) {
  //   // TODO(ionel): Decide which eps to use in the globalUpdate.
  //   GlobalUpdate(1);
  // }
}

void Graph::WriteAssignments(const string& out_file_name) {
  FILE* out_file = fopen(out_file_name.c_str(), "w");
  CHECK(out_file != NULL) << "Could not open assignments file for writing: "
                          << out_file_name;
  WriteAssignments(out_file);
  fclose(out_file);
}

void Graph::WriteFlowGraph(const string& out_file_name) {
  FILE* flow_file = fopen(out_file_name.c_str(), "w");
  CHECK(flow_file != NULL) << "Could not open flow file for writing: "
                          << out_file_name;
  WriteFlowGraph(flow_file);
  fclose(flow_file);
}

void Graph::WriteGraph(const string& out_file_name) {
  FILE *graph_file = fopen(out_file_name.c_str(), "w");
  CHECK(graph_file != NULL) << "Could no open graph file for writing: "
                            << out_file_name;
  WriteGraph(graph_file);
  fclose(graph_file);
}

} // namespace flowlessly
