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

#include "graphs/graph_generator.h"

#include <set>
#include <vector>

#include "graphs/adjacency_map_graph.h"
#include "misc/statistics.h"

DEFINE_int64(max_arc_cost, 1000, "The maximum cost of an arc");
DEFINE_uint64(num_cores_per_machine, 8, "Number of cores per machine");
DEFINE_uint64(max_preference_arcs, 15,
             "The maximum number of preference arcs per task");
DEFINE_uint64(num_machines_per_rack, 20,
              "The number of machines per rack node");
DEFINE_int64(percentage_machines_removed, 1,
             "Number of machines removed in an iteration");
DEFINE_int64(percentage_machines_added, 1,
             "Number of machines added in an iteration");
DEFINE_int64(percentage_tasks_removed, 3,
             "Number of tasks removed in an iteration");
DEFINE_int64(percentage_tasks_added, 3,
             "Number of tasks added in an iteration");

DECLARE_uint64(num_cores_per_machine);

namespace flowlessly {

GraphGenerator::GraphGenerator() : seed_(time(NULL)) {
}

GraphGenerator::GraphGenerator(uint32_t seed) : seed_(seed) {
}

uint32_t GraphGenerator::Generate(uint32_t num_tasks, uint32_t num_machines,
                                  vector<uint32_t>* machine_node_ids,
                                  Graph* graph, FILE* out_graph_file) {
  // Add the sink node.
  uint32_t sink_node_id = graph->AddNode(-num_tasks, 0, SINK, true);
  // Add the cluster aggregator node.
  uint32_t cluster_agg_node_id = graph->AddNode(0, 0, OTHER, true);
  uint32_t cur_rack_node_id = 0;
  // Add the machines.
  uint32_t machines_per_current_rack = FLAGS_num_machines_per_rack;
  for (uint32_t machinde_index = 0; machinde_index < num_machines;
       ++machinde_index) {
    uint32_t machine_node_id =
      GenerateMachine(sink_node_id, cluster_agg_node_id, true, graph,
                      &cur_rack_node_id, &machines_per_current_rack,
                      out_graph_file);
    machine_node_ids->push_back(machine_node_id);
  }
  // Add the tasks.
  for (; num_tasks > 0; --num_tasks) {
    GenerateTask(cluster_agg_node_id, num_machines, *machine_node_ids, true,
                 graph, out_graph_file);
  }
  graph->WriteGraph(out_graph_file);
  return cluster_agg_node_id;
}

void GraphGenerator::GenerateArcChanges(uint32_t num_arc_changes,
                                        vector<uint32_t>* machine_node_ids,
                                        Graph* graph,
                                        FILE* out_changes_file) {
  for (; num_arc_changes > 0; --num_arc_changes) {
    uint32_t machine_index = rand_r(&seed_) % machine_node_ids->size();
    uint32_t machine_node_id = (*machine_node_ids)[machine_index];
    Arc* arc = graph->GetRandomArc(machine_node_id);
    if (arc == NULL) {
      num_arc_changes++;
      continue;
    }
    uint32_t old_capacity =
      arc->residual_cap + arc->reverse_arc->residual_cap + arc->min_flow;
    int64_t new_arc_cost = rand_r(&seed_) % FLAGS_max_arc_cost;
    graph->ChangeArc(arc, 0, old_capacity, new_arc_cost, 0);
    // TODO(ionel): Generate arc capacity changes.
  }
}

void GraphGenerator::GenerateChanges(uint32_t cluster_agg_node_id,
                                     uint32_t* num_machines,
                                     vector<uint32_t>* machine_node_ids,
                                     uint32_t* num_tasks,
                                     Graph* graph,
                                     FILE* out_changes_file) {
  // NOTE: The calculations for number of machines and tasks to be added
  // and removed are not exactly correct when num_machines or num_tasks
  // are smaller than 100.

  // Removing machines.
  uint32_t num_removed_machines =
    FLAGS_percentage_machines_removed * (*num_machines) / 100;
  (*num_machines) -= num_removed_machines;
  for (; num_removed_machines > 0; num_removed_machines--) {
    uint32_t machine_index = rand_r(&seed_) % machine_node_ids->size();
    uint32_t machine_node_id = (*machine_node_ids)[machine_index];
    set<uint32_t> pu_ids;
    graph->GetMachinePUs(machine_node_id, &pu_ids);
    for (auto& pu_id : pu_ids) {
      graph->RemoveNode(pu_id);
      fprintf(out_changes_file, "r %u\n", pu_id);
    }
    graph->RemoveNode(machine_node_id);
    fprintf(out_changes_file, "r %u\n", machine_node_id);
    machine_node_ids->erase(machine_node_ids->begin() + machine_index);
  }

  // Removing tasks.
  uint32_t num_tasks_removed =
    FLAGS_percentage_tasks_removed * (*num_tasks) / 100;
  (*num_tasks) -= num_tasks_removed;
  for (; num_tasks_removed > 0; num_tasks_removed--) {
    // NOTE: Assumes that all source nodes are task nodes.
    set<uint32_t>& source_nodes = graph->get_source_nodes();
    CHECK_GE(source_nodes.size(), 1);
    uint32_t task_node_id = *(source_nodes.begin());
    graph->RemoveNode(task_node_id);
    fprintf(out_changes_file, "r %u\n", task_node_id);
  }

  // Adding new tasks.
  uint32_t num_new_tasks = FLAGS_percentage_tasks_added * (*num_tasks) / 100;
  (*num_tasks) += num_new_tasks;
  for (; num_new_tasks > 0; num_new_tasks--) {
    GenerateTask(cluster_agg_node_id, *num_machines, *machine_node_ids,
                 false, graph, out_changes_file);
  }

  // Adding new machines.
  uint32_t sink_node_id = graph->get_sink_node();
  uint32_t num_new_machines =
    FLAGS_percentage_machines_added * (*num_machines) / 100;
  uint32_t cur_rack_node_id = 0;
  uint32_t machines_per_current_rack = FLAGS_num_machines_per_rack;
  for (; num_new_machines > 0; num_new_machines--) {
    uint32_t machine_node_id =
      GenerateMachine(sink_node_id, cluster_agg_node_id, false, graph,
                      &cur_rack_node_id, &machines_per_current_rack,
                      out_changes_file);
    machine_node_ids->push_back(machine_node_id);
  }

  // Arc changes.
  uint32_t num_arc_changes = rand_r(&seed_) % *num_tasks;
  GenerateArcChanges(num_arc_changes, machine_node_ids, graph,
                     out_changes_file);
}

uint32_t GraphGenerator::GenerateMachine(uint32_t sink_node_id,
                                         uint32_t cluster_agg_node_id,
                                         bool first_scheduling_iteration,
                                         Graph* graph,
                                         uint32_t* cur_rack_node_id,
                                         uint32_t* machines_per_current_rack,
                                         FILE* out_changes_file) {
  int64_t arc_cost = 0;
  uint32_t machine_node_id =
    graph->AddNode(0, 0, OTHER, first_scheduling_iteration);
  if (!first_scheduling_iteration) {
    fprintf(out_changes_file, "n %u 0 %u\n", machine_node_id, OTHER);
  }
  // Add core nodes and their arcs.
  for (uint32_t core_index = 0;
       core_index < FLAGS_num_cores_per_machine;
       ++core_index) {
    uint32_t core_node_id =
      graph->AddNode(0, 0, PU, first_scheduling_iteration);
    arc_cost = rand_r(&seed_) % FLAGS_max_arc_cost;
    graph->AddArc(core_node_id, sink_node_id, 0, 1, arc_cost, 0);
    int64_t core_arc_cost = rand_r(&seed_) % FLAGS_max_arc_cost;
    graph->AddArc(machine_node_id, core_node_id, 0, 1, core_arc_cost, 0);
    if (!first_scheduling_iteration) {
      fprintf(out_changes_file, "n %u 0 %u\n", core_node_id, PU);
      fprintf(out_changes_file, "a %u %u %u %u %" PRId64 "%u\n", core_node_id,
              sink_node_id, 0, 1, arc_cost, 0);
      fprintf(out_changes_file, "a %u %u %u %u %" PRId64 "%u\n",
              machine_node_id, core_node_id, 0, 1, core_arc_cost, 0);
    }
  }
  if (*machines_per_current_rack == FLAGS_num_machines_per_rack) {
    // The current rack is full. Add a new one.
    *cur_rack_node_id = graph->AddNode(0, 0, OTHER, first_scheduling_iteration);
    // Add arc from the cluster aggregator to the rack node.
    arc_cost = rand_r(&seed_) % FLAGS_max_arc_cost;
    graph->AddArc(cluster_agg_node_id, *cur_rack_node_id, 0,
                 FLAGS_num_machines_per_rack * FLAGS_num_cores_per_machine,
                 arc_cost, 0);
    if (!first_scheduling_iteration) {
      fprintf(out_changes_file, "n %u 0 %u\n", *cur_rack_node_id, OTHER);
      fprintf(out_changes_file, "a %u %u %u %" PRIu64 " %" PRId64 "%u\n",
              cluster_agg_node_id, *cur_rack_node_id, 0,
              FLAGS_num_machines_per_rack * FLAGS_num_cores_per_machine,
              arc_cost, 0);
    }
    *machines_per_current_rack = 0;
  }
  // Add arc between the rack and the machine node.
  arc_cost = rand_r(&seed_) % FLAGS_max_arc_cost;
  graph->AddArc(*cur_rack_node_id, machine_node_id, 0,
                FLAGS_num_cores_per_machine, arc_cost, 0);
  if (!first_scheduling_iteration) {
    fprintf(out_changes_file, "a %u %u %u %" PRIu64 " %" PRId64 "%u\n",
            *cur_rack_node_id, machine_node_id, 0, FLAGS_num_cores_per_machine,
            arc_cost, 0);
  }
  (*machines_per_current_rack)++;
  return machine_node_id;
}

void GraphGenerator::GenerateTask(uint32_t cluster_agg_node_id,
                                  uint32_t num_machines,
                                  const vector<uint32_t>& machine_node_ids,
                                  bool first_scheduling_iteration,
                                  Graph* graph,
                                  FILE* out_changes_file) {
  // Add the task node.
  uint32_t task_node_id =
    graph->AddNode(1, 0, TASK, first_scheduling_iteration);
  // Arc to cluster aggregator.
  int64_t arc_cost = rand_r(&seed_) % FLAGS_max_arc_cost;
  graph->AddArc(task_node_id, cluster_agg_node_id, 0, 1, arc_cost, 0);
  // Preference arcs.
  uint32_t num_preference_arcs = rand_r(&seed_) % FLAGS_max_preference_arcs;
  if (num_preference_arcs > num_machines) {
    num_preference_arcs = num_machines;
  }
  if (!first_scheduling_iteration) {
    // Write the incremental graph changes.
    fprintf(out_changes_file, "n %u 1 %u\n", task_node_id, TASK);
    fprintf(out_changes_file, "a %u %u %u %u %" PRId64 "%u\n", task_node_id,
            cluster_agg_node_id, 0, 1, arc_cost, 0);
  }
  set<uint32_t> prefered_machines;
  // Make sure that the arcs are pointing to different machines.
  while (prefered_machines.size() < num_preference_arcs) {
    uint32_t machine_index = rand_r(&seed_) % machine_node_ids.size();
    prefered_machines.insert(machine_index);
  }
  for (uint32_t machine_index : prefered_machines) {
    arc_cost = rand_r(&seed_) % FLAGS_max_arc_cost;
    graph->AddArc(task_node_id, machine_node_ids[machine_index], 0, 1,
                  arc_cost, 0);
    if (!first_scheduling_iteration) {
      // Write the incremental graph changes.
      fprintf(out_changes_file, "a %u %u %u %u %" PRId64 "%u\n", task_node_id,
              machine_node_ids[machine_index], 0, 1, arc_cost, 0);
    }
  }
}

} // namespace flowlessly
