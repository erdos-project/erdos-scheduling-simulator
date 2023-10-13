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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "graphs/adjacency_map_graph.h"

DEFINE_bool(graph_has_node_types, false, "Graph input contains node types");

namespace flowlessly {

class AdjacencyMapGraphTest : public ::testing::Test {
 protected:
  AdjacencyMapGraphTest() {
    graph_ = new AdjacencyMapGraph(&stats);
  }

  ~AdjacencyMapGraphTest() {
    delete graph_;
  }

  virtual void SetUp() {
    FLAGS_logtostderr = true;
    FLAGS_stderrthreshold = 0;
    FLAGS_v = 3;
    /*
      Setup the following graph.
      4 \
         3 - 2 - 1
      5 /
    */
    uint32_t node_id = graph_->AddNode(-1, 0, static_cast<NodeType>(0), false);
    CHECK_EQ(node_id, 1);
    node_id = graph_->AddNode(0, 0, static_cast<NodeType>(0), false);
    CHECK_EQ(node_id, 2);
    node_id = graph_->AddNode(0, 0, static_cast<NodeType>(0), false);
    CHECK_EQ(node_id, 3);
    node_id = graph_->AddNode(1, 0, static_cast<NodeType>(0), false);
    CHECK_EQ(node_id, 4);
    node_id = graph_->AddNode(1, 0, static_cast<NodeType>(0), false);
    CHECK_EQ(node_id, 5);
    graph_->AddArc(4, 3, 0, 1, 1, 0);
    graph_->AddArc(5, 3, 0, 1, 2, 0);
    graph_->AddArc(3, 2, 0, 2, 1, 0);
    graph_->AddArc(2, 1, 0, 2, 0, 0);
  }

  // virtual void TearDown() {}

  Statistics stats;
  AdjacencyMapGraph* graph_;
};

TEST_F(AdjacencyMapGraphTest, AddArc) {
  auto& arcs = graph_->get_arcs();
  auto& nodes = graph_->get_nodes();
  EXPECT_EQ(nodes[5].supply, 1);
  EXPECT_EQ(nodes[2].supply, 0);
  EXPECT_EQ(arcs[5].size(), 1);
  EXPECT_EQ(arcs[2].size(), 2);
  graph_->AddArc(5, 2, 0, 1, 1, 0);
  EXPECT_EQ(nodes[5].supply, 1);
  EXPECT_EQ(nodes[2].supply, 0);
  // Check the arc was actually added.
  EXPECT_EQ(arcs[5].size(), 2);
  EXPECT_EQ(arcs[2].size(), 3);
  Arc* arc = arcs[5][2];
  // Check the arc was setup correctly.
  EXPECT_EQ(arc->src_node_id, 5);
  EXPECT_EQ(arc->dst_node_id, 2);
  EXPECT_TRUE(arc->is_fwd);
  EXPECT_FALSE(arc->is_running);
  EXPECT_EQ(arc->residual_cap, 1);
  EXPECT_EQ(arc->min_flow, 0);
  EXPECT_EQ(arc->cost, 1);

  // Trying to add the same arc again.
  EXPECT_DEATH(graph_->AddArc(5, 2, 0, 2, 42, 0), "");
}

TEST_F(AdjacencyMapGraphTest, AddArcMinFlow) {
  // Testing adding an arc with lower capacity bound.
  auto& arcs = graph_->get_arcs();
  auto& nodes = graph_->get_nodes();
  EXPECT_EQ(nodes[5].supply, 1);
  EXPECT_EQ(nodes[2].supply, 0);
  EXPECT_EQ(arcs[5].size(), 1);
  EXPECT_EQ(arcs[2].size(), 2);
  graph_->AddArc(5, 2, 1, 2, 1, 0);
  // Check the supply has been adjusted.
  EXPECT_EQ(nodes[5].supply, 0);
  EXPECT_EQ(nodes[2].supply, 1);
  // Check the arc was actually added.
  EXPECT_EQ(arcs[5].size(), 2);
  EXPECT_EQ(arcs[2].size(), 3);
  Arc* arc = arcs[5][2];
  // Check the arc was setup correctly.
  EXPECT_EQ(arc->src_node_id, 5);
  EXPECT_EQ(arc->dst_node_id, 2);
  EXPECT_TRUE(arc->is_fwd);
  EXPECT_FALSE(arc->is_running);
  EXPECT_EQ(arc->residual_cap, 1);
  EXPECT_EQ(arc->min_flow, 1);
  EXPECT_EQ(arc->cost, 1);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 0);
}

TEST(AdjacencyMapGraph, AddNode) {
  Statistics stats;
  AdjacencyMapGraph graph(&stats);
  uint32_t sink_node_id =
    graph.AddNode(-1, 0, static_cast<NodeType>(0), false);
  // Graph has 1 node.
  EXPECT_EQ(graph.get_max_node_id(), 1);
  // The new node is the sink.
  EXPECT_EQ(graph.get_sink_node(), sink_node_id);
  uint32_t node_id = graph.AddNode(0, 0, static_cast<NodeType>(0), false);
  // Graph has 2 nodes.
  EXPECT_EQ(graph.get_max_node_id(), 2);
  // The new node is not a sink.
  EXPECT_NE(graph.get_sink_node(), node_id);
  // Adding a source node.
  node_id = graph.AddNode(1, 0, static_cast<NodeType>(0), false);
  // Graph has 3 nodes.
  EXPECT_EQ(graph.get_max_node_id(), 3);
  // The new node is not a sink.
  EXPECT_NE(graph.get_sink_node(), node_id);
  // The graph has one source.
  EXPECT_EQ(graph.get_source_nodes().size(), 1);
  EXPECT_NE(graph.get_source_nodes().find(node_id),
            graph.get_source_nodes().end());
  auto& nodes = graph.get_nodes();
  // The sink has a supply of -2 because of its initial supply
  // and the additional demand created when we added the source node.
  EXPECT_EQ(nodes[1].supply, -2);
  EXPECT_EQ(nodes[2].supply, 0);
  EXPECT_EQ(nodes[3].supply, 1);
}

TEST(AdjacencyMapGraph, AddSourceBeforeSink) {
  Statistics stats;
  AdjacencyMapGraph graph(&stats);
  // Expect failure because sink is not set.
  EXPECT_DEATH(graph.AddNode(1, 0, static_cast<NodeType>(0), false), "");
}

TEST(AdjacencyMapGraph, AddInvalidNode) {
  Statistics stats;
  AdjacencyMapGraph graph(&stats);
  uint32_t node_id = graph.AddNode(-1, 0, static_cast<NodeType>(0), false);
  auto& nodes = graph.get_nodes();
  // Cannot update sink's supply to a positive value.
  EXPECT_DEATH(graph.AddNode(node_id, 1, 0, static_cast<NodeType>(0), false),
               "");
  node_id = graph.AddNode(1, 0, static_cast<NodeType>(0), false);
  EXPECT_EQ(nodes[node_id].supply, 1);
  graph.AddNode(node_id, 2, 0, static_cast<NodeType>(0), false);
  EXPECT_EQ(nodes[node_id].supply, 2);
}

TEST_F(AdjacencyMapGraphTest, ChangeArc) {
  auto& arcs = graph_->get_arcs();
  auto& nodes = graph_->get_nodes();
  EXPECT_EQ(nodes[3].supply, 0);
  EXPECT_EQ(nodes[2].supply, 0);
  EXPECT_EQ(arcs[3].size(), 3);
  EXPECT_EQ(arcs[2].size(), 2);
  EXPECT_EQ(graph_->get_active_node_ids().size(), 2);

  // We're setting a minimum flow on the arc.
  graph_->ChangeArc(3, 2, 2, 2, 42, 0, false, 0);
  EXPECT_EQ(arcs[3].size(), 3);
  EXPECT_EQ(arcs[2].size(), 2);
  EXPECT_EQ(nodes[3].supply, -2);
  EXPECT_EQ(nodes[2].supply, 2);
  EXPECT_EQ(graph_->get_active_node_ids().size(), 3);
  Arc* arc = arcs[3][2];
  EXPECT_EQ(arc->src_node_id, 3);
  EXPECT_EQ(arc->dst_node_id, 2);
  EXPECT_TRUE(arc->is_fwd);
  EXPECT_FALSE(arc->is_running);
  EXPECT_EQ(arc->residual_cap, 0);
  EXPECT_EQ(arc->min_flow, 2);
  EXPECT_EQ(arc->cost, 42);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 0);

  // We're increasing capacity.
  graph_->ChangeArc(arc, 2, 3, 42, 0);
  EXPECT_EQ(arcs[3].size(), 3);
  EXPECT_EQ(arcs[2].size(), 2);
  EXPECT_EQ(nodes[3].supply, -2);
  EXPECT_EQ(nodes[2].supply, 2);
  EXPECT_EQ(arc->residual_cap, 1);
  EXPECT_EQ(arc->min_flow, 2);
  EXPECT_EQ(arc->cost, 42);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 0);

  // We're decreasing capacity.
  graph_->ChangeArc(arc, 2, 2, 42, 0);
  EXPECT_EQ(arcs[3].size(), 3);
  EXPECT_EQ(arcs[2].size(), 2);
  EXPECT_EQ(nodes[3].supply, -2);
  EXPECT_EQ(nodes[2].supply, 2);
  EXPECT_EQ(arc->residual_cap, 0);
  EXPECT_EQ(arc->min_flow, 2);
  EXPECT_EQ(arc->cost, 42);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 0);

  // We're decreasing capacity bellow min_flow.
  EXPECT_DEATH(graph_->ChangeArc(arc, 2, 1, 42, 0), "");
}

TEST_F(AdjacencyMapGraphTest, ChangeNonexistentArc) {
  auto& arcs = graph_->get_arcs();
  auto& nodes = graph_->get_nodes();
  EXPECT_EQ(nodes[5].supply, 1);
  EXPECT_EQ(nodes[2].supply, 0);
  EXPECT_EQ(arcs[5].size(), 1);
  EXPECT_EQ(arcs[2].size(), 2);
  // The arc does not exist, but it will get added.
  graph_->ChangeArc(5, 2, 0, 2, 42, 0, false, 0);
  EXPECT_EQ(nodes[5].supply, 1);
  EXPECT_EQ(nodes[2].supply, 0);
  EXPECT_EQ(arcs[5].size(), 2);
  EXPECT_EQ(arcs[2].size(), 3);
}

TEST_F(AdjacencyMapGraphTest, IsEpsOptimal) {
  auto& nodes = graph_->get_nodes();
  // Not eps optimal because mass balance contraints are not met.
  EXPECT_FALSE(graph_->IsEpsOptimal(4));
  // Reset supplies.
  nodes[5].supply = 0;
  nodes[4].supply = 0;
  nodes[1].supply = 0;
  EXPECT_TRUE(graph_->IsEpsOptimal(4));
  // Break optimality.
  nodes[3].potential = 5;
  EXPECT_FALSE(graph_->IsEpsOptimal(1));
}

TEST_F(AdjacencyMapGraphTest, IsFeasible) {
  auto& nodes = graph_->get_nodes();
  // Not feasible because mass balance contraints are not met.
  EXPECT_FALSE(graph_->IsFeasible());
  // Reset supplies.
  nodes[5].supply = 0;
  nodes[4].supply = 0;
  nodes[1].supply = 0;
  EXPECT_TRUE(graph_->IsFeasible());
}

TEST_F(AdjacencyMapGraphTest, RemoveArc) {
  auto& nodes = graph_->get_nodes();
  auto& arcs = graph_->get_arcs();
  Arc* arc = arcs[4][3];
  EXPECT_EQ(arcs[4].size(), 1);
  EXPECT_EQ(arcs[3].size(), 3);
  EXPECT_EQ(nodes[4].supply, 1);
  EXPECT_EQ(nodes[3].supply, 0);
  // Remove an arc without min_flow.
  graph_->RemoveArc(arc);
  EXPECT_EQ(arcs[4].size(), 0);
  EXPECT_EQ(arcs[3].size(), 2);
  EXPECT_EQ(nodes[4].supply, 1);
  EXPECT_EQ(nodes[3].supply, 0);
  EXPECT_EQ(graph_->get_active_node_ids().size(), 2);

  // Add an arc with min_flow.
  graph_->AddArc(4, 3, 1, 1, 42, 0);
  EXPECT_EQ(arcs[4].size(), 1);
  EXPECT_EQ(arcs[3].size(), 3);
  EXPECT_EQ(nodes[4].supply, 0);
  EXPECT_EQ(nodes[3].supply, 1);
  EXPECT_EQ(graph_->get_active_node_ids().size(), 2);
  // Remove an arc with min_flow.
  graph_->RemoveArc(arcs[4][3]);
  EXPECT_EQ(arcs[4].size(), 0);
  EXPECT_EQ(arcs[3].size(), 2);
  EXPECT_EQ(nodes[4].supply, 1);
  EXPECT_EQ(nodes[3].supply, 0);
}

TEST_F(AdjacencyMapGraphTest, RemoveNode) {
  auto& nodes = graph_->get_nodes();
  auto& arcs = graph_->get_arcs();
  EXPECT_EQ(arcs[3].size(), 3);
  EXPECT_EQ(nodes[1].supply, -3);
  EXPECT_EQ(graph_->get_active_node_ids().size(), 2);
  graph_->RemoveNode(4);
  EXPECT_EQ(arcs[3].size(), 2);
  EXPECT_EQ(nodes[4].supply, 0);
  EXPECT_EQ(nodes[4].potential, 0);
  EXPECT_EQ(nodes[1].supply, -2);
  EXPECT_EQ(graph_->get_active_node_ids().size(), 1);
  graph_->RemoveNode(3);
  EXPECT_EQ(arcs[5].size(), 0);
  EXPECT_EQ(arcs[2].size(), 1);
}

TEST_F(AdjacencyMapGraphTest, GetTaskAssignments) {
  /*
    Extending the graph to:
    4 \  / 2 \
    7 -3      1
    5 /- \ 6 /
  */
  uint32_t node_id = graph_->AddNode(0, 0, static_cast<NodeType>(0), false);
  EXPECT_EQ(node_id, 6);
  // Add arc from the new core to the sink.
  graph_->AddArc(6, 1, 0, 1, 0, 0);
  // Add arc from the cluster agg to the new core.
  graph_->AddArc(3, 6, 0, 1, 1, 0);
  // Add preference arc from task at node 4 to the new core.
  graph_->AddArc(5, 6, 0, 1, 1, 0);

  // Add new task node.
  node_id = graph_->AddNode(1, 0, static_cast<NodeType>(0), false);
  EXPECT_EQ(node_id, 7);
  // Add arc to cluster aggregator.
  graph_->AddArc(7, 3, 0, 1, 1, 0);
  // Make sure the types of the nodes are correct.
  auto& nodes = graph_->get_nodes();
  nodes[2].type = PU;
  nodes[4].type = TASK;
  nodes[5].type = TASK;
  nodes[6].type = PU;
  nodes[7].type = TASK;

  // Setting the flow in the graph.
  graph_->ChangeArc(5, 6, 1, 1, 1, 0, false, 0);
  graph_->ChangeArc(6, 1, 1, 1, 1, 0, false, 0);
  graph_->ChangeArc(4, 3, 1, 1, 1, 0, false, 0);
  graph_->ChangeArc(7, 3, 1, 1, 1, 0, false, 0);
  graph_->ChangeArc(3, 2, 2, 2, 1, 0, false, 0);
  graph_->ChangeArc(2, 1, 2, 2, 0, 0, false, 0);

  auto* task_mappings = graph_->GetTaskAssignments();
  // Check node 4 is mapped to 2, node 5 is mapped to 6 and
  // node 7 is mapped to 2.
  auto task4_to_pu = task_mappings->find(4);
  EXPECT_NE(task4_to_pu, task_mappings->end());
  EXPECT_EQ(task4_to_pu->second, 2);
  auto task5_to_pu = task_mappings->find(5);
  EXPECT_NE(task5_to_pu, task_mappings->end());
  EXPECT_EQ(task5_to_pu->second, 6);
  auto task7_to_pu = task_mappings->find(7);
  EXPECT_NE(task7_to_pu, task_mappings->end());
  EXPECT_EQ(task7_to_pu->second, 2);
}

TEST_F(AdjacencyMapGraphTest, TopologicalSort) {
  auto& nodes = graph_->get_nodes();
  vector<uint32_t> ordered_nodes;
  EXPECT_TRUE(graph_->TopologicalSort(&ordered_nodes));
  EXPECT_EQ(ordered_nodes[0], 5);
  EXPECT_EQ(ordered_nodes[1], 4);
  EXPECT_EQ(ordered_nodes[2], 3);
  EXPECT_EQ(ordered_nodes[3], 2);
  EXPECT_EQ(ordered_nodes[4], 1);

  // Create a cycle in the graph.
  nodes[4].potential = 10;
  graph_->AddArc(2, 4, 0, 1, -5, 0);
  nodes[2].potential = 1;
  nodes[3].potential = 3;
  nodes[4].potential = 5;
  vector<uint32_t> updated_nodes;
  for (uint32_t index = 1; index <= 5; ++index) {
    updated_nodes.push_back(index);
  }
  graph_->UpdateAdmissibleGraph(updated_nodes);
  ordered_nodes.clear();
  // There's no topological order in a cyclic graph.
  EXPECT_FALSE(graph_->TopologicalSort(&ordered_nodes));
}

TEST_F(AdjacencyMapGraphTest, MaxRefinePotential) {
  auto& nodes = graph_->get_nodes();
  nodes[3].potential = 0;
  nodes[2].potential = 0;
  nodes[1].potential = 0;
  graph_->AddArc(3, 1, 0, 1, 2, 0);
  EXPECT_EQ(graph_->MaxRefinePotential(3, 2), 3);
  graph_->ChangeArc(3, 2, 0, 2, 3, 0, false, 0);
  EXPECT_EQ(graph_->MaxRefinePotential(3, 2), 4);
}

} // namespace flowlessly
