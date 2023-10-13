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
#include "misc/utils.h"

DEFINE_bool(graph_has_node_types, false, "Graph input contains node types");

namespace flowlessly {

class UtilsTests : public ::testing::Test {
 protected:
  UtilsTests() {
    graph_ = new AdjacencyMapGraph(&stats);
  }

  ~UtilsTests() {
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
    graph_->AddArc(3, 2, 0, 2, 3, 0);
    graph_->AddArc(2, 1, 0, 2, 0, 0);
  }

  //  virtual void TearDown() {}

  Statistics stats;
  AdjacencyMapGraph* graph_;
};

TEST_F(UtilsTests, BellmanFordWihtoutCycle) {
  auto& nodes = graph_->get_nodes();
  vector<uint32_t> predecessor(graph_->get_max_node_id() + 1, 0);
  for (uint32_t node_id = 1; node_id <= graph_->get_max_node_id(); ++node_id) {
    nodes[node_id].distance = numeric_limits<int64_t>::max();
  }
  BellmanFord(graph_, graph_->get_active_node_ids(), &predecessor);
  EXPECT_EQ(nodes[1].distance, 4);
  EXPECT_EQ(nodes[2].distance, 4);
  EXPECT_EQ(nodes[3].distance, 1);
  EXPECT_EQ(nodes[4].distance, 0);
  EXPECT_EQ(nodes[5].distance, 0);
}

TEST_F(UtilsTests, BellmanFordWithCycle) {
  auto& nodes = graph_->get_nodes();
  graph_->AddArc(2, 4, 0, 1, 1, 0);
  vector<uint32_t> predecessor(graph_->get_max_node_id() + 1, 0);
  for (uint32_t node_id = 1; node_id <= graph_->get_max_node_id(); ++node_id) {
    nodes[node_id].distance = numeric_limits<int64_t>::max();
  }
  BellmanFord(graph_, graph_->get_active_node_ids(), &predecessor);
  EXPECT_EQ(nodes[1].distance, 4);
  EXPECT_EQ(nodes[2].distance, 4);
  EXPECT_EQ(nodes[3].distance, 1);
  EXPECT_EQ(nodes[4].distance, 0);
  EXPECT_EQ(nodes[5].distance, 0);
  EXPECT_EQ(predecessor[2], 3);
}

TEST_F(UtilsTests, DijkstraOptimized) {
  auto& nodes = graph_->get_nodes();
  vector<uint32_t> predecessor(graph_->get_max_node_id() + 1, 0);
  for (uint32_t node_id = 1; node_id <= graph_->get_max_node_id(); ++node_id) {
    nodes[node_id].distance = numeric_limits<int64_t>::max();
  }
  DijkstraOptimized(graph_, graph_->get_active_node_ids(), &predecessor);
  EXPECT_EQ(nodes[1].distance, 4);
  EXPECT_EQ(nodes[2].distance, 4);
  EXPECT_EQ(nodes[3].distance, 1);
  EXPECT_EQ(nodes[4].distance, 0);
  EXPECT_EQ(nodes[5].distance, 0);
  EXPECT_EQ(predecessor[2], 3);
}

TEST_F(UtilsTests, DijkstraSimple) {
  auto& nodes = graph_->get_nodes();
  vector<uint32_t> predecessor(graph_->get_max_node_id() + 1, 0);
  for (uint32_t node_id = 1; node_id <= graph_->get_max_node_id(); ++node_id) {
    nodes[node_id].distance = numeric_limits<int64_t>::max();
  }
  DijkstraSimple(graph_, graph_->get_active_node_ids(), &predecessor);
  EXPECT_EQ(nodes[1].distance, 4);
  EXPECT_EQ(nodes[2].distance, 4);
  EXPECT_EQ(nodes[3].distance, 1);
  EXPECT_EQ(nodes[4].distance, 0);
  EXPECT_EQ(nodes[5].distance, 0);
  EXPECT_EQ(predecessor[2], 3);
}

TEST_F(UtilsTests, MaxFlow) {
  // Add arc to create cycle.
  graph_->AddArc(2, 4, 0, 1, 1, 0);
  // Run max flow on the graph.
  MaxFlow(graph_);
  auto& nodes = graph_->get_nodes();
  auto& arcs = graph_->get_arcs();
  for (uint32_t node_id = 2; node_id <= 5; node_id++) {
    EXPECT_EQ(nodes[node_id].supply, 0);
  }
  Arc* arc = arcs[4][3];
  EXPECT_EQ(arc->residual_cap, 0);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 1);
  arc = arcs[5][3];
  EXPECT_EQ(arc->residual_cap, 0);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 1);
  arc = arcs[3][2];
  EXPECT_EQ(arc->residual_cap, 0);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 2);
  arc = arcs[2][1];
  EXPECT_EQ(arc->residual_cap, 0);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 2);
  arc = arcs[2][4];
  EXPECT_EQ(arc->residual_cap, 1);
  EXPECT_EQ(arc->reverse_arc->residual_cap, 0);
}

} // namespace flowlessly
