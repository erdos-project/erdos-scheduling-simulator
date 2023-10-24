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

#include "solvers/cycle_cancelling.h"

// TODO(ionel): The tests that use graph_dir fail to run with make test because
// graph_dir is a relative path to FLOWLESSLY_ROOT.
DEFINE_string(graph_dir, "test_graphs/",
              "Directory where test graphs are located.");
DEFINE_bool(graph_has_node_types, false, "Graph input contains node types");

namespace flowlessly {

class CycleCancellingTests : public ::testing::Test {
 protected:
  CycleCancellingTests() {
    graph_ = new AdjacencyMapGraph(&stats_);
    solver_ = new CycleCancelling(graph_, &stats_);
  }

  ~CycleCancellingTests() {
    delete graph_;
    delete solver_;
  }

  virtual void SetUp() {
    // FLAGS_logtostderr = true;
    // FLAGS_stderrthreshold = 0;
    // FLAGS_v = 3;
  }

  //  virtual void TearDown() {}

  void LoadGraph(std::string graph_path, bool first_scheduling_iteration) {
    FILE* graph_file;
    if ((graph_file = fopen(graph_path.c_str(), "r")) == NULL) {
      LOG(ERROR) << "Failed to open graph file: " << graph_path;
      EXPECT_TRUE(false);
    }
    bool end_of_scheduling = false;
    graph_->ReadGraph(graph_file, first_scheduling_iteration,
                      &end_of_scheduling);
  }

  Statistics stats_;
  AdjacencyMapGraph* graph_;
  CycleCancelling* solver_;
};

TEST_F(CycleCancellingTests, SmallGraph) {
  LoadGraph(FLAGS_graph_dir + "small_graph.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 38);
}

TEST_F(CycleCancellingTests, Manual8Jobs8PUs) {
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_8j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 386);
}

TEST_F(CycleCancellingTests, Manual10Jobs8PUs) {
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
}

TEST_F(CycleCancellingTests, AugmentFlow) {
  /* Construct the following graph:
   1
   | \
   2  |
   |  |
   3  /
   | /
   4  */
  uint32_t node_id = graph_->AddNode(-1, 1, static_cast<NodeType>(0), false);
  EXPECT_EQ(node_id, 1);
  node_id = graph_->AddNode(0, 1, static_cast<NodeType>(0), false);
  EXPECT_EQ(node_id, 2);
  node_id = graph_->AddNode(1, 1, static_cast<NodeType>(0), false);
  EXPECT_EQ(node_id, 3);
  node_id = graph_->AddNode(1, 1, static_cast<NodeType>(0), false);
  EXPECT_EQ(node_id, 4);
  // The arc with the min residual capacity is arc (1, 2)
  Arc* arc12 = graph_->AddArc(1, 2, 0, 1, 1, 0);
  Arc* arc23 = graph_->AddArc(2, 3, 0, 2, 1, 0);
  Arc* arc34 = graph_->AddArc(3, 4, 0, 2, 1, 0);
  Arc* arc41 = graph_->AddArc(4, 1, 0, 2, 1, 0);

  vector<uint32_t> predecessor(5, 0);
  predecessor[4] = 3;
  predecessor[3] = 2;
  predecessor[2] = 1;
  predecessor[1] = 4;
  solver_->AugmentFlow(predecessor, 4, 1);
  EXPECT_EQ(arc23->residual_cap, 1);
  EXPECT_EQ(arc34->residual_cap, 1);
  EXPECT_EQ(arc41->residual_cap, 1);
  EXPECT_EQ(arc12->residual_cap, 0);

  // The arc with the min residual capacity is arc (4, 1)
  arc12->residual_cap += 2;
  arc23->residual_cap++;
  arc34->residual_cap++;
  solver_->AugmentFlow(predecessor, 4, 1);
  EXPECT_EQ(arc12->residual_cap, 1);
  EXPECT_EQ(arc23->residual_cap, 1);
  EXPECT_EQ(arc34->residual_cap, 1);
  EXPECT_EQ(arc41->residual_cap, 0);
}

// // NOTE: The other tests are disabled because they take a long time to run.
// // Run them after every CycleCancelling change!

// TEST_F(CycleCancellingTests, Graph100Machines4Jobs100Tasks) {
//   LoadGraph(FLAGS_graph_dir + "graph_100m_4j_100t_10p.in", true);
//   solver_->Run();
//   EXPECT_EQ(graph_->GetTotalCost(), 268820);
// }

// TEST_F(CycleCancellingTests, Graph100Machines8Jobs100Tasks) {
//   LoadGraph(FLAGS_graph_dir + "graph_100m_8j_100t_10p.in", true);
//   solver_->Run();
//   EXPECT_EQ(graph_->GetTotalCost(), 547488);
// }

// TEST_F(CycleCancellingTests, Graph100Machines16Jobs100Tasks) {
//   LoadGraph(FLAGS_graph_dir + "graph_100m_16j_100t_10p.in", true);
//   solver_->Run();
//   EXPECT_EQ(graph_->GetTotalCost(), 1115043);
// }

TEST_F(CycleCancellingTests, IncrementalRemCoreWithTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_core_with_task.in",
            false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 639);
}

TEST_F(CycleCancellingTests, IncrementalAddArcEquiCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_arc_equiv_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
}

TEST_F(CycleCancellingTests, IncrementalAddArcPrefCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_arc_pref_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 521);
}

TEST_F(CycleCancellingTests, IncrementalAddCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 501);
}

TEST_F(CycleCancellingTests, IncrementalAddTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 703);
}

TEST_F(CycleCancellingTests, IncrementalAddTaskNoPreempt) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task_nopremt.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 577);
}

TEST_F(CycleCancellingTests, IncrementalChgPrefCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_chg_pref_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 552);
}

TEST_F(CycleCancellingTests, IncrementalChgScheduledTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_chg_task_sched.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
}

TEST_F(CycleCancellingTests, IncrementalRemArcPrefCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_arc_pref_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 564);
}

TEST_F(CycleCancellingTests, IncrementalRemMachine) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_machine.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 760);
}

TEST_F(CycleCancellingTests, IncrementalRemScheduledTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_sched_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 451);
}

TEST_F(CycleCancellingTests, IncrementalAddCoreAndTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_core.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 647);
}

TEST_F(CycleCancellingTests, IncrementalRemMachineAddTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_machine.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 975);
}

TEST_F(CycleCancellingTests, IncrementalRemSchedTaskAddTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_sched_task.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 604);
}

TEST_F(CycleCancellingTests, IncrementalRemPrefArcAddTaskAddMachine) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_arc_pref_core.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_core.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 664);
}

TEST_F(CycleCancellingTests, IncrementalRemPrefArcAddTaskAddMachineRemTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_arc_pref_core.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_core.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_sched_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 595);
}

} // namespace flowlessly
