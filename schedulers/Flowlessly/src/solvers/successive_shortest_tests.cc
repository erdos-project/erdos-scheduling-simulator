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

#include "solvers/successive_shortest.h"

DEFINE_string(graph_dir, "test_graphs/",
              "Directory where test graphs are located.");
DEFINE_bool(graph_has_node_types, false, "Graph input contains node types");
DEFINE_string(algorithm, "successive_shortest_path", "");

namespace flowlessly {

class SuccessiveShortestTests : public ::testing::Test {
 protected:
  SuccessiveShortestTests() {
    graph_ = new AdjacencyMapGraph(&stats_);
    solver_ = new SuccessiveShortest(graph_, &stats_);
  }

  ~SuccessiveShortestTests() {
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
  SuccessiveShortest* solver_;
};

TEST_F(SuccessiveShortestTests, SmallGraph) {
  LoadGraph(FLAGS_graph_dir + "small_graph.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 38);
}

TEST_F(SuccessiveShortestTests, Manual8Jobs8PUs) {
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_8j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 386);
}

TEST_F(SuccessiveShortestTests, Manual10Jobs8PUs) {
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
}

TEST_F(SuccessiveShortestTests, Graph100Machines4Jobs100Tasks) {
  LoadGraph(FLAGS_graph_dir + "graph_100m_4j_100t_10p.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 268820);
}

TEST_F(SuccessiveShortestTests, Graph100Machines8Jobs100Tasks) {
  LoadGraph(FLAGS_graph_dir + "graph_100m_8j_100t_10p.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547488);
}

TEST_F(SuccessiveShortestTests, Graph100Machines16Jobs100Tasks) {
  LoadGraph(FLAGS_graph_dir + "graph_100m_16j_100t_10p.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 1115043);
}

TEST_F(SuccessiveShortestTests, IncrementalRemCoreWithTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_core_with_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 639);
}

TEST_F(SuccessiveShortestTests, IncrementalAddArcEquiCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_arc_equiv_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
}

TEST_F(SuccessiveShortestTests, IncrementalAddArcPrefCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_arc_pref_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 521);
}

TEST_F(SuccessiveShortestTests, IncrementalAddCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 501);
}

TEST_F(SuccessiveShortestTests, IncrementalAddTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 703);
}

TEST_F(SuccessiveShortestTests, IncrementalAddTaskNoPreempt) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task_nopremt.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 577);
}

TEST_F(SuccessiveShortestTests, IncrementalChgPrefCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_chg_pref_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 552);
}

TEST_F(SuccessiveShortestTests, IncrementalChgScheduledTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_chg_task_sched.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
}

TEST_F(SuccessiveShortestTests, IncrementalRemArcPrefCore) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_arc_pref_core.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 564);
}

TEST_F(SuccessiveShortestTests, IncrementalRemMachine) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_machine.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 760);
}

TEST_F(SuccessiveShortestTests, IncrementalRemScheduledTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_sched_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 451);
}

TEST_F(SuccessiveShortestTests, IncrementalAddCoreAndTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_core.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 647);
}

TEST_F(SuccessiveShortestTests, IncrementalRemMachineAddTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_machine.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 975);
}

TEST_F(SuccessiveShortestTests, IncrementalRemSchedTaskAddTask) {
  // Load the initial graph.
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j.in", true);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 547);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_rem_sched_task.in", false);
  LoadGraph(FLAGS_graph_dir + "graph_4m_2crs_10j_add_task.in", false);
  solver_->Run();
  EXPECT_EQ(graph_->GetTotalCost(), 604);
}

TEST_F(SuccessiveShortestTests, IncrementalRemPrefArcAddTaskAddMachine) {
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

TEST_F(SuccessiveShortestTests, IncrementalRemPrefArcAddTaskAddMachineRemTask) {
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
