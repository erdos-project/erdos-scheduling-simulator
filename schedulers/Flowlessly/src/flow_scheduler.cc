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

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/thread/latch.hpp>
#include <boost/timer/timer.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cstdio>
#include <limits>

#include "graphs/graph.h"
#include "graphs/adjacency_map_graph.h"
#include "misc/utils.h"
#include "solvers/cycle_cancelling.h"
#include "solvers/successive_shortest.h"

#define NANOSECONDS_IN_MICROSECOND 1000

using namespace flowlessly; // NOLINT

/*! \mainpage Flowlessly Documentation
 * Welcome to the documentation page of Flowlessly.
 */

DEFINE_string(algorithm, "successive_shortest_path",
              "Options: cycle_cancelling, successive_shortest_path");
DEFINE_string(graph_file, "", "Input graph file (stdin if empty)");
DEFINE_string(out_graph_file, "", "Output graph file (stdout if empty)");
DEFINE_bool(statistics, false, "Log statistics flag");
DEFINE_bool(debug_output, false, "Print debug graph after each iteration");
// Set to true if we use the extended DIMACS format in which we specify
// node's type. Having nodes types allows the solver to compute the
// mappings of tasks to PUs.
// NOTE: if we don't have node types then we try do automatically detect the
// sink. We work with the assumption that there is only one node with
// negative supply. This node is the sink. Otherwise, if the solver receives
// the node types then we can have more than one node with negative supply.
DEFINE_bool(graph_has_node_types, false, "Graph input contains node types");
DEFINE_bool(print_assignments, false, "Only print task to PUs assignments");
DEFINE_string(incremental_graphs, "", "File containing paths to the graphs");
DEFINE_bool(daemon, true, "True if the solver should run as a daemon");
DEFINE_int64(alpha_scaling_factor, 2,
             "Value by which Eps is divided in the cost scaling algorithm");

uint32_t num_solver_run = 0;
boost::latch latch_(2);

static bool ValidatePrintAssignments(const char* flagname,
                                     bool print_assignments) {
  if (print_assignments && !FLAGS_graph_has_node_types) {
    LOG(ERROR) << "Cannot print assignments if the graph does not have "
               << "node types.";
    return false;
  }
  return true;
}

static const bool print_assignments_validator =
  google::RegisterFlagValidator(&FLAGS_print_assignments,
                                &ValidatePrintAssignments);

inline void init(int argc, char *argv[]) {
  // Set up usage message.
  string usage("Sample usage:\nflow_scheduler --algorithm cost_scaling");
  google::SetUsageMessage(usage);

  // Use gflags to parse command line flags
  // The final (boolean) argument determines whether gflags-parsed flags should
  // be removed from the array (if true), otherwise they will re-ordered such
  // that all gflags-parsed flags are at the beginning.
  google::ParseCommandLineFlags(&argc, &argv, false);

  // Set up glog for logging output
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = true;
  FLAGS_stderrthreshold = 0;
}

void ReadGraph(flowlessly::Graph* graph, FILE* graph_file,
               bool* end_of_scheduling) {
  VLOG(2) << "----- Reading new graph file -----";
  if (num_solver_run == 0) {
    graph->ReadGraph(graph_file, true, end_of_scheduling);
  } else {
    graph->ReadGraph(graph_file, false, end_of_scheduling);
  }
}

void WriteGraph(flowlessly::Graph* graph, uint64_t algorithm_runtime,
                FILE* out_graph_file) {
  VLOG(2) << "----- Writing output -----";
  if (FLAGS_debug_output) {
    string graph_debug_file = "/tmp/flowlessly_" +
      boost::lexical_cast<string>(num_solver_run) + ".dm";
    graph->WriteGraph(graph_debug_file);
    string assignments_debug_file = "/tmp/flowlessly_assignments_" +
      boost::lexical_cast<string>(num_solver_run) + ".dm";
    graph->WriteAssignments(assignments_debug_file);
    string flow_debug_file = "/tmp/flowlessly_flow_" +
      boost::lexical_cast<string>(num_solver_run) + ".dm";
    graph->WriteFlowGraph(flow_debug_file);
  }
  num_solver_run++;
  if (FLAGS_print_assignments) {
    // Only print task to PU assignments.
    graph->WriteAssignments(out_graph_file);
  } else {
    graph->WriteFlowGraph(out_graph_file);
  }
  fprintf(out_graph_file, "c ALGORITHM TIME %ju\n", algorithm_runtime);
  fprintf(out_graph_file, "c EOI\n");
  fflush(out_graph_file);
}

void RunSolverAlgorithm(flowlessly::Solver* solver) {
  solver->ResetStatistics();
  VLOG(2) << "----- Running solver -----";
  bool completed = solver->Run();
  if (!completed) {
    return;
  }
  if (FLAGS_statistics) {
    solver->LogStatistics();
  }
  try {
    latch_.count_down();
  } catch (const boost::thread_interrupted&) {
  }
}

void RunSolver(flowlessly::Solver* solver, flowlessly::Graph* graph,
               FILE* graph_file, FILE* out_graph_file,
               bool* end_of_scheduling) {
  Statistics stats;
  ReadGraph(graph, graph_file, end_of_scheduling);
  if (*end_of_scheduling) {
    // The solver doesn't need to run because there's no more
    // scheduling to do.
    return;
  }
  boost::timer::cpu_timer algorithm_timer;
  RunSolverAlgorithm(solver);
  uint64_t algorithm_runtime =
    algorithm_timer.elapsed().wall / NANOSECONDS_IN_MICROSECOND;
  WriteGraph(graph, algorithm_runtime, out_graph_file);
  latch_.reset(1);
}

int main(int argc, char *argv[]) {
  init(argc, argv);
  FILE* graph_file = NULL;
  FILE* out_graph_file = NULL;
  Graph* graph;
  Solver* solver = NULL;
  Statistics stats;
  bool end_of_scheduling = false;
  if (FLAGS_out_graph_file.empty()) {
    out_graph_file = stdout;
  } else {
    out_graph_file = fopen(FLAGS_out_graph_file.c_str(), "w");
    CHECK(out_graph_file != NULL) << "Failed to open output graph file: "
                                  << FLAGS_out_graph_file;
  }
  // Setup the algorithm to be executed.
  if (!FLAGS_algorithm.compare("cycle_cancelling")) {
    LOG(INFO) << "----- Cycle cancelling min cost flow -----";
    AdjacencyMapGraph* adj_graph = new AdjacencyMapGraph(&stats);
    solver = new CycleCancelling(adj_graph, &stats);
    graph = adj_graph;
  } else if (!FLAGS_algorithm.compare("successive_shortest_path")) {
    LOG(INFO) << "----- Successive shortest path min cost flow " << "-----";
    AdjacencyMapGraph* adj_graph = new AdjacencyMapGraph(&stats);
    solver =  new SuccessiveShortest(adj_graph, &stats);
    graph = adj_graph;
  } else {
    LOG(FATAL) << "Unknown algorithm: " << FLAGS_algorithm;
  }
  if (!FLAGS_daemon) {
    if (!FLAGS_incremental_graphs.empty()) {
      // Replay changes from the files whose paths are
      // in FLAGS_incrmental_graphs.
      FILE* incremental_graphs = fopen(FLAGS_incremental_graphs.c_str(), "r");
      CHECK(incremental_graphs != NULL)
        << "Could not open incremental graphs file: "
        << FLAGS_incremental_graphs;
      char line[200];
      // While there are file entries in the incremental_graphs file.
      while (!feof(incremental_graphs) &&
             fscanf(incremental_graphs, "%[^\n]%*[\n]", &line[0]) > 0) {
        graph_file = fopen(line, "r");
        CHECK(graph_file != NULL) << "Could not open graph file: " << line;
        RunSolver(solver, graph, graph_file, out_graph_file,
                  &end_of_scheduling);
        CHECK_EQ(end_of_scheduling, false);
      }
    } else {
      // One-off solver run.
      if (FLAGS_graph_file.empty()) {
        graph_file = stdin;
      } else {
        graph_file = fopen(FLAGS_graph_file.c_str(), "r");
        CHECK(graph_file != NULL) << "Failed to open graph file: "
                                  << FLAGS_graph_file;
      }
      RunSolver(solver, graph, graph_file, out_graph_file, &end_of_scheduling);
    }
  } else {
    while (!end_of_scheduling) {
      RunSolver(solver, graph, stdin, out_graph_file, &end_of_scheduling);
    }
  }
  fclose(out_graph_file);
  if (!graph) {
    delete graph;
  }
  if (!solver) {
    delete solver;
  }
  return 0;
}
