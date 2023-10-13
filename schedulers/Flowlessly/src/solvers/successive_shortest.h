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

#ifndef FLOWLESSLY_SUCCESSIVE_SHORTEST_H
#define FLOWLESSLY_SUCCESSIVE_SHORTEST_H

#include "solvers/solver.h"

#include <vector>

#include "graphs/adjacency_map_graph.h"
#include "misc/statistics.h"

namespace flowlessly {

/**
 * Implements the successive shortest min cost maximum flow algorithm.
 */
class SuccessiveShortest : public Solver {
 public:
  SuccessiveShortest(AdjacencyMapGraph* graph, Statistics* stats);
  ~SuccessiveShortest();

  void PrepareState();
  bool Run();

 private:
  /**
   * Implements the successive shortest path algorithm without potentials.
   * As a result, it uses Bellman-Ford to compute the paths.
   * NOTE: It changes the graph.
   **/
  void SuccessiveShortestBellman();

  /**
   * Implements the successive shortest path algorithm with potentials.
   * It uses Dijkstra's algorithm with heaps to compute the paths.
   * NOTE: It changes the graph.
   **/
  void SuccessiveShortestDijkstra();

  AdjacencyMapGraph* graph_;
};

}  // namespace flowlessly
#endif  // FLOWLESSLY_SUCCESSIVE_SHORTEST_H
