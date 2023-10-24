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

#ifndef FLOWLESSLY_CYCLE_CANCELLING_H
#define FLOWLESSLY_CYCLE_CANCELLING_H

#include "solvers/solver.h"

#include <gtest/gtest_prod.h>
#include <vector>

#include "graphs/adjacency_map_graph.h"
#include "misc/statistics.h"

namespace flowlessly {

/**
 * Implements the cycle cancelling min cost maximum flow algorithm.
 */
class CycleCancelling : public Solver {
 public:
  CycleCancelling(AdjacencyMapGraph* graph, Statistics* stats);
  ~CycleCancelling();

  void PrepareState();
  /**
   * Applies the Cycle cancelling algorithm to compute the min cost flow.
   * The complexity is O(F * M + N * M^2 * C * U).
   **/
  bool Run();

 private:
  FRIEND_TEST(CycleCancellingTests, AugmentFlow);

  void AugmentFlow(const vector<uint32_t>& predecessor, uint32_t src_node,
                   uint32_t dst_node);
  // Returns true if it removes a negative cycle.
  bool RemoveNegativeCycles(const vector<uint32_t>& predecessor);

  AdjacencyMapGraph* graph_;
};

} // namespace flowlessly
#endif // FLOWLESSLY_CYCLE_CANCELLING_H
