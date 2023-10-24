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

#ifndef FLOWLESSLY_UTILS_H
#define FLOWLESSLY_UTILS_H

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <unistd.h>
#include <set>
#include <vector>

#include "graphs/adjacency_map_graph.h"

namespace flowlessly {

/**
 * Compute the shortest distance over a graph using Bellman-Ford.
 * @param graph the graph for which to compute the shortest distance
 * @param active_node_ids set containing active node ids
 * @param predecessor vector which will contain predecessor nodes
 **/
void BellmanFord(AdjacencyMapGraph* graph, const set<uint32_t>& active_node_ids,
                 vector<uint32_t>* predecessor);
void BellmanFordWithoutPotentials(AdjacencyMapGraph* graph,
                                  const set<uint32_t>& active_node_ids,
                                  vector<uint32_t>* predecessor);

/**
 * Compute the shortest distance over a graph using Disjkstra's algorithm
 * with heaps. The computation returns upon finishing visiting the first
 * node with negative supply.
 * @param graph the graph for which to compute the shortest distance
 * @param active_node_ids set containing the active node ids
 * @param predecessor vector which will contain predecessor nodes
 * @return the id of the first node with negative supply the algorithm finishes
 * visiting
 **/
uint32_t DijkstraOptimized(AdjacencyMapGraph* graph,
                       const set<uint32_t>& active_node_ids,
                       vector<uint32_t>* predecessor);

/**
 * Compute the shortest distance over a graph using simple Disjkstra's
 * algorithm.
 * @param graph the graph for which to compute the shortest distance
 * @param active_node_ids set containing the active node ids
 * @param predecessor vector which will contain predecessor nodes
 **/
void DijkstraSimple(AdjacencyMapGraph* graph,
                    const set<uint32_t>& active_node_ids,
                    vector<uint32_t>* predecessor);

/**
 * Computes the maximum flow of a graph using the Ford-Fulkerson algorithm.
 * The Complexity of the algorithm is O(E * F). Where F is the max flow value.
 *
 * NOTE: This method changes the graph!
 * NOTE: This method does not update the set of active_node_ids!
 *
 * @param graph the graph over which to compute the maximum flow
 **/
void MaxFlow(AdjacencyMapGraph* graph);

/**
 * Reverts the flow from nodes with demand to the sink. This method is useful
 * for the cases when we want to enforce that only the sink has demand.
 * @param graph the graph over which to revert the flow
 */
void ReverseMaxFlow(AdjacencyMapGraph* graph);

} // namespace flowlessly
#endif // FLOWLESSLY_UTILS_H
