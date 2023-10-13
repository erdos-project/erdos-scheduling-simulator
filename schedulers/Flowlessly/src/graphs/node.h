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

#ifndef FLOWLESSLY_NODE_H
#define FLOWLESSLY_NODE_H

namespace flowlessly {

// Node type is used to construct the mapping of tasks to pus.
// NOTE: Do not reorder types because it will affect the communication with
// the scheduler.
// OTHER is the first in the enum so that an unset type (i.e. 0) defaults
// to OTHER.
enum NodeType {
  OTHER = 0,
  TASK = 1,
  PU = 2,
  SINK = 3,
  MACHINE = 4,
  INTERMEDIATE_RES = 5
};

enum NodeStatus {
  NOT_VISITED = 0,
  VISITING = 1,
  VISITED = 2
};

class Node {
 public:
  Node() : supply(0), potential(0), distance(0), type(NodeType::OTHER),
    status(NodeStatus::NOT_VISITED) {
  }
  Node(const Node& node)
    : supply(node.supply), potential(node.potential),
    distance(node.distance), type(node.type), status(node.status) {
  }

  int32_t supply;
  int64_t potential;
  int64_t distance; // node's distance from the source (in multiples of eps)
  NodeType type;
  NodeStatus status; // node's status during DFS
};

} // namespace flowlessly
#endif // FLOWLESSLY_NODE_H
