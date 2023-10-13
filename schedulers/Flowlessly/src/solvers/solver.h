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

#ifndef FLOWLESSLY_SOLVER_H
#define FLOWLESSLY_SOLVER_H

#include <boost/thread/latch.hpp>

#include "misc/statistics.h"
#include "misc/utils.h"

namespace flowlessly {

class Solver {
 public:
  explicit Solver(Statistics* stats) : stats_(stats), latch_(NULL) {}
  virtual ~Solver() {
    // The solver is not the owner of the stats object.
  }

  inline void LogStatistics() {
    stats_->LogStatistics();
  }

  inline void ResetStatistics() {
    stats_->ResetStatistics();
  }

  /**
   * Must return false if it terminates earlier (i.e. if the thread
   * is interrupted).
   */
  virtual bool Run() = 0;

  virtual void PrepareState() = 0;

  void SetLatch(boost::latch* latch) {
    latch_ = latch;
  }

 protected:
  Statistics* stats_;
  boost::latch* latch_;
};

} // namespace flowlessly
#endif // FLOWLESSLY_SOLVER_H
