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

#include "misc/statistics.h"

#include <glog/logging.h>

namespace flowlessly {

  void Statistics::LogStatistics() {
    LOG(INFO) << "Refine time: " << refine_time_;
    LOG(INFO) << "Discharge time: " << discharge_time_;
    LOG(INFO) << "Global update time: " << global_update_time_;
    LOG(INFO) << "Price refine time: " << price_refine_time_;
    LOG(INFO) << "Arcs fixing time: " << arcs_fixing_time_;
    LOG(INFO) << "Arcs unfixing time: " << arcs_unfixing_time_;
    LOG(INFO) << "Relabel time: " << relabel_time_;
    LOG(INFO) << "Push time: " << push_time_;
    LOG(INFO) << "Update admissible time: " << update_admissible_time_;
    LOG(INFO) << "Number of pushes: " << num_pushes_;
    LOG(INFO) << "Number of refines: " << num_refines_;
    LOG(INFO) << "Number of relabels: " << num_relabels_;
  }

  void Statistics::ResetStatistics() {
    refine_time_ = 0;
    discharge_time_ = 0;
    global_update_time_ = 0;
    price_refine_time_ = 0;
    arcs_fixing_time_ = 0;
    arcs_unfixing_time_ = 0;
    relabel_time_ = 0;
    push_time_ = 0;
    update_admissible_time_ = 0;
    num_pushes_ = 0;
    num_refines_ = 0;
    num_relabels_ = 0;
  }

} // namespace flowlessly
