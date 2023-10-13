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

#ifndef STATISTICS_H
#define STATISTICS_H

#include <gflags/gflags.h>
#include <sys/time.h>
#include <sys/resource.h>

DECLARE_bool(statistics);

namespace flowlessly {

  /**
   * Stores statistics related to various operations such as:
   * total time spent relabeling, total time spent arc fixing etc.
   **/
class Statistics {
 public:
  void ResetStatistics();
  void LogStatistics();

  inline double get_arcs_fixing_time() {
    return arcs_fixing_time_;
  }

  inline double get_arcs_unfixing_time() {
    return arcs_unfixing_time_;
  }

  inline double get_discharge_time() {
    return discharge_time_;
  }

  inline double get_global_update_time() {
    return global_update_time_;
  }

  inline uint32_t get_num_pushes() {
    return num_pushes_;
  }

  inline uint32_t get_num_refines() {
    return num_refines_;
  }

  inline uint32_t get_num_relabels() {
    return num_relabels_;
  }

  inline double get_price_refine_time() {
    return price_refine_time_;
  }

  inline double get_refine_time_() {
    return refine_time_;
  }

  inline double get_relabel_time() {
    return relabel_time_;
  }

  inline double get_push_time() {
    return push_time_;
  }

  inline double get_time() {
    struct rusage r;
    getrusage(0, &r);
    return r.ru_utime.tv_sec + static_cast<double>(r.ru_utime.tv_usec) /
      1000000.0;
  }

  inline double get_update_admissible_time() {
    return update_admissible_time_;
  }

  inline void increment_num_pushes() {
    num_pushes_++;
  }

  inline void increment_num_refines() {
    num_refines_++;
  }

  inline void increment_num_relabels() {
    num_relabels_++;
  }

  inline void update_admissible_start_time() {
    if (FLAGS_statistics) {
      update_admissible_time_ -= get_time();
    }
  }

  inline void update_admissible_end_time() {
    if (FLAGS_statistics) {
      update_admissible_time_ += get_time();
    }
  }

  inline void update_arcs_fixing_start_time() {
    if (FLAGS_statistics) {
      arcs_fixing_time_ -= get_time();
    }
  }

  inline void update_arcs_fixing_end_time() {
    if (FLAGS_statistics) {
      arcs_fixing_time_ += get_time();
    }
  }

  inline void update_arcs_unfixing_start_time() {
    if (FLAGS_statistics) {
      arcs_unfixing_time_ -= get_time();
    }
  }

  inline void update_arcs_unfixing_end_time() {
    if (FLAGS_statistics) {
      arcs_unfixing_time_ += get_time();
    }
  }

  inline void update_discharge_start_time() {
    if (FLAGS_statistics) {
      discharge_time_ -= get_time();
    }
  }

  inline void update_discharge_end_time() {
    if (FLAGS_statistics) {
      discharge_time_ += get_time();
    }
  }

  inline void update_global_update_start_time() {
    if (FLAGS_statistics) {
      global_update_time_ -= get_time();
    }
  }

  inline void update_global_update_end_time() {
    if (FLAGS_statistics) {
      global_update_time_ += get_time();
    }
  }

  inline void update_price_refine_start_time() {
    if (FLAGS_statistics) {
      price_refine_time_ -= get_time();
    }
  }

  inline void update_price_refine_end_time() {
    if (FLAGS_statistics) {
      price_refine_time_ += get_time();
    }
  }

  inline void update_push_start_time() {
    if (FLAGS_statistics) {
      push_time_ -= get_time();
    }
  }

  inline void update_push_end_time() {
    if (FLAGS_statistics) {
      push_time_ += get_time();
    }
  }

  inline void update_refine_start_time() {
    if (FLAGS_statistics) {
      refine_time_ -= get_time();
    }
  }

  inline void update_refine_end_time() {
    if (FLAGS_statistics) {
      refine_time_ += get_time();
    }
  }

  inline void update_relabel_start_time() {
    if (FLAGS_statistics) {
      relabel_time_ -= get_time();
    }
  }

  inline void update_relabel_end_time() {
    if (FLAGS_statistics) {
      relabel_time_ += get_time();
    }
  }

  double arcs_fixing_time_;
  double arcs_unfixing_time_;
  double refine_time_;
  double discharge_time_;
  double global_update_time_;
  double price_refine_time_;
  double relabel_time_;
  double push_time_;
  double update_admissible_time_;
  uint32_t num_relabels_;
  uint32_t num_pushes_;
  uint32_t num_refines_;
};

} // namespace flowlessly
#endif
