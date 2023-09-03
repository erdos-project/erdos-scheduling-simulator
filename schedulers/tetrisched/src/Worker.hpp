#ifndef _TETRISCHED_WORKER_HPP
#define _TETRISCHED_WORKER_HPP

#include "common.hpp"

namespace alsched {
class Worker {
 private:
  uint32_t _workerId;
  std::string _workerName;

 public:
  /// Constructs a Worker with the given ID and Name.
  Worker(uint32_t workerId, std::string workerName)
      : _workerId(workerId), _workerName(workerName) {}

  /// Retrieves the ID of the Worker.
  uint32_t getWorkerId() const { return _workerId; }

  /// Retrieves the name of the Worker.
  const std::string& getWorkerName() const { return _workerName; }
};
}  // namespace alsched
