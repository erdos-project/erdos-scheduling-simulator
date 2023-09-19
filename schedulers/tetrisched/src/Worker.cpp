#include "tetrisched/Worker.hpp"

namespace tetrisched {
Worker::Worker(uint32_t workerId, std::string workerName)
    : _workerId(workerId), _workerName(workerName) {}

uint32_t Worker::getWorkerId() const { return _workerId; }

const std::string& Worker::getWorkerName() const { return _workerName; }
}  // namespace tetrisched
