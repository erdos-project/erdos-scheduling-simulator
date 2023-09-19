#ifndef TETRISCHED_WORKER_HPP_
#define TETRISCHED_WORKER_HPP_
#include <memory>
#include <string>

namespace tetrisched {
class Worker {
 private:
  uint32_t _workerId;
  std::string _workerName;

 public:
  /// Constructs a Worker with the given ID and Name.
  Worker(uint32_t workerId, std::string workerName);

  /// Retrieves the ID of the Worker.
  uint32_t getWorkerId() const;

  /// Retrieves the name of the Worker.
  const std::string &getWorkerName() const;
};

typedef std::shared_ptr<Worker> WorkerPtr;

}  // namespace tetrisched
#endif
