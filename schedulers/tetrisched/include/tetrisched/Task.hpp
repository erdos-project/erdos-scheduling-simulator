#ifndef _TETRISCHED_TASK_HPP_
#define _TETRISCHED_TASK_HPP_

#include <string>

namespace tetrisched {
class Task {
 private:
  uint32_t taskId;
  std::string taskName;

 public:
  Task(uint32_t taskId, std::string taskName)
      : taskId(taskId), taskName(taskName) {}

  uint32_t getTaskId() const { return taskId; }

  const std::string& getTaskName() const { return taskName; }
};

typedef std::shared_ptr<Task> TaskPtr;
}  // namespace tetrisched
#endif // _TETRISCHED_TASK_HPP_
