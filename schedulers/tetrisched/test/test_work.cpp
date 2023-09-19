#include <gtest/gtest.h>

#include "tetrisched/Task.hpp"

TEST(TaskTest, TestTaskInitialization) {
  // Create a Task.
  tetrisched::Task task = tetrisched::Task(1, "task1");
  EXPECT_EQ(task.getTaskId(), 1)
      << "The ID of the Task was expected to be 1.";
  EXPECT_EQ(task.getTaskName(), "task1")
      << "The name of the Task was expected to be task1.";
}
