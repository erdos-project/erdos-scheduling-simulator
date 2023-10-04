#include <gtest/gtest.h>

#include "tetrisched/Partition.hpp"
#include "tetrisched/Worker.hpp"

TEST(WorkerTest, TestWorkerInitialization) {
  // Create a Worker.
  tetrisched::Worker worker = tetrisched::Worker(1, "worker1");
  EXPECT_EQ(worker.getWorkerId(), 1)
      << "The ID of the Worker was expected to be 1.";
  EXPECT_EQ(worker.getWorkerName(), "worker1")
      << "The name of the Worker was expected to be worker1.";
}

TEST(PartitionTest, TestPartitionInitialized) {
  // Create an empty Partition.
  tetrisched::Partition partition = tetrisched::Partition();
  EXPECT_EQ(partition.size(), 0) << "The partition was expected to be empty.";
  EXPECT_GT(partition.getPartitionId(), 0)
      << "The partition ID was expected to be greater than 0.";
}

TEST(PartitionsTest, TestIntersectionEmpty) {
  // Create a set of Workers.
  std::vector<std::pair<tetrisched::WorkerPtr, size_t>> workersInPartition1 = {
      std::make_pair(std::make_shared<tetrisched::Worker>(1, "worker1"), 2),
  };
  std::vector<std::pair<tetrisched::WorkerPtr, size_t>> workersInPartition2 = {
      std::make_pair(std::make_shared<tetrisched::Worker>(2, "worker2"), 2),
  };

  // Create two partitions from the Workers.
  std::vector<tetrisched::PartitionPtr> p1 = {
      std::make_shared<tetrisched::Partition>(workersInPartition1)};
  std::vector<tetrisched::PartitionPtr> p2 = {
      std::make_shared<tetrisched::Partition>(workersInPartition2)};

  // Wrap the individual PartitionPtr into Partitions.
  tetrisched::Partitions firstPartition = tetrisched::Partitions(p1);
  tetrisched::Partitions secondPartition = tetrisched::Partitions(p2);

  auto intersectionPartition = firstPartition | secondPartition;
  EXPECT_EQ(intersectionPartition.size(), 0)
      << "The intersection of disjoint Partitions was expected to be empty.";
}

TEST(PartitionsTest, TestCorrectIntersectionSize) {
  // Create a set of Workers.
  std::vector<std::pair<tetrisched::WorkerPtr, size_t>> workersInPartition1 = {
      std::make_pair(std::make_shared<tetrisched::Worker>(1, "worker1"), 2),
  };
  std::vector<std::pair<tetrisched::WorkerPtr, size_t>> workersInPartition2 = {
      std::make_pair(std::make_shared<tetrisched::Worker>(2, "worker2"), 2),
  };

  // Create the Partition.
  tetrisched::PartitionPtr partition1 =
      std::make_shared<tetrisched::Partition>(workersInPartition1);
  tetrisched::PartitionPtr partition2 =
      std::make_shared<tetrisched::Partition>(workersInPartition2);

  // Create the Partitions.
  tetrisched::Partitions firstPartition = tetrisched::Partitions();
  firstPartition.addPartition(partition1);
  firstPartition.addPartition(partition2);

  tetrisched::Partitions secondPartition = tetrisched::Partitions();
  secondPartition.addPartition(partition2);

  // Check that the intersection is correct.
  auto intersectionPartition = firstPartition | secondPartition;
  EXPECT_EQ(intersectionPartition.size(), 1)
      << "The intersection of the Partitions was expected to be of size 1.";
}
