#include <gtest/gtest.h>

#include "tetrisched/Partition.hpp"

TEST(PartitionTest, TestPartitionInitialized) {
  // Create an empty Partition.
  tetrisched::Partition partition = tetrisched::Partition(0, "partition1");
  EXPECT_EQ(partition.getQuantity(), 0)
      << "The partition was expected to be empty.";
  EXPECT_EQ(partition.getPartitionId(), 0)
      << "The partition ID was expected to be 0.";
}

TEST(PartitionsTest, TestIntersectionEmpty) {
  // Create two Partition objects.
  std::vector<tetrisched::PartitionPtr> p1 = {
      std::make_shared<tetrisched::Partition>(1, "partition1", 1)};
  std::vector<tetrisched::PartitionPtr> p2 = {
      std::make_shared<tetrisched::Partition>(2, "partition2", 1)};

  // Wrap the individual PartitionPtr into Partitions.
  tetrisched::Partitions firstPartition = tetrisched::Partitions(p1);
  tetrisched::Partitions secondPartition = tetrisched::Partitions(p2);

  auto intersectionPartition = firstPartition | secondPartition;
  EXPECT_EQ(intersectionPartition.size(), 0)
      << "The intersection of disjoint Partitions was expected to be empty.";
}

TEST(PartitionsTest, TestCorrectIntersectionSize) {
  // Create the Partition.
  tetrisched::PartitionPtr partition1 =
      std::make_shared<tetrisched::Partition>(1, "partition1", 1);
  tetrisched::PartitionPtr partition2 =
      std::make_shared<tetrisched::Partition>(2, "partition2", 1);

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
