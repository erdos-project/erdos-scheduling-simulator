#ifndef _TETRISCHED_PARTITION_HPP_
#define _TETRISCHED_PARTITION_HPP_
#include <memory>
#include <unordered_map>
#include <vector>

namespace tetrisched {

/// A Partition represents a collection of Workers that are equivalent
/// to each other. A Task can specify that the scheduler can assign any
/// of the Workers in a Partition to it.
class Partition {
 private:
  /// The ID attached to this partition.
  uint32_t partitionId;
  /// The name of this Partition.
  std::string partitionName;
  /// The quantity of this Partition.
  /// A quantity signifies how many units of the Resource being abstracted
  /// by this Partition are available.
  size_t quantity;

 public:
  /// Constructs a Partition with 0 quantity.
  Partition(uint32_t partitionId, std::string partitionName);

  /// Constructs a Partition with the given quantity.
  Partition(uint32_t partitionId, std::string partitionName, size_t quantity);

  /// Returns the ID of this Partition.
  uint32_t getPartitionId() const;

  /// Returns the name of this Partition.
  std::string getPartitionName() const;

  /// Returns the quantity of the resources in this Partition.
  size_t getQuantity() const;

  /// Adds the given quantity to this Partition.
  Partition& operator+=(size_t quantity);

  /// Checks if the two Partitions are equivalent.
  /// Two Partitions are considered equivalent only if they have the same ID.
  bool operator==(const Partition& other) const;
};

/// Partitions are being allowed to be constructed once and shared by
/// multiple Expressions. This allows equivalence checks that currently
/// rely on the ID of the Partition instead of the actual Workers inside it.
using PartitionPtr = std::shared_ptr<Partition>;

/// @brief  A Partitions object represents a collection of Partition.
/// It is used by Tasks to specify a set of different Partitions that can
/// be used to schedule the Task.
class Partitions {
 private:
  std::unordered_map<uint32_t, PartitionPtr> partitions;

 public:
  /// Constructs a Partitions object without any Partitions.
  Partitions();

  /// Constructs a Partitions object with a collection of Partitions.
  Partitions(std::vector<PartitionPtr>& partitions);
  Partitions(std::initializer_list<PartitionPtr> partitions);

  /// Add a Partition to this Partitions object.
  void addPartition(PartitionPtr partition);

  /// Return a new Partitions object that contains the intersection of
  /// the Partitions in this object and the Partitions in the other object.
  Partitions operator|(const Partitions& other) const;

  /// Returns the number of Partition in this Partitions object.
  size_t size() const;

  /// Returns the Partitions in this Partitions object.
  std::vector<PartitionPtr> getPartitions() const;

  /// Returns the Partition with the given ID (if exists).
  std::optional<PartitionPtr> getPartition(uint32_t partitionId) const;
};
}  // namespace tetrisched
#endif
