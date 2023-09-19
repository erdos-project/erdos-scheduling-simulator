#ifndef _TETRISCHED_PARTITION_HPP_
#define _TETRISCHED_PARTITION_HPP_
#include "Worker.hpp"
#include <unordered_map>
#include <vector>

namespace tetrisched {

/// A Partition represents a collection of Workers that are equivalent
/// to each other. A Task can specify that the scheduler can assign any
/// of the Workers in a Partition to it.
class Partition {
 private:
  static uint32_t partitionIdCounter;
  /// The ID attached to this partition.
  uint32_t partitionId;
  /// A map from the ID of the  Worker to an instance of the Worker.
  std::unordered_map<uint32_t, WorkerPtr> workers;

 public:
  /// Constructs a Partition without any Workers.
  Partition();

  /// Constructs a Partition with a collection of Workers.
  Partition(std::vector<WorkerPtr>& workers);

  /// Add a Worker to this Partition.
  void addWorker(WorkerPtr worker);

  /// Returns the ID of this Partition.
  uint32_t getPartitionId() const;

  /// Checks if the two Partitions are equivalent.
  /// Two Partitions are considered equivalent only if they have the same ID.
  // TODO (Sukrit): The assumption is that only one instance of a Partition
  // will be created and will be used by everyone using a shared_pointer.
  // If this assumption does not pan out, this check should be updated to
  // check equivalence of Workers in this Partition.
  bool operator==(const Partition& other) const;

  /// Returns the number of Workers in this Partition.
  size_t size() const;
};

/// Partitions are being allowed to be constructed once and shared by
/// multiple Expressions. This allows equivalence checks that currently
/// rely on the ID of the Partition instead of the actual Workers inside it.
typedef std::shared_ptr<Partition> PartitionPtr;


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

  /// Add a Partition to this Partitions object.
  void addPartition(PartitionPtr partition);

  /// Return a new Partitions object that contains the intersection of
  /// the Partitions in this object and the Partitions in the other object.
  Partitions operator|(const Partitions& other) const;

  /// Returns the number of Partition in this Partitions object.
  size_t size() const;
};
}  // namespace tetrisched
#endif
