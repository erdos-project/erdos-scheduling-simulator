#include "tetrisched/Partition.hpp"

namespace tetrisched {
// Initialize the static counter for Partition IDs.
// This is required by the compiler.
uint32_t Partition::partitionIdCounter = 0;

Partition::Partition() : partitionId(partitionIdCounter++) {}

/// Constructs a Partition with a collection of Workers.
Partition::Partition(std::vector<std::pair<WorkerPtr, size_t>>& workers)
    : partitionId(partitionIdCounter++) {
  for (const auto& workerPair : workers) {
    this->workers[workerPair.first->getWorkerId()] = workerPair;
  }
}

Partition::Partition(
    std::initializer_list<std::pair<WorkerPtr, size_t>> workers)
    : partitionId(partitionIdCounter++) {
  for (const auto& workerPair : workers) {
    this->workers[workerPair.first->getWorkerId()] = workerPair;
  }
}

/// Add a Worker to this Partition.
void Partition::addWorker(WorkerPtr worker, size_t quantity) {
  workers[worker->getWorkerId()] =
      std::pair<WorkerPtr, size_t>(worker, quantity);
}

/// Returns the ID of this Partition.
uint32_t Partition::getPartitionId() const { return partitionId; }

/// The equivalence checks currently rely on the unique ID.
bool Partition::operator==(const Partition& other) const {
  return partitionId == other.partitionId;
}

/// Returns the number of Workers in this Partition.
size_t Partition::size() const {
  size_t totalQuantity = 0;
  for (const auto& [partitionId, worker] : workers) {
    totalQuantity += worker.second;
  }
  return totalQuantity;
}

/// Constructs a Partitions object without any Partitions.
Partitions::Partitions() {}

/// Constructs a Partitions object with a collection of Partitions.
Partitions::Partitions(std::vector<PartitionPtr>& partitions) {
  for (auto partition : partitions) {
    this->partitions[partition->getPartitionId()] = partition;
  }
}

Partitions::Partitions(std::initializer_list<PartitionPtr> partitions) {
  for (auto partition : partitions) {
    this->partitions[partition->getPartitionId()] = partition;
  }
}

/// Add a Partition to this Partitions object.
void Partitions::addPartition(PartitionPtr partition) {
  partitions[partition->getPartitionId()] = partition;
}

/// Return a new Partitions object that contains the intersection of
/// the Partitions in this object and the Partitions in the other object.
Partitions Partitions::operator|(const Partitions& other) const {
  Partitions result;
  for (auto partition : partitions) {
    if (other.partitions.find(partition.first) != other.partitions.end()) {
      result.addPartition(partition.second);
    }
  }
  return result;
}

/// Returns the number of Partition in this Partitions object.
size_t Partitions::size() const { return partitions.size(); }

/// Returns the Partitions in this Partitions object.
std::vector<PartitionPtr> Partitions::getPartitions() const {
  std::vector<PartitionPtr> partitionsVec;
  for (auto& partition : partitions) {
    partitionsVec.push_back(partition.second);
  }
  return partitionsVec;
}
}  // namespace tetrisched
