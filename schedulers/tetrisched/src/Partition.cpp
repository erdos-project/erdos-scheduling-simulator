#include "tetrisched/Partition.hpp"

namespace tetrisched {
Partition::Partition(uint32_t partitionId, std::string partitionName)
    : partitionId(partitionId), partitionName(partitionName), quantity(0) {}

Partition::Partition(uint32_t partitionId, std::string partitionName,
                     size_t quantity)
    : partitionId(partitionId),
      partitionName(partitionName),
      quantity(quantity) {}

uint32_t Partition::getPartitionId() const { return partitionId; }

std::string Partition::getPartitionName() const { return partitionName; }

size_t Partition::getQuantity() const { return quantity; }

Partition& Partition::operator+=(size_t quantity) {
  this->quantity += quantity;
  return *this;
}

/// The equivalence checks currently rely on the unique ID.
bool Partition::operator==(const Partition& other) const {
  return partitionId == other.partitionId;
}

/// Constructs a Partitions object without any Partitions.
Partitions::Partitions() {}

/// Constructs a Partitions object with a collection of Partitions.
Partitions::Partitions(std::vector<PartitionPtr>& partitions) {
  for (auto partition : partitions) {
    this->addPartition(partition);
  }
}

Partitions::Partitions(std::initializer_list<PartitionPtr> partitions) {
  for (auto partition : partitions) {
    this->addPartition(partition);
  }
}

/// Add a Partition to this Partitions object.
void Partitions::addPartition(PartitionPtr partition) {
  if (partitions.find(partition->getPartitionId()) != partitions.end()) {
    throw tetrisched::exceptions::RuntimeException(
        "Partition with ID " + std::to_string(partition->getPartitionId()) +
        " already exists.");
  }
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

/// Returns the Partition with the given ID (if exists).
std::optional<PartitionPtr> Partitions::getPartition(
    uint32_t partitionId) const {
  if (partitions.find(partitionId) != partitions.end()) {
    return partitions.at(partitionId);
  } else {
    return std::nullopt;
  }
}
}  // namespace tetrisched
