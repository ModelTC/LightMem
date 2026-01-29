#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cache {
namespace storage {

/**
 * @brief Cache index class
 */
class LocalCacheIndex {
public:
  using Hook = std::function<void(const std::string &hash)>;

  /**
   * @brief Internal structure to store LRU list iterator and exists call count
   */
  struct IndexEntry {
    std::list<std::string>::iterator lru_iterator;
    size_t slot_id; // Slot index
    bool ready;     // Data is written to disk and readable
    bool writing;   // Slot is allocated but disk write in progress (evictable but not readable)
    uint32_t crc;   // CRC of the data block (0 means unknown)
  };

  /**
   * @brief Constructor
   * @param capacity Maximum number of hash values to store
   */
  explicit LocalCacheIndex(size_t capacity);

  // Optional hooks:
  // - on_ready: called (under index_lock_) when a hash becomes ready/readable.
  // - on_erase: called (under index_lock_) right before a hash is removed/evicted.
  // These are intended for maintaining auxiliary indices (e.g. hash->shard hints).
  void set_hooks(Hook on_ready, Hook on_erase);

  void reset();

  // Insert a ready mapping (used by recovery / redis warmup).
  // If slot is already held by another hash, that entry is removed.
  void put_ready(const std::string &hash, size_t slot_id, uint32_t crc = 0);

  /**
   * @brief Checks if a hash value exists
   *
   * If it exists, returns true and updates the LRU order (moves to the front);
   * If it doesn't exist, returns false.
   */
  bool exists(const std::string &hash);

  /**
   * @brief Acquire a slot for a hash, allocating or reusing storage as needed.
   *
   * @return 1 if newly inserted, 0 if already existed, -1 if temporarily failed (all slots busy)
   */
  int acquire_slot(const std::string &hash, size_t &slot_id, std::string &evicted_hash);

  // True LRU eviction observability.
  // This counts only evictions triggered by acquire_slot() due to full capacity.
  uint64_t eviction_count() const;
  bool eviction_observed() const;

  void mark_ready(const std::string &hash, uint32_t crc = 0);
  void remove(const std::string &hash);

  /**
   * @brief Gets the offset of a hash value
   * @return slot id if exists and ready, size_t(-1) otherwise
   */
  size_t get_offset(const std::string &hash);

  // Returns true iff hash exists and is ready; outputs slot and crc.
  // crc==0 means CRC is unknown/unavailable.
  bool get_offset_and_crc(const std::string &hash, size_t &slot_id, uint32_t &crc);

  // Snapshot operations
  bool saveToSnapshot(const std::string &filename);
  bool loadFromSnapshot(const std::string &filename);

  // Best-effort dump of ready entries for building auxiliary indices.
  // Thread-safe snapshot under the internal mutex.
  void dump_ready(std::vector<std::string> &out);

private:
  size_t capacity_;                                   ///< Maximum number of hash values to store
  std::list<std::string> lru_list_;                   ///< LRU list, head is most recent, tail is least recently used
  std::list<size_t> empty_block_list_;                ///< List of free disk blocks
  std::unordered_map<std::string, IndexEntry> index_; ///< Map from hash value to IndexEntry
  mutable std::mutex index_lock_;                     ///< Mutex protecting index data structures

  Hook on_ready_;
  Hook on_erase_;

  uint64_t eviction_count_{0};
};

} // namespace storage
} // namespace cache
