#pragma once

#include "core/error.h"
#include "storage/storage_engine.h"

#include <fcntl.h>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace cache {
namespace storage {

/**
 * @brief Cache index class
 */
class LocalCacheIndex {
public:
  /**
   * @brief Constructor
   * @param capacity Maximum number of hash values to store
   */
  LocalCacheIndex(size_t capacity) : capacity_(capacity) {
    for (size_t i = 0; i < capacity; i++) {
      empty_block_list_.push_back(i);
    }
  }

  /**
   * @brief Checks if a hash value exists
   *
   * If it exists, returns true and updates the LRU order (moves to the front);
   * If it doesn't exist, returns false.
   *
   * @param hash The hash value to search for
   * @return True if exists, false otherwise
   */
  bool exists(const std::string &hash) {
    std::lock_guard<std::mutex> lock(index_lock_);
    auto it = index_.find(hash);
    // Only return true if data is fully written and readable (not just allocated)
    if (it == index_.end() || !it->second.ready || it->second.writing) {
      return false;
    }
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
    return true;
  }

  /**
   * @brief Acquire a slot for a hash, allocating or reusing storage as needed.
   *
   * If the hash already exists, its offset is returned and the entry promoted to MRU.
   * If the hash is new and free space remains, a free offset is consumed.
   * Otherwise the LRU entry is evicted and its offset reused for the new hash.
   *
   * @param hash The hash value we wish to store.
   * @param offset Output parameter receiving the chosen file offset index.
   * @param evicted_hash Output parameter holding the evicted hash (empty if none).
   * @return 1 if newly inserted, 0 if already existed, -1 if temporarily failed (all slots busy)
   */
  int acquire_slot(const std::string &hash, size_t &offset, std::string &evicted_hash) {
    std::lock_guard<std::mutex> lock(index_lock_);

    auto existing = index_.find(hash);
    if (existing != index_.end()) {
      if (existing->second.writing) {
        return -1;  // Write in progress, caller should retry
      }
      lru_list_.splice(lru_list_.begin(), lru_list_, existing->second.lru_iterator);
      offset = existing->second.foffset;
      evicted_hash.clear();
      return 0;  // Already exists and ready
    }

    if (!empty_block_list_.empty()) {
      offset = empty_block_list_.back();
      empty_block_list_.pop_back();
      evicted_hash.clear();
    } else {
      // Need to evict a victim from LRU list
      // Find a victim that is not currently being written (writing=false)
      auto it = lru_list_.end();
      bool found_victim = false;

      while (it != lru_list_.begin()) {
        --it;
        const std::string &candidate_hash = *it;
        auto candidate_it = index_.find(candidate_hash);

        // Data structure inconsistency detected, return failure to avoid corruption
        if (candidate_it == index_.end()) {
          fprintf(stderr, "[light_mem error] LRU list and index inconsistent, returning failure\n");
          return -1;
        }

        if (!candidate_it->second.writing) {
          // Found a valid victim (not currently being written to disk)
          offset = candidate_it->second.foffset;
          evicted_hash = candidate_hash;
          lru_list_.erase(it);
          index_.erase(candidate_it);
          found_victim = true;
          break;
        }
      }

      if (!found_victim) {
        // All slots are busy writing or LRU list is empty
        // This is temporary congestion, return -1 to let caller retry
        return -1;
      }
    }

    lru_list_.push_front(hash);
    index_[hash] = {lru_list_.begin(), offset, false, true}; // Not ready, writing in progress
    return 1;  // Newly inserted
  }

  /**
   * @brief Gets the offset of a hash value
   *
   * @param hash The hash value to search for
   * @return The offset if exists and ready, -1 otherwise
   */
  size_t get_offset(const std::string &hash) {
    std::lock_guard<std::mutex> lock(index_lock_);
    auto it = index_.find(hash);
    if (it == index_.end() || !it->second.ready || it->second.writing) {
      return -1;
    } else {
      return it->second.foffset;
    }
  }

  /**
   * @brief Mark a hash as ready (data written to disk and readable)
   * @return true if the slot is still valid, false if it was evicted
   */
  bool mark_ready(const std::string &hash) {
    std::lock_guard<std::mutex> lock(index_lock_);
    auto it = index_.find(hash);
    if (it != index_.end()) {
      it->second.ready = true;
      it->second.writing = false;
      return true;
    }
    return false; // Slot was evicted during write
  }

  /**
   * @brief Remove a hash entry and recycle its slot
   * Used for cleaning up failed writes to prevent zombie slots
   */
  void remove(const std::string &hash) {
    std::lock_guard<std::mutex> lock(index_lock_);
    auto it = index_.find(hash);
    if (it != index_.end()) {
      size_t offset = it->second.foffset;
      lru_list_.erase(it->second.lru_iterator);
      index_.erase(it);
      empty_block_list_.push_back(offset);
    }
  }

private:
  /**
   * @brief Internal structure to store LRU list iterator and exists call count
   */
  struct IndexEntry {
    std::list<std::string>::iterator lru_iterator;
    size_t foffset; // File pointer
    bool ready;     // Data is written to disk and readable
    bool writing;   // Slot is allocated but disk write in progress (evictable but not readable)
  };

  size_t capacity_;                                   ///< Maximum number of hash values to store
  std::list<std::string> lru_list_;                   ///< LRU list, head is most recent, tail is least recently used
  std::list<size_t> empty_block_list_;                ///< List of free disk blocks
  std::unordered_map<std::string, IndexEntry> index_; ///< Map from hash value to IndexEntry
  std::mutex index_lock_;                              ///< Mutex protecting index data structures
};

class LocalStorageEngine : public StorageEngine {
public:
  struct HashInfo {
    std::vector<std::shared_ptr<LocalCacheIndex>> caches;
    std::vector<std::shared_ptr<std::mutex>> io_locks;

    HashInfo() = default;
  };

  /**
   * @brief Constructor for IO sharding.
   * @param filename Base filename; actual files will be filename_0, filename_1, ...
   * @param storage_size Total file size across all shards.
   * @param shard Number of shards.
   */
  LocalStorageEngine(const std::string &filename, const size_t storage_size, const size_t shard,
                     const size_t block_size)
      : filename_(filename), storage_size_(storage_size), shard_(shard), block_size_(block_size) {

    // 每个 shard 分到的文件大小
    size_t shard_storage_size = storage_size_ / shard_;
    // 每个 shard 能存储的块数
    size_t shard_capacity = shard_storage_size / block_size;

    // 初始化缓存索引、锁和文件对象
    caches_.resize(shard_);
    io_locks_.resize(shard_);
    files_.resize(shard_);
    file_fds_.resize(shard_, -1);

    try {
      for (size_t i = 0; i < shard_; i++) {
        caches_[i] = std::make_shared<LocalCacheIndex>(shard_capacity);
        io_locks_[i] = std::make_shared<std::mutex>();
      }
      createOrOpenFiles(shard_storage_size);
    } catch (...) {
      // Clean up any partially opened files on exception
      cleanup();
      throw;
    }
  }

  ~LocalStorageEngine() override { cleanup(); }

  bool query(const std::string &hash) override {
    size_t shard_id = getShard(hash);
    return caches_[shard_id]->exists(hash);
  }

  size_t write(const char *buf, const std::string &hash) override {
    size_t shard_id = getShard(hash);
    size_t slot = 0;
    std::string evicted_hash;
    int result = caches_[shard_id]->acquire_slot(hash, slot, evicted_hash);
    (void)evicted_hash;

    // result: 1=newly inserted, 0=already exists, -1=temporarily failed
    if (result <= 0) {
      return 0;
    }

    // 执行磁盘写入，持有 writing=true 防止槽位被驱逐
    size_t offset = slot * block_size_;
    try {
      std::lock_guard<std::mutex> lock(*io_locks_[shard_id]);
      files_[shard_id].seekp(offset, std::ios::beg);
      files_[shard_id].write(buf, block_size_);
      files_[shard_id].flush();

      // 写入完成后立即标记为 ready，同时释放 writing 标志
      caches_[shard_id]->mark_ready(hash);
    } catch (...) {
      // 写入失败，清理槽位防止僵尸状态，返回 0 表示失败（不抛异常）
      caches_[shard_id]->remove(hash);
      return 0;  // Write failed, let caller retry
    }

    // 立即丢弃新写入的页缓存，避免污染读缓存
    // Note: posix_fadvise is not available on macOS
#ifndef __APPLE__
    if (file_fds_[shard_id] >= 0) {
      posix_fadvise(file_fds_[shard_id], offset, block_size_, POSIX_FADV_DONTNEED);
    }
#endif

    return block_size_;
  }

  size_t read(char *buf, const std::string &hash) override {
    size_t shard_id = getShard(hash);
    std::lock_guard<std::mutex> lock(*io_locks_[shard_id]);
    size_t block_idx = caches_[shard_id]->get_offset(hash);
    if (block_idx == static_cast<size_t>(-1)) {
      return 0;  // Hash does not exist or was evicted (cache miss)
    }

    size_t offset = block_idx * block_size_;
    try {
      files_[shard_id].seekg(offset, std::ios::beg);
      files_[shard_id].read(buf, block_size_);
      return block_size_;
    } catch (...) {
      // I/O error during read, return 0 (read failed, treat as cache miss)
      fprintf(stderr, "[light_mem error] read: I/O error for hash %s at offset %zu\n", hash.c_str(), offset);
      return 0;
    }
  }

  std::shared_ptr<HashInfo> getHashInfo() {
    auto info = std::make_shared<HashInfo>();
    info->caches = caches_;
    info->io_locks = io_locks_;
    return info;
  }

  bool setHashInfo(const std::shared_ptr<HashInfo> &info) {
    if (!info) {
      fprintf(stderr, "[light_mem error] setHashInfo: HashInfo is null\n");
      return false;
    }
    if (info->caches.size() != shard_ || info->io_locks.size() != shard_) {
      fprintf(stderr, "[light_mem error] setHashInfo: shard size mismatch (expected %zu, got caches=%zu io_locks=%zu)\n",
              shard_, info->caches.size(), info->io_locks.size());
      return false;
    }
    caches_ = info->caches;
    io_locks_ = info->io_locks;
    return true;
  }

private:
  inline size_t getShard(const std::string &hash) { return std::hash<std::string>{}(hash) % shard_; }

  // Helper function to clean up file resources
  void cleanup() {
    for (size_t i = 0; i < shard_; i++) {
      if (files_[i].is_open()) {
        try {
          files_[i].close();
        } catch (const std::exception &e) {
          fprintf(stderr, "[light_mem warning] LocalStorageEngine::cleanup: failed to close file shard %zu: %s\n", i,
                  e.what());
        } catch (...) {
          fprintf(stderr,
                  "[light_mem warning] LocalStorageEngine::cleanup: failed to close file shard %zu: unknown error\n",
                  i);
        }
      }
      if (file_fds_[i] >= 0) {
        close(file_fds_[i]);
        file_fds_[i] = -1;
      }
    }
  }

  void createOrOpenFiles(size_t shard_storage_size) {
    for (size_t i = 0; i < shard_; i++) {
      std::stringstream ss;
      ss << filename_ << "_" << i;
      std::string shard_filename = ss.str();

      // 打开 fstream 用于读写
      files_[i].open(shard_filename, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
      if (!files_[i].is_open()) {
        throw std::runtime_error("Failed to open file: " + shard_filename);
      }
      files_[i].seekp(shard_storage_size - 1, std::ios::beg);
      files_[i].write("", 1);
      files_[i].seekp(0, std::ios::beg);

      // 同时打开原生 fd 用于 posix_fadvise
      file_fds_[i] = open(shard_filename.c_str(), O_RDWR);
      if (file_fds_[i] < 0) {
        throw std::runtime_error("Failed to open native fd for: " + shard_filename);
      }
    }
  }

  std::string filename_;
  size_t storage_size_;
  size_t shard_;
  size_t block_size_;
  std::vector<std::fstream> files_;
  std::vector<int> file_fds_;                          ///< 原生文件描述符，用于 posix_fadvise
  std::vector<std::shared_ptr<std::mutex>> io_locks_; ///< Per-shard mutexes protecting file I/O operations
  std::vector<std::shared_ptr<LocalCacheIndex>> caches_;
};

} // namespace storage
} // namespace cache
