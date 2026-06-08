#pragma once

#include "storage/local_cache_index.h"
#include "storage/local_storage_wal.h"
#include "storage/redis_client.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/types.h> // off_t

namespace cache {
namespace storage {

class LocalStorageEngine {
public:
  struct HashInfo {
    std::vector<std::shared_ptr<LocalCacheIndex>> caches;
    std::vector<std::shared_ptr<std::shared_mutex>> io_locks;

    HashInfo() = default;
  };

  struct JournalTask {
    size_t shard_id;
    uint64_t epoch;
    uint64_t write_offset;
    uint32_t write_len;
    uint32_t data_crc;
    size_t slot_id;
    std::string hash;
    std::string evicted_hash;

    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    bool success = false;
  };

  LocalStorageEngine(const std::string &filename, size_t storage_size, size_t shard, size_t block_size,
                     const std::string &index_endpoint, const std::string &index_prefix = std::string());
  ~LocalStorageEngine();

  // Batch query variant for high-throughput callers.
  std::vector<bool> queryMany(const std::vector<std::string> &hashs);
  // Length-aware variants used by the cache service.
  size_t write(const char *buf, const std::string &hash, uint32_t data_crc, uint32_t len_bytes);
  size_t read(char *buf, const std::string &hash, uint32_t len_bytes);

  // Distributed/online mode control-plane API.
  // Update per-shard ownership state as seen by this node.
  void updateShardAssignments(const std::vector<size_t> &shard_ids, const std::vector<uint64_t> &epochs,
                              const std::vector<uint8_t> &draining);

  // For ownership handoff: rebuild Redis index from local snapshot + WAL for a shard.
  void recoverShardToRedis(size_t shard_id);

  // For ownership handoff: choose incremental vs full Redis recovery based on Redis health.
  void recoverShardToRedisSmart(size_t shard_id);

  // Observability: number of in-flight operations targeting a shard.
  uint32_t shardInflight(size_t shard_id) const;
  uint64_t shardWrittenBytes(size_t shard_id) const;
  uint64_t writtenBytes() const;

  // True LRU eviction observability (local disk cache). These counts are local to this node.
  // `shardEvictionCount`: evictions for a single shard.
  uint64_t shardEvictionCount(size_t shard_id) const;
  // `evictionCount`: total evictions across all shards.
  uint64_t evictionCount() const;
  // `evictionObserved`: quick boolean check (evictionCount() > 0).
  bool evictionObserved() const;

  // Online-mode observability: shard ownership and effective write capacity.
  size_t ownedShardCount() const;
  uint64_t effectiveWritableCapacityBytes() const;

  std::shared_ptr<HashInfo> getHashInfo();
  bool setHashInfo(const std::shared_ptr<HashInfo> &info);


private:
  size_t getShard(const std::string &hash) const;
  bool isShardWritable(size_t shard_id, uint64_t *epoch_out = nullptr) const;
  std::optional<size_t> findShardInRedis(const std::string &hash, size_t *slot_id_out);
  size_t pickWritableShard(const std::string &hash) const;
  std::optional<size_t> handleExistingLocalDuplicate(size_t shard_id, const std::string &hash, uint32_t data_crc,
                                                     int &result, size_t &slot_id, std::string &evicted_hash,
                                                     bool do_global_dedupe, const std::string &global_key);

  // Fixed-size thread pool servicing concurrent per-stripe disk I/O (scheme A).
  // Workers pull no-arg tasks from a queue; a caller fans out stripe I/O and waits on a local latch.
  class IoThreadPool {
  public:
    explicit IoThreadPool(size_t threads) {
      if (threads == 0) {
        threads = 1;
      }
      workers_.reserve(threads);
      for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
      }
    }

    ~IoThreadPool() {
      {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
      }
      cv_.notify_all();
      for (auto &t : workers_) {
        if (t.joinable()) {
          t.join();
        }
      }
    }

    IoThreadPool(const IoThreadPool &) = delete;
    IoThreadPool &operator=(const IoThreadPool &) = delete;

    size_t size() const { return workers_.size(); }

    void submit(std::function<void()> task) {
      {
        std::lock_guard<std::mutex> lk(mu_);
        tasks_.push(std::move(task));
      }
      cv_.notify_one();
    }

  private:
    void workerLoop() {
      for (;;) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lk(mu_);
          cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
          if (stop_ && tasks_.empty()) {
            return;
          }
          task = std::move(tasks_.front());
          tasks_.pop();
        }
        task();
      }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
  };

  struct InflightCounter {
    struct Guard {
      explicit Guard(std::atomic<uint32_t> *ctr) : ctr_(ctr), active_(ctr != nullptr) {
        if (active_) {
          ctr_->fetch_add(1, std::memory_order_relaxed);
        }
      }

      Guard(const Guard &) = delete;
      Guard &operator=(const Guard &) = delete;

      Guard(Guard &&other) noexcept : ctr_(other.ctr_), active_(other.active_) {
        other.ctr_ = nullptr;
        other.active_ = false;
      }
      Guard &operator=(Guard &&) = delete;

      ~Guard() {
        if (active_ && ctr_) {
          ctr_->fetch_sub(1, std::memory_order_relaxed);
        }
      }

    private:
      std::atomic<uint32_t> *ctr_;
      bool active_;
    };

    Guard acquire() const { return Guard(&ctr_); }
    uint32_t load(std::memory_order order) const { return ctr_.load(order); }
    void store(uint32_t v, std::memory_order order) { ctr_.store(v, order); }

  private:
    mutable std::atomic<uint32_t> ctr_{0};
  };

  // Best-effort local hint to avoid O(shards) scans and repeated Redis lookups.
  // Only used in online_mode_ and always re-validated under the shard io lock.
  struct LocalShardHintHit {
    size_t shard_id;
    InflightCounter::Guard inflight_guard;
  };

  // Returns a shard-local hint plus an inflight guard.
  // Holding the returned guard keeps shardInflight(shard_id) > 0, so shard handoff waits for
  // any local-hit read that is about to proceed.
  std::optional<LocalShardHintHit> localShardHint(const std::string &hash) const;
  void noteLocalShardHint(const std::string &hash, size_t shard_id);
  void eraseLocalShardHint(const std::string &hash);

  void onEpochChangedLocked(size_t shard_id, uint64_t new_epoch);
  void appendEpochMarkerLocked(size_t shard_id, uint64_t epoch);

  void startJournalWorkers();
  void stopJournalWorkers();
  void journalWorkerLoop(size_t shard_id);

  void cleanup();
  void createOrOpenFiles(size_t shard_storage_size);
  void initIndexBackend(const std::string &endpoint, const std::string &index_prefix);

  bool preadAll(int fd, void *buf, size_t len, off_t offset);
  bool pwriteAll(int fd, const void *buf, size_t len, off_t offset);

  // Scheme A: striped data block I/O. A block stored at logical slot `slot_id` is split into
  // `num_stripes_` segments, segment j living in stripe file file_fds_[shard_id][j] at
  // byte offset `slot_id * stripe_size_`. The segments are issued concurrently through io_pool_
  // so a single block fans out to multiple files (the unit of parallelism on afs).
  bool writeDataBlock(size_t shard_id, size_t slot_id, const char *buf, size_t len_bytes);
  bool readDataBlock(size_t shard_id, size_t slot_id, char *buf, size_t len_bytes);
  bool syncDataBlock(size_t shard_id);
  void fadviseDataDontNeed(size_t shard_id, size_t slot_id, size_t len_bytes);

  bool readSuperBlockAt(size_t shard_id, off_t off, SuperBlock &sb);
  void writeSuperBlockAt(size_t shard_id, off_t off, SuperBlock sb);
  void ensureSuperBlocksInitialized(size_t shard_id);

  void appendJournalRecord(size_t shard_id, uint64_t write_offset, uint32_t write_len, uint32_t data_crc,
                           const std::string &hash, const std::string &evicted_hash, bool do_fsync);

  void maybeCheckpoint(size_t shard_id);
  void checkpoint(size_t shard_id);
  void writeCheckpointSuperblockOnly(size_t shard_id);

  void truncateJournalToHeader(size_t shard_id);

  struct JournalOp {
    std::string hash;
    std::string evicted;
    size_t slot_id = 0;
    uint32_t data_crc = 0;
  };
  void scanWalOps(size_t shard_id, size_t shard_capacity, off_t start_off, std::vector<JournalOp> &ops);

  void recoverAllShards(size_t shard_capacity);
  void recoverShard(size_t shard_id, size_t shard_capacity);
  void recoverShardToRedisIncremental(size_t shard_id);
  bool shouldFullRecoverRedis(size_t shard_id);

  std::string filename_;
  size_t storage_size_;
  size_t shard_;
  size_t block_size_;

  // Scheme A: data striping configuration and the worker pool issuing concurrent stripe I/O.
  size_t num_stripes_ = 1;                  // Number of stripe sub-files per shard (>=1).
  size_t stripe_size_ = 0;                  // Bytes of one block segment = block_size_ / num_stripes_.
  std::unique_ptr<IoThreadPool> io_pool_;   // Shared across shards; sized at construction.

  std::vector<int> meta_fds_;
  // Per-shard data fds: file_fds_[shard][stripe]. With num_stripes_ == 1 this degenerates to one file.
  std::vector<std::vector<int>> file_fds_;

  std::vector<std::shared_ptr<std::shared_mutex>> io_locks_;
  std::vector<std::shared_ptr<LocalCacheIndex>> caches_;

  // Redis is used for:
  // - global dedupe/locking on the write hot path
  // - publishing shard/global index updates from journal workers
  std::unique_ptr<RedisClient> redis_lock_;
  std::vector<std::unique_ptr<RedisClient>> redis_pool_;
  std::vector<uint64_t> journal_entries_;
  std::vector<uint64_t> superblock_seq_;
  std::vector<uint64_t> superblock_stable_offset_;
  std::vector<uint64_t> superblock_epoch_;
  std::vector<uint64_t> shard_recovered_epoch_;

  // Shard write permission cache (fed by coordinator).
  std::unique_ptr<std::atomic<uint8_t>[]> shard_writable_;
  std::unique_ptr<std::atomic<uint8_t>[]> shard_draining_;
  std::unique_ptr<std::atomic<uint64_t>[]> shard_epoch_cache_;
  std::unique_ptr<InflightCounter[]> shard_inflight_;
  std::unique_ptr<std::atomic<uint64_t>[]> shard_written_bytes_;

  std::vector<std::thread> journal_threads_;
  std::vector<std::unique_ptr<std::mutex>> journal_mu_;
  std::vector<std::unique_ptr<std::condition_variable>> journal_cv_;
  std::vector<std::deque<std::shared_ptr<JournalTask>>> journal_queue_;
  std::vector<bool> journal_stop_;

  // In online mode, shards can be dynamically assigned and hash->shard is not deterministic.
  // In offline mode, deterministic sharding keeps query/write O(1) per hash.
  bool online_mode_ = false;

  // hash -> shard_id (bounded by eviction).
  static constexpr size_t kLocalHintBuckets = 64;
  static_assert((kLocalHintBuckets & (kLocalHintBuckets - 1)) == 0, "kLocalHintBuckets must be power-of-two");

  size_t localHintBucket(const std::string &hash) const {
    return std::hash<std::string>{}(hash) & (kLocalHintBuckets - 1);
  }

  mutable std::array<std::shared_mutex, kLocalHintBuckets> local_hint_mu_;
  std::array<std::unordered_map<std::string, size_t>, kLocalHintBuckets> local_hash_to_shard_;

  RedisClient *redisForShard(size_t shard_id) const {
    if (redis_pool_.empty()) {
      return nullptr;
    }
    return redis_pool_[shard_id % redis_pool_.size()].get();
  }
};

} // namespace storage
} // namespace cache
