#pragma once

#include "config.h"

#include "core/cache_task.h"
#include "core/error.h"
#include "core/task_queue.h"
#include "service/cache_service.h"
#include "storage/local_storage_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <zlib.h> // crc32

using namespace std;
using namespace cache::task;
using namespace cache::queue;
using namespace cache::storage;

namespace cache::service {

/**
 * @brief LocalCacheService is a local asynchronous caching system.
 *
 * This system is used for asynchronously storing and accessing large model KV caches,
 * supporting storage on local disks or distributed storage systems.
 */
class LocalCacheService : public CacheService {
public:
  /**
   * @brief Constructor
   * @param file Path to the local storage file
   * @param storage_size Size of the storage
   * @param num_shard Num of storage shard
   * @param kvcache KV cache tensor (CPU)
   * @param num_workers Number of worker threads
   */
  LocalCacheService(const string &file, size_t storage_size, size_t num_shard, const torch::Tensor &kvcache,
                    const size_t num_workers, const std::string &index_endpoint = "", bool bandwidth_log = true,
                    const std::string &index_prefix = "")
      : CacheService(kvcache), stop_(false), num_workers_(num_workers), block_size_(0), bandwidth_log_(bandwidth_log),
        online_mode_(!index_endpoint.empty()) {
    block_size_ = static_cast<size_t>(this->block_size());

    if (storage_size < block_size_) {
      throw runtime_error("file size < blocksize is not allowed.");
    }

    ensure_disk_capacity(file, storage_size, num_shard);

    storage_ = make_unique<LocalStorageEngine>(file, storage_size, num_shard, block_size_, index_endpoint, index_prefix);

    // Use unique_ptr for exception safety - if any allocation fails, previous allocations are automatically cleaned up
    r_cpu_buffers_.reserve(num_workers_);
    w_cpu_buffers_.reserve(num_workers_);

    for (size_t i = 0; i < num_workers_; ++i) {
      r_cpu_buffers_.emplace_back(std::make_unique<char[]>(block_size_));
      w_cpu_buffers_.emplace_back(std::make_unique<char[]>(block_size_));
    }
  }

  /**
   * @brief Destructor, stops all worker threads and releases resources.
   */
  ~LocalCacheService() {
    stop_ = true;
    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  /**
   * The query function querys a given hash keys whether exist in current cache system.
   * A bool array will be returned as the query result,
   * where true means hash existence and false means not
   *
   * This function will throw no exception or error.
   */
  std::vector<bool> query(const std::vector<std::string> &hashs) override {
    if (!storage_) {
      return std::vector<bool>(hashs.size(), false);
    }
    return storage_->queryMany(hashs);
  }

  /**
   * @brief Update this node's shard assignment state in online/distributed mode.
   *
   * @param shard_ids Shard IDs owned/managed by this node.
   * @param epochs    Per-shard epoch (generation). Must align 1:1 with shard_ids.
   * @param draining  Per-shard draining flag (0/1). If 1, shard is treated as not writable (no new writes).
   *
   * Requirements: shard_ids/epochs/draining must have the same length.
   */
  void update_shard_assignments(const std::vector<size_t> &shard_ids, const std::vector<uint64_t> &epochs,
                                const std::vector<uint8_t> &draining) {
    if (!storage_) {
      return;
    }
    storage_->updateShardAssignments(shard_ids, epochs, draining);
  }

  /**
   * @brief Rebuild Redis index for a shard from local snapshot + WAL.
   *
   * Intended to be called when a node (re-)acquires write permission for a shard.
   */
  void recover_shard_to_redis(size_t shard_id) {
    if (!storage_) {
      return;
    }
    storage_->recoverShardToRedis(shard_id);
  }

  /**
   * @brief Recover Redis index for a shard using smart policy (full vs incremental).
   */
  void recover_shard_to_redis_smart(size_t shard_id) {
    if (!storage_) {
      return;
    }
    storage_->recoverShardToRedisSmart(shard_id);
  }

  /**
   * @brief Observability: current in-flight write operations for a shard.
   *
   * This is a best-effort counter used by the coordinator/control-plane to decide when a shard is drained.
   */
  uint32_t shard_inflight(size_t shard_id) const {
    if (!storage_) {
      return 0;
    }
    return storage_->shardInflight(shard_id);
  }

  uint64_t shard_written_bytes(size_t shard_id) const {
    if (!storage_) {
      return 0;
    }
    return storage_->shardWrittenBytes(shard_id);
  }

  // True LRU eviction observability (local to this node).
  uint64_t shard_eviction_count(size_t shard_id) const {
    if (!storage_) {
      return 0;
    }
    return storage_->shardEvictionCount(shard_id);
  }

  uint64_t eviction_count() const {
    if (!storage_) {
      return 0;
    }
    return storage_->evictionCount();
  }

  bool eviction_observed() const {
    if (!storage_) {
      return false;
    }
    return storage_->evictionObserved();
  }

  /**
   * @brief Runs worker threads.
   */
  void run() {
    stop_ = false;
    for (size_t i = 0; i < num_workers_; ++i) {
      workers_.emplace_back(&LocalCacheService::work, this, static_cast<int32_t>(i));
    }
  }

protected:
  bool online_mode() const override { return online_mode_; }

  void on_task_finalized(const std::shared_ptr<cache::task::CacheTask> &task) override {
    if (!bandwidth_log_) {
      return;
    }

    if (task->operation_mode == cache::task::Mode::Write) {
      window_write_block_count_.fetch_add(static_cast<uint64_t>(task->blocks.size()), std::memory_order_relaxed);
      write_finalize_in_progress_.fetch_add(1, std::memory_order_relaxed);

      const bool queue_drained = (active_write_creates_.load(std::memory_order_relaxed) == 0);
      const uint64_t remaining_finalize_hooks =
          write_finalize_in_progress_.fetch_sub(1, std::memory_order_acq_rel) - 1;
      if (!queue_drained || remaining_finalize_hooks != 0) {
        return;
      }

      std::lock_guard<std::mutex> guard(log_mutex_);
      if (active_write_creates_.load(std::memory_order_relaxed) != 0 ||
          write_finalize_in_progress_.load(std::memory_order_relaxed) != 0) {
        return;
      }

      const uint64_t total = storage_->writtenBytes();
      const uint64_t previous_bytes = last_logged_bytes_;
      const uint64_t current_bytes = (total >= previous_bytes) ? (total - previous_bytes) : 0;
      const uint64_t current_write_block_num = window_write_block_count_.exchange(0, std::memory_order_relaxed);
      double speed_gbps = 0.0;

      const int64_t first_ticks = first_write_time_ticks_.exchange(0, std::memory_order_relaxed);
      const int64_t last_ticks = last_write_time_ticks_.exchange(0, std::memory_order_relaxed);
      if (first_ticks != 0 && last_ticks > first_ticks) {
        const double elapsed_sec = static_cast<double>(last_ticks - first_ticks) /
                                   static_cast<double>(std::chrono::steady_clock::duration::period::den);
        if (elapsed_sec > 0.0) {
          speed_gbps = (static_cast<double>(current_bytes) / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec;
        }
      }

      last_logged_bytes_ = total;

      const double total_gb = static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
      const double current_gb = static_cast<double>(current_bytes) / (1024.0 * 1024.0 * 1024.0);

      const size_t owned_shards = storage_->ownedShardCount();
      const uint64_t effective_bytes = storage_->effectiveWritableCapacityBytes();
      const double effective_gb = static_cast<double>(effective_bytes) / (1024.0 * 1024.0 * 1024.0);
      const bool evict = storage_->evictionObserved();

      std::fprintf(stderr,
                   "[light_mem] cumulative disk write size: %.2f GB, current disk write size: %.2f GB, "
                   "recent write speed: %.2f GB/s, owned_shards: %zu, "
                   "effective_write_capacity: %.2f GB, evict: %s\n",
                   total_gb, current_gb, speed_gbps, owned_shards, effective_gb,
                   evict ? "True" : "False");
      std::fflush(stderr);
      return;
    }

    if (task->operation_mode != cache::task::Mode::Read) {
      return;
    }

    // Track end-to-end read window (create() -> ready()) using min(start) and max(end).
    // This avoids summing per-task durations (which double-counts time under concurrency).
    const int64_t start_ticks = task->submit_time_ticks.load(std::memory_order_relaxed);
    const int64_t end_ticks = task->finish_time_ticks.load(std::memory_order_relaxed);
    if (start_ticks != 0 && end_ticks != 0 && end_ticks > start_ticks) {
      // window_read_start_ticks_ = min(window_read_start_ticks_, start_ticks)
      int64_t cur = window_read_start_ticks_.load(std::memory_order_relaxed);
      while (cur == 0 || start_ticks < cur) {
        if (window_read_start_ticks_.compare_exchange_weak(cur, start_ticks, std::memory_order_relaxed,
                                                          std::memory_order_relaxed)) {
          break;
        }
      }

      // window_read_end_ticks_ = max(window_read_end_ticks_, end_ticks)
      cur = window_read_end_ticks_.load(std::memory_order_relaxed);
      while (end_ticks > cur) {
        if (window_read_end_ticks_.compare_exchange_weak(cur, end_ticks, std::memory_order_relaxed,
                                                        std::memory_order_relaxed)) {
          break;
        }
      }
    }

    if (active_read_creates_.load(std::memory_order_relaxed) != 0) {
      return;
    }

    std::lock_guard<std::mutex> guard(read_log_mutex_);

    if (active_read_creates_.load(std::memory_order_relaxed) != 0) {
      return;
    }

    const uint64_t window_bytes = total_read_bytes_.exchange(0, std::memory_order_relaxed);
    const int64_t window_start = window_read_start_ticks_.exchange(0, std::memory_order_relaxed);
    const int64_t window_end = window_read_end_ticks_.exchange(0, std::memory_order_relaxed);
    if (window_bytes == 0 || window_start == 0 || window_end == 0 || window_end <= window_start) {
      return;
    }

    const int64_t elapsed_ticks = window_end - window_start;
    const auto elapsed_dur =
        std::chrono::steady_clock::duration(static_cast<std::chrono::steady_clock::duration::rep>(elapsed_ticks));
    const double elapsed_sec = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_dur).count();
    if (elapsed_sec <= 0.0) {
      return;
    }

    const double window_gb = static_cast<double>(window_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double speed_gbps = window_gb / elapsed_sec;
    if (speed_gbps <= 0.0) {
      return;
    }

    std::fprintf(stderr, "[light_mem] batch read size: %.2f GB, read speed: %.2f GB/s\n", window_gb, speed_gbps);
    std::fflush(stderr);
  }

private:
  /**
   * @brief Core logic of the worker thread.
   * @param index Thread index
   */
  void work(int32_t index) {
    while (!stop_) {
      if (auto block = this->queue_->claim()) {
        if (block != nullptr) {
          auto task = block->get_task();
          if (!task) {
            fprintf(stderr, "[light_mem warning] worker %d: task has been destroyed for block with hash %s\n", index,
                    block->hash.c_str());
            continue; // Task has been destroyed
          }
          char *cpu_buffer =
              (task->operation_mode == Mode::Read) ? r_cpu_buffers_[index].get() : w_cpu_buffers_[index].get();
          try {
            processTask(block, cpu_buffer);
          } catch (const std::exception &e) {
            fprintf(stderr,
                    "[light_mem error] worker %d: exception while processing block (hash=%s): %s; aborting block\n",
                    index, block->hash.c_str(), e.what());
            this->abort(block);
          } catch (...) {
            fprintf(stderr,
                    "[light_mem error] worker %d: unknown exception while processing block (hash=%s); aborting block\n",
                    index, block->hash.c_str());
            this->abort(block);
          }
        }
      }
    }
  }

  void processTask(CacheBlock *block, char *cpu_buffer) {
    auto task = block->get_task();
    if (!task) {
      fprintf(stderr, "[light_mem warning] processTask: task has been destroyed for block with hash %s\n",
              block->hash.c_str());
      return; // Task has been destroyed
    }
    torch::Tensor page_tensor = task->page_indexer;
    auto bid = block->block_idx;

    const int64_t page_size = this->cache_info_.page_size;
    const int64_t page_per_block = static_cast<int64_t>(block_size_) / page_size;
    const int64_t remaining_pages = page_tensor.numel() - bid * page_per_block;
    if (remaining_pages <= 0) {
      fprintf(stderr,
              "[light_mem error] processTask: remaining_pages=%lld <= 0 for block %lld (hash=%s), "
              "page_tensor.numel()=%lld, page_per_block=%lld\n",
              static_cast<long long>(remaining_pages), static_cast<long long>(bid), block->hash.c_str(),
              static_cast<long long>(page_tensor.numel()), static_cast<long long>(page_per_block));
      this->abort(block);
      return;
    }
    const int64_t num_of_page = std::min(remaining_pages, page_per_block);
    int32_t *page_ptr = reinterpret_cast<int32_t *>(page_tensor.data_ptr()) + bid * page_per_block;

    if (!page_tensor.device().is_cpu()) {
      throw std::runtime_error("kv_page_indexer tensor must reside on CPU for CPU cache service.");
    }

    bool success = false;
    if (task->operation_mode == Mode::Read) {
      success = handleReadCpu(block, cpu_buffer, page_ptr, num_of_page);
    } else {
      success = handleWriteCpu(block, cpu_buffer, page_ptr, num_of_page);
    }

    success ? this->deliver(block) : this->abort(block);
  }

  bool handleReadCpu(CacheBlock *block, char *cpu_buffer, int32_t *page_ptr, int64_t num_of_page) {
    // Record start time before the first read operation begins
    if (first_read_time_ticks_.load(std::memory_order_relaxed) == 0) {
      const auto now_duration = std::chrono::steady_clock::now().time_since_epoch();
      const int64_t now_ticks = static_cast<int64_t>(now_duration.count());
      int64_t expected = 0;
      first_read_time_ticks_.compare_exchange_strong(expected, now_ticks, std::memory_order_relaxed,
                                                     std::memory_order_relaxed);
    }

    const uint32_t logical_bytes = static_cast<uint32_t>(num_of_page * this->cache_info_.page_size);

    // Zero-copy fast path: when the destination pages are contiguous in the kvcache tensor,
    // read straight into the tensor and skip both the bounce buffer and the cpu_scatter memcpy.
    char *const direct_dst = contiguousDstPtr(this->cache_info_, page_ptr, num_of_page);
    char *const read_dst = direct_dst ? direct_dst : cpu_buffer;

    const size_t read_bytes = storage_->read(read_dst, block->hash, logical_bytes);
    if (read_bytes != static_cast<size_t>(logical_bytes)) {
      // Only log if it's a real I/O error (partial read), not cache miss (read_bytes == 0)
      if (read_bytes != 0) {
        fprintf(stderr,
                "[light_mem error] handleReadCpu: partial read for hash %s, expected %u bytes, got %zu bytes\n",
                block->hash.c_str(), logical_bytes, read_bytes);
      }
      return false;
    }

    total_read_bytes_.fetch_add(static_cast<uint64_t>(read_bytes), std::memory_order_relaxed);

    if (!direct_dst) {
      // Slow path (non-contiguous destination pages): scatter from the bounce buffer.
      cpu_scatter(this->cache_info_, cpu_buffer, page_ptr, num_of_page);
    }

    // Record end time after scatter completes
    const auto now_duration = std::chrono::steady_clock::now().time_since_epoch();
    const int64_t now_ticks = static_cast<int64_t>(now_duration.count());
    last_read_time_ticks_.store(now_ticks, std::memory_order_relaxed);

    return true;
  }

  bool handleWriteCpu(CacheBlock *block, char *cpu_buffer, int32_t *page_ptr, int64_t num_of_page) {
    auto task = block->get_task();
    if (!task) {
      fprintf(stderr, "[light_mem warning] handleWriteCpu: task has been destroyed for block with hash %s\n",
              block->hash.c_str());
      return false; // Task has been destroyed
    }

    // Record start time before the first write operation begins
    if (first_write_time_ticks_.load(std::memory_order_relaxed) == 0) {
      const auto now_duration = std::chrono::steady_clock::now().time_since_epoch();
      const int64_t now_ticks = static_cast<int64_t>(now_duration.count());
      int64_t expected = 0;
      first_write_time_ticks_.compare_exchange_strong(expected, now_ticks, std::memory_order_relaxed,
                                                      std::memory_order_relaxed);
    }

    const uint32_t logical_bytes = static_cast<uint32_t>(num_of_page * this->cache_info_.page_size);

    // Step 1: Gather data from KV cache to temporary buffer.
    // Online mode: fuse CRC with memcpy (cache-friendly) so the storage layer doesn't
    //   need a separate full-block CRC pass.
    // Offline mode: CRC is never used (mark_ready passes 0), so skip it entirely
    //   to avoid wasting memory bandwidth on a ~1 MB CRC that gets discarded.
    uint32_t data_crc = 0;
    if (online_mode_) {
      data_crc = cpu_gather_crc32(this->cache_info_, cpu_buffer, page_ptr, num_of_page);
    } else {
      cpu_gather(this->cache_info_, cpu_buffer, page_ptr, num_of_page);
    }

    // Critical optimization: Mark data as ready immediately after gather completes
    // This allows Python layer to release pages without waiting for disk I/O.
    // Guard with state_mutex + per-block flag so that this increment and a possible
    // abort() of the same block are mutually exclusive and never double-count.
    {
      std::lock_guard<std::mutex> lock(task->state_mutex);
      if (!block->write_data_ready) {
        block->write_data_ready = true;
        task->num_data_ready_blocks.fetch_add(1, std::memory_order_release);
      }
    }

    // Step 2: Write to disk (this happens asynchronously and doesn't block page release)
    // Always use the length-aware write path to skip redundant CRC recomputation.
    const size_t written = storage_->write(cpu_buffer, block->hash, data_crc, logical_bytes);

    // Handle different write results:
    // - written > 0: Success.
    // - written == 0: Skipped (already exists, failed, or temporary congestion)
    if (written != 0 && written != static_cast<size_t>(logical_bytes) && written != block_size_) {
      fprintf(stderr,
              "[light_mem error] handleWriteCpu: unexpected write size for hash %s, expected %u or %zu or 0 bytes, got %zu bytes\n",
              block->hash.c_str(), logical_bytes, block_size_, written);
      return false;
    }

    // If written == 0:
    // - Offline mode: treat as failure; local duplicates return block_size_ from storage.
    // - Online mode: tolerate short transient ownership migration windows with retries.
    size_t final_written = written;
    if (final_written == 0) {
      if (!online_mode()) {
        return false;
      }

      bool readable = false;
      constexpr int kRetry = 3;
      for (int attempt = 0; attempt <= kRetry; ++attempt) {
        // Verify existence on slow-path: dedupe hit or already-published data should be readable.
        const auto exists = storage_->queryMany(std::vector<std::string>{block->hash});
        if (!exists.empty() && exists[0]) {
          readable = true;
          break;
        }

        if (attempt == kRetry) {
          break;
        }

        // Retry write for transient ownership handoff/draining windows.
        std::this_thread::yield();
        final_written = storage_->write(cpu_buffer, block->hash, data_crc, logical_bytes);

        if (final_written != 0 && final_written != static_cast<size_t>(logical_bytes) && final_written != block_size_) {
          fprintf(stderr,
                  "[light_mem error] handleWriteCpu: unexpected retry write size for hash %s, expected %u or %zu or 0 bytes, got %zu bytes\n",
                  block->hash.c_str(), logical_bytes, block_size_, final_written);
          return false;
        }

        if (final_written > 0) {
          break;
        }
      }

      if (final_written == 0 && !readable) {
        std::fprintf(
            stderr,
            "[light_mem warning] handleWriteCpu: transient write miss (possibly shard migration), hash not readable after retries: %s\n",
            block->hash.c_str());
        return false;
      }
    }
    // Record end time after write completes
    const auto now_duration = std::chrono::steady_clock::now().time_since_epoch();
    const int64_t now_ticks = static_cast<int64_t>(now_duration.count());
    last_write_time_ticks_.store(now_ticks, std::memory_order_relaxed);

    return true;  // Success or acceptable skip
  }

  /**
   * @brief Return the contiguous destination pointer in the kvcache tensor, or nullptr.
   *
   * If the destination pages occupy a contiguous, in-range span of the kvcache tensor
   * (page_stride == page_size and indices form an ascending run), a disk read can land
   * directly into the tensor, avoiding the bounce buffer + scatter memcpy entirely.
   * Returns nullptr when the fast path does not apply (caller must use the slow path).
   */
  static char *contiguousDstPtr(const CacheParam_t &info, const int32_t *page_idx, int64_t num_of_page) {
    if (num_of_page <= 0 || info.page_stride != info.page_size) {
      return nullptr;
    }
    const int32_t first = page_idx[0];
    if (first < 0 || first >= info.num_of_page) {
      return nullptr;
    }
    for (int64_t local_page = 1; local_page < num_of_page; ++local_page) {
      if (page_idx[local_page] != first + static_cast<int32_t>(local_page)) {
        return nullptr;
      }
    }
    const int64_t last = static_cast<int64_t>(first) + num_of_page - 1;
    if (last >= info.num_of_page) {
      return nullptr;
    }
    return info.base_ptr + static_cast<int64_t>(first) * info.page_stride;
  }

  /**
   * @brief Scatter data from a continuous block buffer to KV cache pages in memory.
   *
   * This function is used during read operations to distribute data read from disk
   * into the appropriate page locations in the KV cache tensor. Each page in the
   * block buffer is copied to its corresponding destination page in the cache.
   *
   * @param info KV cache configuration containing base pointer, page sizes and strides
   * @param block Source buffer containing continuous block data read from disk
   * @param page_idx Array of destination page indices in the KV cache
   * @param num_of_page Number of pages to scatter from the block
   *
   * @throws std::runtime_error if any page index is out of valid range
   */
  static void cpu_scatter(const CacheParam_t &info, const char *block, const int32_t *page_idx, int64_t num_of_page) {
    const int64_t page_size = info.page_size;
    const int64_t page_stride = info.page_stride;
    const int64_t total_pages = info.num_of_page;
    const int64_t page_bytes = page_size;

    // Fast path: if destination pages are contiguous in memory and indices form a contiguous range,
    // we can copy the entire span in one memcpy.
    if (num_of_page > 0 && page_stride == page_bytes) {
      const int32_t first = page_idx[0];
      if (first < 0 || first >= total_pages) {
        throw std::runtime_error("kv page index out of range in cpu_scatter.");
      }
      bool contiguous = true;
      for (int64_t local_page = 1; local_page < num_of_page; ++local_page) {
        const int32_t expected = first + static_cast<int32_t>(local_page);
        if (page_idx[local_page] != expected) {
          contiguous = false;
          break;
        }
      }
      if (contiguous) {
        const int32_t last = first + static_cast<int32_t>(num_of_page - 1);
        if (last < 0 || last >= total_pages) {
          throw std::runtime_error("kv page index out of range in cpu_scatter.");
        }
        char *dst_ptr = info.base_ptr + static_cast<int64_t>(first) * page_stride;
        std::memcpy(dst_ptr, block, static_cast<size_t>(num_of_page) * static_cast<size_t>(page_bytes));
        return;
      }
    }

    for (int64_t local_page = 0; local_page < num_of_page; ++local_page) {
      const int32_t dst_page = page_idx[local_page];
      if (dst_page < 0 || dst_page >= total_pages) {
        throw std::runtime_error("kv page index out of range in cpu_scatter.");
      }

      char *dst_page_ptr = info.base_ptr + static_cast<int64_t>(dst_page) * page_stride;
      const char *src_page_ptr = block + local_page * page_bytes;

      std::memcpy(dst_page_ptr, src_page_ptr, page_bytes);
    }
  }

  /**
   * @brief Gather data from KV cache pages in memory to a continuous block buffer.
   *
   * This function is used during write operations to collect data from scattered
   * page locations in the KV cache tensor into a continuous buffer for disk writing.
   * Each source page in the cache is copied to its corresponding position in the block.
   *
   * @param info KV cache configuration containing base pointer, page sizes and strides
   * @param block Destination buffer to store the gathered continuous block data
   * @param page_idx Array of source page indices in the KV cache
   * @param num_of_page Number of pages to gather into the block
   *
   * @throws std::runtime_error if any page index is out of valid range
   */
  static void cpu_gather(const CacheParam_t &info, char *block, const int32_t *page_idx, int64_t num_of_page) {
    const int64_t page_size = info.page_size;
    const int64_t page_stride = info.page_stride;
    const int64_t total_pages = info.num_of_page;
    const int64_t page_bytes = page_size;

    // Fast path: if source pages are contiguous in memory and indices form a contiguous range,
    // we can gather the entire span in one memcpy.
    if (num_of_page > 0 && page_stride == page_bytes) {
      const int32_t first = page_idx[0];
      if (first < 0 || first >= total_pages) {
        throw std::runtime_error("kv page index out of range in cpu_gather.");
      }
      bool contiguous = true;
      for (int64_t local_page = 1; local_page < num_of_page; ++local_page) {
        const int32_t expected = first + static_cast<int32_t>(local_page);
        if (page_idx[local_page] != expected) {
          contiguous = false;
          break;
        }
      }
      if (contiguous) {
        const int32_t last = first + static_cast<int32_t>(num_of_page - 1);
        if (last < 0 || last >= total_pages) {
          throw std::runtime_error("kv page index out of range in cpu_gather.");
        }
        const char *src_ptr = info.base_ptr + static_cast<int64_t>(first) * page_stride;
        std::memcpy(block, src_ptr, static_cast<size_t>(num_of_page) * static_cast<size_t>(page_bytes));
        return;
      }
    }

    for (int64_t local_page = 0; local_page < num_of_page; ++local_page) {
      const int32_t src_page = page_idx[local_page];
      if (src_page < 0 || src_page >= total_pages) {
        throw std::runtime_error("kv page index out of range in cpu_gather.");
      }

      const char *src_page_ptr = info.base_ptr + static_cast<int64_t>(src_page) * page_stride;
      char *dst_page_ptr = block + local_page * page_bytes;

      std::memcpy(dst_page_ptr, src_page_ptr, page_bytes);
    }
  }

  // Gather + CRC in one pass (CRC over the logical bytes only: num_of_page * page_size).
  static uint32_t cpu_gather_crc32(const CacheParam_t &info, char *block, const int32_t *page_idx,
                                  int64_t num_of_page) {
    const int64_t page_size = info.page_size;
    const int64_t page_stride = info.page_stride;
    const int64_t total_pages = info.num_of_page;
    const int64_t page_bytes = page_size;

    uLong crc = ::crc32(0, Z_NULL, 0);

    // Fast path: if source pages are contiguous in memory and indices form a contiguous range.
    if (num_of_page > 0 && page_stride == page_bytes) {
      const int32_t first = page_idx[0];
      if (first < 0 || first >= total_pages) {
        throw std::runtime_error("kv page index out of range in cpu_gather_crc32.");
      }
      bool contiguous = true;
      for (int64_t local_page = 1; local_page < num_of_page; ++local_page) {
        const int32_t expected = first + static_cast<int32_t>(local_page);
        if (page_idx[local_page] != expected) {
          contiguous = false;
          break;
        }
      }
      if (contiguous) {
        const int32_t last = first + static_cast<int32_t>(num_of_page - 1);
        if (last < 0 || last >= total_pages) {
          throw std::runtime_error("kv page index out of range in cpu_gather_crc32.");
        }
        const char *src_ptr = info.base_ptr + static_cast<int64_t>(first) * page_stride;
        const size_t nbytes = static_cast<size_t>(num_of_page) * static_cast<size_t>(page_bytes);
        std::memcpy(block, src_ptr, nbytes);
        crc = ::crc32(crc, reinterpret_cast<const Bytef *>(src_ptr), nbytes);
        return static_cast<uint32_t>(crc);
      }
    }

    for (int64_t local_page = 0; local_page < num_of_page; ++local_page) {
      const int32_t src_page = page_idx[local_page];
      if (src_page < 0 || src_page >= total_pages) {
        throw std::runtime_error("kv page index out of range in cpu_gather_crc32.");
      }

      const char *src_page_ptr = info.base_ptr + static_cast<int64_t>(src_page) * page_stride;
      char *dst_page_ptr = block + local_page * page_bytes;
      std::memcpy(dst_page_ptr, src_page_ptr, page_bytes);
      crc = ::crc32(crc, reinterpret_cast<const Bytef *>(src_page_ptr), static_cast<size_t>(page_bytes));
    }
    return static_cast<uint32_t>(crc);
  }

  size_t block_size_;                        ///< Block size
  unique_ptr<LocalStorageEngine> storage_;   ///< Local storage engine
  vector<thread> workers_;                   ///< Worker threads
  bool stop_;                                ///< Thread stop flag
  size_t num_workers_;                       ///< Number of worker threads
  vector<unique_ptr<char[]>> r_cpu_buffers_; ///< CPU buffers for read worker (RAII managed)
  vector<unique_ptr<char[]>> w_cpu_buffers_; ///< CPU buffers for write worker (RAII managed)

  bool bandwidth_log_{true};
  bool online_mode_{false};

  std::atomic<uint64_t> total_written_bytes_{0};
  std::atomic<uint64_t> window_write_block_count_{0};
  std::atomic<uint64_t> write_finalize_in_progress_{0};
  std::atomic<uint64_t> window_logical_written_bytes_{0};
  std::atomic<int64_t> window_logical_write_time_ticks_{0};
  std::atomic<int64_t> first_write_time_ticks_{0};
  std::atomic<int64_t> last_write_time_ticks_{0};
  std::mutex log_mutex_;
  std::chrono::steady_clock::time_point last_log_time_{};
  uint64_t last_logged_bytes_{0};

  std::atomic<uint64_t> total_read_bytes_{0};
  std::atomic<uint64_t> window_logical_read_bytes_{0};
  std::atomic<int64_t> window_logical_read_time_ticks_{0};
  std::atomic<int64_t> window_read_start_ticks_{0};
  std::atomic<int64_t> window_read_end_ticks_{0};
  std::atomic<int64_t> first_read_time_ticks_{0};
  std::atomic<int64_t> last_read_time_ticks_{0};
  std::mutex read_log_mutex_;
  std::chrono::steady_clock::time_point read_last_log_time_{};
  uint64_t read_last_logged_bytes_{0};

  // Ensure the backing storage path exposes enough disk capacity for the requested cache size.
  static void ensure_disk_capacity(const string &file, size_t storage_size, size_t num_shard) {
    namespace fs = std::filesystem;

    if (num_shard == 0) {
      throw std::runtime_error("num_shard must be greater than zero");
    }

    fs::path base_path(file);
    fs::path target_dir = base_path.parent_path();
    if (target_dir.empty()) {
      target_dir = fs::current_path();
    }

    std::error_code ec;
    fs::path probe_dir = target_dir;
    while (!probe_dir.empty() && !fs::exists(probe_dir, ec)) {
      ec.clear();
      probe_dir = probe_dir.parent_path();
    }

    if (probe_dir.empty()) {
      probe_dir = fs::current_path();
    }

    ec.clear();
    fs::space_info info = fs::space(probe_dir, ec);
    if (ec) {
      throw std::runtime_error("Failed to query available space for path: " + probe_dir.string() +
                               ", reason: " + ec.message());
    }

    uintmax_t reclaimable = 0;
    for (size_t i = 0; i < num_shard; ++i) {
      fs::path shard_path = base_path;
      shard_path += "_" + std::to_string(i);

      std::error_code exists_ec;
      if (!fs::exists(shard_path, exists_ec) || exists_ec) {
        continue;
      }

      std::error_code size_ec;
      const uintmax_t shard_size = fs::file_size(shard_path, size_ec);
      if (size_ec) {
        continue;
      }

      if (std::numeric_limits<uintmax_t>::max() - reclaimable < shard_size) {
        reclaimable = std::numeric_limits<uintmax_t>::max();
        break;
      }
      reclaimable += shard_size;
    }

    uintmax_t total_available = info.available;
    if (std::numeric_limits<uintmax_t>::max() - total_available < reclaimable) {
      total_available = std::numeric_limits<uintmax_t>::max();
    } else {
      total_available += reclaimable;
    }

    const uintmax_t required = static_cast<uintmax_t>(storage_size);
    if (total_available < required) {
      throw std::runtime_error("Insufficient disk space for local cache service. Required " + std::to_string(required) +
                               " bytes but only " + std::to_string(total_available) +
                               " bytes available including reclaimable shards.");
    }
  }
};

} // namespace cache::service
