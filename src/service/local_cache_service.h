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
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
                    const size_t num_workers)
      : CacheService(kvcache), stop_(false), num_workers_(num_workers), block_size_(0), total_written_bytes_(0),
        first_write_time_ticks_(0), last_write_time_ticks_(0), last_log_time_(), last_logged_bytes_(0),
        total_read_bytes_(0), first_read_time_ticks_(0), last_read_time_ticks_(0), read_last_log_time_(),
        read_last_logged_bytes_(0) {
    block_size_ = static_cast<size_t>(this->block_size());

    if (storage_size < block_size_) {
      throw runtime_error("file size < blocksize is not allowed.");
    }

    ensure_disk_capacity(file, storage_size, num_shard);

    storage_ = make_unique<LocalStorageEngine>(file, storage_size, num_shard, block_size_);

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
    std::vector<bool> ret;
    ret.reserve(hashs.size());
    std::transform(hashs.begin(), hashs.end(), std::back_inserter(ret),
                   [this](const auto &hash) { return storage_->query(hash); });
    return ret;
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
  void on_task_finalized(const std::shared_ptr<cache::task::CacheTask> &task) override {
    if (task->operation_mode == cache::task::Mode::Write) {
      // Try to acquire the lock, skip logging if contention occurs
      std::unique_lock<std::mutex> guard(log_mutex_, std::try_to_lock);
      if (!guard.owns_lock()) {
        return;
      }

      const uint64_t total = total_written_bytes_.load(std::memory_order_relaxed);
      if (total == 0) {
        return;
      }

      const auto now = std::chrono::steady_clock::now();
      if (last_log_time_ != std::chrono::steady_clock::time_point{}) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time_);
        if (elapsed < std::chrono::seconds(3)) {
          return;
        }
      }

      const uint64_t previous_bytes = last_logged_bytes_;
      const uint64_t delta_bytes = (total >= previous_bytes) ? (total - previous_bytes) : 0;
      double speed_gbps = 0.0;

      // Use actual I/O time for speed calculation
      if (last_log_time_ != std::chrono::steady_clock::time_point{}) {
        const double elapsed_sec =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - last_log_time_).count();
        if (elapsed_sec > 0.0) {
          speed_gbps = (static_cast<double>(delta_bytes) / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec;
        }
      } else {
        // For the first log, use time from first write start to now
        const int64_t first_ticks = first_write_time_ticks_.load(std::memory_order_relaxed);
        if (first_ticks != 0) {
          const auto first_duration =
              std::chrono::steady_clock::duration(static_cast<std::chrono::steady_clock::duration::rep>(first_ticks));
          const auto first_time = std::chrono::steady_clock::time_point(first_duration);
          const double elapsed_sec =
              std::chrono::duration_cast<std::chrono::duration<double>>(now - first_time).count();
          if (elapsed_sec > 0.0) {
            speed_gbps = (static_cast<double>(delta_bytes) / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec;
          }
        }
      }

      last_log_time_ = now;
      last_logged_bytes_ = total;

      const double total_gb = static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
      printf("[light_mem] cumulative disk write size: %.2f GB, recent write speed: %.2f GB/s\n", total_gb, speed_gbps);
      return;
    }

    if (task->operation_mode != cache::task::Mode::Read) {
      return;
    }
    if (active_read_creates_.load(std::memory_order_relaxed) != 0) {
      return;
    }

    std::lock_guard<std::mutex> guard(read_log_mutex_);

    const uint64_t total_read = total_read_bytes_.load(std::memory_order_relaxed);
    if (total_read == 0) {
      return;
    }

    // Calculate batch read amount (since last queue empty)
    const uint64_t previous_read = read_last_logged_bytes_;
    const uint64_t delta_read = (total_read >= previous_read) ? (total_read - previous_read) : 0;
    if (delta_read == 0) {
      return;
    }

    // Use actual I/O time: from first read start to last read completion
    const int64_t first_ticks = first_read_time_ticks_.load(std::memory_order_relaxed);
    const int64_t last_ticks = last_read_time_ticks_.load(std::memory_order_relaxed);

    double speed_gbps = 0.0;
    if (first_ticks != 0 && last_ticks != 0 && last_ticks > first_ticks) {
      const auto first_duration =
          std::chrono::steady_clock::duration(static_cast<std::chrono::steady_clock::duration::rep>(first_ticks));
      const auto last_duration =
          std::chrono::steady_clock::duration(static_cast<std::chrono::steady_clock::duration::rep>(last_ticks));
      const auto first_time = std::chrono::steady_clock::time_point(first_duration);
      const auto last_time = std::chrono::steady_clock::time_point(last_duration);
      const double elapsed_sec =
          std::chrono::duration_cast<std::chrono::duration<double>>(last_time - first_time).count();
      if (elapsed_sec > 0.0) {
        speed_gbps = (static_cast<double>(delta_read) / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec;
      }
    }

    if (speed_gbps <= 0.0) {
      return;
    }

    const double delta_read_gb = static_cast<double>(delta_read) / (1024.0 * 1024.0 * 1024.0);

    // Reset counters for next batch after queue is empty
    read_last_log_time_ = std::chrono::steady_clock::now();
    read_last_logged_bytes_ = 0;                                // Reset to 0 instead of total_read
    total_read_bytes_.store(0, std::memory_order_relaxed);      // Clear accumulated bytes
    first_read_time_ticks_.store(0, std::memory_order_relaxed); // Reset timing
    last_read_time_ticks_.store(0, std::memory_order_relaxed);  // Reset timing
    printf("[light_mem] batch read size: %.2f GB, read speed: %.2f GB/s\n", delta_read_gb, speed_gbps);
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
          processTask(block, cpu_buffer);
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

    const int64_t page_size = this->cache_info_.page_size * this->cache_info_.num_of_layer;
    const int64_t page_per_block = (static_cast<int64_t>(block_size_) + page_size - 1) / page_size;
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

    const size_t read_bytes = storage_->read(cpu_buffer, block->hash);
    if (read_bytes != block_size_) {
      // Only log if it's a real I/O error (partial read), not cache miss (read_bytes == 0)
      if (read_bytes != 0) {
        fprintf(stderr,
                "[light_mem error] handleReadCpu: partial read for hash %s, expected %zu bytes, got %zu bytes\n",
                block->hash.c_str(), block_size_, read_bytes);
      }
      return false;
    }

    {
      auto task = block->get_task();
      if (!task) {
        fprintf(stderr, "[light_mem warning] handleReadCpu: task has been destroyed for block with hash %s\n",
                block->hash.c_str());
        return false;
      }
      std::lock_guard<std::mutex> lock(task->state_mutex);
      if (block->state != cache::task::State::Working) {
        fprintf(stderr, "[light_mem warning] handleReadCpu: block state is not Working for hash %s, state=%d\n",
                block->hash.c_str(), static_cast<int>(block->state));
        return false;
      }
    }

    total_read_bytes_.fetch_add(static_cast<uint64_t>(read_bytes), std::memory_order_relaxed);

    cpu_scatter(this->cache_info_, cpu_buffer, page_ptr, num_of_page);

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

    {
      std::lock_guard<std::mutex> lock(task->state_mutex);
      if (block->state != cache::task::State::Working) {
        fprintf(stderr, "[light_mem warning] handleWriteCpu: block state is not Working for hash %s, state=%d\n",
                block->hash.c_str(), static_cast<int>(block->state));
        return false;
      }
    }

    // Record start time before the first write operation begins
    if (first_write_time_ticks_.load(std::memory_order_relaxed) == 0) {
      const auto now_duration = std::chrono::steady_clock::now().time_since_epoch();
      const int64_t now_ticks = static_cast<int64_t>(now_duration.count());
      int64_t expected = 0;
      first_write_time_ticks_.compare_exchange_strong(expected, now_ticks, std::memory_order_relaxed,
                                                      std::memory_order_relaxed);
    }

    // Step 1: Gather data from KV cache to temporary buffer
    cpu_gather(this->cache_info_, cpu_buffer, page_ptr, num_of_page);

    // Critical optimization: Mark data as ready immediately after gather completes
    // This allows Python layer to release pages without waiting for disk I/O
    task->num_data_ready_blocks.fetch_add(1, std::memory_order_release);

    // Step 2: Write to disk (this happens asynchronously and doesn't block page release)
    const size_t written = storage_->write(cpu_buffer, block->hash);
    if (written != block_size_ && written != 0) {
      fprintf(stderr,
              "[light_mem error] handleWriteCpu: write failed for hash %s, expected %zu or 0 bytes, got %zu bytes\n",
              block->hash.c_str(), block_size_, written);
      return false;
    }
    if (written == block_size_) {
      total_written_bytes_.fetch_add(static_cast<uint64_t>(written), std::memory_order_relaxed);
    }

    // Record end time after write completes
    const auto now_duration = std::chrono::steady_clock::now().time_since_epoch();
    const int64_t now_ticks = static_cast<int64_t>(now_duration.count());
    last_write_time_ticks_.store(now_ticks, std::memory_order_relaxed);

    return true;
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
    const int64_t num_of_layer = info.num_of_layer;
    const int64_t page_stride = info.page_stride;
    const int64_t total_pages = info.num_of_page;
    const int64_t page_bytes = page_size * num_of_layer;

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
    const int64_t num_of_layer = info.num_of_layer;
    const int64_t page_stride = info.page_stride;
    const int64_t total_pages = info.num_of_page;
    const int64_t page_bytes = page_size * num_of_layer;

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

  size_t block_size_;                                   ///< Block size
  unique_ptr<LocalStorageEngine> storage_;              ///< Local storage engine
  vector<thread> workers_;                              ///< Worker threads
  bool stop_;                                           ///< Thread stop flag
  size_t num_workers_;                                  ///< Number of worker threads
  vector<unique_ptr<char[]>> r_cpu_buffers_;            ///< CPU buffers for read worker (RAII managed)
  vector<unique_ptr<char[]>> w_cpu_buffers_;            ///< CPU buffers for write worker (RAII managed)
  std::atomic<uint64_t> total_written_bytes_;           ///< Total bytes written to disk
  std::atomic<int64_t> first_write_time_ticks_;         ///< First write start time in steady clock ticks
  std::atomic<int64_t> last_write_time_ticks_;          ///< Last write completion time in steady clock ticks
  std::mutex log_mutex_;                                ///< Protects write rate reporting
  std::chrono::steady_clock::time_point last_log_time_; ///< Last write log timestamp
  uint64_t last_logged_bytes_;                          ///< Bytes recorded at last write log

  std::atomic<uint64_t> total_read_bytes_;                   ///< Total bytes read from disk
  std::atomic<int64_t> first_read_time_ticks_;               ///< First read start time in steady clock ticks
  std::atomic<int64_t> last_read_time_ticks_;                ///< Last read completion time in steady clock ticks
  std::mutex read_log_mutex_;                                ///< Protects read rate reporting
  std::chrono::steady_clock::time_point read_last_log_time_; ///< Last read log timestamp
  uint64_t read_last_logged_bytes_;                          ///< Bytes recorded at last read log

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
