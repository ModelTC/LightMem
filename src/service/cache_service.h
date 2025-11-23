#pragma once

#include <torch/extension.h>

#include "config.h"
#include "core/cache_task.h"
#include "core/error.h"
#include "core/task_queue.h"
#include "storage/local_storage_engine.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

int64_t get_env_int64(const char *name) {
  if (!name) {
    return 0;
  }
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return 0;
  }
  errno = 0;
  char *end = nullptr;
  long long parsed = std::strtoll(value, &end, 10);
  if (end == value || errno == ERANGE) {
    return 0;
  }
  return static_cast<int64_t>(parsed);
}

int64_t resolve_max_block_size_bytes() {
  const int64_t env_override_mb = get_env_int64(LM_MaxBlockSizeEnvVar);
  if (env_override_mb > 0) {
    return env_override_mb * 1024ll * 1024ll;  // Convert MB to bytes
  }
  return LM_DefaultMaxBlockSizeBytes;
}

} // namespace

namespace cache::service {

struct alignas(8) CacheParam_t {
  char *base_ptr;
  int64_t page_size;
  int64_t num_of_layer;
  int64_t layer_stride;
  int64_t page_stride;
  int64_t num_of_page;

  CacheParam_t() : base_ptr(nullptr), page_size(0), num_of_layer(0), layer_stride(0), page_stride(0), num_of_page(0) {}

  CacheParam_t(char *ptr, int64_t page, int64_t layer, int64_t layer_strd, int64_t page_strd, int64_t pages)
      : base_ptr(ptr), page_size(page), num_of_layer(layer), layer_stride(layer_strd), page_stride(page_strd),
        num_of_page(pages) {}
};

// Generic Cache Service Base Class
class CacheService {
public:
  /**
   * Constructor
   * @param kvcache Reference to the CPU kv-cache tensor backing this service
   */
  explicit CacheService(const torch::Tensor &kvcache)
      : queue_(std::make_unique<cache::queue::TaskQueue>()), cache_info_(), block_size_(0) {

    if (kvcache.dim() != 3) {
      throw std::runtime_error("The input kv-cache tensor has an incorrect dimension; it must be a 3D tensor of shape "
                               "[num page, num layer, page size].");
    }

    if (kvcache.is_cuda()) {
      throw std::runtime_error("GPU kv-cache tensors are not supported. Please provide a CPU tensor.");
    }

    if (!kvcache.is_contiguous()) {
      throw std::runtime_error("The kv-cache tensor must be stored contiguously.");
    }

    const int64_t element_size = kvcache.element_size();
    const auto sizes = kvcache.sizes();
    const auto strides = kvcache.strides();

    int64_t num_layers = 0;
    int64_t num_pages = 0;
    int64_t layer_stride = 0;
    int64_t page_stride = 0;

    num_layers = sizes[1];
    num_pages = sizes[0];
    if (strides[2] != 1 || strides[1] != sizes[2]) {
      throw std::runtime_error("Page-major kv-cache tensor must have contiguous layer dimension.");
    }
    if (strides[0] != sizes[1] * sizes[2]) {
      throw std::runtime_error("Page-major kv-cache tensor must be contiguous along the page dimension.");
    }

    layer_stride = strides[1] * element_size;
    page_stride = strides[0] * element_size;

    const int64_t inferred_page_size = sizes[2] * element_size;
    cache_info_ =
        CacheParam_t((char *)kvcache.data_ptr(), inferred_page_size, num_layers, layer_stride, page_stride, num_pages);

    const int64_t page_bytes = cache_info_.page_size * cache_info_.num_of_layer;
    const int64_t block_limit = resolve_max_block_size_bytes();

    int64_t pages_per_block = std::max<int64_t>(1, LM_TokensPerBlock);
    if (block_limit > 0 && block_limit < std::numeric_limits<int64_t>::max()) {
      const int64_t max_pages_by_size = std::max<int64_t>(1, block_limit / std::max<int64_t>(page_bytes, int64_t(1)));
      pages_per_block = std::max<int64_t>(1, std::min<int64_t>(pages_per_block, max_pages_by_size));
    }
    block_size_ = page_bytes * pages_per_block;

    printf("CacheService created with following cache info: \n");
    printf("\tNum of layer: %lld \n", static_cast<long long>(cache_info_.num_of_layer));
    printf("\tNum of page: %lld \n", static_cast<long long>(cache_info_.num_of_page));
    printf("\tPage Size: %lld \n", static_cast<long long>(cache_info_.page_size));
    printf("\tLayer Stride: %lld \n", static_cast<long long>(cache_info_.layer_stride));
    printf("\tPage Stride: %lld \n", static_cast<long long>(cache_info_.page_stride));
    printf("\tPages Per Block: %lld \n", static_cast<long long>(pages_per_block));
    printf("\tBlock Size: %lld \n", static_cast<long long>(block_size_));
  }

  /**
   * The query function querys a given hash keys whether exist in current cache system.
   * A bool array will be returned as the query result,
   * where true means hash existence and false means not
   *
   * This function will throw no exception or error.
   */
  virtual std::vector<bool> query(const std::vector<std::string> &hashs) = 0;

  int64_t block_size() const;
  int64_t page_size() const;

  int64_t active_create_count(const std::string &mode) const;

  /**
   * The create function creates an asynchronous read/write task for kvcache.
   * All asynchronous read/write tasks are divided into a series of TaskBlocks,
   * each with a fixed size of block_size_.
   * This behavior is to facilitate efficient execution of subsequent transfers.
   * To create such a task, the user needs to provide the hash value corresponding
   * to each TaskBlock according to certain rules.
   * Asynchronous read requests will index the data based on these hash values.
   * The returned task object is jointly held by CacheService and Python.
   * The resources will only be destroyed when the task execution is completed
   * and the reference on the Python side is released.
   *
   * @param hashs Vector of hash strings to process
   * @param mode Operation mode of the cache task
   * @param kv_page_indexer A pytorch tensor of type int64 used to specify the kvcache
   * space corresponding to this task
   * @return Shared pointer to a newly created CacheTask
   * @exception If there is too many tasks(>)
   */
  std::shared_ptr<cache::task::CacheTask> create(const std::vector<std::string> &hashs,
                                                 const torch::Tensor &kv_page_indexer_, const std::string &mode) {
    std::atomic<int64_t> *active_counter = nullptr;
    if (mode == "r") {
      active_counter = &active_read_creates_;
    } else if (mode == "w") {
      active_counter = &active_write_creates_;
    } else {
      throw std::runtime_error("Invalid mode string. Use 'r' for Read or 'w' for Write.");
    }

    // Split task into blocks
    const int64_t page_size = cache_info_.page_size * cache_info_.num_of_layer;
    const int64_t page_per_block = (block_size_ + page_size - 1) / page_size;
    const int64_t num_of_pages = kv_page_indexer_.numel();
    const int64_t num_of_blocks = (num_of_pages + page_per_block - 1) / page_per_block;

    if (kv_page_indexer_.dtype() != torch::kInt32) {
      throw std::runtime_error("kv_page_indexer must be a tensor of type int32_t.");
    }

    if (!kv_page_indexer_.is_cpu()) {
      throw std::runtime_error("kv_page_indexer tensor must reside on CPU.");
    }

    auto kv_page_indexer = kv_page_indexer_;

    // Page mismatch
    if (num_of_blocks != hashs.size()) {
      throw std::runtime_error("The hash array length is incorrect, "
                               "want num of hash value: " +
                               std::to_string(num_of_blocks) + ", got: " + std::to_string(hashs.size()));
    }
    auto task = std::make_shared<cache::task::CacheTask>(hashs, kv_page_indexer, mode);

    // Initialize blocks after task is created (needed for weak_ptr in blocks)
    task->initialize_blocks(hashs);

    // For write mode, query which pages are already in disk cache
    if (mode == "w") {
      std::vector<bool> query_result = query(hashs);
      const int32_t *page_ptr = reinterpret_cast<int32_t *>(kv_page_indexer.data_ptr());

      for (int64_t block_idx = 0; block_idx < static_cast<int64_t>(hashs.size()); ++block_idx) {
        if (query_result[block_idx]) {
          // This block is already in disk cache, add its page indices to page_already_list
          const int64_t start_page_in_block = block_idx * page_per_block;
          const int64_t end_page_in_block = std::min(start_page_in_block + page_per_block, num_of_pages);

          for (int64_t page_idx = start_page_in_block; page_idx < end_page_in_block; ++page_idx) {
            task->page_already_list.push_back(page_ptr[page_idx]);
          }
        }
      }
    }

    active_counter->fetch_add(1, std::memory_order_relaxed);

    {
      std::lock_guard<std::mutex> lock(lock_);
      taskpool_.push_back(task);
    }

    // Submit task to the task queue
    if (queue_->submit(task) != cache::error::LM_SUCCESS) {
      {
        std::lock_guard<std::mutex> lock(lock_);
        for (auto it = taskpool_.begin(); it != taskpool_.end(); ++it) {
          if (it->get() == task.get()) {
            taskpool_.erase(it);
            break;
          }
        }
      }
      active_counter->fetch_sub(1, std::memory_order_relaxed);
      throw std::runtime_error("Failed to submit task to the queue. Too many tasks are currently queued.");
    }
    return task;
  }

  /**
   * The abort_task function notifies the system to immediately
   * abandon the subsequent execution of a task.
   * Since CacheService tasks are executed asynchronously in units of TaskBlocks,
   * some blocks in this task may have completed asynchronous access,
   * some may be running, and some may not have been launched yet.
   * The abort_task function will stop the launch of subsequent tasks
   * and immediately terminate the currently running blocks.
   * Blocks that have completed access are not affected.
   * Threads that are currently working may not be able to immediately end the task and respond,
   * and resources can only be reclaimed after the last thread completes execution.
   * After calling the abort_task function, the current task will no longer perform
   * read/write operations on the kvcache.
   */
  void abort_task(const std::shared_ptr<cache::task::CacheTask> &task) {
    for (const auto &block : task->blocks) {
      abort(block.get());
    }
  }

  /**
   * Notify the system to immediately abandon the subsequent execution of a Block.
   */
  void abort(cache::task::CacheBlock *block) {
    auto task = block->get_task();
    if (!task) {
      fprintf(stderr, "[light_mem warning] abort: task has been destroyed for block with hash %s\n",
              block->hash.c_str());
      return; // Task has been destroyed, nothing to do
    }

    std::lock_guard<std::mutex> lock(task->state_mutex);
    if (block->state == cache::task::State::Initial || block->state == cache::task::State::Working) {
      block->state = cache::task::State::Aborted;
      task->num_finished_blocks.fetch_add(1, std::memory_order_release);
      if (task->ready()) {
        finalize_task(task);
      }
    }
  }

  /**
   * Mark a CacheBlock as completed and update its state.
   * This function is called by worker threads when they finish processing a block.
   * It atomically updates the block state to Finished and increments the task's
   * finished block counter. If all blocks of the task are completed, it triggers
   * task finalization to release resources.
   *
   * @param block Pointer to the CacheBlock that has completed execution
   */
  void deliver(cache::task::CacheBlock *block) {
    auto task = block->get_task();
    if (!task) {
      fprintf(stderr, "[light_mem warning] deliver: task has been destroyed for block with hash %s\n",
              block->hash.c_str());
      return; // Task has been destroyed, nothing to do
    }

    {
      std::lock_guard<std::mutex> lock(task->state_mutex);

      if (block->state == cache::task::State::Working) {
        block->state = cache::task::State::Finished;
        task->num_finished_blocks.fetch_add(1, std::memory_order_release);
      }
    }

    // If all blocks of this task have completed execution,
    if (task->ready()) {
      finalize_task(task);
    }
  }

protected:
  void finalize_task(const std::shared_ptr<cache::task::CacheTask> &task);
  virtual void on_task_finalized(const std::shared_ptr<cache::task::CacheTask> &task);

  std::mutex lock_;
  std::vector<std::shared_ptr<cache::task::CacheTask>> taskpool_;
  std::unique_ptr<cache::queue::TaskQueue> queue_;

  CacheParam_t cache_info_;

  int64_t block_size_;
  std::atomic<int64_t> active_read_creates_{0};
  std::atomic<int64_t> active_write_creates_{0};
};

inline int64_t CacheService::block_size() const { return block_size_; }

inline int64_t CacheService::page_size() const { return cache_info_.page_size; }

inline int64_t CacheService::active_create_count(const std::string &mode) const {
  if (mode == "r") {
    return active_read_creates_.load(std::memory_order_relaxed);
  }
  if (mode == "w") {
    return active_write_creates_.load(std::memory_order_relaxed);
  }
  throw std::runtime_error("Invalid mode string. Use 'r' for Read or 'w' for Write.");
}

inline void CacheService::finalize_task(const std::shared_ptr<cache::task::CacheTask> &task) {
  if (!task->mark_completion_notified()) {
    return;
  }

  std::atomic<int64_t> *active_counter =
      (task->operation_mode == cache::task::Mode::Read) ? &active_read_creates_ : &active_write_creates_;
  active_counter->fetch_sub(1, std::memory_order_relaxed);
  on_task_finalized(task);

  {
    std::lock_guard<std::mutex> lock(lock_);
    for (auto it = taskpool_.begin(); it != taskpool_.end(); ++it) {
      if (it->get() == task.get()) {
        taskpool_.erase(it);
        break;
      }
    }
  }
}

} // namespace cache::service

inline void cache::service::CacheService::on_task_finalized(const std::shared_ptr<cache::task::CacheTask> &task) {}
