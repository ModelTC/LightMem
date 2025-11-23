#pragma once

#include <torch/torch.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cache::task {

enum State { Initial, Working, Finished, Aborted };

enum Mode { Write, Read };

class CacheTask;

/**
 * @brief Represents a single block of data within a cache task
 *
 * Each CacheBlock corresponds to a fixed-size chunk of KV cache data that can be
 * independently read from or written to storage. Blocks are processed asynchronously
 * by worker threads and track their own state throughout the operation lifecycle.
 */
class CacheBlock {
public:
  CacheBlock(std::string hash_, const int64_t block_idx_, std::shared_ptr<CacheTask> task_)
      : hash(std::move(hash_)), task(task_), block_idx(block_idx_) {}

  bool ready() const { return state == State::Finished; }

  // Get a shared_ptr to the parent task, returns nullptr if task has been destroyed
  std::shared_ptr<CacheTask> get_task() const { return task.lock(); }

  int64_t block_idx;
  std::weak_ptr<CacheTask> task;
  std::string hash;
  State state{};
};

/**
 * @brief Manages a collection of cache blocks for a single read or write operation
 *
 * CacheTask represents a complete cache operation request from Python code, containing
 * multiple blocks that are processed in parallel by worker threads. It tracks completion
 * status, provides thread-safe access to shared state, and manages the lifecycle of all
 * associated blocks through RAII-managed unique_ptr ownership.
 *
 * Thread Safety: Atomic members (num_finished_blocks, num_data_ready_blocks, completion_notified)
 * are lock-free for high-frequency access. The state_mutex protects page_already_list_ updates.
 */
class CacheTask : public std::enable_shared_from_this<CacheTask> {
public:
  CacheTask(const std::vector<std::string> &hashs, torch::Tensor kv_page_indexer, const std::string &mode_str)
      : num_finished_blocks(0), num_data_ready_blocks(0), page_indexer(std::move(kv_page_indexer)),
        completion_notified(false) {

    if (mode_str == "r") {
      operation_mode = Mode::Read;
    } else if (mode_str == "w") {
      operation_mode = Mode::Write;
    } else {
      throw std::invalid_argument("Invalid mode string. Use 'r' for Read or 'w' for Write.");
    }

    blocks.reserve(hashs.size());
  }

  void initialize_blocks(const std::vector<std::string> &hashs) {
    int64_t idx = 0;
    for (const auto &hash : hashs) {
      blocks.emplace_back(std::make_unique<CacheBlock>(hash, idx++, shared_from_this()));
    }
  }

  bool ready() const { return num_finished_blocks.load(std::memory_order_acquire) == blocks.size(); }

  bool data_safe() const {
    if (operation_mode == Mode::Write) {
      return num_data_ready_blocks.load(std::memory_order_acquire) >= static_cast<int64_t>(blocks.size());
    }
    return ready();
  }

  bool mark_completion_notified() { return !completion_notified.exchange(true, std::memory_order_acq_rel); }

  std::vector<State> state() const {
    std::vector<State> ret;
    ret.reserve(blocks.size());
    for (const auto &block : blocks) {
      ret.push_back(block->state);
    }
    return ret;
  }

  std::vector<int32_t> get_page_already_list() const {
    std::lock_guard<std::mutex> lock_guard(state_mutex);
    return page_already_list;
  }

  torch::Tensor page_indexer;
  mutable std::mutex state_mutex;
  std::vector<std::unique_ptr<CacheBlock>> blocks;
  std::atomic<int64_t> num_finished_blocks;
  std::atomic<int64_t> num_data_ready_blocks;
  Mode operation_mode;
  std::atomic<bool> completion_notified;
  std::vector<int32_t> page_already_list;
};

} // namespace cache::task
