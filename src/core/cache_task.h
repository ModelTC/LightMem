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

enum State { Initial = 0, Working = 1, Finished = 2, Aborted = 3 };

enum Mode { Write = 1, Read = 2 };

class CacheTask;

class CacheBlock {
public:
  /// @brief Constructor, calculates the SHA-256 hash of the data.
  /// @param hash Data hash.
  /// @param m Mode: Read or Write
  /// @param task Corresponding CacheTask
  CacheBlock(std::string hash_, const int64_t block_idx_, CacheTask *task_)
      : hash(std::move(hash_)), task(task_), block_idx(block_idx_) {}

  /// @brief 禁止拷贝和移动
  CacheBlock(CacheBlock &&other) = delete;
  CacheBlock &operator=(CacheBlock &&other) = delete;

  bool ready() const { return state == State::Finished; }

  int64_t block_idx;
  CacheTask *task;  ///< Corresponding Task
  std::string hash; ///< Hash of the block.
  State state{};    ///< Read/write state of the block.
};

/// @brief Task cache class, storing only cache blocks.
class CacheTask {
public:
  CacheTask() = delete;

  /// @brief Constructor, determines mode from user input ('r' or 'w').
  /// @param hashs Hash sequence.
  /// @param mode_str Mode string: "r" for Read, "w" for Write
  CacheTask(const std::vector<std::string> &hashs, torch::Tensor kv_page_indexer, const std::string &mode_str)
      : num_finished_blocks(0), num_data_ready_blocks(0), page_indexer(std::move(kv_page_indexer)),
        completion_notified(false) {

    if (mode_str == "r") {
      mode = Mode::Read;
    } else if (mode_str == "w") {
      mode = Mode::Write;
    } else {
      throw std::invalid_argument("Invalid mode string. Use 'r' for Read or 'w' for Write.");
    }

    blocks.reserve(hashs.size());
    for (int64_t idx = 0; idx < hashs.size(); ++idx) {
      blocks.emplace_back(new CacheBlock(hashs[idx], idx, this));
    }
  }

  ~CacheTask() {
    for (auto block : blocks) {
      delete block;
    }
  }

  /// @brief 禁止拷贝和移动
  CacheTask(CacheTask &&other) = delete;
  CacheTask &operator=(CacheTask &&other) = delete;

  bool ready() const { return num_finished_blocks.load(std::memory_order_acquire) == blocks.size(); }

  /// @brief Check if data is safe to release pages (for write mode)
  /// For write mode: returns true when data has been copied from KV cache
  /// For read mode: equivalent to ready()
  bool data_safe() const {
    if (mode == Mode::Write) {
      return num_data_ready_blocks.load(std::memory_order_acquire) >= static_cast<int64_t>(blocks.size());
    }
    return ready(); // For read mode, data_safe is same as ready
  }

  bool mark_completion_notified() { return !completion_notified.exchange(true, std::memory_order_acq_rel); }

  std::vector<State> state() {
    auto ret = std::vector<State>(blocks.size());
    for (int32_t i = 0; i < blocks.size(); ++i) {
      ret[i] = blocks[i]->state;
    }
    return ret;
  }

  std::vector<int32_t> get_page_already_list() const {
    std::lock_guard<std::mutex> lock_guard(const_cast<std::mutex &>(lock));
    return page_already_list;
  }

  // 这个指针是用来标记 task 中的数据存取位置的
  torch::Tensor page_indexer;

  std::mutex lock;                            ///< Task state lock
  std::vector<CacheBlock *> blocks;           ///< Blocks stored as shared_ptr
  std::atomic<int64_t> num_finished_blocks;   ///< Number of finished blocks (atomic for thread-safe reading)
  std::atomic<int64_t> num_data_ready_blocks; ///< Number of blocks with data copied (for write mode)
  Mode mode;                                  ///< Read/write mode of the task.
  std::atomic<bool> completion_notified;
  std::vector<int32_t> page_already_list; ///< List of page indices already persisted to disk (for write mode)
};

} // namespace cache::task
