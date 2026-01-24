#pragma once

#include "config.h"
#include "core/cache_task.h"
#include "core/error.h"

#include <boost/lockfree/queue.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace cache::queue {

/**
 * @brief Task queue class that stores all tasks & controls task state.
 *
 */
class TaskQueue {
public:
  TaskQueue() = default;
  TaskQueue(const TaskQueue &) = delete;
  TaskQueue &operator=(const TaskQueue &) = delete;

  cache::error::LMError_t submit(const std::shared_ptr<cache::task::CacheTask> &task) {
    for (const auto &block : task->blocks) {
      blocks_.push(block.get());
    }
    return cache::error::LM_SUCCESS;
  }

  cache::task::CacheBlock *claim() {
    cache::task::CacheBlock *block = nullptr;
    if (blocks_.pop(block)) {
      // Get the task to check if it's been aborted
      auto task = block->get_task();
      if (!task) {
        return nullptr;  // Task destroyed, skip this block
      }
      // Check if this block was already aborted before claiming it
      {
        std::lock_guard<std::mutex> lock(task->state_mutex);
        if (block->state == cache::task::State::Aborted) {
          return nullptr;  // Block was aborted, don't process it
        }
        block->state = cache::task::State::Working;
      }
      return block;
    }
    return nullptr;
  }

private:
  boost::lockfree::queue<cache::task::CacheBlock *> blocks_{LM_QueueSize};
};

} // namespace cache::queue
