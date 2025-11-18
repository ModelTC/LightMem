#pragma once

namespace cache::error {

/**
 * @brief 错误码类型定义
 *
 * 下面的错误码用于指示 CacheService 的各种操作结果。
 */
enum LMError_t {
  LM_SUCCESS = 0,              // 操作成功
  LM_MISSING_FILE = 1,         // 文件不存在
  LM_FILE_EXISTS = 2,          // 文件已存在（冲突）
  LM_FILE_CREATION_FAILED = 3, // 文件或目录创建失败
  LM_TIMEOUT = 4,              // 操作超时
  LM_CUDA_ERROR = 5,           // Cuda 操作失败
  LM_QUEUE_LENGTH_EXCEED = 6,  // 任务队列超长
  LM_HASH_ERROR = 7,           // 哈希值错误
  LM_CACHE_ERROR = 8,          // 缓存分配错误
  LM_TASK_MODE_ERROR = 9,      // 任务模式错误
  LM_BUFFER_EXCEED = 10        // 缓存不够
};

} // namespace cache::error
