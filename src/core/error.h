#pragma once

namespace cache::error {

/**
 * @brief 错误码类型定义
 *
 * 下面的错误码用于指示 CacheService 的各种操作结果。
 */
enum LMError_t {
  LM_SUCCESS,              // 操作成功
  LM_MISSING_FILE,         // 文件不存在
  LM_FILE_EXISTS,          // 文件已存在（冲突）
  LM_FILE_CREATION_FAILED, // 文件或目录创建失败
  LM_TIMEOUT,              // 操作超时
  LM_CUDA_ERROR,           // Cuda 操作失败
  LM_QUEUE_LENGTH_EXCEED,  // 任务队列超长
  LM_HASH_ERROR,           // 哈希值错误
  LM_CACHE_ERROR,          // 缓存分配错误
  LM_TASK_MODE_ERROR,      // 任务模式错误
  LM_BUFFER_EXCEED         // 缓存不够
};

} // namespace cache::error