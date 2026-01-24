#pragma once

#include <cstdint>

// number of tokens per asynchronous block
inline constexpr int64_t LM_TokensPerBlock = 64ll;
// upper bound for bytes staged in a single block copy (can be overridden via env)
inline constexpr int64_t LM_DefaultMaxBlockSizeMB = 64ll;
inline constexpr int64_t LM_DefaultMaxBlockSizeBytes = LM_DefaultMaxBlockSizeMB * 1024ll * 1024ll;
inline constexpr const char *LM_MaxBlockSizeEnvVar = "LIGHTMEM_MAX_BLOCK_SIZE_MB";
// for control SM usage
inline constexpr int64_t LM_KernelBlocks = 64ll;
// task queue length
inline constexpr int64_t LM_QueueSize = 131072ll;
