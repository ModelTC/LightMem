#pragma once

#include <cstdint>

// number of tokens per asynchronous block
inline constexpr int64_t LM_TokensPerBlock = 64ll;
// upper bound for bytes staged in a single block copy (can be overridden via env)
inline constexpr int64_t LM_DefaultMaxBlockSizeMB = 64ll;
inline constexpr int64_t LM_DefaultMaxBlockSizeBytes = LM_DefaultMaxBlockSizeMB * 1024ll * 1024ll;
inline constexpr const char *LM_MaxBlockSizeEnvVar = "LIGHTMEM_MAX_BLOCK_SIZE_MB";

// Scheme A: data file striping.
// Each shard's data is split across multiple stripe sub-files (data.0 .. data.S-1).
// A single (potentially large) block is split into S contiguous segments that are written
// to S different files in parallel. On distributed/afs-style filesystems the unit of write
// parallelism is the FILE, so fanning one block across multiple files raises effective
// bandwidth even when there are few concurrent blocks.
inline constexpr int64_t LM_DefaultDataStripes = 1ll;
inline constexpr const char *LM_DataStripesEnvVar = "LIGHTMEM_DATA_STRIPES";
// Background threads issuing concurrent per-stripe I/O for a single block.
inline constexpr int64_t LM_DefaultIoPoolThreads = 1ll;
inline constexpr const char *LM_IoPoolThreadsEnvVar = "LIGHTMEM_IO_POOL_THREADS";

// for control SM usage
inline constexpr int64_t LM_KernelBlocks = 64ll;
// task queue length
inline constexpr int64_t LM_QueueSize = 131072ll;
