#pragma once

#include <cstddef>
#include <cstdint>

#include <zlib.h> // crc32

namespace cache {
namespace storage {

#pragma pack(push, 1)

struct SuperBlock {
  uint32_t magic;          // 0x4C4D454D (LMEM)
  uint32_t version;        // Format version
  uint64_t shard_id;       // Shard ID
  uint64_t sequence_id;    // Monotonically increasing ID
  uint64_t stable_offset;  // Offset in meta file where Journal scanning starts (truncate-mode WAL)
  uint64_t total_capacity; // Max capacity of the shard
  uint32_t block_size;     // Block size
  uint64_t current_epoch;  // Epoch for fencing
  uint8_t padding[4096 - 56];
  uint32_t crc32; // CRC of the first 4092 bytes
};

// Journal record:
// - Includes epoch for fencing.
// - Supports an epoch-marker record (flags & kFlagEpochMarker) to establish epoch boundaries in the append stream.
// Variable bytes layout: [hash bytes][evicted_hash bytes][record_crc32]
struct JournalRecord {
  uint32_t magic;            // 0x4A524E4C (JRNL)
  uint16_t version;          // Format version
  uint16_t flags;            // bitmask
  uint64_t epoch;            // Fencing epoch (monotonic per shard ownership)
  uint64_t write_offset;     // Offset in data file
  uint32_t write_len;        // Length of data written
  uint32_t data_crc;         // CRC of the data
  uint32_t hash_len;         // Length of hash key
  uint32_t evicted_hash_len; // Length of evicted hash (0 if none)
#if defined(__GNUC__) || defined(__clang__)
  uint8_t hash_key[0]; // Variable-length bytes follow this header (compiler extension)
#endif
};

#pragma pack(pop)

inline constexpr uint32_t SUPERBLOCK_MAGIC = 0x4C4D454D;
inline constexpr uint32_t SUPERBLOCK_VERSION = 1;
inline constexpr uint32_t JOURNAL_MAGIC = 0x4A524E4C;
inline constexpr uint16_t JOURNAL_VERSION = 1;
inline constexpr uint16_t JOURNAL_FLAG_EPOCH_MARKER = 1u << 0;
inline constexpr size_t SUPERBLOCK_SIZE = 4096;
inline constexpr size_t META_HEADER_SIZE = 8192; // 2 * SuperBlock

static_assert(sizeof(SuperBlock) == SUPERBLOCK_SIZE, "SuperBlock must be exactly 4096 bytes");

inline uint32_t compute_crc32(const void *data, size_t len) {
  return ::crc32(0, reinterpret_cast<const Bytef *>(data), len);
}

} // namespace storage
} // namespace cache
