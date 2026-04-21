#include "storage/local_storage_engine.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cstdlib>

namespace cache {
namespace storage {

// Conditionally delete a hash entry from the global index and CRC map.
static void conditionalDeleteGlobalEntry(RedisClient *redis, const std::string &hash, const std::string &expected) {
  const std::string global_index_key = redis->globalIndexKey();
  const std::string global_crc_key = redis->globalCrcKey();
  auto cur = redis->hget(global_index_key, hash);
  if (cur.has_value() && *cur == expected) {
    (void)redis->hdel(global_index_key, hash);
    (void)redis->hdel(global_crc_key, hash);
  }
}

void LocalStorageEngine::recoverShardToRedis(size_t shard_id) {
  if (shard_id >= shard_) {
    return;
  }
  RedisClient *redis = redisForShard(shard_id);
  if (!redis || !redis->connect()) {
    std::fprintf(stderr, "[light_mem warning] recoverShardToRedis: Redis not reachable (shard=%zu)\n", shard_id);
    return;
  }

  const size_t shard_capacity = (storage_size_ / shard_) / block_size_;

  // Rebuild from local snapshot + WAL. Redis is NOT a source of truth.
  std::unique_lock<std::shared_mutex> lock(*io_locks_[shard_id]);

  // Load mapping from snapshot if present.
  std::unordered_map<std::string, size_t> mapping;
  std::unordered_map<std::string, uint32_t> crc_map;
  std::unordered_set<std::string> crc_present;
  std::stringstream ss;
  ss << filename_ << "_" << shard_id << "/index";
  const std::string snap_path = ss.str();
  const bool snapshot_loaded = caches_[shard_id]->loadSnapshotToMapping(snap_path, mapping, crc_map, crc_present);
  if (!snapshot_loaded && ::access(snap_path.c_str(), F_OK) == 0) {
    std::fprintf(stderr,
                 "[light_mem warning] recoverShardToRedis: snapshot exists but load failed, continue with WAL "
                 "(shard=%zu file=%s)\n",
                 shard_id, snap_path.c_str());
  }

  // Apply WAL tail (after last checkpoint) to mapping.
  std::vector<JournalOp> ops;
  const off_t start_off =
      static_cast<off_t>((superblock_stable_offset_[shard_id] >= META_HEADER_SIZE) ? superblock_stable_offset_[shard_id]
                                                                                   : META_HEADER_SIZE);
  scanWalOps(shard_id, shard_capacity, start_off, ops);
  for (const auto &op : ops) {
    if (!op.evicted.empty()) {
      mapping.erase(op.evicted);
      crc_map.erase(op.evicted);
      crc_present.erase(op.evicted);
    }
    if (!op.hash.empty() && op.slot_id < shard_capacity) {
      mapping[op.hash] = op.slot_id;
      crc_map[op.hash] = op.data_crc;
      crc_present.insert(op.hash);
    }
  }

  const std::string shard_index_key = redis->shardIndexKey(shard_id);
  const std::string shard_seq_key = redis->shardSeqKey(shard_id);
  const std::string global_index_key = redis->globalIndexKey();
  const std::string global_crc_key = redis->globalCrcKey();

  // Cleanup stale global-index entries for this shard using the previous shard index.
  if (auto prev = redis->hgetall(shard_index_key); prev.has_value() && (prev->size() % 2 == 0)) {
    for (size_t i = 0; i + 1 < prev->size(); i += 2) {
      const std::string &old_hash = (*prev)[i];
      const std::string &old_slot = (*prev)[i + 1];
      if (mapping.find(old_hash) == mapping.end()) {
        conditionalDeleteGlobalEntry(redis, old_hash, std::to_string(shard_id) + ":" + old_slot);
      }
    }
  }

  std::vector<std::vector<std::string>> cmds;
  cmds.reserve(mapping.size() * 3 + 2);

  // Replace shard index with our rebuilt mapping.
  // Keep DEL in the same pipeline to reduce the chance of leaving Redis empty on a pipeline failure.
  cmds.push_back({"DEL", shard_index_key});

  std::vector<uint8_t> buf;
  buf.resize(block_size_);

  size_t skipped_crc = 0;
  for (const auto &kv : mapping) {
    const size_t slot_id = kv.second;
    uint32_t crc = 0;
    bool crc_ok = false;
    if (crc_present.find(kv.first) != crc_present.end()) {
      auto it = crc_map.find(kv.first);
      if (it != crc_map.end()) {
        crc = it->second;
        crc_ok = true;
      }
    }
    if (!crc_ok) {
      const size_t offset_bytes = slot_id * block_size_;
      if (file_fds_[shard_id] >= 0 &&
          preadAll(file_fds_[shard_id], buf.data(), block_size_, static_cast<off_t>(offset_bytes))) {
        crc = compute_crc32(buf.data(), block_size_);
        crc_ok = true;
      }
    }

    if (!crc_ok) {
      skipped_crc++;
      continue;
    }

    cmds.push_back({"HSET", shard_index_key, kv.first, std::to_string(slot_id)});
    cmds.push_back({"HSET", global_index_key, kv.first, std::to_string(shard_id) + ":" + std::to_string(slot_id)});
    cmds.push_back({"HSET", global_crc_key, kv.first, std::to_string(crc)});
  }
  cmds.push_back({"SET", shard_seq_key, std::to_string(superblock_seq_[shard_id])});
  if (!redis->pipeline(cmds)) {
    throw std::runtime_error("Recovery Redis pipeline failed");
  }
}

bool LocalStorageEngine::shouldFullRecoverRedis(size_t shard_id) {
  if (shard_id >= shard_) {
    return false;
  }
  RedisClient *redis = redisForShard(shard_id);
  if (!redis || !redis->connect()) {
    return false;
  }

  const std::string shard_index_key = redis->shardIndexKey(shard_id);
  const std::string shard_seq_key = redis->shardSeqKey(shard_id);

  auto hlen = redis->hlen(shard_index_key);
  if (!hlen.has_value() || *hlen == 0) {
    return true;
  }

  auto rseq = redis->getString(shard_seq_key);
  if (!rseq.has_value() || rseq->empty()) {
    return true;
  }
  try {
    const uint64_t seq = static_cast<uint64_t>(std::stoull(*rseq));
    if (seq != superblock_seq_[shard_id]) {
      return true;
    }
  } catch (...) {
    return true;
  }

  return false;
}

void LocalStorageEngine::recoverShardToRedisIncremental(size_t shard_id) {
  if (shard_id >= shard_) {
    return;
  }
  RedisClient *redis = redisForShard(shard_id);
  if (!redis || !redis->connect()) {
    std::fprintf(stderr, "[light_mem warning] recoverShardToRedisIncremental: Redis not reachable (shard=%zu)\n",
                 shard_id);
    return;
  }

  const size_t shard_capacity = (storage_size_ / shard_) / block_size_;
  const off_t start_off =
      static_cast<off_t>((superblock_stable_offset_[shard_id] >= META_HEADER_SIZE) ? superblock_stable_offset_[shard_id]
                                                                                   : META_HEADER_SIZE);
  const std::string shard_seq_key = redis->shardSeqKey(shard_id);

  std::vector<JournalOp> ops;
  scanWalOps(shard_id, shard_capacity, start_off, ops);
  if (ops.empty()) {
    // Still sync the shard seq so control-plane checks don't treat this as stale.
    if (!redis->setString(shard_seq_key, std::to_string(superblock_seq_[shard_id]))) {
      std::fprintf(stderr,
                   "[light_mem warning] recoverShardToRedisIncremental: Redis SET shard seq failed (shard=%zu)\n",
                   shard_id);
    }
    return;
  }

  const std::string shard_index_key = redis->shardIndexKey(shard_id);
  const std::string global_index_key = redis->globalIndexKey();
  const std::string global_crc_key = redis->globalCrcKey();

  std::vector<std::vector<std::string>> cmds;
  cmds.reserve(ops.size() * 5 + 1);
  for (const auto &op : ops) {
    if (!op.evicted.empty()) {
      cmds.push_back({"HDEL", shard_index_key, op.evicted});
    }
    cmds.push_back({"HSET", shard_index_key, op.hash, std::to_string(op.slot_id)});
    cmds.push_back({"HSET", global_index_key, op.hash, std::to_string(shard_id) + ":" + std::to_string(op.slot_id)});
    cmds.push_back({"HSET", global_crc_key, op.hash, std::to_string(op.data_crc)});
  }
  cmds.push_back({"SET", shard_seq_key, std::to_string(superblock_seq_[shard_id])});

  const bool pipeline_ok = redis->pipeline(cmds);
  if (!pipeline_ok) {
    std::fprintf(stderr,
                 "[light_mem warning] recoverShardToRedisIncremental: Redis pipeline failed (shard=%zu cmds=%zu)\n",
                 shard_id, cmds.size());
    // IMPORTANT: do not perform post-pipeline cleanup if the pipeline failed; otherwise we can delete valid
    // global-index entries while failing to install the corresponding new shard-index state.
    return;
  }

  // Conditional cleanup for evictions (do after pipeline).
  // The evicted hash previously occupied the same slot being overwritten by op.hash.
  for (const auto &op : ops) {
    if (!op.evicted.empty()) {
      conditionalDeleteGlobalEntry(redis, op.evicted, std::to_string(shard_id) + ":" + std::to_string(op.slot_id));
    }
  }
}

void LocalStorageEngine::recoverShardToRedisSmart(size_t shard_id) {
  if (shouldFullRecoverRedis(shard_id)) {
    recoverShardToRedis(shard_id);
    return;
  }
  recoverShardToRedisIncremental(shard_id);
}

void LocalStorageEngine::scanWalOps(size_t shard_id, size_t shard_capacity, off_t start_off,
                                    std::vector<JournalOp> &ops) {
  struct stat st{};
  if (::fstat(meta_fds_[shard_id], &st) != 0) {
    std::fprintf(stderr,
                 "[light_mem warning] wal scan: fstat failed (shard=%zu fd=%d errno=%d %s)\n",
                 shard_id, meta_fds_[shard_id], errno, std::strerror(errno));
    return;
  }
  const off_t end = st.st_size;
  off_t off = start_off;

  uint64_t max_epoch_seen = 0;
  bool warned_io = false;
  bool warned_crc = false;
  bool warned_bad_record = false;

  while (off + static_cast<off_t>(sizeof(uint32_t)) <= end) {
    const off_t record_start = off;
    uint32_t magic = 0;
    if (!preadAll(meta_fds_[shard_id], &magic, sizeof(uint32_t), off)) {
      if (!warned_io) {
        std::fprintf(stderr,
                     "[light_mem warning] wal scan: read magic failed (shard=%zu off=%lld start=%lld end=%lld errno=%d %s)\n",
                     shard_id, static_cast<long long>(off), static_cast<long long>(start_off),
                     static_cast<long long>(end), errno, std::strerror(errno));
        warned_io = true;
      }
      break;
    }
    if (magic != JOURNAL_MAGIC) {
      // Not necessarily an error: tail may contain junk/partial writes. Log only if this happens at the scan start.
      if (off == start_off) {
        std::fprintf(stderr,
                     "[light_mem warning] wal scan: magic mismatch at start (shard=%zu off=%lld got=0x%08x expect=0x%08x)\n",
                     shard_id, static_cast<long long>(off), magic, JOURNAL_MAGIC);
      }
      break;
    }

    // Parse journal record (epoch-aware).
    bool parsed = false;
    if (off + static_cast<off_t>(sizeof(JournalRecord)) <= end) {
      JournalRecord rec{};
      if (preadAll(meta_fds_[shard_id], &rec, sizeof(JournalRecord), off) && rec.magic == JOURNAL_MAGIC &&
          rec.version == JOURNAL_VERSION && (rec.flags & ~JOURNAL_FLAG_EPOCH_MARKER) == 0 && rec.hash_len <= 4096 &&
          rec.evicted_hash_len <= 4096) {
        off += static_cast<off_t>(sizeof(JournalRecord));
        if (off + static_cast<off_t>(rec.hash_len) + static_cast<off_t>(rec.evicted_hash_len) +
                static_cast<off_t>(sizeof(uint32_t)) >
            end) {
          break;
        }

        std::string hash;
        hash.resize(rec.hash_len);
        if (rec.hash_len > 0) {
          if (!preadAll(meta_fds_[shard_id], hash.data(), rec.hash_len, off)) {
            break;
          }
        }
        off += static_cast<off_t>(rec.hash_len);

        std::string evicted;
        evicted.resize(rec.evicted_hash_len);
        if (rec.evicted_hash_len > 0) {
          if (!preadAll(meta_fds_[shard_id], evicted.data(), rec.evicted_hash_len, off)) {
            break;
          }
        }
        off += static_cast<off_t>(rec.evicted_hash_len);

        uint32_t record_crc = 0;
        if (!preadAll(meta_fds_[shard_id], &record_crc, sizeof(uint32_t), off)) {
          break;
        }
        off += static_cast<off_t>(sizeof(uint32_t));

        std::vector<uint8_t> tmp;
        tmp.resize(sizeof(JournalRecord) + hash.size() + evicted.size());
        std::memcpy(tmp.data(), &rec, sizeof(JournalRecord));
        if (!hash.empty()) {
          std::memcpy(tmp.data() + sizeof(JournalRecord), hash.data(), hash.size());
        }
        if (!evicted.empty()) {
          std::memcpy(tmp.data() + sizeof(JournalRecord) + hash.size(), evicted.data(), evicted.size());
        }
        const uint32_t expect = compute_crc32(tmp.data(), tmp.size());
        if (expect != record_crc) {
          if (!warned_crc) {
            std::fprintf(stderr,
                         "[light_mem warning] wal scan: record crc mismatch (shard=%zu off=%lld)\n",
                         shard_id, static_cast<long long>(record_start));
            warned_crc = true;
          }
          parsed = true;
          continue;
        }

        if ((rec.flags & JOURNAL_FLAG_EPOCH_MARKER) != 0) {
          if (rec.epoch >= max_epoch_seen) {
            max_epoch_seen = rec.epoch;
          }
          parsed = true;
          continue;
        }

        if (rec.epoch < max_epoch_seen) {
          parsed = true;
          continue;
        }
        if (rec.epoch > max_epoch_seen) {
          max_epoch_seen = rec.epoch;
        }

        if (rec.write_offset % block_size_ != 0) {
          parsed = true;
          continue;
        }
        const size_t slot_id = static_cast<size_t>(rec.write_offset / block_size_);
        if (slot_id >= shard_capacity) {
          parsed = true;
          continue;
        }

        JournalOp op;
        op.hash = std::move(hash);
        op.evicted = std::move(evicted);
        op.slot_id = slot_id;
        op.data_crc = rec.data_crc;
        ops.emplace_back(std::move(op));
        parsed = true;
      }
    }

    if (parsed) {
      continue;
    }

    // Not a valid record at this offset; stop scanning.
    if (!warned_bad_record) {
      std::fprintf(stderr,
                   "[light_mem warning] wal scan: invalid record, stop scanning (shard=%zu off=%lld start=%lld end=%lld)\n",
                   shard_id, static_cast<long long>(record_start), static_cast<long long>(start_off),
                   static_cast<long long>(end));
      warned_bad_record = true;
    }
    break;
  }
}

void LocalStorageEngine::recoverAllShards(size_t shard_capacity) {
  for (size_t i = 0; i < shard_; i++) {
    recoverShard(i, shard_capacity);
  }
}

void LocalStorageEngine::recoverShard(size_t shard_id, size_t shard_capacity) {
  RedisClient *redis = redisForShard(shard_id);
  const bool redis_ok = (redis && redis->connect());
  caches_[shard_id]->reset();

  // 0. Try to load from Local Snapshot first
  std::stringstream ss;
  ss << filename_ << "_" << shard_id << "/index";
  std::string snap_path = ss.str();
  bool snapshot_loaded = caches_[shard_id]->loadSnapshotToIndex(snap_path);
  if (!snapshot_loaded && ::access(snap_path.c_str(), F_OK) == 0) {
    std::fprintf(stderr,
                 "[light_mem warning] recoverShard: snapshot exists but load failed, continue with WAL "
                 "(shard=%zu file=%s)\n",
                 shard_id, snap_path.c_str());
  }

  // 1. Scan WAL
  // We rely SOLELY on Snapshot + WAL.
  // Redis is treated as a cache/index that reflects our state, not the source of truth.
  const off_t start_off =
      static_cast<off_t>((superblock_stable_offset_[shard_id] >= META_HEADER_SIZE) ? superblock_stable_offset_[shard_id]
                                                                                   : META_HEADER_SIZE);

  std::vector<JournalOp> ops;
  scanWalOps(shard_id, shard_capacity, start_off, ops);

  // 2. Apply WAL to Memory
  if (!ops.empty()) {
    for (const auto &op : ops) {
      if (!op.evicted.empty()) {
        caches_[shard_id]->remove(op.evicted);
      }
      caches_[shard_id]->put_ready(op.hash, op.slot_id, op.data_crc);
    }
  }

  // 3. Sync WAL updates to Redis (and truncate WAL if successful)
  if (!ops.empty() && redis_ok) {
    writeCheckpointSuperblockOnly(shard_id);

    bool replay_ok = true;
    const std::string shard_index_key = redis->shardIndexKey(shard_id);
    const std::string global_index_key = redis->globalIndexKey();
    const std::string global_crc_key = redis->globalCrcKey();
    const std::string shard_seq_key = redis->shardSeqKey(shard_id);
    for (const auto &op : ops) {
      if (!op.evicted.empty()) {
        if (!redis->hdel(shard_index_key, op.evicted)) {
          std::fprintf(stderr, "[light_mem warning] recoverShard: Redis HDEL shard index failed (shard=%zu hash=%s)\n",
                       shard_id, op.evicted.c_str());
          replay_ok = false;
          break;
        }
        // Conditional global delete to avoid deleting a newer mapping for the same hash.
        conditionalDeleteGlobalEntry(redis, op.evicted, std::to_string(shard_id) + ":" + std::to_string(op.slot_id));
      }
      if (!redis->hset(shard_index_key, op.hash, std::to_string(op.slot_id))) {
        std::fprintf(stderr, "[light_mem warning] recoverShard: Redis HSET shard index failed (shard=%zu hash=%s)\n",
                     shard_id, op.hash.c_str());
        replay_ok = false;
        break;
      }

      if (!redis->hset(global_crc_key, op.hash, std::to_string(op.data_crc))) {
        std::fprintf(stderr, "[light_mem warning] recoverShard: Redis HSET crc failed (shard=%zu hash=%s)\n", shard_id,
                     op.hash.c_str());
        replay_ok = false;
        break;
      }

      // Global mapping: only recover if missing; duplicates are intentionally ignored.
      (void)redis->hsetnx(global_index_key, op.hash, std::to_string(shard_id) + ":" + std::to_string(op.slot_id));
    }
    if (replay_ok) {
      if (!redis->setString(shard_seq_key, std::to_string(superblock_seq_[shard_id]))) {
        std::fprintf(stderr, "[light_mem warning] recoverShard: Redis SET shard seq failed (shard=%zu)\n", shard_id);
        replay_ok = false;
      }
    }

    if (replay_ok) {
      std::stringstream css;
      css << filename_ << "_" << shard_id << "/index";
      const std::string snap_path2 = css.str();

      std::unique_lock<std::shared_mutex> io_lock(*io_locks_[shard_id]);
      if (!caches_[shard_id]->saveToSnapshot(snap_path2)) {
        std::fprintf(stderr,
                     "[light_mem warning] recoverShard: failed to save snapshot; keep WAL (shard=%zu file=%s)\n",
                     shard_id, snap_path2.c_str());
      } else {
        // Snapshot now reflects the replayed WAL tail; it is safe to truncate WAL.
        truncateJournalToHeader(shard_id);
      }
    }
  }

  // 4. Handle Empty State / Fresh Shard
  // If we have no snapshot and no WAL logs, we are effectively empty.
  // We must ensure Redis doesn't hold stale data pointing to us.
  if (!snapshot_loaded && ops.empty() && redis_ok) {
    (void)redis->del(redis->shardIndexKey(shard_id));
  }
}

} // namespace storage
} // namespace cache
