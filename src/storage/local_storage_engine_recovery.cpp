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

namespace cache {
namespace storage {

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
  // This routine is intended to be called when a node (re-)acquires ownership of a shard.
  std::unique_lock<std::shared_mutex> lock(*io_locks_[shard_id]);

  // Load mapping from snapshot if present.
  std::unordered_map<std::string, size_t> mapping;
  std::unordered_map<std::string, uint32_t> crc_map;
  std::unordered_set<std::string> crc_present;
  {
    constexpr uint32_t kSnapshotMagic = 0x534E4150; // SNAP
    constexpr uint32_t kSnapshotVersion = 1;

    std::stringstream ss;
    ss << filename_ << "_" << shard_id << "/index";
    const std::string snap_path = ss.str();

    int fd = ::open(snap_path.c_str(), O_RDONLY);
    if (fd >= 0) {
      bool warned_snapshot = false;
      int last_errno = 0;
      bool last_eof = false;
      auto read_all = [&](void *p, size_t n) -> bool {
        char *buf = static_cast<char *>(p);
        size_t left = n;
        while (left > 0) {
          ssize_t r = ::read(fd, buf, left);
          if (r == 0) {
            last_eof = true;
            return false;
          }
          if (r < 0) {
            last_errno = errno;
            return false;
          }
          buf += static_cast<size_t>(r);
          left -= static_cast<size_t>(r);
        }
        return true;
      };

      uint32_t magic = 0;
      uint32_t version = 0;
      uint64_t count = 0;
      if (read_all(&magic, sizeof(magic)) && read_all(&version, sizeof(version)) && read_all(&count, sizeof(count)) &&
          magic == kSnapshotMagic && version == kSnapshotVersion) {
        for (uint64_t i = 0; i < count; i++) {
          uint32_t hash_len = 0;
          if (!read_all(&hash_len, sizeof(hash_len)) || hash_len > 4096) {
            if (!warned_snapshot) {
              std::fprintf(stderr, "[light_mem warning] recoverShardToRedis: snapshot truncated/corrupt (file=%s)\n",
                           snap_path.c_str());
              warned_snapshot = true;
            }
            break;
          }
          std::string hash;
          hash.resize(hash_len);
          if (hash_len > 0 && !read_all(hash.data(), hash_len)) {
            if (!warned_snapshot) {
              std::fprintf(stderr, "[light_mem warning] recoverShardToRedis: snapshot truncated/corrupt (file=%s)\n",
                           snap_path.c_str());
              warned_snapshot = true;
            }
            break;
          }
          uint64_t slot = 0;
          if (!read_all(&slot, sizeof(slot))) {
            if (!warned_snapshot) {
              std::fprintf(stderr, "[light_mem warning] recoverShardToRedis: snapshot truncated/corrupt (file=%s)\n",
                           snap_path.c_str());
              warned_snapshot = true;
            }
            break;
          }
          uint32_t crc = 0;
          if (!read_all(&crc, sizeof(crc))) {
            if (!warned_snapshot) {
              std::fprintf(stderr, "[light_mem warning] recoverShardToRedis: snapshot truncated/corrupt (file=%s)\n",
                           snap_path.c_str());
              warned_snapshot = true;
            }
            break;
          }
          if (slot < shard_capacity) {
            mapping[hash] = static_cast<size_t>(slot);
            crc_map[hash] = crc;
            crc_present.insert(hash);
          }
        }
      } else {
        // Header read failed or magic/version mismatch.
        if (!warned_snapshot) {
          if (last_eof) {
            std::fprintf(stderr, "[light_mem warning] recoverShardToRedis: snapshot header truncated (file=%s)\n",
                         snap_path.c_str());
          } else if (last_errno != 0) {
            std::fprintf(stderr,
                         "[light_mem warning] recoverShardToRedis: snapshot header read failed (file=%s errno=%d %s)\n",
                         snap_path.c_str(), last_errno, std::strerror(last_errno));
          } else {
            std::fprintf(stderr,
                         "[light_mem warning] recoverShardToRedis: snapshot header mismatch (file=%s magic=0x%08x version=%u)\n",
                         snap_path.c_str(), magic, version);
          }
          warned_snapshot = true;
        }
      }
      ::close(fd);
    } else if (errno != ENOENT) {
      std::fprintf(stderr, "[light_mem warning] recoverShardToRedis: open snapshot failed (file=%s errno=%d %s)\n",
                   snap_path.c_str(), errno, std::strerror(errno));
    }
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

  const std::string key = redis->shardIndexKey(shard_id);
  const std::string gkey = redis->globalIndexKey();

  // Cleanup stale global-index entries for this shard using the previous shard index.
  if (auto prev = redis->hgetall(key); prev.has_value() && (prev->size() % 2 == 0)) {
    for (size_t i = 0; i + 1 < prev->size(); i += 2) {
      const std::string &old_hash = (*prev)[i];
      if (mapping.find(old_hash) == mapping.end()) {
        (void)redis->hdel(gkey, old_hash);
        (void)redis->hdel(redis->globalCrcKey(), old_hash);
      }
    }
  }

  // Replace shard index with our rebuilt mapping.
  (void)redis->del(key);

  std::vector<std::vector<std::string>> cmds;
  cmds.reserve(mapping.size() * 3 + 1);

  std::vector<uint8_t> buf;
  buf.resize(block_size_);

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
      // Unable to compute CRC; skip publishing this mapping to avoid missing CRC in Redis.
      continue;
    }

    cmds.push_back({"HSET", key, kv.first, std::to_string(slot_id)});
    cmds.push_back({"HSET", gkey, kv.first, std::to_string(shard_id) + ":" + std::to_string(slot_id)});
    cmds.push_back({"HSET", redis->globalCrcKey(), kv.first, std::to_string(crc)});
  }
  cmds.push_back({"SET", redis->shardSeqKey(shard_id), std::to_string(superblock_seq_[shard_id])});
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

  const std::string key = redis->shardIndexKey(shard_id);

  auto hlen = redis->hlen(key);
  if (!hlen.has_value() || *hlen == 0) {
    return true;
  }

  auto rseq = redis->getString(redis->shardSeqKey(shard_id));
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

  std::vector<JournalOp> ops;
  scanWalOps(shard_id, shard_capacity, start_off, ops);
  if (ops.empty()) {
    return;
  }

  const std::string key = redis->shardIndexKey(shard_id);
  const std::string gkey = redis->globalIndexKey();

  std::vector<std::vector<std::string>> cmds;
  cmds.reserve(ops.size() * 5 + 1);
  for (const auto &op : ops) {
    if (!op.evicted.empty()) {
      cmds.push_back({"HDEL", key, op.evicted});
      cmds.push_back({"HDEL", gkey, op.evicted});
      cmds.push_back({"HDEL", redis->globalCrcKey(), op.evicted});
    }
    cmds.push_back({"HSET", key, op.hash, std::to_string(op.slot_id)});
    cmds.push_back({"HSET", gkey, op.hash, std::to_string(shard_id) + ":" + std::to_string(op.slot_id)});
    cmds.push_back({"HSET", redis->globalCrcKey(), op.hash, std::to_string(op.data_crc)});
  }
  cmds.push_back({"SET", redis->shardSeqKey(shard_id), std::to_string(superblock_seq_[shard_id])});

  if (!redis->pipeline(cmds)) {
    std::fprintf(stderr, "[light_mem warning] recoverShardToRedisIncremental: Redis pipeline failed (shard=%zu cmds=%zu)\n",
                 shard_id, cmds.size());
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
  bool snapshot_loaded = caches_[shard_id]->loadFromSnapshot(snap_path);

  // If snapshot is loaded, also warm up the local hash->shard hint map so distributed-mode
  // reads/queries can locate the shard in O(1) without scanning all shards or hitting Redis.
  if (snapshot_loaded && online_mode_) {
    std::vector<std::string> hashes;
    caches_[shard_id]->dump_ready(hashes);
    for (const auto &hash : hashes) {
      noteLocalShardHint(hash, shard_id);
    }
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
        if (online_mode_) {
          eraseLocalShardHint(op.evicted);
        }
      }
      caches_[shard_id]->put_ready(op.hash, op.slot_id, op.data_crc);
      if (online_mode_) {
        noteLocalShardHint(op.hash, shard_id);
      }
    }
  }

  // 3. Sync WAL updates to Redis (and truncate WAL if successful)
  if (!ops.empty() && redis_ok) {
    writeCheckpointSuperblockOnly(shard_id);

    bool replay_ok = true;
    const std::string key = redis->shardIndexKey(shard_id);
    const std::string gkey = redis->globalIndexKey();
    for (const auto &op : ops) {
      if (!op.evicted.empty()) {
        if (!redis->hdel(key, op.evicted)) {
          std::fprintf(stderr, "[light_mem warning] recoverShard: Redis HDEL shard index failed (shard=%zu hash=%s)\n",
                       shard_id, op.evicted.c_str());
          replay_ok = false;
          break;
        }
        (void)redis->hdel(gkey, op.evicted);
        (void)redis->hdel(redis->globalCrcKey(), op.evicted);
      }
      if (!redis->hset(key, op.hash, std::to_string(op.slot_id))) {
        std::fprintf(stderr, "[light_mem warning] recoverShard: Redis HSET shard index failed (shard=%zu hash=%s)\n",
                     shard_id, op.hash.c_str());
        replay_ok = false;
        break;
      }

      if (!redis->hset(redis->globalCrcKey(), op.hash, std::to_string(op.data_crc))) {
        std::fprintf(stderr, "[light_mem warning] recoverShard: Redis HSET crc failed (shard=%zu hash=%s)\n", shard_id,
                     op.hash.c_str());
        replay_ok = false;
        break;
      }

      // Global mapping: only recover if missing; duplicates are intentionally ignored.
      (void)redis->hsetnx(gkey, op.hash, std::to_string(shard_id) + ":" + std::to_string(op.slot_id));
    }
    if (replay_ok) {
      if (!redis->setString(redis->shardSeqKey(shard_id), std::to_string(superblock_seq_[shard_id]))) {
        std::fprintf(stderr, "[light_mem warning] recoverShard: Redis SET shard seq failed (shard=%zu)\n", shard_id);
        replay_ok = false;
      }
    }

    if (replay_ok) {
      truncateJournalToHeader(shard_id);
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
