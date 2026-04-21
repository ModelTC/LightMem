#include "storage/local_storage_engine.h"

#include "config.h"

#include <fcntl.h>
#include <unistd.h>

#include "utils/fsync_compat.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace cache {
namespace storage {

std::optional<LocalStorageEngine::LocalShardHintHit> LocalStorageEngine::localShardHint(const std::string &hash) const {
  if (!online_mode_) {
    std::fprintf(stderr, "[light_mem error] localShardHint called in offline mode\n");
    return std::nullopt;
  }

  const size_t b = localHintBucket(hash);
  std::shared_lock<std::shared_mutex> lk(local_hint_mu_[b]);
  auto it = local_hash_to_shard_[b].find(hash);
  if (it == local_hash_to_shard_[b].end()) {
    return std::nullopt;
  }
  const size_t shard_id = it->second;
  if (shard_id >= shard_) {
    return std::nullopt;
  }

  // Acquire inflight inside localShardHint(). This closes the window between a successful hint
  // check and the caller starting the local-hit read path.
  auto inflight_guard = shard_inflight_[shard_id].acquire();

  // In online mode, a shard in draining state is treated as not eligible for local-hit fast path.
  // Re-check after acquiring inflight to avoid returning a hint during/after handoff.
  if (!isShardWritable(shard_id, nullptr)) {
    return std::nullopt;
  }

  return LocalShardHintHit{shard_id, std::move(inflight_guard)};
}

void LocalStorageEngine::noteLocalShardHint(const std::string &hash, size_t shard_id) {
  if (!online_mode_) {
    std::fprintf(stderr, "[light_mem error] noteLocalShardHint called in offline mode\n");
    return;
  }

  if (shard_id >= shard_) {
    return;
  }
  // Only record hints for shards eligible for local-hit fast path.
  if (!isShardWritable(shard_id, nullptr)) {
    return;
  }

  // Fast-path: avoid exclusive lock when the mapping is already up-to-date.
  const size_t b = localHintBucket(hash);
  {
    std::shared_lock<std::shared_mutex> lk(local_hint_mu_[b]);
    auto it = local_hash_to_shard_[b].find(hash);
    if (it != local_hash_to_shard_[b].end() && it->second == shard_id) {
      return;
    }
  }
  std::unique_lock<std::shared_mutex> lk(local_hint_mu_[b]);
  local_hash_to_shard_[b][hash] = shard_id;
}

void LocalStorageEngine::eraseLocalShardHint(const std::string &hash) {
  if (!online_mode_) {
    std::fprintf(stderr, "[light_mem error] eraseLocalShardHint called in offline mode\n");
    return;
  }

  // Fast-path: avoid exclusive lock if the key is absent.
  const size_t b = localHintBucket(hash);
  {
    std::shared_lock<std::shared_mutex> lk(local_hint_mu_[b]);
    if (local_hash_to_shard_[b].find(hash) == local_hash_to_shard_[b].end()) {
      return;
    }
  }
  std::unique_lock<std::shared_mutex> lk(local_hint_mu_[b]);
  local_hash_to_shard_[b].erase(hash);
}

LocalStorageEngine::LocalStorageEngine(const std::string &filename, const size_t storage_size, const size_t shard,
                                       const size_t block_size, const std::string &index_endpoint,
                                       const std::string &index_prefix)
    : filename_(filename), storage_size_(storage_size), shard_(shard), block_size_(block_size),
      online_mode_(!index_endpoint.empty()) {
  // 每个 shard 分到的文件大小
  const size_t shard_storage_size = storage_size_ / shard_;
  // 每个 shard 能存储的块数
  const size_t shard_capacity = shard_storage_size / block_size;

  caches_.resize(shard_);
  io_locks_.resize(shard_);
  file_fds_.resize(shard_, -1);
  meta_fds_.resize(shard_, -1);
  journal_entries_.assign(shard_, 0);
  superblock_seq_.assign(shard_, 0);
  superblock_stable_offset_.assign(shard_, META_HEADER_SIZE);
  superblock_epoch_.assign(shard_, 0);
  shard_recovered_epoch_.assign(shard_, 0);

  shard_writable_ = std::make_unique<std::atomic<uint8_t>[]>(shard_);
  shard_draining_ = std::make_unique<std::atomic<uint8_t>[]>(shard_);
  shard_epoch_cache_ = std::make_unique<std::atomic<uint64_t>[]>(shard_);
  shard_inflight_ = std::make_unique<InflightCounter[]>(shard_);
  shard_written_bytes_ = std::make_unique<std::atomic<uint64_t>[]>(shard_);
  for (size_t i = 0; i < shard_; i++) {
    shard_writable_[i].store(1, std::memory_order_relaxed);
    shard_draining_[i].store(0, std::memory_order_relaxed);
    shard_epoch_cache_[i].store(0, std::memory_order_relaxed);
    shard_inflight_[i].store(0, std::memory_order_relaxed);
    shard_written_bytes_[i].store(0, std::memory_order_relaxed);
  }

  journal_mu_.resize(shard_);
  journal_cv_.resize(shard_);
  journal_queue_.resize(shard_);
  journal_stop_.assign(shard_, false);

  initIndexBackend(index_endpoint, index_prefix);

  // Best-effort startup: if index backend is configured but unreachable, warn and proceed without it.
  if (redis_lock_) {
    if (!redis_lock_->connect()) {
      std::fprintf(stderr, "[light_mem warning] index backend is configured but not reachable; "
                           "continuing without index persistence\n");
      redis_lock_.reset();
      redis_pool_.clear();
    }
  }

  try {
    for (size_t i = 0; i < shard_; i++) {
      caches_[i] = std::make_shared<LocalCacheIndex>(shard_capacity);
      io_locks_[i] = std::make_shared<std::shared_mutex>();
      journal_mu_[i] = std::make_unique<std::mutex>();
      journal_cv_[i] = std::make_unique<std::condition_variable>();
    }

    // Make localShardHint authoritative: update hint under the same LocalCacheIndex mutex
    // for any ready insertions and removals/evictions.
    if (online_mode_) {
      for (size_t i = 0; i < shard_; i++) {
        caches_[i]->set_hooks(
            [this, i](const std::string &hash) { noteLocalShardHint(hash, i); },
            [this](const std::string &hash) { eraseLocalShardHint(hash); });
      }
    }
    createOrOpenFiles(shard_storage_size);
    if (!online_mode_) {
      recoverAllShards(shard_capacity);
    }
    startJournalWorkers();
  } catch (...) {
    cleanup();
    throw;
  }
}

LocalStorageEngine::~LocalStorageEngine() {
  stopJournalWorkers();
  cleanup();
}

std::vector<bool> LocalStorageEngine::queryMany(const std::vector<std::string> &hashs) {
  std::vector<bool> ret;
  ret.assign(hashs.size(), false);

  // Offline mode: local lookup only.
  if (!online_mode_) {
    for (size_t i = 0; i < hashs.size(); i++) {
      const std::string &hash = hashs[i];
      const size_t shard_id = getShard(hash);
      ret[i] = caches_[shard_id]->exists(hash);
    }
    return ret;
  }

  // First pass: local hint + shard-local index.
  // We only treat it as a hit without Redis re-validation if the shard is currently owned.
  std::vector<std::string> need_redis_hash;
  std::vector<size_t> need_redis_idx;
  need_redis_hash.reserve(hashs.size());
  need_redis_idx.reserve(hashs.size());

  for (size_t i = 0; i < hashs.size(); i++) {
    const std::string &hash = hashs[i];
    if (auto hinted = localShardHint(hash); hinted.has_value() && hinted->shard_id < shard_) {
      const size_t sid = hinted->shard_id;
      if (sid < shard_ && caches_[sid] && caches_[sid]->exists(hash)) {
        ret[i] = true;
        continue;
      }
      eraseLocalShardHint(hash);
    }
    need_redis_hash.emplace_back(hash);
    need_redis_idx.emplace_back(i);
  }

  if (need_redis_hash.empty()) {
    return ret;
  }

  // Second pass: Redis global index (batched).
  if (redis_lock_ && redis_lock_->connect()) {
    auto vals = redis_lock_->hmget(redis_lock_->globalIndexKey(), need_redis_hash);
    if (vals.has_value() && vals->size() == need_redis_hash.size()) {
      for (size_t j = 0; j < vals->size(); j++) {
        const std::string &s = (*vals)[j];
        if (s.empty()) {
          continue;
        }
        const auto pos = s.find(':');
        if (pos == std::string::npos) {
          continue;
        }
        try {
          const size_t shard_id = static_cast<size_t>(std::stoull(s.substr(0, pos)));
          if (shard_id >= shard_) {
            continue;
          }
          ret[need_redis_idx[j]] = true;
        } catch (...) {
          continue;
        }
      }
      return ret;
    }
  }
  return ret;
}

std::optional<size_t> LocalStorageEngine::handleExistingLocalDuplicate(size_t shard_id, const std::string &hash,
                                                                       uint32_t data_crc, int &result,
                                                                       size_t &slot_id, std::string &evicted_hash,
                                                                       bool do_global_dedupe,
                                                                       const std::string &global_key) {
  if (result != 0) {
    return std::nullopt;
  }

  size_t existing_slot = 0;
  uint32_t existing_crc = 0;
  if (caches_[shard_id]->get_offset_and_crc(hash, existing_slot, existing_crc)) {
    if (existing_crc != 0 && data_crc != 0 && existing_crc != data_crc) {
      caches_[shard_id]->remove(hash);

      result = -1;
      evicted_hash.clear();
      for (int attempt = 0; attempt < 64; ++attempt) {
        result = caches_[shard_id]->acquire_slot(hash, slot_id, evicted_hash);
        if (result >= 0) {
          break;
        }
        std::this_thread::yield();
      }
      if (result < 0) {
        return static_cast<size_t>(0);
      }
    } else if (existing_crc == 0 && data_crc != 0) {
      caches_[shard_id]->mark_ready(hash, data_crc);
    }
  }

  if (result == 0) {
    if (do_global_dedupe) {
      size_t duplicate_slot = 0;
      uint32_t duplicate_crc = 0;
      if (caches_[shard_id]->get_offset_and_crc(hash, duplicate_slot, duplicate_crc)) {
        const std::string shard_slot = std::to_string(shard_id) + ":" + std::to_string(duplicate_slot);
        std::vector<std::vector<std::string>> cmds;
        cmds.reserve(duplicate_crc != 0 ? 3 : 2);
        cmds.push_back({"HSET", redis_lock_->shardIndexKey(shard_id), hash, std::to_string(duplicate_slot)});
        cmds.push_back({"HSET", global_key, hash, shard_slot});
        if (duplicate_crc != 0) {
          cmds.push_back({"HSET", redis_lock_->globalCrcKey(), hash, std::to_string(duplicate_crc)});
        }
        (void)redis_lock_->pipeline(cmds);
      }
    }
    return block_size_;
  }

  return std::nullopt;
}

size_t LocalStorageEngine::write(const char *buf, const std::string &hash, uint32_t data_crc, uint32_t len_bytes) {
  if (len_bytes == 0 || static_cast<size_t>(len_bytes) > block_size_) {
    return 0;
  }

  // Offline mode (local cache):
  // Online mode requires durability/consistency (WAL + CRC + fsync + fencing).
  if (!online_mode_) {
    const size_t shard_id = getShard(hash);
    if (shard_id >= shard_) {
      return 0;
    }

    size_t slot_id = 0;
    std::string evicted_hash;

    // Acquire a slot. In offline mode we should not silently drop writes due to temporary congestion.
    // Retry briefly to wait for other in-flight writes to finish.
    int result = -1;
    for (int attempt = 0; attempt < 64; ++attempt) {
      result = caches_[shard_id]->acquire_slot(hash, slot_id, evicted_hash);
      if (result >= 0) {
        break;
      }
      std::this_thread::yield();
    }
    if (result < 0) {
      return 0;
    }
    if (result == 0) {
      return block_size_; // Local duplicate: data already on disk and ready.
    }

    const size_t offset_bytes = slot_id * block_size_;
    try {
      std::unique_lock<std::shared_mutex> lock(*io_locks_[shard_id]);
      // Offline hot-path: only write the valid logical bytes, not the full block.
      // For a 1-page write this reduces I/O from ~64 MB to ~1 MB.
      if (!pwriteAll(file_fds_[shard_id], buf, static_cast<size_t>(len_bytes), static_cast<off_t>(offset_bytes))) {
        throw std::runtime_error("pwrite failed, errno=" + std::to_string(errno));
      }
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[light_mem error] write(offline): I/O failed (shard=%zu hash=%s): %s\n",
                   shard_id, hash.c_str(), e.what());
      caches_[shard_id]->remove(hash);
      return 0;
    }

    // Mark ready immediately after the data write completes.
    caches_[shard_id]->mark_ready(hash, 0);

    // Offline mode: also enqueue journal metadata asynchronously so restart recovery
    // can rebuild from snapshot + WAL without blocking the write hot path.
    auto task = std::make_shared<JournalTask>();
    task->shard_id = shard_id;
    task->epoch = 0;
    task->write_offset = static_cast<uint64_t>(offset_bytes);
    task->write_len = len_bytes;
    task->data_crc = 0;
    task->slot_id = slot_id;
    task->hash = hash;
    task->evicted_hash = evicted_hash;
    {
      std::lock_guard<std::mutex> lk(*journal_mu_[shard_id]);
      journal_queue_[shard_id].push_back(task);
    }
    journal_cv_[shard_id]->notify_one();

#ifndef __APPLE__
    // Linux regression fix: keep offline write path aligned with historical behavior.
    // Drop newly written data pages from page cache to avoid polluting subsequent read cache.
    (void)posix_fadvise(file_fds_[shard_id], static_cast<off_t>(offset_bytes), static_cast<off_t>(len_bytes),
                        POSIX_FADV_DONTNEED);
#endif

    shard_written_bytes_[shard_id].fetch_add(static_cast<uint64_t>(block_size_), std::memory_order_relaxed);

    return block_size_;
  }

  const bool redis_connected = (redis_lock_ && redis_lock_->connect());
  // Global dedupe/locking is required in online mode.
  const bool do_global_dedupe = redis_connected;
  const std::string global_key = do_global_dedupe ? redis_lock_->globalIndexKey() : std::string();
  const std::string lock_key = do_global_dedupe ? redis_lock_->hashLockKey(hash) : std::string();

  // Track whether we hold the distributed Redis lock for this hash.
  bool lock_held = false;
  if (do_global_dedupe) {
    // 返回true就跳过写入(去重), 返回false继续写入并覆盖原数据
    auto shouldDedupeGlobalHit = [&](const std::string &mapping) -> bool {
      const auto pos = mapping.find(':');
      if (pos == std::string::npos) {
        return false;
      }

      size_t mapped_shard = 0;
      try {
        mapped_shard = static_cast<size_t>(std::stoull(mapping.substr(0, pos)));
      } catch (...) {
        return false;
      }

      if (mapped_shard >= shard_) {
        return false;
      }

      // If this node doesn't currently own/write the shard, keep global dedupe behavior.
      if (!isShardWritable(mapped_shard, nullptr)) {
        return true;
      }

      size_t local_slot = 0;
      uint32_t local_crc = 0;
      if (!caches_[mapped_shard]->get_offset_and_crc(hash, local_slot, local_crc)) {
        return false;
      }

      if (data_crc != 0 && local_crc != 0 && local_crc != data_crc) {
        return false;
      }

      return true;
    };

    if (auto cur = redis_lock_->hget(global_key, hash); cur.has_value() && shouldDedupeGlobalHit(*cur)) {
      return 0;
    }

    if (!redis_lock_->setStringNxPx(lock_key, "1", 30000)) {
      return 0;
    }
    lock_held = true;

    // Re-check under lock to close races with concurrent writers.
    if (auto cur = redis_lock_->hget(global_key, hash); cur.has_value() && shouldDedupeGlobalHit(*cur)) {
      return 0;
    }
  }

  // RAII: release the distributed Redis lock on any exit path.
  struct LockRelease {
    bool &held;
    RedisClient *client;
    const std::string &key;
    ~LockRelease() {
      if (held && client) {
        (void)client->del(key);
      }
    }
  } lock_release{lock_held, redis_lock_.get(), lock_key};

  const size_t shard_id = pickWritableShard(hash);
  if (shard_id >= shard_) {
    return 0;
  }

  auto inflight_guard = shard_inflight_[shard_id].acquire();
  uint64_t epoch = 0;
  if (!isShardWritable(shard_id, &epoch)) {
    return 0;
  }

  size_t slot_id = 0;
  std::string evicted_hash;

  // 1. Acquire slot (LRU)
  int result = -1;
  for (int attempt = 0; attempt < 64; ++attempt) {
    result = caches_[shard_id]->acquire_slot(hash, slot_id, evicted_hash);
    if (result >= 0) {
      break;
    }
    // Temporary congestion: other in-flight writers hold all slots. Yield and retry.
    std::this_thread::yield();
  }
  if (result < 0) {
    return 0;
  }

  if (auto duplicate_result =
          handleExistingLocalDuplicate(shard_id, hash, data_crc, result, slot_id, evicted_hash, do_global_dedupe,
                                       global_key);
      duplicate_result.has_value()) {
    return *duplicate_result;
  }

  // If LRU evicted something, delete its Redis mappings *before* we overwrite the slot.
  // This avoids a window where stale Redis mapping points to overwritten data.
  if (do_global_dedupe && !evicted_hash.empty()) {
    // NOTE: best-effort; a Redis failure here can lead to stale mapping.
    (void)redis_lock_->pipeline({
        {"HDEL", redis_lock_->shardIndexKey(shard_id), evicted_hash},
        {"HDEL", global_key, evicted_hash},
        {"HDEL", redis_lock_->globalCrcKey(), evicted_hash},
    });
  }

  // 2. Write data to data file (overwrite).
  const size_t write_bytes = block_size_;
  const size_t offset_bytes = slot_id * block_size_;
  try {
    std::unique_lock<std::shared_mutex> lock(*io_locks_[shard_id]);
    if (!pwriteAll(file_fds_[shard_id], buf, write_bytes, static_cast<off_t>(offset_bytes))) {
      throw std::runtime_error("pwrite failed, errno=" + std::to_string(errno));
    }
    if (cache::utils::fdatasync_compat(file_fds_[shard_id]) != 0) {
      throw std::runtime_error("fdatasync failed, errno=" + std::to_string(errno));
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[light_mem error] write: I/O failed (shard=%zu hash=%s): %s\n",
                 shard_id, hash.c_str(), e.what());
    caches_[shard_id]->remove(hash);
    return 0;
  }

  // 3. Enqueue journal+redis update; handled by the per-shard journal worker.
  // CRC is computed by the caller over `len_bytes` to avoid hashing unused tail bytes.
  auto task = std::make_shared<JournalTask>();
  task->shard_id = shard_id;
  task->epoch = epoch;
  task->write_offset = static_cast<uint64_t>(offset_bytes);
  task->write_len = len_bytes;
  task->data_crc = data_crc;
  task->slot_id = slot_id;
  task->hash = hash;
  task->evicted_hash = evicted_hash;

  {
    std::lock_guard<std::mutex> lk(*journal_mu_[shard_id]);
    journal_queue_[shard_id].push_back(task);
  }
  journal_cv_[shard_id]->notify_one();

  // Synchronous commit: wait until journal worker makes the record durable.
  {
    std::unique_lock<std::mutex> lk(task->mu);
    task->cv.wait(lk, [&] { return task->done; });
  }

  if (!task->success) {
    caches_[shard_id]->remove(hash);
    return 0;
  }
  shard_written_bytes_[shard_id].fetch_add(static_cast<uint64_t>(write_bytes), std::memory_order_relaxed);

#ifndef __APPLE__
  posix_fadvise(file_fds_[shard_id], offset_bytes, write_bytes, POSIX_FADV_DONTNEED);
#endif

  return write_bytes;
}

size_t LocalStorageEngine::read(char *buf, const std::string &hash, uint32_t len_bytes) {
  if (len_bytes == 0 || static_cast<size_t>(len_bytes) > block_size_) {
    return 0;
  }

  // Offline mode: local lookup only.
  if (!online_mode_) {
    const size_t shard_id = getShard(hash);
    const size_t slot_id = caches_[shard_id]->get_offset(hash);
    if (slot_id == static_cast<size_t>(-1)) {
      return 0;
    }
    std::shared_lock<std::shared_mutex> lock(*io_locks_[shard_id]);
    if (caches_[shard_id]->get_offset(hash) != slot_id) {
      return 0;
    }
    const size_t offset_bytes = slot_id * block_size_;
    if (!preadAll(file_fds_[shard_id], buf, static_cast<size_t>(len_bytes), static_cast<off_t>(offset_bytes))) {
      std::fprintf(stderr, "[light_mem error] read: I/O error for hash %s\n", hash.c_str());
      return 0;
    }
    return static_cast<size_t>(len_bytes);
  }

  // 1) Local-hit path (authoritative local hint). No CRC validation needed.
  if (auto hinted = localShardHint(hash); hinted.has_value() && hinted->shard_id < shard_) {
    const size_t i = hinted->shard_id;
    std::shared_lock<std::shared_mutex> lock(*io_locks_[i]);
    const size_t slot_id = caches_[i]->get_offset(hash);
    if (slot_id != static_cast<size_t>(-1)) {
      const size_t offset_bytes = slot_id * block_size_;
      if (!preadAll(file_fds_[i], buf, static_cast<size_t>(len_bytes), static_cast<off_t>(offset_bytes))) {
        std::fprintf(stderr, "[light_mem error] read: I/O error for hash %s\n", hash.c_str());
        return 0;
      }
      return static_cast<size_t>(len_bytes);
    }
    eraseLocalShardHint(hash);
  }

  // 2) Redis-resolved path (with CRC validation).
  if (!redis_lock_ || !redis_lock_->connect()) {
    return 0;
  }

  size_t slot_id = static_cast<size_t>(-1);
  auto resolved = findShardInRedis(hash, &slot_id);
  if (!resolved.has_value() || slot_id == static_cast<size_t>(-1)) {
    return 0;
  }
  const size_t shard_id = *resolved;
  if (shard_id >= shard_) {
    return 0;
  }

  // Fetch and verify CRC.
  uint32_t expected_crc = 0;
  auto crc_s = redis_lock_->hget(redis_lock_->globalCrcKey(), hash);
  if (crc_s.has_value() && !crc_s->empty()) {
    try { expected_crc = static_cast<uint32_t>(std::stoul(*crc_s)); } catch (...) {}
  }

  if (expected_crc != 0) {
    const size_t offset_bytes = slot_id * block_size_;
    if (!preadAll(file_fds_[shard_id], buf, static_cast<size_t>(len_bytes), static_cast<off_t>(offset_bytes))) {
      std::fprintf(stderr, "[light_mem error] read: I/O error for hash %s (shard=%zu slot=%zu)\n",
                   hash.c_str(), shard_id, slot_id);
      return 0;
    }
    if (compute_crc32(buf, static_cast<size_t>(len_bytes)) == expected_crc) {
      return static_cast<size_t>(len_bytes);
    }
  }

  // CRC unavailable or mismatch: clean up stale Redis mappings.
  (void)redis_lock_->pipeline({
      {"HDEL", redis_lock_->globalIndexKey(), hash},
      {"HDEL", redis_lock_->shardIndexKey(shard_id), hash},
      {"HDEL", redis_lock_->globalCrcKey(), hash},
  });
  return 0;
}

std::shared_ptr<LocalStorageEngine::HashInfo> LocalStorageEngine::getHashInfo() {
  auto info = std::make_shared<HashInfo>();
  info->caches = caches_;
  info->io_locks = io_locks_;
  return info;
}

bool LocalStorageEngine::setHashInfo(const std::shared_ptr<HashInfo> &info) {
  if (!info) {
    std::fprintf(stderr, "[light_mem error] setHashInfo: HashInfo is null\n");
    return false;
  }
  if (info->caches.size() != shard_ || info->io_locks.size() != shard_) {
    std::fprintf(stderr,
                 "[light_mem error] setHashInfo: shard size mismatch (expected %zu, got caches=%zu io_locks=%zu)\n",
                 shard_, info->caches.size(), info->io_locks.size());
    return false;
  }
  caches_ = info->caches;
  io_locks_ = info->io_locks;
  return true;
}

} // namespace storage
} // namespace cache
