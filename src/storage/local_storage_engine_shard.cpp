#include "storage/local_storage_engine.h"

#include <cstdio>
#include <cstdlib>

namespace cache {
namespace storage {

namespace {

// Small, fast 64-bit mixer (SplitMix64). Suitable for generating a pseudo-random
// probe sequence from a stable seed.
static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

} // namespace

size_t LocalStorageEngine::getShard(const std::string &hash) const { return std::hash<std::string>{}(hash) % shard_; }

bool LocalStorageEngine::isShardWritable(size_t shard_id, uint64_t *epoch_out) const {
  if (shard_id >= shard_) {
    return false;
  }
  if (shard_writable_[shard_id].load(std::memory_order_relaxed) == 0) {
    return false;
  }
  if (shard_draining_[shard_id].load(std::memory_order_relaxed) != 0) {
    return false;
  }
  if (epoch_out) {
    *epoch_out = shard_epoch_cache_[shard_id].load(std::memory_order_relaxed);
  }
  return true;
}

std::optional<size_t> LocalStorageEngine::findShardInRedis(const std::string &hash, size_t *slot_id_out) {
  if (!redis_lock_ || !redis_lock_->connect()) {
    return std::nullopt;
  }
  const auto v = redis_lock_->hget(redis_lock_->globalIndexKey(), hash);
  if (!v.has_value()) {
    return std::nullopt;
  }
  const auto &s = *v;
  const auto pos = s.find(':');
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  try {
    const size_t shard_id = static_cast<size_t>(std::stoull(s.substr(0, pos)));
    const size_t slot_id = static_cast<size_t>(std::stoull(s.substr(pos + 1)));
    if (shard_id >= shard_) {
      return std::nullopt;
    }
    if (slot_id_out) {
      *slot_id_out = slot_id;
    }
    return shard_id;
  } catch (...) {
    return std::nullopt;
  }
}

size_t LocalStorageEngine::pickWritableShard(const std::string &hash) const {
  if (shard_ == 0) {
    return 0;
  }

  const size_t preferred = getShard(hash);
  if (isShardWritable(preferred, nullptr)) {
    return preferred;
  }

  // Seed from std::hash (process-stable). We only need per-process stability here.
  uint64_t seed = static_cast<uint64_t>(std::hash<std::string>{}(hash));
  seed = splitmix64(seed ^ 0x6a09e667f3bcc909ull);

  // Fast-path: try a small number of randomized probes (expected to succeed quickly when a
  // non-trivial fraction of shards are writable).
  const size_t max_probes = (shard_ < 16) ? shard_ : 16;
  for (size_t i = 0; i < max_probes; i++) {
    seed = splitmix64(seed);
    const size_t sid = static_cast<size_t>(seed % static_cast<uint64_t>(shard_));
    if (isShardWritable(sid, nullptr)) {
      return sid;
    }
  }

  // Slow-path: guarantee progress.
  for (size_t sid = 0; sid < shard_; sid++) {
    if (isShardWritable(sid, nullptr)) {
      return sid;
    }
  }
  return shard_;
}

void LocalStorageEngine::updateShardAssignments(const std::vector<size_t> &shard_ids,
                                                const std::vector<uint64_t> &epochs,
                                                const std::vector<uint8_t> &draining) {
  // If index backend is disabled/unavailable, treat as single-node mode.
  // Avoid enabling strict multi-node semantics and avoid doing shard recovery work.
  if (!online_mode_ || !redis_lock_) {
    return;
  }

  // Once the coordinator starts pushing assignments, we must use strict multi-node semantics.
  coordinated_mode_.store(1, std::memory_order_relaxed);

  if (shard_ids.size() != epochs.size() || shard_ids.size() != draining.size()) {
    std::fprintf(stderr, "[light_mem error] updateShardAssignments: size mismatch (ids=%zu epochs=%zu draining=%zu)\n",
                 shard_ids.size(), epochs.size(), draining.size());
    return;
  }

  // Build desired state.
  std::vector<uint8_t> desired_writable(shard_, 0);
  std::vector<uint8_t> desired_draining(shard_, 1); // default: treat as draining (no new writes)
  std::vector<uint64_t> desired_epoch(shard_, 0);

  for (size_t i = 0; i < shard_ids.size(); i++) {
    const size_t sid = shard_ids[i];
    if (sid >= shard_) {
      continue;
    }
    desired_writable[sid] = 1;
    desired_draining[sid] = (draining[i] != 0) ? 1 : 0;
    desired_epoch[sid] = epochs[i];
  }

  const size_t shard_capacity = (storage_size_ / shard_) / block_size_;

  for (size_t sid = 0; sid < shard_; sid++) {
    const uint8_t new_w = desired_writable[sid];
    const uint8_t new_d = desired_draining[sid];
    const uint64_t new_e = desired_epoch[sid];

    const uint64_t old_e = shard_epoch_cache_[sid].load(std::memory_order_relaxed);
    const uint8_t old_w = shard_writable_[sid].load(std::memory_order_relaxed);

    shard_writable_[sid].store(new_w, std::memory_order_relaxed);
    shard_draining_[sid].store(new_d, std::memory_order_relaxed);
    shard_epoch_cache_[sid].store(new_e, std::memory_order_relaxed);

    // Persist epoch changes for shards that are (now) writable.
    if (new_w != 0 && new_e != 0 && new_e != old_e) {
      std::unique_lock<std::shared_mutex> io_lock(*io_locks_[sid]);
      try {
        onEpochChangedLocked(sid, new_e);
      } catch (const std::exception &e) {
        std::fprintf(stderr, "[light_mem warning] updateShardAssignments: failed to persist epoch for shard %zu: %s\n",
                     sid, e.what());
      } catch (...) {
        std::fprintf(
            stderr,
            "[light_mem warning] updateShardAssignments: failed to persist epoch for shard %zu: unknown error\n", sid);
      }
    }

    // If shard ownership is lost, clear local index and related hints.
    if (old_w != 0 && new_w == 0) {
      if (caches_[sid]) {
        caches_[sid]->reset();
      }
      shard_recovered_epoch_[sid] = 0;
    }

    // In online mode, rebuild local index only for owned/writable shards when ownership changes.
    if (online_mode_ && new_w != 0 && new_e != 0) {
      const bool need_recover = (old_w == 0) || (new_e != old_e) || (shard_recovered_epoch_[sid] != new_e);
      if (need_recover) {
        try {
          recoverShard(sid, shard_capacity);
          shard_recovered_epoch_[sid] = new_e;
        } catch (...) {
          // Best-effort: leave shard_recovered_epoch_ unchanged on failures.
        }
      }
    }
  }
}

uint32_t LocalStorageEngine::shardInflight(size_t shard_id) const {
  if (shard_id >= shard_) {
    return 0;
  }
  return shard_inflight_[shard_id].load(std::memory_order_relaxed);
}

uint64_t LocalStorageEngine::shardWrittenBytes(size_t shard_id) const {
  if (shard_id >= shard_) {
    return 0;
  }
  return shard_written_bytes_[shard_id].load(std::memory_order_relaxed);
}

uint64_t LocalStorageEngine::shardEvictionCount(size_t shard_id) const {
  if (shard_id >= shard_) {
    return 0;
  }
  const auto &c = caches_[shard_id];
  if (!c) {
    return 0;
  }
  return c->eviction_count();
}

uint64_t LocalStorageEngine::evictionCount() const {
  uint64_t total = 0;
  for (size_t i = 0; i < shard_; i++) {
    const auto &c = caches_[i];
    if (!c) {
      continue;
    }
    total += c->eviction_count();
  }
  return total;
}

bool LocalStorageEngine::evictionObserved() const { return evictionCount() > 0; }

size_t LocalStorageEngine::ownedShardCount() const {
  size_t owned = 0;
  for (size_t i = 0; i < shard_; i++) {
    if (shard_writable_[i].load(std::memory_order_relaxed) != 0) {
      owned++;
    }
  }
  return owned;
}

uint64_t LocalStorageEngine::effectiveWritableCapacityBytes() const {
  if (shard_ == 0) {
    return 0;
  }

  size_t writable = 0;
  for (size_t i = 0; i < shard_; i++) {
    if (isShardWritable(i, nullptr)) {
      writable++;
    }
  }

  const uint64_t per_shard = static_cast<uint64_t>(storage_size_ / shard_);
  const uint64_t w = static_cast<uint64_t>(writable);
  // Best-effort overflow guard.
  if (per_shard != 0 && w > (std::numeric_limits<uint64_t>::max() / per_shard)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return per_shard * w;
}

} // namespace storage
} // namespace cache
