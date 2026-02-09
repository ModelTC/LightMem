#include "storage/local_storage_engine.h"

#include <unistd.h>

#include "utils/fsync_compat.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cache {
namespace storage {

void LocalStorageEngine::startJournalWorkers() {
  journal_threads_.resize(shard_);
  for (size_t shard_id = 0; shard_id < shard_; shard_id++) {
    journal_threads_[shard_id] = std::thread([this, shard_id] { journalWorkerLoop(shard_id); });
  }
}

void LocalStorageEngine::stopJournalWorkers() {
  for (size_t shard_id = 0; shard_id < shard_; shard_id++) {
    {
      std::lock_guard<std::mutex> lk(*journal_mu_[shard_id]);
      journal_stop_[shard_id] = true;
    }
    journal_cv_[shard_id]->notify_one();
  }
  for (size_t shard_id = 0; shard_id < shard_; shard_id++) {
    if (journal_threads_.size() > shard_id && journal_threads_[shard_id].joinable()) {
      journal_threads_[shard_id].join();
    }
  }
}

void LocalStorageEngine::journalWorkerLoop(size_t shard_id) {
  while (true) {
    std::vector<std::shared_ptr<JournalTask>> batch;
    {
      std::unique_lock<std::mutex> lk(*journal_mu_[shard_id]);
      journal_cv_[shard_id]->wait(lk, [&] { return journal_stop_[shard_id] || !journal_queue_[shard_id].empty(); });
      if (journal_queue_[shard_id].empty()) {
        if (journal_stop_[shard_id]) {
          break;
        }
        continue;
      }
      while (!journal_queue_[shard_id].empty()) {
        batch.emplace_back(journal_queue_[shard_id].front());
        journal_queue_[shard_id].pop_front();
      }
    }

    const bool strict = coordinatedMode();
    bool ok = true;
    try {
      if (strict) {
        {
          std::unique_lock<std::shared_mutex> io_lock(*io_locks_[shard_id]);

          for (auto &task : batch) {
            if (task->epoch != 0 && task->epoch != superblock_epoch_[shard_id]) {
              onEpochChangedLocked(shard_id, task->epoch);
            }
            appendJournalRecord(shard_id, task->write_offset, task->write_len, task->data_crc, task->hash,
                                task->evicted_hash);
            journal_entries_[shard_id]++;
          }

          writeCheckpointSuperblockOnly(shard_id);
        }
      }

      RedisClient *r = redisForShard(shard_id);
      const bool redis_connected = (r && r->connect());
      if (redis_connected) {
        const std::string key = r->shardIndexKey(shard_id);
        const std::string gkey = r->globalIndexKey();

        std::vector<std::vector<std::string>> cmds;
        cmds.reserve(batch.size() * 4 + 1);

        for (auto &task : batch) {
          if (!task->evicted_hash.empty()) {
            cmds.push_back({"HDEL", key, task->evicted_hash});
            cmds.push_back({"HDEL", gkey, task->evicted_hash});
            cmds.push_back({"HDEL", r->globalCrcKey(), task->evicted_hash});
          }
          cmds.push_back({"HSET", key, task->hash, std::to_string(task->slot_id)});
          cmds.push_back({"HSET", gkey, task->hash, std::to_string(shard_id) + ":" + std::to_string(task->slot_id)});
          // Only publish CRC when available (strict mode). Single-node fast mode may use 0.
          if (task->data_crc != 0) {
            cmds.push_back({"HSET", r->globalCrcKey(), task->hash, std::to_string(task->data_crc)});
          }
        }
        if (strict) {
          cmds.push_back({"SET", r->shardSeqKey(shard_id), std::to_string(superblock_seq_[shard_id])});
        }

        if (!r->pipeline(cmds)) {
          std::fprintf(stderr, "[light_mem error] journalWorkerLoop: Redis pipeline failed (shard=%zu, cmds=%zu)\n",
                       shard_id, cmds.size());
          ok = false;
        }
      }

      if (strict) {
        // Keep local WAL bounded even when Redis is down.
        // Only checkpoint after the write queue is drained (to avoid frequent checkpointing
        // in the middle of a steady stream of writes).
        bool queue_empty = false;
        {
          std::lock_guard<std::mutex> lk(*journal_mu_[shard_id]);
          queue_empty = journal_queue_[shard_id].empty();
        }
        if (queue_empty) {
          maybeCheckpoint(shard_id);
        }
      }
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[light_mem error] journalWorkerLoop: exception (shard=%zu): %s\n", shard_id, e.what());
      ok = false;
    } catch (...) {
      std::fprintf(stderr, "[light_mem error] journalWorkerLoop: unknown exception (shard=%zu)\n", shard_id);
      ok = false;
    }

    for (auto &task : batch) {
      {
        std::lock_guard<std::mutex> lk(task->mu);
        task->success = ok;
        task->done = true;
      }
      task->cv.notify_one();
    }
  }

  // Drain any queued tasks (fail them) if shutting down.
  while (true) {
    std::shared_ptr<JournalTask> task;
    {
      std::lock_guard<std::mutex> lk(*journal_mu_[shard_id]);
      if (journal_queue_[shard_id].empty()) {
        break;
      }
      task = journal_queue_[shard_id].front();
      journal_queue_[shard_id].pop_front();
    }
    {
      std::lock_guard<std::mutex> lk(task->mu);
      task->success = false;
      task->done = true;
    }
    task->cv.notify_one();
  }
}

void LocalStorageEngine::appendJournalRecord(size_t shard_id, uint64_t write_offset, uint32_t write_len,
                                             uint32_t data_crc, const std::string &hash,
                                             const std::string &evicted_hash) {
  JournalRecord rec{};
  rec.magic = JOURNAL_MAGIC;
  rec.version = JOURNAL_VERSION;
  rec.flags = 0;
  rec.epoch = superblock_epoch_[shard_id];
  rec.write_offset = write_offset;
  rec.write_len = write_len;
  rec.data_crc = data_crc;
  rec.hash_len = static_cast<uint32_t>(hash.size());
  rec.evicted_hash_len = static_cast<uint32_t>(evicted_hash.size());

  std::vector<uint8_t> tmp;
  tmp.resize(sizeof(JournalRecord) + hash.size() + evicted_hash.size());
  std::memcpy(tmp.data(), &rec, sizeof(JournalRecord));
  if (!hash.empty()) {
    std::memcpy(tmp.data() + sizeof(JournalRecord), hash.data(), hash.size());
  }
  if (!evicted_hash.empty()) {
    std::memcpy(tmp.data() + sizeof(JournalRecord) + hash.size(), evicted_hash.data(), evicted_hash.size());
  }
  const uint32_t record_crc = compute_crc32(tmp.data(), tmp.size());

  off_t end = ::lseek(meta_fds_[shard_id], 0, SEEK_END);
  if (end < 0) {
    const int err = errno;
    throw std::runtime_error(std::string("Failed to seek meta file end, errno=") + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }

  if (!pwriteAll(meta_fds_[shard_id], &rec, sizeof(JournalRecord), end)) {
    const int err = errno;
    throw std::runtime_error(std::string("Failed to append journal header, errno=") + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
  end += static_cast<off_t>(sizeof(JournalRecord));

  if (!hash.empty()) {
    if (!pwriteAll(meta_fds_[shard_id], hash.data(), hash.size(), end)) {
      const int err = errno;
      throw std::runtime_error(std::string("Failed to append journal hash, errno=") + std::to_string(err) +
                               ", reason=" + std::string(::strerror(err)));
    }
    end += static_cast<off_t>(hash.size());
  }

  if (!evicted_hash.empty()) {
    if (!pwriteAll(meta_fds_[shard_id], evicted_hash.data(), evicted_hash.size(), end)) {
      const int err = errno;
      throw std::runtime_error(std::string("Failed to append journal evicted_hash, errno=") + std::to_string(err) +
                               ", reason=" + std::string(::strerror(err)));
    }
    end += static_cast<off_t>(evicted_hash.size());
  }

  if (!pwriteAll(meta_fds_[shard_id], &record_crc, sizeof(uint32_t), end)) {
    const int err = errno;
    throw std::runtime_error(std::string("Failed to append journal crc, errno=") + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
  if (cache::utils::fdatasync_compat(meta_fds_[shard_id]) != 0) {
    const int err = errno;
    throw std::runtime_error(std::string("Failed to fdatasync meta, errno=") + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
}

void LocalStorageEngine::appendEpochMarkerLocked(size_t shard_id, uint64_t epoch) {
  JournalRecord rec{};
  rec.magic = JOURNAL_MAGIC;
  rec.version = JOURNAL_VERSION;
  rec.flags = JOURNAL_FLAG_EPOCH_MARKER;
  rec.epoch = epoch;
  rec.write_offset = 0;
  rec.write_len = 0;
  rec.data_crc = 0;
  rec.hash_len = 0;
  rec.evicted_hash_len = 0;

  std::vector<uint8_t> tmp;
  tmp.resize(sizeof(JournalRecord));
  std::memcpy(tmp.data(), &rec, sizeof(JournalRecord));
  const uint32_t record_crc = compute_crc32(tmp.data(), tmp.size());

  off_t end = ::lseek(meta_fds_[shard_id], 0, SEEK_END);
  if (end < 0) {
    const int err = errno;
    throw std::runtime_error(std::string("Failed to seek meta file end (epoch marker), errno=") +
                             std::to_string(err) + ", reason=" + std::string(::strerror(err)));
  }

  if (!pwriteAll(meta_fds_[shard_id], &rec, sizeof(JournalRecord), end)) {
    const int err = errno;
    throw std::runtime_error(std::string("Failed to append epoch marker, errno=") + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
  end += static_cast<off_t>(sizeof(JournalRecord));

  if (!pwriteAll(meta_fds_[shard_id], &record_crc, sizeof(uint32_t), end)) {
    const int err = errno;
    throw std::runtime_error(std::string("Failed to append epoch marker crc, errno=") + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
  if (cache::utils::fdatasync_compat(meta_fds_[shard_id]) != 0) {
    const int err = errno;
    throw std::runtime_error(std::string("Failed to fdatasync meta (epoch marker), errno=") + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
}

void LocalStorageEngine::onEpochChangedLocked(size_t shard_id, uint64_t new_epoch) {
  // Persist the new epoch into memory and WAL so later scans can ignore stale tail writes.
  superblock_epoch_[shard_id] = new_epoch;
  shard_epoch_cache_[shard_id].store(new_epoch, std::memory_order_relaxed);
  appendEpochMarkerLocked(shard_id, new_epoch);
  writeCheckpointSuperblockOnly(shard_id);
}

void LocalStorageEngine::maybeCheckpoint(size_t shard_id) {
  // Trigger checkpoint based on accumulated write volume (about 1GiB), which scales with block size.
  // NOTE: block_size_ is derived from the cache service block size and is affected by
  static constexpr uint64_t kCheckpointBytes = 1ull * 1024ull * 1024ull * 1024ull; // 1GiB

  const uint64_t bs = static_cast<uint64_t>(block_size_);
  if (bs == 0) {
    return;
  }
  const uint64_t blocks_per_checkpoint = (kCheckpointBytes + bs - 1) / bs; // ceil
  const uint64_t threshold = (blocks_per_checkpoint == 0) ? 1 : blocks_per_checkpoint;

  if (journal_entries_[shard_id] < threshold) {
    return;
  }
  checkpoint(shard_id);
}

void LocalStorageEngine::checkpoint(size_t shard_id) {
  std::unique_lock<std::shared_mutex> lock(*io_locks_[shard_id]);

  // 1. Create Local Index Snapshot
  std::stringstream ss;
  ss << filename_ << "_" << shard_id << "/index";
  std::string snap_path = ss.str();

  if (!caches_[shard_id]->saveToSnapshot(snap_path)) {
    std::fprintf(stderr, "[light_mem warning] checkpoint: failed to save snapshot for shard %zu\n", shard_id);
    // We continue even if snapshot fails? No, if snapshot fails, we shouldn't truncate WAL.
    return;
  }

  SuperBlock sb{};
  sb.magic = SUPERBLOCK_MAGIC;
  sb.version = SUPERBLOCK_VERSION;
  sb.shard_id = shard_id;
  sb.sequence_id = superblock_seq_[shard_id] + 1;
  sb.stable_offset = META_HEADER_SIZE;
  sb.total_capacity = (storage_size_ / shard_) / block_size_;
  sb.block_size = static_cast<uint32_t>(block_size_);
  sb.current_epoch = superblock_epoch_[shard_id];

  const off_t target = (sb.sequence_id % 2 == 1) ? 0 : static_cast<off_t>(SUPERBLOCK_SIZE);
  writeSuperBlockAt(shard_id, target, sb);
  superblock_seq_[shard_id] = sb.sequence_id;
  superblock_stable_offset_[shard_id] = sb.stable_offset;

  // Best-effort: publish local sequence so Redis consumers can detect staleness.
  RedisClient *r = redisForShard(shard_id);
  if (r && r->connect()) {
    (void)r->setString(r->shardSeqKey(shard_id), std::to_string(superblock_seq_[shard_id]));
  }

  if (::ftruncate(meta_fds_[shard_id], META_HEADER_SIZE) != 0) {
    std::fprintf(stderr, "[light_mem warning] checkpoint: ftruncate meta failed for shard %zu (errno=%d %s)\n",
                 shard_id, errno, std::strerror(errno));
    return;
  }
  (void)::fsync(meta_fds_[shard_id]);
  journal_entries_[shard_id] = 0;
}

void LocalStorageEngine::writeCheckpointSuperblockOnly(size_t shard_id) {
  SuperBlock sb{};
  sb.magic = SUPERBLOCK_MAGIC;
  sb.version = SUPERBLOCK_VERSION;
  sb.shard_id = shard_id;
  sb.sequence_id = superblock_seq_[shard_id] + 1;
  sb.stable_offset = META_HEADER_SIZE;
  sb.total_capacity = (storage_size_ / shard_) / block_size_;
  sb.block_size = static_cast<uint32_t>(block_size_);
  sb.current_epoch = superblock_epoch_[shard_id];

  const off_t target = (sb.sequence_id % 2 == 1) ? 0 : static_cast<off_t>(SUPERBLOCK_SIZE);
  writeSuperBlockAt(shard_id, target, sb);
  superblock_seq_[shard_id] = sb.sequence_id;
  superblock_stable_offset_[shard_id] = sb.stable_offset;
}

void LocalStorageEngine::truncateJournalToHeader(size_t shard_id) {
  if (meta_fds_[shard_id] < 0) {
    return;
  }
  if (::ftruncate(meta_fds_[shard_id], META_HEADER_SIZE) != 0) {
    std::fprintf(stderr, "[light_mem warning] truncateJournalToHeader: ftruncate failed for shard %zu (errno=%d %s)\n",
                 shard_id, errno, std::strerror(errno));
    return;
  }
  (void)::fsync(meta_fds_[shard_id]);
  journal_entries_[shard_id] = 0;
}

} // namespace storage
} // namespace cache
