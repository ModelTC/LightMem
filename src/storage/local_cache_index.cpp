#include "storage/local_cache_index.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <vector>

namespace cache {
namespace storage {

static const uint32_t SNAPSHOT_MAGIC = 0x534E4150; // SNAP
static const uint32_t SNAPSHOT_VERSION = 1;

LocalCacheIndex::LocalCacheIndex(size_t capacity) : capacity_(capacity) {
  // Allocate low offsets first.
  // empty_block_list_.back() is used as the next free slot; so we push in reverse order
  // and pop from the back to get 0, 1, 2, ...
  for (size_t i = capacity; i > 0; --i) {
    empty_block_list_.push_back(i - 1);
  }
}

void LocalCacheIndex::set_hooks(Hook on_ready, Hook on_erase) {
  std::lock_guard<std::mutex> lock(index_lock_);
  on_ready_ = std::move(on_ready);
  on_erase_ = std::move(on_erase);
}

void LocalCacheIndex::reset() {
  std::lock_guard<std::mutex> lock(index_lock_);

  if (on_erase_) {
    for (const auto &kv : index_) {
      on_erase_(kv.first);
    }
  }

  lru_list_.clear();
  index_.clear();
  empty_block_list_.clear();
  eviction_count_ = 0;
  for (size_t i = capacity_; i > 0; --i) {
    empty_block_list_.push_back(i - 1);
  }
}

void LocalCacheIndex::put_ready(const std::string &hash, size_t slot_id, uint32_t crc) {
  std::lock_guard<std::mutex> lock(index_lock_);

  if (slot_id >= capacity_) {
    return;
  }

  // If hash already exists, just update slot and promote.
  auto it = index_.find(hash);
  if (it != index_.end()) {
    it->second.slot_id = slot_id;
    it->second.ready = true;
    it->second.writing = false;
    it->second.crc = crc;
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
    if (on_ready_) {
      on_ready_(hash);
    }
    return;
  }

  // If slot is already used by someone else, evict that hash.
  for (auto map_it = index_.begin(); map_it != index_.end(); ++map_it) {
    if (map_it->second.slot_id == slot_id) {
      if (on_erase_) {
        on_erase_(map_it->first);
      }
      lru_list_.erase(map_it->second.lru_iterator);
      index_.erase(map_it);
      break;
    }
  }

  // Ensure slot is not in empty list.
  for (auto e = empty_block_list_.begin(); e != empty_block_list_.end(); ++e) {
    if (*e == slot_id) {
      empty_block_list_.erase(e);
      break;
    }
  }

  // If over capacity (shouldn't happen if slot_id is from valid range), evict LRU.
  if (index_.size() >= capacity_ && !lru_list_.empty()) {
    const std::string victim = lru_list_.back();
    auto vit = index_.find(victim);
    if (vit != index_.end()) {
      if (on_erase_) {
        on_erase_(victim);
      }
      size_t freed = vit->second.slot_id;
      lru_list_.pop_back();
      index_.erase(vit);
      empty_block_list_.push_back(freed);
    }
  }

  lru_list_.push_front(hash);
  index_[hash] = {lru_list_.begin(), slot_id, true, false, crc};

  if (on_ready_) {
    on_ready_(hash);
  }
}

bool LocalCacheIndex::exists(const std::string &hash) {
  std::lock_guard<std::mutex> lock(index_lock_);
  auto it = index_.find(hash);
  // Only return true if data is fully written and readable (not just allocated)
  if (it == index_.end() || !it->second.ready || it->second.writing) {
    return false;
  }
  lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
  return true;
}

int LocalCacheIndex::acquire_slot(const std::string &hash, size_t &slot_id, std::string &evicted_hash) {
  std::lock_guard<std::mutex> lock(index_lock_);

  auto existing = index_.find(hash);
  if (existing != index_.end()) {
    if (existing->second.writing) {
      return -1; // Write in progress, caller should retry
    }
    lru_list_.splice(lru_list_.begin(), lru_list_, existing->second.lru_iterator);
    slot_id = existing->second.slot_id;
    evicted_hash.clear();
    return 0; // Already exists and ready
  }

  if (!empty_block_list_.empty()) {
    slot_id = empty_block_list_.back();
    empty_block_list_.pop_back();
    evicted_hash.clear();
  } else {
    // Need to evict a victim from LRU list
    // Find a victim that is not currently being written (writing=false)
    auto it = lru_list_.end();
    bool found_victim = false;

    while (it != lru_list_.begin()) {
      --it;
      const std::string &candidate_hash = *it;
      auto candidate_it = index_.find(candidate_hash);

      // Data structure inconsistency detected, return failure to avoid corruption
      if (candidate_it == index_.end()) {
        std::fprintf(stderr, "[light_mem error] LRU list and index inconsistent, returning failure\n");
        return -1;
      }

      if (!candidate_it->second.writing) {
        slot_id = candidate_it->second.slot_id;
        evicted_hash = candidate_hash;

        if (on_erase_) {
          on_erase_(candidate_hash);
        }

        lru_list_.erase(it);
        index_.erase(candidate_it);
        eviction_count_++;
        found_victim = true;
        break;
      }
    }

    if (!found_victim) {
      // All slots are busy writing or LRU list is empty
      // This is temporary congestion, return -1 to let caller retry
      return -1;
    }
  }

  lru_list_.push_front(hash);
  // Mark as writing=true, ready=false
  index_[hash] = {lru_list_.begin(), slot_id, false, true, 0};
  return 1;
}

uint64_t LocalCacheIndex::eviction_count() const {
  std::lock_guard<std::mutex> lock(index_lock_);
  return eviction_count_;
}

bool LocalCacheIndex::eviction_observed() const { return eviction_count() > 0; }

void LocalCacheIndex::mark_ready(const std::string &hash, uint32_t crc) {
  std::lock_guard<std::mutex> lock(index_lock_);
  auto it = index_.find(hash);
  if (it != index_.end()) {
    it->second.ready = true;
    it->second.writing = false;
    if (crc != 0) {
      it->second.crc = crc;
    }

    if (on_ready_) {
      on_ready_(hash);
    }
  }
}

void LocalCacheIndex::remove(const std::string &hash) {
  std::lock_guard<std::mutex> lock(index_lock_);
  auto it = index_.find(hash);
  if (it != index_.end()) {
    if (on_erase_) {
      on_erase_(hash);
    }
    size_t slot_id = it->second.slot_id;
    lru_list_.erase(it->second.lru_iterator);
    index_.erase(it);
    empty_block_list_.push_back(slot_id);
  }
}

size_t LocalCacheIndex::get_offset(const std::string &hash) {
  std::lock_guard<std::mutex> lock(index_lock_);
  auto it = index_.find(hash);
  if (it == index_.end() || !it->second.ready || it->second.writing) {
    return static_cast<size_t>(-1);
  }
  return it->second.slot_id;
}

bool LocalCacheIndex::get_offset_and_crc(const std::string &hash, size_t &slot_id, uint32_t &crc) {
  std::lock_guard<std::mutex> lock(index_lock_);
  auto it = index_.find(hash);
  if (it == index_.end() || !it->second.ready || it->second.writing) {
    return false;
  }
  slot_id = it->second.slot_id;
  crc = it->second.crc;
  return true;
}

void LocalCacheIndex::dump_ready(std::vector<std::string> &out) {
  std::lock_guard<std::mutex> lock(index_lock_);
  out.clear();
  out.reserve(index_.size());
  for (const auto &kv : index_) {
    const auto &e = kv.second;
    if (e.ready && !e.writing) {
      out.emplace_back(kv.first);
    }
  }
}

bool LocalCacheIndex::saveToSnapshot(const std::string &filename) {
  std::lock_guard<std::mutex> lock(index_lock_);

  std::string tmp_filename = filename + ".tmp";
  // Shared storage: allow other nodes/users to read+write snapshots.
  int fd = ::open(tmp_filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
  if (fd < 0) {
    std::fprintf(stderr, "[light_mem warning] snapshot save: open failed (file=%s errno=%d %s)\n", filename.c_str(),
                 errno, std::strerror(errno));
    return false;
  }

  // Best-effort: umask may have reduced permission bits.
  (void)::fchmod(fd, 0666);

  // Header: Magic(4) + Version(4) + Count(8)
  uint32_t magic = SNAPSHOT_MAGIC;
  uint32_t version = SNAPSHOT_VERSION;
  uint64_t count = 0;

  // Count valid entries (ready=true)
  for (const auto &kv : index_) {
    if (kv.second.ready && !kv.second.writing) {
      count++;
    }
  }

  auto write_all = [&](const void *p, size_t n) -> bool {
    const char *buf = static_cast<const char *>(p);
    size_t left = n;
    while (left > 0) {
      ssize_t w = ::write(fd, buf, left);
      if (w <= 0) {
        return false;
      }
      buf += static_cast<size_t>(w);
      left -= static_cast<size_t>(w);
    }
    return true;
  };

  if (!write_all(&magic, sizeof(magic)) || !write_all(&version, sizeof(version)) || !write_all(&count, sizeof(count))) {
    std::fprintf(stderr, "[light_mem warning] snapshot save: write header failed (file=%s errno=%d %s)\n",
                 filename.c_str(), errno, std::strerror(errno));
    ::close(fd);
    std::remove(tmp_filename.c_str());
    return false;
  }

  // Write entries in Cold-to-Hot order (reverse LRU)
  // lru_list_: front=Hot, back=Cold
  // We iterate from rbegin() (Cold) to rend() (Hot)
  for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
    const std::string &hash = *it;
    auto idx_it = index_.find(hash);
    if (idx_it != index_.end()) {
      const auto &entry = idx_it->second;
      if (entry.ready && !entry.writing) {
        uint32_t hash_len = static_cast<uint32_t>(hash.size());
        uint64_t slot_id = static_cast<uint64_t>(entry.slot_id);
        uint32_t crc = static_cast<uint32_t>(entry.crc);

        if (!write_all(&hash_len, sizeof(hash_len)) || !write_all(hash.data(), hash_len) ||
            !write_all(&slot_id, sizeof(slot_id)) || !write_all(&crc, sizeof(crc))) {
          std::fprintf(stderr, "[light_mem warning] snapshot save: write entry failed (file=%s errno=%d %s)\n",
                       filename.c_str(), errno, std::strerror(errno));
          ::close(fd);
          std::remove(tmp_filename.c_str());
          return false;
        }
      }
    }
  }

  // Ensure file contents are durable before rename.
  if (::fsync(fd) != 0) {
    std::fprintf(stderr, "[light_mem warning] snapshot save: fsync failed (file=%s errno=%d %s)\n", filename.c_str(),
                 errno, std::strerror(errno));
    ::close(fd);
    std::remove(tmp_filename.c_str());
    return false;
  }
  ::close(fd);

  // Atomic replace.
  if (std::rename(tmp_filename.c_str(), filename.c_str()) != 0) {
    std::fprintf(stderr, "[light_mem warning] snapshot save: rename failed (file=%s errno=%d %s)\n", filename.c_str(),
                 errno, std::strerror(errno));
    std::remove(tmp_filename.c_str());
    return false;
  }

  // Best-effort: fsync directory to persist rename.
  std::string dir = ".";
  auto pos = filename.find_last_of('/');
  if (pos != std::string::npos) {
    dir = filename.substr(0, pos);
  }
  int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (dfd >= 0) {
    (void)::fsync(dfd);
    ::close(dfd);
  }

  return true;
}

bool LocalCacheIndex::loadSnapshotToIndex(const std::string &filename) {
  return loadSnapshotEntries(
      filename, capacity_, SnapshotLoadMode::Strict,
      [this](const std::string &hash, size_t slot_id, uint32_t crc) { put_ready(hash, slot_id, crc); },
      "snapshot load");
}

bool LocalCacheIndex::loadSnapshotToMapping(const std::string &filename,
                                            std::unordered_map<std::string, size_t> &mapping,
                                            std::unordered_map<std::string, uint32_t> &crc_map,
                                            std::unordered_set<std::string> &crc_present) {
  return loadSnapshotEntries(
      filename, capacity_, SnapshotLoadMode::BestEffort,
      [&mapping, &crc_map, &crc_present](const std::string &hash, size_t slot_id, uint32_t crc) {
        mapping[hash] = slot_id;
        crc_map[hash] = crc;
        crc_present.insert(hash);
      },
      "snapshot load mapping");
}

bool LocalCacheIndex::loadSnapshotEntries(const std::string &filename, size_t max_slot_exclusive, SnapshotLoadMode mode,
                                          const SnapshotEntryConsumer &consumer, const char *warn_prefix) {
  int fd = ::open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    if (errno != ENOENT) {
      std::fprintf(stderr, "[light_mem warning] %s: open failed (file=%s errno=%d %s)\n", warn_prefix,
                   filename.c_str(), errno, std::strerror(errno));
    }
    return false;
  }

  bool warned = false;
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
  if (!read_all(&magic, sizeof(magic)) || !read_all(&version, sizeof(version)) || !read_all(&count, sizeof(count))) {
    if (!warned) {
      if (last_eof) {
        std::fprintf(stderr, "[light_mem warning] %s: truncated header (file=%s)\n", warn_prefix, filename.c_str());
      } else if (last_errno != 0) {
        std::fprintf(stderr, "[light_mem warning] %s: read header failed (file=%s errno=%d %s)\n", warn_prefix,
                     filename.c_str(), last_errno, std::strerror(last_errno));
      }
      warned = true;
    }
    ::close(fd);
    return false;
  }

  if (magic != SNAPSHOT_MAGIC || version != SNAPSHOT_VERSION) {
    if (!warned) {
      std::fprintf(stderr,
                   "[light_mem warning] %s: bad header (file=%s magic=0x%08x version=%u expect_magic=0x%08x expect_version=%u)\n",
                   warn_prefix, filename.c_str(), magic, version, SNAPSHOT_MAGIC, SNAPSHOT_VERSION);
      warned = true;
    }
    ::close(fd);
    return false;
  }

  for (uint64_t i = 0; i < count; i++) {
    uint32_t hash_len = 0;
    if (!read_all(&hash_len, sizeof(hash_len)) || hash_len > 4096) {
      if (!warned) {
        std::fprintf(stderr, "[light_mem warning] %s: bad hash_len (file=%s)\n", warn_prefix, filename.c_str());
        warned = true;
      }
      if (mode == SnapshotLoadMode::Strict) {
        ::close(fd);
        return false;
      }
      break;
    }

    std::string hash;
    hash.resize(hash_len);
    if (hash_len > 0 && !read_all(hash.data(), hash_len)) {
      if (!warned) {
        std::fprintf(stderr, "[light_mem warning] %s: read hash failed (file=%s)\n", warn_prefix, filename.c_str());
        warned = true;
      }
      if (mode == SnapshotLoadMode::Strict) {
        ::close(fd);
        return false;
      }
      break;
    }

    uint64_t slot_id = 0;
    if (!read_all(&slot_id, sizeof(slot_id))) {
      if (!warned) {
        std::fprintf(stderr, "[light_mem warning] %s: read slot_id failed (file=%s)\n", warn_prefix,
                     filename.c_str());
        warned = true;
      }
      if (mode == SnapshotLoadMode::Strict) {
        ::close(fd);
        return false;
      }
      break;
    }

    uint32_t crc = 0;
    if (!read_all(&crc, sizeof(crc))) {
      if (!warned) {
        std::fprintf(stderr, "[light_mem warning] %s: read crc failed (file=%s)\n", warn_prefix, filename.c_str());
        warned = true;
      }
      if (mode == SnapshotLoadMode::Strict) {
        ::close(fd);
        return false;
      }
      break;
    }

    const size_t sid = static_cast<size_t>(slot_id);
    if (sid < max_slot_exclusive) {
      consumer(hash, sid, crc);
    }
  }

  ::close(fd);
  return true;
}

} // namespace storage
} // namespace cache