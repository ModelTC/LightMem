#include "storage/local_storage_engine.h"

#include "config.h"

#include "utils/fsync_compat.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <stdexcept>

namespace cache {
namespace storage {

static void chmod_best_effort(const std::string &path, mode_t mode) {
  // Best-effort: on shared filesystems the owner should be able to chmod.
  // If it fails (e.g., permission/readonly), leave it as-is.
  (void)::chmod(path.c_str(), mode);
}

static void ensure_parent_dir_shareable(const std::string &path) {
  namespace fs = std::filesystem;
  std::error_code ec;

  fs::path p(path);
  fs::path parent = p.parent_path();
  if (parent.empty()) {
    parent = fs::current_path(ec);
    if (ec) {
      // If current_path fails, just skip parent handling.
      return;
    }
  }

  ec.clear();
  fs::create_directories(parent, ec);
  // Even if create_directories fails (e.g. already exists), try chmod.
  chmod_best_effort(parent.string(), 0777);
}

static void ensure_directory(const std::string &path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) {
    if (::mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
      const int err = errno;
      throw std::runtime_error("Failed to create directory: " + path + ", errno=" + std::to_string(err) +
                               ", reason=" + std::string(::strerror(err)));
    }
  } else if (!S_ISDIR(st.st_mode)) {
    throw std::runtime_error("Path exists but is not a directory: " + path);
  }
  chmod_best_effort(path, 0777);
}

static void require_directory_exists(const std::string &path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) {
    const int err = errno;
    throw std::runtime_error("Missing directory: " + path + ", errno=" + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
  if (!S_ISDIR(st.st_mode)) {
    throw std::runtime_error("Path exists but is not a directory: " + path);
  }
}

// Initializer-only: open existing OR create new file without truncating an existing file.
// If created, optionally preallocates and writes an initial header.
static int open_existing_or_create_new(const std::string &path, size_t preallocate_size, const void *init_data,
                                       size_t init_size) {
  bool created = false;
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0666);
  if (fd >= 0) {
    created = true;
  } else if (errno == EEXIST) {
    fd = ::open(path.c_str(), O_RDWR);
  }
  if (fd < 0) {
    const int err = errno;
    throw std::runtime_error("Failed to open file: " + path + ", errno=" + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }

  if (created) {
    if (preallocate_size > 0) {
      // Fast sparse preallocation (matches historical behavior): set the apparent file size
      // without forcing physical block allocation.
      if (::lseek(fd, static_cast<off_t>(preallocate_size - 1), SEEK_SET) < 0 || ::write(fd, "", 1) != 1) {
        const int err = errno;
        ::close(fd);
        throw std::runtime_error("Failed to preallocate file: " + path + ", errno=" + std::to_string(err) +
                                 ", reason=" + std::string(::strerror(err)));
      }
    }
    if (init_data && init_size > 0) {
      if (::pwrite(fd, init_data, init_size, 0) != static_cast<ssize_t>(init_size)) {
        const int err = errno;
        ::close(fd);
        throw std::runtime_error("Failed to initialize file: " + path + ", errno=" + std::to_string(err) +
                                 ", reason=" + std::string(::strerror(err)));
      }
    }
  }

  chmod_best_effort(path, 0666);
  return fd;
}

// Follower-only: open existing file; never creates/truncates.
static int open_existing_file(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDWR);
  if (fd < 0) {
    const int err = errno;
    throw std::runtime_error("Failed to open existing file: " + path + ", errno=" + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
  return fd;
}

static bool file_exists(const std::string &path) {
  struct stat st;
  return (::stat(path.c_str(), &st) == 0);
}

static void write_init_marker(const std::string &path) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
  if (fd < 0) {
    const int err = errno;
    throw std::runtime_error("Failed to write init marker: " + path + ", errno=" + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }
  static const char kMsg[] = "ok\n";
  (void)::write(fd, kMsg, sizeof(kMsg) - 1);
  (void)::fsync(fd);
  ::close(fd);
  chmod_best_effort(path, 0666);
}

void LocalStorageEngine::cleanup() {
  for (size_t i = 0; i < shard_; i++) {
    if (i < file_fds_.size()) {
      for (int &fd : file_fds_[i]) {
        if (fd >= 0) {
          ::close(fd);
          fd = -1;
        }
      }
    }
    if (meta_fds_[i] >= 0) {
      ::close(meta_fds_[i]);
      meta_fds_[i] = -1;
    }
  }
}

void LocalStorageEngine::createOrOpenFiles(size_t shard_storage_size) {
  ensure_parent_dir_shareable(filename_);

  static const char kMetaHeader[META_HEADER_SIZE] = {0};

  // Global init lock to avoid TOCTOU races when multiple services start against the same directory.
  // Only the initializer creates directories/files; followers wait for the marker and then only open existing files.
  const std::string init_lock_dir = filename_ + ".init.lock";
  const std::string init_marker = filename_ + ".initialized";

  bool initializer = false;
  if (::mkdir(init_lock_dir.c_str(), 0777) == 0) {
    initializer = true;
    chmod_best_effort(init_lock_dir, 0777);
  } else if (errno == EEXIST) {
    initializer = false;
  } else {
    const int err = errno;
    throw std::runtime_error("Failed to create init lock: " + init_lock_dir + ", errno=" + std::to_string(err) +
                             ", reason=" + std::string(::strerror(err)));
  }

  if (!initializer) {
    // If marker already exists, proceed immediately. Otherwise, wait for initializer to finish.
    if (!file_exists(init_marker)) {
      static constexpr int kWaitMs = 50;
      static constexpr int kTimeoutMs = 60000; // 60s
      int waited = 0;
      while (!file_exists(init_marker) && waited < kTimeoutMs) {
        ::usleep(kWaitMs * 1000);
        waited += kWaitMs;
      }
      if (!file_exists(init_marker)) {
        throw std::runtime_error("Timed out waiting for storage init marker: " + init_marker +
                                 ". Another process may be stuck initializing (lock=" + init_lock_dir + ")");
      }
    }
  }

  for (size_t i = 0; i < shard_; i++) {
    const std::string shard_path = filename_ + "_" + std::to_string(i);
    if (initializer) {
      ensure_directory(shard_path);
    } else {
      require_directory_exists(shard_path);
    }

    const std::string data_filename = shard_path + "/data";
    const std::string meta_filename = shard_path + "/meta";

    // Scheme A: each shard's data is split across num_stripes_ stripe sub-files.
    // Stripe j holds one block segment of stripe_size_ bytes per slot, so the per-stripe
    // file is sized shard_storage_size / num_stripes_. With num_stripes_ == 1, the single
    // stripe file is named "data" (backward compatible); otherwise "data.0 .. data.S-1".
    const size_t stripe_file_size = shard_storage_size / num_stripes_;
    file_fds_[i].assign(num_stripes_, -1);
    for (size_t j = 0; j < num_stripes_; ++j) {
      const std::string stripe_path =
          (num_stripes_ == 1) ? data_filename : (data_filename + "." + std::to_string(j));
      if (initializer) {
        // Open data stripe (preallocate only when newly created).
        file_fds_[i][j] = open_existing_or_create_new(stripe_path, stripe_file_size, nullptr, 0);
      } else {
        // Follower: never create/truncate.
        file_fds_[i][j] = open_existing_file(stripe_path);
      }
    }

    if (initializer) {
      // Online mode only: the metadata journal (WAL) backs cross-node recovery.
      // Offline mode skips it entirely to prioritize read/write performance.
      if (online_mode_) {
        meta_fds_[i] = open_existing_or_create_new(meta_filename, 0, kMetaHeader, META_HEADER_SIZE);
      }
    } else {
      // Follower: never create/truncate.
      if (online_mode_) {
        meta_fds_[i] = open_existing_file(meta_filename);
      }
    }

#ifdef __APPLE__
    // Keep metadata journal I/O from polluting page cache and affecting subsequent
    // high-throughput reads in offline mode.
    if (meta_fds_[i] >= 0) {
      (void)::fcntl(meta_fds_[i], F_NOCACHE, 1);
    }
#endif

    // Superblocks are part of the WAL/recovery machinery; only needed online.
    if (online_mode_) {
      ensureSuperBlocksInitialized(i);
    }
  }

  if (initializer) {
    // Publish completion marker then release lock.
    write_init_marker(init_marker);
    (void)::rmdir(init_lock_dir.c_str());
  }
}

void LocalStorageEngine::initIndexBackend(const std::string &endpoint, const std::string &index_prefix) {
  // Index backend is opt-in: only enable when endpoint is set.
  std::string ep = endpoint;
  if (ep.empty()) {
    return;
  }

  RedisClient::Options opt;

  // Optional: override Redis key prefix to isolate multiple runs.
  // Unified with coordinator prefix via the single init arg `index_prefix`.
  {
    std::string v = index_prefix;
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) {
      v.erase(v.begin());
    }
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) {
      v.pop_back();
    }
    if (!v.empty()) {
      opt.key_prefix = std::move(v);
    }
  }

  // Strip scheme if someone passes it.
  const std::string http_prefix = "http://";
  const std::string https_prefix = "https://";
  if (ep.rfind(http_prefix, 0) == 0) {
    ep = ep.substr(http_prefix.size());
  } else if (ep.rfind(https_prefix, 0) == 0) {
    ep = ep.substr(https_prefix.size());
  }

  // Take first if comma-separated.
  const auto comma = ep.find(',');
  if (comma != std::string::npos) {
    ep = ep.substr(0, comma);
  }

  // Trim whitespace (minimal).
  while (!ep.empty() && (ep.front() == ' ' || ep.front() == '\t')) {
    ep.erase(ep.begin());
  }
  while (!ep.empty() && (ep.back() == ' ' || ep.back() == '\t')) {
    ep.pop_back();
  }

  if (!ep.empty()) {
    auto colon = ep.rfind(':');
    if (colon != std::string::npos) {
      const std::string host_part = ep.substr(0, colon);
      const std::string port_part = ep.substr(colon + 1);
      if (!host_part.empty()) {
        opt.host = host_part;
      }
      if (!port_part.empty()) {
        try {
          opt.port = std::stoi(port_part);
        } catch (...) {
        }
      }
    } else {
      opt.host = ep;
    }
  }

  // Pool size: controls number of TCP connections used for shard/journal metadata updates.
  // A larger pool reduces mutex/queue contention under high concurrency.
  int pool_size = 32;

  // Dedicated client for write-hot-path lock/dedupe.
  redis_lock_ = std::make_unique<RedisClient>(opt);

  // Pool for journal workers / lookups.
  redis_pool_.clear();
  redis_pool_.reserve(static_cast<size_t>(pool_size));
  for (int i = 0; i < pool_size; i++) {
    redis_pool_.emplace_back(std::make_unique<RedisClient>(opt));
  }
}

bool LocalStorageEngine::preadAll(int fd, void *buf, size_t len, off_t offset) {
  char *p = static_cast<char *>(buf);
  size_t left = len;
  while (left > 0) {
    ssize_t n = ::pread(fd, p, left, offset);
    if (n <= 0) {
      return false;
    }
    p += static_cast<size_t>(n);
    left -= static_cast<size_t>(n);
    offset += static_cast<off_t>(n);
  }
  return true;
}

bool LocalStorageEngine::pwriteAll(int fd, const void *buf, size_t len, off_t offset) {
  const char *p = static_cast<const char *>(buf);
  size_t left = len;
  while (left > 0) {
    ssize_t n = ::pwrite(fd, p, left, offset);
    if (n <= 0) {
      return false;
    }
    p += static_cast<size_t>(n);
    left -= static_cast<size_t>(n);
    offset += static_cast<off_t>(n);
  }
  return true;
}

// Scheme A: split [0, len_bytes) of a block into up to num_stripes_ segments and issue them
// concurrently across the stripe files via io_pool_. Segment j (j*stripe_size_ .. ) maps to
// file_fds_[shard][j] at byte offset slot_id * stripe_size_. Segments target distinct files,
// so concurrent pread/pwrite are independent. The caller services segment 0 inline and waits
// for the rest on a local latch; the per-shard io lock held by the caller is unchanged.
namespace {
struct StripeIoLatch {
  std::mutex mu;
  std::condition_variable cv;
  size_t remaining = 0;
  std::atomic<bool> ok{true};

  void completeOne() {
    std::lock_guard<std::mutex> lk(mu);
    if (--remaining == 0) {
      cv.notify_one();
    }
  }

  void wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [this] { return remaining == 0; });
  }
};
} // namespace

bool LocalStorageEngine::writeDataBlock(size_t shard_id, size_t slot_id, const char *buf, size_t len_bytes) {
  const std::vector<int> &fds = file_fds_[shard_id];
  const off_t base = static_cast<off_t>(slot_id) * static_cast<off_t>(stripe_size_);

  if (num_stripes_ <= 1 || !io_pool_ || len_bytes <= stripe_size_) {
    // Single stripe touched (small/partial write or striping disabled): one syscall.
    return pwriteAll(fds[0], buf, len_bytes, base);
  }

  // Number of stripes that actually carry data for this logical length.
  const size_t active = std::min(num_stripes_, (len_bytes + stripe_size_ - 1) / stripe_size_);

  StripeIoLatch latch;
  latch.remaining = active - 1; // segment 0 serviced inline below.

  for (size_t j = 1; j < active; ++j) {
    const size_t soff = j * stripe_size_;
    const size_t clen = std::min(stripe_size_, len_bytes - soff);
    const int fd = fds[j];
    io_pool_->submit([this, fd, buf, soff, clen, base, &latch]() {
      if (!pwriteAll(fd, buf + soff, clen, base)) {
        latch.ok.store(false, std::memory_order_relaxed);
      }
      latch.completeOne();
    });
  }

  // Segment 0 inline on the caller thread.
  if (!pwriteAll(fds[0], buf, std::min(stripe_size_, len_bytes), base)) {
    latch.ok.store(false, std::memory_order_relaxed);
  }

  latch.wait();
  return latch.ok.load(std::memory_order_relaxed);
}

bool LocalStorageEngine::readDataBlock(size_t shard_id, size_t slot_id, char *buf, size_t len_bytes) {
  const std::vector<int> &fds = file_fds_[shard_id];
  const off_t base = static_cast<off_t>(slot_id) * static_cast<off_t>(stripe_size_);

  if (num_stripes_ <= 1 || !io_pool_ || len_bytes <= stripe_size_) {
    return preadAll(fds[0], buf, len_bytes, base);
  }

  const size_t active = std::min(num_stripes_, (len_bytes + stripe_size_ - 1) / stripe_size_);

  StripeIoLatch latch;
  latch.remaining = active - 1;

  for (size_t j = 1; j < active; ++j) {
    const size_t soff = j * stripe_size_;
    const size_t clen = std::min(stripe_size_, len_bytes - soff);
    const int fd = fds[j];
    io_pool_->submit([this, fd, buf, soff, clen, base, &latch]() {
      if (!preadAll(fd, buf + soff, clen, base)) {
        latch.ok.store(false, std::memory_order_relaxed);
      }
      latch.completeOne();
    });
  }

  if (!preadAll(fds[0], buf, std::min(stripe_size_, len_bytes), base)) {
    latch.ok.store(false, std::memory_order_relaxed);
  }

  latch.wait();
  return latch.ok.load(std::memory_order_relaxed);
}

bool LocalStorageEngine::syncDataBlock(size_t shard_id) {
  // Online durability: a block write touches all stripes (full block_size_), so fdatasync each.
  for (int fd : file_fds_[shard_id]) {
    if (fd >= 0 && cache::utils::fdatasync_compat(fd) != 0) {
      return false;
    }
  }
  return true;
}

void LocalStorageEngine::fadviseDataDontNeed(size_t shard_id, size_t slot_id, size_t len_bytes) {
#ifndef __APPLE__
  const off_t base = static_cast<off_t>(slot_id) * static_cast<off_t>(stripe_size_);
  const std::vector<int> &fds = file_fds_[shard_id];
  const size_t active =
      (num_stripes_ <= 1) ? 1 : std::min(num_stripes_, (len_bytes + stripe_size_ - 1) / stripe_size_);
  for (size_t j = 0; j < active; ++j) {
    const size_t soff = j * stripe_size_;
    const size_t clen = (len_bytes > soff) ? std::min(stripe_size_, len_bytes - soff) : 0;
    if (clen > 0 && fds[j] >= 0) {
      (void)posix_fadvise(fds[j], base, static_cast<off_t>(clen), POSIX_FADV_DONTNEED);
    }
  }
#else
  (void)shard_id;
  (void)slot_id;
  (void)len_bytes;
#endif
}

bool LocalStorageEngine::readSuperBlockAt(size_t shard_id, off_t off, SuperBlock &sb) {
  if (!preadAll(meta_fds_[shard_id], &sb, sizeof(SuperBlock), off)) {
    return false;
  }
  if (sb.magic != SUPERBLOCK_MAGIC || sb.version != SUPERBLOCK_VERSION || sb.shard_id != shard_id) {
    return false;
  }
  const uint32_t expect = compute_crc32(&sb, SUPERBLOCK_SIZE - sizeof(uint32_t));
  if (expect != sb.crc32) {
    return false;
  }
  if (sb.block_size != block_size_) {
    return false;
  }
  return true;
}

void LocalStorageEngine::writeSuperBlockAt(size_t shard_id, off_t off, SuperBlock sb) {
  sb.crc32 = 0;
  sb.crc32 = compute_crc32(&sb, SUPERBLOCK_SIZE - sizeof(uint32_t));
  if (!pwriteAll(meta_fds_[shard_id], &sb, sizeof(SuperBlock), off)) {
    throw std::runtime_error("Failed to write superblock");
  }
  ::fsync(meta_fds_[shard_id]);
}

void LocalStorageEngine::ensureSuperBlocksInitialized(size_t shard_id) {
  SuperBlock a{}, b{};
  bool va = readSuperBlockAt(shard_id, 0, a);
  bool vb = readSuperBlockAt(shard_id, static_cast<off_t>(SUPERBLOCK_SIZE), b);

  if (va || vb) {
    const SuperBlock &chosen = (!va) ? b : (!vb) ? a : (a.sequence_id >= b.sequence_id ? a : b);
    superblock_seq_[shard_id] = chosen.sequence_id;
    superblock_stable_offset_[shard_id] =
        (chosen.stable_offset >= META_HEADER_SIZE) ? chosen.stable_offset : META_HEADER_SIZE;
    superblock_epoch_[shard_id] = chosen.current_epoch;
    return;
  }

  SuperBlock sb{};
  sb.magic = SUPERBLOCK_MAGIC;
  sb.version = SUPERBLOCK_VERSION;
  sb.shard_id = shard_id;
  sb.sequence_id = 1;
  sb.stable_offset = META_HEADER_SIZE;
  sb.total_capacity = (storage_size_ / shard_) / block_size_;
  sb.block_size = static_cast<uint32_t>(block_size_);
  sb.current_epoch = 0;
  writeSuperBlockAt(shard_id, 0, sb);
  sb.sequence_id = 0;
  writeSuperBlockAt(shard_id, static_cast<off_t>(SUPERBLOCK_SIZE), sb);
  superblock_seq_[shard_id] = 1;
  superblock_stable_offset_[shard_id] = META_HEADER_SIZE;
  superblock_epoch_[shard_id] = 0;

  if (::ftruncate(meta_fds_[shard_id], META_HEADER_SIZE) != 0) {
    throw std::runtime_error("Failed to ftruncate meta to header");
  }
  (void)::fsync(meta_fds_[shard_id]);
}

} // namespace storage
} // namespace cache
