#include "storage/local_storage_engine.h"

#include "config.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
    if (file_fds_[i] >= 0) {
      ::close(file_fds_[i]);
      file_fds_[i] = -1;
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

    if (initializer) {
      // Open data file (preallocate only when newly created).
      file_fds_[i] = open_existing_or_create_new(data_filename, shard_storage_size, nullptr, 0);
      // Open meta file (initialize header only when newly created).
      meta_fds_[i] = open_existing_or_create_new(meta_filename, 0, kMetaHeader, META_HEADER_SIZE);
    } else {
      // Follower: never create/truncate.
      file_fds_[i] = open_existing_file(data_filename);
      meta_fds_[i] = open_existing_file(meta_filename);
    }

    ensureSuperBlocksInitialized(i);
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
