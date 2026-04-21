#pragma once

#include <unistd.h>

#if defined(__APPLE__)
#include <fcntl.h>
#endif

namespace cache {
namespace utils {

inline int fdatasync_compat(int fd) {
#if defined(__APPLE__)
  // macOS doesn't provide fdatasync(2). Best-effort equivalent is F_FULLFSYNC
  // (flushes to physical media). If unavailable/fails, fall back to fsync.
#ifdef F_FULLFSYNC
  if (::fcntl(fd, F_FULLFSYNC) == 0) {
    return 0;
  }
#endif
  return ::fsync(fd);
#else
  return ::fdatasync(fd);
#endif
}

} // namespace utils
} // namespace cache
