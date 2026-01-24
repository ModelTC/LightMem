#pragma once

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <netdb.h>
#include <netinet/tcp.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace cache::storage {

class RedisClient {
public:
  struct Options {
    std::string host = "127.0.0.1";
    int port = 6379;
    std::string password;
    int db = 0;
    std::string key_prefix;
    int connect_timeout_ms = 300;
    // NOTE: Redis metadata updates can be pipelined in large batches.
    // Keep this comfortably above typical batch drain time to avoid
    // silent timeouts that desynchronize the connection.
    int io_timeout_ms = 5000;
  };

  explicit RedisClient(Options opt) : opt_(std::move(opt)) {}

  ~RedisClient() { close(); }

  RedisClient(const RedisClient &) = delete;
  RedisClient &operator=(const RedisClient &) = delete;

  bool connect() {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (sock_ >= 0) {
      return true;
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *res = nullptr;
    const std::string port_str = std::to_string(opt_.port);
    if (::getaddrinfo(opt_.host.c_str(), port_str.c_str(), &hints, &res) != 0) {
      return false;
    }

    int sock = -1;
    for (addrinfo *p = res; p != nullptr; p = p->ai_next) {
      sock = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
      if (sock < 0) {
        continue;
      }

      setSocketOptions(sock);

      if (::connect(sock, p->ai_addr, p->ai_addrlen) == 0) {
        break;
      }

      ::close(sock);
      sock = -1;
    }
    ::freeaddrinfo(res);

    if (sock < 0) {
      return false;
    }

    sock_ = sock;

    // AUTH
    if (!opt_.password.empty()) {
      auto resp = command({"AUTH", opt_.password});
      if (!resp.ok) {
        close();
        return false;
      }
    }

    // SELECT DB
    if (opt_.db != 0) {
      auto resp = command({"SELECT", std::to_string(opt_.db)});
      if (!resp.ok) {
        close();
        return false;
      }
    }

    // PING
    auto ping = command({"PING"});
    if (!ping.ok) {
      close();
      return false;
    }

    return true;
  }

  void close() {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (sock_ >= 0) {
      ::close(sock_);
      sock_ = -1;
    }
  }

  bool isConnected() const { return sock_ >= 0; }

  bool setString(const std::string &key, const std::string &value) {
    auto resp = command({"SET", key, value});
    return resp.ok;
  }

  std::optional<std::string> getString(const std::string &key) {
    auto resp = command({"GET", key});
    if (!resp.ok) {
      return std::nullopt;
    }
    return resp.bulk;
  }

  bool hset(const std::string &key, const std::string &field, const std::string &value) {
    auto resp = command({"HSET", key, field, value});
    return resp.ok;
  }

  bool hsetnx(const std::string &key, const std::string &field, const std::string &value) {
    auto resp = command({"HSETNX", key, field, value});
    if (!resp.ok || !resp.bulk.has_value()) {
      return false;
    }
    // Integer reply: 1 if field is a new field, 0 if it was already present.
    try {
      return std::stoll(*resp.bulk) == 1;
    } catch (...) {
      return false;
    }
  }

  bool hexists(const std::string &key, const std::string &field) {
    auto resp = command({"HEXISTS", key, field});
    if (!resp.ok || !resp.bulk.has_value()) {
      return false;
    }
    try {
      return std::stoll(*resp.bulk) == 1;
    } catch (...) {
      return false;
    }
  }

  std::optional<std::string> hget(const std::string &key, const std::string &field) {
    auto resp = command({"HGET", key, field});
    if (!resp.ok) {
      return std::nullopt;
    }
    return resp.bulk;
  }

  std::optional<uint64_t> hlen(const std::string &key) {
    auto resp = command({"HLEN", key});
    if (!resp.ok || !resp.bulk.has_value()) {
      return std::nullopt;
    }
    try {
      return static_cast<uint64_t>(std::stoull(*resp.bulk));
    } catch (...) {
      return std::nullopt;
    }
  }

  // HMGET key field1 field2 ...
  // Returns one entry per requested field. Missing fields are returned as empty strings.
  // NOTE: this matches parseResp() behavior for nil bulk strings inside arrays.
  std::optional<std::vector<std::string>> hmget(const std::string &key, const std::vector<std::string> &fields) {
    if (fields.empty()) {
      return std::vector<std::string>{};
    }
    std::vector<std::string> argv;
    argv.reserve(2 + fields.size());
    argv.emplace_back("HMGET");
    argv.emplace_back(key);
    for (const auto &f : fields) {
      argv.emplace_back(f);
    }
    auto resp = command(argv);
    if (!resp.ok) {
      return std::nullopt;
    }
    // RESP array length should match requested fields.
    if (resp.array.size() != fields.size()) {
      return std::nullopt;
    }
    return resp.array;
  }

  bool hdel(const std::string &key, const std::string &field) {
    auto resp = command({"HDEL", key, field});
    return resp.ok;
  }

  bool del(const std::string &key) {
    auto resp = command({"DEL", key});
    return resp.ok;
  }

  // Send multiple commands back-to-back (RESP pipelining) and parse all replies.
  // Returns true if the I/O succeeded; individual Redis error replies are tolerated.
  // This is intended for high-throughput metadata updates (e.g., HSET/HDEL) where
  // callers historically ignored per-command success.
  bool pipeline(const std::vector<std::vector<std::string>> &argv_list) {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (sock_ < 0) {
      return false;
    }
    if (argv_list.empty()) {
      return true;
    }

    std::string req;
    req.reserve(argv_list.size() * 64);
    for (const auto &argv : argv_list) {
      req += "*" + std::to_string(argv.size()) + "\r\n";
      for (const auto &a : argv) {
        req += "$" + std::to_string(a.size()) + "\r\n";
        req += a;
        req += "\r\n";
      }
    }

    if (!writeAll(req.data(), req.size())) {
      close();
      return false;
    }

    // Drain replies to keep the connection in sync.
    for (size_t i = 0; i < argv_list.size(); i++) {
      (void)parseResp();
      if (sock_ < 0) {
        return false;
      }
    }
    return true;
  }

  // EVAL helper (used to reduce round-trips on hot paths).
  std::optional<int64_t> evalInt(const std::string &script, const std::vector<std::string> &keys,
                                 const std::vector<std::string> &args) {
    std::vector<std::string> argv;
    argv.reserve(3 + keys.size() + args.size());
    argv.emplace_back("EVAL");
    argv.emplace_back(script);
    argv.emplace_back(std::to_string(keys.size()));
    for (const auto &k : keys) {
      argv.emplace_back(k);
    }
    for (const auto &a : args) {
      argv.emplace_back(a);
    }
    auto resp = command(argv);
    if (!resp.ok || !resp.bulk.has_value()) {
      return std::nullopt;
    }
    try {
      return std::stoll(*resp.bulk);
    } catch (...) {
      return std::nullopt;
    }
  }

  // SET key value NX PX ttl_ms
  bool setStringNxPx(const std::string &key, const std::string &value, uint64_t ttl_ms) {
    auto resp = command({"SET", key, value, "NX", "PX", std::to_string(ttl_ms)});
    // Success returns +OK; failure returns nil bulk.
    return resp.ok && resp.bulk.has_value();
  }

  // Returns alternating field/value list
  std::optional<std::vector<std::string>> hgetall(const std::string &key) {
    auto resp = command({"HGETALL", key});
    if (!resp.ok) {
      return std::nullopt;
    }
    return resp.array;
  }

  std::string shardIndexKey(size_t shard_id) const {
    return opt_.key_prefix + ":" + std::to_string(shard_id) + ":index";
  }

  // Global hash -> "shard_id:slot_id" mapping.
  std::string globalIndexKey() const { return opt_.key_prefix + ":global:index"; }

  // Global hash -> data CRC (uint32 as string).
  std::string globalCrcKey() const { return opt_.key_prefix + ":global:crc"; }

  // Per-hash lock key used to prevent concurrent writers from racing.
  std::string hashLockKey(const std::string &hash) const { return opt_.key_prefix + ":lock:" + hash; }

  std::string shardSeqKey(size_t shard_id) const { return opt_.key_prefix + ":" + std::to_string(shard_id) + ":seq"; }

private:
  struct Resp {
    bool ok = false;
    std::optional<std::string> bulk;
    std::vector<std::string> array;
  };

  void setSocketOptions(int sock) {
    int yes = 1;
    ::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

    timeval tv{};
    tv.tv_sec = opt_.io_timeout_ms / 1000;
    tv.tv_usec = (opt_.io_timeout_ms % 1000) * 1000;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  }

  bool writeAll(const void *data, size_t len) {
    const char *p = static_cast<const char *>(data);
    size_t left = len;
    while (left > 0) {
      ssize_t n = ::send(sock_, p, left, 0);
      if (n <= 0) {
        return false;
      }
      p += static_cast<size_t>(n);
      left -= static_cast<size_t>(n);
    }
    return true;
  }

  bool readExact(void *buf, size_t len) {
    char *p = static_cast<char *>(buf);
    size_t left = len;
    while (left > 0) {
      ssize_t n = ::recv(sock_, p, left, 0);
      if (n <= 0) {
        return false;
      }
      p += static_cast<size_t>(n);
      left -= static_cast<size_t>(n);
    }
    return true;
  }

  bool readLine(std::string &out) {
    out.clear();
    char c;
    while (true) {
      if (!readExact(&c, 1)) {
        return false;
      }
      if (c == '\r') {
        char lf;
        if (!readExact(&lf, 1)) {
          return false;
        }
        if (lf != '\n') {
          return false;
        }
        return true;
      }
      out.push_back(c);
    }
  }

  Resp command(const std::vector<std::string> &argv) {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (sock_ < 0) {
      return {};
    }

    std::string req;
    req.reserve(64);
    req += "*" + std::to_string(argv.size()) + "\r\n";
    for (const auto &a : argv) {
      req += "$" + std::to_string(a.size()) + "\r\n";
      req += a;
      req += "\r\n";
    }

    if (!writeAll(req.data(), req.size())) {
      close();
      return {};
    }

    return parseResp();
  }

  Resp parseResp() {
    char type;
    if (!readExact(&type, 1)) {
      close();
      return {};
    }

    std::string line;
    if (!readLine(line)) {
      close();
      return {};
    }

    if (type == '+') {
      // Simple string OK
      Resp r;
      r.ok = true;
      r.bulk = line;
      return r;
    }

    if (type == '-') {
      // Error
      Resp r;
      r.ok = false;
      return r;
    }

    if (type == ':') {
      Resp r;
      r.ok = true;
      r.bulk = line;
      return r;
    }

    if (type == '$') {
      int64_t n = 0;
      try {
        n = std::stoll(line);
      } catch (...) {
        close();
        return {};
      }
      if (n < 0) {
        Resp r;
        r.ok = true;
        r.bulk = std::nullopt;
        return r;
      }
      std::string bulk;
      bulk.resize(static_cast<size_t>(n));
      if (!readExact(bulk.data(), bulk.size())) {
        close();
        return {};
      }
      // consume CRLF
      char crlf[2];
      if (!readExact(crlf, 2) || crlf[0] != '\r' || crlf[1] != '\n') {
        close();
        return {};
      }
      Resp r;
      r.ok = true;
      r.bulk = std::move(bulk);
      return r;
    }

    if (type == '*') {
      int64_t count = 0;
      try {
        count = std::stoll(line);
      } catch (...) {
        close();
        return {};
      }
      if (count < 0) {
        Resp r;
        r.ok = true;
        return r;
      }
      std::vector<std::string> arr;
      arr.reserve(static_cast<size_t>(count));
      for (int64_t i = 0; i < count; i++) {
        char t;
        if (!readExact(&t, 1)) {
          close();
          return {};
        }
        std::string l;
        if (!readLine(l)) {
          close();
          return {};
        }
        if (t == '$') {
          int64_t n = 0;
          try {
            n = std::stoll(l);
          } catch (...) {
            close();
            return {};
          }
          if (n < 0) {
            arr.emplace_back("");
            continue;
          }
          std::string bulk;
          bulk.resize(static_cast<size_t>(n));
          if (!readExact(bulk.data(), bulk.size())) {
            close();
            return {};
          }
          char crlf[2];
          if (!readExact(crlf, 2) || crlf[0] != '\r' || crlf[1] != '\n') {
            close();
            return {};
          }
          arr.emplace_back(std::move(bulk));
        } else if (t == ':') {
          arr.emplace_back(l);
        } else if (t == '+') {
          arr.emplace_back(l);
        } else {
          // Unsupported nested types
          close();
          return {};
        }
      }
      Resp r;
      r.ok = true;
      r.array = std::move(arr);
      return r;
    }

    close();
    return {};
  }

  Options opt_;
  int sock_ = -1;
  mutable std::recursive_mutex mu_;
};

} // namespace cache::storage
