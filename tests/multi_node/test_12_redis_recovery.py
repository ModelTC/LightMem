#!/usr/bin/env python3
"""Redis + disk recovery robustness test.

This test exercises:
1) Start Redis (local redis-server preferred; docker compose fallback)
2) Start LightMem, write data, verify query/read
3) Stop process (normal) and restart, verify recovery from Redis + existing disk files
4) Crash process (abnormal exit) during an in-flight write, restart and verify previously committed data is intact

It is written as a single file that acts as both orchestrator and phase worker.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch

# Make repo sources importable when running this test directly without installing.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
PY_SRC = REPO_ROOT / "python"
TESTS_SRC = REPO_ROOT / "tests"

# Always allow importing test helpers from tests/.
if str(TESTS_SRC) not in sys.path:
    sys.path.insert(0, str(TESTS_SRC))

# IMPORTANT: do NOT unconditionally prepend repo's python/.
# If the user has installed light_mem (with the compiled extension) in the environment,
# adding repo python/ first will shadow site-packages and break imports.
try:
    from light_mem import PyLocalCacheService, PyState  # type: ignore
except Exception:
    if str(PY_SRC) not in sys.path:
        sys.path.insert(0, str(PY_SRC))
    from light_mem import PyLocalCacheService, PyState  # type: ignore

from test_utils import generate_cumulative_hashes


ROOT = REPO_ROOT
TESTS_DIR = HERE.parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Accept the common multi_node run_all arguments so this script can be run
    # under tests/multi_node/run_all.py without failing on unknown flags.
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--reuse-services", action="store_true")
    ap.add_argument("--redis-host", default="")
    ap.add_argument("--redis-port", type=int, default=0)
    ap.add_argument("--etcd-host", default="")
    ap.add_argument("--etcd-port", type=int, default=0)
    ap.add_argument("--storage-dir", default="")

    ap.add_argument("--phase", default="orchestrate")
    ap.add_argument("--storage", default="")
    ap.add_argument("--index-endpoint", default="")
    ap.add_argument("--index-prefix", default="lightmem")

    ns, _unknown = ap.parse_known_args(argv)
    return ns


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 120) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True, timeout=timeout)


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(host: str, port: int, *, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _redis_resp_command(host: str, port: int, argv: list[str], *, timeout_s: float = 1.0) -> str:
    """Send a single Redis command over RESP and return the first line reply.

    We keep this minimal to avoid adding external Python dependencies.
    """
    payload = "*" + str(len(argv)) + "\r\n"
    for a in argv:
        b = a.encode("utf-8")
        payload += "$" + str(len(b)) + "\r\n" + a + "\r\n"

    with socket.create_connection((host, port), timeout=timeout_s) as s:
        s.settimeout(timeout_s)
        s.sendall(payload.encode("utf-8"))

        # Read a single RESP line (type + content + CRLF). Good enough for +OK / :int / -ERR.
        buf = bytearray()
        while True:
            ch = s.recv(1)
            if not ch:
                break
            buf += ch
            if buf.endswith(b"\r\n"):
                break

        return buf.decode("utf-8", errors="replace").strip()


def _redis_delete_prefix_keys(host: str, port: int, *, prefix: str, num_shard: int) -> None:
    keys: list[str] = []
    for shard_id in range(num_shard):
        keys.append(f"{prefix}:{shard_id}:index")
        keys.append(f"{prefix}:{shard_id}:seq")

    # DEL supports variadic keys.
    # If keys do not exist, Redis returns :0 which is fine.
    resp = _redis_resp_command(host, port, ["DEL", *keys])
    if not resp or resp[0] not in (":", "+"):
        raise RuntimeError(f"unexpected Redis DEL response: {resp}")


class _RedisHandle:
    def __init__(self, kind: str, host: str, port: int, cleanup_cb):
        self.kind = kind
        self.host = host
        self.port = port
        self._cleanup_cb = cleanup_cb

    def cleanup(self) -> None:
        if self._cleanup_cb:
            try:
                self._cleanup_cb()
            finally:
                self._cleanup_cb = None


def _start_redis_local() -> _RedisHandle | None:
    redis_server = _which("redis-server")
    if not redis_server:
        return None

    port = _find_free_port()
    tmpdir = Path(tempfile.mkdtemp(prefix="lightmem-redis-"))

    # Minimal persistence: AOF enabled so Redis survives its own restart, and so we emulate realistic settings.
    # (Our test mainly needs Redis to stay up while LightMem restarts.)
    cmd = [
        redis_server,
        "--bind",
        "127.0.0.1",
        "--port",
        str(port),
        "--save",
        "",
        "--appendonly",
        "yes",
        "--appendfsync",
        "everysec",
        "--dir",
        str(tmpdir),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not _wait_port("127.0.0.1", port, timeout_s=10.0):
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    def _cleanup():
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return _RedisHandle("local", "127.0.0.1", port, _cleanup)


def _start_redis_docker() -> _RedisHandle | None:
    docker = _which("docker")
    if not docker:
        return None

    # Need `docker compose` subcommand.
    try:
        _run([docker, "compose", "version"], cwd=ROOT, timeout=10)
    except Exception:
        return None

    # Compose file maps host 6379:6379; skip if port is busy.
    if _wait_port("127.0.0.1", 6379, timeout_s=0.2):
        return None

    compose_file = ROOT / "docker-compose.redis.yml"
    if not compose_file.exists():
        return None

    try:
        _run([docker, "compose", "-f", str(compose_file), "up", "-d"], cwd=ROOT, timeout=180)
    except Exception:
        return None

    if not _wait_port("127.0.0.1", 6379, timeout_s=20.0):
        try:
            _run([docker, "compose", "-f", str(compose_file), "down", "-v"], cwd=ROOT, timeout=60)
        except Exception:
            pass
        return None

    def _cleanup():
        try:
            _run([docker, "compose", "-f", str(compose_file), "down", "-v"], cwd=ROOT, timeout=60)
        except Exception:
            pass

    return _RedisHandle("docker", "127.0.0.1", 6379, _cleanup)


def _start_redis_or_skip() -> _RedisHandle:
    h = _start_redis_local()
    if h:
        print(f"[redis] started local redis-server on {h.host}:{h.port}")
        return h

    h = _start_redis_docker()
    if h:
        print(f"[redis] started docker redis on {h.host}:{h.port}")
        return h

    print("SKIP: neither redis-server nor docker compose is available (or port 6379 is busy).")
    sys.exit(0)


def _make_kvcache(*, seed: int, num_pages: int, page_bytes: int) -> torch.Tensor:
    torch.manual_seed(seed)
    # uint8 tensor; element size is 1 byte, so shape[1] == page_bytes.
    return torch.randint(0, 256, size=(num_pages, page_bytes), dtype=torch.uint8, device="cpu")


def _wait_task(task, *, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while not task.ready():
        if time.time() > deadline:
            raise TimeoutError("LightMem task did not finish in time")
        time.sleep(0.001)


def _assert_task_finished(task, label: str) -> None:
    states = task.state()
    if not all(s == PyState.Finished for s in states):
        raise AssertionError(f"{label} failed: states={states}")


def _make_service(*, kvcache: torch.Tensor, storage_dir: Path, storage_size: int, num_shard: int, num_worker: int, index_endpoint: str, index_prefix: str) -> PyLocalCacheService:
    """Create PyLocalCacheService with backward-compatible args.

    Some environments may run an older installed light_mem that doesn't accept index_prefix.
    """
    try:
        return PyLocalCacheService(
            kvcache_tensor=kvcache,
            file=str(storage_dir),
            storage_size=int(storage_size),
            num_shard=int(num_shard),
            num_worker=int(num_worker),
            index_endpoint=str(index_endpoint),
            index_prefix=str(index_prefix),
        )
    except TypeError:
        return PyLocalCacheService(
            kvcache_tensor=kvcache,
            file=str(storage_dir),
            storage_size=int(storage_size),
            num_shard=int(num_shard),
            num_worker=int(num_worker),
            index_endpoint=str(index_endpoint),
        )


def _phase_normal(storage_dir: Path, redis_host: str, redis_port: int, redis_prefix: str) -> None:

    # Keep block size small-ish for test speed.

    page_bytes = 4096
    num_pages = 128
    storage_size = 32 * 1024 * 1024
    num_shard = 4

    # Deterministic payload
    kvcache = _make_kvcache(seed=123, num_pages=num_pages, page_bytes=page_bytes)
    expected = kvcache.clone()

    service = _make_service(
        kvcache=kvcache,
        storage_dir=storage_dir,
        storage_size=storage_size,
        num_shard=num_shard,
        num_worker=8,
        index_endpoint=f"{redis_host}:{redis_port}",
        index_prefix=str(redis_prefix),
    )

    data = list(range(num_pages))
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(num_pages, dtype=torch.int32)

    # Write
    t = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    _wait_task(t)
    _assert_task_finished(t, "write")

    # Query should hit local index
    q = service.query(hash_128s)
    if not all(q):
        raise AssertionError(f"query after write expected all True, got {q}")

    # Readback in same process
    kvcache.zero_()
    t = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "read")

    if not torch.equal(kvcache, expected):
        raise AssertionError("readback mismatch in phase_normal")

    print("[phase_normal] ok")


def _phase_restart_verify(storage_dir: Path, redis_host: str, redis_port: int, redis_prefix: str) -> None:

    page_bytes = 4096
    num_pages = 128
    storage_size = 32 * 1024 * 1024
    num_shard = 4

    # Fresh tensor, deterministic expected
    kvcache = torch.zeros((num_pages, page_bytes), dtype=torch.uint8)
    expected = _make_kvcache(seed=123, num_pages=num_pages, page_bytes=page_bytes)

    service = _make_service(
        kvcache=kvcache,
        storage_dir=storage_dir,
        storage_size=storage_size,
        num_shard=num_shard,
        num_worker=8,
        index_endpoint=f"{redis_host}:{redis_port}",
        index_prefix=str(redis_prefix),
    )

    # If Redis lost its dataset (keys deleted), we need to explicitly recover shard indices
    # back to Redis. In multi-node mode this is normally triggered by the coordinator on
    # shard ownership changes; here we do it manually.
    for shard_id in range(num_shard):
        if hasattr(service, "recover_shard_to_redis_smart"):
            service.recover_shard_to_redis_smart(int(shard_id))
        elif hasattr(service, "recover_shard_to_redis"):
            service.recover_shard_to_redis(int(shard_id))

    data = list(range(num_pages))
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(num_pages, dtype=torch.int32)

    # On restart, query should already return True (rebuilt from Redis HGETALL)
    q = service.query(hash_128s)
    if not all(q):
        raise AssertionError(f"query after restart expected all True, got {q}")

    # Read from disk
    t = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "read after restart")

    if not torch.equal(kvcache, expected):
        raise AssertionError("readback mismatch after restart")

    # Append additional writes with new hashes to ensure continued operation.
    # Use a disjoint token stream so hashes differ.
    kvcache2 = _make_kvcache(seed=456, num_pages=num_pages, page_bytes=page_bytes)
    kvcache.copy_(kvcache2)

    data2 = list(range(10_000, 10_000 + num_pages))
    hash_128s2 = generate_cumulative_hashes(data2)
    t = service.create(hash_128s=hash_128s2, kv_page_indexer=indexer, mode="w")
    _wait_task(t)
    _assert_task_finished(t, "append write")

    # Read back appended data
    kvcache.zero_()
    t = service.create(hash_128s=hash_128s2, kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "append read")

    if not torch.equal(kvcache, kvcache2):
        raise AssertionError("append readback mismatch")

    print("[phase_restart_verify] ok")


def _phase_eviction_setup(storage_dir: Path, redis_host: str, redis_port: int, redis_prefix: str) -> None:
    """Fill cache beyond capacity to trigger LRU eviction, then verify within-process behavior."""

    page_bytes = 4096
    pages_per_block = 64
    num_pages = pages_per_block
    # Capacity 2 blocks (block_size is 256KB with our params).
    storage_size = 512 * 1024
    num_shard = 1

    indexer = torch.arange(num_pages, dtype=torch.int32)
    kvcache = torch.zeros((num_pages, page_bytes), dtype=torch.uint8)

    service = _make_service(
        kvcache=kvcache,
        storage_dir=storage_dir,
        storage_size=storage_size,
        num_shard=num_shard,
        num_worker=4,
        index_endpoint=f"{redis_host}:{redis_port}",
        index_prefix=str(redis_prefix),
    )

    def _block_hashes(tag: int) -> list[int]:
        data = list(range(tag * 10_000, tag * 10_000 + num_pages))
        return generate_cumulative_hashes(data)

    def _write_block(tag: int, seed: int) -> torch.Tensor:
        payload = _make_kvcache(seed=seed, num_pages=num_pages, page_bytes=page_bytes)
        kvcache.copy_(payload)
        t = service.create(hash_128s=_block_hashes(tag), kv_page_indexer=indexer, mode="w")
        _wait_task(t)
        _assert_task_finished(t, f"eviction write tag={tag}")
        return payload

    # Capacity is 2 blocks. To make the eviction deterministic:
    # 1) Write A, then B
    # 2) Touch B so it becomes MRU
    # 3) Insert C => A should be evicted
    expected_a = _write_block(tag=1, seed=111)
    expected_b = _write_block(tag=2, seed=222)

    # Touch B to make it MRU.
    t = service.create(hash_128s=_block_hashes(2), kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "eviction read touch B")
    if not torch.equal(kvcache, expected_b):
        raise AssertionError("eviction: readback of B before eviction mismatch")

    # Insert C, which should evict A.
    expected_c = _write_block(tag=3, seed=333)

    # (Optional sanity) Read C back immediately.
    kvcache.zero_()
    t = service.create(hash_128s=_block_hashes(3), kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "eviction read C")
    if not torch.equal(kvcache, expected_c):
        raise AssertionError("eviction: readback of C after insertion mismatch")

    # Now A should be miss, B and C should be hit.
    q_a = service.query(_block_hashes(1))
    q_b = service.query(_block_hashes(2))
    q_c = service.query(_block_hashes(3))

    if any(q_a):
        raise AssertionError(f"eviction: expected A to be evicted, got query={q_a}")
    if not all(q_b):
        raise AssertionError(f"eviction: expected B to exist, got query={q_b}")
    if not all(q_c):
        raise AssertionError(f"eviction: expected C to exist, got query={q_c}")

    print("[phase_eviction_setup] ok")


def _phase_eviction_restart_verify(storage_dir: Path, redis_host: str, redis_port: int, redis_prefix: str) -> None:
    """Restart and verify eviction state persists (A missing, B/C readable), and service remains writable."""

    page_bytes = 4096
    pages_per_block = 64
    num_pages = pages_per_block
    storage_size = 512 * 1024
    num_shard = 1

    indexer = torch.arange(num_pages, dtype=torch.int32)
    kvcache = torch.zeros((num_pages, page_bytes), dtype=torch.uint8)

    service = _make_service(
        kvcache=kvcache,
        storage_dir=storage_dir,
        storage_size=storage_size,
        num_shard=num_shard,
        num_worker=4,
        index_endpoint=f"{redis_host}:{redis_port}",
        index_prefix=str(redis_prefix),
    )

    def _block_hashes(tag: int) -> list[int]:
        data = list(range(tag * 10_000, tag * 10_000 + num_pages))
        return generate_cumulative_hashes(data)

    expected_b = _make_kvcache(seed=222, num_pages=num_pages, page_bytes=page_bytes)
    expected_c = _make_kvcache(seed=333, num_pages=num_pages, page_bytes=page_bytes)

    # A should be missing
    q_a = service.query(_block_hashes(1))
    if any(q_a):
        raise AssertionError(f"eviction restart: expected A miss, got query={q_a}")

    # B/C should be readable
    t = service.create(hash_128s=_block_hashes(2), kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "eviction restart read B")
    if not torch.equal(kvcache, expected_b):
        raise AssertionError("eviction restart: B mismatch")

    kvcache.zero_()
    t = service.create(hash_128s=_block_hashes(3), kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "eviction restart read C")
    if not torch.equal(kvcache, expected_c):
        raise AssertionError("eviction restart: C mismatch")

    # Ensure continued operation: write/read D
    expected_d = _make_kvcache(seed=444, num_pages=num_pages, page_bytes=page_bytes)
    kvcache.copy_(expected_d)
    t = service.create(hash_128s=_block_hashes(4), kv_page_indexer=indexer, mode="w")
    _wait_task(t)
    _assert_task_finished(t, "eviction restart write D")

    kvcache.zero_()
    t = service.create(hash_128s=_block_hashes(4), kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "eviction restart read D")
    if not torch.equal(kvcache, expected_d):
        raise AssertionError("eviction restart: D mismatch")

    print("[phase_eviction_restart_verify] ok")


def _phase_crash_writer(storage_dir: Path, redis_host: str, redis_port: int, redis_prefix: str) -> None:
    page_bytes = 4096
    num_pages = 128
    storage_size = 32 * 1024 * 1024
    num_shard = 4

    # Baseline committed data
    kvcache = _make_kvcache(seed=777, num_pages=num_pages, page_bytes=page_bytes)
    expected = kvcache.clone()

    service = _make_service(
        kvcache=kvcache,
        storage_dir=storage_dir,
        storage_size=storage_size,
        num_shard=num_shard,
        num_worker=8,
        index_endpoint=f"{redis_host}:{redis_port}",
        index_prefix=str(redis_prefix),
    )

    indexer = torch.arange(num_pages, dtype=torch.int32)

    data = list(range(20_000, 20_000 + num_pages))
    hash_128s = generate_cumulative_hashes(data)

    t = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    _wait_task(t)
    _assert_task_finished(t, "baseline write")

    # Start another write but do NOT wait; crash immediately.
    kvcache.copy_(_make_kvcache(seed=888, num_pages=num_pages, page_bytes=page_bytes))
    data2 = list(range(30_000, 30_000 + num_pages))
    hash_128s2 = generate_cumulative_hashes(data2)
    _ = service.create(hash_128s=hash_128s2, kv_page_indexer=indexer, mode="w")

    # Abrupt termination: bypass destructors.
    # This simulates SIGKILL / power loss more closely than sys.exit.
    print("[phase_crash_writer] crashing now")
    os._exit(137)


def _phase_crash_verify(storage_dir: Path, redis_host: str, redis_port: int, redis_prefix: str) -> None:
    page_bytes = 4096
    num_pages = 128
    storage_size = 32 * 1024 * 1024
    num_shard = 4

    kvcache = torch.zeros((num_pages, page_bytes), dtype=torch.uint8)
    expected = _make_kvcache(seed=777, num_pages=num_pages, page_bytes=page_bytes)

    service = _make_service(
        kvcache=kvcache,
        storage_dir=storage_dir,
        storage_size=storage_size,
        num_shard=num_shard,
        num_worker=8,
        index_endpoint=f"{redis_host}:{redis_port}",
        index_prefix=str(redis_prefix),
    )

    indexer = torch.arange(num_pages, dtype=torch.int32)
    data = list(range(20_000, 20_000 + num_pages))
    hash_128s = generate_cumulative_hashes(data)

    # Baseline should still be readable after crash.
    q = service.query(hash_128s)
    if not all(q):
        raise AssertionError(f"query after crash expected all True for committed baseline, got {q}")

    t = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "read after crash")

    if not torch.equal(kvcache, expected):
        raise AssertionError("baseline readback mismatch after crash")

    # Ensure service remains usable: do another write/read.
    kvcache2 = _make_kvcache(seed=999, num_pages=num_pages, page_bytes=page_bytes)
    kvcache.copy_(kvcache2)
    data3 = list(range(40_000, 40_000 + num_pages))
    hash_128s3 = generate_cumulative_hashes(data3)

    t = service.create(hash_128s=hash_128s3, kv_page_indexer=indexer, mode="w")
    _wait_task(t)
    _assert_task_finished(t, "post-crash write")

    kvcache.zero_()
    t = service.create(hash_128s=hash_128s3, kv_page_indexer=indexer, mode="r")
    _wait_task(t)
    _assert_task_finished(t, "post-crash read")

    if not torch.equal(kvcache, kvcache2):
        raise AssertionError("post-crash readback mismatch")

    print("[phase_crash_verify] ok")


def _run_phase_self(args: list[str], *, env: dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        cwd=str(TESTS_DIR),
        env=env,
        capture_output=False,
        text=True,
        timeout=timeout,
    )


def _orchestrate(ns: argparse.Namespace) -> None:
    cleanup_storage = not bool(str(getattr(ns, "storage_dir", "") or "").strip())

    if bool(getattr(ns, "reuse_services", False)):
        host = str(getattr(ns, "redis_host", "") or "").strip() or "127.0.0.1"
        port = int(getattr(ns, "redis_port", 0) or 0)
        if port <= 0:
            raise SystemExit("--reuse-services requires --redis-port")
        redis = _RedisHandle("reuse", host, port, lambda: None)
    else:
        redis = _start_redis_or_skip()

    storage_root_cli = str(getattr(ns, "storage_dir", "") or "").strip()
    if storage_root_cli:
        storage_root = Path(storage_root_cli).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
    else:
        storage_root = TESTS_DIR / "cache"
        storage_root.mkdir(parents=True, exist_ok=True)

    storage_dir = storage_root / "recovery"
    redis_prefix = "lightmem_test_" + uuid.uuid4().hex

    storage_dir_evict = storage_root / "eviction"
    redis_prefix_evict = "lightmem_test_evict_" + uuid.uuid4().hex

    base_env = os.environ.copy()

    try:
        # Normal shutdown / restart
        r1 = _run_phase_self(
            [
                "--phase",
                "normal",
                "--storage",
                str(storage_dir),
                "--index-endpoint",
                f"{redis.host}:{redis.port}",
                "--index-prefix",
                str(redis_prefix),
            ],
            env=base_env,
            timeout=120,
        )
        if r1.returncode != 0:
            raise RuntimeError(f"phase normal failed: rc={r1.returncode}")

        # Simulate partial Redis key loss while disk/WAL remains.
        # (Full dataset loss recovery isn't guaranteed in all builds/configs.)
        _redis_delete_prefix_keys(redis.host, redis.port, prefix=redis_prefix, num_shard=4)

        r2 = _run_phase_self(
            [
                "--phase",
                "restart_verify",
                "--storage",
                str(storage_dir),
                "--index-endpoint",
                f"{redis.host}:{redis.port}",
                "--index-prefix",
                str(redis_prefix),
            ],
            env=base_env,
            timeout=120,
        )
        if r2.returncode != 0:
            raise RuntimeError(f"phase restart_verify failed: rc={r2.returncode}")

        # Abnormal exit / restart
        r3 = _run_phase_self(
            [
                "--phase",
                "crash_writer",
                "--storage",
                str(storage_dir),
                "--index-endpoint",
                f"{redis.host}:{redis.port}",
                "--index-prefix",
                str(redis_prefix),
            ],
            env=base_env,
            timeout=120,
        )
        if r3.returncode == 0:
            raise RuntimeError("phase crash_writer unexpectedly exited cleanly")

        r4 = _run_phase_self(
            [
                "--phase",
                "crash_verify",
                "--storage",
                str(storage_dir),
                "--index-endpoint",
                f"{redis.host}:{redis.port}",
                "--index-prefix",
                str(redis_prefix),
            ],
            env=base_env,
            timeout=120,
        )
        if r4.returncode != 0:
            raise RuntimeError(f"phase crash_verify failed: rc={r4.returncode}")

        # LRU eviction correctness across restart
        env_evict = base_env.copy()
        r5 = _run_phase_self(
            [
                "--phase",
                "eviction_setup",
                "--storage",
                str(storage_dir_evict),
                "--index-endpoint",
                f"{redis.host}:{redis.port}",
                "--index-prefix",
                str(redis_prefix_evict),
            ],
            env=env_evict,
            timeout=120,
        )
        if r5.returncode != 0:
            raise RuntimeError(f"phase eviction_setup failed: rc={r5.returncode}")

        r6 = _run_phase_self(
            [
                "--phase",
                "eviction_restart_verify",
                "--storage",
                str(storage_dir_evict),
                "--index-endpoint",
                f"{redis.host}:{redis.port}",
                "--index-prefix",
                str(redis_prefix_evict),
            ],
            env=env_evict,
            timeout=120,
        )
        if r6.returncode != 0:
            raise RuntimeError(f"phase eviction_restart_verify failed: rc={r6.returncode}")

        print("✓ Redis + disk recovery robustness test passed")

    finally:
        # Best-effort cleanup
        if cleanup_storage:
            try:
                shutil.rmtree(storage_dir, ignore_errors=True)
                shutil.rmtree(storage_dir_evict, ignore_errors=True)
            except Exception:
                pass
        redis.cleanup()


def main() -> None:
    args = _parse_args(sys.argv[1:])

    if args.phase == "orchestrate":
        _orchestrate(args)
        return

    storage_dir = Path(args.storage)
    if not storage_dir:
        raise SystemExit("--storage is required for phase mode")

    endpoint = (args.index_endpoint or "").strip() or "127.0.0.1:6379"
    if ":" in endpoint:
        host, port_s = endpoint.rsplit(":", 1)
        port = int(port_s)
    else:
        host, port = endpoint, 6379
    prefix = str(args.index_prefix or "lightmem")

    if args.phase == "normal":
        _phase_normal(storage_dir, host, port, prefix)
    elif args.phase == "restart_verify":
        _phase_restart_verify(storage_dir, host, port, prefix)
    elif args.phase == "crash_writer":
        _phase_crash_writer(storage_dir, host, port, prefix)
    elif args.phase == "crash_verify":
        _phase_crash_verify(storage_dir, host, port, prefix)
    elif args.phase == "eviction_setup":
        _phase_eviction_setup(storage_dir, host, port, prefix)
    elif args.phase == "eviction_restart_verify":
        _phase_eviction_restart_verify(storage_dir, host, port, prefix)
    else:
        raise SystemExit(f"unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
