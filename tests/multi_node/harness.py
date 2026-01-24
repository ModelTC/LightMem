#!/usr/bin/env python3
"""Multi-node test harness for LightMem.

Goals:
- Each test file can run standalone: it starts/stops Redis+Etcd as needed.
- run_all.py can start Redis+Etcd once and pass connection info to each test to reuse.

We keep this harness dependency-light:
- Redis: prefer local redis-server; docker compose fallback.
- Etcd: prefer local etcd; docker (single container) fallback.

All tests are designed to run on a single machine.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[2]

# Ensure repo's Python sources are importable when running tests directly
# without installing the package (editable install still works fine).
_PY_SRC = ROOT / "python"
if _PY_SRC.exists():
    sys.path.insert(0, str(_PY_SRC))


def _load_etcd_http_client_cls():
    """Load EtcdV3HttpClient from repo sources without requiring installed package."""
    path = ROOT / "python" / "light_mem" / "etcd_v3_http.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("light_mem_etcd_v3_http", str(path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Register into sys.modules before exec_module so decorators (e.g. dataclasses)
    # can resolve module globals via sys.modules[__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, "EtcdV3HttpClient", None)


def which(name: str) -> str | None:
    return shutil.which(name)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 180) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True, timeout=timeout)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_port(host: str, port: int, *, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def redis_ping(host: str, port: int, *, timeout_s: float = 0.5) -> bool:
    try:
        return redis_resp_command(host, port, ["PING"], timeout_s=timeout_s).startswith("+PONG")
    except Exception:
        return False


def etcd_health(host: str, port: int, *, timeout_s: float = 0.8) -> bool:
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            raw = resp.read()
        obj = json.loads(raw.decode("utf-8")) if raw else {}
        return str(obj.get("health", "")).lower() == "true"
    except Exception:
        return False


def _wait_lightmem_server_ready(*, host: str, redis_port: int, etcd_port: int, proc: subprocess.Popen, timeout_s: float = 20.0) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"lightmem_server exited early rc={proc.returncode}")
        if redis_ping(host, redis_port) and etcd_health(host, etcd_port):
            return
        time.sleep(0.1)
    raise TimeoutError("timeout waiting for lightmem_server to become ready")


@dataclass
class LightMemServerHandle:
    proc: subprocess.Popen
    kind: str
    cmd: list[str]

    def cleanup(self) -> None:
        p = self.proc
        if p.poll() is not None:
            return

        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass

        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
            p.wait(timeout=8)


def _lightmem_server_command() -> tuple[list[str] | None, str]:
    """Return (cmd, kind). Prefer console script, fallback to python -m."""
    exe = which("lightmem_server")
    if exe:
        return [exe], "lightmem_server"

    # Fallback: run module from repo source tree.
    # This still uses the same implementation (server_cli.py) without modifying it.
    return [sys.executable, "-m", "light_mem.server_cli"], "python -m light_mem.server_cli"


def start_lightmem_server_or_skip(*, host: str, redis_port: int, etcd_port: int, etcd_peer_port: int) -> LightMemServerHandle:
    base_cmd, kind = _lightmem_server_command()
    if not base_cmd:
        print("SKIP: lightmem_server is unavailable.")
        raise SystemExit(0)

    cmd = [
        *base_cmd,
        "--mode",
        "local",
        "--index-port",
        str(int(redis_port)),
        "--coord-port",
        str(int(etcd_port)),
        "--coord-peer-port",
        str(int(etcd_peer_port)),
    ]

    env = os.environ.copy()
    if kind.startswith("python -m"):
        # Ensure repo Python package is importable in the subprocess.
        py_src = str(ROOT / "python")
        env["PYTHONPATH"] = (py_src + os.pathsep + env.get("PYTHONPATH", "")).rstrip(os.pathsep)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_lightmem_server_ready(host=host, redis_port=redis_port, etcd_port=etcd_port, proc=proc, timeout_s=25.0)
    except Exception:
        out = ""
        try:
            if proc.stdout is not None:
                out = proc.stdout.read()[-4000:]
        except Exception:
            pass

        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        if out:
            print("[lightmem_server output tail]\n" + out)

        # Preserve old behavior: skip tests if dependencies are not available.
        print("SKIP: failed to start lightmem_server local services (need redis-server and etcd).")
        raise SystemExit(0)

    return LightMemServerHandle(proc=proc, kind=kind, cmd=cmd)


def redis_resp_command(host: str, port: int, argv: list[str], *, timeout_s: float = 1.0) -> str:
    """Send one Redis command over RESP, return first-line reply."""
    payload = "*" + str(len(argv)) + "\r\n"
    for a in argv:
        b = a.encode("utf-8")
        payload += "$" + str(len(b)) + "\r\n" + a + "\r\n"

    with socket.create_connection((host, port), timeout=timeout_s) as s:
        s.settimeout(timeout_s)
        s.sendall(payload.encode("utf-8"))

        buf = bytearray()
        while True:
            ch = s.recv(1)
            if not ch:
                break
            buf += ch
            if buf.endswith(b"\r\n"):
                break
        return buf.decode("utf-8", errors="replace").strip()


def _redis_read_line(sock: socket.socket) -> bytes:
    buf = bytearray()
    while True:
        ch = sock.recv(1)
        if not ch:
            raise ConnectionError("redis connection closed")
        buf += ch
        if buf.endswith(b"\r\n"):
            return bytes(buf[:-2])


def _redis_read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("redis connection closed")
        buf += chunk
    return bytes(buf)


def redis_command(host: str, port: int, argv: list[str], *, timeout_s: float = 1.0) -> str | None:
    """Execute a Redis command and return decoded string result.

    Supported replies:
    - simple string (+OK)
    - error (-ERR ...): raises RuntimeError
    - integer (:1)
    - bulk string ($len\r\n...): returns str; nil ($-1) -> None

    For multi-node tests we mostly need HGET/HEXISTS/HDEL/DEL.
    """
    payload = "*" + str(len(argv)) + "\r\n"
    for a in argv:
        b = a.encode("utf-8")
        payload += "$" + str(len(b)) + "\r\n" + a + "\r\n"

    with socket.create_connection((host, port), timeout=timeout_s) as s:
        s.settimeout(timeout_s)
        s.sendall(payload.encode("utf-8"))

        first = _redis_read_line(s)
        if not first:
            return None

        prefix = chr(first[0])
        rest = first[1:]

        if prefix == '+':
            return rest.decode("utf-8", errors="replace")
        if prefix == '-':
            raise RuntimeError(rest.decode("utf-8", errors="replace"))
        if prefix == ':':
            return rest.decode("utf-8", errors="replace")
        if prefix == '$':
            try:
                ln = int(rest.decode("utf-8", errors="replace"))
            except Exception:
                return None
            if ln < 0:
                return None
            data = _redis_read_exact(s, ln)
            _ = _redis_read_exact(s, 2)  # CRLF
            return data.decode("utf-8", errors="replace")

        # Not needed for our tests.
        raise RuntimeError(f"unsupported RESP reply: {first!r}")


def redis_hget_str(host: str, port: int, *, key: str, field: str) -> str | None:
    return redis_command(host, port, ["HGET", key, field])


def redis_hexists(host: str, port: int, *, key: str, field: str) -> bool:
    v = redis_command(host, port, ["HEXISTS", key, field])
    try:
        return int(v or "0") == 1
    except Exception:
        return False


def redis_hdel(host: str, port: int, *, key: str, field: str) -> None:
    _ = redis_command(host, port, ["HDEL", key, field])


def parse_global_mapping(value: str) -> tuple[int, int] | None:
    # value like "<shard_id>:<slot_id>"
    if not value:
        return None
    if ":" not in value:
        return None
    a, b = value.split(":", 1)
    try:
        return int(a), int(b)
    except Exception:
        return None


@dataclass
class RedisHandle:
    host: str
    port: int
    kind: str
    _cleanup_cb: callable | None

    def cleanup(self) -> None:
        if self._cleanup_cb is None:
            return
        cb = self._cleanup_cb
        self._cleanup_cb = None
        cb()


@dataclass
class EtcdHandle:
    host: str
    port: int
    kind: str
    _cleanup_cb: callable | None

    def cleanup(self) -> None:
        if self._cleanup_cb is None:
            return
        cb = self._cleanup_cb
        self._cleanup_cb = None
        cb()


def start_redis_local() -> RedisHandle | None:
    redis_server = which("redis-server")
    if not redis_server:
        return None

    port = find_free_port()
    tmpdir = Path(tempfile.mkdtemp(prefix="lightmem-test-redis-"))

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

    if not wait_port("127.0.0.1", port, timeout_s=10.0):
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    def _cleanup() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return RedisHandle(host="127.0.0.1", port=port, kind="local", _cleanup_cb=_cleanup)


def docker_compose_cmd() -> list[str] | None:
    if which("docker"):
        try:
            run(["docker", "compose", "version"], cwd=ROOT, timeout=10)
            return ["docker", "compose"]
        except Exception:
            pass
    if which("docker-compose"):
        return ["docker-compose"]
    return None


def etcd_client(host: str, port: int):
    cls = _load_etcd_http_client_cls()
    if cls is None:
        raise RuntimeError("EtcdV3HttpClient not found in repo sources; cannot create etcd client")
    return cls(host=host, port=port)


def wait_shard_owners(*, host: str, port: int, prefix: str, num_shards: int, timeout_s: float = 30.0) -> dict[int, str]:
    """Wait until every shard has an owner key in etcd."""
    client = etcd_client(host, port)
    base = f"{prefix.rstrip('/')}/shards/"

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        owners: dict[int, str] = {}
        # Scan owners.
        for value, meta in client.get_prefix(base):
            try:
                key = meta.key.decode("utf-8")
            except Exception:
                continue
            if not key.endswith("/owner"):
                continue
            try:
                sid_str = key[len(base):].split("/", 1)[0]
                sid = int(sid_str)
            except Exception:
                continue
            if 0 <= sid < num_shards and value is not None:
                try:
                    owners[sid] = value.decode("utf-8")
                except Exception:
                    continue

        if len(owners) >= num_shards:
            return owners
        time.sleep(0.2)

    raise TimeoutError(f"etcd shard owners not ready: have {len(owners)}/{num_shards}")


def start_redis_docker_compose_fixed_port() -> RedisHandle | None:
    compose = docker_compose_cmd()
    if not compose:
        return None

    if wait_port("127.0.0.1", 6379, timeout_s=0.2):
        return None

    compose_file = ROOT / "docker-compose.redis.yml"
    if not compose_file.exists():
        return None

    try:
        run([*compose, "-f", str(compose_file), "up", "-d"], cwd=ROOT, timeout=180)
    except Exception:
        return None

    if not wait_port("127.0.0.1", 6379, timeout_s=20.0):
        try:
            run([*compose, "-f", str(compose_file), "down", "-v"], cwd=ROOT, timeout=60)
        except Exception:
            pass
        return None

    def _cleanup() -> None:
        try:
            run([*compose, "-f", str(compose_file), "down", "-v"], cwd=ROOT, timeout=60)
        except Exception:
            pass

    return RedisHandle(host="127.0.0.1", port=6379, kind="docker-compose", _cleanup_cb=_cleanup)


def start_redis_or_skip(*, reuse: bool, host: str, port: int) -> RedisHandle:
    if reuse:
        return RedisHandle(host=host, port=port, kind="reuse", _cleanup_cb=None)

    h = start_redis_local()
    if h:
        return h
    h = start_redis_docker_compose_fixed_port()
    if h:
        return h

    print("SKIP: redis-server/docker compose unavailable (or port 6379 busy).")
    raise SystemExit(0)


def start_etcd_local() -> EtcdHandle | None:
    etcd = which("etcd")
    if not etcd:
        return None

    host = "127.0.0.1"
    client_port = find_free_port()
    peer_port = find_free_port()
    data_dir = Path(tempfile.mkdtemp(prefix="lightmem-test-etcd-"))

    name = "coord"
    cmd = [
        etcd,
        "--name",
        name,
        "--data-dir",
        str(data_dir),
        "--listen-client-urls",
        f"http://{host}:{client_port}",
        "--advertise-client-urls",
        f"http://{host}:{client_port}",
        "--listen-peer-urls",
        f"http://{host}:{peer_port}",
        "--initial-advertise-peer-urls",
        f"http://{host}:{peer_port}",
        "--initial-cluster",
        f"{name}=http://{host}:{peer_port}",
        "--initial-cluster-state",
        "new",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not wait_port(host, client_port, timeout_s=15.0):
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(data_dir, ignore_errors=True)
        return None

    def _cleanup() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(data_dir, ignore_errors=True)

    return EtcdHandle(host=host, port=client_port, kind="local", _cleanup_cb=_cleanup)


def start_etcd_docker() -> EtcdHandle | None:
    docker = which("docker")
    if not docker:
        return None

    host = "127.0.0.1"
    client_port = find_free_port()
    peer_port = find_free_port()

    # bitnami/etcd image is used elsewhere in repo; keep consistent.
    # Use --rm? We want to cleanup deterministically; we'll stop container by name.
    name = f"lightmem-test-etcd-{client_port}"

    try:
        run(
            [
                docker,
                "run",
                "-d",
                "--name",
                name,
                "-e",
                "ALLOW_NONE_AUTHENTICATION=yes",
                "-e",
                "ETCD_NAME=coord",
                "-p",
                f"{client_port}:2379",
                "-p",
                f"{peer_port}:2380",
                "bitnami/etcd:3.5",
            ],
            cwd=ROOT,
            timeout=180,
        )
    except Exception:
        return None

    if not wait_port(host, client_port, timeout_s=20.0):
        try:
            run([docker, "rm", "-f", name], cwd=ROOT, timeout=30)
        except Exception:
            pass
        return None

    def _cleanup() -> None:
        try:
            run([docker, "rm", "-f", name], cwd=ROOT, timeout=30)
        except Exception:
            pass

    return EtcdHandle(host=host, port=client_port, kind="docker", _cleanup_cb=_cleanup)


def start_etcd_or_skip(*, reuse: bool, host: str, port: int) -> EtcdHandle:
    if reuse:
        return EtcdHandle(host=host, port=port, kind="reuse", _cleanup_cb=None)

    h = start_etcd_local()
    if h:
        return h
    h = start_etcd_docker()
    if h:
        return h

    print("SKIP: etcd/docker unavailable.")
    raise SystemExit(0)


@dataclass
class ClusterEnv:
    redis: RedisHandle
    etcd: EtcdHandle
    server: LightMemServerHandle | None
    storage_dir: Path
    cleanup_storage: bool

    def cleanup(self) -> None:
        # Stop services first to release file handles.
        if self.server is not None:
            try:
                self.server.cleanup()
            finally:
                self.server = None
        else:
            try:
                self.etcd.cleanup()
            finally:
                self.redis.cleanup()

        if self.cleanup_storage:
            shutil.rmtree(self.storage_dir, ignore_errors=True)


def make_storage_dir(*, base: str | None = None) -> Path:
    if base:
        p = Path(base)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(tempfile.mkdtemp(prefix="lightmem-multi-node-cache-"))


def start_cluster_env(*, reuse_services: bool, redis_host: str, redis_port: int, etcd_host: str, etcd_port: int,
                      storage_dir: str | None = None, cleanup_storage: bool = True) -> ClusterEnv:
    # Multi-node tests now rely on lightmem_server to start/manage both Redis (index)
    # and Etcd (coord). Ports are fixed by convention unless explicitly provided.
    index_port = int(redis_port) if int(redis_port) > 0 else 6379
    coord_port = int(etcd_port) if int(etcd_port) > 0 else 2379

    server: LightMemServerHandle | None = None
    if not reuse_services:
        # lightmem_server --mode local binds to 127.0.0.1.
        # Keep behavior explicit to avoid surprising cross-host binds.
        if str(redis_host) not in ("127.0.0.1", "localhost") or str(etcd_host) not in ("127.0.0.1", "localhost"):
            raise SystemExit("multi_node tests require --redis-host/--etcd-host to be 127.0.0.1 when not using --reuse-services")
        server = start_lightmem_server_or_skip(host="127.0.0.1", redis_port=index_port, etcd_port=coord_port, etcd_peer_port=2380)

    r = RedisHandle(host=str(redis_host), port=index_port, kind=("reuse" if reuse_services else "lightmem_server"), _cleanup_cb=None)
    e = EtcdHandle(host=str(etcd_host), port=coord_port, kind=("reuse" if reuse_services else "lightmem_server"), _cleanup_cb=None)
    sd = make_storage_dir(base=storage_dir)
    return ClusterEnv(redis=r, etcd=e, server=server, storage_dir=sd, cleanup_storage=cleanup_storage)


def parse_common_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--reuse-services", action="store_true", help="Reuse already running redis/etcd")
    p.add_argument("--redis-host", default="127.0.0.1")
    p.add_argument("--redis-port", type=int, default=0)
    p.add_argument("--etcd-host", default="127.0.0.1")
    p.add_argument("--etcd-port", type=int, default=0)
    p.add_argument("--storage-dir", default="")
    return p.parse_args(argv)


def require_ports(ns: argparse.Namespace) -> tuple[str, int, str, int]:
    rh = str(ns.redis_host)
    rp = int(ns.redis_port)
    eh = str(ns.etcd_host)
    ep = int(ns.etcd_port)

    if ns.reuse_services:
        if rp <= 0 or ep <= 0:
            raise SystemExit("--reuse-services requires --redis-port and --etcd-port")
    else:
        # allow 0 to mean 'auto' for independent mode
        if rp < 0 or ep < 0:
            raise SystemExit("invalid port")

    return rh, rp, eh, ep


def iter_hash_ids(*, count: int, seed: int = 0) -> list[int]:
    """Deterministic 128-bit integers for block hashes."""
    out: list[int] = []
    x = (seed & 0xFFFFFFFFFFFFFFFF) | (seed << 64)
    for i in range(count):
        # A tiny LCG-ish mix into 128-bit space.
        x = (x * 6364136223846793005 + 1442695040888963407 + i) & ((1 << 128) - 1)
        out.append(x)
    return out


def build_hash_128s_for_blocks(*, block_hash_ids: Iterable[int], pages_per_block: int) -> list[int]:
    """Build a list[int] whose last element of each block is the hash id.

    PyLocalCacheService will take every `pages_per_block`-th element (the last of each block)
    as the block hash string.
    """
    result: list[int] = []
    dummy = 1
    for hid in block_hash_ids:
        for _ in range(pages_per_block - 1):
            result.append(dummy)
            dummy += 1
        result.append(int(hid))
    return result
