#!/usr/bin/env python3
"""Concurrent init against a shared storage directory.

Regression test for TOCTOU-style races in LocalStorageEngine::createOrOpenFiles when multiple
services start at the same time pointing to the same `--storage-dir`.

Expected behavior after the fix:
- Exactly one process becomes the initializer (creates dirs/files and writes `<storage>.initialized`).
- Other processes wait for `<storage>.initialized` and only open existing files (no create/truncate).

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

# Allow running this test from repo root where the script dir isn't on sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from harness import parse_common_args, require_ports, start_cluster_env


WORKER = Path(__file__).with_name("worker_node_ops.py")


def _wait_path_exists(path: Path, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"timeout waiting for {path}")


def _tail(p: subprocess.Popen[str] | None, *, max_chars: int = 4000) -> str:
    try:
        if p is None or p.stdout is None:
            return ""
        return (p.stdout.read() or "")[-int(max_chars) :]
    except Exception:
        return ""


def _choose_num_shard() -> int:
    # Each shard opens 2 fds (data/meta) per process. Add a conservative reserve for
    # stdio, sockets, dylibs, etc.
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft == resource.RLIM_INFINITY:
            soft = 4096
        soft = int(soft)
    except Exception:
        soft = 1024

    reserve = 256
    per_shard_fds = 2
    max_by_limit = max(32, (max(0, soft - reserve) // per_shard_fds))

    # We don't need huge scale here; we only need concurrent init ordering.
    return int(max(64, min(256, max_by_limit)))


def _wait_marker_or_fail(
    *,
    marker: Path,
    procs: dict[str, subprocess.Popen[str] | None],
    timeout_s: float,
) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if marker.exists():
            return
        for name, p in procs.items():
            if p is None:
                continue
            rc = p.poll()
            if rc is not None and rc != 0:
                out = _tail(p)
                raise RuntimeError(f"worker {name} exited rc={rc}\n{out}")
        time.sleep(0.05)
    raise TimeoutError(f"timeout waiting for {marker}")


def main(argv: list[str] | None = None) -> int:
    ns = parse_common_args(argv)
    redis_host, redis_port, etcd_host, etcd_port = require_ports(ns)

    num_shard = _choose_num_shard()
    storage_size = 24 * 1024 * 1024 * 1024  # sparse file preallocation

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        soft, hard = -1, -1
    print(f"[test] RLIMIT_NOFILE soft={soft} hard={hard} num_shard={num_shard}", flush=True)

    env = start_cluster_env(
        reuse_services=bool(ns.reuse_services),
        redis_host=redis_host,
        redis_port=redis_port,
        etcd_host=etcd_host,
        etcd_port=etcd_port,
        storage_dir=(ns.storage_dir or None),
        cleanup_storage=not bool(ns.storage_dir),
    )

    p0: subprocess.Popen[str] | None = None
    p1: subprocess.Popen[str] | None = None
    try:
        marker_dir = Path(env.storage_dir) / "_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)

        # These are created by LocalStorageEngine::createOrOpenFiles.
        init_lock_dir = Path(str(env.storage_dir) + ".init.lock")
        init_marker = Path(str(env.storage_dir) + ".initialized")

        node0_started = marker_dir / "node0_started"
        node1_started = marker_dir / "node1_started"

        # Start node-0 first; it should win the global init lock in most cases.
        p0 = subprocess.Popen(
            [
                sys.executable,
                str(WORKER),
                "--storage-dir",
                str(env.storage_dir),
                "--storage-size",
                str(storage_size),
                "--num-shard",
                str(num_shard),
                "--num-worker",
                "1",
                "--redis",
                f"{env.redis.host}:{env.redis.port}",
                "--etcd",
                f"{env.etcd.host}:{env.etcd.port}",
                "--node-id",
                "node-0",
                "--ttl",
                "6",
                "--reconcile-sec",
                "1.0",
                "--op",
                "idle",
                "--duration-sec",
                "6",
                "--started-file",
                str(node0_started),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait until we observe the lock dir (or marker, if init is extremely fast).
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if init_lock_dir.exists() or init_marker.exists():
                break
            time.sleep(0.01)

        p1 = subprocess.Popen(
            [
                sys.executable,
                str(WORKER),
                "--storage-dir",
                str(env.storage_dir),
                "--storage-size",
                str(storage_size),
                "--num-shard",
                str(num_shard),
                "--num-worker",
                "1",
                "--redis",
                f"{env.redis.host}:{env.redis.port}",
                "--etcd",
                f"{env.etcd.host}:{env.etcd.port}",
                "--node-id",
                "node-1",
                "--ttl",
                "6",
                "--reconcile-sec",
                "1.0",
                "--op",
                "idle",
                "--duration-sec",
                "6",
                "--started-file",
                str(node1_started),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        procs = {"node-0": p0, "node-1": p1}

        # Regression check:
        # If the follower reaches "started" before the marker exists, it means it didn't wait.
        # (If init is very fast, the marker may already exist and this is fine.)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if node1_started.exists() and not init_marker.exists():
                raise AssertionError("follower started before init marker existed")
            if init_marker.exists():
                break
            # Also fail-fast if a worker exits.
            for name, p in procs.items():
                if p is None:
                    continue
                rc = p.poll()
                if rc is not None and rc != 0:
                    raise RuntimeError(f"worker {name} exited rc={rc}\n{_tail(p)}")
            time.sleep(0.02)

        _wait_marker_or_fail(marker=init_marker, procs=procs, timeout_s=90.0)
        _wait_path_exists(node0_started, timeout_s=30.0)
        _wait_path_exists(node1_started, timeout_s=30.0)

        # After completion, lock dir should be released (best-effort rmdir).
        assert not init_lock_dir.exists(), f"init lock dir still exists: {init_lock_dir}"

        # Spot-check a few shards for expected files.
        sample_sids = [0]
        if num_shard > 1:
            sample_sids.append(1)
        if num_shard > 2:
            sample_sids.append(num_shard - 1)
        for sid in sample_sids:
            shard_dir = Path(f"{env.storage_dir}_{sid}")
            assert shard_dir.exists() and shard_dir.is_dir(), f"missing shard dir {shard_dir}"
            for name in ("data", "meta"):
                p = shard_dir / name
                assert p.exists(), f"missing {p}"

        return 0
    finally:
        for p in (p0, p1):
            if p is None:
                continue
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)

        # Print tails to aid debugging on failures.
        for name, p in (("node-0", p0), ("node-1", p1)):
            out = _tail(p)
            if out:
                print(f"[{name} output tail]\n" + out)

        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
