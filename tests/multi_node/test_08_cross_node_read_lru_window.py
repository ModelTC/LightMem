#!/usr/bin/env python3
"""Cross-node read under LRU eviction window.

We attempt to create a situation where:
- A reader resolves mapping via Redis
- A writer evicts/overwrites
- Reader should either read correctly (before eviction) or fail (after eviction),
  but must not crash.

This is a best-effort race test; it mainly ensures the read path is robust.

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from harness import (
    iter_hash_ids,
    parse_common_args,
    require_ports,
    redis_hget_str,
    start_cluster_env,
    wait_shard_owners,
)


WORKER = Path(__file__).with_name("worker_node_ops.py")


def _wait_redis_mapping(*, host: str, port: int, field_hex: str, timeout_s: float = 30.0) -> str:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        v = redis_hget_str(host, port, key="lightmem:global:index", field=field_hex)
        if v is not None:
            return v
        time.sleep(0.01)
    raise TimeoutError(f"missing global mapping for {field_hex}")


def main(argv: list[str] | None = None) -> int:
    ns = parse_common_args(argv)
    redis_host, redis_port, etcd_host, etcd_port = require_ports(ns)

    num_shard = 192
    storage_size = 6 * 1024 * 1024 * 1024  # 6GB total => ~32MB per shard

    page_bytes = 4096
    num_pages = 2048

    env = start_cluster_env(
        reuse_services=bool(ns.reuse_services),
        redis_host=redis_host,
        redis_port=redis_port,
        etcd_host=etcd_host,
        etcd_port=etcd_port,
        storage_dir=(ns.storage_dir or None),
        cleanup_storage=not bool(ns.storage_dir),
    )

    try:
        peer = subprocess.Popen(
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
                "2",
                "--redis",
                f"{env.redis.host}:{env.redis.port}",
                "--etcd",
                f"{env.etcd.host}:{env.etcd.port}",
                "--node-id",
                "node-peer",
                "--ttl",
                "6",
                "--reconcile-sec",
                "1.0",
                "--num-pages",
                str(num_pages),
                "--page-bytes",
                str(page_bytes),
                "--op",
                "idle",
                "--duration-sec",
                "60",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)

        # Prime one key.
        hid0 = iter_hash_ids(count=1, seed=9001)[0]
        h0 = format(hid0, "032x")
        prime = subprocess.run(
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
                "2",
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
                "--num-pages",
                str(num_pages),
                "--page-bytes",
                str(page_bytes),
                "--op",
                "write",
                "--hash-id",
                str(hid0),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert prime.returncode == 0, (prime.stdout or "") + (prime.stderr or "")
        _ = _wait_redis_mapping(host=env.redis.host, port=env.redis.port, field_hex=h0, timeout_s=30)

        # Stress writer + reader concurrently; test should not crash/hang.
        writer = subprocess.Popen(
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
                "2",
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
                "--num-pages",
                str(num_pages),
                "--page-bytes",
                str(page_bytes),
                "--op",
                "stress_write",
                "--hash-id",
                str(910000),
                "--duration-sec",
                "6",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        reader = subprocess.Popen(
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
                "2",
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
                "--num-pages",
                str(num_pages),
                "--page-bytes",
                str(page_bytes),
                "--op",
                "stress_read",
                "--hash-id",
                str(hid0),
                "--duration-sec",
                "6",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        out_w, _ = writer.communicate(timeout=180)
        out_r, _ = reader.communicate(timeout=180)
        assert writer.returncode == 0, out_w
        assert reader.returncode == 0, out_r

        return 0
    finally:
        try:
            peer.terminate()
        except Exception:
            pass
        try:
            peer.wait(timeout=5)
        except Exception:
            try:
                peer.kill()
            except Exception:
                pass
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
